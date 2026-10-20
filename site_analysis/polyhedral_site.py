"""Polyhedral site representation for crystal structure analysis.

This module provides the PolyhedralSite class, which represents a site defined
by a polyhedron formed by a set of vertex atoms. These sites are commonly used
to represent coordination environments in crystal structures, such as tetrahedral
or octahedral sites.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError
from site_analysis.site import Site, _as_index
from site_analysis.tools import _SHIFTS
from site_analysis.atom import Atom
from site_analysis._compat import HAS_NUMBA
from site_analysis.distances import _as_point
from site_analysis.pbc_utils import correct_pbc, update_pbc_shifts

if HAS_NUMBA:
    import numba

    @numba.njit(cache=True)
    def _numba_update_hull_triangles(
        vertex_coords: np.ndarray,
        triangles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """JIT-compiled hull triangle normal and centre sign computation.

        Args:
            vertex_coords: (N_vertices, 3) array of vertex positions.
            triangles: (N_triangles, 3) array of vertex indices per hull
                triangle.

        Returns:
            Tuple of (triangle_normals, triangle_ref_points, centre_signs,
            within_two_cells), where within_two_cells is True if every
            vertex coordinate is in [0, 2).
        """
        n_triangles = triangles.shape[0]
        n_vertices = vertex_coords.shape[0]
        triangle_normals = np.empty((n_triangles, 3))
        triangle_ref_points = np.empty((n_triangles, 3))
        centre_signs = np.empty(n_triangles)

        centre = np.zeros(3)
        within_two_cells = True
        for i in range(n_vertices):
            for k in range(3):
                centre[k] += vertex_coords[i, k]
                if not (0.0 <= vertex_coords[i, k] < 2.0):
                    within_two_cells = False
        for k in range(3):
            centre[k] /= n_vertices

        for j in range(n_triangles):
            i0 = triangles[j, 0]
            i1 = triangles[j, 1]
            i2 = triangles[j, 2]

            # Edge vectors from vertex 2
            e0x = vertex_coords[i0, 0] - vertex_coords[i2, 0]
            e0y = vertex_coords[i0, 1] - vertex_coords[i2, 1]
            e0z = vertex_coords[i0, 2] - vertex_coords[i2, 2]
            e1x = vertex_coords[i1, 0] - vertex_coords[i2, 0]
            e1y = vertex_coords[i1, 1] - vertex_coords[i2, 1]
            e1z = vertex_coords[i1, 2] - vertex_coords[i2, 2]

            # Cross product
            triangle_normals[j, 0] = e0y * e1z - e0z * e1y
            triangle_normals[j, 1] = e0z * e1x - e0x * e1z
            triangle_normals[j, 2] = e0x * e1y - e0y * e1x

            # Reference point (first vertex of triangle)
            for k in range(3):
                triangle_ref_points[j, k] = vertex_coords[i0, k]

            # Centre sign
            dot = 0.0
            for k in range(3):
                dot += triangle_normals[j, k] * (centre[k] - triangle_ref_points[j, k])
            centre_signs[j] = np.sign(dot)

        return triangle_normals, triangle_ref_points, centre_signs, within_two_cells

    @numba.njit(cache=True)
    def _numba_sn_query(
        x: np.ndarray,
        image_shifts: np.ndarray,
        triangle_normals: np.ndarray,
        triangle_ref_points: np.ndarray,
        centre_signs: np.ndarray,
    ) -> bool:
        """JIT-compiled containment check using surface normals.

        Wraps the point into [0, 1) and tests each of its images given by
        the shifts: an image is inside when it lies on the same side as
        the polyhedron centre for every hull triangle. Exits early on the
        first failing triangle of an image, and on the first image inside.

        Args:
            x: (3,) fractional coordinates of the point, in any periodic
                image.
            image_shifts: (M, 3) integer array of whole-cell shifts from
                the wrapped point to the images to test.
            triangle_normals: (N_triangles, 3) precomputed outward triangle normals.
            triangle_ref_points: (N_triangles, 3) reference vertex per triangle.
            centre_signs: (N_triangles,) sign of centre dot product per triangle.

        Returns:
            True if any image is inside the polyhedron.
        """
        wrapped = np.empty(3)
        for k in range(3):
            wrapped[k] = x[k] - np.floor(x[k])
        n_images = image_shifts.shape[0]
        n_triangles = triangle_normals.shape[0]
        for i in range(n_images):
            all_match = True
            for j in range(n_triangles):
                dot = 0.0
                for k in range(3):
                    dot += ((wrapped[k] + image_shifts[i, k] - triangle_ref_points[j, k])
                            * triangle_normals[j, k])
                # dot == 0.0 means the point lies exactly on the triangle
                # plane; np.sign(0.0) returns 0.0 which would never
                # match centre_signs (always +/-1), so we skip the
                # check to avoid falsely rejecting boundary points.
                if dot != 0.0 and np.sign(dot) != centre_signs[j]:
                    all_match = False
                    break
            if all_match:
                return True
        return False


# The most cells a polyhedral site's vertices may span, counted over all
# three axes, for the site to test the images of a point in each of them.
_MAX_IMAGE_SHIFTS = 1000


def _image_shifts(vertex_coords: np.ndarray) -> np.ndarray:
    """Whole-cell shifts from a point in [0, 1) to its images that can lie inside a polyhedron.

    An image ``x + n`` of a point ``x`` in [0, 1) lies inside the
    vertices' bounding box only if ``floor(min_k) <= n_k <= floor(max_k)``
    along each axis ``k``, where ``min_k`` and ``max_k`` are the lowest and
    highest vertex coordinates.

    Args:
        vertex_coords: (N_vertices, 3) array of finite vertex positions.

    Returns:
        (M, 3) integer array of the shifts, one for each cell the bounding
        box meets.

    Raises:
        ValueError: If the bounding box meets more than
            ``_MAX_IMAGE_SHIFTS`` cells, as it can for a site in a
            strongly skewed or unreduced cell.
    """
    lower = np.floor(vertex_coords.min(axis=0)).astype(np.int64)
    upper = np.floor(vertex_coords.max(axis=0)).astype(np.int64)
    cells = upper - lower + 1
    if np.prod(cells, dtype=np.float64) > _MAX_IMAGE_SHIFTS:
        raise ValueError(
            f"vertices span {cells.tolist()} cells along the lattice axes, more than "
            f"the {_MAX_IMAGE_SHIFTS} cells in which a site's images can be tested; "
            "use a reduced cell")
    axes = [np.arange(lo, hi + 1) for lo, hi in zip(lower, upper)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


class _HullTriangleCache:
    """Cached hull triangle topology and surface normal data for containment.

    The topology (which vertex triples form each hull triangle) is computed
    once from an initial ``ConvexHull`` and cached permanently -- it depends
    only on vertex connectivity, not on coordinates. Per-timestep, the
    triangle normals, reference points, centre signs and image shifts are
    recomputed from new vertex coordinates using the cached topology.

    Attributes:
        triangles: (N_triangles, 3) array of vertex indices per hull
            triangle.
        image_shifts: (M, 3) integer array of whole-cell shifts from a
            point in [0, 1) to its images that can lie inside the
            polyhedron: the eight shifts with components in {0, 1} when
            every vertex coordinate is in [0, 2), otherwise those from
            ``_image_shifts``.
    """

    def __init__(self, vertex_coords: np.ndarray) -> None:
        """Compute hull triangle topology from initial vertex coordinates.

        Args:
            vertex_coords: (N_vertices, 3) array of vertex positions.
                Used to build a ConvexHull and extract triangle connectivity.

        Raises:
            RuntimeError: If numba is not installed.
            ValueError: If the vertices span more than ``_MAX_IMAGE_SHIFTS``
                cells, from ``_image_shifts``.
        """
        if not HAS_NUMBA:
            raise RuntimeError(
                "_HullTriangleCache requires numba. "
                "Install it with: pip install site-analysis[fast]"
            )
        hull = ConvexHull(vertex_coords)
        self.triangles: np.ndarray = hull.simplices
        self._triangle_normals: np.ndarray
        self._triangle_ref_points: np.ndarray
        self._centre_signs: np.ndarray
        self.image_shifts: np.ndarray
        self.update(vertex_coords)

    def update(self, vertex_coords: np.ndarray) -> None:
        """Recompute triangle normals and image shifts from new coordinates.

        Recomputes the triangle normals, reference points and centre signs
        from the cached topology, and sets ``image_shifts``. Called once per
        timestep after vertex coordinates are assigned.

        Args:
            vertex_coords: (N_vertices, 3) array of current vertex positions.

        Raises:
            ValueError: If the vertices span more than ``_MAX_IMAGE_SHIFTS``
                cells, from ``_image_shifts``.
        """
        (self._triangle_normals, self._triangle_ref_points, self._centre_signs,
         within_two_cells) = _numba_update_hull_triangles(vertex_coords, self.triangles)
        self.image_shifts = _SHIFTS if within_two_cells else _image_shifts(vertex_coords)

    def contains_point(self, x: np.ndarray) -> bool:
        """Test whether an image of a point is inside the polyhedron.

        The point is wrapped into [0, 1) and each of its images given by
        ``image_shifts`` is tested.

        Args:
            x: (3,) fractional coordinates of the point, in any periodic
                image.

        Returns:
            True if an image of the point is inside the polyhedron.
        """
        return bool(_numba_sn_query(
            x,
            self.image_shifts,
            self._triangle_normals,
            self._triangle_ref_points,
            self._centre_signs,
        ))


class PolyhedralSite(Site):
    """Describes a site defined by the polyhedral volume enclosed by a set
    of vertex atoms.

    A PolyhedralSite determines whether atoms are inside the site volume by
    constructing a convex polyhedron from the vertex atoms and checking whether
    points lie within this polyhedron. The containment algorithm is selected
    automatically: when numba is available, a JIT-compiled surface normal method
    with cached hull triangle topology is used; otherwise, falls back to
    Delaunay tessellation via scipy.

    The polyhedron vertices are defined using atom indices in a structure, and
    their coordinates are assigned from the structure when needed. This allows the
    polyhedron shape to adapt to changes in the crystal structure.

    Attributes:
        vertex_indices (list[int]): List of integer indices for the vertex atoms
            (counting from 0).
        vertex_coords (np.ndarray or None): Fractional coordinates of the vertices.
            Set using assign_vertex_coords() or notify_structure_changed().
        reference_center (np.ndarray or None): Optional reference centre for PBC handling.

    See Also:
        :class:`~site_analysis.site.Site`: Parent class documenting inherited attributes
            (index, label, contains_atoms, trajectory, points, transitions).
    """

    def __init__(self,
        vertex_indices: Iterable[int],
        label: str | None=None,
        reference_center: np.ndarray | None=None):
        """Create a PolyhedralSite instance.
        
        Args:
            vertex_indices: Integer indices for the vertex atoms (counting from 0),
                as any iterable, such as a list, an array or a generator. They are
                stored as a list of Python ``int``.
            label: Optional label for this site.
            reference_center: Optional reference centre for PBC handling,
                three finite numbers, stored as a float array.
        
        Returns:
            None
        
        Raises:
            ValueError: If vertex_indices is empty, contains a negative index,
                or lists an atom more than once, or if reference_center is
                not None and not three finite numbers.
            TypeError: If any element in vertex_indices is not an integer
                (a ``bool`` is not accepted).
        """
        vertex_indices = list(vertex_indices)

        if not vertex_indices:
            raise ValueError("vertex_indices cannot be empty")

        vertex_indices = [_as_index(idx, "vertex index") for idx in vertex_indices]

        seen: set[int] = set()
        for k in vertex_indices:
            if k < 0:
                raise ValueError(f"vertex_indices contains negative atom index {k}")
            if k in seen:
                raise ValueError(f"vertex_indices lists atom {k} more than once")
            seen.add(k)
        if reference_center is not None:
            reference_center = _as_point(reference_center, "reference_center")

        super(PolyhedralSite, self).__init__(label=label)
        self.vertex_indices = vertex_indices
        self.vertex_coords: np.ndarray | None = None
        self._delaunay: Delaunay | None = None
        self._point_image_shifts: np.ndarray | None = None
        self._hull_triangle_cache: _HullTriangleCache | None = None
        self._cache_stale: bool = True
        self._pending_frac_coords: np.ndarray | None = None
        self._pending_lattice_matrix: np.ndarray | None = None
        self._pbc_image_shifts: np.ndarray | None = None
        self._pbc_cached_raw_frac: np.ndarray | None = None
        self.reference_center = reference_center

    def __repr__(self) -> str:
        string = ('site_analysis.PolyhedralSite('
                  f'index={self.index}, '
                  f'label={self.label}, '
                  f'vertex_indices={self.vertex_indices}, '
                  f'contains_atoms={self.contains_atoms})')
        return string
                  
    def reset(self) -> None:
        """Clear the recorded occupation data and cached geometry for this site.

        Clears the occupation data as in ``Site.reset``, and unsets the
        vertex coordinates, Delaunay tessellation, image shifts of points
        and PBC shift caches. The cached hull triangles are kept.
        """
        super(PolyhedralSite, self).reset()
        self.vertex_coords = None
        self._delaunay = None
        self._point_image_shifts = None
        self._cache_stale = True
        self._pending_frac_coords = None
        self._pending_lattice_matrix = None
        self._pbc_image_shifts = None
        self._pbc_cached_raw_frac = None
 
    @property
    def delaunay(self) -> Delaunay:
        """Delaunay tessellation of the vertex coordinates for this site.

        This is calculated the first time the attribute is requested,
        and then stored for reuse, unless the site is reset.

        Returns:
            scipy.spatial.Delaunay

        """
        if not self._delaunay:
            if self.vertex_coords is None:
                raise RuntimeError("Vertex coordinates have not been assigned.")
            self._delaunay = Delaunay(self.vertex_coords)
        return self._delaunay

    @property
    def coordination_number(self) -> int:
        """Coordination number for this site, defined as the number of 
        vertices

        Returns:
            int

        """
        return len(self.vertex_indices)
    
    @property
    def cn(self) -> int:
        """Coordination number for this site, defined as the number of
        vertices

        Convenience property for coordination_number()

        Returns:
            int

        """
        return self.coordination_number
        
    def notify_structure_changed(self,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Mark vertex coordinates as stale for lazy reassignment.

        Stores a reference to the full coordinate array so that
        PBC-corrected vertex coordinates can be computed on demand
        when ``contains_point`` is next called.

        Args:
            all_frac_coords: Full fractional coordinate array from the
                structure, shape ``(n_atoms, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are
                lattice vectors.
        """
        self._pending_frac_coords = all_frac_coords
        self._pending_lattice_matrix = lattice_matrix

    def assign_vertex_coords(self,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Assign fractional coordinates to the polyhedra vertices.

        Resets the cached Delaunay tessellation, so the next
        ``contains_point`` call will recompute it.

        Args:
            all_frac_coords: Full fractional coordinate array, shape
                ``(n_atoms, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Note:
            For bulk analysis prefer ``notify_structure_changed``, which
            pre-extracts coordinates once and defers PBC correction
            until the site is actually queried.
        """
        frac_coords = all_frac_coords[self.vertex_indices]
        self._store_vertex_coords(frac_coords, lattice_matrix)

    def _assign_from_pending(self,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Compute PBC-corrected vertex coords from pending data.

        Args:
            all_frac_coords: Full fractional coordinate array.
            lattice_matrix: (3, 3) lattice matrix where rows are
                lattice vectors.
        """
        self._pending_frac_coords = None
        self._pending_lattice_matrix = None
        frac_coords = all_frac_coords[self.vertex_indices]
        self._store_vertex_coords(frac_coords, lattice_matrix)

    def _store_vertex_coords(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Apply PBC correction and store vertex coordinates.

        On the first call (or after an anomalous displacement invalidates
        the cache), delegates to ``correct_pbc()`` for full PBC
        unwrapping. On subsequent calls, updates the cached integer
        image shifts incrementally by detecting coordinate wraps
        (jumps of ~1.0), avoiding the expensive nearest-image search.

        Sets ``vertex_coords``, clears the Delaunay tessellation and the
        image shifts of points, and marks the cached hull triangle normals
        to be recomputed.

        Args:
            frac_coords: Raw fractional coordinates of the vertices,
                shape ``(n_vertices, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are
                lattice vectors. Passed to ``correct_pbc()``; unused
                by the incremental update path.
        """
        if self._pbc_image_shifts is not None and self._pbc_cached_raw_frac is not None:
            valid, vertex_coords, new_shifts = update_pbc_shifts(
                frac_coords, self._pbc_cached_raw_frac, self._pbc_image_shifts)
            if valid:
                self._pbc_image_shifts = new_shifts
                self._pbc_cached_raw_frac = frac_coords.copy()
                self.vertex_coords = vertex_coords
                self._delaunay = None
                self._point_image_shifts = None
                self._cache_stale = True
                return

        # Full computation — first call only (or after anomalous displacement)
        corrected, image_shifts = correct_pbc(
            frac_coords, self.reference_center, lattice_matrix)
        self._pbc_image_shifts = image_shifts
        self._pbc_cached_raw_frac = frac_coords.copy()
        self.vertex_coords = corrected
        self._delaunay = None
        self._point_image_shifts = None
        self._cache_stale = True

    def get_vertex_species(self,
            species: list[str]) -> list[str]:
        """Return species strings for this site's vertex atoms.

        Args:
            species: List of species strings for all atoms in the
                structure, indexed by atom index.

        Returns:
            Species strings for this site's vertex atoms.
        """
        return [species[i] for i in self.vertex_indices]

    def contains_point(self,
            x: np.ndarray,
            *,
            algo: str | None = None,
            pbc_images: np.ndarray | None = None) -> bool:
        """Test whether a specific point is enclosed by this polyhedral site.

        The point may be given in any periodic image. It is wrapped into
        [0, 1), and each of its images that can lie inside the vertices'
        bounding box is tested, in every cell the vertices span.

        The containment algorithm is selected automatically based on available
        dependencies. When numba is installed, uses a JIT-compiled surface
        normal method. Otherwise, falls back to Delaunay tessellation.

        Args:
            x: Fractional coordinates of the point to test (length 3 array).
            algo: Deprecated. Previously selected the algorithm. Now ignored;
                the best available method is used automatically.
            pbc_images: Deprecated. Previously images of the point to test
                in place of its own. Now ignored; the site tests the images
                of the point that can lie inside it.

        Returns:
            True if the point is inside the polyhedron.

        Raises:
            RuntimeError: If no vertex coordinates have been assigned, or
                the vertices are degenerate.
            ValueError: If the vertices span more than ``_MAX_IMAGE_SHIFTS``
                cells, as they can in a strongly skewed or unreduced cell.
        """
        if algo is not None:
            warnings.warn(
                "The 'algo' parameter is deprecated and will be removed in a "
                "future version. The best available containment algorithm is "
                "now selected automatically.",
                DeprecationWarning,
                stacklevel=2,
            )
        if pbc_images is not None:
            warnings.warn(
                "The 'pbc_images' parameter is deprecated and will be removed in "
                "version 2.0. The site now tests the images of the point itself.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self._pending_frac_coords is not None and self._pending_lattice_matrix is not None:
            self._assign_from_pending(self._pending_frac_coords, self._pending_lattice_matrix)
        if self.vertex_coords is None:
            raise RuntimeError(
                f'no vertex coordinates set for polyhedral_site {self.index}'
            )
        try:
            if HAS_NUMBA:
                if self._hull_triangle_cache is None:
                    self._hull_triangle_cache = _HullTriangleCache(self.vertex_coords)
                    self._cache_stale = False
                elif self._cache_stale:
                    self._hull_triangle_cache.update(self.vertex_coords)
                    self._cache_stale = False
                return self._hull_triangle_cache.contains_point(x)
            if self._point_image_shifts is None:
                self._point_image_shifts = _image_shifts(self.vertex_coords)
            return self._contains_point_delaunay(
                (x - np.floor(x)) + self._point_image_shifts)
        except QhullError as e:
            raise RuntimeError(
                f"Degenerate vertex geometry for polyhedral_site {self.index} "
                f"(vertices {self.vertex_indices})"
            ) from e
   
    def _contains_point_delaunay(self, x: np.ndarray) -> bool:
        """Test containment using Delaunay tessellation.

        Args:
            x: Fractional coordinates as (3,) or (N, 3) array.

        Returns:
            True if any point is inside a simplex of the tessellation.
        """
        return bool(np.any(self.delaunay.find_simplex(x) >= 0))

    def contains_atom(self,
            atom: Atom,
            *,
            algo: str | None = None,
            pbc_images: np.ndarray | None = None) -> bool:
        """Test whether an atom is inside this polyhedron.

        Args:
            atom: The atom to test.
            algo: Deprecated. Previously selected the algorithm. Now ignored;
                the best available method is used automatically.
            pbc_images: Deprecated. Previously images of the atom to test
                in place of its own. Now ignored; the site tests the images
                of the atom that can lie inside it.

        Returns:
            True if the atom is inside the polyhedron.
        """
        if algo is not None:
            warnings.warn(
                "The 'algo' parameter is deprecated and will be removed in a "
                "future version. The best available containment algorithm is "
                "now selected automatically.",
                DeprecationWarning,
                stacklevel=2,
            )
        if pbc_images is not None:
            warnings.warn(
                "The 'pbc_images' parameter is deprecated and will be removed in "
                "version 2.0. The site now tests the images of the atom itself.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.contains_point(atom.frac_coords)

    def as_dict(self) -> dict:
        """Dictionary of this site's attributes, which may contain numpy arrays.

        Besides the attributes of :class:`~site_analysis.site.Site`, it contains
        the ``vertex_indices``, the ``vertex_coords`` (``None`` if not yet
        assigned) and the ``reference_center`` as a list of three floats
        (``None`` if the site has none).

        Returns:
            dict: dictionary representation of this site.
        """
        d = super(PolyhedralSite, self).as_dict()
        d['vertex_indices'] = self.vertex_indices
        d['vertex_coords'] = self.vertex_coords
        d['reference_center'] = (
            None if self.reference_center is None else self.reference_center.tolist())
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PolyhedralSite:
        """Create a PolyhedralSite object from a dict representation.

        Restores the vertex indices, vertex coordinates, contained atoms,
        label and reference centre. A dict without a ``reference_center``
        key, as written by earlier versions, gives a site with no reference
        centre.

        Args:
            d (dict): The dict representation of this site.

        Returns:
            PolyhedralSite: A new PolyhedralSite instance.

        Raises:
            ValueError: If ``reference_center`` is not None and not three
                finite numbers.
        """
        polyhedral_site = cls(
            vertex_indices=d['vertex_indices'],
            reference_center=d.get('reference_center'))
        polyhedral_site.vertex_coords = d['vertex_coords']
        polyhedral_site.contains_atoms = d['contains_atoms']
        polyhedral_site.label = d.get('label')
        return polyhedral_site

    @property
    def centre(self) -> np.ndarray:
        """Returns the fractional coordinates of the centre point of
        this polyhedral site.

        Args:
            None

        Returns:
            (np.array): (3,) numpy array.
 
        """
        if self.vertex_coords is None:
            raise RuntimeError("Vertex coordinates have not been assigned.")
        centre_coords = np.mean(self.vertex_coords, axis=0)
        return np.array(centre_coords) 

    @classmethod
    def sites_from_vertex_indices(cls,
        vertex_indices: list[list[int]],
        label: str | None=None) -> list[PolyhedralSite]:
        sites = [cls(vertex_indices=vi, label=label) for vi in vertex_indices]
        return sites
