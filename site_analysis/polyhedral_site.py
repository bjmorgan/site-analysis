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
from scipy.spatial import Delaunay, QhullError
from site_analysis.site import Site, _as_index
from site_analysis.atom import Atom
from site_analysis.distances import _as_point
from site_analysis.pbc_utils import correct_pbc, update_pbc_shifts
from site_analysis.polyhedron_surface import _SurfaceCache, _surface_triangles


class PolyhedralSite(Site):
    """Describes a site defined by the polyhedral volume enclosed by a set
    of vertex atoms.

    The site is the polyhedron bounded by a fixed set of triangles over its
    vertices: those set with ``set_triangles`` (as
    ``PolyhedralSiteCollection.find_shared_faces`` does, so that sites
    sharing a face triangulate it identically), or else the triangles of
    the convex hull of the vertices in the first frame the site is tested
    in. A point is inside the site when it is inside one of the tetrahedra
    formed by the site's centre (the mean of its vertex coordinates) and a
    triangle. Containment uses numba when it is installed and numpy
    otherwise, with identical results.

    The polyhedron vertices are defined using atom indices in a structure, and
    their coordinates are assigned from the structure when needed. This allows the
    polyhedron shape to adapt to changes in the crystal structure.

    Attributes:
        vertex_indices (list[int]): List of integer indices for the vertex atoms
            (counting from 0).
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
        self._vertex_coords: np.ndarray | None = None
        self._lattice_matrix: np.ndarray | None = None
        self._delaunay: Delaunay | None = None
        self._triangles: np.ndarray | None = None
        self._surface_cache: _SurfaceCache | None = None
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
        vertex coordinates, their lattice, the Delaunay tessellation and the
        PBC shift caches. The triangles are kept.
        """
        super(PolyhedralSite, self).reset()
        self._vertex_coords = None
        self._lattice_matrix = None
        self._delaunay = None
        self._cache_stale = True
        self._pending_frac_coords = None
        self._pending_lattice_matrix = None
        self._pbc_image_shifts = None
        self._pbc_cached_raw_frac = None
 
    @property
    def vertex_coords(self) -> np.ndarray | None:
        """Unwrapped fractional coordinates of the vertices, or None if none have been assigned.

        Read-only: set with ``assign_vertex_coords`` or
        ``notify_structure_changed``. Coordinates notified with
        ``notify_structure_changed`` are assigned (unwrapped and stored) when
        this is read, so it always gives the coordinates of the most recent
        structure.

        Returns:
            (n_vertices, 3) array in the order of ``vertex_indices``, or None.

        Raises:
            ValueError: If notified coordinates are assigned and are not
                finite.
        """
        if self._pending_frac_coords is not None and self._pending_lattice_matrix is not None:
            self._assign_from_pending(self._pending_frac_coords, self._pending_lattice_matrix)
        return self._vertex_coords

    @property
    def delaunay(self) -> Delaunay:
        """Delaunay tessellation of the vertex coordinates for this site.

        This is calculated the first time the attribute is requested,
        and then stored for reuse until the vertex coordinates are next
        assigned (including coordinates notified with
        ``notify_structure_changed``) or the site is reset. It does not
        decide containment.

        Returns:
            scipy.spatial.Delaunay

        """
        vertex_coords = self.vertex_coords
        if self._delaunay is None:
            if vertex_coords is None:
                raise RuntimeError("Vertex coordinates have not been assigned.")
            self._delaunay = Delaunay(vertex_coords)
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

        Stores the lattice with the coordinates, clears any coordinates
        notified with ``notify_structure_changed`` and the Delaunay
        tessellation, and marks the containment planes to be recomputed on
        the next ``contains_point`` call.

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

        The pending data is cleared when the coordinates are stored, so
        pending data that cannot be stored stays pending.

        Args:
            all_frac_coords: Full fractional coordinate array.
            lattice_matrix: (3, 3) lattice matrix where rows are
                lattice vectors.
        """
        frac_coords = all_frac_coords[self.vertex_indices]
        self._store_vertex_coords(frac_coords, lattice_matrix)

    def _store_vertex_coords(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Apply PBC correction and store vertex coordinates.

        On the first call (or after an anomalous displacement invalidates
        the cache), unwraps the coordinates as ``unwrapped_vertex_coords``
        does. On subsequent calls, updates the cached integer image shifts
        incrementally by detecting coordinate wraps (jumps of ~1.0),
        avoiding the expensive nearest-image search.

        Sets ``vertex_coords``, stores the lattice, clears any notified
        coordinates and the Delaunay tessellation, and marks the
        containment planes to be recomputed.

        Args:
            frac_coords: Raw fractional coordinates of the vertices,
                shape ``(n_vertices, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are
                lattice vectors.
        """
        if self._pbc_image_shifts is not None and self._pbc_cached_raw_frac is not None:
            valid, vertex_coords, new_shifts = update_pbc_shifts(
                frac_coords, self._pbc_cached_raw_frac, self._pbc_image_shifts)
            if valid:
                self._pbc_image_shifts = new_shifts
                self._pbc_cached_raw_frac = frac_coords.copy()
                self._set_vertex_coords(vertex_coords, lattice_matrix)
                return

        # Full computation — first call only (or after anomalous displacement)
        wrapped, corrected, image_shifts = self._unwrap(frac_coords, lattice_matrix)
        self._pbc_image_shifts = image_shifts
        self._pbc_cached_raw_frac = wrapped
        self._set_vertex_coords(corrected, lattice_matrix)

    def _set_vertex_coords(self,
            vertex_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Store unwrapped vertex coordinates and the lattice they are in.

        Clears any coordinates notified with ``notify_structure_changed``
        and the Delaunay tessellation, and marks the containment planes to
        be recomputed.

        Args:
            vertex_coords: (n_vertices, 3) unwrapped fractional coordinates.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.
        """
        self._vertex_coords = vertex_coords
        self._lattice_matrix = lattice_matrix
        self._pending_frac_coords = None
        self._pending_lattice_matrix = None
        self._delaunay = None
        self._cache_stale = True

    def _unwrap(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Wrap vertex coordinates into [0, 1), then unwrap them around the site.

        Args:
            frac_coords: Fractional coordinates of the vertices, shape
                ``(n_vertices, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Returns:
            Tuple of ``(wrapped, unwrapped, image_shifts)``: the wrapped
            coordinates, the coordinates unwrapped by ``correct_pbc``, and
            the integer image shifts from the wrapped coordinates, as
            returned by ``correct_pbc``.
        """
        wrapped = frac_coords - np.floor(frac_coords)
        unwrapped, image_shifts = correct_pbc(wrapped, self.reference_center, lattice_matrix)
        return wrapped, unwrapped, image_shifts

    def unwrapped_vertex_coords(self,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> np.ndarray:
        """Unwrapped fractional coordinates of this site's vertices in a structure.

        The vertex coordinates are wrapped into [0, 1), then unwrapped
        around the site by ``correct_pbc``: to their images nearest the
        reference centre, or by the spread-based correction if the site has
        none. The result is the coordinates ``assign_vertex_coords`` stores
        for a site with no earlier coordinates. The site is not changed.

        Args:
            all_frac_coords: Full fractional coordinate array, shape
                ``(n_atoms, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Returns:
            (n_vertices, 3) array of the unwrapped coordinates, in the order
            of ``vertex_indices``.

        Raises:
            ValueError: If the vertex coordinates are not finite, or
                ``correct_pbc`` rejects them.
        """
        _, unwrapped, _ = self._unwrap(all_frac_coords[self.vertex_indices], lattice_matrix)
        return unwrapped

    def set_triangles(self, triangles: np.ndarray) -> None:
        """Set the triangles that bound this site.

        Containment uses these triangles from the next ``contains_point``
        call on. They are kept when the site is reset.

        Args:
            triangles: (T, 3) array of local vertex indices (positions in
                ``vertex_indices``) per triangle, each wound so that
                ``(b - a) x (c - a)`` points away from the site's centre,
                as from ``PolyhedralSiteCollection.find_shared_faces``.

        Raises:
            ValueError: If ``triangles`` is not a (T, 3) array of integers
                from 0 to one less than the number of vertices.
        """
        array = np.asarray(triangles)
        if (not np.issubdtype(array.dtype, np.integer)
                or array.ndim != 2 or array.shape[1] != 3
                or not np.all((array >= 0) & (array < len(self.vertex_indices)))):
            raise ValueError(
                "triangles must be a (T, 3) array of local vertex indices from 0 to "
                f"{len(self.vertex_indices) - 1}, got {triangles!r}")
        self._triangles = array.astype(np.int64)
        self._surface_cache = None

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

        The site is bounded by its triangles (see the class description).
        If none have been set, they are fixed on the first call from the
        convex hull of the vertices in the lattice they were assigned with.
        The containment planes are recomputed on the first call after the
        vertex coordinates change.

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
            RuntimeError: If no vertex coordinates have been assigned; if
                no triangles have been set and the coordinates were set
                without a lattice; if the vertices are degenerate; or if
                the site is too distorted for its triangles to enclose its
                centre.
            ValueError: If coordinates notified with
                ``notify_structure_changed`` are not finite, or if the
                vertices span more than 1000 cells, counted over the three
                axes, as they can in a strongly skewed or unreduced cell.
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
        if self._vertex_coords is None:
            raise RuntimeError(
                f'no vertex coordinates set for polyhedral_site {self.index}'
            )
        surface = self._surface_cache
        if surface is None or self._cache_stale:
            surface = self._refreshed_surface_cache(self._vertex_coords)
        return surface.contains_point(x)

    def _refreshed_surface_cache(self, vertex_coords: np.ndarray) -> _SurfaceCache:
        """Compute the containment planes for the current vertex coordinates.

        Fixes the triangles from the convex hull first if none are set.

        Args:
            vertex_coords: (N_vertices, 3) array of the current vertex
                coordinates.

        Returns:
            The site's surface cache, up to date with ``vertex_coords``.

        Raises:
            RuntimeError: If no triangles are set and the coordinates were
                set without a lattice; if the vertices are degenerate; or if
                the site is too distorted for its triangles to enclose its
                centre.
            ValueError: If the vertices span more than 1000 cells, counted
                over the three axes.
        """
        if self._triangles is None:
            if self._lattice_matrix is None:
                raise RuntimeError(
                    f"Site {self.index} (vertices {self.vertex_indices}): has no "
                    "triangles and its vertex coordinates have no lattice: assign "
                    "them with assign_vertex_coords or notify_structure_changed, "
                    "or call set_triangles")
            try:
                self._triangles = _surface_triangles(
                    self.vertex_indices, vertex_coords, self._lattice_matrix, ())
            except QhullError as error:
                raise RuntimeError(
                    f"Degenerate vertex geometry for polyhedral_site {self.index} "
                    f"(vertices {self.vertex_indices})"
                ) from error
        try:
            if self._surface_cache is None:
                self._surface_cache = _SurfaceCache(self._triangles, vertex_coords)
            else:
                self._surface_cache.update(vertex_coords)
        except (RuntimeError, ValueError) as error:
            raise type(error)(
                f"Site {self.index} (vertices {self.vertex_indices}): {error}") from error
        self._cache_stale = False
        return self._surface_cache

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
        assigned), the ``reference_center`` as a list of three floats
        (``None`` if the site has none) and the ``triangles`` that bound the
        site as a list of local vertex index triples (``None`` if none have
        been fixed).

        Returns:
            dict: dictionary representation of this site.
        """
        d = super(PolyhedralSite, self).as_dict()
        d['vertex_indices'] = self.vertex_indices
        d['vertex_coords'] = self.vertex_coords
        d['reference_center'] = (
            None if self.reference_center is None else self.reference_center.tolist())
        d['triangles'] = None if self._triangles is None else self._triangles.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PolyhedralSite:
        """Create a PolyhedralSite object from a dict representation.

        Restores the vertex indices, vertex coordinates, contained atoms,
        label, reference centre and triangles. A dict without a
        ``reference_center`` or ``triangles`` key, as written by earlier
        versions, gives a site with no reference centre or no triangles. A
        site restored without triangles cannot test containment until its
        vertex coordinates are assigned with a lattice. A collection of
        restored sites that has not found its shared faces replaces their
        triangles on its first analysis; calling ``find_shared_faces`` with
        the original structure first keeps them.

        Args:
            d (dict): The dict representation of this site.

        Returns:
            PolyhedralSite: A new PolyhedralSite instance.

        Raises:
            ValueError: If ``reference_center`` is not None and not three
                finite numbers, or ``triangles`` is not None and not
                accepted by ``set_triangles``.
        """
        polyhedral_site = cls(
            vertex_indices=d['vertex_indices'],
            reference_center=d.get('reference_center'))
        if d.get('triangles') is not None:
            polyhedral_site.set_triangles(d['triangles'])
        polyhedral_site._vertex_coords = (
            None if d['vertex_coords'] is None
            else np.asarray(d['vertex_coords'], dtype=float))
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
        vertex_coords = self.vertex_coords
        if vertex_coords is None:
            raise RuntimeError("Vertex coordinates have not been assigned.")
        centre_coords = np.mean(vertex_coords, axis=0)
        return np.array(centre_coords) 

    @classmethod
    def sites_from_vertex_indices(cls,
        vertex_indices: list[list[int]],
        label: str | None=None) -> list[PolyhedralSite]:
        sites = [cls(vertex_indices=vi, label=label) for vi in vertex_indices]
        return sites
