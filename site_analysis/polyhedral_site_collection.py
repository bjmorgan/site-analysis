"""Collection manager for polyhedral sites in crystal structures.

This module provides the PolyhedralSiteCollection class, which manages a
collection of PolyhedralSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The PolyhedralSiteCollection extends the base SiteCollection class with
specific functionality for polyhedral sites, including:

- Maintaining a map of neighbouring polyhedral sites that share three or
  more vertex indices
- Finding the faces that pairs of sites share, and which sites are
  adjacent through a shared face, and fixing each site's triangles so that
  sites sharing a face triangulate it identically
- Optimised atom assignment via the PriorityAssignmentMixin
- Distance-ranked site ordering when reference centres are available

The module also includes utility functions:

- ``construct_neighbouring_sites``: analyses polyhedral sites to determine
  which ones share three or more vertex indices.
- ``_collect_reference_centres``: extracts reference centres from polyhedral
  sites for distance-ranked ordering.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable

from .faces import Face, _find_shared_faces, _narrow_cell_hint
from .site_collection import SiteCollection, PriorityAssignmentMixin
from .polyhedral_site import PolyhedralSite
from .polyhedron_surface import _SharedFaceError, _first_overlap, _surface_triangles
from .atom import Atom
from pymatgen.core import Structure
import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import QhullError


class PolyhedralSiteCollection(PriorityAssignmentMixin[PolyhedralSite], SiteCollection):
    """A collection of PolyhedralSite objects.
    
    Extends the base SiteCollection class with specific functionality for
    polyhedral sites, including maintaining a map of neighbouring polyhedral
    sites that share three or more vertex indices and implementing optimised
    atom assignment based on spatial relationships and learned transition
    patterns.

    After an atom's recent sites and learned transitions, the remaining
    sites are searched in order of the distance of their reference centres
    from the atom. If any site lacks a reference centre (for example, sites
    built with ``use_reference_centers=False``), the search instead falls
    back to the neighbours of the atom's most recent site, then list order,
    for the whole collection.
    
    Attributes:
        sites (list): List of ``PolyhedralSite`` objects.
    
    """

    def __init__(self,
            sites: Iterable[PolyhedralSite]) -> None:
        """Create a PolyhedralSiteCollection instance.

        Args:
            sites (iterable(PolyhedralSite)): PolyhedralSite objects, such as
                a list or a generator.

        Returns:
            None

        Raises:
            TypeError: If any site is not a ``PolyhedralSite``.

        """
        sites = list(sites)
        for s in sites:
            if not isinstance(s, PolyhedralSite):
                raise TypeError(f"Expected PolyhedralSite, got {type(s).__name__}")
        super().__init__(sites)
        self.sites: list[PolyhedralSite]
        self._neighbouring_sites = construct_neighbouring_sites(self.sites)
        self._shared_faces: tuple[Face, ...] | None = None
        self._largest_vertex_index = max(
            (max(site.vertex_indices) for site in self.sites), default=-1)
        centres, site_indices = _collect_reference_centres(self.sites)
        if centres is not None:
            self._init_priority_ranking(centres, site_indices)

    def analyse_structure(self,
            atoms: list[Atom],
            structure: Structure) -> None:
        """Analyse a structure to assign atoms to polyhedral sites.

        Finds the shared faces from the structure, as ``find_shared_faces``
        does, if none have been found. Then assigns coordinates to atoms,
        notifies sites of the new structure, and assigns atoms to sites.

        Args:
            atoms: List of Atom objects to be assigned to sites.
            structure: Pymatgen Structure containing atom positions.

        Raises:
            ValueError: If a site has a vertex index that is not an atom of
                the structure, or finding the shared faces fails (see
                ``find_shared_faces``).
            RuntimeError: If finding the shared faces fails on degenerate
                vertex geometry, or a site tested for an atom is too
                distorted for its triangles to enclose its centre.
        """
        all_frac_coords = structure.frac_coords
        lattice_matrix = structure.lattice.matrix
        self.validate_atom_count(len(all_frac_coords))
        if self._shared_faces is None:
            self.find_shared_faces(all_frac_coords, lattice_matrix)
        for a in atoms:
            a.assign_coords(all_frac_coords)
        for s in self.sites:
            s.notify_structure_changed(all_frac_coords, lattice_matrix)
        self.assign_site_occupations(atoms, lattice_matrix)

    def assign_site_occupations(self,
            atoms: list[Atom],
            lattice_matrix: np.ndarray) -> None:
        """Assign atoms to polyhedral sites based on their positions.

        This method implements an optimised assignment logic using a priority-based
        site checking approach.

        Args:
            atoms: List of Atom objects to be assigned to sites.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors. Used to rank candidate sites by the distance of
                their reference centres from each atom; the containment
                checks themselves do not use it.

        Raises:
            ValueError: When every site has a reference centre and an
                atom's search reaches the distance ranking (the atom is
                not in its recent sites or learned transition
                destinations): if ``lattice_matrix`` is not a finite,
                non-singular (3, 3) matrix, or if the atom's coordinates
                are not finite.
        """
        self.reset_site_occupations()
        for atom in atoms:
            atom.in_site = None

            # Check sites in priority order until found
            for site in self._get_priority_sites(atom, lattice_matrix):
                if site.contains_atom(atom):
                    self.update_occupation(site, atom)
                    break
    
    def validate_atom_count(self, n_atoms: int) -> None:
        """Check that every vertex index of every site is an atom of a structure.

        Args:
            n_atoms: The number of atoms in the structure.

        Raises:
            ValueError: If a site has a vertex index that is not less than
                ``n_atoms``. The message names the first such site in index
                order, and its largest vertex index.
        """
        if self._largest_vertex_index < n_atoms:
            return
        site = min((site for site in self.sites if max(site.vertex_indices) >= n_atoms),
                   key=lambda site: site.index)
        raise ValueError(
            f"Site {site.index} has vertex index {max(site.vertex_indices)}, "
            f"but the structure has {n_atoms} atoms")

    def neighbouring_sites(self,
            index: int) -> list[PolyhedralSite]:
        """Return the sites that share three or more vertex indices with a site.

        Args:
            index: Index of the site.

        Returns:
            The neighbouring sites.

        Raises:
            KeyError: If no site has this index.
        """
        return self._neighbouring_sites[index]

    def find_shared_faces(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Find the faces shared by pairs of sites, and fix each site's triangles.

        Two sites share a face when they share three or more vertex atoms
        under one lattice translation. Each site's triangles are then fixed
        from its vertex coordinates unwrapped as by
        ``PolyhedralSite.unwrapped_vertex_coords``: the convex hull of its
        vertices, with each face it shares triangulated by a rule that
        depends only on the face's atoms, so that the sites sharing it
        triangulate it identically (see ``PolyhedralSite.set_triangles``).
        The faces and triangles replace any earlier ones, and are kept when
        the collection is reset. If an error is raised, the earlier faces
        and triangles are kept.

        Args:
            frac_coords: Fractional coordinates of the atoms, shape
                ``(n_atoms, 3)``, indexed by atom index.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Raises:
            ValueError: If a site has a vertex index that is not an atom of
                the structure (see ``validate_atom_count``); the vertex
                coordinates are not finite; two sites coincide or one lies
                inside another; the atoms two sites share cannot be
                triangulated identically on both; or two sites, or a site
                and a periodic image of itself, overlap with positive
                volume.
            RuntimeError: If a site's vertices are degenerate.
        """
        frac_coords = np.asarray(frac_coords, dtype=float)
        lattice_matrix = np.asarray(lattice_matrix, dtype=float)
        self.validate_atom_count(len(frac_coords))
        faces = _find_shared_faces(self.sites, frac_coords, lattice_matrix)
        sites = sorted(self.sites, key=lambda site: site.index)
        sites_by_index = {site.index: site for site in sites}
        faces_by_site: defaultdict[int, list[Face]] = defaultdict(list)
        for face in faces:
            for i in face.site_indices:
                faces_by_site[i].append(face)
        coords = [site.unwrapped_vertex_coords(frac_coords, lattice_matrix) for site in sites]
        triangles = [
            _site_triangles(site, site_coords, lattice_matrix,
                            faces_by_site[site.index], sites_by_index)
            for site, site_coords in zip(sites, coords)]
        overlap = _first_overlap(coords, triangles)
        if overlap is not None:
            first, second = sites[overlap[0]], sites[overlap[1]]
            if first is second:
                message = f"Site {first.index} overlaps a periodic image of itself"
            else:
                message = f"Sites {first.index} and {second.index} overlap"
            raise ValueError(message + _narrow_cell_hint(first, second))
        self._shared_faces = faces
        for site, site_triangles in zip(sites, triangles):
            site.set_triangles(site_triangles)

    def shared_faces(self) -> tuple[Face, ...]:
        """Return the faces shared by pairs of sites.

        Returns:
            The shared faces, sorted by site indices then vertex indices.

        Raises:
            RuntimeError: If the shared faces have not been found.
        """
        if self._shared_faces is None:
            raise RuntimeError(
                "Shared faces have not been found: call Trajectory.find_shared_faces(structure)")
        return self._shared_faces

    def site_adjacency(self) -> csr_array:
        """Return which sites share a face.

        Returns:
            A new symmetric boolean sparse array of shape ``(n, n)``, where
            ``n`` is one more than the largest site index. Entry ``(i, j)``
            is True when the sites with indices ``i`` and ``j`` share a face.

        Raises:
            RuntimeError: If the shared faces have not been found.
        """
        faces = self.shared_faces()
        n = max((site.index for site in self.sites), default=-1) + 1
        rows = np.array([face.site_indices[0] for face in faces], dtype=int)
        columns = np.array([face.site_indices[1] for face in faces], dtype=int)
        return csr_array(
            (np.ones(2 * len(faces), dtype=bool),
             (np.concatenate([rows, columns]), np.concatenate([columns, rows]))),
            shape=(n, n))

    def sites_contain_points(self,
            points: np.ndarray,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> bool:
        """Check whether the set of sites contain corresponding points.

        Finds the shared faces from the structure, as ``find_shared_faces``
        does, if none have been found.

        Args:
            points: (N, 3) array of fractional coordinates.
                One coordinate per site being checked.
            all_frac_coords: Full fractional coordinate array from the
                structure, shape ``(n_atoms, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Returns:
            True if every point is contained by its corresponding site.

        Raises:
            ValueError: If the number of points does not match the number
                of sites, a site has a vertex index that is not an atom
                of the structure, or finding the shared faces fails (see
                ``find_shared_faces``).
            RuntimeError: If finding the shared faces fails on degenerate
                vertex geometry, or a site is too distorted for its
                triangles to enclose its centre.
        """
        if len(points) != len(self.sites):
            raise ValueError(
                f"Expected {len(self.sites)} points (one per site), "
                f"got {len(points)}"
            )
        self.validate_atom_count(len(all_frac_coords))
        if self._shared_faces is None:
            self.find_shared_faces(all_frac_coords, lattice_matrix)
        for s in self.sites:
            s.notify_structure_changed(all_frac_coords, lattice_matrix)
        return all(s.contains_point(p) for s, p in zip(self.sites, points))

def _site_triangles(
        site: PolyhedralSite,
        coords: np.ndarray,
        lattice_matrix: np.ndarray,
        faces: list[Face],
        sites_by_index: dict[int, PolyhedralSite]) -> np.ndarray:
    """Surface triangles of a site, triangulating each face it shares by its atoms alone.

    Args:
        site: The site.
        coords: (N_vertices, 3) unwrapped fractional coordinates of the
            site's vertices.
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors.
        faces: The faces the site shares with other sites.
        sites_by_index: Every site of the collection, by site index.

    Returns:
        (N_triangles, 3) int64 array of local vertex indices per triangle,
        as from ``_surface_triangles``.

    Raises:
        ValueError: If the atoms of a shared face cannot be triangulated
            identically on the site, naming the two sites that share it.
        RuntimeError: If the site's vertices are degenerate.
    """
    try:
        return _surface_triangles(
            site.vertex_indices, coords, lattice_matrix,
            [face.vertex_indices for face in faces])
    except _SharedFaceError as error:
        face = next(face for face in faces if face.vertex_indices == error.atoms)
        i, j = face.site_indices
        raise ValueError(
            f"Sites {i} and {j}: {error}; the sites may overlap"
            f"{_narrow_cell_hint(sites_by_index[i], sites_by_index[j])}") from error
    except QhullError as error:
        raise RuntimeError(
            f"Degenerate vertex geometry for polyhedral_site {site.index} "
            f"(vertices {site.vertex_indices})") from error


def _collect_reference_centres(
        sites: list[PolyhedralSite],
) -> tuple[np.ndarray | None, list[int]]:
    """Collect reference centres from polyhedral sites.

    Args:
        sites: List of PolyhedralSite objects.

    Returns:
        A tuple of (centres, site_indices) where:
        - centres is an (N, 3) array of fractional coordinates, or None
          if any site lacks a reference centre.
        - site_indices is a list of site indices.
    """
    site_indices = [s.index for s in sites]
    centres = [s.reference_center for s in sites]
    if any(centre is None for centre in centres):
        return None, site_indices
    return np.array(centres), site_indices


def construct_neighbouring_sites(
        sites: list[PolyhedralSite]) -> dict[int, list[PolyhedralSite]]:
    """Find the sites that share three or more vertex indices with each site.

    Args:
        sites: List of PolyhedralSite objects to check for shared vertices.

    Returns:
        (dict): Dictionary of `int`: `list` entries.
            Keys are site indices. Values are lists of ``PolyhedralSite``
            objects, in the order they appear in ``sites``.

    """
    positions_by_vertex: defaultdict[int, list[int]] = defaultdict(list)
    for position, site in enumerate(sites):
        for vertex in site.vertex_indices:
            positions_by_vertex[vertex].append(position)
    neighbours: dict[int, list[PolyhedralSite]] = {}
    for site_i in sites:
        n_shared_vertices = Counter(
            position for vertex in site_i.vertex_indices
            for position in positions_by_vertex[vertex])
        neighbours[site_i.index] = [
            sites[position] for position in sorted(n_shared_vertices)
            if n_shared_vertices[position] >= 3 and sites[position] is not site_i]
    return neighbours
 
