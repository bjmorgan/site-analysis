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

import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable

from .faces import Face, _find_shared_faces, _narrow_cell_hint
from .site_collection import SiteCollection, PriorityAssignmentMixin
from .polyhedral_site import PolyhedralSite
from .polyhedron_surface import _SharedFaceError
from .atom import Atom
from pymatgen.core import Structure
import numpy as np
from scipy.sparse import csr_array


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
        with ``PolyhedralSite.fix_triangles``: the convex hull of its
        vertices, with each face it shares triangulated by a rule that
        depends only on the face's atoms, so that the sites sharing it
        triangulate it identically. The faces and triangles replace any
        earlier ones, and are kept when the collection is reset. If the
        faces differ from those found before, a ``UserWarning`` says so,
        with the numbers of faces added and removed. If an error is
        raised, the earlier faces are kept, as are the triangles of the
        sites not yet fixed. Overlapping sites are not supported.

        Args:
            frac_coords: Fractional coordinates of the atoms, shape
                ``(n_atoms, 3)``, indexed by atom index.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Raises:
            ValueError: If a site has a vertex index that is not an atom of
                the structure (see ``validate_atom_count``); the vertex
                coordinates are not finite; two sites coincide or one lies
                inside another; or the atoms two sites share cannot be
                triangulated identically on both, naming the two sites.
            RuntimeError: If a site's vertices are degenerate, or its
                triangles do not enclose its centre, as when the fan of a
                shared face folds through it.
        """
        frac_coords = np.asarray(frac_coords, dtype=float)
        lattice_matrix = np.asarray(lattice_matrix, dtype=float)
        self.validate_atom_count(len(frac_coords))
        faces = _find_shared_faces(self.sites, frac_coords, lattice_matrix)
        if self._shared_faces is not None and set(faces) != set(self._shared_faces):
            added = len(set(faces) - set(self._shared_faces))
            removed = len(set(self._shared_faces) - set(faces))
            warnings.warn(
                f"The shared faces differ from those found before ({added} added, {removed} "
                "removed), so the connectivity of the sites has changed",
                UserWarning, stacklevel=2)
        sites_by_index = {site.index: site for site in self.sites}
        faces_by_site: defaultdict[int, list[Face]] = defaultdict(list)
        for face in faces:
            for i in face.site_indices:
                faces_by_site[i].append(face)
        for index in sorted(sites_by_index):
            site_faces = faces_by_site[index]
            try:
                sites_by_index[index].fix_triangles(
                    frac_coords, lattice_matrix, [face.vertex_indices for face in site_faces])
            except _SharedFaceError as error:
                face = next(face for face in site_faces if face.vertex_indices == error.atoms)
                i, j = face.site_indices
                raise ValueError(
                    f"Sites {i} and {j}: {error}; the sites may overlap"
                    f"{_narrow_cell_hint(sites_by_index[i], sites_by_index[j])}") from error
        self._shared_faces = faces

    def shared_faces(self) -> tuple[Face, ...]:
        """Return the faces shared by pairs of sites.

        Returns:
            The shared faces, sorted by site indices then vertex indices.

        Raises:
            RuntimeError: If the shared faces have not been found.
        """
        if self._shared_faces is None:
            raise RuntimeError(
                "Shared faces have not been found: analyse a structure, or call "
                "find_shared_faces with one")
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
 
