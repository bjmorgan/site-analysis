"""Collection manager for polyhedral sites in crystal structures.

This module provides the PolyhedralSiteCollection class, which manages a
collection of PolyhedralSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The PolyhedralSiteCollection extends the base SiteCollection class with
specific functionality for polyhedral sites, including:

- Maintaining a map of neighbouring polyhedral sites that share three or
  more vertex indices
- Optimised atom assignment via the PriorityAssignmentMixin
- Distance-ranked site ordering when reference centres are available

The module also includes utility functions:

- ``construct_neighbouring_sites``: analyses polyhedral sites to determine
  which ones share three or more vertex indices.
- ``_collect_reference_centres``: extracts reference centres from polyhedral
  sites for distance-ranked ordering.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations

from .site_collection import SiteCollection, PriorityAssignmentMixin
from .polyhedral_site import PolyhedralSite
from .atom import Atom
from .tools import x_pbc
from .faces import Face, FaceKey, _faces_and_keys
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
            ValueError: If a site's reference centre is not three finite
                numbers.

        """
        sites = list(sites)
        for s in sites:
            if not isinstance(s, PolyhedralSite):
                raise TypeError(f"Expected PolyhedralSite, got {type(s).__name__}")
        super().__init__(sites)
        self.sites: list[PolyhedralSite]
        self._neighbouring_sites = construct_neighbouring_sites(self.sites)
        self._adjacency: csr_array | None = None
        self._shared_faces: dict[tuple[int, int], tuple[Face, ...]] = {}
        centres, site_indices = _collect_reference_centres(self.sites)
        if centres is not None:
            self._init_priority_ranking(centres, site_indices)

    def analyse_structure(self,
            atoms: list[Atom],
            structure: Structure) -> None:
        """Analyse a structure to assign atoms to polyhedral sites.

        Assigns coordinates to atoms, notifies sites of the new
        structure, and assigns atoms to sites.

        Args:
            atoms: List of Atom objects to be assigned to sites.
            structure: Pymatgen Structure containing atom positions.
        """
        all_frac_coords = structure.frac_coords
        for a in atoms:
            a.assign_coords(all_frac_coords)
        lattice_matrix = structure.lattice.matrix
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
            pbc_images = x_pbc(atom.frac_coords)

            # Check sites in priority order until found
            for site in self._get_priority_sites(atom, lattice_matrix):
                if site.contains_atom(atom, pbc_images=pbc_images):
                    self.update_occupation(site, atom)
                    break
    
    def neighbouring_sites(self,
            index: int) -> list[PolyhedralSite]:
        return self._neighbouring_sites[index] 

    def assign_faces(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Assign every site's faces, and find which faces sites share.

        Faces are found from the convex hull of each site's vertices, merging
        hull triangles that lie in one plane (to within 0.01 Angstrom). A
        polygon that is not planar to within this tolerance gives separate
        triangular faces, which the sites on either side of it divide
        differently, so they do not share those faces.

        Args:
            frac_coords: Fractional coordinates of all atoms, indexed by
                atom index.
            lattice_matrix: (3, 3) lattice matrix, rows as lattice vectors.

        Raises:
            ValueError: If a site's vertices do not form a polyhedron.
        """
        coords = np.asarray(frac_coords, dtype=float)
        self._assign_faces_from_vertex_coords(
            [coords[site.vertex_indices] for site in self.sites], lattice_matrix)

    def _assign_faces_from_vertex_coords(self,
            vertex_frac_coords: Sequence[np.ndarray],
            lattice_matrix: np.ndarray) -> None:
        """Assign faces from each site's vertex coordinates.

        Vertex coordinates are wrapped into the unit cell first. Two sites
        share a face when one lattice translation maps the atoms of a face of
        one onto the atoms of a face of the other.

        Args:
            vertex_frac_coords: For each site, in the order of ``sites``,
                the fractional coordinates of its vertices in the order of
                its ``vertex_indices``.
            lattice_matrix: (3, 3) lattice matrix, rows as lattice vectors.

        Raises:
            ValueError: If a site's vertices do not form a polyhedron.
        """
        sites_by_key: defaultdict[FaceKey, list[tuple[int, Face]]] = defaultdict(list)
        site_faces = []
        for site, coords in zip(self.sites, vertex_frac_coords):
            wrapped = np.asarray(coords, dtype=float)
            wrapped = wrapped - np.floor(wrapped)
            try:
                faces_and_keys = _faces_and_keys(
                    site.vertex_indices, wrapped, site.reference_center, lattice_matrix)
            except ValueError as error:
                raise ValueError(f"Site {site.index}: {error}") from error
            site_faces.append(tuple(face for face, _ in faces_and_keys))
            for face, key in faces_and_keys:
                sites_by_key[key].append((site.index, face))
        shared: defaultdict[tuple[int, int], set[Face]] = defaultdict(set)
        for entries in sites_by_key.values():
            for (i, face), (j, _) in combinations(entries, 2):
                if i != j:
                    shared[(min(i, j), max(i, j))].add(face)
        for site, faces in zip(self.sites, site_faces):
            site.faces = faces
        self._shared_faces = {
            pair: tuple(sorted(faces, key=lambda face: face.vertex_indices))
            for pair, faces in shared.items()}
        rows = [i for i, j in shared] + [j for i, j in shared]
        columns = [j for i, j in shared] + [i for i, j in shared]
        n = max((site.index for site in self.sites), default=-1) + 1
        self._adjacency = csr_array(
            (np.ones(len(rows), dtype=bool),
             (np.array(rows, dtype=np.int64), np.array(columns, dtype=np.int64))),
            shape=(n, n))

    def _check_faces_assigned(self) -> csr_array:
        """The adjacency matrix, if faces have been assigned.

        Raises:
            ValueError: If faces have not been assigned.
        """
        if self._adjacency is None:
            raise ValueError(
                "Faces have not been assigned: call Trajectory.assign_faces(structure)")
        return self._adjacency

    def site_adjacency(self) -> csr_array:
        """Return which sites share a face.

        Returns:
            A copy of the symmetric boolean sparse matrix, indexed by
            ``site.index``, whose entry ``(i, j)`` is True when sites ``i``
            and ``j`` share a face.

        Raises:
            ValueError: If faces have not been assigned.
        """
        return self._check_faces_assigned().copy()

    def shared_faces(self, i: int, j: int) -> tuple[Face, ...]:
        """Return the faces two sites share.

        Args:
            i: Index of one site.
            j: Index of the other site.

        Returns:
            The shared faces, sorted by vertex indices; empty if the sites
            share no face.

        Raises:
            ValueError: If faces have not been assigned.
        """
        self._check_faces_assigned()
        return self._shared_faces.get((min(i, j), max(i, j)), ())

    def sites_contain_points(self,
            points: np.ndarray,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> bool:
        """Check whether the set of sites contain corresponding points.

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
                of sites.
        """
        if len(points) != len(self.sites):
            raise ValueError(
                f"Expected {len(self.sites)} points (one per site), "
                f"got {len(points)}"
            )
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

    Raises:
        ValueError: If a site has a reference centre that is not three
            finite numbers, even if another site lacks one.
    """
    centres = []
    for s in sites:
        if s.reference_center is None:
            continue
        try:
            centre = np.asarray(s.reference_center, dtype=np.float64)
            valid = centre.shape == (3,) and bool(np.isfinite(centre).all())
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"reference centre of site {s.index} must be three finite numbers, "
                f"got {s.reference_center}")
        centres.append(centre)
    site_indices = [s.index for s in sites]
    if len(centres) < len(sites):
        return None, site_indices
    return np.array(centres), site_indices


def construct_neighbouring_sites(
        sites: list[PolyhedralSite]) -> dict[int, list[PolyhedralSite]]:
    """
    Find, for each polyhedral site, the other sites that share three or more
    vertex indices with it.

    Args:
        sites: List of PolyhedralSite objects to check for shared vertices.

    Returns:
        (dict): Dictionary of `int`: `list` entries.
            Keys are site indices. Values are lists of ``PolyhedralSite``
            objects, in the order they appear in ``sites``.

    """
    vertex_sets = [set(site.vertex_indices) for site in sites]
    positions_by_vertex: defaultdict[int, list[int]] = defaultdict(list)
    for position, vertices in enumerate(vertex_sets):
        for vertex in vertices:
            positions_by_vertex[vertex].append(position)
    neighbours: dict[int, list[PolyhedralSite]] = {}
    for site_i, vertices in zip(sites, vertex_sets):
        n_shared_vertices = Counter(
            position for vertex in vertices for position in positions_by_vertex[vertex])
        neighbours[site_i.index] = [
            sites[position] for position in sorted(n_shared_vertices)
            if n_shared_vertices[position] >= 3 and sites[position] is not site_i]
    return neighbours
 
