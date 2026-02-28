"""Collection manager for polyhedral sites in crystal structures.

This module provides the PolyhedralSiteCollection class, which manages a
collection of PolyhedralSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The PolyhedralSiteCollection extends the base SiteCollection class with
specific functionality for polyhedral sites, including:
- Maintaining a map of neighbouring polyhedral sites that share faces
- Optimised atom assignment via the PriorityAssignmentMixin
- Precomputed distance-ranked site ordering when reference centres are
  available

The module also includes utility functions:
- construct_neighbouring_sites: analyses polyhedral sites to determine which
  ones are face-sharing neighbours (sharing three or more vertices).
- _collect_reference_centres: extracts reference centres from polyhedral
  sites for distance-ranked ordering.
"""

from .site_collection import SiteCollection, PriorityAssignmentMixin
from .polyhedral_site import PolyhedralSite
from .atom import Atom
from .site import Site
from .tools import x_pbc
from pymatgen.core import Structure # type: ignore
import numpy as np


class PolyhedralSiteCollection(PriorityAssignmentMixin, SiteCollection):
    """A collection of PolyhedralSite objects.
    
    Extends the base SiteCollection class with specific functionality for
    polyhedral sites, including maintaining a map of neighbouring polyhedral
    sites that share faces and implementing optimised atom assignment based
    on spatial relationships and learned transition patterns.
    
    Attributes:
        sites (list): List of ``PolyhedralSite`` objects.
    
    """

    def __init__(self,
            sites: list[Site]) -> None:
        """Create a PolyhedralSiteCollection instance.

        Args:
            sites (list(PolyhedralSite)): List of PolyhedralSite objects.

        Returns:
            None

        """
        for s in sites:
            if not isinstance(s, PolyhedralSite):
                raise TypeError
        super().__init__(sites)
        self.sites: list[PolyhedralSite]
        self._neighbouring_sites = construct_neighbouring_sites(self.sites)
        centres, site_indices = _collect_reference_centres(self.sites)
        if centres is not None:
            self._init_priority_ranking(centres, site_indices)

    def analyse_structure(self,
            atoms: list[Atom],
            structure: Structure):
        for a in atoms:
            a.assign_coords(structure)
        all_frac_coords = structure.frac_coords
        lattice = structure.lattice
        for s in self.sites:
            s.notify_structure_changed(all_frac_coords, lattice)
        self.assign_site_occupations(atoms, structure)
                    
    def assign_site_occupations(self, atoms, structure) -> None:
        """Assign atoms to polyhedral sites based on their positions.
        
        This method implements an optimised assignment logic using a priority-based
        site checking approach.
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
        """
        self.reset_site_occupations()
        for atom in atoms:
            atom.in_site = None
            pbc_images = x_pbc(atom.frac_coords)

            # Check sites in priority order until found
            for site in self._get_priority_sites(atom):
                if site.contains_atom(atom, pbc_images=pbc_images):
                    self.update_occupation(site, atom)
                    break
    
    def neighbouring_sites(self,
            index: int) -> list[PolyhedralSite]:
        return self._neighbouring_sites[index] 

    def sites_contain_points(self,
            points: np.ndarray,
            structure: Structure | None=None) -> bool:
        """Checks whether the set of sites contain 
        a corresponding set of fractional coordinates.

        Args:
            points (np.array): 3xN numpy array of fractional coordinates.
                There should be one coordinate for each site being checked.
            structure (Structure): Pymatgen Structure used to define the
                vertex coordinates of each polyhedral site.
        
        Returns:
            (bool)

        """
        if not isinstance(structure, Structure):
            raise TypeError(f"Expected a Structure, got {type(structure).__name__}")
        check = all([s.contains_point(p,structure) for s, p in zip(self.sites, points)])
        return check

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
    centres = []
    for s in sites:
        if s.reference_center is None:
            return None, [s.index for s in sites]
        centres.append(s.reference_center)
    return np.array(centres), [s.index for s in sites]


def construct_neighbouring_sites(
        sites: list[PolyhedralSite]) -> dict[int, list[PolyhedralSite]]:
    """
    Find all polyhedral sites that are face-sharing neighbours.

    Any polyhedral sites that share 3 or more vertices are considered
    to share a face.

    Args:
        sites: List of PolyhedralSite objects to check for shared faces.

    Returns:
        (dict): Dictionary of `int`: `list` entries.
            Keys are site indices. Values are lists of ``PolyhedralSite`` objects.

    """
    neighbours: dict[int, list[PolyhedralSite]] = {}
    for site_i in sites:
        neighbours[site_i.index] = []
        for site_j in sites:
            if site_i is site_j:
                continue
            # 3 or more common vertices indicated a shared face.
            n_shared_vertices = len(set(site_i.vertex_indices) & set(site_j.vertex_indices))
            if n_shared_vertices >= 3:
                neighbours[site_i.index].append(site_j)
    return neighbours
 
