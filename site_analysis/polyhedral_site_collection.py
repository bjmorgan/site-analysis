"""Collection manager for polyhedral sites in crystal structures.

This module provides the PolyhedralSiteCollection class, which manages a
collection of PolyhedralSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The PolyhedralSiteCollection extends the base SiteCollection class with
specific functionality for polyhedral sites, including:
- Maintaining a map of neighboring polyhedral sites that share faces
- Efficiently assigning atoms to sites based on their positions
- Checking whether a set of sites contains a corresponding set of points

The module also includes a utility function, construct_neighbouring_sites,
which analyzes a set of polyhedral sites to determine which ones are
face-sharing neighbors (defined as sites sharing three or more vertices).

For atom assignment, the collection implements an efficient prioritization:
1. First check if an atom is still in its previously assigned site
2. If not, sequentially check all sites until a match is found
3. The first site found to contain the atom claims it (no further checks)

This approach optimizes performance when atom movements between structures
are small, as most atoms will remain in their previously assigned sites.
"""

from .site_collection import SiteCollection
from typing import List, Any, Optional, Dict
from .polyhedral_site import PolyhedralSite
from .atom import Atom
from .site import Site
from pymatgen.core import Structure # type: ignore
import numpy as np

class PolyhedralSiteCollection(SiteCollection):
    """A collection of PolyhedralSite objects.

    Attributes:
        sites (list): List of ``Site``-like objects.

    """

    def __init__(self,
            sites: List[Site]) -> None:
        """Create a PolyhedralSiteCollection instance.

        Args:
            sites (list(PolyhedralSite)): List of PolyhedralSite objects.

        Returns:
            None

        """
        for s in sites:
            if not isinstance(s, PolyhedralSite):
                raise TypeError
        super(PolyhedralSiteCollection, self).__init__(sites)
        self.sites = self.sites # type: List[PolyhedralSite]
        self._neighbouring_sites = construct_neighbouring_sites(self.sites)

    def analyse_structure(self,
            atoms: List[Atom],
            structure: Structure):
        for a in atoms:
            a.assign_coords(structure)
        for s in self.sites:
            s.assign_vertex_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self, atoms, structure):
        """Assign atoms to polyhedral sites based on their positions.
        
        This method implements an improved assignment logic:
        1. All site occupations are reset (emptied) at the beginning
        2. For each atom, check if it's still in its current site, or if not, check
        its most recent site from trajectory history
        3. If the atom is not in either of these sites, check all sites sequentially
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
        """
        self.reset_site_occupations()
        for atom in atoms:
            # Check current site or most recent site first
            most_recent_site = atom.most_recent_site
            if most_recent_site is not None:
                site = self.site_by_index(most_recent_site)
                if site and site.contains_atom(atom):
                    self.update_occupation(site, atom)
                    continue
            # Reset in_site since we didn't find the atom in its previous site
            atom.in_site = None
            # Check all sites sequentially
            for site in self.sites:
                if site.contains_atom(atom):
                    self.update_occupation(site, atom)
                    break

    def neighbouring_sites(self,
            index: int) -> List[PolyhedralSite]:
        return self._neighbouring_sites[index] 

    def sites_contain_points(self,
            points: np.ndarray,
            structure: Optional[Structure]=None) -> bool:
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
        assert isinstance(structure, Structure)
        check = all([s.contains_point(p,structure) for s, p in zip(self.sites, points)])
        return check

def construct_neighbouring_sites(
        sites: List[PolyhedralSite]) -> Dict[int, List[PolyhedralSite]]:
    """
    Find all polyhedral sites that are face-sharing neighbours.

    Any polyhedral sites that share 3 or more vertices are considered
    to share a face.

    Args:
        None

    Returns:
        (dict): Dictionary of `int`: `list` entries. 
            Keys are site indices. Values are lists of ``PolyhedralSite`` objects.

    """
    neighbours: Dict[int, List[PolyhedralSite]] = {}
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
 
