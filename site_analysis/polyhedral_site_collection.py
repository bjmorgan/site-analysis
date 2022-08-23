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

    def assign_site_occupations(self,
                                atoms: List[Atom],
                                structure: Structure):
        self.reset_site_occupations()
        for atom in atoms:
            if atom.in_site:
                # first check the site last occupied
                previous_site = next(s for s in self.sites if s.index == atom.in_site)
                if previous_site.contains_atom(atom):
                    self.update_occupation( previous_site, atom )
                    continue # atom has not moved
                else: # default is atom does not occupy any sites
                    atom.in_site = None
            for s in self.sites:
                if s.contains_atom(atom):
                    self.update_occupation( s, atom )
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
 
