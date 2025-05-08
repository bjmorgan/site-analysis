from .site_collection import SiteCollection
import numpy as np
from .atom import Atom
from pymatgen.core import Structure
from typing import List
from .site import Site
from .voronoi_site import VoronoiSite

class VoronoiSiteCollection(SiteCollection):

    def __init__(self,
            sites: List[Site]) -> None:
        """Create a VoronoiSiteCollection instance.

        Args:
            sites (list(VoronoiSite)): List of VoronoiSite objects.

        Returns:
            None

        """
        for s in sites:
            if not isinstance(s, VoronoiSite):
                raise TypeError
        super(VoronoiSiteCollection, self).__init__(sites)
        self.sites = self.sites # type: List[VoronoiSite]

    def analyse_structure(self,
                          atoms: List[Atom],
                          structure: Structure) -> None:
        for a in atoms:
            a.assign_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self,
                                atoms: List[Atom],
                                structure: Structure):
        self.reset_site_occupations()
        if not atoms:
            return
        lattice = structure.lattice
        site_coords = np.array([s.frac_coords for s in self.sites])
        atom_coords = np.array([a.frac_coords for a in atoms])
        dist_matrix = lattice.get_all_distances(site_coords, atom_coords)
        site_list_indices = np.argmin(dist_matrix, axis=0)
        for atom, site_list_index in zip( atoms, site_list_indices):
            site = self.sites[site_list_index]
            self.update_occupation(site, atom)

    
 
