"""Collection manager for Voronoi sites in crystal structures.

This module provides the VoronoiSiteCollection class, which manages a
collection of VoronoiSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The VoronoiSiteCollection extends the base SiteCollection class with
specific functionality for Voronoi sites, implementing a different assignment
logic than other site collections:

For atom assignment, the collection:
1. Calculates distances from each site center to each atom
2. Assigns each atom to the site with the nearest center
3. Uses the structure's lattice to correctly handle distances across
   periodic boundaries

Unlike other site types where individual sites can determine containment,
Voronoi site assignment is a global operation that depends on the relative
positions of all sites. This collection efficiently implements the Voronoi
tessellation logic using distance calculations rather than explicit
geometric construction of the Voronoi cells.
"""

import numpy as np
from pymatgen.core import Structure
from site_analysis.site_collection import SiteCollection
from site_analysis.atom import Atom
from site_analysis.site import Site
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.distances import all_mic_distances

class VoronoiSiteCollection(SiteCollection):

    def __init__(self,
            sites: list[Site]) -> None:
        """Create a VoronoiSiteCollection instance.

        Args:
            sites (list(VoronoiSite)): list of VoronoiSite objects.

        Returns:
            None

        """
        for s in sites:
            if not isinstance(s, VoronoiSite):
                raise TypeError
        super(VoronoiSiteCollection, self).__init__(sites)
        self.sites: list[VoronoiSite]

    def analyse_structure(self,
                          atoms: list[Atom],
                          structure: Structure) -> None:
        """Analyze a structure to assign atoms to Voronoi sites.
        
        This method:
        1. Assigns fractional coordinates to each atom based on the structure
        2. Delegates to assign_site_occupations to determine which atoms
           belong in which sites based on Voronoi tessellation principles
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
            
        Returns:
            None
        """
        all_frac_coords = structure.frac_coords
        for a in atoms:
            a.assign_coords(all_frac_coords)
        self.assign_site_occupations(atoms, structure.lattice.matrix)

    def assign_site_occupations(self,
                                atoms: list[Atom],
                                lattice_matrix: np.ndarray) -> None:
        """Assign atoms to Voronoi sites based on closest site centres.

        Uses minimum-image convention distances to assign each atom to the
        nearest site centre.

        Args:
            atoms: List of Atom objects to be assigned to sites.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.
        """
        self.reset_site_occupations()
        if not atoms:
            return
        site_coords = np.array([s.frac_coords for s in self.sites])
        atom_coords = np.array([a.frac_coords for a in atoms])
        dist_matrix = all_mic_distances(site_coords, atom_coords, lattice_matrix)
        site_list_indices = np.argmin(dist_matrix, axis=0)
        for atom, site_list_index in zip(atoms, site_list_indices):
            site = self.sites[site_list_index]
            self.update_occupation(site, atom)

    
 
