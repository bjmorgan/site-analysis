"""Collection manager for spherical sites in crystal structures.

This module provides the SphericalSiteCollection class, which manages a
collection of SphericalSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The SphericalSiteCollection extends the base SiteCollection class with
specific functionality for spherical sites, focusing primarily on efficient
atom assignment.

For atom assignment, the collection implements an efficient prioritization:
1. First check if an atom is still in its previously assigned site
2. If not, sequentially check all sites until a match is found
3. The first site found to contain the atom claims it (no further checks)

This approach handles overlapping spherical sites in a consistent way - 
if an atom is in a region where multiple sites overlap, it will remain assigned
to its original site as long as it stays within that site's volume. This 
persistence can be useful for tracking atoms through small oscillations
without generating spurious site transitions.
"""

import numpy as np
from typing import List
from pymatgen.core import Structure
from site_analysis.atom import Atom
from site_analysis.site_collection import SiteCollection

class SphericalSiteCollection(SiteCollection):


    def analyse_structure(self,
            atoms: List[Atom],
            structure: Structure) -> None:
        """Analyze a structure to assign atoms to spherical sites.
        
        This method:
        1. Assigns fractional coordinates to each atom based on the structure
        2. Delegates to assign_site_occupations to determine which atoms
           belong in which sites
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
            
        Returns:
            None
        """
        for a in atoms:
            a.assign_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self,
            atoms: List[Atom],
            structure: Structure) -> None:
        """Assign atoms to spherical sites based on their positions.
        
        This method implements the spherical site assignment logic:
        1. All site occupations are reset (emptied) at the beginning
        2. For each atom, first check if it's still in its previously assigned site
           (prioritizing site persistence when atoms are in overlapping regions)
        3. If the atom has moved out of its previous site (or had no previous assignment),
           check all sites sequentially until a site containing the atom is found
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
            
        Returns:
            None
            
        Note:
            This approach optimizes for small atom movements and handles overlapping
            sites consistently by prioritizing an atom's previous site assignment.
        """
        self.reset_site_occupations()
        for atom in atoms:
            # Check current site or most recent site first
            most_recent_site = atom.most_recent_site
            if most_recent_site is not None:
                site = self.site_by_index(most_recent_site)
                if site and site.contains_atom(atom, structure.lattice):
                    self.update_occupation(site, atom)
                    continue
            # Reset in_site since we didn't find the atom in its previous site
            atom.in_site = None
            for site in self.sites:
                if site.contains_atom(atom, structure.lattice):
                    self.update_occupation(site, atom)
                    break

 
