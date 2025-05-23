"""Abstract base class for collections of sites in crystal structures.

This module defines the SiteCollection abstraction, which manages groups of
related Site objects and provides methods for assigning atoms to these sites
based on their positions in a crystal structure.

The SiteCollection class serves as an abstract base class that all specific
site collection types (PolyhedralSiteCollection, SphericalSiteCollection, etc.)
must inherit from. It defines the interface for site-atom assignment logic
and provides common functionality for managing site occupations.

Concrete site collection classes must implement methods to:
- Assign atoms to sites for a given structure
- Analyze structure and update site assignments
- Define structure-specific site relationships like neighboring sites

This class should not be instantiated directly; use one of the concrete
subclasses instead.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Sequence
from pymatgen.core import Structure # type: ignore
from .site import Site

class SiteCollection(ABC):
    """Parent class for collections of sites.

    Collections of specific site types should inherit from this class.

    Attributes:
        sites (list): List of ``Site``-like objects.

    """

    def __init__(self, sites: Sequence[Site]) -> None:
        """Create a SiteCollection object.
        
        Args:
            sites (list): List of ``Site`` objects.
            
        Raises:
            ValueError: If there are duplicate site indices.
        
        """
        self.sites = sites
        
        # Create lookup dictionary for efficient site access by index
        self._site_lookup: dict[int, Site] = {}
        for site in sites:
            if site.index in self._site_lookup:
                raise ValueError(f"Duplicate site index detected: {site.index}. Site indices must be unique.")
            self._site_lookup[site.index] = site

    @abstractmethod
    def assign_site_occupations(self, atoms, structure):
        """Assigns atoms to sites for a specific structure.

        This method should be implemented in the derived subclass

        Args:
            atoms (list(Atom)): List of Atom objects to be assigned to sites.
            struture (pymatgen.Structure): Pymatgen Structure object used to specificy
                the atomic coordinates.

        Returns:
            None

        Notes:
            The atom coordinates should already be consistent with the coordinates
            in `structure`. Recommended usage is via the ``analyse_structure()`` method.

        """
        raise NotImplementedError('assign_site_occupations should be implemented in'
            ' the derived class')

    @abstractmethod
    def analyse_structure(self, atoms, structure):
        """Perform a site analysis for a set of atoms on a specific structure.

        This method should be implemented in the derived subclass.

        Args:
            atoms (list(Atom)): List of Atom objects to be assigned to sites.
            struture (pymatgen.Structure): Pymatgen Structure object used to specificy
                the atomic coordinates.

        Returns:
            None

        """
        raise NotImplementedError('analyse_structure should be implemented in the derived class')

    def neighbouring_sites(self, site_index):
        """If implemented, returns a list of sites that neighbour
        a given site.

        This method should be implemented in the derived subclass.
        
        Args:
            site_index (int): Index of the site to return a list of neighbours for.

        """
        raise NotImplementedError('neighbouring_sites should be implemented'
            'in the derived class')

    def site_by_index(self, index):
        """Returns the site with a specific index.
        
        Args:
            index (int): index for the site to be returned.
        
        Returns:
            (Site)
        
        Raises:
            ValueError: If a site with the specified index is not contained
                in this SiteCollection.
        
        """
        site = self._site_lookup.get(index)
        if site is None:
            raise ValueError(f'No site with index {index} found')
        return site

    def update_occupation(self, site, atom):
        """Updates site and atom attributes for this atom occupying this site.

        Args:
            site (Site): The site to be updated.
            atom (Atom): The atom to be updated.

        Returns:
            None

        Notes:

            This method does the following:

            1. If the atom has moved to a new site, record a old_site --> new_site transition.
            2. Add this atom's index to the list of atoms occupying this site.
            3. Add this atom's fractional coordinates to the list of
               coordinates observed occupying this site.
            4. Assign this atom this site index.

        """
        previous_site_index = None
        if atom.trajectory:
            previous_site_index = atom.trajectory[-1]
        if previous_site_index:
            if previous_site_index != site.index: # this atom has moved
                previous_site = self.site_by_index(previous_site_index)
                previous_site.transitions[site.index] += 1
        site.contains_atoms.append(atom.index)
        site.points.append(atom.frac_coords)
        atom.in_site = site.index

    def reset_site_occupations(self):
        """Occupations of all sites in this site collection are set as empty.

        Args:
            None

        Returns:
            None

        """
        for s in self.sites:
            s.contains_atoms = []

    def sites_contain_points(self,
                             points: np.ndarray,
                             structure: Optional[Structure]=None) -> bool:
        """If implemented, Checks whether the set of sites contain
        a corresponding set of fractional coordinates.
        Args:
            points (np.array): 3xN numpy array of fractional coordinates.
                There should be one coordinate for each site being checked.

        Returns:
            (bool)
            
        Notes:
            Specific SiteCollection subclass implementations may require
            additional arguments to be passed.
        """
        raise NotImplementedError('sites_contain_points() should be'
            ' implemented in the derived class')
