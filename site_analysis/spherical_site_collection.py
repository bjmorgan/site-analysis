"""Collection manager for spherical sites in crystal structures.

This module provides the SphericalSiteCollection class, which manages a
collection of SphericalSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The SphericalSiteCollection extends the base SiteCollection class with
specific functionality for spherical sites, including maintaining a map of 
neighbouring sites within a distance cutoff and optimized atom assignment 
that leverages spatial relationships and learned transition patterns for 
improved performance in large systems.

For atom assignment, the collection uses a priority-based approach:

1. First check if an atom is still in its most recently assigned site
2. Then check observed transition destinations from that site in decreasing 
   frequency order
3. Then check neighbouring sites within the distance cutoff of the most recent site 
   (if these have not yet been checked)
4. Finally check all remaining sites if not found in the priority categories

This handles overlapping spherical sites in a consistent way - if an atom is 
in a region where multiple sites overlap, it will remain assigned to its 
original site as long as it stays within that site's volume. This persistence 
can be useful for tracking atoms through small oscillations without generating 
spurious site transitions.
"""

import numpy as np
from typing import Iterator
from pymatgen.core import Structure, Lattice
from site_analysis.atom import Atom
from site_analysis.spherical_site import SphericalSite
from site_analysis.site_collection import SiteCollection

class SphericalSiteCollection(SiteCollection):


    def __init__(self,
        sites: list[SphericalSite],
        neighbour_cutoff: float = 10.0) -> None:
        """A collection of SphericalSite objects with optimised atom assignment.
        
        Extends the base SiteCollection class with specific functionality for 
        spherical sites, including maintaining a map of neighboring spherical 
        sites within a distance cutoff and implementing optimized atom assignment 
        based on spatial relationships and learned transition patterns.
        
        The collection uses a priority-based site checking approach that leverages:
        - Most recently occupied sites (spatial locality)
        - Observed transition frequencies (learned behavior)  
        - Distance-based site neighborhoods (spatial relationships)
        
        This provides improved performance for large systems while handling 
        overlapping sites consistently.
        
        Args:
            sites (list): List of ``SphericalSite`` objects.
            neighbour_cutoff (float, optional): Distance cutoff in Ångström for 
                determining neighbouring sites. Default is 10.0 Å.
        
        Attributes:
            sites (list): List of ``SphericalSite`` objects.
        
        """
        for s in sites:
            if not isinstance(s, SphericalSite):
                raise TypeError
        super(SphericalSiteCollection, self).__init__(sites)
        self.sites = self.sites  # type: list[SphericalSite]
        self._neighbouring_sites: dict[int, list[SphericalSite]] | None = None
        self._current_lattice: Lattice | None = None
        self._neighbour_cutoff = neighbour_cutoff
        
    def neighbouring_sites(self, index: int) -> list[SphericalSite]:
        """Returns the neighbouring sites for a given site index.
        
        Neighbours are defined as sites within 10.0 Å of the given site.
        The neighbour calculation is performed lazily on first access and cached.
        
        Args:
            index (int): Index of the site to return neighbours for.
            
        Returns:
            list[SphericalSite]: List of neighbouring SphericalSite objects.
            
        Raises:
            RuntimeError: If no lattice is available for distance calculations.
        """
        if self._neighbouring_sites is None:
            lattice = self._get_current_lattice()
            if lattice is None:
                raise RuntimeError("No lattice available for neighbour calculation. "
                                "Call assign_site_occupations or analyse_structure first.")
            self._neighbouring_sites = self._calculate_all_neighbouring_sites(lattice)
        
        return self._neighbouring_sites[index]
        
    def _get_current_lattice(self) -> Lattice | None:
        """Get the current lattice for distance calculations.
        
        Returns:
            Lattice or None: The current lattice if available, None otherwise.
        """
        return self._current_lattice
        
    def _calculate_all_neighbouring_sites(self,
        lattice: Lattice
    ) -> dict[int, list[SphericalSite]]:
        """Calculate all site-site distances and determine neighbours within 10.0 Å.
        
        Args:
            lattice : Pymatgen Lattice object for distance calculations.
            
        Returns:
            dict: Dictionary mapping site indices to lists
                of neighbouring SphericalSite objects within 10.0 Å, sorted by 
                increasing distance.
        """
        if not self.sites:
            return {}
        
        # Get coordinates of all sites
        site_coords = np.array([site.frac_coords for site in self.sites])
        
        # Calculate all pairwise distances
        distance_matrix = lattice.get_all_distances(site_coords, site_coords)
        
        # Set diagonal to infinity to exclude self-references
        np.fill_diagonal(distance_matrix, np.inf)
        
        # Create mask for valid neighbors
        neighbor_mask = distance_matrix <= self._neighbour_cutoff
        
        # Build neighbour dictionary
        neighbours = {}
        for i, site_i in enumerate(self.sites):
            neighbor_indices = np.where(neighbor_mask[i])[0]
            if len(neighbor_indices) > 0:
                # Sort neighbor indices by distance
                neighbor_distances = distance_matrix[i, neighbor_indices]
                sorted_indices = neighbor_indices[np.argsort(neighbor_distances)]
                neighbours[site_i.index] = [self.sites[j] for j in sorted_indices]
            else:
                neighbours[site_i.index] = []
        
        return neighbours

    def analyse_structure(self,
            atoms: list[Atom],
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
        atoms: list[Atom], 
        structure: Structure) -> None:
        """Assign atoms to spherical sites based on their positions.
        
        This method implements an optimised assignment logic using a priority-based
        site checking approach:
        1. First check if an atom is still in its most recently assigned site
        2. Then check sites we have observed transitions to from the most recent site,
        in order of reverse observed frequency  
        3. Then check neighbouring sites of the most recent site
        4. Finally check all remaining sites if not found in the priority categories
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
        """
        # Store lattice for neighbour calculations
        self._current_lattice = structure.lattice
        
        self.reset_site_occupations()
        for atom in atoms:
            atom.in_site = None
            
            # Check sites in priority order until found
            for site in self._get_priority_sites(atom):
                if site.contains_atom(atom, lattice=structure.lattice):
                    self.update_occupation(site, atom)
                    break
                    
    def _get_priority_sites(self, atom: Atom) -> Iterator[SphericalSite]:
        """Generator that yields sites in priority order for optimised atom assignment.
        
        This generator implements an optimised site-checking sequence:
        1. First yield the most recent site from atom.most_recent_site (if atom has trajectory history)
        2. Then yield transition destinations from that site in frequency order using site.most_frequent_transitions()
        3. Then yield neighbouring sites of the most recent site using self.neighbouring_sites() 
        4. Finally yield all remaining sites not already checked
        
        Each site is yielded at most once by tracking checked indices. If the atom
        has no trajectory history (atom.most_recent_site is None), steps 1-3 are
        skipped and only step 4 (all sites) is performed.
        
        The optimisation reduces average search complexity from O(N) to O(k) where:
        - N = total number of sites in the collection
        - k = typical number of sites checked before finding atom (usually much smaller)
        
        Args:
            atom: Atom object with trajectory history used to determine
                site priorities.
        
        Yields:
            SphericalSite: Sites in optimal checking order, stopping when the
                calling method finds the atom.
        
        Notes:
            This method is called internally by assign_site_occupations()
            and should not be called directly.
        """
        checked_indices = set()
        
        # 1. Most recent site first (if atom has trajectory)
        most_recent_index = atom.most_recent_site
        if most_recent_index is not None:
            most_recent_site = self.site_by_index(most_recent_index)
            yield most_recent_site
            checked_indices.add(most_recent_site.index)
            
            # 2. Transition destinations in frequency order
            for dest_index in most_recent_site.most_frequent_transitions():
                if dest_index not in checked_indices:
                    dest_site = self.site_by_index(dest_index)
                    yield dest_site
                    checked_indices.add(dest_index)
            
            # 3. neighbouring sites not already checked
            for neighbour_site in self.neighbouring_sites(most_recent_index):
                if neighbour_site.index not in checked_indices:
                    yield neighbour_site
                    checked_indices.add(neighbour_site.index)
        
        # 4. All remaining sites
        for site in self.sites:
            if site.index not in checked_indices:
                yield site

 
