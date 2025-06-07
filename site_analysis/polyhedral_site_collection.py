"""Collection manager for polyhedral sites in crystal structures.

This module provides the PolyhedralSiteCollection class, which manages a
collection of PolyhedralSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The PolyhedralSiteCollection extends the base SiteCollection class with
specific functionality for polyhedral sites, including:
- Maintaining a map of neighbouring polyhedral sites that share faces
- Optimised atom assignment using priority-based site checking
- Using observed transition patterns between sites for performance optimisation

The module also includes a utility function, construct_neighbouring_sites,
which analyses a set of polyhedral sites to determine which ones are
face-sharing neighbours (defined as sites sharing three or more vertices).

For atom assignment, the collection implements an intelligent optimisation
that reduces average search complexity from O(N) to O(k):

1. First check if an atom is still in its most recently assigned site
2. Then check observed transition destinations from that site in decreasing 
    frequency order
3. Then check neighbouring sites that share faces with the most recent site 
    (if these have not yet been checked)
4. Finally check all remaining sites if not found in the priority categories

This approach leverages both spatial relationships (face-sharing neighbors)
and learned behavior (observed transition patterns) to dramatically reduce
the number of containment checks required.
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
    
    Extends the base SiteCollection class with specific functionality for 
    polyhedral sites, including maintaining a map of neighboring polyhedral 
    sites that share faces and implementing optimized atom assignment based 
    on spatial relationships and learned transition patterns.
    
    Attributes:
        sites (list): List of ``PolyhedralSite`` objects.
    
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
            
            # Check sites in priority order until found
            for site in self._get_priority_sites(atom):
                if site.contains_atom(atom):
                    self.update_occupation(site, atom)
                    break
    
    def _get_priority_sites(self, atom):
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
            atom (Atom): Atom object with trajectory history used to determine
                site priorities.
        
        Yields:
            PolyhedralSite: Sites in optimal checking order, stopping when the
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
 
