from .site_collection import SiteCollection
from typing import List, Any, Optional, Dict
from .polyhedral_site import PolyhedralSite
from .tools import generate_site_atom_distance_matrix
from .atom import Atom
from .site import Site
from pymatgen.core import Structure # type: ignore
import numpy as np
from scipy.spatial import Delaunay # type: ignore

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
        self.sites: List[PolyhedralSite] = self.sites
        self._neighbouring_sites: Optional[List[PolyhedralSite]]  = None 

    def analyse_structure(self,
            atoms: List[Atom],
            structure: Structure):
        for a in atoms:
            a.assign_coords(structure)
        for s in self.sites:
            s.assign_vertex_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self, atoms: List[Atom], structure: Structure) -> None:
        """Assign atoms to sites by checking sites in order of proximity.

        Args:
            atoms (list(Atom)): List of Atom objects.
            structure (Structure): Crystal structure information.

        Returns:
            None

        Idea:
            1. Reset site occupations
            2. For each atom, check if it is already in a site
            3. If not, sort sites by distance and check them in order
            4. If the atom is in a site, update the site occupation
        """
        self.reset_site_occupations()

        atoms = atoms.copy() # Copy the list to avoid some wonky behaviour

        # First handle atoms that are already assigned to sites
        for atom in atoms:
            if atom.in_site:
                previous_site = self.sites[atom.in_site]
                if previous_site.contains_atom(atom):
                    self.update_occupation(previous_site, atom)
                    atoms.remove(atom)

        if len(atoms) == 0:
            return
        
        # Distance matrix for all sites and atoms by minimum pbc distance
        distance_matrix = generate_site_atom_distance_matrix(self.sites, atoms)

        not_assigned_atoms = []
        for i, atom in enumerate(atoms):
            
            sorted_site_indices = np.argsort(distance_matrix[:, i])
            
            assigned = False
            for site_idx in sorted_site_indices:
                site = self.sites[site_idx]
                if site.contains_atom(atom):
                    self.update_occupation(site, atom)
                    # Modify this part if you want to allow an atom to be in multiple sites.
                    assigned = True
                    break
            if not assigned:
                not_assigned_atoms.append(atom)
        
        if not_assigned_atoms:
            print("Not assigned atoms: ", [atom.index for atom in not_assigned_atoms])
            print(f'Atom frac coords: {not_assigned_atoms[0].frac_coords} for atom {not_assigned_atoms[0].index}')

    def neighbouring_sites(self, index: int) -> List[PolyhedralSite]:
        """Get list of neighboring sites for a given site index.
        
        Args:
            index (int): Index of the site to get neighbors for
            
        Returns:
            List[PolyhedralSite]: List of neighboring polyhedral sites
        """
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
    
    @property
    def neighbouring_sites(self) -> Dict[int, List[PolyhedralSite]]:
        """
        Get all polyhedral sites that are face-sharing neighbours.

        Returns:
            Dict[int, List[PolyhedralSite]]: Dictionary mapping site indices to lists of 
                                            neighbouring PolyhedralSite objects.
        """
        if self._neighbouring_sites is None:
            self._neighbouring_sites = self._construct_neighbouring_sites(self.sites)
        return self._neighbouring_sites

    def _construct_neighbouring_sites(
            self,
            sites: List[PolyhedralSite]) -> Dict[int, List[PolyhedralSite]]:
        """
        Find all polyhedral sites that are face-sharing neighbours.

        Any polyhedral sites that share 3 or more vertices are considered
        to share a face.

        Args:
            sites (List[PolyhedralSite]): List of polyhedral sites to analyze.

        Returns:
            Dict[int, List[PolyhedralSite]]: Dictionary mapping site indices to lists of 
                                            neighbouring PolyhedralSite objects.
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