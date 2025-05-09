"""Trajectory analysis for tracking site occupations over time.

This module provides the Trajectory class, which is responsible for analyzing
and tracking atom movements through crystallographic sites in a simulation
trajectory.

The Trajectory class manages the relationship between atoms and sites, analyzes
structures to assign atoms to sites, and records the movement history of atoms
between sites over time.

Key functionality includes:
- Assigning atoms to sites based on their positions in a structure
- Tracking atom migrations between sites over a sequence of structures
- Recording site occupation and transition data
- Supporting different site definitions via appropriate SiteCollection types

Note:
    Trajectory objects should typically be created using the TrajectoryBuilder class
    rather than directly instantiated. The builder provides an interface for
    configuring all aspects of the trajectory:

    >>> from site_analysis.builders import TrajectoryBuilder
    >>> trajectory = (TrajectoryBuilder()
    ...              .with_structure(structure)
    ...              .with_mobile_species("Li")
    ...              .with_spherical_sites(centres=[[0.5, 0.5, 0.5]], radii=[2.0])
    ...              .build())
"""

from collections import Counter
from tqdm import tqdm, tqdm_notebook # type: ignore
from .polyhedral_site_collection import PolyhedralSiteCollection
from .polyhedral_site import PolyhedralSite
from .voronoi_site import VoronoiSite
from .voronoi_site_collection import VoronoiSiteCollection
from .spherical_site import SphericalSite
from .spherical_site_collection import SphericalSiteCollection
from .dynamic_voronoi_site import DynamicVoronoiSite
from .dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection
from .site_collection import SiteCollection
from .site import Site
from .atom import Atom
from typing import Union, Optional, Type, Sequence
from pymatgen.core import Structure

class Trajectory(object):
    """Class for performing sites analysis on simulation trajectories."""

    def __init__(self,
            sites: Sequence[Site],
            atoms: list[Atom]) -> None:
        """Initialize a Trajectory object for site analysis of simulation trajectories.
        
        This constructor ensures all sites are of the same type and initializes the
        appropriate site collection based on the type of sites provided.
        
        Args:
            sites: list of Site objects (must all be of the same type).
            atoms: list of Atom objects to track during the trajectory analysis.
            
        Raises:
            ValueError: If sites or atoms list is empty.
            TypeError: If sites contains mixed site types or an unrecognised site type.
        """
        # Validate sites is not empty
        if not sites:
            raise ValueError("Cannot initialize Trajectory with empty sites list")
        
        # Validate atoms is not empty
        if not atoms:
            raise ValueError("Cannot initialize Trajectory with empty atoms list")
        
        # ensure that all sites are of the same type
        if len(set([type(s) for s in sites])) > 1:
            raise TypeError("A Trajectory cannot be initialised with mixed Site types")
        
        # Map site types to their corresponding collection classes
        site_collection_map: dict[Type[Site], Type[SiteCollection]] = {
            PolyhedralSite: PolyhedralSiteCollection,
            VoronoiSite: VoronoiSiteCollection,
            SphericalSite: SphericalSiteCollection,
            DynamicVoronoiSite: DynamicVoronoiSiteCollection
        }
        
        site_type = type(sites[0])
        # Find the appropriate site collection class
        for site_type, collection_class in site_collection_map.items():
            if isinstance(sites[0], site_type):
                self.site_collection = collection_class(sites)
                break
        else:  # This executes if no break occurs in the for loop
            raise TypeError(f"Site type {type(sites[0])} not recognised for Trajectory initialisation")
        
        self.sites = sites
        self.atoms = atoms
        self.timesteps: list[int] = []
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        self.site_lookup = {s.index: i for i, s in enumerate(sites)}

    def atom_by_index(self,
            i: int) -> Atom:
        """Return the atom with the specified index.
        
        Args:
            i: Index of the atom to return.
            
        Returns:
            The Atom object with the specified index.
        """
        return self.atoms[self.atom_lookup[i]] 

    def site_by_index(self,
            i: int) -> Site:
        """Return the site with the specified index.
        
        Args:
            i: Index of the site to return.
            
        Returns:
            The Site object with the specified index.
        """
        return self.sites[self.site_lookup[i]] 

    def analyse_structure(self,
            structure: Structure) -> None:
        """Analyse a structure to assign atoms to sites.
        
        This delegates the analysis to the site collection's analyse_structure method.
        
        Args:
            structure: A pymatgen Structure object to be analysed.
        """
        self.site_collection.analyse_structure(self.atoms, structure)
        
    def assign_site_occupations(self,
            structure: Structure) -> None:
        """Assign atoms to sites for a specific structure.
        
        This delegates the assignment to the site collection's assign_site_occupations method.
        
        Args:
            structure: A pymatgen Structure object to be analysed.
        """
        self.site_collection.assign_site_occupations(self.atoms, structure)
                    
    def site_coordination_numbers(self) -> Counter:
        """Return the coordination numbers of all sites.
        
        Returns:
            A Counter object mapping coordination numbers to their frequencies.
        """
        return Counter([s.coordination_number for s in self.sites])

    def site_labels(self) -> list[Optional[str]]:
        """Return the labels of all sites.
        
        Returns:
            A list of site labels (or None for sites without labels).
        """
        return [s.label for s in self.sites]
   
    @property
    def atom_sites(self) -> list[Optional[int]]:
        """Return the sites that each atom currently occupies.
        
        Returns:
            A list of site indices (or None for unoccupied atoms), one for each atom.
        """
        return [atom.in_site for atom in self.atoms]
        
    @property
    def site_occupations(self) -> list[list[int]]:
        """Return the atoms occupying each site.
        
        Returns:
            A list of lists, where each inner list contains the indices of atoms
            occupying a site.
        """
        return [s.contains_atoms for s in self.sites]

    def append_timestep(self,
        structure: Structure,
        t: Optional[int]=None) -> None:
        """Append a new timestep to the trajectory.
        
        This method:
        1. Analyses the structure to assign atoms to sites
        2. Updates the trajectory information for atoms and sites
        3. Adds the timestep to the list of timesteps if provided
        
        Args:
            structure: A pymatgen Structure object for this timestep.
            t: Optional timestep index to record. If None, no timestep is recorded.
        """
        self.analyse_structure(structure)
        for atom in self.atoms:
            # assert(isinstance(atom.in_site, int))
            atom.trajectory.append(atom.in_site)
        for site in self.sites:
            site.trajectory.append(site.contains_atoms)
        if t:
            self.timesteps.append(t)

    def reset(self) -> None:
        """Reset the trajectory.
        
        This clears all trajectory information for atoms and sites and
        resets the timesteps list.
        """
        for atom in self.atoms:
            atom.reset()
        for site in self.sites:
            site.reset()
        self.timesteps = [] 

    @property
    def atoms_trajectory(self):
        """Return the trajectory of all atoms.
        
        Returns:
            A list of lists, where each inner list represents a timestep and
            contains the site indices occupied by each atom at that timestep.
        """
        return list(map(list, zip(*[atom.trajectory for atom in self.atoms])))

    @property
    def sites_trajectory(self):
        """Return the trajectory of all sites.
        
        Returns:
            A list of lists, where each inner list represents a timestep and
            contains the atom indices occupying each site at that timestep.
        """
        return list(map(list, zip(*[site.trajectory for site in self.sites])))

    @property
    def at(self):
        """Shorthand for atoms_trajectory.
        
        Returns:
            The atoms_trajectory property.
        """
        return self.atoms_trajectory

    @property
    def st(self):
        """Shorthand for sites_trajectory.
        
        Returns:
            The sites_trajectory property.
        """
        return self.sites_trajectory

    def trajectory_from_structures(self, structures, progress=False):
        """Generate a trajectory from a list of structures.
        
        This method processes each structure in sequence, appending a timestep
        for each one.
        
        Args:
            structures: list of pymatgen Structure objects to analyse.
            progress: If False, no progress is shown. If True, a progress bar is displayed.
                If 'notebook', a notebook-friendly progress bar is displayed.
                
        Notes:
            This method uses tqdm for progress tracking when enabled.
        """
        generator = enumerate(structures, 1)
        if progress:
            if progress=='notebook':
                generator = tqdm_notebook(generator, total=len(structures), unit=' steps')
            else:
                generator = tqdm(generator, total=len(structures), unit=' steps')
        for timestep, s in generator:
            self.append_timestep(s, t=timestep)
   
    def __len__(self):
        """Return the length of the trajectory.
        
        Returns:
            The number of timesteps in the trajectory.
        """
        return len(self.timesteps)
 
def update_occupation(site, atom):
    """Update the occupation record for a site and atom pair.
    
    This utility function updates the occupation records when an atom
    is assigned to a site. It:
    
    1. Adds the atom's index to the site's list of contained atoms
    2. Sets the atom's in_site attribute to the site's index
    
    Args:
        site (Site): The site that contains the atom
        atom (Atom): The atom to be assigned to the site
        
    Returns:
        None
    
    Note:
        This is a simplified version of the update_occupation method in
        SiteCollection classes, used for direct assignments without
        tracking transitions.
    """
    site.contains_atoms.append(atom.index)
    atom.in_site = site.index

