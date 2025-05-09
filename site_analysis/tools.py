"""Utility functions for site analysis and crystal structure manipulation.

This module provides a collection of helper functions for working with crystal
structures, finding coordination environments, mapping between structures, and
handling periodic boundary conditions.

Key functions include:

Coordination and site analysis:
- get_coordination_indices: Find atoms with specific coordination environments
- get_nearest_neighbour_indices: Get indices of nearest neighbors for each site
- get_vertex_indices: (Deprecated) Find vertex atoms for coordination polyhedra

Structure mapping and comparison:
- site_index_mapping: Map site indices between two structures
- calculate_species_distances: Calculate distances between matching atoms in structures

Periodic boundary handling:
- x_pbc: Generate fractional coordinates for all periodic images in neighboring cells

These utilities provide low-level functionality that can be used directly or are
used internally by the higher-level site and trajectory analysis classes.
"""

import warnings
import numpy as np

from typing import Optional, Union, cast
from pymatgen.core import Structure, Site, PeriodicSite

def get_coordination_indices(
        structure: Structure,
        centre_species: str, 
        coordination_species: Union[str, list[str]],
        cutoff: float,
        n_coord: Union[int, list[int]]) -> list[list[int]]:
    """
    Find atoms with exactly the specified coordination environment.
    
    For each atom of centre_species, finds environments with exactly n_coord
    coordinating atoms of coordination_species within the cutoff distance.
    
    Args:
        structure: A pymatgen Structure object.
        centre_species: Species string identifying the atoms at the centres.
        coordination_species: Species string or list of strings identifying 
            the coordinating atoms.
        cutoff: Distance cutoff for neighbour search in Angstroms.
        n_coord: Number(s) of coordinating atoms required for 
            each environment. If an int is provided, the same number is used for all
            centre atoms. If a list is provided, it should have the same length as 
            the number of centre atoms found.
            
    Returns:
        list(list(int)): Nested list of integers, giving the atom indices for each
            complete coordination environment. Only includes environments with 
            exactly n_coord coordinating atoms within cutoff.
            
    Raises:
        ValueError: If no centre atoms are found, or if a list of n_coord
            has incorrect length.
    """
    # Standardize coordination_species to list
    if isinstance(coordination_species, str):
        coordination_species = [coordination_species]
        
    # Find all centre atoms
    centre_atoms = [i for i, site in enumerate(structure) 
                    if site.species_string == centre_species]
    
    if not centre_atoms:
        raise ValueError(f"No atoms of species '{centre_species}' found in structure")
    
    # Standardize n_coord to list
    if isinstance(n_coord, int):
        required_coord = [n_coord] * len(centre_atoms)
    else:
        if len(n_coord) != len(centre_atoms):
            raise ValueError(f"Length of n_coord list ({len(n_coord)}) does not match "
                             f"number of {centre_species} atoms ({len(centre_atoms)})")
        required_coord = n_coord
    
    # Find coordinating environments
    complete_environments = []
    
    for i, (centre_idx, required) in enumerate(zip(centre_atoms, required_coord)):
        centre_site = structure[centre_idx]
        
        # Get all neighbors within cutoff that match coordination_species
        neighbors = []
        # structure.get_neighbors returns a list of PeriodicNeighbor objects
        # PeriodicNeighbor is a PeriodicSite subclass with additional properties
        # for neighbor info (nn_distance, index, image)
        for neighbor in structure.get_neighbors(cast(PeriodicSite, centre_site), cutoff):
            if neighbor.species_string in coordination_species:
                neighbors.append((int(neighbor.index), int(neighbor.nn_distance)))
        
        # Only include environments with exactly the required number of coordinating atoms
        if len(neighbors) == required:
            # Sort by distance
            neighbors.sort(key=lambda x: x[1])
            neighbor_indices = [idx for idx, _ in neighbors]
            complete_environments.append(neighbor_indices)
    
    return complete_environments

def get_nearest_neighbour_indices(
        structure: Structure,
        ref_structure: Structure,
        vertex_species: list[str],
        n_coord: int) -> list[list[int]]:
    """
    Returns the atom indices for the N nearest neighbours to each site in a reference
    structure.
    
    Args:
        structure (`pymatgen.Structure`): A pymatgen Structure object, used to select
            the nearest neighbour indices.
        ref_structure (`pymatgen.Structure`): A pymatgen Structure object. Each site
            is used to find the set of N nearest neighbours (of the specified atomic species)
            in ``structure``.
        vertex_species (list(str)): list of strings specifying the atomic species of
            the vertex atoms, e.g. ``[ 'S', 'I' ]``.
        n_coord (int): Number of matching nearest neighbours to return for each site in 
            ``ref_structure``.
    
    Returns:
        (list(list(int)): N_sites x N_neighbours nested list of vertex atom indices.
        
    Raises:
        ValueError: If structure or ref_structure is empty, if vertex_species is empty,
            if n_coord is not positive, if no atoms match vertex_species, or if there
            are fewer matching atoms than n_coord.
    """
    if len(structure) == 0:
        raise ValueError("Empty structure provided")
        
    if len(ref_structure) == 0:
        raise ValueError("Empty reference structure provided")
        
    if not vertex_species:
        raise ValueError("No vertex species specified")
        
    if n_coord <= 0:
        raise ValueError(f"n_coord must be positive, got {n_coord}")
    
    vertex_indices = [i for i, s in enumerate(structure)
            if s.species_string in vertex_species]
    if not vertex_indices:
        raise ValueError(f"No atoms of species {vertex_species} found in structure")
        
    if len(vertex_indices) < n_coord:
        raise ValueError(f"Requested {n_coord} neighbors but only {len(vertex_indices)} matching atoms found")
    
    struc1_coords = np.array([structure[i].frac_coords for i in vertex_indices])
    struc2_coords = ref_structure.frac_coords
    lattice = structure[0].lattice
    dr_ij = lattice.get_all_distances(struc1_coords, struc2_coords).T
    nn_indices = []
    for dr_i in dr_ij:
        idx = np.argpartition(dr_i, n_coord)
        nn_indices.append( sorted([ vertex_indices[i] for i in idx[:n_coord] ]) )
    return nn_indices

def get_vertex_indices(
        structure: Structure,
        centre_species: str, 
        vertex_species: Union[str, list[str]],
        cutoff: float,
        n_vertices: Union[int, list[int]]) -> list[list[int]]:
    """
    DEPRECATED: Find the atom indices for atoms defining the vertices of coordination polyhedra.
    
    This function is deprecated and will be removed in a future version.
    
    Please use one of the following alternatives:
    - get_coordinating_indices(): For finding atoms with exact coordination environments
    - ReferenceBasedSites workflow: For generating sites based on reference structures
    
    Given the elemental species of a set of central atoms, A, 
    and of the polyhedral vertices, B, this function finds:
    for each A, then N closest neighbours B (within some cutoff).
    The number of neighbours found per central atom can be a single
    value for all A, or can be provided as a list of values for each A.
    
    Args:
        structure: A pymatgen Structure object.
        centre_species: Species string identifying the atoms at the centres.
        vertex_species: Species string or list of strings identifying the vertex atoms.
        cutoff: Distance cutoff for neighbour search.
        n_vertices: Number(s) of nearest neighbours to return
            for each set of vertices. If an int is passed, this should be the same
            length as the number of atoms of centre species A.
            
    Returns:
        list(list(int)): Nested list of integers, giving the atom indices for each
            coordination environment.
    """
    warnings.warn(
        "get_vertex_indices is deprecated and will be removed in a future version. "
        "Please use get_coordinating_indices() for finding atoms with exact coordination "
        "environments, or use the ReferenceBasedSites workflow for generating sites based "
        "on reference structures.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Original implementation remains unchanged
    # Standardize vertex_species to list
    if isinstance(vertex_species, str):
        vertex_species = [vertex_species]
        
    # Find all centre atoms
    central_sites = [s for s in structure if s.species_string == centre_species]
    
    if isinstance(n_vertices, int):
        n_vertices = [n_vertices] * len(central_sites)
    
    if len(n_vertices) != len(central_sites):
        raise ValueError(f"Length of n_vertices list ({len(n_vertices)}) does not match "
                         f"number of {centre_species} atoms ({len(central_sites)})")
    
    vertex_indices = []
    for site, n_vert in zip(central_sites, n_vertices):
        # Get all neighbors within cutoff that match vertex_species
        neighbors = []
        for neighbor in structure.get_neighbors(cast(PeriodicSite, site), cutoff):
            if neighbor.species_string in vertex_species:
                neighbors.append((neighbor.index, neighbor.nn_distance))
        
        # Sort by distance
        neighbors.sort(key=lambda x: x[1])
        
        # Get the n_vert closest neighbors
        neighbor_indices = [idx for idx, _ in neighbors[:n_vert]]
        vertex_indices.append(neighbor_indices)
    
    return vertex_indices

def x_pbc(x: np.ndarray):
    """Return an array of fractional coordinates mapped into all positive neighbouring 
    periodic cells.

    Args:
        x (np.array): Input fractional coordinates.

    Returns:
        np.array: (9,3) numpy array of all mapped fractional coordinates, including the
                  original coordinates in the origin calculation cell.

    Example:
        >>> x = np.array([0.1, 0.2, 0.3])
        >>> x_pbc(x)
        array([[0.1, 0.2, 0.3],
               [1.1, 0.2, 0.3],
               [0.1, 1.2, 0.3],
               [0.1, 0.2, 1.3],
               [1.1, 1.2, 0.3],
               [1.1, 0.2, 1.3],
               [0.1, 1.2, 1.3],
               [1.1, 1.2, 1.3]])

    """       
    all_x =  np.array([[0,0,0],
                       [1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [1,1,0],
                       [1,0,1],
                       [0,1,1],
                       [1,1,1]]) + x
    return all_x

def species_string_from_site(site: Site) -> str:
    """Extract the species string from a pymatgen Site object.
    
    Args:
        site: A pymatgen Site object
        
    Returns:
        String representation of the site's species
    """
    if hasattr(site._species, 'keys'):
        species_keys = [k.__str__() for k in site._species.keys()]
        if species_keys:
            return str(species_keys[0])
        return "" 
    elif hasattr(site, 'species_string'):
        return site.species_string
    else:
        return str(site._species)

def site_index_mapping(structure1: Structure, 
                       structure2: Structure,
                       species1: Optional[Union[str, list[str]]] = None,
                       species2: Optional[Union[str, list[str]]] = None,
                       one_to_one_mapping: Optional[bool] = True,
                       return_mapping_distances: Optional[bool] = False) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute the site index mapping between two structures based on the closest corresponding site in
    structure2 to each selected site in structure1.
    
    Args:
        structure1 (pymatgen.Structure): The structure to map from.
        structure2 (pymatgen.Structure): The structure to map to.
        species1 (optional, str or list(str)): Optional argument to select a subset of atomic species
            to map site indices from.
        species2 (optional, str of list(str)): Optional argument to specify a subset of atomic species
            to map site indices to.
        one_to_one_mapping (optional, bool): Optional argument to check that a one-to-one mapping is found
            between the relevant subsets of sites in structure1 and structure2. Default is `True`.
            
    Returns:
        np.ndarray
        
    Raises:
        ValueError: if `one_to_one_mapping = True` and a one-to-one mapping is not found.
 
    """
    # Ensure species1 and species2 are lists of site species strings.
    if species1 is None:
        species1 = list(set([site.species_string for site in structure1]))
    if isinstance(species1, str):
        species1 = [species1]
    if isinstance(species2, str):
        species2 = [species2]
    if species2 is None:
        species2 = list(set([site.species_string for site in structure2]))
    assert(isinstance(species1, list))
    assert(isinstance(species2, list))
    
    structure2_mask = np.array([site.species_string in species2 for site in structure2])
    lattice = structure1.lattice
    dr_ij = np.array(lattice.get_all_distances(structure1.frac_coords, structure2.frac_coords))
    to_return = []
    dr_ij_to_return = []
    for site1, dr_i in zip(structure1, dr_ij):
        if site1.species_string in species1:
                dr_i_array = np.asarray(dr_i)
                subset_idx = np.argmin(dr_i_array[structure2_mask])
                parent_idx = np.arange(dr_i_array.size)[structure2_mask][subset_idx] 
                to_return.append(parent_idx)
                dr_ij_to_return.append(dr_i_array[parent_idx])
    if one_to_one_mapping:
        if len(to_return) != len(set(to_return)):
            raise ValueError("One-to-one mapping between structures not found.")   
    if return_mapping_distances:
        return np.array(to_return), np.array(dr_ij_to_return)     
    else:
        return np.array(to_return)
        
def calculate_species_distances(structure1, structure2, species=None):
    """Calculate minimum distances between atoms of the same species in two structures.
    
    Args:
        structure1: First structure to compare
        structure2: Second structure to compare
        species: list of species to include. If None, includes all species
                 present in both structures.
                 
    Returns:
        dict: Dictionary mapping species to lists of minimum distances for each atom
        list: Flattened list of all minimum distances
    """
    # Determine which species to include
    if species is None:
        species = set([site.species_string for site in structure1])
        species = species.intersection([site.species_string for site in structure2])
        species = list(species)
    
    # Calculate minimum distances for each atom by species
    species_distances = {}
    all_distances = []
    
    for sp in species:
        indices1 = list(structure1.indices_from_symbol(sp))
        indices2 = list(structure2.indices_from_symbol(sp))
        
        if not indices1 or not indices2:
            continue
        
        # Get coordinates for this species
        coords1 = structure1.frac_coords[indices1]
        coords2 = structure2.frac_coords[indices2]
        
        # Calculate distance matrix for this species
        distance_matrix = structure1.lattice.get_all_distances(coords1, coords2)
        
        # Find minimum distance for each atom in structure1
        min_distances = np.min(distance_matrix, axis=1)
        
        species_distances[sp] = min_distances.tolist()
        all_distances.extend(min_distances)
    
    return species_distances, all_distances