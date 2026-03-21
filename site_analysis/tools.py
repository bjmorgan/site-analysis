"""Utility functions for site analysis.

This module provides helper functions for finding coordination
environments, mapping atoms between structures, and handling
periodic boundary conditions. Most functions accept numpy arrays
and species lists rather than pymatgen Structure objects.

Key functions include:

Coordination and site analysis:
- get_coordination_indices: Find atoms with specific coordination environments
- indices_for_species: Get atom indices matching a species string

Structure mapping and comparison:
- site_index_mapping: Map site indices between two structures
- calculate_species_distances: Calculate distances between matching atoms

Periodic boundary handling:
- x_pbc: Generate fractional coordinates for all periodic images in neighbouring cells

These utilities provide low-level functionality that can be used directly or are
used internally by the higher-level site and trajectory analysis classes.
"""

import warnings
import numpy as np

from typing import cast
from pymatgen.core import Structure, Site, PeriodicSite
from site_analysis.distances import all_mic_distances

def get_coordination_indices(
    frac_coords: np.ndarray,
    lattice_matrix: np.ndarray,
    species: list[str],
    centre_species: str,
    coordination_species: str | list[str],
    cutoff: float,
    n_coord: int | list[int],
) -> dict[int, list[int]]:
    """Find atoms with exactly the specified coordination environment.

    For each atom of centre_species, finds environments with exactly n_coord
    coordinating atoms of coordination_species within the cutoff distance.

    Args:
        frac_coords: Fractional coordinates for all atoms, shape ``(N, 3)``.
        lattice_matrix: Lattice matrix (3x3) for distance calculations.
        species: Species strings for each atom.
        centre_species: Species string identifying the atoms at the centres.
        coordination_species: Species string or list of strings identifying
            the coordinating atoms.
        cutoff: Distance cutoff for neighbour search in Angstroms.
        n_coord: Number(s) of coordinating atoms required for
            each environment. If an int is provided, the same number is used
            for all centre atoms. If a list is provided, it should have the
            same length as the number of centre atoms found.

    Returns:
        Dictionary mapping centre atom indices to lists of coordinating atom
        indices. Only includes environments with exactly n_coord coordinating
        atoms within cutoff.

    Raises:
        ValueError: If no centre atoms are found, or if a list of n_coord
            has incorrect length.
    """
    if len(species) != len(frac_coords):
        raise ValueError(
            f"species length ({len(species)}) does not match "
            f"frac_coords rows ({len(frac_coords)})"
        )
    if isinstance(coordination_species, str):
        coordination_species = [coordination_species]

    coordination_species_set = set(coordination_species)
    centre_atoms = indices_for_species(species, centre_species)

    if not centre_atoms:
        raise ValueError(f"No atoms of species '{centre_species}' found in structure")

    coord_atoms = [i for i, s in enumerate(species)
                   if s in coordination_species_set]

    if isinstance(n_coord, int):
        required_coord = [n_coord] * len(centre_atoms)
    else:
        if len(n_coord) != len(centre_atoms):
            raise ValueError(f"Length of n_coord list ({len(n_coord)}) does not match "
                            f"number of {centre_species} atoms ({len(centre_atoms)})")
        required_coord = n_coord

    # Compute distance matrix between centre and coordinating atoms
    if coord_atoms:
        centre_coords = frac_coords[centre_atoms]
        coord_coords = frac_coords[coord_atoms]
        dist_matrix = all_mic_distances(centre_coords, coord_coords, lattice_matrix)
    else:
        dist_matrix = np.empty((len(centre_atoms), 0))

    complete_environments: dict[int, list[int]] = {}
    for i, (centre_idx, required) in enumerate(zip(centre_atoms, required_coord)):
        distances = dist_matrix[i]
        within_cutoff = [(coord_atoms[j], float(distances[j]))
                         for j in range(len(coord_atoms))
                         if distances[j] <= cutoff and coord_atoms[j] != centre_idx]
        if len(within_cutoff) == required:
            within_cutoff.sort(key=lambda x: x[1])
            complete_environments[centre_idx] = [idx for idx, _ in within_cutoff]

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
        vertex_species: str | list[str],
        cutoff: float,
        n_vertices: int | list[int]) -> list[list[int]]:
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
        "Please use get_coordination_indices() for finding atoms with exact coordination "
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

_SHIFTS = np.array([[0,0,0],
                    [1,0,0],
                    [0,1,0],
                    [0,0,1],
                    [1,1,0],
                    [1,0,1],
                    [0,1,1],
                    [1,1,1]])
                    
def x_pbc(x: np.ndarray):
    """Return an array of fractional coordinates mapped into all positive neighbouring 
    periodic cells.

    Args:
        x (np.array): Input fractional coordinates.

    Returns:
        np.array: (8,3) numpy array of all mapped fractional coordinates, including the
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
    return _SHIFTS + x

def species_string_from_site(site: Site) -> str:
    """Extract the species string from a pymatgen Site object.

    Args:
        site: A pymatgen Site object

    Returns:
        String representation of the site's species
    """
    return site.species_string

def site_index_mapping(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
    species1: list[str],
    species2: list[str],
    species1_filter: str | list[str] | None = None,
    species2_filter: str | list[str] | None = None,
    one_to_one_mapping: bool = True,
    return_mapping_distances: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the site index mapping based on closest distances.

    For each selected site in ``frac_coords1`` (filtered by
    ``species1_filter``), finds the closest site in ``frac_coords2``
    (filtered by ``species2_filter``) using minimum-image convention
    distances.

    Args:
        frac_coords1: Fractional coordinates to map from, shape (N, 3).
        frac_coords2: Fractional coordinates to map to, shape (M, 3).
        lattice_matrix: Lattice matrix (3x3) for distance calculations.
        species1: Species strings for each site in ``frac_coords1``.
        species2: Species strings for each site in ``frac_coords2``.
        species1_filter: If given, only map from sites whose species
            are in this list. Defaults to all species in ``species1``.
        species2_filter: If given, only map to sites whose species
            are in this list. Defaults to all species in ``species2``.
        one_to_one_mapping: If ``True``, raise ``ValueError`` when the
            mapping is not one-to-one. Default is ``True``.
        return_mapping_distances: If ``True``, also return the distances
            for each mapped pair.

    Returns:
        Array of mapped indices into ``frac_coords2``. If
        ``return_mapping_distances`` is ``True``, returns a tuple of
        (mapping, distances).

    Raises:
        ValueError: If ``one_to_one_mapping`` is ``True`` and the
            mapping is not one-to-one.
    """
    if len(species1) != len(frac_coords1):
        raise ValueError(
            f"species1 length ({len(species1)}) does not match "
            f"frac_coords1 rows ({len(frac_coords1)})"
        )
    if len(species2) != len(frac_coords2):
        raise ValueError(
            f"species2 length ({len(species2)}) does not match "
            f"frac_coords2 rows ({len(frac_coords2)})"
        )
    if isinstance(species1_filter, str):
        species1_filter = [species1_filter]
    if isinstance(species2_filter, str):
        species2_filter = [species2_filter]
    if species1_filter is None:
        species1_filter = list(set(species1))
    if species2_filter is None:
        species2_filter = list(set(species2))
    structure2_mask = np.array([s in species2_filter for s in species2])
    dr_ij = all_mic_distances(frac_coords1, frac_coords2, lattice_matrix)
    to_return = []
    dr_ij_to_return = []
    for i, dr_i in enumerate(dr_ij):
        if species1[i] in species1_filter:
            subset_idx = np.argmin(dr_i[structure2_mask])
            parent_idx = np.arange(dr_i.size)[structure2_mask][subset_idx]
            to_return.append(parent_idx)
            dr_ij_to_return.append(dr_i[parent_idx])
    if one_to_one_mapping:
        if len(to_return) != len(set(to_return)):
            raise ValueError("One-to-one mapping between structures not found.")
    if return_mapping_distances:
        return np.array(to_return), np.array(dr_ij_to_return)
    else:
        return np.array(to_return)
        
def indices_for_species(
    all_species: list[str],
    target: str,
) -> list[int]:
    """Return indices where all_species matches target.

    Args:
        all_species: List of species strings for all atoms.
        target: Species string to match.

    Returns:
        List of indices where species matches target.
    """
    return [i for i, s in enumerate(all_species) if s == target]


def calculate_species_distances(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
    species1: list[str],
    species2: list[str],
    species: list[str] | None = None,
) -> tuple[dict[str, list[float]], list[float]]:
    """Calculate minimum distances between atoms of the same species.

    For each species, computes the distance from each atom of that
    species in frac_coords1 to the nearest atom of the same species
    in frac_coords2.

    Args:
        frac_coords1: Fractional coordinates of first structure,
            shape ``(N, 3)``.
        frac_coords2: Fractional coordinates of second structure,
            shape ``(M, 3)``.
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors.
        species1: Species strings for each atom in frac_coords1.
        species2: Species strings for each atom in frac_coords2.
        species: Optional filter - only include these species.
            If None, includes all species present in both structures.

    Returns:
        A tuple of (species_distances, all_distances) where
        species_distances maps species to lists of minimum distances,
        and all_distances is a flat list of all minimum distances.
    """
    if len(species1) != len(frac_coords1):
        raise ValueError(
            f"species1 length ({len(species1)}) does not match "
            f"frac_coords1 rows ({len(frac_coords1)})"
        )
    if len(species2) != len(frac_coords2):
        raise ValueError(
            f"species2 length ({len(species2)}) does not match "
            f"frac_coords2 rows ({len(frac_coords2)})"
        )
    if species is None:
        species = sorted(set(species1) & set(species2))

    species_distances: dict[str, list[float]] = {}
    all_distances: list[float] = []

    for sp in species:
        idx1 = indices_for_species(species1, sp)
        idx2 = indices_for_species(species2, sp)

        if not idx1 or not idx2:
            continue

        coords1 = frac_coords1[idx1]
        coords2 = frac_coords2[idx2]
        dist_matrix = all_mic_distances(coords1, coords2, lattice_matrix)
        min_dists = np.min(dist_matrix, axis=1).tolist()

        species_distances[sp] = min_dists
        all_distances.extend(min_dists)

    return species_distances, all_distances
