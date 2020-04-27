"""site_analysis.tools module

This module contains tools for 

"""
import numpy as np

def get_nearest_neighbour_indices(structure, ref_structure, vertex_species, n_coord):
    """
    Returns the atom indices for the N nearest neighbours to each site in a reference
    structure.

    Args:
        structure (`pymatgen.Structure`): A pymatgen Structure object, used to select
            the nearest neighbour indices.
        ref_structure (`pymatgen.Structure`): A pymatgen Structure object. Each site
            is used to find the set of N nearest neighbours (of the specified atomic species)
            in ``structure``.
        vertex_species (list(str)): List of strings specifying the atomic species of
            the vertex atoms, e.g. ``[ 'S', 'I' ]``.
        n_coord (int): Number of matching nearest neighbours to return for each site in 
            ``ref_structure``.

    Returns:
        (list(list(int)): N_sites x N_neighbours nested list of vertex atom indices.

    """
    vertex_indices = [ i for i, s in enumerate(structure) if s.species_string in vertex_species ]
    struc1_coords = np.array([structure[i].frac_coords for i in vertex_indices])
    struc2_coords = ref_structure.frac_coords
    lattice = structure[0].lattice
    dr_ij = lattice.get_all_distances(struc1_coords, struc2_coords).T
    nn_indices = []
    for dr_i in dr_ij:
        idx = np.argpartition(dr_i, n_coord)
        nn_indices.append( sorted([ vertex_indices[i] for i in idx[:n_coord] ]) )
    return nn_indices

def get_vertex_indices( structure, centre_species, vertex_species, cutoff=4.5, n_vertices=6 ):
    """
    Find the atom indices for atoms defining the vertices of coordination polyhedra, from 
    a pymatgen Structure object.

    Given the elemental species of a set of central atoms, A, 
    and of the polyhedral vertices, B, this function finds:
    for each A, then N closest neighbours B (within some cutoff).
    The number of neighbours found per central atom can be a single
    value for all A, or can be provided as a list of values for each A.

    Args:
        structure (`pymatgen.Structure`): A pymatgen Structure object, used to
            find the coordination polyhedra vertices..
        centre_species (str): Species string identifying the atoms at the centres
            of each coordination environment, e.g. "Na".
        vertex_species (str or list(str)): Species string identifying the atoms at the vertices
            of each coordination environment, e.g. "S"., or a list of strings, e.g. ``["S", "I"]``.
        cutoff (float): Distance cutoff for neighbour search.
        n_vertices (int or list(int)): Number(s) of nearest neighbours to return
            for each set of vertices. If a list is passed, this should be the same
            length as the number of atoms of centre species A.

    Returns:
        list(list(int)): Nested list of integers, giving the atom indices for each
            coordination environment.

    """
    central_sites = [ s for s in structure if s.species_string == centre_species ]
    if isinstance(n_vertices, int):
        n_vertices = [n_vertices] * len(central_sites)
    if isinstance(vertex_species, str):
        vertex_species = [ vertex_species ]
    vertex_indices = []
    for site, n_vert in zip(central_sites, n_vertices):
        neighbours = [ s for s in structure.get_neighbors(site, r=cutoff, include_index=True) 
                       if s[0].species_string in vertex_species ]
        neighbours.sort(key=lambda x: x[1])
        atom_indices = [ n[2] for n in neighbours[:n_vert] ]
        vertex_indices.append( atom_indices )
    return vertex_indices

def x_pbc(x):
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

def species_string_from_site(site):
    return [k.__str__() for k in site._species.keys()][0]
