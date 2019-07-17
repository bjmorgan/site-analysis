import numpy as np

def get_vertex_indices( structure, centre_species, vertex_species, cutoff=4.5, n_vertices=6 ):
    """
    TODO: modify this to accept a list of cutoff distances and vertex numbers to allow more
    flexible site construction.
    """
    central_sites = [ s for s in structure if s.species_string == centre_species ]
    if isinstance(n_vertices, int):
        n_vertices = [n_vertices] * len(central_sites)
    vertex_species_indices = [ i for i, s in enumerate( structure ) if s.species_string == vertex_species ]
    vertex_indices = []
    for site, n_vert in zip(central_sites, n_vertices):
        neighbours = [ s for s in structure.get_neighbors(site, r=cutoff, include_index=True) 
                       if s[0].species_string == vertex_species ]
        neighbours.sort(key=lambda x: x[1])
        atom_indices = [ n[2] for n in neighbours[:n_vert] ]
        vertex_indices.append( [ i for i, n in enumerate(vertex_species_indices, 1) if n in atom_indices ] )
    return np.array(vertex_indices)
