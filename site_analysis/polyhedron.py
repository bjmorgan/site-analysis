import itertools
import numpy as np
from scipy.spatial import Delaunay

class Polyhedron(object):
    
    newid = itertools.count(1)

    def __init__(self, vertex_species, vertex_indices):
        self.index = next(Polyhedron.newid)
        self.vertex_species = vertex_species
        self.vertex_indices = vertex_indices
        self.vertex_coords = None
        self._hull = None
        self.contains_atoms = []
        
    @property
    def hull(self):
        if not self._hull:
            self._hull = Delaunay(self.vertex_coords)         
        return self._hull
    
    @property
    def coordination_number(self):
        return len(self.vertex_indices)
    
    @property
    def cn(self):
        return self.coordination_number
        
    def get_vertex_coords(self, structure):
        vertex_species_sites = [ s for s in structure 
                                if s.species_string is self.vertex_species ]
        frac_coords = np.array([ s.frac_coords for i, s in
                                enumerate(vertex_species_sites, 1) 
                                if i in self.vertex_indices ])
        for i in range(3):
            spread = max(frac_coords[:,i]) - min(frac_coords[:,i])
            if spread > 0.5:
                for j, fc in enumerate(frac_coords):
                    if fc[i] < 0.5:
                        frac_coords[j,i] += 1.0
        self.vertex_coords = frac_coords
        
    def contains_point(self, x):
        if self.vertex_coords is None:
            raise RuntimeError('no vertex coordinates set for polyhedron {}'.format(self.index))
        x_pbc = np.array([[0,0,0],
                          [1,0,0],
                          [0,1,0],
                          [0,0,1],
                          [1,1,0],
                          [1,0,1],
                          [0,1,1],
                          [1,1,1]]) + x
        return np.any( self.hull.find_simplex(x_pbc) >= 0 )
    
    def contains_atom(self, atom):
        return self.contains_point(atom.frac_coords)

    def to_dict(self):
        d = {'index': self.index,
             'vertex_species': self.vertex_species,
             'vertex_indices': self.vertex_indices,
             'vertex_coords': self.vertex_coords,
             'contains_atoms': self.contains_atoms}
        return d

    @classmethod
    def from_dict(cls, d):
        polyhedron = cls( vertex_species=d['vertex_species'],
                          vertex_indices=d['vertex_indices'] )
        polyhedron.vertex_coords = d['vertex_coords']
        polyhedron.contains_atoms = d['contains_atoms']
        return polyhedron 
