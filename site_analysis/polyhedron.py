import itertools
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import linprog

class Polyhedron(object):
    
    newid = itertools.count(1)

    def __init__(self, vertex_species, vertex_indices, label=None):
        self.index = next(Polyhedron.newid)
        self.vertex_species = vertex_species
        self.vertex_indices = vertex_indices
        self.label = label
        self.vertex_coords = None
        self._delaunay = None
        self.contains_atoms = []
        self.trajectory = []

    def reset(self):
        self.vertex_coords = None
        self._delaunay = None
        self.contains_atoms = []
        self.trajectory = []
 
    @property
    def delaunay(self):
        if not self._delaunay:
            self._delaunay = Delaunay(self.vertex_coords)         
        return self._delaunay

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
        self._delaunay = None
 
    def contains_point(self, x):
        if self.vertex_coords is None:
            raise RuntimeError('no vertex coordinates set for polyhedron {}'.format(self.index))
        return np.any( self.delaunay.find_simplex(x_pbc(x)) >= 0 )
    
    def contains_point_new(self, x):
        if self.vertex_coords is None:
            raise RuntimeError('no vertex coordinates set for polyhedron {}'.format(self.index))
        for p in x_pbc(x):
            if np.any( self.contains_point_alt(p)):
                return True
        return False
 
    def contains_point_alt(self, x):
        """Alternative algorithm for calculating whether a point sits
        inside a convex hull.

        This algorithm is a potential target for optimisation at some
        future time.
   
        """
        hull = ConvexHull(self.vertex_coords)
        faces = hull.points[hull.simplices]
        centre = self.centre()
        dotsum = 0
        for f in faces:
            surface_normal = np.cross(f[0]-f[2], f[1]-f[2])
            c_sign = np.sign(np.dot( surface_normal, centre-f[0]))
            p_sign = np.sign(np.dot( surface_normal, x-f[0]))
            dotsum += c_sign * p_sign
        return dotsum == len(faces)

    def contains_atom(self, atom):
        return self.contains_point(atom.frac_coords)

    def contains_atom_accurate(self, atom):
        return self.contains_point_accurate(atom.frac_coords)

    def contains_atom_new(self, atom):
        return self.contains_point_new(atom.frac_coords)

    def as_dict(self):
        d = {'index': self.index,
             'vertex_species': self.vertex_species,
             'vertex_indices': self.vertex_indices,
             'vertex_coords': self.vertex_coords,
             'contains_atoms': self.contains_atoms}
        if self.label:
            d['label'] = self.label
        return d

    @classmethod
    def from_dict(cls, d):
        polyhedron = cls( vertex_species=d['vertex_species'],
                          vertex_indices=d['vertex_indices'] )
        polyhedron.vertex_coords = d['vertex_coords']
        polyhedron.contains_atoms = d['contains_atoms']
        polyhedron.label = d.get('label')
        return polyhedron 

    def centre(self):
        return np.mean(self.vertex_coords, axis=0)

def x_pbc(x):
    all_x =  np.array([[0,0,0],
                       [1,0,0],
                       [0,1,0],
                       [0,0,1],
                       [1,1,0],
                       [1,0,1],
                       [0,1,1],
                       [1,1,1]]) + x
    return all_x

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T,np.ones((1,n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

