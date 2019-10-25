import itertools
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import linprog

class Site(object):

    newid = itertools.count(1)

    def __init__(self, label=None):
        self.index = next(Site.newid)
        self.label = label
        self.contains_atoms = []
        self.trajectory = []
        self.points = []

    def reset(self):
        self.contains_atoms = []
        self.trajectory = []
 
    def contains_point(self, x):
        raise NotImplementedError('contains_point should be implemented '
                                  'in the inherited class')

    def contains_atom(self, atom):
        return self.contains_point(atom.frac_coords)

    def as_dict(self):
        d = {'index': self.index,
             'contains_atoms': self.contains_atoms,
             'trajectory': self.trajectory}
        if self.label:
            d['label'] = self.label
        return d

    @classmethod
    def from_dict(cls, d):
        site = cls()
        site.trajectory = d['trajectory']
        site.contains_atoms = d['contains_atoms']
        site.label = d.get('label')
        return site 

    def centre(self):
        raise NotImplementedError('centre should be implemeneted '
                                  'in the inherited class')

    @classmethod
    def reset_index(cls):
        cls.newid = itertools.count(1)
