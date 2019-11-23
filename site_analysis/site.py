import itertools
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.optimize import linprog

class Site(object):
    """Parent class for defining sites.

    A Site is a bounded volume that can contain none, one, or more atoms.
    This class defines the attributes and methods expected for specific
    Site subclasses.

    Attributes:
        index (int): Numerical ID, intended to be unique to each site.
        label (`str`: optional): Optional string given as a label for this site.
            Default is `None`.
        contains_atoms (list): List of the atoms contained by this site in the
            structure last processed.
        trajectory (list): TODO: List of sites this atom has visited at each timestep?
        points (list): List of fractional coordinates for atoms assigned as
            occupying this site.
  
    """

    _newid = 1
    # Site._newid provides a counter that is incremented each time a 
    # new site is initialised. This allows each site to have a 
    # unique numerical index.
    # Site._newid can be reset to 1 by calling Site.reset_index()
    # with the default arguments.
    
    def __init__(self, label=None):
        """Initialise a Site object.

        Args:
            label (`str`: optional): Optional string used to label this site.

        Retuns:
            None

        """
        self.index = Site._newid
        Site._newid += 1
        self.label = label
        self.contains_atoms = []
        self.trajectory = []
        self.points = []

    def reset(self):
        """Reset the trajectory for this site.

        Returns the contains_atoms and trajectory attributes
        to empty lists.

        Args:
            None

        Returns:
            None

        """
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
    def reset_index(cls, newid=1):
        Site._newid = newid
