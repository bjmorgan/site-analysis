from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Optional, List, Dict
from .atom import Atom
import numpy as np
from pymatgen.core import Structure

class Site(ABC):
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
        trajectory (list(list(int))): Nested list of atoms that have visited this
            site at each timestep.
        points (list): List of fractional coordinates for atoms assigned as
            occupying this site.
        transitions (collections.Counter): Stores observed transitions from this
            site to other sites. Format is {index: count} with ``index`` giving
            the index of each destination site, and ``count`` giving the number 
            of observed transitions to this site.
 
    """

    _newid = 0
    # Site._newid provides a counter that is incremented each time a 
    # new site is initialised. This allows each site to have a 
    # unique numerical index.
    # Site._newid can be reset to 0 by calling Site.reset_index()
    # with the default arguments.
    
    def __init__(self,
            label: Optional[str]=None) -> None:
        """Initialise a Site object.

        Args:
            label (`str`: optional): Optional string used to label this site.

        Retuns:
            None

        """
        self.index = Site._newid
        Site._newid += 1
        self.label = label
        self.contains_atoms: List[int] = []
        self.trajectory: List[List[int]] = []
        self.points: List[np.ndarray] = []
        self.transitions: Counter = Counter()

    def reset(self) -> None:
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
        self.transitions = Counter()
 
    @abstractmethod
    def contains_point(self,
            x: np.ndarray,
            *args: Any,
            **kwargs: Any) -> bool:
        """Test whether the fractional coordinate x is contained by this site.

        This method should be implemented in the derived subclass

        Args:
            x (np.array): Fractional coordinate.

        Returns:
            (bool)

        Note:
            Specific Site subclasses may require additional arguments to be passed.

        """
        raise NotImplementedError('contains_point should be implemented '
                                  'in the derived class')

    def contains_atom(self,
            atom: Atom,
            *args: Any,
            **kwargs: Any) -> bool:
        """Test whether this site contains a specific atom.

        Args:
            atom (Atom): The atom to test.

        Returns:
            (bool)

        """
        return self.contains_point(atom.frac_coords)

    def as_dict(self) -> Dict:
        """Json-serializable dict representation of this Site.

        Args:
            None

        Returns:
            (dict)

        """
        d = {'index': self.index,
             'contains_atoms': self.contains_atoms,
             'trajectory': self.trajectory,
             'points': self.points,
             'transitions': self.transitions}
        if self.label:
            d['label'] = self.label
        return d

    @classmethod
    def from_dict(cls,
            d: Dict) -> Site:
        """Create a Site object from a dict representation.

        Args:
            d (dict): The dict representation of this Site.

        Returns:
            (Site)

        """
        site = cls()
        site.set_attributes_from_dict(d)
        return site 
        
    def set_attributes_from_dict(self,
            d: dict) -> None:
            """TODO
            
            """
            self.index = d['index']
            self.trajectory = d['trajectory']
            self.contains_atoms = d['contains_atoms']
            self.points = d['points']
            self.transitions = d['transitions']
            self.label = d.get('label')
        
    @abstractmethod
    def centre(self) -> np.ndarray:
        """Returns the centre point of this site.

        This method should be implemented in the derived subclass.

        Args:
            None

        Returns:
            None

        """ 
        raise NotImplementedError('centre should be implemented '
                                  'in the derived class')

    @classmethod
    def reset_index(cls,
            newid: int=0) -> None:
        """Reset the site index counter.

        Args:
            newid (`int`: optional): New starting index. Default is 1.

        Returns:
            None

        """ 
        Site._newid = newid

    @property
    def coordination_number(self) -> int:
        """Returns the coordination number of this site.

        This method should be implemented in the derived subclass.

        Args:
            None

        Returns:
            int

        """
        raise NotImplementedError('coordination_number should be implemented '
                                  'in the derived class')

