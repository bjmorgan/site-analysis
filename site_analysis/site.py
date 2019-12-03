from collections import Counter

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
        trajectory (list): List of sites this atom has visited at each timestep?
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
        self.transitions = Counter()

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
        self.transitions = Counter()
 
    def contains_point(self, x):
        """Test whether the fractional coordinate x is contained by this site.

        This method should be implemented in the inherited subclass

        Args:
            x (np.array): Fractional coordinate.

        Returns:
            (bool)

        Note:
            Specific Site subclasses may require additional arguments to be passed.

        """
        raise NotImplementedError('contains_point should be implemented '
                                  'in the inherited class')

    def contains_atom(self, atom):
        """Test whether this site contains a specific atom.

        Args:
            atom (Atom): The atom to test.

        Returns:
            (bool)

        """
        return self.contains_point(atom.frac_coords)

    def as_dict(self):
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
    def from_dict(cls, d):
        """Create a Site object from a dict representation.

        Args:
            d (dict): The dict representation of this Site.

        Returns:
            (Site)

        """
        site = cls()
        site.index = d['index']
        site.trajectory = d['trajectory']
        site.contains_atoms = d['contains_atoms']
        site.points = d['points']
        site.transitions = d['transitions']
        site.label = d.get('label')
        return site 

    def centre(self):
        """Returns the centre point of this site.

        This method should be implemented in the inherited subclass.

        Args:
            None

        Returns:
            None

        """ 
        raise NotImplementedError('centre should be implemeneted '
                                  'in the inherited class')

    @classmethod
    def reset_index(cls, newid=0):
        """Reset the site index counter.

        Args:
            newid (`int`: optional): New starting index. Default is 1.

        Returns:
            None

        """ 
        Site._newid = newid
