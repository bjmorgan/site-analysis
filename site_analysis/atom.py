import itertools

class Atom(object):
    """Represents a single persistent atom during a simulation

    Attributes:
        species_strings (str): """
    
    newid = itertools.count(1)
    
    def __init__(self, species_string):
        """Initialise an Atom object

        Args:
            species_string (str): String for this atom species, e.g. 'Li'.

        Returns:
            None
        """
        self.species_string = species_string
        self.index = next(Atom.newid)
        self.in_polyhedron = None
        self._frac_coords = None
        
    def get_coords(self, structure):
        atom_species_sites = [ s for s in structure 
                                if s.species_string is self.species_string ]
        self._frac_coords = atom_species_sites[self.index-1].frac_coords
        
    @property
    def frac_coords(self):
        if self._frac_coords is None:
            raise AttributeError('Coordinates not set for atom {}'.format(self.index))
        else:
            return self._frac_coords
