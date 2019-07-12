import itertools

class Atom(object):
    
    newid = itertools.count(1)
    
    def __init__(self, species):
        self.atom_species = species
        self.index = next(Atom.newid)
        self.in_polyhedron = None
        self._frac_coords = None
        
    def get_coords(self, structure):
        atom_species_sites = [ s for s in structure 
                                if s.species_string is self.atom_species ]
        self._frac_coords = atom_species_sites[self.index-1].frac_coords
        
    @property
    def frac_coords(self):
        if self._frac_coords is None:
            raise AttributeError('Coordinates not set for atom {}'.format(self.index))
        else:
            return self._frac_coords
