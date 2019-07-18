import numpy as np

class AtomsTrajectory(object):
    
    def __init__(self, atoms):
        self.atoms = atoms
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        
    def by_atom_indices(self, indices):
        return AtomsTrajectory( atoms=[a for a in self.atoms if a.index in indices] )
   
    @property
    def data(self):
        return list(map(list, zip(*[atom.trajectory for atom in self.atoms])))

    def as_dict(self):
        d = {'atoms': self.atoms}
        return d

    @classmethod
    def from_dict(cls, d):
        at = cls(atoms=d['atoms'])
        return at
        
