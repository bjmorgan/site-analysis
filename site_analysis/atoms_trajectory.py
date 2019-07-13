import numpy as np

class AtomsTrajectory(object):
    
    def __init__(self, atoms=None, atom_lookup=None):
        self.data = []
        self.timesteps = []
        if atoms:
            self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        elif atom_lookup:
            self.atom_lookup = atom_lookup
        else:
            raise ArgumentError('atoms or atom_lookup arguments are needed to initialise an AtomsTrajectory')
        
    def append_timestep(self, atom_sites, t=None):
        self.data.append(atom_sites)
        self.timesteps.append(t)

    def by_atom_index(self, i):
        return np.array(self.data)[:,self.atom_lookup[i]]

    def reset(self):
        self.data = []
        self.timesteps = []

    def to_dict(self):
        d = {'data': self.data,
             'timesteps': self.timesteps,
             'atom_lookup': self.atom_lookup}
        return d

    @classmethod
    def from_dict(cls, d):
        at = cls( atom_lookup=d['atom_lookup'] )
        at.data = d['data']
        at.timesteps = d['timesteps']
        return at
        
