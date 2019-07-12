import numpy as np

class AtomsTrajectory(object):
    
    def __init__(self, atoms):
        self.data = []
        self.timesteps = []
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        
    def append_timestep(self, atom_sites, t=None):
        self.data.append(atom_sites)
        self.timesteps.append(t)

    def by_atom_index(self, i):
        return np.array(self.data)[:,self.atom_lookup[i]]

    def reset(self):
        self.data = []
        self.timesteps = []
