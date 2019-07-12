class AtomTrajectory(object):
    
    def __init__(self):
        self.data = []
        self.timesteps = []
        
    def append_timestep(self, atom_sites, t=None):
        self.data.append(atom_sites)
        self.timesteps.append(t)
