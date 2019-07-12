class SitesTrajectory(object):
    
    def __init__(self, sites):
        self.data = []
        self.timesteps = []
        self.site_lookup = {s.index: i for i, s in enumerate(sites)}
        
    def append_timestep(self, site_occupations, t=None):
        self.data.append(site_occupations)
        self.timesteps.append(t)

    def by_site_index(self, i):
        return [ d[self.site_lookup[i]] for d in self.data ]

    def reset(self):
        self.data = []
        self.timesteps = []

