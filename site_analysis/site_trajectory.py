class SiteTrajectory(object):
    
    def __init__(self):
        self.data = []
        self.timesteps = []
        
    def append_timestep(self, site_occupations, t=None):
        self.data.append(site_occupations)
        self.timesteps.append(t)
