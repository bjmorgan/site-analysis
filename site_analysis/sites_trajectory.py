class SitesTrajectory(object):
    
    def __init__(self, sites=None, site_lookup=None):
        self.data = []
        self.timesteps = []
        if sites:
            self.site_lookup = {s.index: i for i, s in enumerate(sites)}
        elif site_lookup:
            self.site_lookup = site_lookup
        else:
            raise ArgumentError('sites or site_lookup arguments are needed to initialise an SitesTrajectory') 

    def append_timestep(self, site_occupations, t=None):
        self.data.append(site_occupations)
        self.timesteps.append(t)

    def by_site_index(self, i):
        return [ d[self.site_lookup[i]] for d in self.data ]

    def reset(self):
        self.data = []
        self.timesteps = []

    def to_dict(self):
        d = {'data': self.data,
             'timesteps': self.timesteps,
             'site_lookup': self.site_lookup}

        return d

    @classmethod
    def from_dict(cls, d):
        st = cls( site_lookup=d['site_lookup'])
        st.data = d['data']
        st.timesteps = d['timesteps']
        return st
