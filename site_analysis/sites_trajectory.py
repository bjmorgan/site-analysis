class SitesTrajectory(object):
    
    def __init__(self, sites):
        self.sites = sites
        #self.site_lookup = {s.index: i for i, s in enumerate(sites)}

    def by_site_indices(self, indices):
        return SitesTrajectory( sites=[s for s in self.sites if s.index in indices ] )

    def by_site_label(self, label):
        return SitesTrajectory( sites=[s for s in self.sites is s.label is label] )

    @property
    def data(self):
        return list(map(list, zip(*[site.trajectory for site in self.sites])))

    def as_dict(self):
        d = {'sites': self.sites}
        return d

    @classmethod
    def from_dict(cls, d):
        st = cls(sites=d['sites'])
        return st
