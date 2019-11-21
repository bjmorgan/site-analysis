from .site import Site

class VoronoiSite(Site):
    
    def __init__(self, frac_coords, label=None):
        super(VoronoiSite, self).__init__(label=label)
        self.frac_coords = frac_coords

    def as_dict(self):
        d = super(VoronoiSite, self).as_dict()
        d['frac_coords'] = self.frac_coords
        return d

    @classmethod
    def from_dict(cls, d):
        voronoi_site = cls( frac_coords=d['frac_coords'] )
        voronoi_site.label = d.get('label')
        return voronoi_site 

