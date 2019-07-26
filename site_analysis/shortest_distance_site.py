from .site import Site

class ShortestDistanceSite(Site):
    
    def __init__(self, frac_coords, label=None):
        super(ShortestDistanceSite, self).__init__(label=label)
        self.frac_coords = frac_coords

    def as_dict(self):
        d = super(ShortestDistanceSite, self).as_dict()
        d['frac_coords'] = self.frac_coords
        return d

    @classmethod
    def from_dict(cls, d):
        shortest_distance_site = cls( frac_coords=d['frac_coords'] )
        shortest_distance_site.label = d.get('label')
        return shortest_distance_site 

