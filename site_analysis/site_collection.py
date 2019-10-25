class SiteCollection(object):

    def __init__(self, sites):
        self.sites = sites

    def assign_site_occupations(self, atoms, structure):
        raise NotImplementedError('assign_site_occupations should be implemented in the inherited class')

    def analyse_structure(self, atoms, structure):
        raise NotImplementedError('analyse_structure should be implemented in the inherited class')

    def update_occupation(self, site, atom):
        site.contains_atoms.append( atom.index )
        site.points.append( atom.frac_coords )
        atom.in_site = site.index

    def reset_site_occupations(self):
        for s in self.sites:
            s.contains_atoms = []

    def sites_contain_points(self, structure, points):
        check = all([s.contains_point(p,structure=structure) for s, p in zip(self.sites, points)])
        self.reset_site_occupations()
        return check
  
        

