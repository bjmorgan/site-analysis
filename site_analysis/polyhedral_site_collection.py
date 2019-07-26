from .site_collection import SiteCollection

class PolyhedralSiteCollection(SiteCollection):

    def analyse_structure(self, atoms, structure):
        for a in atoms:
            a.get_coords(structure)
        for s in self.sites:
            s.get_vertex_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self, atoms, structure):
        self.reset_site_occupations()
        for atom in atoms:
            if atom.in_site:
                # first check the site last occupied
                previous_site = next(s for s in self.sites if s.index == atom.in_site)
                if previous_site.contains_atom(atom):
                    self.update_occupation( previous_site, atom )
                    continue # atom has not moved
                else: # default is atom does not occupy any sites
                    atom.in_site = None
            for s in self.sites:
                if s.contains_atom(atom):
                    self.update_occupation( s, atom )
                    break

    
 
