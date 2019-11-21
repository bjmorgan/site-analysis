from .site_collection import SiteCollection

class PolyhedralSiteCollection(SiteCollection):

    def __init__(self, sites):
        super(PolyhedralSiteCollection, self).__init__(sites)
        self._neighbouring_sites = self.construct_neighbouring_sites()

    def construct_neighbouring_sites(self):
        """
        Find all polyhedral sites that are face-sharing neighbours.

        Any polyhedral sites that share 3 or more vertices are considered
        to share a face.

        Args:
            None

        Returns:
            (dict): Dictionary of `int`: `list` entries. 
                Keys are site indices. Values are lists of PolyhedralSite objects.

        """
        neighbours = {}
        for site_i in self.sites:
            neighbours[site_i.index] = []
            for site_j in self.sites:
                if site_i is site_j:
                    continue
                # count the number of shared vertices.
                n_shared_vertices = len( set(site_i.vertex_indices) & set(site_j.vertex_indices) ) 
                # if this is >= 3 these sites share a face.
                if n_shared_vertices >= 3:
                    neighbours[site_i.index].append(site_j)
        return neighbours

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

    def neighbouring_sites(self, index):
        return self._neighbouring_sites[index] 


    
 
