from .site_collection import SiteCollection
import numpy as np

class VoronoiSiteCollection(SiteCollection):

    def analyse_structure(self, atoms, structure):
        for a in atoms:
            a.assign_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self, atoms, structure):
        self.reset_site_occupations()
        lattice = structure.lattice
        site_coords = np.array( [ s.frac_coords for s in self.sites ] )
        atom_coords = np.array( [ a.frac_coords for a in atoms ] )
        dist_matrix = lattice.get_all_distances(site_coords, atom_coords)
        site_list_indices = np.argmin( dist_matrix, axis=0 )
        for atom, site_list_index in zip( atoms, site_list_indices):
            site = self.sites[site_list_index]
            self.update_occupation( site, atom )

    
 
