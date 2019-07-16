from collections import Counter
from .atoms_trajectory import AtomsTrajectory
from .sites_trajectory import SitesTrajectory

class Analysis(object):
    
    def __init__(self, sites, atoms):
        self.sites = sites
        self.atoms = atoms
#        self.atoms_trajectory = AtomsTrajectory(atoms)
#        self.sites_trajectory = SitesTrajectory(sites)
        self.timesteps = []
        self.previous_occupations = {}
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        self.site_lookup = {s.index: i for i, s in enumerate(sites)}

    def atom_by_index(self, i):
        return self.atoms[self.atom_lookup[i]] 

    def site_by_index(self, i):
        return self.sites[self.site_lookup[i]] 

    def analyse_structure(self, structure):
        for a in self.atoms:
            a.get_coords(structure)
        for s in self.sites:
            s.get_vertex_coords(structure)
        self.assign_site_occupations(structure)
        
    def assign_site_occupations(self, structure):
        for s in self.sites:
            s.contains_atoms = []
        for atom in self.atoms:
            if atom.in_site:
                # first check the site last occupied
                previous_site = next(s for s in self.sites if s.index == atom.in_site)
                if previous_site.contains_atom(atom):
                    update_occupation( previous_site, atom )
                    continue # atom has not moved
                else: # default is atom does not occupy any sites
                    atom.in_site = None
            for s in self.sites:
                if s.contains_atom(atom):
                    update_occupation( s, atom )
                    break
                    
    def site_coordination_numbers(self):
        return Counter( [ s.coordination_number for s in self.sites ] )

    def site_labels(self):
        return [ s.label for s in self.sites ]
   
    @property
    def atom_sites(self):
        return [ atom.in_site for atom in self.atoms ]
        
    @property
    def site_occupations(self):
        return [ s.contains_atoms for s in self.sites ]

    def append_timestep(self, structure, t=None):
        self.analyse_structure(structure)
        for atom in self.atoms:
            atom.trajectory.append( atom.in_site )
        for site in self.sites:
            site.trajectory.append( site.contains_atoms )
        self.timesteps.append(t)

    def reset(self):
        for atom in self.atoms:
            atom.reset()
        for site in self.sites:
            site.reset()
        self.timesteps = [] 

    @property
    def atoms_trajectory(self):
        return list(map(list, zip(*[atom.trajectory for atom in self.atoms])))

    @property
    def sites_trajectory(self):
        return list(map(list, zip(*[site.trajectory for site in self.sites])))

    @property
    def at(self):
        return self.atoms_trajectory

    @property
    def st(self):
        return self.sites_trajectory

def update_occupation( site, atom ):
    site.contains_atoms.append( atom.index )
    atom.in_site = site.index
