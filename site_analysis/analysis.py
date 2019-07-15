from collections import Counter
from .atoms_trajectory import AtomsTrajectory
from .sites_trajectory import SitesTrajectory

class Analysis(object):
    
    def __init__(self, sites, atoms):
        self.sites = sites
        self.atoms = atoms
        self.atoms_trajectory = AtomsTrajectory(atoms)
        self.sites_trajectory = SitesTrajectory(sites)
        self.timesteps = []
        self.previous_occupations = {}
 
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
                    continue
                else: # default is atom does not occupy any sites
                    atom.in_site = None
            for s in self.sites:
                if s.contains_atom(atom):
                    update_occupation( s, atom )
                    break
            if atom.in_site is None:
                # Not able to find this atom inside a polyhedron
                # Recalculate using more accurate, slower algorithm for unassigned atoms
                if previous_site.contains_atom_accurate(atom):
                    update_occupation( previous_site, atom )
                    continue
                for s in self.sites:
                    if s.contains_atom_accurate(atom):
                        update_occupation( s, atom )
                        break
                    
    def coordination_summary(self):
        return Counter( [ s.coordination_number for s in self.sites ] )
    
    @property
    def atom_sites(self):
        return [ atom.in_site for atom in self.atoms ]
        
    @property
    def site_occupations(self):
        return [ s.contains_atoms for s in self.sites ]

    def append_timestep(self, structure, t=None):
        self.analyse_structure(structure)
        self.atoms_trajectory.append_timestep(self.atom_sites, t=t)
        self.sites_trajectory.append_timestep(self.site_occupations, t=t)
        self.timesteps.append(t)

    def reset(self):
        self.atoms_trajectory.reset()
        self.sites_trajectory.reset()
        self.timesteps = [] 

    @property
    def at(self):
        return self.atoms_trajectory

    @property
    def st(self):
        return self.sites_trajectory

def update_occupation( site, atom ):
    site.contains_atoms.append( atom.index )
    atom.in_site = site.index
