from collections import Counter
from .atoms_trajectory import AtomsTrajectory
from .sites_trajectory import SitesTrajectory

class Analysis(object):
    
    def __init__(self, polyhedra, atoms):
        self.polyhedra = polyhedra
        self.atoms = atoms
        self.atoms_trajectory = AtomsTrajectory(atoms)
        self.sites_trajectory = SitesTrajectory(polyhedra)
        self.timesteps = []
        
    def analyse_structure(self, structure):
        for a in self.atoms:
            a.get_coords(structure)
        for p in self.polyhedra:
            p.get_vertex_coords(structure)
        self.assign_site_occupations(structure)
        
    def assign_site_occupations(self, structure):
        for p in self.polyhedra:
            p.contains_atoms = []
        for atom in self.atoms:
            for p in self.polyhedra:
                if p.contains_atom(atom):
                    atom.in_site = p.index
                    p.contains_atoms.append( atom.index )
                    
    def coordination_summary(self):
        return Counter( [ p.coordination_number for p in self.polyhedra ] )
    
    @property
    def atom_sites(self):
        return [ atom.in_site for atom in self.atoms ]
        
    @property
    def site_occupations(self):
        return [ p.contains_atoms for p in self.polyhedra ]

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
