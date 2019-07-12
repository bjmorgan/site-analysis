from collections import Counter

class Analysis(object):
    
    def __init__(self, polyhedra, atoms):
        self.polyhedra = polyhedra
        self.atoms = atoms
        
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
                    atom.in_polyhedron = p.index
                    p.contains_atoms.append( atom.index )
                    
    def coordination_summary(self):
        return Counter( [ p.coordination_number for p in self.polyhedra ] )
    
    @property
    def atom_sites(self):
        return [ atom.in_polyhedron for atom in self.atoms ]
        
    @property
    def site_occupations(self):
        return [ p.contains_atoms for p in self.polyhedra ]
