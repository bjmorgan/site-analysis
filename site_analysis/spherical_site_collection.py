import numpy as np
from typing import List
from pymatgen.core import Structure
from site_analysis.atom import Atom
from site_analysis.site_collection import SiteCollection

class SphericalSiteCollection(SiteCollection):


    def analyse_structure(self,
            atoms: List[Atom],
            structure: Structure) -> None:
        for a in atoms:
            a.assign_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self,
            atoms: List[Atom],
            structure: Structure) -> None:
        self.reset_site_occupations()
        for atom in atoms:
            if atom.in_site:
                # first check the site last occupied
                previous_site = next(s for s in self.sites if s.index == atom.in_site)
                if previous_site.contains_atom(atom, structure.lattice):
                    self.update_occupation( previous_site, atom )
                    continue # atom has not moved
                else: # default is atom does not occupy any sites
                    atom.in_site = None
            for s in self.sites:
                if s.contains_atom(atom, structure.lattice):
                    self.update_occupation( s, atom )
                    break

 
