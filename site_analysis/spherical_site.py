from .site import Site
from typing import Optional, Dict
from pymatgen.core.lattice import Lattice
import numpy as np

class SphericalSite(Site):
    
    def __init__(self,
            frac_coords: np.ndarray,
            rcut: float,
            label: Optional[str]=None) -> None:
        super(SphericalSite, self).__init__(label=label)
        self.frac_coords = frac_coords
        self.rcut = rcut

    def as_dict(self) -> Dict:
        d = super(SphericalSite, self).as_dict()
        d['frac_coords'] = self.frac_coords
        d['rcut'] = self.rcut
        return d

    def contains_atom(self, atom, lattice):
        return self.contains_point(atom.frac_coords, lattice)

    def contains_point(self, x, lattice):
        dr = lattice.get_distance_and_image( self.frac_coords, x )[0]
        return dr <= self.rcut

    @classmethod
    def from_dict(cls, d):
        spherical_site = cls( frac_coords=d['frac_coords'],
                              rcut=d['rcut'] )
        spherical_site.label = d.get('label')
        return spherical_site 


