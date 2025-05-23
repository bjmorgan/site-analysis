"""Spherical site representation for crystal structure analysis.

This module provides the SphericalSite class, which represents a site defined
by a sphere with a specific center position and radius. Spherical sites are the
simplest site geometry, useful for quick analysis or when the exact shape of
the site is less important than its location.

SphericalSite determines whether atoms are inside the site volume by checking
if the distance between the atom and the site center is less than or equal to
the site's radius. This calculation considers periodic boundary conditions using
the structure's lattice.

Unlike polyhedral sites, spherical sites have a fixed geometry independent of
atom positions in the structure, making them suitable for applications where
consistent site volumes are needed regardless of structural distortions.

Note:
    SphericalSite objects are typically used with SphericalSiteCollection,
    which implements specific assignment logic for handling atoms in overlapping
    sites.
"""

from __future__ import annotations
from .site import Site
from typing import Optional, Dict, Any
from .atom import Atom
from pymatgen.core.lattice import Lattice
import numpy as np


class SphericalSite(Site):
    """A site defined by a spherical volume in real space.
    
    Represents a spherical site centered at a position in fractional coordinates
    with a radius in Angstroms (not fractional coordinates).
    
    Attributes:
        index (int): Unique numerical identifier for this site. Automatically
            assigned during initialisation by the parent Site class.
        frac_coords (np.ndarray): Fractional coordinates of the sphere center.
        rcut (float): Cutoff radius in Angstroms.
        label (str, optional): Optional label for this site.
        contains_atoms (list): Indices of atoms contained within this site.
        trajectory (list): History of atom occupations at each timestep.
        points (list): Fractional coordinates of atoms assigned to this site.
        transitions (collections.Counter): Record of atom transitions between sites.
            Keys are destination site indices, values are transition counts.
    """
    
    def __init__(self,
        frac_coords: np.ndarray,
        rcut: float,
        label: Optional[str]=None) -> None:
        """Create a SphericalSite instance.
        
        Args:
            frac_coords: Fractional coordinates of the sphere center.
            rcut: Cutoff radius in Angstroms.
            label: Optional label for this site. Default is None.
        
        Returns:
            None
        """
        super(SphericalSite, self).__init__(label=label)
        self.frac_coords = frac_coords
        self.rcut = rcut
        
    def __repr__(self) -> str:
        """Return a string representation of this spherical site.
        
        Returns:
            str: A string representation of the site including its 
                class name and important attributes.
        """
        string = ('site_analysis.SphericalSite('
                 f'index={self.index}, '
                 f'label={self.label}, '
                 f'frac_coords={self.frac_coords}, '
                 f'rcut={self.rcut}, '
                 f'contains_atoms={self.contains_atoms})')
        return string

    @property
    def centre(self) -> np.ndarray:
        """Returns the fractional coordinates of the spherical site's centre.
        
        Returns:
            np.ndarray: Fractional coordinates of the site center.
        """
        return self.frac_coords
    
    def as_dict(self) -> Dict:
        """Returns a dictionary representation of this SphericalSite.
        
        Creates a JSON-serializable dictionary containing all the attributes
        needed to reconstruct this SphericalSite object.
        
        Returns:
            Dict: Dictionary containing the SphericalSite's attributes, including
                attributes from the parent Site class plus 'frac_coords' and 'rcut'.
        """
        d = super(SphericalSite, self).as_dict()
        d['frac_coords'] = self.frac_coords
        d['rcut'] = self.rcut
        return d

    def contains_atom(self,
        atom: Atom,
        lattice: Optional[Lattice] = None,  # Technically optional for signature compatibility, but required
        *args: Any,
        **kwargs: Any) -> bool:
        """Test whether this spherical site contains a specific atom.
        
        Args:
            atom: The atom to test.
            lattice: Lattice object for distance calculations.
                Although marked as optional for inheritance reasons,
                this parameter is actually required.
                
        Returns:
            True if the atom is contained within this site, False otherwise.
            
        Raises:
            ValueError: If lattice is not provided.
            TypeError: If lattice is not a Lattice object.
        """
        if not lattice:
            raise ValueError("Lattice is required for SphericalSite.contains_atom() to calculate real-space distances")
        elif not isinstance(lattice, Lattice):
            raise TypeError(f"Expected Lattice object, got {type(lattice).__name__}. SphericalSite requires a valid Lattice to calculate distances.")
        return self.contains_point(
                x=atom.frac_coords,
                lattice=lattice)

    def contains_point(self,
        x: np.ndarray,
        lattice: Optional[Lattice]=None,  # Technically optional for signature compatibility, but required
        *args: Any,
        **kwargs: Any) -> bool:
        """Test if the point x is contained by this spherical site.
        
        Args:
            x: Fractional coordinates to test.
            lattice: Lattice object for distance calculations. 
                Although marked as optional for inheritance reasons, 
                this parameter is actually required.
                
        Returns:
            True if the point is within the cutoff radius, False otherwise.
                
        Raises:
            ValueError: If lattice is not provided.
            TypeError: If lattice is not a Lattice object.
        """
        if not lattice:
            raise ValueError("Lattice is required for SphericalSite.contains_point() to calculate real-space distances")
        elif not isinstance(lattice, Lattice):
            raise TypeError(f"Expected Lattice object, got {type(lattice).__name__}. SphericalSite requires a valid Lattice to calculate distances.")
        dr = float(lattice.get_distance_and_image(self.frac_coords, x)[0])
        return dr <= self.rcut

    @classmethod
    def from_dict(cls,
            d: Dict) -> SphericalSite:
        """Create a SphericalSite object from a dictionary representation.
        
        This is the complementary method to `as_dict()`, allowing SphericalSite
        objects to be reconstructed from their dictionary representation.
        
        Args:
            d: Dictionary containing at minimum 'frac_coords' and 'rcut' keys.
               May also contain 'label' and other attributes from the parent
               Site class.
        
        Returns:
            SphericalSite: A new SphericalSite instance with attributes set
                according to the provided dictionary.
        """
        spherical_site = cls(frac_coords=d['frac_coords'],
                             rcut=d['rcut'])
        spherical_site.label = d.get('label')
        return spherical_site


