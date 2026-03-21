"""Spherical site representation for crystal structure analysis.

This module provides the SphericalSite class, which represents a site defined
by a sphere with a specific centre position and radius. Spherical sites are the
simplest site geometry, useful for quick analysis or when the exact shape of
the site is less important than its location.
"""

from __future__ import annotations
from .site import Site
from .atom import Atom
from site_analysis.distances import mic_distance
import numpy as np


class SphericalSite(Site):
    """A site defined by a spherical volume in real space.
    
    Represents a spherical site centred at a position in fractional coordinates
    with a radius in Angstroms (not fractional coordinates).
    
    SphericalSite determines whether atoms are inside the site volume by checking
    if the distance between the atom and the site centre is less than or equal to
    the site's radius. This calculation considers periodic boundary conditions
    via minimum-image convention distances.
    
    Unlike polyhedral sites, spherical sites have a fixed geometry independent of
    atom positions in the structure, making them suitable for applications where
    consistent site volumes are needed regardless of structural distortions.
    
    Attributes:
        frac_coords (np.ndarray): Fractional coordinates of the sphere centre.
        rcut (float): Cutoff radius in Angstroms.
        
    See Also:
        :class:`~site_analysis.site.Site`: Parent class documenting inherited attributes
            (index, label, contains_atoms, trajectory, points, transitions, average_occupation).
            
    Note:
        SphericalSite objects are typically used with 
        :class:`~site_analysis.spherical_site_collection.SphericalSiteCollection`,
        which implements specific assignment logic for handling atoms in overlapping
        sites.
    """
    
    def __init__(self,
        frac_coords: np.ndarray,
        rcut: float,
        label: str | None=None) -> None:
        """Create a SphericalSite instance.
        
        Args:
            frac_coords: Fractional coordinates of the sphere centre.
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
            np.ndarray: Fractional coordinates of the site centre.
        """
        return self.frac_coords
    
    def as_dict(self) -> dict:
        """Returns a dictionary representation of this SphericalSite.
        
        Creates a JSON-serializable dictionary containing all the attributes
        needed to reconstruct this SphericalSite object.
        
        Returns:
            dict: Dictionary containing the SphericalSite's attributes, including
                attributes from the parent Site class plus 'frac_coords' and 'rcut'.
        """
        d = super(SphericalSite, self).as_dict()
        d['frac_coords'] = self.frac_coords
        d['rcut'] = self.rcut
        return d

    def contains_atom(self,
        atom: Atom,
        *,
        lattice_matrix: np.ndarray | None = None) -> bool:
        """Test whether this spherical site contains a specific atom.

        Args:
            atom: The atom to test.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors. Required for distance calculations.

        Returns:
            True if the atom is contained within this site, False otherwise.

        Raises:
            ValueError: If lattice_matrix is not provided.
        """
        if lattice_matrix is None:
            raise ValueError("lattice_matrix is required for SphericalSite.contains_atom()")
        return self.contains_point(
                x=atom.frac_coords,
                lattice_matrix=lattice_matrix)

    def contains_point(self,
        x: np.ndarray,
        *,
        lattice_matrix: np.ndarray | None = None) -> bool:
        """Test if the point x is contained by this spherical site.

        Args:
            x: Fractional coordinates to test.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors. Required for distance calculations.

        Returns:
            True if the point is within the cutoff radius, False otherwise.

        Raises:
            ValueError: If lattice_matrix is not provided.
        """
        if lattice_matrix is None:
            raise ValueError("lattice_matrix is required for SphericalSite.contains_point()")
        dr = mic_distance(self.frac_coords, x, lattice_matrix)
        return dr <= self.rcut

    @classmethod
    def from_dict(cls,
            d: dict) -> SphericalSite:
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


