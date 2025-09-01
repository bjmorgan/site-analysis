"""Voronoi site representation for crystal structure analysis.

This module provides the VoronoiSite class, which represents a site defined
by a Voronoi cell centered at a fixed position.
"""

from site_analysis.site import Site
from typing import Any, Optional
import numpy as np

class VoronoiSite(Site):
    """Site subclass corresponding to Voronoi cells centered on fixed positions.
    
    Voronoi sites divide space into regions where each point in a region is closer 
    to its site center than to any other site center.
    
    Unlike other site types, a single VoronoiSite cannot determine whether an atom
    is contained within it, as this calculation depends on the positions of all
    other Voronoi sites in the structure. Therefore, the contains_point method
    is not implemented directly in this class.
    
    Attributes:
        frac_coords (np.ndarray): Fractional coordinates of the site centre.
        
    See Also:
        :class:`~site_analysis.site.Site`: Parent class documenting inherited attributes
            (index, label, contains_atoms, trajectory, points, transitions, average_occupation).
            
    Important:
        Do not use VoronoiSite.contains_point() directly. Instead, use
        :class:`~site_analysis.voronoi_site_collection.VoronoiSiteCollection` and its
        assign_site_occupations() method to determine which atoms belong to which 
        Voronoi sites.
    """
    
    def __init__(self,
                 frac_coords: np.ndarray,
                 label: Optional[str]=None) -> None:
        """Create a ``VoronoiSite`` instance.
        
        Args:
            frac_coords (np.array): Fractional coordinates of the site centre.
            label (:str:, optional): Optional label for this site. Default is `None`.

        Returns:
            None

        """
        super(VoronoiSite, self).__init__(label=label)
        self.frac_coords = frac_coords
        
    def __repr__(self) -> str:
        """Return a string representation of this Voronoi site.
        
        Returns:
            str: A string representation of the site including its 
                class name and important attributes.
        """
        string = ('site_analysis.VoronoiSite('
                f'index={self.index}, '
                f'label={self.label}, '
                f'frac_coords={self.frac_coords}, '
                f'contains_atoms={self.contains_atoms})')
        return string

    def as_dict(self) -> dict:
        """Json-serializable dict representation of this VoronoiSite.

        Args:
            None

        Returns:
            (dict)

        """
        d = super(VoronoiSite, self).as_dict()
        d['frac_coords'] = self.frac_coords
        return d

    @classmethod
    def from_dict(cls, d):
        """Create a VoronoiSite object from a dict representation.

        Args:
            d (dict): The dict representation of this Site.

        Returns:
            (VoronoiSite)

        """
        voronoi_site = cls(frac_coords=d['frac_coords'])
        voronoi_site.label = d.get('label')
        return voronoi_site 

    @property
    def centre(self) -> np.ndarray:
        """Returns the centre position of this site.

        Args:
            None

        Returns:
            np.ndarray

        """
        return self.frac_coords

    def contains_point(self,
                       x: np.ndarray,
                       *args: Any,
                       **kwargs: Any) -> bool:
        """A single Voronoi site cannot determine whether it contains a given point, because
        the site boundaries are defined by the set of all Voronoi sites.

        Use VoronoiSiteCollection.assign_site_occupations() instead.

        """
        raise NotImplementedError
