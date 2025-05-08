from .site import Site
from typing import Dict, Any, Optional
import numpy as np

class VoronoiSite(Site):
    """Site subclass corresponding to Voronoi cells centered
    on fixed positions.

    Attributes:
        frac_coords (np.array): Fractional coordinates of the site centre.

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

    def as_dict(self) -> Dict:
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
