"""Dynamic Voronoi site representation for crystal structure analysis.

This module provides the DynamicVoronoiSite class, which represents a site with
a centre that is dynamically calculated from the positions of a set of reference
atoms.
"""

from __future__ import annotations
import numpy as np
from .site import Site
from .atom import Atom
from site_analysis.distances import _as_point
from site_analysis.pbc_utils import correct_pbc
from site_analysis._reference_centre_deprecation import (
    on_users_behalf, warn_without_reference_centres)

class DynamicVoronoiSite(Site):
    """Site subclass corresponding to Voronoi cells with centres dynamically 
    calculated from the positions of sets of reference atoms.
    
    Unlike standard VoronoiSite objects, which have fixed centres, the positions
    of DynamicVoronoiSite objects adapt to structural changes as the reference
    atoms move. The site centre is calculated as the mean position of the reference
    atoms, with special handling for periodic boundary conditions.
    
    This makes DynamicVoronoiSite particularly useful for tracking sites in mobile 
    frameworks where the crystal structure deforms during simulation.
    
    Similar to standard Voronoi sites, a single DynamicVoronoiSite cannot determine
    whether an atom is contained within it, as this depends on the positions of all
    other sites. The coordination number of a DynamicVoronoiSite is defined as the 
    number of reference atoms used to calculate its centre.
    
    Attributes:
        reference_indices (list[int]): Indices of atoms used as reference to calculate
            the dynamic centre of the site.
        centre_coords (np.ndarray or None): Fractional coordinates of the dynamically 
            calculated site centre. This is None initially and calculated on demand.
        reference_center (np.ndarray or None): Reference centre for PBC
            handling, or None for a site created without one (deprecated).
            
    See Also:
        :class:`~site_analysis.site.Site`: Parent class documenting inherited attributes
            (index, label, contains_atoms, trajectory, points, transitions, average_occupation).
            
    Important:
        This class is designed to be used with 
        :class:`~site_analysis.dynamic_voronoi_site_collection.DynamicVoronoiSiteCollection`,
        which handles the Voronoi tessellation logic.
    """
        
    def __init__(self,
        reference_indices: list[int],
        label: str | None = None,
        reference_center: np.ndarray | None = None) -> None:
        """Create a ``DynamicVoronoiSite`` instance.
            
        Args:
            reference_indices: List of atom indices whose positions will be used to dynamically calculate the centre of this site.
            label: Optional label for this site.
            reference_center: Reference centre for PBC handling, usually the
                position of the site's central atom, as three finite numbers,
                stored as a float array. The reference atoms are unwrapped to
                their images nearest it. Deprecated: without it, the reference
                atoms are unwrapped by the spread-based rule
                (``pbc_utils.apply_legacy_pbc_correction``); version 2.0
                will require it.
        
        Returns:
            None

        Warns:
            FutureWarning: If ``reference_center`` is None.

        Raises:
            ValueError: If reference_center is not None and not three
                finite numbers.
        """
        if reference_center is None:
            warn_without_reference_centres(
                "DynamicVoronoiSite without a reference_center is deprecated and will "
                "be removed in version 2.0; use reference_center, usually the "
                "position of the site's central atom.")
        else:
            reference_center = _as_point(reference_center, "reference_center")
        super(DynamicVoronoiSite, self).__init__(label=label)
        self.reference_indices = reference_indices
        self._centre_coords: np.ndarray | None = None
        self.reference_center = reference_center
        
    def __repr__(self) -> str:
        string = ('site_analysis.DynamicVoronoiSite('
                f'index={self.index}, '
                f'label={self.label}, '
                f'reference_indices={self.reference_indices}, '
                f'contains_atoms={self.contains_atoms})')
        return string

    def reset(self) -> None:
        """Reset the site state.
        
        Clears the calculated centre coordinates and resets the site occupation data.
        
        Args:
            None
            
        Returns:
            None
        """
        super(DynamicVoronoiSite, self).reset()
        self._centre_coords = None
        
    def calculate_centre(self,
            all_frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Calculate the centre of this site from reference atom positions.

        Handles periodic boundary conditions and wraps the result into
        [0, 1).

        Args:
            all_frac_coords: Full fractional coordinate array from the
                structure, shape ``(n_atoms, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.
        """
        ref_coords = all_frac_coords[self.reference_indices]
        corrected, _ = correct_pbc(ref_coords, self.reference_center, lattice_matrix)
        self._centre_coords = np.mean(corrected, axis=0) % 1.0
        
    @property
    def centre(self) -> np.ndarray:
        """Returns the centre position of this site.
        
        This method returns the cached centre coordinates or raises an error
        if they haven't been calculated yet.
        
        Args:
            None
            
        Returns:
            np.ndarray: Fractional coordinates of the site centre.
            
        Raises:
            RuntimeError: If the centre coordinates have not been calculated yet.
            
        """
        if self._centre_coords is None:
            raise RuntimeError("Centre coordinates for this DynamicVoronoiSite have not been calculated yet.")
        return self._centre_coords
        
    def contains_point(self,
                       x: np.ndarray) -> bool:
        """A single dynamic Voronoi site cannot determine whether it contains a given point,
        because the site boundaries are defined by the set of all dynamic Voronoi sites.
        
        Use DynamicVoronoiSiteCollection.assign_site_occupations() instead.
        
        """
        raise NotImplementedError
        
    def as_dict(self) -> dict:
        """Dictionary of this site's attributes, which may contain numpy arrays.

        For example, the recorded ``points`` are numpy arrays. It also contains
        the ``reference_center`` as a list of three floats (``None`` if the site
        has none) and, if it has been calculated, the ``centre_coords``.
        
        Args:
            None
            
        Returns:
            dict: dictionary representation of this site.
        """
        d = super(DynamicVoronoiSite, self).as_dict()
        d['reference_indices'] = self.reference_indices
        d['reference_center'] = (
            None if self.reference_center is None else self.reference_center.tolist())
        if self._centre_coords is not None:
            d['centre_coords'] = self._centre_coords.tolist()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> DynamicVoronoiSite:
        """Create a DynamicVoronoiSite object from a dict representation.

        Deprecated for a dict without a reference centre, where
        ``reference_center`` is None or the key is missing, as in dicts
        written by earlier versions: it gives a site with no reference
        centre, and support for such dicts will be removed in version 2.0.
        
        Args:
            d (dict): The dict representation of this Site.
            
        Returns:
            DynamicVoronoiSite: A new DynamicVoronoiSite instance.

        Warns:
            FutureWarning: If the dict has no reference centre.

        Raises:
            ValueError: If ``reference_center`` is not None and not three
                finite numbers.
        """
        reference_center = d.get('reference_center')
        if reference_center is None:
            warn_without_reference_centres(
                "DynamicVoronoiSite.from_dict for a dict without a reference_center is "
                "deprecated and will be removed in version 2.0; use a dict with a "
                "reference_center, usually the position of the site's central atom.")
        with on_users_behalf():
            site = cls(reference_indices=d['reference_indices'],
                       reference_center=reference_center)
        site.label = d.get('label')
        site.contains_atoms = d.get('contains_atoms', [])
        site.trajectory = d.get('trajectory', [])
        site.points = d.get('points', [])
        
        if 'transitions' in d:
            site.transitions = d['transitions']
            
        if 'centre_coords' in d:
            site._centre_coords = np.array(d['centre_coords'])
            
        return site
        
    @property
    def coordination_number(self) -> int:
        """Returns the coordination number of this site.
        
        For a DynamicVoronoiSite, the "coordination number" is the number of reference atoms.
        
        Args:
            None
            
        Returns:
            int: The number of reference atoms.
        """
        return len(self.reference_indices)
        
    @property
    def cn(self) -> int:
        """Coordination number for this site, defined as the number of
        reference atoms.
    
        Convenience property for coordination_number()
    
        Returns:
            int
    
        """
        return self.coordination_number
        
    @classmethod
    def sites_from_reference_indices(cls,
        reference_indices_list: list[list[int]],
        label: str | None = None) -> list[DynamicVoronoiSite]:
        """Create sites without reference centres from lists of reference indices.

        Deprecated: use ``TrajectoryBuilder.with_dynamic_voronoi_sites`` (or
        ``DynamicVoronoiSite`` with ``reference_center``). This will be
        removed in version 2.0, with sites without reference centres.
        
        Args:
            reference_indices_list: List of lists, where each inner list contains
                reference indices for a site.
            label: Optional label for all sites. Default is None.
            
        Returns:
            A list of DynamicVoronoiSite objects, without reference centres.

        Warns:
            FutureWarning: Always, once per call.
        """
        warn_without_reference_centres(
            "DynamicVoronoiSite.sites_from_reference_indices is deprecated and will be "
            "removed in version 2.0; use TrajectoryBuilder.with_dynamic_voronoi_sites "
            "(or DynamicVoronoiSite with reference_center).")
        with on_users_behalf():
            return [cls(reference_indices=ri, label=label) for ri in reference_indices_list]
