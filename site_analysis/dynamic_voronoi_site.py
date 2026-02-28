"""Dynamic Voronoi site representation for crystal structure analysis.

This module provides the DynamicVoronoiSite class, which represents a site with
a center that is dynamically calculated from the positions of a set of reference
atoms.
"""

from __future__ import annotations
import numpy as np
from typing import Any
from .site import Site
from .atom import Atom
from pymatgen.core import Lattice, Structure
from site_analysis.containment import update_pbc_shifts
from site_analysis.pbc_utils import apply_legacy_pbc_correction, unwrap_vertices_to_reference_center

class DynamicVoronoiSite(Site):
    """Site subclass corresponding to Voronoi cells with centres dynamically 
    calculated from the positions of sets of reference atoms.
    
    Unlike standard VoronoiSite objects, which have fixed centers, the positions
    of DynamicVoronoiSite objects adapt to structural changes as the reference 
    atoms move. The site center is calculated as the mean position of the reference 
    atoms, with special handling for periodic boundary conditions.
    
    This makes DynamicVoronoiSite particularly useful for tracking sites in mobile 
    frameworks where the crystal structure deforms during simulation.
    
    Similar to standard Voronoi sites, a single DynamicVoronoiSite cannot determine
    whether an atom is contained within it, as this depends on the positions of all
    other sites. The coordination number of a DynamicVoronoiSite is defined as the 
    number of reference atoms used to calculate its center.
    
    Attributes:
        reference_indices (list[int]): Indices of atoms used as reference to calculate
            the dynamic centre of the site.
        centre_coords (np.ndarray or None): Fractional coordinates of the dynamically 
            calculated site centre. This is None initially and calculated on demand.
            
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
            reference_center: Optional reference centre for PBC handling.
        
        Returns:
            None
        """
        super(DynamicVoronoiSite, self).__init__(label=label)
        self.reference_indices = reference_indices
        self._centre_coords: np.ndarray | None = None
        self._pbc_image_shifts: np.ndarray | None = None
        self._pbc_cached_raw_frac: np.ndarray | None = None
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
        self._pbc_image_shifts = None
        self._pbc_cached_raw_frac = None
        
    def calculate_centre(self, structure: Structure) -> None:
        """Calculate the centre of this site based on the positions of reference atoms.

        Args:
            structure: The pymatgen Structure used to assign
                fractional coordinates to the reference atoms.

        Notes:
            This method handles periodic boundary conditions and calculates
            the centre as the mean of the reference atom positions.

            For bulk analysis prefer ``calculate_centre_from_bulk`` via the
            collection, which pre-extracts coordinates once and avoids
            creating individual ``PeriodicSite`` objects.
        """
        ref_coords = np.array([structure[i].frac_coords for i in self.reference_indices])
        self._compute_corrected_coords(ref_coords, structure.lattice)

    def calculate_centre_from_bulk(self,
            all_frac_coords: np.ndarray,
            lattice: Lattice) -> None:
        """Calculate the site centre from pre-extracted bulk coordinates.

        This is the preferred method for bulk analysis, where the collection
        extracts ``structure.frac_coords`` once and passes it to all sites.
        Avoids creating individual ``PeriodicSite`` objects per reference atom.

        Args:
            all_frac_coords: Full fractional coordinate array from the
                structure, shape ``(n_atoms, 3)``.
            lattice: Lattice for PBC distance calculations.
        """
        ref_coords = all_frac_coords[self.reference_indices]
        self._compute_corrected_coords(ref_coords, lattice)

    def _compute_corrected_coords(self,
            frac_coords: np.ndarray,
            lattice: Lattice) -> None:
        """Apply PBC correction with shift caching and compute the site centre.

        On the first call (or after an anomalous displacement invalidates
        the cache), performs full PBC unwrapping using either the reference
        centre method or the legacy spread-based method. On subsequent
        calls, updates the cached integer image shifts incrementally by
        detecting coordinate wraps (jumps of ~1.0), avoiding the expensive
        27-image distance search.

        Sets ``_centre_coords`` from the mean of PBC-corrected coordinates.

        Args:
            frac_coords: Raw fractional coordinates of the reference atoms,
                shape ``(n_reference, 3)``.
            lattice: Lattice for Cartesian distance calculations
                (used only on full recomputation with reference centres).
        """
        if self._pbc_image_shifts is not None and self._pbc_cached_raw_frac is not None:
            valid, corrected, new_shifts = update_pbc_shifts(
                frac_coords, self._pbc_cached_raw_frac, self._pbc_image_shifts)
            if valid:
                self._pbc_image_shifts = new_shifts
                self._pbc_cached_raw_frac = frac_coords.copy()
                self._centre_coords = np.mean(corrected, axis=0) % 1.0
                return

        # Full computation -- first call or after anomalous displacement
        if self.reference_center is not None:
            corrected, image_shifts = unwrap_vertices_to_reference_center(
                frac_coords, self.reference_center, lattice,
                return_image_shifts=True)
        else:
            corrected = apply_legacy_pbc_correction(frac_coords)
            image_shifts = np.round(corrected - frac_coords).astype(int)
        self._pbc_image_shifts = image_shifts
        self._pbc_cached_raw_frac = frac_coords.copy()
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
                       x: np.ndarray,
                       *args: Any,
                       **kwargs: Any) -> bool:
        """A single dynamic Voronoi site cannot determine whether it contains a given point,
        because the site boundaries are defined by the set of all dynamic Voronoi sites.
        
        Use DynamicVoronoiSiteCollection.assign_site_occupations() instead.
        
        """
        raise NotImplementedError
        
    def as_dict(self) -> dict:
        """Json-serializable dict representation of this DynamicVoronoiSite.
        
        Args:
            None
            
        Returns:
            dict: dictionary representation of this site.
        """
        d = super(DynamicVoronoiSite, self).as_dict()
        d['reference_indices'] = self.reference_indices
        if self._centre_coords is not None:
            d['centre_coords'] = self._centre_coords.tolist()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> DynamicVoronoiSite:
        """Create a DynamicVoronoiSite object from a dict representation.
        
        Args:
            d (dict): The dict representation of this Site.
            
        Returns:
            DynamicVoronoiSite: A new DynamicVoronoiSite instance.
        """
        site = cls(reference_indices=d['reference_indices'])
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
        """Create a list of DynamicVoronoiSite objects from a list of reference indices.
        
        Args:
            reference_indices_list: List of lists, where each inner list contains
                reference indices for a site.
            label: Optional label for all sites. Default is None.
            
        Returns:
            A list of DynamicVoronoiSite objects.
        """
        sites = [cls(reference_indices=ri, label=label) for ri in reference_indices_list]
        return sites
