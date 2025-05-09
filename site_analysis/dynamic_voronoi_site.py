"""Dynamic Voronoi site representation for crystal structure analysis.

This module provides the DynamicVoronoiSite class, which represents a site with
a center that is dynamically calculated from the positions of a set of reference
atoms. Unlike standard VoronoiSite objects, which have fixed centers, the positions
of DynamicVoronoiSite objects adapt to structural changes as the reference atoms move.

The site center is calculated as the mean position of the reference atoms, with special
handling for periodic boundary conditions. This makes DynamicVoronoiSite particularly
useful for tracking sites in mobile frameworks where the crystal structure deforms
during simulation.

Similar to standard Voronoi sites, a single DynamicVoronoiSite cannot determine
whether an atom is contained within it, as this depends on the positions of all
other sites. This class is designed to be used with DynamicVoronoiSiteCollection,
which handles the Voronoi tessellation logic.

The coordination number of a DynamicVoronoiSite is defined as the number of 
reference atoms used to calculate its center.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Any, Dict
from .site import Site
from .atom import Atom
from pymatgen.core import Structure

class DynamicVoronoiSite(Site):
	"""Site subclass corresponding to Voronoi cells with centres
	dynamically calculated from the positions of sets of reference
	atoms.
	
	Attributes:
	    reference_indices (List[int]): Indices of atoms used as reference to calculate
		    the dynamic centre of the site.
		_centre_coords (np.ndarray): Fractional coordinates of the dynamically calculated
			site centre. This is None initially and calculated on demand.
			
		"""
		
	def __init__(self,
		reference_indices: List[int],
		label: Optional[str] = None) -> None:
		"""Create a ``DynamicVoronoiSite`` instance.__abs__()
			
		Args:
			reference_indices (List[int]): List of atom indices whose
			positions will be used to dynamically calculate the centre of this site.
			label (str, optional): Optional label for this site. Default is `None`.
		
		Returns:
			None
		"""
		super(DynamicVoronoiSite, self).__init__(label=label)
		self.reference_indices = reference_indices
		self._centre_coords: Optional[np.ndarray] = None
		
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
		
	def calculate_centre(self, structure: Structure) -> None:
		"""Calculate the centre of this site based on the positions of reference atoms.
		
		Args:
			structure (Structure): The pymatgen Structure used to assign
				fractional coordinates to the reference atoms.
		
		Returns:
			None
			
		Notes:
			This method handles periodic boundary conditions and calculates
			the centre as the mean of the reference atom positions.
		"""
		# Get fractional coordinates of reference atoms
		ref_coords = np.array([structure[i].frac_coords for i in self.reference_indices])
		
		# Handle periodic boundary conditions:
		# If the range of fractional coordinates along x, y, or z 
		# exceeds 0.5, assume that this site wraps around the 
		# periodic boundary in that dimension. 
		# Fractional coordinates for that dimension that are less 
		# than 0.5 will be incremented by 1.0 
		for dim in range(3):
			spread = np.max(ref_coords[:, dim]) - np.min(ref_coords[:, dim])
			if spread > 0.5:
				ref_coords[ref_coords[:, dim] < 0.5, dim] += 1.0
		
		# Calculate the centre as the mean of the reference atom coordinates
		centre = np.mean(ref_coords, axis=0)
		
		# Wrap the centre back into the unit cell [0, 1)
		centre = centre % 1.0
		
		self._centre_coords = centre
		
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
		
	def as_dict(self) -> Dict:
		"""Json-serializable dict representation of this DynamicVoronoiSite.
		
		Args:
			None
			
		Returns:
			Dict: Dictionary representation of this site.
		"""
		d = super(DynamicVoronoiSite, self).as_dict()
		d['reference_indices'] = self.reference_indices
		if self._centre_coords is not None:
			d['centre_coords'] = self._centre_coords.tolist()
		return d
	
	@classmethod
	def from_dict(cls, d: Dict) -> DynamicVoronoiSite:
		"""Create a DynamicVoronoiSite object from a dict representation.
		
		Args:
			d (Dict): The dict representation of this Site.
			
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
		reference_indices_list: List[List[int]],
		label: Optional[str] = None) -> List[DynamicVoronoiSite]:
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
