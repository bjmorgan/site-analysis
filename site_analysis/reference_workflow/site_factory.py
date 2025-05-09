"""Factory for creating site objects from coordination environments.

This module provides the SiteFactory class, which creates different types of site
objects based on coordination environments defined by atom indices in a crystal
structure. This factory simplifies the creation of complex site objects by handling
the details of vertex coordinate assignment and validation.

The SiteFactory supports creating:
- PolyhedralSite objects: Sites defined by polyhedra with vertices at specific atom positions
- DynamicVoronoiSite objects: Sites with centers dynamically calculated from reference atom positions

It provides validation of coordination environments, ensures consistent labeling,
and manages the initialisation of site objects with the appropriate structural data.

This module is part of the reference-based workflow, which creates sites in one
structure based on coordination environments identified in a reference structure.
"""

from typing import List, Optional, Union, Any
import numpy as np
from pymatgen.core import Structure

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site


class SiteFactory:
	"""Factory for creating site objects from coordination environments.
	
	This class creates PolyhedralSite or DynamicVoronoiSite objects from
	environments defined as lists of atom indices.
	
	Attributes:
		structure (Structure): The structure providing atom coordinates.
	"""
	
	def __init__(self, structure: Structure):
		"""Initialize SiteFactory with a structure.
		
		Args:
			structure: A pymatgen Structure used to assign coordinates to sites.
		"""
		self.structure = structure
	
	def create_polyhedral_sites(
		self, 
		environments: List[List[int]], 
		label: Optional[str] = None,
		labels: Optional[List[str]] = None
	) -> List[PolyhedralSite]:
		"""Create PolyhedralSite objects from coordination environments.
		
		Args:
			environments: List of environments, where each environment is a list
				of atom indices defining the vertices of a polyhedral site.
			label: Optional label to assign to all sites.
			labels: Optional list of labels, one for each environment.
			
		Returns:
			List of PolyhedralSite objects.
			
		Raises:
			ValueError: If environments are invalid or if minimum vertex count
				is not met for polyhedral sites (must have at least 3 vertices).
		"""
		# Validate inputs
		self._validate_environments(environments)
		self._validate_labels(label, labels, len(environments))
		
		# Validate minimum vertices for PolyhedralSite (3 for a polygon)
		for i, env in enumerate(environments):
			if len(env) < 3:
				raise ValueError(
					f"Environment {i} has {len(env)} vertices, but PolyhedralSite "
					f"requires at least 3 vertices to form a polyhedron."
				)
		
		# Create sites
		sites = []
		for i, env in enumerate(environments):
			# Determine label for this site
			site_label = label if label is not None else (
				labels[i] if labels is not None else None
			)
			
			# Create site
			site = PolyhedralSite(vertex_indices=env, label=site_label)
			
			# Assign vertex coordinates
			self._assign_vertex_coords(site)
			
			sites.append(site)
		
		return sites
	
	def create_dynamic_voronoi_sites(
		self, 
		environments: List[List[int]], 
		label: Optional[str] = None,
		labels: Optional[List[str]] = None
	) -> List[DynamicVoronoiSite]:
		"""Create DynamicVoronoiSite objects from coordination environments.
		
		Args:
			environments: List of environments, where each environment is a list
				of atom indices defining the reference atoms for a dynamic
				Voronoi site.
			label: Optional label to assign to all sites.
			labels: Optional list of labels, one for each environment.
			
		Returns:
			List of DynamicVoronoiSite objects.
			
		Raises:
			ValueError: If environments are invalid.
		"""
		# Validate inputs
		self._validate_environments(environments)
		self._validate_labels(label, labels, len(environments))
		
		# Create sites
		sites = []
		for i, env in enumerate(environments):
			# Determine label for this site
			site_label = label if label is not None else (
				labels[i] if labels is not None else None
			)
			
			# Create site
			site = DynamicVoronoiSite(reference_indices=env, label=site_label)
			
			sites.append(site)
		
		return sites
	
	def _validate_environments(self, environments: Any) -> None:
		"""Validate that environments have the correct format.
		
		Args:
			environments: Environments to validate.
			
		Raises:
			ValueError: If environments have invalid format or reference
				non-existent atoms.
		"""
		# Check that environments is a list
		if not isinstance(environments, list):
			raise ValueError(
				f"Environments must be a list, got {type(environments)}"
			)
		
		# Empty environments list is valid
		if not environments:
			return
		
		# Check that each environment is a list of integers
		for i, env in enumerate(environments):
			if not isinstance(env, list):
				raise ValueError(
					f"Environment {i} must be a list, got {type(env)}"
				)
			
			for j, idx in enumerate(env):
				if not isinstance(idx, int):
					raise ValueError(
						f"Index {j} in environment {i} must be an integer, got {type(idx)}"
					)
				
				# Check that index is in range
				if idx < 0 or idx >= len(self.structure):
					raise ValueError(
						f"Index {idx} in environment {i} is out of range "
						f"(structure has {len(self.structure)} atoms)"
					)
	
	def _validate_labels(
		self, 
		label: Optional[str], 
		labels: Optional[List[str]], 
		num_environments: int
	) -> None:
		"""Validate label options.
		
		Args:
			label: Single label option.
			labels: Multiple labels option.
			num_environments: Number of environments.
			
		Raises:
			ValueError: If both label and labels are provided, or if labels
				length doesn't match the number of environments.
		"""
		# Check that not both label and labels are provided
		if label is not None and labels is not None:
			raise ValueError(
				"Cannot provide both 'label' and 'labels' arguments"
			)
		
		# Check that labels length matches environments length
		if labels is not None and len(labels) != num_environments:
			raise ValueError(
				f"Number of labels ({len(labels)}) does not match "
				f"number of environments ({num_environments})"
			)
	
	def _assign_vertex_coords(self, site: PolyhedralSite) -> None:
		"""Assign vertex coordinates to a PolyhedralSite.
		
		Args:
			site: PolyhedralSite to assign coordinates to.
		"""
		site.assign_vertex_coords(self.structure)