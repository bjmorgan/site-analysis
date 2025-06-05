"""Reference-based workflow for defining sites in crystal structures.

This module provides the ReferenceBasedSites class, which is the main orchestrator
for defining crystallographic sites in a target structure based on coordination
environments identified in a reference structure. This approach is particularly
useful for:

1. Analyzing structures with distortions relative to an ideal reference
2. Tracking specific site types across different structures or simulation frames
3. Creating consistent site definitions across diverse structures

The ReferenceBasedSites class integrates several components to accomplish this workflow:
- StructureAligner: Aligns the structures to find the optimal translation vector
- CoordinationEnvironmentFinder: Identifies coordination environments in the reference
- IndexMapper: Maps atom indices between reference and target structures
- SiteFactory: Creates appropriate site objects in the target structure

This approach lets users define sites based on ideal coordination environments in
a reference structure, then create corresponding sites in real or distorted structures
where those same environments might be harder to identify directly.
"""

import numpy as np
from typing import Any, Optional

from pymatgen.core import Structure

from site_analysis.reference_workflow.structure_aligner import StructureAligner
from site_analysis.reference_workflow.coord_finder import CoordinationEnvironmentFinder
from site_analysis.reference_workflow.index_mapper import IndexMapper
from site_analysis.reference_workflow.site_factory import SiteFactory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite


class ReferenceBasedSites:
	"""Main orchestrator for defining sites using a reference structure approach.
	
	This class ties together all components needed to define sites in crystal structures
	using a reference structure as a template:
	
	1. StructureAligner - to align the target structure to the reference
	2. CoordinationEnvironmentFinder - to find coordination environments in the reference
	3. IndexMapper - to map environments from reference to target structure
	4. SiteFactory - to create appropriate site objects
	
	Attributes:
		reference_structure: Reference structure defining ideal coordination environments
		target_structure: Target structure where sites will be created
		aligned_structure: Target structure aligned to the reference (same as target if align=False)
		translation_vector: Translation vector used for alignment (None if align=False)
		alignment_metrics: Metrics describing quality of structure alignment (None if align=False)
	"""
	
	def __init__(self, 
			reference_structure: Structure, 
			target_structure: Structure, 
			align: bool = True, 
			align_species: list[str] | None = None, 
			align_metric: str = 'rmsd',
			align_algorithm: str = 'Nelder-Mead',
			align_minimizer_options: dict[str, Any] | None = None,
			align_tolerance: float = 1e-4) -> None:
		"""Initialise ReferenceBasedSites with reference and target structures.
		
		Args:
			reference_structure: Reference structure defining ideal coordination environments
			target_structure: Target structure where sites will be created
			align: Whether to perform structure alignment. Default is True.
			align_species: Species to use for alignment. Default is all species.
			align_metric: Metric for alignment ('rmsd', 'max_dist'). Default is 'rmsd'.
			align_algorithm: Algorithm for optimization ('Nelder-Mead', 'differential_evolution'). 
							Default is 'Nelder-Mead'.
			align_minimizer_options: Additional options for the minimizer. Default is None.
			align_tolerance: Convergence tolerance for alignment optimizer. Default is 1e-4.
		"""
		self.reference_structure = reference_structure
		self.target_structure = target_structure
		
		# Initialise alignment attributes
		self.aligned_structure: Structure | None = None
		self.translation_vector: np.ndarray | None = None
		self.alignment_metrics: dict[str, float] | None = None
		
		# Perform alignment if requested
		if align:
			self._align_structures(
				align_species, 
				align_metric, 
				align_algorithm, 
				align_minimizer_options,
				align_tolerance
			)
		
		# These will be initialised on first use
		self._coord_finder: CoordinationEnvironmentFinder | None = None
		self._index_mapper: IndexMapper | None = None
		self._site_factory: SiteFactory | None = None
		
	def create_polyhedral_sites(self, 
							center_species: str, 
							vertex_species: str | list[str], 
							cutoff: float, 
							n_vertices: int,
							label: str | None = None, 
							labels: list[str] | None = None, 
							target_species: str | list[str] | None = None,
							use_reference_centers: bool = True) -> list[PolyhedralSite]:
		"""Create PolyhedralSite objects based on coordination environments in the reference structure.
		
		Args:
			center_species: Species at the center of coordination environments
			vertex_species: Species at vertices of coordination environments
			cutoff: Cutoff distance for coordination environment (required)
			n_vertices: Number of vertices per environment (required)
			label: Label to apply to all created sites. Default is None.
			labels: List of labels for each site. Default is None.
			target_species: Species to map to in the target structure. Default is None.
			use_reference_centers: Whether to use reference centers for 
				PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
				Default is True.
			
		Returns:
			List of PolyhedralSite objects
			
		Raises:
			ValueError: If coordination environments cannot be found or mapped,
				or if both label and labels are provided.
		"""
		# Find coordination environments in reference structure
		ref_environments = self._find_coordination_environments(
			center_species=center_species,
			coordination_species=vertex_species,
			cutoff=cutoff,
			n_coord=n_vertices
		)
		
		# Check we do not have repeat periodic images in the coordination environments
		self._validate_unique_environments(ref_environments)
		
		# Calculate reference centers if requested
		if use_reference_centers:
			center_indices = list(ref_environments.keys())
			reference_centers = self._calculate_reference_centers_from_indices(center_indices)
		else:
			reference_centers = None
			
		# Map environments to target structure
		mapped_environments = self._map_environments(
			list(ref_environments.values()),
			target_species
		)
		
		# Create site factory if not already initialised
		if self._site_factory is None:
			self._site_factory = SiteFactory(self.target_structure)
		
		# Create polyhedral sites
		# At this point we know self._site_factory is not None
		assert self._site_factory is not None
		sites = self._site_factory.create_polyhedral_sites(
			mapped_environments,
			reference_centers=reference_centers,
			label=label,
			labels=labels
		)
		
		return sites
	
	def create_dynamic_voronoi_sites(self, 
								center_species: str, 
								reference_species: str | list[str], 
								cutoff: float, 
								n_reference: int,
								label: str | None = None, 
								labels: list[str] | None = None, 
								target_species: str | list[str] | None = None,
								use_reference_centers: bool = True) -> list[DynamicVoronoiSite]:
		"""Create DynamicVoronoiSite objects based on coordination environments in the reference structure.
		
		Args:
			center_species: Species at the center of coordination environments
			reference_species: Species of reference atoms used to define the dynamic site centers
			cutoff: Cutoff distance for finding reference atoms (required)
			n_reference: Number of reference atoms per site (required)
			label: Label to apply to all created sites. Default is None.
			labels: List of labels for each site. Default is None.
			target_species: Species to map to in the target structure. Default is None.
			use_reference_centers: Whether to use reference centers for 
				PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
				Default is True.
			
		Returns:
			List of DynamicVoronoiSite objects
			
		Raises:
			ValueError: If coordination environments cannot be found or mapped,
				or if both label and labels are provided.
	"""
		# Find coordination environments in reference structure
		# Note: CoordinationEnvironmentFinder uses vertex_species terminology,
		# but conceptually these are reference atoms for dynamic Voronoi sites
		ref_environments = self._find_coordination_environments(
			center_species, reference_species, cutoff, n_reference
		)
		
		# Check we do not have repeat periodic images in the coordination environments
		self._validate_unique_environments(ref_environments)
		
		# Calculate reference centers if requested
		if use_reference_centers:
			center_indices = list(ref_environments.keys())
			reference_centers = self._calculate_reference_centers_from_indices(center_indices)
		else:
			reference_centers = None
			
		# Map environments to target structure
		mapped_environments = self._map_environments(
			list(ref_environments.values()),
			target_species
		)
		
		# Create site factory if not already initialised
		if self._site_factory is None:
			self._site_factory = SiteFactory(self.target_structure)
		
		# Create dynamic Voronoi sites
		# At this point we know self._site_factory is not None
		assert self._site_factory is not None
		sites = self._site_factory.create_dynamic_voronoi_sites(
			mapped_environments,
			reference_centers=reference_centers,
			label=label,
			labels=labels
		)
		
		return sites
	
	def _align_structures(self, 
						align_species: list[str] | None = None, 
						align_metric: str = 'rmsd',
						align_algorithm: str = 'Nelder-Mead',
						align_minimizer_options: dict[str, Any] | None = None,
						align_tolerance: float = 1e-4) -> None:
		"""Align target structure to reference structure.
		
		Args:
			align_species: Species to use for alignment. Default is all species.
			align_metric: Metric for alignment ('rmsd', 'max_dist'). Default is 'rmsd'.
			align_algorithm: Algorithm for optimization ('Nelder-Mead', 'differential_evolution'). 
						Default is 'Nelder-Mead'.
			align_minimizer_options: Additional options for the minimizer. Default is None.
			align_tolerance: Convergence tolerance for alignment optimizer. Default is 1e-4.
			
		Raises:
			ValueError: If alignment fails.
		"""
		try:
			# Create a structure aligner
			aligner = StructureAligner()
			
			# Align structures
			aligned_structure, translation_vector, metrics = aligner.align(
				self.reference_structure, 
				self.target_structure, 
				species=align_species, 
				metric=align_metric,
				tolerance=align_tolerance,
				algorithm=align_algorithm,
				minimizer_options=align_minimizer_options,
			)
			
			# Update attributes
			self.aligned_structure = aligned_structure
			self.translation_vector = translation_vector
			self.alignment_metrics = metrics
			
		except Exception as e:
			# Re-raise with more context
			raise ValueError(f"Structure alignment failed: {str(e)}") from e
	
	def _find_coordination_environments(self, 
									  center_species: str, 
									  coordination_species: str | list[str], 
									  cutoff: float, 
									  n_coord: int) -> dict[int, list[int]]:
		"""Find coordination environments in the reference structure.
		
		Args:
			center_species: Species at the center of coordination environments
			coordination_species: Coordination atom species
			cutoff: Cutoff distance for coordination environment
			n_coord: Number of coordination atoms per environment
			
		Returns:
			Dictionary mapping center atom indices to lists of coordinating atom indices.
			Keys are indices of center atoms, values are lists of coordinating atom indices.
			
		Raises:
			ValueError: If coordination environments cannot be found.
		"""
		try:
			# Create coordination environment finder if not already initialised
			if self._coord_finder is None:
				self._coord_finder = CoordinationEnvironmentFinder(self.reference_structure)
			
			# Find coordination environments
			# At this point we know self._coord_finder is not None
			assert self._coord_finder is not None
			environments_dict = self._coord_finder.find_environments(
				center_species=center_species,
				coordination_species=coordination_species,
				n_coord=n_coord,
				cutoff=cutoff
			)
			
			return environments_dict
			
		except Exception as e:
			# Re-raise with more context
			raise ValueError(
				f"Failed to find coordination environments for {center_species} centers "
				f"and {coordination_species} coordinating atoms: {str(e)}"
			) from e
	
	def _map_environments(self, 
		ref_environments: list[list[int]], 
		target_species: str | list[str] | None = None) -> list[list[int]]:
		"""Map coordination environments from reference to target structure.
		
		Args:
			ref_environments: List of environments from reference structure
			target_species: Species to map to in the target structure. Default is None.
			
		Returns:
			List of mapped environments for the target structure
			
		Raises:
			ValueError: If environments cannot be mapped between structures.
		"""
		# If no environments were found, return an empty list immediately
		if not ref_environments:
			return []
		
		try:
			# Create index mapper if not already initialised
			if self._index_mapper is None:
				self._index_mapper = IndexMapper()
			
			# Use aligned reference if available, otherwise use original reference
			reference_to_use = (
				self.aligned_structure  # When alignment was performed
				if self.aligned_structure is not None 
				else self.reference_structure  # When no alignment was performed
			)
			
			# Map environments
			mapped_environments = self._index_mapper.map_coordinating_atoms(
				reference_to_use,      # Aligned reference if available
				self.target_structure, # Always use original target
				ref_environments,
				target_species=target_species
			)
			
			return mapped_environments
			
		except Exception as e:
			# Re-raise with more context
			species_str = f" for {target_species} species" if target_species else ""
			raise ValueError(
				f"Failed to map coordination environments{species_str}: {str(e)}"
			) from e
			
	def _initialise_site_factory(self):
		"""Initialise the site factory if not already done.
		
		Note: This method exists primarily for testing purposes.
		"""
		if self._site_factory is None:
			self._site_factory = SiteFactory(self.target_structure)
		return self._site_factory
	
	def _validate_unique_environments(self, environments: dict[int, list[int]]) -> None:
		"""Validate that each environment contains unique atom indices.
		
		Args:
			environments: Dict of environments, where keys are center atom indices 
				and values are lists of coordinating atom indices.
			
		Raises:
			ValueError: If any environment contains duplicate atom indices.
		"""
		for center_idx, env in environments.items():
			if len(env) != len(set(env)):
				# Find the duplicates
				counts: dict[int, int] = {}
				for idx in env:
					counts[idx] = counts.get(idx, 0) + 1
				duplicates = [idx for idx, count in counts.items() if count > 1]
				
				raise ValueError(
					f"Environment for center atom {center_idx} contains duplicate atom indices {duplicates}. "
					f"This typically occurs in small unit cells where the same atom "
					f"appears as a neighbor in multiple periodic images. "
					f"Please use a larger supercell for your analysis."
				)
	def _calculate_reference_centers_from_indices(self, center_indices: list[int]) -> list[np.ndarray]:
		"""Calculate reference centres from center atom indices."""
		# Use aligned reference structure if available, otherwise original
		structure_to_use = self.aligned_structure if self.aligned_structure else self.reference_structure
		
		reference_centers = []
		for center_idx in center_indices:
			reference_center = structure_to_use[center_idx].frac_coords.copy()
			reference_centers.append(reference_center)
		
		return reference_centers