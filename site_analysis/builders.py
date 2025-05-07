"""Builder pattern implementation for site_analysis.

This module provides a fluent builder interface for creating site analysis
objects, making it easier to set up and run site analysis workflows.

Examples:
	Creating a simple trajectory with spherical sites:
	
	```python
	# Using the builder directly
	trajectory = (TrajectoryBuilder()
				 .with_structure(structure)
				 .with_mobile_species("Li")
				 .with_spherical_sites(
					 centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
					 radii=[2.0, 2.0],
					 labels=["octahedral", "tetrahedral"]
				 )
				 .build())
	
	# Using a factory function
	trajectory = create_trajectory_with_spherical_sites(
		structure=structure,
		mobile_species="Li",
		centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
		radii=[2.0, 2.0],
		labels=["octahedral", "tetrahedral"]
	)
	```
	
	Creating a trajectory with polyhedral sites:
	
	```python
	trajectory = create_trajectory_with_polyhedral_sites(
		structure=target_structure,
		reference_structure=reference_structure,
		mobile_species="Li",
		centre_species="O",
		vertex_species=["Li", "Na"], 
		cutoff=3.0,
		n_vertices=4,
		label="tetrahedral"
	)
	```
"""

from __future__ import annotations

from pymatgen.core import Structure
from site_analysis.atom import Atom, atoms_from_structure
from site_analysis.site import Site
from site_analysis.spherical_site import SphericalSite
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.trajectory import Trajectory
from site_analysis.reference_workflow.reference_based_sites import ReferenceBasedSites
import numpy as np
from typing import Union, Optional, cast


class TrajectoryBuilder:
	"""Builder for creating Trajectory objects for site analysis.
	
	This class provides a step-by-step approach to creating a Trajectory
	object for analysing site occupations in crystal structures.
	
	Example:
		```python
		# Create a builder
		builder = TrajectoryBuilder()
		
		# Configure it
		builder.with_structure(structure)
			   .with_mobile_species("Li")
			   .with_spherical_sites(
				   centres=[[0.5, 0.5, 0.5]],
				   radii=[2.0]
			   )
		
		# Build the trajectory
		trajectory = builder.build()
		
		# Analyse a trajectory
		trajectory.trajectory_from_structures(structures)
		```
	"""
	
	def __init__(self) -> None:
		"""Initialize a TrajectoryBuilder."""
		self._structure: Optional[Structure] = None
		self._reference_structure: Optional[Structure] = None
		self._mobile_species: Optional[Union[str, list[str]]] = None
		self._sites: Optional[list[Site]] = None
		self._atoms: Optional[list[Atom]] = None
		# Alignment options
		self._align: bool = True
		self._align_species: Optional[list[str]] = None
		self._align_metric: str = 'rmsd'
		
	def with_structure(self, structure) -> TrajectoryBuilder:
		"""Set the structure to analyse.
		
		Args:
			structure: A pymatgen Structure object
			
		Returns:
			self: For method chaining
		"""
		self._structure = structure
		return self
		
	def with_reference_structure(self, reference_structure) -> TrajectoryBuilder:
		"""Set the reference structure for complex site types.
		
		Args:
			reference_structure: A pymatgen Structure object representing
				the ideal reference structure
				
		Returns:
			self: For method chaining
		"""
		self._reference_structure = reference_structure
		return self
		
	def with_mobile_species(self, species: Union[str, list[str]]) -> TrajectoryBuilder:
		"""Set the mobile species to track.
		
		Args:
			species: Species string or list of species strings for mobile atoms
				
		Returns:
			self: For method chaining
		"""
		self._mobile_species = species
		return self
	
	def with_alignment_options(self, 
							 align: bool = True, 
							 align_species: Optional[list[str]] = None, 
							 align_metric: str = 'rmsd') -> TrajectoryBuilder:
		"""Set options for aligning reference and target structures.
		
		Args:
			align: Whether to perform structure alignment. Default is True.
			align_species: Species to use for alignment. Default is all species.
			align_metric: Metric for alignment ('rmsd', 'max_dist'). 
				Default is 'rmsd'.
				
		Returns:
			self: For method chaining
		"""
		self._align = align
		self._align_species = align_species
		self._align_metric = align_metric
		return self
		
	def with_spherical_sites(self, 
						   centres: list[list[float]], 
						   radii: list[float], 
						   labels: Optional[list[str]] = None) -> TrajectoryBuilder:
		"""Define spherical sites.
		
		Args:
			centres: list of fractional coordinate centres for spherical sites
			radii: list of radii for spherical sites (in Angstroms)
			labels: Optional list of labels for sites
				
		Returns:
			self: For method chaining
			
		Raises:
			ValueError: If centres and radii have different lengths
		"""
		if len(centres) != len(radii):
			raise ValueError("Number of centres must match number of radii")
			
		# Create spherical sites
		self._sites = []
		for i, (centre, radius) in enumerate(zip(centres, radii)):
			label = labels[i] if labels and i < len(labels) else None
			self._sites.append(SphericalSite(
				frac_coords=np.array(centre), 
				rcut=radius, 
				label=label
			))
		return self
		
	def with_voronoi_sites(self, 
						centres: list[list[float]], 
						labels: Optional[list[str]] = None) -> TrajectoryBuilder:
		"""Define Voronoi sites.
		
		Args:
			centres: list of fractional coordinate centres for Voronoi sites
			labels: Optional list of labels for sites
				
		Returns:
			self: For method chaining
		"""
		# Create Voronoi sites
		self._sites = []
		for i, centre in enumerate(centres):
			label = labels[i] if labels and i < len(labels) else None
			self._sites.append(VoronoiSite(
				frac_coords=np.array(centre), 
				label=label
			))
		return self
		
	def with_polyhedral_sites(self, 
							centre_species: str, 
							vertex_species: Union[str, list[str]], 
							cutoff: float, 
							n_vertices: int, 
							label: Optional[str] = None) -> TrajectoryBuilder:
		"""Define polyhedral sites using the ReferenceBasedSites workflow.
		
		Args:
			centre_species: Species at the centre of coordination environments
			vertex_species: Species at vertices of coordination environments
			cutoff: Cutoff distance for coordination environment
			n_vertices: Number of vertices per environment
			label: Label to apply to all created sites
				
		Returns:
			self: For method chaining
			
		Raises:
			ValueError: If reference_structure or structure is not set
		"""
		if not self._structure or not self._reference_structure:
			raise ValueError("Both structure and reference_structure must be set")
			
		# Create ReferenceBasedSites
		rbs = ReferenceBasedSites(
			reference_structure=self._reference_structure,
			target_structure=self._structure,
			align=self._align,
			align_species=self._align_species,
			align_metric=self._align_metric
		)
		
		# Create sites
		self._sites = cast(list[Site], 
			rbs.create_polyhedral_sites(
				center_species=centre_species,
				vertex_species=vertex_species,
				cutoff=cutoff,
				n_vertices=n_vertices,
				label=label
			)
		)
		
		return self
		
	def with_dynamic_voronoi_sites(self, 
								 centre_species: str, 
								 reference_species: Union[str, list[str]], 
								 cutoff: float, 
								 n_reference: int, 
								 label: Optional[str] = None) -> TrajectoryBuilder:
		"""Define dynamic Voronoi sites using the ReferenceBasedSites workflow.
		
		Args:
			centre_species: Species at the centre of coordination environments
			reference_species: Species of reference atoms for dynamic site centres
			cutoff: Cutoff distance for finding reference atoms
			n_reference: Number of reference atoms per site
			label: Label to apply to all created sites
				
		Returns:
			self: For method chaining
			
		Raises:
			ValueError: If reference_structure or structure is not set
		"""
		if not self._structure or not self._reference_structure:
			raise ValueError("Both structure and reference_structure must be set")
			
		# Create ReferenceBasedSites
		rbs = ReferenceBasedSites(
			reference_structure=self._reference_structure,
			target_structure=self._structure,
			align=self._align,
			align_species=self._align_species,
			align_metric=self._align_metric
		)
		
		# Create sites
		self._sites = cast(list[Site],
			rbs.create_dynamic_voronoi_sites(
				center_species=centre_species,
				reference_species=reference_species,
				cutoff=cutoff,
				n_reference=n_reference,
				label=label
			)
		)
		
		return self
		
	def with_existing_sites(self, sites: list) -> TrajectoryBuilder:
		"""Use existing site objects.
		
		Args:
			sites: list of site objects
				
		Returns:
			self: For method chaining
		"""
		self._sites = sites
		return self
		
	def with_existing_atoms(self, atoms: list) -> TrajectoryBuilder:
		"""Use existing atom objects.
		
		Args:
			atoms: list of atom objects
				
		Returns:
			self: For method chaining
		"""
		self._atoms = atoms
		return self
		
	def build(self) -> Trajectory:
		"""Build and return the Trajectory object.
		
		Returns:
			Trajectory: The constructed Trajectory object
			
		Raises:
			ValueError: If structure, sites, or mobile_species is not set
		"""
		# Validate state
		if not self._structure:
			raise ValueError("Structure must be set")
		if not self._sites:
			raise ValueError("Sites must be defined")
			
		# Create atoms if not already set
		if not self._atoms:
			if not self._mobile_species:
				raise ValueError("Mobile species must be set")
			self._atoms = atoms_from_structure(self._structure, self._mobile_species)
			
		# Create trajectory
		trajectory = Trajectory(sites=self._sites, atoms=self._atoms)
		
		# Pre-analyse the structure
		trajectory.analyse_structure(self._structure)
		
		return trajectory


def create_trajectory_with_spherical_sites(
	structure, 
	mobile_species: Union[str, list[str]], 
	centres: list[list[float]], 
	radii: list[float], 
	labels: Optional[list[str]] = None
) -> Trajectory:
	"""Create a Trajectory with spherical sites.
	
	Args:
		structure: Structure containing the atoms to analyse
		mobile_species: Species string or list of species strings for mobile atoms
		centres: list of fractional coordinate centres for spherical sites
		radii: list of radii for spherical sites (in Angstroms)
		labels: Optional list of labels for sites
		
	Returns:
		Trajectory: The constructed Trajectory object
	"""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_mobile_species(mobile_species)
			.with_spherical_sites(centres, radii, labels)
			.build())


def create_trajectory_with_voronoi_sites(
	structure, 
	mobile_species: Union[str, list[str]], 
	centres: list[list[float]], 
	labels: Optional[list[str]] = None
) -> Trajectory:
	"""Create a Trajectory with Voronoi sites.
	
	Args:
		structure: Structure containing the atoms to analyse
		mobile_species: Species string or list of species strings for mobile atoms
		centres: list of fractional coordinate centres for Voronoi sites
		labels: Optional list of labels for sites
		
	Returns:
		Trajectory: The constructed Trajectory object
	"""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_mobile_species(mobile_species)
			.with_voronoi_sites(centres, labels)
			.build())


def create_trajectory_with_polyhedral_sites(
	structure, 
	reference_structure, 
	mobile_species: Union[str, list[str]], 
	centre_species: str, 
	vertex_species: Union[str, list[str]], 
	cutoff: float, 
	n_vertices: int, 
	label: Optional[str] = None,
	align: bool = True,
	align_species: Optional[list[str]] = None,
	align_metric: str = 'rmsd'
) -> Trajectory:
	"""Create a Trajectory with polyhedral sites.
	
	Args:
		structure: Structure to analyse
		reference_structure: Reference structure for site definition
		mobile_species: Species string or list of species strings for mobile atoms
		centre_species: Species at the centre of coordination environments
		vertex_species: Species at vertices of coordination environments
		cutoff: Cutoff distance for coordination environment
		n_vertices: Number of vertices per environment
		label: Label to apply to all created sites
		align: Whether to perform structure alignment. Default is True.
		align_species: Species to use for alignment. Default is all species.
		align_metric: Metric for alignment ('rmsd', 'max_dist'). 
			Default is 'rmsd'.
		
	Returns:
		Trajectory: The constructed Trajectory object
	"""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_reference_structure(reference_structure)
			.with_mobile_species(mobile_species)
			.with_alignment_options(align, align_species, align_metric)
			.with_polyhedral_sites(centre_species, vertex_species, cutoff, n_vertices, label)
			.build())


def create_trajectory_with_dynamic_voronoi_sites(
	structure, 
	reference_structure, 
	mobile_species: Union[str, list[str]], 
	centre_species: str, 
	reference_species: Union[str, list[str]], 
	cutoff: float, 
	n_reference: int, 
	label: Optional[str] = None,
	align: bool = True,
	align_species: Optional[list[str]] = None,
	align_metric: str = 'rmsd'
) -> Trajectory:
	"""Create a Trajectory with dynamic Voronoi sites.
	
	Args:
		structure: Structure to analyse
		reference_structure: Reference structure for site definition
		mobile_species: Species string or list of species strings for mobile atoms
		centre_species: Species at the centre of coordination environments
		reference_species: Species of reference atoms for dynamic site centres
		cutoff: Cutoff distance for finding reference atoms
		n_reference: Number of reference atoms per site
		label: Label to apply to all created sites
		align: Whether to perform structure alignment. Default is True.
		align_species: Species to use for alignment. Default is all species.
		align_metric: Metric for alignment ('rmsd', 'max_dist'). 
			Default is 'rmsd'.
		
	Returns:
		Trajectory: The constructed Trajectory object
	"""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_reference_structure(reference_structure)
			.with_mobile_species(mobile_species)
			.with_alignment_options(align, align_species, align_metric)
			.with_dynamic_voronoi_sites(centre_species, reference_species, cutoff, n_reference, label)
			.build())