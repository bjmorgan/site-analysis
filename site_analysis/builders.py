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
					 radii=2.0,
					 labels=["octahedral", "tetrahedral"]
				 )
				 .build())
	
	# Using a factory function
	trajectory = create_trajectory_with_spherical_sites(
		structure=structure,
		mobile_species="Li",
		centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
		radii=2.0,
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
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.trajectory import Trajectory
from site_analysis.reference_workflow.reference_based_sites import ReferenceBasedSites
import numpy as np
from typing import Union, Optional, cast, Callable, Sequence, Any


class TrajectoryBuilder:
	"""Builder for creating Trajectory objects for site analysis.
	
	This class provides a step-by-step approach to creating a Trajectory
	object for analysing site occupations in crystal structures.
	
	Structure Alignment and Site Mapping:
	------------------------------------
	The builder supports separate control over structure alignment (finding optimal
	translations) and site mapping (identifying corresponding sites):
	
	- Structure alignment: Use with_structure_alignment() to control whether and how
	structures are aligned before site creation.
	
	- Site mapping: Use with_site_mapping() to specify which species are used to
	identify corresponding sites between structures.
	
	Default Behaviors:
	- Alignment is enabled by default
	- If mapping species are specified but alignment species are not, mapping species
	will be used for alignment
	- If alignment species are specified but mapping species are not, alignment species
	will be used for mapping
	
	Example:
		```python
		# Create a builder
		builder = TrajectoryBuilder()
		
		# Configure it
		builder.with_structure(structure)
			.with_reference_structure(reference_structure)
			.with_mobile_species("Li")
			.with_structure_alignment(align=True, align_species=["O"])  # Align on framework
			.with_site_mapping(mapping_species=["Na"])  # Map using Na atoms
			.with_polyhedral_sites(
				centre_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4
			)
		
		# Build the trajectory
		trajectory = builder.build()
		```
	"""
	
	def __init__(self) -> None:
		"""Initialize a TrajectoryBuilder.
		
		Creates a new builder with all attributes set to their default values.
		"""
		# Call reset() to set all attributes to their default values
		self.reset()
	
	def reset(self) -> 'TrajectoryBuilder':
		"""Reset the builder state to default values.
		
		This method clears all configuration and returns the builder to its
		initial state. It is called automatically during initialization and
		after build(), but can also be called explicitly if needed.
		
		Returns:
			self: For method chaining
		"""
		self._structure: Optional[Structure] = None
		self._reference_structure: Optional[Structure]= None
		self._mobile_species: Optional[str|list[str]] = None
		self._atoms: Optional[list[Atom]] = None
		
		# Alignment options
		self._align = True
		self._align_species: Optional[list[str]] = None
		self._align_metric = 'rmsd'
		self._align_algorithm = 'Nelder-Mead'
		self._align_minimizer_options: Optional[dict[str, Any]] = None
		self._align_tolerance = 1e-4  # Default tolerance value
		
		# Mapping options
		self._mapping_species: Optional[list[str]] = None
		
		# Functions to be called during build() to create sites
		self._site_generators: list[Callable] = []
		
		return self
		
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
	
	def with_structure_alignment(self, 
				align: bool = True, 
				align_species: Optional[Union[str, list[str]]] = None, 
				align_metric: str = 'rmsd',
				align_algorithm: str = 'Nelder-Mead',
				align_minimizer_options: Optional[dict[str, Any]] = None,
				align_tolerance: float = 1e-4) -> 'TrajectoryBuilder':
		"""Set options for aligning reference and target structures.
		
		Structure alignment finds the optimal translation vector to superimpose
		the reference structure onto the target structure, minimizing distances
		between corresponding atoms.
		
		Note:
			Structure alignment is ENABLED by default when using polyhedral or dynamic
			Voronoi sites, even if this method is not explicitly called. To disable
			alignment, call this method with align=False.
		
		All parameters are optional and have sensible defaults:
		
		Args:
			align: Whether to perform structure alignment. Default is True.
			align_species: Species to use for alignment. Can be a string or list of strings.
				Default is None, which means:
				- If mapping species have been set with with_site_mapping(), those species will be used
				- Otherwise, all common species between structures will be used
			align_metric: Metric for alignment. Options are:
				- 'rmsd': Root-mean-square deviation (default)
				- 'max_dist': Maximum distance between any atom pair
			align_algorithm: Algorithm for optimization. Options are:
				- 'Nelder-Mead': Local optimizer, faster but may find local minima (default)
				- 'differential_evolution': Global optimizer, more robust but slower
			align_minimizer_options: Additional options for the minimizer as a dictionary.
				Default is None (use algorithm defaults).
			align_tolerance: Convergence tolerance for alignment optimizer. Default is 1e-4.
				Lower values (e.g., 1e-5) give more precise alignment but may take longer.
				
		Returns:
			self: For method chaining
			
		Examples:
			# Use default alignment (enabled, all species)
			builder.with_reference_structure(reference)
				.with_polyhedral_sites(...)
			
			# Specify alignment species explicitly
			builder.with_structure_alignment(align_species=["O", "Ti"])
				.with_polyhedral_sites(...)
			
			# Disable alignment
			builder.with_structure_alignment(align=False)
				.with_polyhedral_sites(...)
			
			# Use global optimization for challenging alignments
			builder.with_structure_alignment(
					align_algorithm='differential_evolution',
					align_minimizer_options={'popsize': 20}
				)
		"""
		self._align = align
		
		# Convert single species string to a list
		if isinstance(align_species, str):
			self._align_species = [align_species]
		else:
			self._align_species = align_species
			
		self._align_metric = align_metric
		self._align_algorithm = align_algorithm
		self._align_minimizer_options = align_minimizer_options
		self._align_tolerance = align_tolerance
		return self
		
	def with_site_mapping(self, mapping_species: Optional[Union[str, list[str]]]) -> TrajectoryBuilder:
		"""Set the species to use for mapping sites between reference and target structures.
		
		Site mapping identifies corresponding sites between structures even when
		atom counts differ, for example when structures have different numbers
		of mobile ions but the same framework atoms.
		
		If mapping species are specified but alignment species are not:
		- The mapping species will also be used for alignment (unless alignment is disabled)
		
		If mapping species are not specified:
		- The alignment species will be used for mapping
		
		Args:
			mapping_species: Species to use for mapping. Can be a string or list of strings.
				If None, alignment species will be used for mapping.
				
		Returns:
			self: For method chaining
		"""
		# Convert single species string to a list
		if isinstance(mapping_species, str):
			self._mapping_species = [mapping_species]
		else:
			self._mapping_species = mapping_species
		return self
		
	def with_spherical_sites(self, 
					centres: list[list[float]], 
					radii: Union[float, list[float]], 
					labels: Optional[Union[str, list[str]]] = None) -> TrajectoryBuilder:
		"""Define spherical sites.
		
		Note: Sites will be generated when build() is called.
		
		Args:
			centres: list of fractional coordinate centres for spherical sites
			radii: either a single radius (float) to use for all sites, or a list 
				of radii (one per centre)
			labels: either a single label (str) to use for all sites, a list of 
				labels (one per centre), or None
			
		Returns:
			self: For method chaining
		"""
		# Convert single radius to list if needed
		if isinstance(radii, (float, int)):
			radii = [radii] * len(centres)
			
		# Convert single label to list if needed
		if isinstance(labels, str):
			labels = [labels] * len(centres)
			
		# Validate lengths
		if len(centres) != len(radii):
			raise ValueError("Number of centres must match number of radii")
			
		if labels is not None and len(centres) != len(labels):
			raise ValueError("Number of centres must match number of labels")
			
		# Define the site generation function but don't execute it yet
		def create_spherical_sites() -> Sequence[SphericalSite]:
			# Create spherical sites
			sites = []
			for i, (centre, radius) in enumerate(zip(centres, radii)):
				label = labels[i] if labels and i < len(labels) else None
				sites.append(SphericalSite(
					frac_coords=np.array(centre), 
					rcut=radius, 
					label=label
				))
			return sites
			
		# Store the function for later execution
		self._site_generators.append(create_spherical_sites)
		return self
		
	def with_voronoi_sites(self, 
						centres: list[list[float]], 
						labels: Optional[list[str]] = None) -> TrajectoryBuilder:
		"""Define Voronoi sites.
		
		Note: Sites will be generated when build() is called.
		"""
		# Define the site generation function but don't execute it yet
		def create_voronoi_sites() -> Sequence[VoronoiSite]:
			# Create Voronoi sites
			sites = []
			for i, centre in enumerate(centres):
				label = labels[i] if labels and i < len(labels) else None
				sites.append(VoronoiSite(
					frac_coords=np.array(centre), 
					label=label
				))
			return sites
			
		# Store the function for later execution
		self._site_generators.append(create_voronoi_sites)
		return self
		
	def with_polyhedral_sites(self, 
							centre_species: str, 
							vertex_species: Union[str, list[str]], 
							cutoff: float, 
							n_vertices: int, 
							label: Optional[str] = None) -> TrajectoryBuilder:
		"""Define polyhedral sites using the ReferenceBasedSites workflow.
		
		Note: Sites will be generated when build() is called.
		"""
		# Define the site generation function but don't execute it yet
		def create_polyhedral_sites() -> Sequence[PolyhedralSite]:
			if not self._structure or not self._reference_structure:
				raise ValueError("Both structure and reference_structure must be set for polyhedral sites")
			
			# If alignment is enabled but no alignment species are specified, 
			# use mapping species for alignment if available 
			align_species = self._align_species
			if self._align and align_species is None and self._mapping_species is not None:
				align_species = self._mapping_species
				
			# Create ReferenceBasedSites
			rbs = ReferenceBasedSites(
				reference_structure=self._reference_structure,
				target_structure=self._structure,
				align=self._align,
				align_species=align_species,
				align_metric=self._align_metric,
				align_algorithm=self._align_algorithm,
				align_minimizer_options=self._align_minimizer_options,
				align_tolerance=self._align_tolerance  # Pass the tolerance
			)
			
			# Determine mapping species (use alignment species if mapping species not specified)
			target_species = self._mapping_species if self._mapping_species is not None else self._align_species
			
			# Create sites
			sites = rbs.create_polyhedral_sites(
						center_species=centre_species,
						vertex_species=vertex_species,
						cutoff=cutoff,
						n_vertices=n_vertices,
						label=label,
						target_species=target_species
					)
		
			
			# Check if any sites were found
			if not sites:
				raise ValueError(
					f"No polyhedral sites found for centre_species='{centre_species}', "
					f"vertex_species='{vertex_species}', cutoff={cutoff}, n_vertices={n_vertices}. "
					f"Try adjusting these parameters or verify that the specified species exist "
					f"in the structure."
				)
			
			return sites
			
		# Store the function for later execution
		self._site_generators.append(create_polyhedral_sites)
		return self
		
	def with_dynamic_voronoi_sites(self,
		centre_species: str,
		reference_species: Union[str, list[str]],
		cutoff: float,
		n_reference: int,
		label: Optional[str] = None) -> TrajectoryBuilder:
		"""Define dynamic Voronoi sites using the ReferenceBasedSites workflow.
		
		Note: Sites will be generated when build() is called.
		"""
		# Define the site generation function but don't execute it yet
		def create_dynamic_voronoi_sites() -> Sequence[DynamicVoronoiSite]:
			if not self._structure or not self._reference_structure:
				raise ValueError("Both structure and reference_structure must be set for dynamic Voronoi sites")
			
			# If alignment is enabled but no alignment species are specified, 
			# use mapping species for alignment if available 
			align_species = self._align_species
			if self._align and align_species is None and self._mapping_species is not None:
				align_species = self._mapping_species
				
			# Create ReferenceBasedSites
			rbs = ReferenceBasedSites(
				reference_structure=self._reference_structure,
				target_structure=self._structure,
				align=self._align,
				align_species=align_species,
				align_metric=self._align_metric,
				align_algorithm=self._align_algorithm,
				align_minimizer_options=self._align_minimizer_options,
				align_tolerance=self._align_tolerance 
			)
			
			# Determine mapping species (use alignment species if mapping species not specified)
			target_species = self._mapping_species if self._mapping_species is not None else self._align_species
			
			# Create sites
			sites = rbs.create_dynamic_voronoi_sites(
						center_species=centre_species,
						reference_species=reference_species,
						cutoff=cutoff,
						n_reference=n_reference,
						label=label,
						target_species=target_species
					)
			
			# Check if any sites were found
			if not sites:
				raise ValueError(
					f"No dynamic Voronoi sites found for centre_species='{centre_species}', "
					f"reference_species='{reference_species}', cutoff={cutoff}, n_reference={n_reference}. "
					f"Try adjusting these parameters or verify that the specified species exist "
					f"in the structure."
				)
			
			return sites
			
		# Store the function for later execution
		self._site_generators.append(create_dynamic_voronoi_sites)
		return self
		
	def with_existing_sites(self, sites: list) -> TrajectoryBuilder:
		"""Use existing site objects."""
		# Define a simple function that returns the provided sites
		def return_existing_sites() -> list[Site]:
			return sites
			
		# Store the function for later execution
		self._site_generators.append(return_existing_sites)
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
		
		This method validates all required parameters and generates sites
		using the previously configured site generator.
		
		Returns:
			Trajectory: The constructed Trajectory object
			
		Raises:
			ValueError: If required parameters are missing
		"""
		# Validate basic requirements
		if not self._structure:
			raise ValueError("Structure must be set")
		if not self._site_generators:
			raise ValueError("Site type must be defined using one of the with_*_sites methods")
			
		# Reset the site index counter
		Site.reset_index()
			
		# Generate all sites
		sites: list[Site] = []
		site_type = None
		
		for generator in self._site_generators:
			generated_sites = generator()
			
			# Verify site type consistency
			if generated_sites:
				current_type = type(generated_sites[0])
				if site_type is None:
					site_type = current_type
				elif site_type != current_type:
					raise TypeError(f"Cannot mix site types: {site_type.__name__} and {current_type.__name__}")
			
			sites.extend(generated_sites)
		
		# Create atoms if not already set
		if not self._atoms:
			if not self._mobile_species:
				raise ValueError("Mobile species must be set")
			self._atoms = atoms_from_structure(self._structure, self._mobile_species)
			
		# Create trajectory
		trajectory = Trajectory(sites=sites, atoms=self._atoms)
		
		# Reset the builder state for future use
		self.reset()
		
		return trajectory


def create_trajectory_with_spherical_sites(
	structure, 
	mobile_species: Union[str, list[str]], 
	centres: list[list[float]], 
	radii: Union[float, list[float]], 
	labels: Optional[Union[str, list[str]]] = None
) -> Trajectory:
	"""Create a Trajectory with spherical sites.
	
	Args:
		structure: Structure containing the atoms to analyse
		mobile_species: Species string or list of species strings for mobile atoms
		centres: list of fractional coordinate centres for spherical sites
		radii: either a single radius (float) to use for all sites, or a list 
			of radii (one per centre) in Angstroms
		labels: Optional single label (str) to use for all sites, or list of 
			labels (one per centre)
		
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
	align_species: Optional[Union[str, list[str]]] = None,
	align_metric: str = 'rmsd',
	align_algorithm: str = 'Nelder-Mead',
	align_minimizer_options: Optional[dict[str, Any]] = None,
	mapping_species: Optional[Union[str, list[str]]] = None,
	align_tolerance: float = 1e-4 
) -> Trajectory:
	"""Create a Trajectory with polyhedral sites."""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_reference_structure(reference_structure)
			.with_mobile_species(mobile_species)
			.with_structure_alignment(
				align=align,
				align_species=align_species,
				align_metric=align_metric,
				align_algorithm=align_algorithm,
				align_minimizer_options=align_minimizer_options,
				align_tolerance=align_tolerance
			)
			.with_site_mapping(mapping_species)
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
	align_species: Optional[Union[str, list[str]]] = None,
	align_metric: str = 'rmsd',
	align_algorithm: str = 'Nelder-Mead',
	align_minimizer_options: Optional[dict[str, Any]] = None,
	mapping_species: Optional[Union[str, list[str]]] = None,
	align_tolerance: float = 1e-4 
) -> Trajectory:
	"""Create a Trajectory with dynamic Voronoi sites."""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_reference_structure(reference_structure)
			.with_mobile_species(mobile_species)
			.with_structure_alignment(
				align=align,
				align_species=align_species,
				align_metric=align_metric,
				align_algorithm=align_algorithm,
				align_minimizer_options=align_minimizer_options,
				align_tolerance=align_tolerance
			)
			.with_site_mapping(mapping_species)
			.with_dynamic_voronoi_sites(centre_species, reference_species, cutoff, n_reference, label)
			.build())
