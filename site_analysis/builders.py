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
from typing import cast, Callable, Sequence, Any


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
		self._structure: Structure | None = None
		self._reference_structure: Structure | None= None
		self._mobile_species: str | list[str] | None = None
		self._atoms: list[Atom] | None = None
		
		# Alignment options
		self._align = True
		self._align_species: list[str] | None = None
		self._align_metric = 'rmsd'
		self._align_algorithm = 'Nelder-Mead'
		self._align_minimizer_options: dict[str, Any] | None = None
		self._align_tolerance = 1e-4  # Default tolerance value
		
		# Mapping options
		self._mapping_species: list[str] | None = None
		
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
		
	def with_mobile_species(self, species: str | list[str]) -> TrajectoryBuilder:
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
				align_species: str | list[str] | None = None, 
				align_metric: str = 'rmsd',
				align_algorithm: str = 'Nelder-Mead',
				align_minimizer_options: dict[str, Any] | None = None,
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
		
	def with_site_mapping(self, mapping_species: str | list[str] | None) -> TrajectoryBuilder:
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
					radii: float | list[float], 
					labels: str | list[str] | None = None) -> TrajectoryBuilder:
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
						labels: list[str] | None = None) -> TrajectoryBuilder:
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
						vertex_species: str | list[str], 
						cutoff: float, 
						n_vertices: int, 
						label: str | None = None,
						use_reference_centers: bool = True) -> TrajectoryBuilder:
		"""Define polyhedral sites using the ReferenceBasedSites workflow.
		
		Creates polyhedral sites by identifying coordination environments in the reference 
		structure and mapping them to corresponding sites in the target structure. Each site 
		is defined by a polyhedron formed by vertex atoms around a central atom.
		
		The workflow involves:
		1. Finding coordination environments in the reference structure
		2. Mapping these environments to the target structure  
		3. Creating PolyhedralSite objects with proper periodic boundary handling
		
		Note:
			Sites will be generated when build() is called, not immediately.
			Requires both structure and reference_structure to be set.
		
		Args:
			centre_species: Atomic species at the centre of coordination environments.
			vertex_species: Atomic species at vertices of coordination polyhedra. 
				Can be a single species string or list of species strings.
			cutoff: Maximum distance (in Ångströms) for vertex atoms to be considered 
				part of the coordination environment.
			n_vertices: Number of vertex atoms required for each polyhedral site.
			label: Optional label to assign to all created sites. Default is None.
			use_reference_centers: Controls periodic boundary condition handling for 
				reference-based sites (polyhedral and dynamic Voronoi sites). Default is True.
				
				- True (recommended): Reference-based PBC correction. Defines a reference 
				center for each site and unwraps vertex coordinates relative to this center.
				Correctly handles sites that naturally span >50% of unit cell dimensions,
				even in small simulation cells.
				
				- False (advanced usage): Spread-based PBC correction. If vertex coordinates 
				span >50% of the unit cell in any dimension, assumes this indicates PBC 
				wrapping and shifts coordinates accordingly. 
				
				WARNING: Gives incorrect results when sites legitimately span >50% of 
				the unit cell (e.g., octahedral sites in a 2×2×2 FCC supercell). May 
				offer performance benefits for some setups.
				
				Only use False after verifying it works correctly for your structures.
				
		Returns:
			self: For method chaining
			
		Raises:
			ValueError: At build() time if no coordination environments are found,
				or if required structures are not set.
				
		Examples:
			# Tetrahedral sites around Li atoms
			builder.with_polyhedral_sites(
				centre_species="Li",
				vertex_species="O", 
				cutoff=2.5,
				n_vertices=4,
				label="tetrahedral"
			)
			
			# Octahedral sites with mixed vertex species
			builder.with_polyhedral_sites(
				centre_species="Ti",
				vertex_species=["O", "F"],
				cutoff=2.8,
				n_vertices=6,
				label="octahedral"
			)
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
						target_species=target_species,
						use_reference_centers=use_reference_centers
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
	reference_species: str | list[str],
	cutoff: float,
	n_reference: int,
	label: str | None = None,
	use_reference_centers: bool = True) -> TrajectoryBuilder:
		"""Define dynamic Voronoi sites using the ReferenceBasedSites workflow.
		
		Creates dynamic Voronoi sites where the site centres are dynamically calculated
		from the positions of reference atoms. Unlike fixed Voronoi sites, these adapt
		to structural changes as the reference atoms move.
		
		The workflow involves:
		1. Finding coordination environments in the reference structure
		2. Mapping these environments to the target structure
		3. Creating DynamicVoronoiSite objects that calculate centres from reference atoms
		
		Note:
			Sites will be generated when build() is called, not immediately.
			Requires both structure and reference_structure to be set.
		
		Args:
			centre_species: Atomic species at centres where sites will be located.
			reference_species: Atomic species used as reference atoms to define 
				dynamic site centres. Can be a single species string or list of species strings.
			cutoff: Maximum distance (in Ångströms) for reference atoms to be considered 
				part of the coordination environment.
			n_reference: Number of reference atoms required for each dynamic site.
			label: Optional label to assign to all created sites. Default is None.
			use_reference_centers: Controls periodic boundary condition handling for 
				reference-based sites (polyhedral and dynamic Voronoi sites). Default is True.
				
				- True (recommended): Reference-based PBC correction. Defines a reference 
				center for each site and unwraps vertex coordinates relative to this center.
				Correctly handles sites that naturally span >50% of unit cell dimensions,
				even in small simulation cells.
				
				- False (advanced usage): Spread-based PBC correction. If vertex coordinates 
				span >50% of the unit cell in any dimension, assumes this indicates PBC 
				wrapping and shifts coordinates accordingly. 
				
				WARNING: Gives incorrect results when sites legitimately span >50% of 
				the unit cell (e.g., octahedral sites in a 2×2×2 FCC supercell). May 
				offer performance benefits for some setups.
				
				Only use False after verifying it works correctly for your structures.
				
		Returns:
			self: For method chaining
			
		Raises:
			ValueError: At build() time if no coordination environments are found,
				or if required structures are not set.
				
		Examples:
			# Dynamic sites at Li positions defined by neighbouring O atoms
			builder.with_dynamic_voronoi_sites(
				centre_species="Li",
				reference_species="O",
				cutoff=3.0,
				n_reference=4,
				label="tetrahedral_dynamic"
			)
			
			# Sites with mixed reference species
			builder.with_dynamic_voronoi_sites(
				centre_species="Na",
				reference_species=["O", "F"],
				cutoff=2.8,
				n_reference=6,
				label="mixed_coordination"
			)
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
						target_species=target_species,
						use_reference_centers=use_reference_centers
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
	structure: Structure, 
	mobile_species: str | list[str], 
	centres: list[list[float]], 
	radii: float | list[float], 
	labels: str | list[str] | None = None
) -> Trajectory:
	"""Create a Trajectory with spherical sites for site analysis.
	
	Creates a trajectory object configured with spherical sites defined by centres
	and radii. Each spherical site represents a spherical volume in the crystal
	structure where atoms can be assigned based on distance from the centre.
	
	Args:
		structure: Pymatgen Structure object containing the crystal structure to analyse.
		mobile_species: Species string (e.g., "Li") or list of species strings 
			(e.g., ["Li", "Na"]) identifying the mobile atoms to track.
		centres: List of fractional coordinate triplets defining the centres of 
			spherical sites. Each centre should be a list of three floats [x, y, z].
		radii: Cutoff radius in Ångströms for spherical sites. If a single float 
			is provided, all sites use the same radius. If a list is provided, 
			it must have the same length as centres.
		labels: Optional labels for the sites. Can be a single string applied to 
			all sites, a list of strings (one per site), or None for no labels.
			
	Returns:
		Trajectory: Configured trajectory object ready for site analysis.
		
	Raises:
		ValueError: If the number of centres doesn't match the number of radii 
			or labels when lists are provided.
	"""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_mobile_species(mobile_species)
			.with_spherical_sites(centres, radii, labels)
			.build())


def create_trajectory_with_voronoi_sites(
	structure: Structure, 
	mobile_species: str | list[str], 
	centres: list[list[float]], 
	labels: list[str] | None = None
) -> Trajectory:
	"""Create a Trajectory with Voronoi sites for site analysis.
	
	Creates a trajectory object configured with Voronoi sites that partition space
	based on proximity to site centres. Each point in space is assigned to the
	Voronoi site with the nearest centre, ensuring complete space-filling coverage.
	
	Args:
		structure: Pymatgen Structure object containing the crystal structure to analyse.
		mobile_species: Species string (e.g., "Li") or list of species strings 
			(e.g., ["Li", "Na"]) identifying the mobile atoms to track.
		centres: List of fractional coordinate triplets defining the centres of 
			Voronoi sites. Each centre should be a list of three floats [x, y, z].
		labels: Optional list of string labels for the sites. Must have the same 
			length as centres if provided, or None for no labels.
			
	Returns:
		Trajectory: Configured trajectory object ready for site analysis.
		
	Raises:
		ValueError: If the number of labels doesn't match the number of centres.
	"""
	builder = TrajectoryBuilder()
	return (builder
			.with_structure(structure)
			.with_mobile_species(mobile_species)
			.with_voronoi_sites(centres, labels)
			.build())


def create_trajectory_with_polyhedral_sites(
	structure: Structure, 
	reference_structure: Structure, 
	mobile_species: str | list[str], 
	centre_species: str, 
	vertex_species: str | list[str], 
	cutoff: float, 
	n_vertices: int, 
	label: str | None = None,
	align: bool = True,
	align_species: str | list[str] | None = None,
	align_metric: str = 'rmsd',
	align_algorithm: str = 'Nelder-Mead',
	align_minimizer_options: dict[str, Any] | None = None,
	mapping_species: str | list[str] | None = None,
	align_tolerance: float = 1e-4,
	use_reference_centers: bool = True
) -> Trajectory:
	"""Create a Trajectory with polyhedral sites using reference-based workflow.
	
	Creates a trajectory object with polyhedral sites by identifying coordination
	environments in a reference structure and mapping them to corresponding sites
	in the target structure. Each site is defined by a polyhedron formed by vertex
	atoms around a central atom.
	
	Args:
		structure: Pymatgen Structure object containing the target structure to analyse.
		reference_structure: Pymatgen Structure object defining ideal coordination 
			environments to identify and map.
		mobile_species: Species string (e.g., "Li") or list of species strings 
			identifying the mobile atoms to track.
		centre_species: Atomic species at the centre of coordination environments
			in the reference structure.
		vertex_species: Atomic species at vertices of coordination polyhedra. 
			Can be a single species string or list of species strings.
		cutoff: Maximum distance in Ångströms for vertex atoms to be considered 
			part of the coordination environment.
		n_vertices: Number of vertex atoms required for each polyhedral site.
		label: Optional string label to assign to all created sites.
		align: Whether to perform structure alignment before site mapping.
		align_species: Species to use for structure alignment. If None and 
			mapping_species is specified, mapping_species will be used.
		align_metric: Alignment metric. Options are 'rmsd' for root-mean-square 
			deviation or 'max_dist' for maximum distance.
		align_algorithm: Optimisation algorithm. Options are 'Nelder-Mead' for 
			local optimisation or 'differential_evolution' for global optimisation.
		align_minimizer_options: Additional options dictionary for the alignment 
			optimiser.
		mapping_species: Species to use for mapping sites between structures. 
			If None, align_species will be used.
		align_tolerance: Convergence tolerance for alignment optimiser.
		use_reference_centers: Whether to use reference centers for 
			PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
			Default is True.
			
	Returns:
		Trajectory: Configured trajectory object ready for site analysis.
		
	Raises:
		ValueError: If no coordination environments are found, if required 
			structures are not set, or if structure alignment fails.
	"""
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
			.with_polyhedral_sites(
				centre_species,
				vertex_species,
				cutoff,
				n_vertices,
				label,
				use_reference_centers)
			.build())


def create_trajectory_with_dynamic_voronoi_sites(
	structure: Structure, 
	reference_structure: Structure, 
	mobile_species: str | list[str], 
	centre_species: str, 
	reference_species: str | list[str], 
	cutoff: float, 
	n_reference: int, 
	label: str | None = None,
	align: bool = True,
	align_species: str | list[str] | None = None,
	align_metric: str = 'rmsd',
	align_algorithm: str = 'Nelder-Mead',
	align_minimizer_options: dict[str, Any] | None = None,
	mapping_species: str | list[str] | None = None,
	align_tolerance: float = 1e-4,
	use_reference_centers: bool = True
) -> Trajectory:
	"""Create a Trajectory with dynamic Voronoi sites using reference-based workflow.
	
	Creates a trajectory object with dynamic Voronoi sites where site centres are
	dynamically calculated from the positions of reference atoms. Unlike fixed 
	Voronoi sites, these adapt to structural changes as the reference atoms move,
	making them suitable for analysing deformable frameworks.
	
	Args:
		structure: Pymatgen Structure object containing the target structure to analyse.
		reference_structure: Pymatgen Structure object defining coordination 
			environments to identify and map.
		mobile_species: Species string (e.g., "Li") or list of species strings 
			identifying the mobile atoms to track.
		centre_species: Atomic species at centres where dynamic sites will be located
			in the reference structure.
		reference_species: Atomic species used as reference atoms to define dynamic 
			site centres. Can be a single species string or list of species strings.
		cutoff: Maximum distance in Ångströms for reference atoms to be considered 
			part of the coordination environment.
		n_reference: Number of reference atoms required for each dynamic site.
		label: Optional string label to assign to all created sites.
		align: Whether to perform structure alignment before site mapping.
		align_species: Species to use for structure alignment. If None and 
			mapping_species is specified, mapping_species will be used.
		align_metric: Alignment metric. Options are 'rmsd' for root-mean-square 
			deviation or 'max_dist' for maximum distance.
		align_algorithm: Optimisation algorithm. Options are 'Nelder-Mead' for 
			local optimisation or 'differential_evolution' for global optimisation.
		align_minimizer_options: Additional options dictionary for the alignment 
			optimiser.
		mapping_species: Species to use for mapping sites between structures. 
			If None, align_species will be used.
		align_tolerance: Convergence tolerance for alignment optimiser.
		use_reference_centers: Whether to use reference centers for 
			PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
			Default is True.
			
	Returns:
		Trajectory: Configured trajectory object ready for site analysis.
		
	Raises:
		ValueError: If no coordination environments are found, if required 
			structures are not set, or if structure alignment fails.
	"""
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
			.with_dynamic_voronoi_sites(
				centre_species,
				reference_species,
				cutoff,
				n_reference,
				label,
				use_reference_centers)
			.build())
