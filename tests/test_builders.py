import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pymatgen.core import Structure, Lattice
from site_analysis.site import Site
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.atom import Atom

from site_analysis.builders import (
	TrajectoryBuilder,
	create_trajectory_with_spherical_sites,
	create_trajectory_with_voronoi_sites,
	create_trajectory_with_polyhedral_sites,
	create_trajectory_with_dynamic_voronoi_sites
)

class TestTrajectoryBuilder(unittest.TestCase):
	"""Tests for the TrajectoryBuilder class."""

	def setUp(self):
		"""Set up test fixtures."""
		# Create a simple test structure
		self.lattice = Lattice.cubic(5.0)
		self.structure = Structure(
			lattice=self.lattice,
			species=["Li", "O", "Li", "O"],
			coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]]
		)
		
		# Create a reference structure (slightly different)
		self.reference_structure = Structure(
			lattice=self.lattice,
			species=["Li", "O", "Li", "O"],
			coords=[[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.6, 0.1, 0.1], [0.1, 0.6, 0.6]]
		)
		
		# Sample site parameters
		self.centres = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
		self.radii = [1.0, 1.5]
		self.labels = ["tetrahedral", "octahedral"]
		
		# Create builder instance for tests
		self.builder = TrajectoryBuilder()
	
	def test_initialization(self):
		"""Test that TrajectoryBuilder initializes with empty state."""
		builder = TrajectoryBuilder()
		
		self.assertIsNone(builder._structure)
		self.assertIsNone(builder._reference_structure)
		self.assertIsNone(builder._mobile_species)
		self.assertIsNone(builder._atoms)
		self.assertEqual(builder._site_generators, [])
	
	def test_method_chaining(self):
		"""Test that all builder methods return self for method chaining."""
		# Start with a fresh builder
		builder = TrajectoryBuilder()
		
		# Chain several methods
		result = (builder
				.with_structure(self.structure)
				.with_reference_structure(self.reference_structure)
				.with_mobile_species("Li"))
		
		# Verify that the result is the same builder instance
		self.assertIs(result, builder)
		
		# Verify that the builder state was updated
		self.assertEqual(builder._structure, self.structure)
		self.assertEqual(builder._reference_structure, self.reference_structure)
		self.assertEqual(builder._mobile_species, "Li")
	
	def test_with_spherical_sites_validation(self):
		"""Test validation in with_spherical_sites."""
		# Test validation for centres and radii length mismatch
		with self.assertRaises(ValueError) as context:
			self.builder.with_spherical_sites(
				centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
				radii=[1.0]  # Only one radius for two centres
			)
		
		# Check error message
		self.assertIn("match", str(context.exception).lower())
	
	def test_with_spherical_sites_sets_generator(self):
		"""Test that with_spherical_sites sets a site generator function."""
		# Call the method
		builder = self.builder.with_spherical_sites(
			centres=self.centres,
			radii=self.radii,
			labels=self.labels
		)
		
		# Verify a site generator was set
		self.assertNotEqual(builder._site_generators, [])
		self.assertTrue(callable(builder._site_generators[0]))
	
	def test_with_voronoi_sites_sets_generator(self):
		"""Test that with_voronoi_sites sets a site generator function."""
		# Call the method
		builder = self.builder.with_voronoi_sites(
			centres=self.centres,
			labels=self.labels
		)
		
		# Verify a site generator was set
		self.assertNotEqual(builder._site_generators, [])
		self.assertTrue(callable(builder._site_generators[0]))
	
	def test_with_existing_sites_sets_generator(self):
		"""Test that with_existing_sites sets a site generator function."""
		mock_sites = [Mock(), Mock()]
		
		# Call the method
		builder = self.builder.with_existing_sites(mock_sites)
		
		# Verify a site generator was set
		self.assertNotEqual(builder._site_generators, [])
		self.assertTrue(callable(builder._site_generators[0]))
	
	def test_with_polyhedral_sites_sets_generator(self):
		"""Test that with_polyhedral_sites sets a site generator function."""
		# Call the method
		builder = self.builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Verify a site generator was set
		self.assertNotEqual(builder._site_generators, [])
		self.assertTrue(callable(builder._site_generators[0]))
	
	def test_with_dynamic_voronoi_sites_sets_generator(self):
		"""Test that with_dynamic_voronoi_sites sets a site generator function."""
		# Call the method
		builder = self.builder.with_dynamic_voronoi_sites(
			centre_species="Li",
			reference_species="O",
			cutoff=2.0,
			n_reference=4,
			label="tetrahedral"
		)
		
		# Verify a site generator was set
		self.assertNotEqual(builder._site_generators, [])
		self.assertTrue(callable(builder._site_generators[0]))
	
	def test_deferred_site_creation_spherical_sites(self):
		"""Test that spherical sites are created at build time."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_mobile_species("Li")
		builder.with_spherical_sites(
			centres=self.centres,
			radii=self.radii,
			labels=self.labels
		)
		
		# Mock SphericalSite to verify when it's called
		with patch('site_analysis.builders.SphericalSite') as mock_spherical_site, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return site objects
			mock_site1 = Mock()
			mock_site2 = Mock()
			mock_spherical_site.side_effect = [mock_site1, mock_site2]
			
			# Verify SphericalSite is not called yet
			mock_spherical_site.assert_not_called()
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify SphericalSite was called during build
			self.assertEqual(mock_spherical_site.call_count, 2)
			
			# Check that site parameters were used correctly
			args1, kwargs1 = mock_spherical_site.call_args_list[0]
			self.assertEqual(kwargs1['frac_coords'].tolist(), self.centres[0])
			self.assertEqual(kwargs1['rcut'], self.radii[0])
			self.assertEqual(kwargs1['label'], self.labels[0])
			
			args2, kwargs2 = mock_spherical_site.call_args_list[1]
			self.assertEqual(kwargs2['frac_coords'].tolist(), self.centres[1])
			self.assertEqual(kwargs2['rcut'], self.radii[1])
			self.assertEqual(kwargs2['label'], self.labels[1])
	
	def test_deferred_polyhedral_site_creation(self):
		"""Test that polyhedral sites are created at build time."""
		# Configure the builder first without reference structure
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_mobile_species("Li")
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Add reference structure after site configuration
		builder.with_reference_structure(self.reference_structure)
		
		# Mock ReferenceBasedSites to verify when it's called
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_polyhedral_sites.return_value = mock_sites
			
			# Verify ReferenceBasedSites is not called yet
			mock_rbs_class.assert_not_called()
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify ReferenceBasedSites was called during build
			mock_rbs_class.assert_called_once()
			
			# Verify the correct parameters were used
			mock_rbs.create_polyhedral_sites.assert_called_once_with(
				center_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4,
				label="tetrahedral",
				target_species=None
			)
	
	def test_flexible_method_call_order(self):
		"""Test that methods can be called in any order."""
		# Start with a fresh builder
		builder = TrajectoryBuilder()
		
		# Configure in a different order than typical
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Add structure after site configuration
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		
		# Mock to avoid actual site creation
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_polyhedral_sites.return_value = mock_sites
			
			# Build should succeed despite unusual order
			builder.build()
			
			# Verify the correct parameters were used
			mock_rbs.create_polyhedral_sites.assert_called_once()
	
	def test_build_checks_required_parameters(self):
		"""Test that build validates required parameters."""
		# Configure without setting structure
		builder = TrajectoryBuilder()
		builder.with_spherical_sites(
			centres=self.centres,
			radii=self.radii
		)
		
		# Build should fail
		with self.assertRaises(ValueError) as context:
			builder.build()
		
		# Check error message
		self.assertIn("structure must be set", str(context.exception).lower())
		
		# Fix structure but leave out mobile species
		builder.with_structure(self.structure)
		
		# Build should still fail
		with self.assertRaises(ValueError) as context:
			builder.build()
		
		# Check error message
		self.assertIn("mobile species must be set", str(context.exception).lower())
	
	def test_reference_structure_checked_at_build_time(self):
		"""Test that reference structure is checked at build time for complex sites."""
		# Configure the builder without reference structure
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_mobile_species("Li")
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4
		)
		
		# Build should fail because reference structure is missing
		with self.assertRaises(ValueError) as context:
			builder.build()
		
		# Check error message
		self.assertIn("reference_structure must be set", str(context.exception).lower())
	
	@patch('site_analysis.builders.atoms_from_structure')
	@patch('site_analysis.builders.Trajectory')
	def test_build_with_existing_atoms(self, mock_trajectory_class, mock_atoms_from_structure):
		"""Test building a trajectory with pre-existing atoms."""
		# Configure the trajectory mock
		mock_trajectory = Mock()
		mock_trajectory_class.return_value = mock_trajectory
		
		# Configure the builder
		mock_sites = [Mock(), Mock()]
		mock_atoms = [Mock(), Mock()]
		
		self.builder.with_structure(self.structure)
		self.builder.with_existing_sites(mock_sites)
		self.builder.with_existing_atoms(mock_atoms)
		
		# Call build
		result = self.builder.build()
		
		# Verify atoms_from_structure was NOT called
		mock_atoms_from_structure.assert_not_called()
		
		# Verify trajectory was created with expected arguments
		mock_trajectory_class.assert_called_once()
		args, kwargs = mock_trajectory_class.call_args
		self.assertEqual(kwargs['atoms'], mock_atoms)
		
	def test_with_structure_alignment(self):
		"""Test setting alignment options."""
		# Start with a fresh builder
		builder = TrajectoryBuilder()
		
		# Set alignment options with a list of species
		result = builder.with_structure_alignment(
			align=False,
			align_species=["Li"],
			align_metric="max_dist"
		)
		
		# Verify chaining works
		self.assertIs(result, builder)
		
		# Verify options were stored
		self.assertEqual(builder._align, False)
		self.assertEqual(builder._align_species, ["Li"])
		self.assertEqual(builder._align_metric, "max_dist")
		
		# Test with a single species string
		builder.with_structure_alignment(
			align=True,
			align_species="Na",
			align_metric="rmsd"
		)
		
		# Verify single species string is converted to list
		self.assertEqual(builder._align, True)
		self.assertEqual(builder._align_species, ["Na"])
		self.assertEqual(builder._align_metric, "rmsd")
		
		# Test with default align value
		builder.with_structure_alignment(
			align_species="Ca",
			align_metric="max_dist"
		)
		
		# Verify align defaulted to True
		self.assertEqual(builder._align, True)
		self.assertEqual(builder._align_species, ["Ca"])
		self.assertEqual(builder._align_metric, "max_dist")
	
	def test_polyhedral_sites_no_sites_found(self):
		"""Test that build() raises an error when polyhedral sites return an empty list."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to return empty site list
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class:
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			mock_rbs.create_polyhedral_sites.return_value = []
			
			# Build should fail when no sites are found
			with self.assertRaises(ValueError) as context:
				builder.build()
			
			# Check error message contains useful information
			error_msg = str(context.exception)
			self.assertIn("No polyhedral sites found", error_msg)
			self.assertIn("Li", error_msg)  # centre_species
			self.assertIn("O", error_msg)   # vertex_species
			self.assertIn("2.0", error_msg) # cutoff
			self.assertIn("4", error_msg)   # n_vertices
	
	def test_dynamic_voronoi_sites_no_sites_found(self):
		"""Test that build() raises an error when dynamic Voronoi sites return an empty list."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		builder.with_dynamic_voronoi_sites(
			centre_species="Li",
			reference_species="O",
			cutoff=2.0,
			n_reference=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to return empty site list
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class:
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			mock_rbs.create_dynamic_voronoi_sites.return_value = []
			
			# Build should fail when no sites are found
			with self.assertRaises(ValueError) as context:
				builder.build()
			
			# Check error message contains useful information
			error_msg = str(context.exception)
			self.assertIn("No dynamic Voronoi sites found", error_msg)
			self.assertIn("Li", error_msg)  # centre_species
			self.assertIn("O", error_msg)   # reference_species
			self.assertIn("2.0", error_msg) # cutoff
			self.assertIn("4", error_msg)   # n_reference
	
	def test_build_complete(self):
		"""Test building a complete trajectory with all required components."""
		# Configure the mocks
		mock_sites = [Mock(), Mock()]
		mock_atoms = [Mock(), Mock()]
		
		with patch('site_analysis.builders.atoms_from_structure') as mock_atoms_from_structure, \
			 patch('site_analysis.builders.Trajectory') as mock_trajectory_class:
			
			# Configure atoms_from_structure to return mock atoms
			mock_atoms_from_structure.return_value = mock_atoms
			
			# Configure Trajectory to return a mock trajectory
			mock_trajectory = Mock()
			mock_trajectory_class.return_value = mock_trajectory
			
			# Configure the builder with existing sites to avoid mocking site creation
			builder = TrajectoryBuilder()
			builder.with_structure(self.structure)
			builder.with_mobile_species("Li")
			builder.with_existing_sites(mock_sites)
			
			# Call build
			result = builder.build()
			
			# Verify atoms were created
			mock_atoms_from_structure.assert_called_once_with(self.structure, "Li")
			
			# Verify trajectory was created
			mock_trajectory_class.assert_called_once()
			args, kwargs = mock_trajectory_class.call_args
			self.assertEqual(kwargs['sites'], mock_sites)
			self.assertEqual(kwargs['atoms'], mock_atoms)
			
			# Result should be the trajectory
			self.assertEqual(result, mock_trajectory)
	
	def test_build_requires_site_generator(self):
		"""Test that build() requires a site generator to be defined."""
		# Configure the builder without defining any sites
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_mobile_species("Li")
		
		# Build should fail
		with self.assertRaises(ValueError) as context:
			builder.build()
		
		# Check error message
		self.assertIn("site type must be defined", str(context.exception).lower())
		
	def test_builder_with_interstitial_in_simple_cubic(self):
		"""Test handling of structures with an interstitial mobile ion in a simple cubic lattice."""
		# Create a simple cubic structure with O at (0,0,0)
		lattice = Lattice.cubic(5.0)  # 5 Å lattice parameter
		primitive_structure = Structure(
			lattice=lattice,
			species=["O"],  # Oxygen at the origin
			coords=[[0.0, 0.0, 0.0]]
		)
		
		# Create a 3x3x3 supercell with 27 oxygen atoms
		reference_structure = primitive_structure * [3, 3, 3]
		
		# Create target structure as a copy of the reference structure
		target_structure = reference_structure.copy()
		
		# Add a Li atom at the interstitial position (center of the supercell)
		target_structure.append("Li", [0.5, 0.5, 0.5])
		
		# Test two configurations:
		
		# 1. First test: Aligning on oxygen (should work)
		builder1 = TrajectoryBuilder()
		builder1.with_structure(target_structure)
		builder1.with_reference_structure(reference_structure)
		builder1.with_mobile_species("Li")
		
		# Create polyhedral sites aligned on oxygen atoms
		builder1.with_polyhedral_sites(
			centre_species="O",         # Sites centered on oxygen
			vertex_species="O",         # Vertices defined by oxygen
			cutoff=5.0,
			n_vertices=6,
			label="test_site"
		).with_structure_alignment(align=True, align_species=["O"])  # Align on oxygen
		
		# This should work fine since oxygen atoms match in both structures
		try:
			trajectory = builder1.build()
			# Verify we have a valid trajectory with the expected properties
			self.assertEqual(len(trajectory.atoms), 1)  # 1 Li atom
			self.assertGreater(len(trajectory.sites), 0)  # Should have created sites
		except ValueError as e:
			self.fail(f"Builder with aligned oxygen framework raised an error: {str(e)}")
		
		# 2. Second test: Trying to align on Li (should fail)
		builder2 = TrajectoryBuilder()
		builder2.with_structure(target_structure)
		builder2.with_reference_structure(reference_structure)
		builder2.with_mobile_species("Li")
		
		# Create polyhedral sites but try to align on Li atoms (which mismatch)
		builder2.with_polyhedral_sites(
			centre_species="O",
			vertex_species="O", 
			cutoff=3.0,
			n_vertices=3,
			label="test_site"
		).with_structure_alignment(align=True, align_species=["Li"])  # Align on Li
		
		# This should fail because Li counts don't match (0 vs 1)
		with self.assertRaises(ValueError) as context:
			builder2.build()
		
		# Check that the error message mentions the mismatch in atom counts
		error_msg = str(context.exception)
		self.assertIn("Li", error_msg)  # Should mention Li species
		self.assertIn("not found in reference structure", error_msg)  # Since Li is only in target
		
	def test_builder_with_vacancy_in_simple_cubic(self):
		"""Test handling when the target has a vacancy compared to the reference structure."""
		# Create a primitive cell containing both O and Li
		lattice = Lattice.cubic(5.0)  # 5 Å lattice parameter
		primitive_structure = Structure(
			lattice=lattice,
			species=["O", "Li"],  # Include both O and Li in primitive cell
			coords=[
				[0.0, 0.0, 0.0],  # O at origin
				[0.5, 0.5, 0.5]   # Li at center
			]
		)
		
		# Create a 3x3x3 supercell with all sites filled
		reference_structure = primitive_structure * [3, 3, 3]
		# This gives us 27 O atoms and 27 Li atoms
		
		# Create target structure by removing the central Li atom
		target_structure = reference_structure.copy()
		
		# Find the index of the Li atom at the center of the supercell [0.5, 0.5, 0.5]
		center_position = np.array([0.5, 0.5, 0.5])
		center_index = None
		
		for i, site in enumerate(target_structure):
			if site.species_string == "Li":
				# Check if this is the center position
				if np.allclose(site.frac_coords, center_position, atol=0.01):
					center_index = i
					break
		
		# Remove the Li atom at the center
		if center_index is not None:
			target_structure.remove_sites([center_index])
		
		# Verify we have the expected number of atoms
		self.assertEqual(len(reference_structure), 27 + 27)  # 27 O + 27 Li
		self.assertEqual(len(target_structure), 27 + 26)     # 27 O + 26 Li
		
		# Test two configurations:
		
		# 1. First test: Aligning on oxygen (should work)
		builder1 = TrajectoryBuilder()
		builder1.with_structure(target_structure)
		builder1.with_reference_structure(reference_structure)
		builder1.with_mobile_species("Li")  # Track all Li atoms
		
		builder1.with_polyhedral_sites(
			centre_species="O",
			vertex_species="O",
			cutoff=5.0,
			n_vertices=6,
			label="test_site"
		).with_structure_alignment(align=True, align_species=["O"])  # Align on oxygen
		
		# This should work fine since oxygen atoms match
		try:
			trajectory = builder1.build()
			# Verify we have a valid trajectory with all Li atoms
			self.assertEqual(len(trajectory.atoms), 26)  # 26 Li atoms (one missing)
			self.assertGreater(len(trajectory.sites), 0)  # Should have created sites
		except ValueError as e:
			self.fail(f"Builder with aligned oxygen framework raised an error: {str(e)}")
		
		# 2. Second test: Trying to align on Li (should fail)
		builder2 = TrajectoryBuilder()
		builder2.with_structure(target_structure)
		builder2.with_reference_structure(reference_structure)
		builder2.with_mobile_species("Li")
		
		# Create polyhedral sites but try to align on Li atoms (which mismatch)
		builder2.with_polyhedral_sites(
			centre_species="O",
			vertex_species="O", 
			cutoff=5.0,
			n_vertices=6,
			label="test_site"
		).with_structure_alignment(align=True, align_species=["Li"])  # Align on Li
		
		# This should fail because Li counts don't match (27 vs 26)
		with self.assertRaises(ValueError) as context:
			builder2.build()
		
		# Check that the error message mentions the mismatch in atom counts
		error_msg = str(context.exception)
		self.assertIn("Different number of Li atoms", error_msg)
		self.assertIn("27 in reference", error_msg)  # 27 Li atoms in reference
		self.assertIn("26 in target", error_msg)     # 26 Li atoms in target
		
	def test_site_indices_reset_between_trajectories(self):
		"""Test that site indices are reset for each new trajectory build."""
		# Create a simple structure for testing
		lattice = Lattice.cubic(5.0)
		structure = Structure(
			lattice=lattice,
			species=["Li", "O"],
			coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
		)
		
		# Reset the site counter to ensure a clean test
		Site.reset_index()
		
		# Configure first builder
		builder1 = TrajectoryBuilder()
		builder1.with_structure(structure)
		builder1.with_mobile_species("Li")
		builder1.with_spherical_sites(
			centres=[[0.5, 0.5, 0.5]],
			radii=[1.0],
			labels=["test_site"]
		)
		
		# Build first trajectory (without mocking)
		first_trajectory = builder1.build()
		
		# Verify first site has index 0
		self.assertEqual(first_trajectory.sites[0].index, 0)
		
		# Verify Site._newid is now 1 (after creating one site)
		self.assertEqual(Site._newid, 1)
		
		# Configure second builder
		builder2 = TrajectoryBuilder()
		builder2.with_structure(structure)
		builder2.with_mobile_species("Li")
		builder2.with_spherical_sites(
			centres=[[0.5, 0.5, 0.5]],
			radii=[1.0],
			labels=["test_site"]
		)
		
		# Build second trajectory (without mocking)
		second_trajectory = builder2.build()
		
		# The second site should also have index 0
		self.assertEqual(second_trajectory.sites[0].index, 0)
		
		# And Site._newid should still be 1 after the second build
		self.assertEqual(Site._newid, 1)

	def test_multiple_polyhedral_site_groups_with_dummy_atoms(self):
		"""Test creating multiple groups of polyhedral sites in a single trajectory.
		
		This test creates an FCC structure with dummy atoms at tetrahedral and octahedral
		interstitial sites, then constructs polyhedral sites for both types.
		"""
		# Create FCC structure with Cu atoms and dummy T/O atoms at interstitial sites
		lattice = Lattice.cubic(5.64)  # FCC lattice parameter
		
		# Define all sites in the FCC lattice in a single call:
		# - Cu atoms at Wyckoff position 4a: (0, 0, 0)
		# - Tetrahedral sites (T) at Wyckoff position 8c: (1/4, 1/4, 1/4)
		# - Octahedral sites (O) at Wyckoff position 4b: (1/2, 0, 0)
		fcc_structure = Structure.from_spacegroup(
			sg="Fm-3m",
			lattice=lattice,
			species=["Cu", "S", "O"],
			coords=[
				[0.0, 0.0, 0.0],    # Cu at FCC positions
				[0.25, 0.25, 0.25],  # T at tetrahedral sites 
				[0.5, 0.0, 0.0]      # O at octahedral sites
			]
		)
		
		# Make a 2x2x2 supercell to avoid boundary issues
		supercell = fcc_structure * [2, 2, 2]
		
		# Add a Li atom to the structure (needed as the mobile species)
		supercell.append("Li", [0.1, 0.1, 0.1])
		
		# Configure builder
		builder = TrajectoryBuilder()
		builder.with_structure(supercell)
		builder.with_reference_structure(supercell.copy())
		builder.with_mobile_species("Li")  # Hypothetical mobile species
		
		# First group: polyhedral sites centered at T atoms with Cu atoms as vertices
		builder.with_polyhedral_sites(
			centre_species="S",
			vertex_species="Cu",
			cutoff=3.0,
			n_vertices=4,  # Each tetrahedral site has 4 Cu atoms as vertices
			label="tetrahedral"
		)
		
		# Second group: polyhedral sites centered at O atoms with Cu atoms as vertices
		builder.with_polyhedral_sites(
			centre_species="O",
			vertex_species="Cu",
			cutoff=3.0,
			n_vertices=6,  # Each octahedral site has 6 Cu atoms as vertices
			label="octahedral"
		)
		
		# Build trajectory
		trajectory = builder.build()
		
		# Verify sites were created
		self.assertGreater(len(trajectory.sites), 0)
		
		# Get counts of tetrahedral and octahedral dummy atoms in the supercell
		t_count = len([site for site in supercell if site.species_string == "S"])
		o_count = len([site for site in supercell if site.species_string == "O"])
		
		# Verify we have both tetrahedral and octahedral sites
		tetrahedral_sites = [s for s in trajectory.sites if s.label == "tetrahedral"]
		octahedral_sites = [s for s in trajectory.sites if s.label == "octahedral"]
		
		# Verify correct counts of sites
		self.assertEqual(len(tetrahedral_sites), t_count)  # Should match dummy T atoms
		self.assertEqual(len(octahedral_sites), o_count)   # Should match dummy O atoms		
		
		# Verify all sites are PolyhedralSite instances
		for site in trajectory.sites:
			self.assertIsInstance(site, PolyhedralSite)
		
		# Verify site indices are sequential
		self.assertEqual(min(site.index for site in trajectory.sites), 0)
		expected_count = len(tetrahedral_sites) + len(octahedral_sites)
		self.assertEqual(max(site.index for site in trajectory.sites), expected_count - 1)
		
	def test_with_site_mapping(self):
		"""Test setting site mapping options."""
		# Start with a fresh builder
		builder = TrajectoryBuilder()
		
		# Set mapping options with a list of species
		result = builder.with_site_mapping(
			mapping_species=["Li"]
		)
		
		# Verify chaining works
		self.assertIs(result, builder)
		
		# Verify options were stored
		self.assertEqual(builder._mapping_species, ["Li"])
		
		# Test with a single species string
		builder.with_site_mapping(
			mapping_species="Na"
		)
		
		# Verify single species string is converted to list
		self.assertEqual(builder._mapping_species, ["Na"])
		
	def test_mapping_species_passed_to_reference_based_sites(self):
		"""Test that mapping species are correctly passed to ReferenceBasedSites."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		
		# Set different species for alignment and mapping
		builder.with_structure_alignment(align=True, align_species=["O"])
		builder.with_site_mapping(mapping_species=["Na"])
		
		# Set up polyhedral sites
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to verify correct parameters are passed
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_polyhedral_sites.return_value = mock_sites
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify ReferenceBasedSites was created with the correct alignment parameters
			mock_rbs_class.assert_called_once_with(
				reference_structure=self.reference_structure,
				target_structure=self.structure,
				align=True,
				align_species=["O"],
				align_metric='rmsd',
				align_algorithm='Nelder-Mead',
				align_minimizer_options=None,
				align_tolerance=0.0001
			)
			
			# Verify create_polyhedral_sites was called with the correct mapping parameters
			mock_rbs.create_polyhedral_sites.assert_called_once_with(
				center_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4,
				label="tetrahedral",
				target_species=["Na"]  # This is the key assertion - mapping species should be passed here
			)
			
	def test_mapping_uses_alignment_species_by_default(self):
		"""Test that mapping uses alignment species when mapping species is not specified."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		
		# Set alignment species but NOT mapping species
		builder.with_structure_alignment(align=True, align_species=["O"])
		# Deliberately NOT calling with_site_mapping()
		
		# Set up polyhedral sites
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to verify correct parameters are passed
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_polyhedral_sites.return_value = mock_sites
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify create_polyhedral_sites was called with the alignment species as target_species
			mock_rbs.create_polyhedral_sites.assert_called_once_with(
				center_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4,
				label="tetrahedral",
				target_species=["O"]  # This should be the same as align_species
			)
			
	def test_factory_function_with_mapping_species(self):
		"""Test that factory functions correctly use mapping_species."""
		# Mock the necessary classes
		with patch('site_analysis.builders.TrajectoryBuilder') as mock_builder_class, \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock builder
			mock_builder = Mock()
			mock_builder_class.return_value = mock_builder
			
			# Configure method chaining
			for method in ['with_structure', 'with_reference_structure', 'with_mobile_species',
						'with_structure_alignment', 'with_site_mapping', 'with_polyhedral_sites', 'build']:
				setattr(mock_builder, method, Mock(return_value=mock_builder))
			
			# Call the factory function with mapping_species
			create_trajectory_with_polyhedral_sites(
				structure=self.structure,
				reference_structure=self.reference_structure,
				mobile_species="Li",
				centre_species="O",
				vertex_species="Li",
				cutoff=2.0,
				n_vertices=4,
				label="test",
				align_species=["O"],
				mapping_species=["Li"]
			)
			
			# Verify with_site_mapping was called with correct parameter
			mock_builder.with_site_mapping.assert_called_once_with(["Li"])
			
			# Reset the mock
			mock_builder.reset_mock()
			
			# Also test with dynamic Voronoi sites
			create_trajectory_with_dynamic_voronoi_sites(
				structure=self.structure,
				reference_structure=self.reference_structure,
				mobile_species="Li",
				centre_species="O",
				reference_species="Li",
				cutoff=2.0,
				n_reference=4,
				label="test",
				align_species=["O"],
				mapping_species=["Li"]
			)
			
			# Verify with_site_mapping was called for dynamic sites too
			mock_builder.with_site_mapping.assert_called_once_with(["Li"])
			
	def test_mapping_species_used_for_alignment_when_align_species_not_set(self):
		"""Test that mapping species are used for alignment when alignment species are not set."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		
		# Set mapping species but NOT alignment species
		# Don't call with_structure_alignment()
		builder.with_site_mapping(mapping_species=["O"])
		
		# Set up polyhedral sites
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to verify correct parameters are passed
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_polyhedral_sites.return_value = mock_sites
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify ReferenceBasedSites was created with the mapping species as alignment species
			mock_rbs_class.assert_called_once_with(
				reference_structure=self.reference_structure,
				target_structure=self.structure,
				align=True,  # Alignment should be enabled
				align_species=["O"],  # Should use mapping species for alignment
				align_metric='rmsd',
				align_algorithm='Nelder-Mead',
				align_minimizer_options=None,
				align_tolerance=0.0001
			)
			
	def test_with_spherical_sites_single_radius(self):
		"""Test that a single radius value is expanded to match the number of centres."""
		centres = [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
		radius = 1.5  # Single float instead of list
		
		builder = self.builder.with_spherical_sites(centres=centres, radii=radius)
		
		# Execute the site generator function to see what it produces
		site_generator = builder._site_generators[0]
		
		# Mock SphericalSite to capture what it's called with
		with patch('site_analysis.builders.SphericalSite') as mock_spherical_site:
			# Call the generator
			site_generator()
			
			# Check that SphericalSite was called 3 times (once per centre)
			self.assertEqual(mock_spherical_site.call_count, 3)
			
			# Check that each call used the same radius
			for call in mock_spherical_site.call_args_list:
				args, kwargs = call
				self.assertEqual(kwargs['rcut'], 1.5)
				
	def test_with_spherical_sites_single_label(self):
		"""Test that a single label value is expanded to match the number of centres."""
		centres = [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]
		radii = [1.0, 1.5, 2.0]
		label = "test_site"  # Single string instead of list
		
		builder = self.builder.with_spherical_sites(centres=centres, radii=radii, labels=label)
		
		# Execute the site generator function to see what it produces
		site_generator = builder._site_generators[0]
		
		# Mock SphericalSite to capture what it's called with
		with patch('site_analysis.builders.SphericalSite') as mock_spherical_site:
			# Call the generator
			site_generator()
			
			# Check that SphericalSite was called 3 times (once per centre)
			self.assertEqual(mock_spherical_site.call_count, 3)
			
			# Check that each call used the same label
			for call in mock_spherical_site.call_args_list:
				args, kwargs = call
				self.assertEqual(kwargs['label'], "test_site")
				
	def test_create_trajectory_with_spherical_sites_single_radius(self):
		"""Test factory function accepts a single radius value."""
		centres = [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]
		radius = 1.0  # Single float
		
		# Mock TrajectoryBuilder to verify it receives the single radius
		with patch('site_analysis.builders.TrajectoryBuilder') as mock_builder_class:
			mock_builder = Mock()
			mock_builder_class.return_value = mock_builder
			
			# Configure method chaining
			for method in ['with_structure', 'with_mobile_species', 'with_spherical_sites', 'build']:
				setattr(mock_builder, method, Mock(return_value=mock_builder))
			
			# Call the factory function
			create_trajectory_with_spherical_sites(
				structure=self.structure,
				mobile_species="Li",
				centres=centres,
				radii=radius
			)
			
			# Verify with_spherical_sites was called with the single radius
			mock_builder.with_spherical_sites.assert_called_once_with(centres, radius, None)
	
	def test_create_trajectory_with_spherical_sites_single_label(self):
		"""Test factory function accepts a single label value."""
		centres = [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]
		radii = [1.0, 1.5]
		label = "test_site"  # Single string
		
		# Mock TrajectoryBuilder to verify it receives the single label
		with patch('site_analysis.builders.TrajectoryBuilder') as mock_builder_class:
			mock_builder = Mock()
			mock_builder_class.return_value = mock_builder
			
			# Configure method chaining
			for method in ['with_structure', 'with_mobile_species', 'with_spherical_sites', 'build']:
				setattr(mock_builder, method, Mock(return_value=mock_builder))
			
			# Call the factory function
			create_trajectory_with_spherical_sites(
				structure=self.structure,
				mobile_species="Li",
				centres=centres,
				radii=radii,
				labels=label
			)
			
			# Verify with_spherical_sites was called with the single label
			mock_builder.with_spherical_sites.assert_called_once_with(centres, radii, label)
			
	def test_builder_state_reset_after_build(self):
		"""Test that the builder resets its entire state after build() is called."""
		# Create a builder
		builder = TrajectoryBuilder()
		
		# Set state directly to non-default values
		builder._structure = Mock(spec=Structure)
		builder._reference_structure = Mock(spec=Structure)
		builder._mobile_species = "Li"
		builder._atoms = [Mock(spec=Atom)]
		builder._align = False
		builder._align_species = ["Na"]
		builder._align_metric = "max_dist"
		builder._mapping_species = ["Cl"]
		builder._site_generators = [lambda: []]
		
		# Mock the methods called within build
		with patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'), \
			patch('site_analysis.builders.Site.reset_index'):
			
			# Call build
			builder.build()
			
			# Verify entire state is reset
			self.assertIsNone(builder._structure)
			self.assertIsNone(builder._reference_structure)
			self.assertIsNone(builder._mobile_species)
			self.assertIsNone(builder._atoms)
			self.assertTrue(builder._align)  # Default is True
			self.assertIsNone(builder._align_species)
			self.assertEqual(builder._align_metric, 'rmsd')
			self.assertIsNone(builder._mapping_species)
			self.assertEqual(builder._site_generators, [])
			
	def test_reset(self):
		"""Test that the reset() method returns all builder attributes to default values."""
		# Create a builder
		builder = TrajectoryBuilder()
		
		# Set state to non-default values
		builder._structure = Mock(spec=Structure)
		builder._reference_structure = Mock(spec=Structure)
		builder._mobile_species = "Li"
		builder._atoms = [Mock(spec=Atom)]
		builder._align = False
		builder._align_species = ["Na"]
		builder._align_metric = "max_dist"
		builder._mapping_species = ["Cl"]
		builder._site_generators = [lambda: []]
		
		# Call reset and verify it returns self for chaining
		result = builder.reset()
		self.assertIs(result, builder, "reset() should return self for method chaining")
		
		# Verify all attributes are reset to defaults
		self.assertIsNone(builder._structure)
		self.assertIsNone(builder._reference_structure)
		self.assertIsNone(builder._mobile_species)
		self.assertIsNone(builder._atoms)
		self.assertTrue(builder._align)  # Default is True
		self.assertIsNone(builder._align_species)
		self.assertEqual(builder._align_metric, 'rmsd')
		self.assertIsNone(builder._mapping_species)
		self.assertEqual(builder._site_generators, [])
		
	def test_initialization_calls_reset(self):
		"""Test that TrajectoryBuilder.__init__() calls reset()."""
		# Patch the reset method
		with patch.object(TrajectoryBuilder, 'reset') as mock_reset:
			# Create a new TrajectoryBuilder instance
			builder = TrajectoryBuilder()
			
			# Verify reset() was called exactly once
			mock_reset.assert_called_once()
			
	def test_with_structure_alignment_unit(self):
		"""Unit test that with_structure_alignment correctly stores alignment parameters."""
		# Create a test builder
		builder = TrajectoryBuilder()
		
		# Define test parameters
		align = True
		align_species = ["Na"]
		align_metric = "max_dist"
		align_algorithm = "differential_evolution"
		align_minimizer_options = {"popsize": 20, "maxiter": 500}
		
		# Call with_structure_alignment with our parameters
		result = builder.with_structure_alignment(
			align=align,
			align_species=align_species,
			align_metric=align_metric,
			align_algorithm=align_algorithm,
			align_minimizer_options=align_minimizer_options
		)
		
		# Verify method returns self for chaining
		self.assertIs(result, builder)
		
		# Verify parameters are correctly stored in instance variables
		self.assertEqual(builder._align, align)
		self.assertEqual(builder._align_species, align_species)
		self.assertEqual(builder._align_metric, align_metric)
		self.assertEqual(builder._align_algorithm, align_algorithm)
		self.assertEqual(builder._align_minimizer_options, align_minimizer_options)
		
		# Test with single string for align_species
		builder.with_structure_alignment(align_species="Ca")
		self.assertEqual(builder._align_species, ["Ca"], 
						"Single string species should be converted to a list")
						
	def test_factory_functions_with_alignment_options(self):
		"""Test that factory functions accept and pass alignment algorithm, options, and tolerance."""
		# Mock necessary classes
		with patch('site_analysis.builders.TrajectoryBuilder') as MockBuilder, \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock builder
			mock_builder = Mock()
			MockBuilder.return_value = mock_builder
			
			# Configure method chaining
			for method in ['with_structure', 'with_reference_structure', 'with_mobile_species',
						'with_structure_alignment', 'with_polyhedral_sites', 'build']:
				setattr(mock_builder, method, Mock(return_value=mock_builder))
			
			# Test with polyhedral sites
			algorithm = 'differential_evolution'
			minimizer_options = {'popsize': 20, 'maxiter': 500}
			custom_tolerance = 1e-5  # Add custom tolerance
			
			create_trajectory_with_polyhedral_sites(
				structure=Mock(spec=Structure),
				reference_structure=Mock(spec=Structure),
				mobile_species="Li",
				centre_species="O",
				vertex_species="Li",
				cutoff=2.0,
				n_vertices=4,
				label="test",
				align=True,
				align_species=["O"],
				align_metric="rmsd",
				align_algorithm=algorithm,
				align_minimizer_options=minimizer_options,
				align_tolerance=custom_tolerance  
			)
			
			# Verify with_structure_alignment was called with all parameters
			mock_builder.with_structure_alignment.assert_called_once()
			args, kwargs = mock_builder.with_structure_alignment.call_args
			self.assertEqual(kwargs['align'], True)
			self.assertEqual(kwargs['align_species'], ["O"])
			self.assertEqual(kwargs['align_metric'], "rmsd")
			self.assertEqual(kwargs['align_algorithm'], algorithm)
			self.assertEqual(kwargs['align_minimizer_options'], minimizer_options)
			self.assertEqual(kwargs['align_tolerance'], custom_tolerance)  
			
			# Reset mocks for dynamic voronoi sites test
			mock_builder.reset_mock()
			for method in ['with_structure', 'with_reference_structure', 'with_mobile_species',
						'with_structure_alignment', 'with_dynamic_voronoi_sites', 'build']:
				setattr(mock_builder, method, Mock(return_value=mock_builder))
			
			# Test with dynamic voronoi sites
			create_trajectory_with_dynamic_voronoi_sites(
				structure=Mock(spec=Structure),
				reference_structure=Mock(spec=Structure),
				mobile_species="Li",
				centre_species="O",
				reference_species="Li",
				cutoff=2.0,
				n_reference=4,
				label="test",
				align=True,
				align_species=["O"],
				align_metric="rmsd",
				align_algorithm=algorithm,
				align_minimizer_options=minimizer_options,
				align_tolerance=custom_tolerance  
			)
			
			# Verify with_structure_alignment was called with all parameters
			mock_builder.with_structure_alignment.assert_called_once()
			args, kwargs = mock_builder.with_structure_alignment.call_args
			self.assertEqual(kwargs['align_tolerance'], custom_tolerance)  
			
	def test_tolerance_passed_to_reference_based_sites(self):
		"""Test that align_tolerance is correctly passed to ReferenceBasedSites."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		
		# Set a custom tolerance
		custom_tolerance = 1e-5
		builder.with_structure_alignment(
			align=True, 
			align_species=["O"],
			align_tolerance=custom_tolerance
		)
		
		# Set up polyhedral sites
		builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to verify correct parameters are passed
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_polyhedral_sites.return_value = mock_sites
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify ReferenceBasedSites was created with the correct tolerance
			mock_rbs_class.assert_called_once()
			_, kwargs = mock_rbs_class.call_args
			self.assertEqual(kwargs["align_tolerance"], custom_tolerance)
			
	def test_tolerance_passed_to_reference_based_sites_for_dynamic_voronoi(self):
		"""Test that align_tolerance is correctly passed to ReferenceBasedSites for dynamic Voronoi sites."""
		# Configure the builder
		builder = TrajectoryBuilder()
		builder.with_structure(self.structure)
		builder.with_reference_structure(self.reference_structure)
		builder.with_mobile_species("Li")
		
		# Set a custom tolerance
		custom_tolerance = 1e-5
		builder.with_structure_alignment(
			align=True, 
			align_species=["O"],
			align_tolerance=custom_tolerance
		)
		
		# Set up dynamic Voronoi sites
		builder.with_dynamic_voronoi_sites(
			centre_species="Li",
			reference_species="O",
			cutoff=2.0,
			n_reference=4,
			label="tetrahedral"
		)
		
		# Mock ReferenceBasedSites to verify correct parameters are passed
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class, \
			patch('site_analysis.builders.atoms_from_structure'), \
			patch('site_analysis.builders.Trajectory'):
			
			# Configure mock to return a mock RBS instance
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			
			# Configure mock to return site objects
			mock_sites = [Mock(), Mock()]
			mock_rbs.create_dynamic_voronoi_sites.return_value = mock_sites
			
			# Call build to trigger site creation
			builder.build()
			
			# Verify ReferenceBasedSites was created with the correct tolerance
			mock_rbs_class.assert_called_once()
			_, kwargs = mock_rbs_class.call_args
			self.assertEqual(kwargs["align_tolerance"], custom_tolerance)
	
if __name__ == '__main__':
	unittest.main()