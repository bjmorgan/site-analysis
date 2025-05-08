import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.builders import (
	TrajectoryBuilder,
	create_trajectory_with_spherical_sites,
	create_trajectory_with_voronoi_sites,
	create_trajectory_with_polyhedral_sites
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
		self.assertIsNone(builder._site_generator)
	
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
		self.assertIsNotNone(builder._site_generator)
		self.assertTrue(callable(builder._site_generator))
	
	def test_with_voronoi_sites_sets_generator(self):
		"""Test that with_voronoi_sites sets a site generator function."""
		# Call the method
		builder = self.builder.with_voronoi_sites(
			centres=self.centres,
			labels=self.labels
		)
		
		# Verify a site generator was set
		self.assertIsNotNone(builder._site_generator)
		self.assertTrue(callable(builder._site_generator))
	
	def test_with_existing_sites_sets_generator(self):
		"""Test that with_existing_sites sets a site generator function."""
		mock_sites = [Mock(), Mock()]
		
		# Call the method
		builder = self.builder.with_existing_sites(mock_sites)
		
		# Verify a site generator was set
		self.assertIsNotNone(builder._site_generator)
		self.assertTrue(callable(builder._site_generator))
	
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
		self.assertIsNotNone(builder._site_generator)
		self.assertTrue(callable(builder._site_generator))
	
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
		self.assertIsNotNone(builder._site_generator)
		self.assertTrue(callable(builder._site_generator))
	
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
				label="tetrahedral"
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
		
	def test_with_alignment_options(self):
		"""Test setting alignment options."""
		# Start with a fresh builder
		builder = TrajectoryBuilder()
		
		# Set alignment options with a list of species
		result = builder.with_alignment_options(
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
		builder.with_alignment_options(
			align=True,
			align_species="Na",
			align_metric="rmsd"
		)
		
		# Verify single species string is converted to list
		self.assertEqual(builder._align, True)
		self.assertEqual(builder._align_species, ["Na"])
		self.assertEqual(builder._align_metric, "rmsd")
		
		# Test with default align value
		builder.with_alignment_options(
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
			
			# Verify analyse_structure was called
			mock_trajectory.analyse_structure.assert_called_once_with(self.structure)
			
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

if __name__ == '__main__':
	unittest.main()