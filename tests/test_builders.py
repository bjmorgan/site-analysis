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
		self.assertIsNone(builder._sites)
		self.assertIsNone(builder._atoms)
	
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
	
	@patch('site_analysis.builders.SphericalSite')
	def test_with_spherical_sites(self, mock_spherical_site):
		"""Test building with spherical sites."""
		# Configure the mock
		mock_site1 = Mock()
		mock_site2 = Mock()
		mock_spherical_site.side_effect = [mock_site1, mock_site2]
		
		# Call the method
		self.builder.with_spherical_sites(
			centres=self.centres,
			radii=self.radii,
			labels=self.labels
		)
		
		# Verify SphericalSite was called correctly
		self.assertEqual(mock_spherical_site.call_count, 2)
		
		# First call should use first values
		args1, kwargs1 = mock_spherical_site.call_args_list[0]
		np.testing.assert_array_equal(kwargs1['frac_coords'], np.array(self.centres[0]))
		self.assertEqual(kwargs1['rcut'], self.radii[0])
		self.assertEqual(kwargs1['label'], self.labels[0])
		
		# Second call should use second values
		args2, kwargs2 = mock_spherical_site.call_args_list[1]
		np.testing.assert_array_equal(kwargs2['frac_coords'], np.array(self.centres[1]))
		self.assertEqual(kwargs2['rcut'], self.radii[1])
		self.assertEqual(kwargs2['label'], self.labels[1])
		
		# Sites should be stored in the builder
		self.assertEqual(self.builder._sites, [mock_site1, mock_site2])
	
	def test_with_spherical_sites_length_mismatch(self):
		"""Test validation for centres and radii length mismatch."""
		# Mismatch lengths
		with self.assertRaises(ValueError) as context:
			self.builder.with_spherical_sites(
				centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
				radii=[1.0]  # Only one radius for two centres
			)
		
		# Check error message
		self.assertIn("match", str(context.exception).lower())
	
	@patch('site_analysis.builders.VoronoiSite')
	def test_with_voronoi_sites(self, mock_voronoi_site):
		"""Test building with Voronoi sites."""
		# Configure the mock
		mock_site1 = Mock()
		mock_site2 = Mock()
		mock_voronoi_site.side_effect = [mock_site1, mock_site2]
		
		# Call the method
		self.builder.with_voronoi_sites(
			centres=self.centres,
			labels=self.labels
		)
		
		# Verify VoronoiSite was called correctly
		self.assertEqual(mock_voronoi_site.call_count, 2)
		
		# First call should use first values
		args1, kwargs1 = mock_voronoi_site.call_args_list[0]
		np.testing.assert_array_equal(kwargs1['frac_coords'], np.array(self.centres[0]))
		self.assertEqual(kwargs1['label'], self.labels[0])
		
		# Second call should use second values
		args2, kwargs2 = mock_voronoi_site.call_args_list[1]
		np.testing.assert_array_equal(kwargs2['frac_coords'], np.array(self.centres[1]))
		self.assertEqual(kwargs2['label'], self.labels[1])
		
		# Sites should be stored in the builder
		self.assertEqual(self.builder._sites, [mock_site1, mock_site2])
	
	@patch('site_analysis.builders.ReferenceBasedSites')
	def test_with_polyhedral_sites(self, mock_rbs_class):
		"""Test building with polyhedral sites."""
		# Configure the mock ReferenceBasedSites
		mock_rbs = Mock()
		mock_rbs_class.return_value = mock_rbs
		
		# Mock the sites that would be returned
		mock_sites = [Mock(), Mock()]
		mock_rbs.create_polyhedral_sites.return_value = mock_sites
		
		# Configure the builder with structures
		self.builder.with_structure(self.structure)
		self.builder.with_reference_structure(self.reference_structure)
		
		# Call the method
		self.builder.with_polyhedral_sites(
			centre_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Verify ReferenceBasedSites was created correctly
		mock_rbs_class.assert_called_once_with(
			reference_structure=self.reference_structure,
			target_structure=self.structure
		)
		
		# Verify create_polyhedral_sites was called correctly
		mock_rbs.create_polyhedral_sites.assert_called_once_with(
			center_species="Li",
			vertex_species="O",
			cutoff=2.0,
			n_vertices=4,
			label="tetrahedral"
		)
		
		# Sites should be stored in the builder
		self.assertEqual(self.builder._sites, mock_sites)
	
	def test_with_polyhedral_sites_requires_structures(self):
		"""Test validation of reference and target structures for polyhedral sites."""
		# Try to create sites without setting structures
		with self.assertRaises(ValueError) as context:
			self.builder.with_polyhedral_sites(
				centre_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4
			)
		
		# Check error message
		self.assertIn("structure", str(context.exception).lower())
	
	def test_with_polyhedral_sites(self):
		"""Test building with polyhedral sites."""
		# Configure the mock ReferenceBasedSites
		mock_rbs = Mock()
		mock_sites = [Mock(), Mock()]
		mock_rbs.create_polyhedral_sites.return_value = mock_sites
		
		# Create a mock for ReferenceBasedSites class
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class:
			# Configure mock to return our mock_rbs instance
			mock_rbs_class.return_value = mock_rbs
			
			# Configure the builder with structures
			self.builder.with_structure(self.structure)
			self.builder.with_reference_structure(self.reference_structure)
			
			# Call the method
			self.builder.with_polyhedral_sites(
				centre_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4,
				label="tetrahedral"
			)
			
			# Verify ReferenceBasedSites was created correctly with alignment parameters
			mock_rbs_class.assert_called_once_with(
				reference_structure=self.reference_structure,
				target_structure=self.structure,
				align=True,                # Default is True
				align_species=None,        # Default is None
				align_metric='rmsd'        # Default is 'rmsd'
			)
			
			# Verify create_polyhedral_sites was called correctly
			mock_rbs.create_polyhedral_sites.assert_called_once_with(
				center_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4,
				label="tetrahedral"
			)
			
			# Sites should be stored in the builder
			self.assertEqual(self.builder._sites, mock_sites)
	
	def test_with_dynamic_voronoi_sites(self):
		"""Test building with dynamic Voronoi sites."""
		# Configure the mock ReferenceBasedSites
		mock_rbs = Mock()
		mock_sites = [Mock(), Mock()]
		mock_rbs.create_dynamic_voronoi_sites.return_value = mock_sites
		
		# Create a mock for ReferenceBasedSites class
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class:
			# Configure mock to return our mock_rbs instance
			mock_rbs_class.return_value = mock_rbs
			
			# Configure the builder with structures
			self.builder.with_structure(self.structure)
			self.builder.with_reference_structure(self.reference_structure)
			
			# Call the method
			self.builder.with_dynamic_voronoi_sites(
				centre_species="Li",
				reference_species="O",
				cutoff=2.0,
				n_reference=4,
				label="tetrahedral"
			)
			
			# Verify ReferenceBasedSites was created correctly with alignment parameters
			mock_rbs_class.assert_called_once_with(
				reference_structure=self.reference_structure,
				target_structure=self.structure,
				align=True,                # Default is True
				align_species=None,        # Default is None
				align_metric='rmsd'        # Default is 'rmsd'
			)
			
			# Verify create_dynamic_voronoi_sites was called correctly
			mock_rbs.create_dynamic_voronoi_sites.assert_called_once_with(
				center_species="Li",
				reference_species="O",
				cutoff=2.0,
				n_reference=4,
				label="tetrahedral"
			)
			
			# Sites should be stored in the builder
			self.assertEqual(self.builder._sites, mock_sites)
	
	def test_with_alignment_options(self):
		"""Test setting alignment options."""
		# Start with a fresh builder
		builder = TrajectoryBuilder()
		
		# Set alignment options
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
		
		# Test with polyhedral sites to verify options are passed correctly
		with patch('site_analysis.builders.ReferenceBasedSites') as mock_rbs_class:
			mock_rbs = Mock()
			mock_rbs_class.return_value = mock_rbs
			mock_rbs.create_polyhedral_sites.return_value = [Mock()]
			
			# Configure the builder
			builder.with_structure(self.structure)
			builder.with_reference_structure(self.reference_structure)
			
			# Call with_polyhedral_sites
			builder.with_polyhedral_sites(
				centre_species="Li",
				vertex_species="O",
				cutoff=2.0,
				n_vertices=4
			)
			
			# Verify ReferenceBasedSites was created with our custom alignment options
			mock_rbs_class.assert_called_once_with(
				reference_structure=self.reference_structure,
				target_structure=self.structure,
				align=False,
				align_species=["Li"],
				align_metric="max_dist"
			)
	
	def test_with_dynamic_voronoi_sites_requires_structures(self):
		"""Test validation of reference and target structures for dynamic Voronoi sites."""
		# Try to create sites without setting structures
		with self.assertRaises(ValueError) as context:
			self.builder.with_dynamic_voronoi_sites(
				centre_species="Li",
				reference_species="O",
				cutoff=2.0,
				n_reference=4
			)
		
		# Check error message
		self.assertIn("structure", str(context.exception).lower())
	
	def test_with_existing_sites(self):
		"""Test using existing site objects."""
		# Create mock sites
		mock_sites = [Mock(), Mock()]
		
		# Call the method
		self.builder.with_existing_sites(mock_sites)
		
		# Sites should be stored directly
		self.assertEqual(self.builder._sites, mock_sites)
	
	def test_with_existing_atoms(self):
		"""Test using existing atom objects."""
		# Create mock atoms
		mock_atoms = [Mock(), Mock()]
		
		# Call the method
		self.builder.with_existing_atoms(mock_atoms)
		
		# Atoms should be stored directly
		self.assertEqual(self.builder._atoms, mock_atoms)
	
	@patch('site_analysis.builders.atoms_from_structure')
	@patch('site_analysis.builders.Trajectory')
	def test_build_complete(self, mock_trajectory_class, mock_atoms_from_structure):
		"""Test building a trajectory with all required components."""
		# Configure the mocks
		mock_atoms = [Mock(), Mock()]
		mock_atoms_from_structure.return_value = mock_atoms
		
		mock_trajectory = Mock()
		mock_trajectory_class.return_value = mock_trajectory
		
		# Configure the builder
		mock_sites = [Mock(), Mock()]
		self.builder.with_structure(self.structure)
		self.builder.with_mobile_species("Li")
		self.builder.with_existing_sites(mock_sites)
		
		# Call build
		result = self.builder.build()
		
		# Verify atoms were created
		mock_atoms_from_structure.assert_called_once_with(self.structure, "Li")
		
		# Verify trajectory was created
		mock_trajectory_class.assert_called_once_with(sites=mock_sites, atoms=mock_atoms)
		
		# Verify analyse_structure was called
		mock_trajectory.analyse_structure.assert_called_once_with(self.structure)
		
		# Result should be the trajectory
		self.assertEqual(result, mock_trajectory)
	
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
		
		# Verify trajectory was created with existing atoms
		mock_trajectory_class.assert_called_once_with(sites=mock_sites, atoms=mock_atoms)
		
		# Verify analyse_structure was called
		mock_trajectory.analyse_structure.assert_called_once_with(self.structure)
		
		# Result should be the trajectory
		self.assertEqual(result, mock_trajectory)
	
	def test_build_requires_structure(self):
		"""Test validation that structure is set before building."""
		# Configure builder without structure
		self.builder.with_mobile_species("Li")
		self.builder.with_existing_sites([Mock()])
		
		# Build should fail
		with self.assertRaises(ValueError) as context:
			self.builder.build()
		
		# Check error message
		self.assertIn("structure must be set", str(context.exception).lower())
	
	def test_build_requires_sites(self):
		"""Test validation that sites are defined before building."""
		# Configure builder without sites
		self.builder.with_structure(self.structure)
		self.builder.with_mobile_species("Li")
		
		# Build should fail
		with self.assertRaises(ValueError) as context:
			self.builder.build()
		
		# Check error message
		self.assertIn("sites must be defined", str(context.exception).lower())
	
	def test_build_requires_mobile_species(self):
		"""Test validation that mobile species is set before building with auto atoms."""
		# Configure builder without mobile species
		self.builder.with_structure(self.structure)
		self.builder.with_existing_sites([Mock()])
		
		# Build should fail
		with self.assertRaises(ValueError) as context:
			self.builder.build()
		
		# Check error message
		self.assertIn("mobile species must be set", str(context.exception).lower())


class TestFactoryFunctions(unittest.TestCase):
	"""Tests for the factory functions that use TrajectoryBuilder."""
	
	def setUp(self):
		"""Set up test fixtures."""
		# Create a simple test structure
		self.lattice = Lattice.cubic(5.0)
		self.structure = Structure(
			lattice=self.lattice,
			species=["Li", "O", "Li", "O"],
			coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]]
		)
		
		# Reference structure
		self.reference_structure = Structure(
			lattice=self.lattice,
			species=["Li", "O", "Li", "O"],
			coords=[[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.6, 0.1, 0.1], [0.1, 0.6, 0.6]]
		)
	
	@patch('site_analysis.builders.TrajectoryBuilder')
	def test_create_trajectory_with_spherical_sites(self, mock_builder_class):
		"""Test factory function for creating trajectory with spherical sites."""
		# Configure the mock builder
		mock_builder = Mock()
		mock_builder_class.return_value = mock_builder
		
		# Configure builder method chaining
		mock_builder.with_structure.return_value = mock_builder
		mock_builder.with_mobile_species.return_value = mock_builder
		mock_builder.with_spherical_sites.return_value = mock_builder
		
		# Mock the final trajectory
		mock_trajectory = Mock()
		mock_builder.build.return_value = mock_trajectory
		
		# Call the factory function
		centres = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
		radii = [1.0, 1.5]
		labels = ["octahedral", "tetrahedral"]
		
		result = create_trajectory_with_spherical_sites(
			self.structure,
			"Li",
			centres,
			radii,
			labels
		)
		
		# Verify builder was configured correctly
		mock_builder.with_structure.assert_called_once_with(self.structure)
		mock_builder.with_mobile_species.assert_called_once_with("Li")
		mock_builder.with_spherical_sites.assert_called_once_with(centres, radii, labels)
		mock_builder.build.assert_called_once()
		
		# Result should be the built trajectory
		self.assertEqual(result, mock_trajectory)
	
	@patch('site_analysis.builders.TrajectoryBuilder')
	def test_create_trajectory_with_voronoi_sites(self, mock_builder_class):
		"""Test factory function for creating trajectory with Voronoi sites."""
		# Configure the mock builder
		mock_builder = Mock()
		mock_builder_class.return_value = mock_builder
		
		# Configure builder method chaining
		mock_builder.with_structure.return_value = mock_builder
		mock_builder.with_mobile_species.return_value = mock_builder
		mock_builder.with_voronoi_sites.return_value = mock_builder
		
		# Mock the final trajectory
		mock_trajectory = Mock()
		mock_builder.build.return_value = mock_trajectory
		
		# Call the factory function
		centres = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
		labels = ["octahedral", "tetrahedral"]
		
		result = create_trajectory_with_voronoi_sites(
			self.structure,
			"Li",
			centres,
			labels
		)
		
		# Verify builder was configured correctly
		mock_builder.with_structure.assert_called_once_with(self.structure)
		mock_builder.with_mobile_species.assert_called_once_with("Li")
		mock_builder.with_voronoi_sites.assert_called_once_with(centres, labels)
		mock_builder.build.assert_called_once()
		
		# Result should be the built trajectory
		self.assertEqual(result, mock_trajectory)
	
	@patch('site_analysis.builders.TrajectoryBuilder')
	def test_create_trajectory_with_polyhedral_sites(self, mock_builder_class):
		"""Test factory function for creating trajectory with polyhedral sites."""
		# Configure the mock builder
		mock_builder = Mock()
		mock_builder_class.return_value = mock_builder
		
		# Configure builder method chaining
		mock_builder.with_structure.return_value = mock_builder
		mock_builder.with_reference_structure.return_value = mock_builder
		mock_builder.with_mobile_species.return_value = mock_builder
		mock_builder.with_alignment_options.return_value = mock_builder
		mock_builder.with_polyhedral_sites.return_value = mock_builder
		
		# Mock the final trajectory
		mock_trajectory = Mock()
		mock_builder.build.return_value = mock_trajectory
		
		# Call the factory function
		result = create_trajectory_with_polyhedral_sites(
			self.structure,
			self.reference_structure,
			"Li",
			"O",
			"Li",
			2.0,
			4,
			"tetrahedral"
		)
		
		# Verify builder was configured correctly
		mock_builder.with_structure.assert_called_once_with(self.structure)
		mock_builder.with_reference_structure.assert_called_once_with(self.reference_structure)
		mock_builder.with_mobile_species.assert_called_once_with("Li")
		
		# Verify alignment options were set with defaults
		mock_builder.with_alignment_options.assert_called_once_with(True, None, 'rmsd')
		
		# Verify polyhedral sites were created
		mock_builder.with_polyhedral_sites.assert_called_once_with("O", "Li", 2.0, 4, "tetrahedral")
		mock_builder.build.assert_called_once()
		
		# Result should be the built trajectory
		self.assertEqual(result, mock_trajectory)


if __name__ == '__main__':
	unittest.main()