"""Unit tests for the SiteFactory class.

These tests verify that SiteFactory correctly creates site objects from
coordination environments.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from pymatgen.core import Structure, Lattice

from site_analysis.reference_workflow.site_factory import SiteFactory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site
from tests.test_helpers import environment_centres, recorded_future_warnings


class TestSiteFactory(unittest.TestCase):
	"""Test cases for the SiteFactory class."""
	
	def setUp(self):
		"""Set up test structures and environments."""
		# Reset Site._newid counter between tests
		Site._newid = 0
		
		# Create a simple test structure (coordinates chosen so all
		# 4-atom subsets form valid, non-degenerate tetrahedra)
		lattice = Lattice.cubic(5.0)
		species = ["Na", "Cl", "Na", "Cl", "Na", "Cl"]
		coords = [
			[0.1, 0.1, 0.1],
			[0.3, 0.1, 0.3],
			[0.1, 0.3, 0.3],
			[0.3, 0.3, 0.1],
			[0.2, 0.2, 0.4],
			[0.2, 0.4, 0.2],
		]
		self.structure = Structure(lattice, species, coords)
		self.site_factory = SiteFactory(self.structure)
		
		# Define some coordination environments as lists of atom indices
		self.tetrahedral_env = [[1, 2, 3, 4]]  # A tetrahedral environment
		self.multiple_envs = [
			[1, 2, 4, 5],  # First environment (4 atoms)
			[0, 1, 3, 4, 5]   # Second environment (5 atoms)
		]
	
	def test_validate_environments_valid(self):
		"""Test environment validation with valid environments."""
		# This should not raise any exception
		self.site_factory._validate_environments(self.tetrahedral_env)
		self.site_factory._validate_environments(self.multiple_envs)
	
	def test_validate_environments_invalid_type(self):
		"""Test environment validation with invalid environment type."""
		# Not a list
		with self.assertRaises(ValueError):
			self.site_factory._validate_environments(123)
		
		# Not a list of lists
		with self.assertRaises(ValueError):
			self.site_factory._validate_environments([1, 2, 3])
	
	def test_validate_environments_invalid_indices(self):
		"""Test environment validation with invalid indices."""
		factory = SiteFactory(self.structure)
		
		# String indices instead of integers
		with self.assertRaises(ValueError):
			factory._validate_environments([["a", "b", "c"]])
		
		# Indices out of range
		with self.assertRaises(ValueError):
			factory._validate_environments([[1, 2, 10]])  # 10 > 5, out of range
	
	def test_validate_labels_valid(self):
		"""Test label validation with valid labels."""
		# Valid cases
		self.site_factory._validate_labels(None, None, 2)  # No labels
		self.site_factory._validate_labels("label", None, 2)  # Single label
		self.site_factory._validate_labels(None, ["label1", "label2"], 2)  # Multiple labels
	
	def test_validate_labels_invalid(self):
		"""Test label validation with invalid labels."""
		# Both label and labels provided
		with self.assertRaises(ValueError):
			self.site_factory._validate_labels("label", ["label1", "label2"], 2)
		
		# Wrong number of labels
		with self.assertRaises(ValueError):
			self.site_factory._validate_labels(None, ["label1"], 2)  # Too few
		
		with self.assertRaises(ValueError):
			self.site_factory._validate_labels(None, ["label1", "label2", "label3"], 2)  # Too many
	
	def test_create_polyhedral_sites(self):
		"""Test creation of PolyhedralSite objects."""
		# Mock the PolyhedralSite class
		with patch('site_analysis.reference_workflow.site_factory.PolyhedralSite') as mock_class:
			# Configure mock to store the parameters it was called with
			mock_sites = []
			def side_effect(**kwargs):
				mock_site = Mock()
				# Store the kwargs used to create this mock
				mock_site.creation_kwargs = kwargs
				mock_sites.append(mock_site)
				return mock_site
			
			mock_class.side_effect = side_effect
			
			# Test with single environment
			sites = self.site_factory.create_polyhedral_sites(
				self.tetrahedral_env,
				reference_centers=environment_centres(self.structure, self.tetrahedral_env))
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 1)
			
			# Check that the vertex indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['vertex_indices'], [1, 2, 3, 4])
			
			# Check that no label was set
			self.assertEqual(mock_sites[0].creation_kwargs['label'], None)
			
			# Test with multiple environments and custom label
			mock_sites.clear()
			sites = self.site_factory.create_polyhedral_sites(
				self.multiple_envs, label="test_label",
				reference_centers=environment_centres(self.structure, self.multiple_envs))
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 2)
			
			# Check that vertex indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['vertex_indices'], [1, 2, 4, 5])
			self.assertEqual(mock_sites[1].creation_kwargs['vertex_indices'], [0, 1, 3, 4, 5])
			
			# Check that labels are set correctly
			self.assertEqual(mock_sites[0].creation_kwargs['label'], "test_label")
			self.assertEqual(mock_sites[1].creation_kwargs['label'], "test_label")
	
	def test_create_dynamic_voronoi_sites(self):
		"""Test creation of DynamicVoronoiSite objects."""
		
		# Mock the DynamicVoronoiSite class
		with patch('site_analysis.reference_workflow.site_factory.DynamicVoronoiSite') as mock_class:
			# Configure mock to store the parameters it was called with
			mock_sites = []
			def side_effect(**kwargs):
				mock_site = Mock()
				# Store the kwargs used to create this mock
				mock_site.creation_kwargs = kwargs
				mock_sites.append(mock_site)
				return mock_site
			
			mock_class.side_effect = side_effect
			
			# Test with single environment
			sites = self.site_factory.create_dynamic_voronoi_sites(
				self.tetrahedral_env,
				reference_centers=environment_centres(self.structure, self.tetrahedral_env))
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 1)
			
			# Check that the reference indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['reference_indices'], [1, 2, 3, 4])
			
			# Check that no label was set
			self.assertEqual(mock_sites[0].creation_kwargs['label'], None)
			
			# Test with multiple environments and multiple labels
			mock_sites.clear()
			sites = self.site_factory.create_dynamic_voronoi_sites(
				self.multiple_envs, labels=["label1", "label2"],
				reference_centers=environment_centres(self.structure, self.multiple_envs)
			)
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 2)
			
			# Check that reference indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['reference_indices'], [1, 2, 4, 5])
			self.assertEqual(mock_sites[1].creation_kwargs['reference_indices'], [0, 1, 3, 4, 5])
			
			# Check that labels are set correctly
			self.assertEqual(mock_sites[0].creation_kwargs['label'], "label1")
			self.assertEqual(mock_sites[1].creation_kwargs['label'], "label2")
	
	def test_assign_vertex_coords(self):
		"""Test assignment of vertex coordinates to PolyhedralSite."""
		# Create a mock PolyhedralSite
		mock_site = Mock(spec=PolyhedralSite)
		
		# Call the method directly with the mock
		self.site_factory._assign_vertex_coords(mock_site)
		
		# Verify method was called with correct arguments
		call_args = mock_site.assign_vertex_coords.call_args
		np.testing.assert_array_equal(call_args[0][0], self.structure.frac_coords)
		np.testing.assert_array_equal(call_args[0][1], self.structure.lattice.matrix)
	
	def test_vertex_coords_assignment(self):
		"""Test that vertex coordinates are correctly assigned for PolyhedralSite."""
		# Mock the PolyhedralSite class
		with patch('site_analysis.reference_workflow.site_factory.PolyhedralSite') as mock_class:
			# Return a mock site with a mock assign_vertex_coords method
			mock_site = Mock(spec=PolyhedralSite)
			mock_class.return_value = mock_site
			
			# Test creating a single site
			self.site_factory.create_polyhedral_sites(
				self.tetrahedral_env,
				reference_centers=environment_centres(self.structure, self.tetrahedral_env))
			
			# Verify assign_vertex_coords was called with the correct arrays
			call_args = mock_site.assign_vertex_coords.call_args
			np.testing.assert_array_equal(call_args[0][0], self.structure.frac_coords)
			np.testing.assert_array_equal(call_args[0][1], self.structure.lattice.matrix)
	
	def test_minimum_vertices_for_polyhedral_sites(self):
		"""Test minimum vertices validation for PolyhedralSite."""
		# Not enough vertices for a polyhedron (need at least 3)
		with self.assertRaises(ValueError):
			self.site_factory.create_polyhedral_sites([[1, 2]])
	
	def test_both_labels_error(self):
		"""Test error when both label and labels are provided."""
		with self.assertRaises(ValueError):
			self.site_factory.create_polyhedral_sites(
				self.multiple_envs,
				label="single_label",
				labels=["label1", "label2"]
			)
		
		with self.assertRaises(ValueError):
			self.site_factory.create_dynamic_voronoi_sites(
				self.multiple_envs,
				label="single_label",
				labels=["label1", "label2"]
			)
	
	def test_empty_environments(self):
		"""Test behavior with empty environments list."""
		# Mock the site classes to ensure we're testing in isolation
		with patch('site_analysis.reference_workflow.site_factory.PolyhedralSite'):
			# Create sites with empty environments
			sites = self.site_factory.create_polyhedral_sites([], reference_centers=[])
			self.assertEqual(len(sites), 0)
		
		with patch('site_analysis.reference_workflow.site_factory.DynamicVoronoiSite'):
			# Create sites with empty environments
			sites = self.site_factory.create_dynamic_voronoi_sites([], reference_centers=[])
			self.assertEqual(len(sites), 0)
			
	def test_create_polyhedral_sites_with_reference_centers(self):
		"""Test creating polyhedral sites with reference centres."""
		environments = [[0, 1, 2, 3], [1, 2, 3, 4]]
		reference_centers = [
			np.array([0.1, 0.1, 0.1]),
			np.array([0.2, 0.2, 0.2])
		]
		
		sites = self.site_factory.create_polyhedral_sites(
			environments,
			reference_centers=reference_centers,
			label="test_label"
		)
		
		# Check that sites were created with reference centres
		self.assertEqual(len(sites), 2)
		
		for site, expected_centre in zip(sites, reference_centers):
			self.assertIsInstance(site, PolyhedralSite)
			self.assertIsNotNone(site.reference_center)
			np.testing.assert_array_equal(site.reference_center, expected_centre)
	
	def test_create_polyhedral_sites_without_reference_centers(self):
		"""Creating polyhedral sites without reference centres is deprecated, warning once per call."""
		environments = [[1, 2, 3, 4], [0, 1, 2, 3]]
		
		with recorded_future_warnings() as caught:
			sites = self.site_factory.create_polyhedral_sites(environments, label="test_label")
		
		self.assertEqual(len(caught), 1)
		self.assertRegex(
			str(caught[0].message),
			r"SiteFactory\.create_polyhedral_sites without reference_centers is deprecated "
			r"and will be removed in version 2\.0; use reference_centers")
		self.assertEqual(caught[0].filename, __file__)

		# Check that sites were created without reference centres
		self.assertEqual(len(sites), 2)
		
		for site in sites:
			self.assertIsInstance(site, PolyhedralSite)
			self.assertIsNone(site.reference_center)
	
	def test_create_dynamic_voronoi_sites_with_reference_centers(self):
		"""Test creating dynamic voronoi sites with reference centres."""
		environments = [[1, 2], [3, 4]]
		reference_centers = [
			np.array([0.3, 0.3, 0.3]),
			np.array([0.4, 0.4, 0.4])
		]
		
		sites = self.site_factory.create_dynamic_voronoi_sites(
			environments,
			reference_centers=reference_centers,
			label="test_label"
		)
		
		# Check that sites were created with reference centres
		self.assertEqual(len(sites), 2)
		
		for site, expected_centre in zip(sites, reference_centers):
			self.assertIsInstance(site, DynamicVoronoiSite)
			self.assertIsNotNone(site.reference_center)
			np.testing.assert_array_equal(site.reference_center, expected_centre)
	
	def test_create_dynamic_voronoi_sites_without_reference_centers(self):
		"""Creating dynamic Voronoi sites without reference centres is deprecated, warning once per call."""
		environments = [[1, 2], [3, 4]]
		
		with recorded_future_warnings() as caught:
			sites = self.site_factory.create_dynamic_voronoi_sites(environments, label="test_label")
		
		self.assertEqual(len(caught), 1)
		self.assertRegex(
			str(caught[0].message),
			r"SiteFactory\.create_dynamic_voronoi_sites without reference_centers is "
			r"deprecated and will be removed in version 2\.0; use reference_centers")
		self.assertEqual(caught[0].filename, __file__)

		# Check that sites were created without reference centres
		self.assertEqual(len(sites), 2)
		
		for site in sites:
			self.assertIsInstance(site, DynamicVoronoiSite)
			self.assertIsNone(site.reference_center)
	
	def test_create_sites_with_reference_centers_does_not_warn(self):
		"""Creating sites with reference centres issues no warning."""
		environments = [[0, 1, 2, 3], [1, 2, 3, 4]]
		reference_centers = [np.array([0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2])]
		with recorded_future_warnings() as caught:
			self.site_factory.create_polyhedral_sites(environments, reference_centers=reference_centers)
			self.site_factory.create_dynamic_voronoi_sites(
				environments, reference_centers=reference_centers)
		self.assertEqual(caught, [])

	def test_create_polyhedral_sites_mismatched_lengths_raises_error(self):
		"""Test that mismatched lengths between environments and reference centres raises error."""
		environments = [[0, 1, 2, 3], [1, 2, 3, 4]]  # 2 environments
		reference_centers = [np.array([0.1, 0.1, 0.1])]  # 1 reference centre
		
		with self.assertRaises(ValueError) as context:
			self.site_factory.create_polyhedral_sites(
				environments,
				reference_centers=reference_centers
			)
		
		self.assertIn("reference_centers", str(context.exception))
		self.assertIn("environments", str(context.exception))
		
	def test_create_dynamic_voronoi_sites_mismatched_lengths_raises_error(self):
		"""Test that mismatched lengths between environments and reference centres raises error."""
		environments = [[1, 2], [3, 4]]  # 2 environments
		reference_centers = [np.array([0.1, 0.1, 0.1])]  # 1 reference centre
		
		with self.assertRaises(ValueError) as context:
			self.site_factory.create_dynamic_voronoi_sites(
				environments,
				reference_centers=reference_centers
			)
		
		self.assertIn("reference_centers", str(context.exception))
		self.assertIn("environments", str(context.exception))


	def test_create_sites_with_a_none_reference_centre_raises_error(self):
		"""A None entry in reference_centers raises ValueError naming its index."""
		environments = [[0, 1, 2, 3], [1, 2, 3, 4]]
		reference_centers = [np.array([0.1, 0.1, 0.1]), None]
		for create in (self.site_factory.create_polyhedral_sites,
				self.site_factory.create_dynamic_voronoi_sites):
			with self.subTest(create=create.__name__):
				with self.assertRaisesRegex(ValueError, r"reference_centers\[1\]"):
					create(environments, reference_centers=reference_centers)


if __name__ == '__main__':
	unittest.main()