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


class TestSiteFactory(unittest.TestCase):
	"""Test cases for the SiteFactory class."""
	
	def setUp(self):
		"""Set up test structures and environments."""
		# Reset Site._newid counter between tests
		Site._newid = 0
		
		# Create a simple test structure
		lattice = Lattice.cubic(5.0)
		species = ["Na", "Cl", "Na", "Cl", "Na"]
		coords = [
			[0.0, 0.0, 0.0],  # Na0
			[0.2, 0.2, 0.2],  # Cl1
			[0.5, 0.0, 0.0],  # Na2
			[0.7, 0.2, 0.2],  # Cl3
			[0.0, 0.5, 0.0],  # Na4
		]
		self.structure = Structure(lattice, species, coords)
		
		# Define some coordination environments as lists of atom indices
		self.tetrahedral_env = [[1, 2, 3, 4]]  # A tetrahedral environment
		self.multiple_envs = [
			[1, 2, 4],     # First environment (3 atoms)
			[0, 1, 3, 4]   # Second environment (4 atoms)
		]
	
	def test_validate_environments_valid(self):
		"""Test environment validation with valid environments."""
		factory = SiteFactory(self.structure)
		
		# This should not raise any exception
		factory._validate_environments(self.tetrahedral_env)
		factory._validate_environments(self.multiple_envs)
	
	def test_validate_environments_invalid_type(self):
		"""Test environment validation with invalid environment type."""
		factory = SiteFactory(self.structure)
		
		# Not a list
		with self.assertRaises(ValueError):
			factory._validate_environments(123)
		
		# Not a list of lists
		with self.assertRaises(ValueError):
			factory._validate_environments([1, 2, 3])
	
	def test_validate_environments_invalid_indices(self):
		"""Test environment validation with invalid indices."""
		factory = SiteFactory(self.structure)
		
		# String indices instead of integers
		with self.assertRaises(ValueError):
			factory._validate_environments([["a", "b", "c"]])
		
		# Indices out of range
		with self.assertRaises(ValueError):
			factory._validate_environments([[1, 2, 10]])  # 10 is out of range
	
	def test_validate_labels_valid(self):
		"""Test label validation with valid labels."""
		factory = SiteFactory(self.structure)
		
		# Valid cases
		factory._validate_labels(None, None, 2)  # No labels
		factory._validate_labels("label", None, 2)  # Single label
		factory._validate_labels(None, ["label1", "label2"], 2)  # Multiple labels
	
	def test_validate_labels_invalid(self):
		"""Test label validation with invalid labels."""
		factory = SiteFactory(self.structure)
		
		# Both label and labels provided
		with self.assertRaises(ValueError):
			factory._validate_labels("label", ["label1", "label2"], 2)
		
		# Wrong number of labels
		with self.assertRaises(ValueError):
			factory._validate_labels(None, ["label1"], 2)  # Too few
		
		with self.assertRaises(ValueError):
			factory._validate_labels(None, ["label1", "label2", "label3"], 2)  # Too many
	
	def test_create_polyhedral_sites(self):
		"""Test creation of PolyhedralSite objects."""
		factory = SiteFactory(self.structure)
		
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
			sites = factory.create_polyhedral_sites(self.tetrahedral_env)
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 1)
			
			# Check that the vertex indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['vertex_indices'], [1, 2, 3, 4])
			
			# Check that no label was set
			self.assertEqual(mock_sites[0].creation_kwargs['label'], None)
			
			# Test with multiple environments and custom label
			mock_sites.clear()
			sites = factory.create_polyhedral_sites(self.multiple_envs, label="test_label")
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 2)
			
			# Check that vertex indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['vertex_indices'], [1, 2, 4])
			self.assertEqual(mock_sites[1].creation_kwargs['vertex_indices'], [0, 1, 3, 4])
			
			# Check that labels are set correctly
			self.assertEqual(mock_sites[0].creation_kwargs['label'], "test_label")
			self.assertEqual(mock_sites[1].creation_kwargs['label'], "test_label")
	
	def test_create_dynamic_voronoi_sites(self):
		"""Test creation of DynamicVoronoiSite objects."""
		factory = SiteFactory(self.structure)
		
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
			sites = factory.create_dynamic_voronoi_sites(self.tetrahedral_env)
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 1)
			
			# Check that the reference indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['reference_indices'], [1, 2, 3, 4])
			
			# Check that no label was set
			self.assertEqual(mock_sites[0].creation_kwargs['label'], None)
			
			# Test with multiple environments and multiple labels
			mock_sites.clear()
			sites = factory.create_dynamic_voronoi_sites(
				self.multiple_envs, labels=["label1", "label2"]
			)
			
			# Check that we got the right number of sites
			self.assertEqual(len(sites), 2)
			
			# Check that reference indices are correct
			self.assertEqual(mock_sites[0].creation_kwargs['reference_indices'], [1, 2, 4])
			self.assertEqual(mock_sites[1].creation_kwargs['reference_indices'], [0, 1, 3, 4])
			
			# Check that labels are set correctly
			self.assertEqual(mock_sites[0].creation_kwargs['label'], "label1")
			self.assertEqual(mock_sites[1].creation_kwargs['label'], "label2")
	
	def test_assign_vertex_coords(self):
		"""Test assignment of vertex coordinates to PolyhedralSite."""
		factory = SiteFactory(self.structure)
		
		# Create a mock PolyhedralSite
		mock_site = Mock(spec=PolyhedralSite)
		
		# Call the method directly with the mock
		factory._assign_vertex_coords(mock_site)
		
		# Verify method was called with correct argument
		mock_site.assign_vertex_coords.assert_called_once_with(self.structure)
	
	def test_vertex_coords_assignment(self):
		"""Test that vertex coordinates are correctly assigned for PolyhedralSite."""
		factory = SiteFactory(self.structure)
		
		# Mock the PolyhedralSite class
		with patch('site_analysis.reference_workflow.site_factory.PolyhedralSite') as mock_class:
			# Return a mock site with a mock assign_vertex_coords method
			mock_site = Mock(spec=PolyhedralSite)
			mock_class.return_value = mock_site
			
			# Test creating a single site
			factory.create_polyhedral_sites(self.tetrahedral_env)
			
			# Verify assign_vertex_coords was called with the correct structure
			mock_site.assign_vertex_coords.assert_called_once_with(self.structure)
	
	def test_minimum_vertices_for_polyhedral_sites(self):
		"""Test minimum vertices validation for PolyhedralSite."""
		factory = SiteFactory(self.structure)
		
		# Not enough vertices for a polyhedron (need at least 3)
		with self.assertRaises(ValueError):
			factory.create_polyhedral_sites([[1, 2]])
	
	def test_both_labels_error(self):
		"""Test error when both label and labels are provided."""
		factory = SiteFactory(self.structure)
		
		with self.assertRaises(ValueError):
			factory.create_polyhedral_sites(
				self.multiple_envs,
				label="single_label",
				labels=["label1", "label2"]
			)
		
		with self.assertRaises(ValueError):
			factory.create_dynamic_voronoi_sites(
				self.multiple_envs,
				label="single_label",
				labels=["label1", "label2"]
			)
	
	def test_empty_environments(self):
		"""Test behavior with empty environments list."""
		factory = SiteFactory(self.structure)
		
		# Mock the site classes to ensure we're testing in isolation
		with patch('site_analysis.reference_workflow.site_factory.PolyhedralSite'):
			# Create sites with empty environments
			sites = factory.create_polyhedral_sites([])
			self.assertEqual(len(sites), 0)
		
		with patch('site_analysis.reference_workflow.site_factory.DynamicVoronoiSite'):
			# Create sites with empty environments
			sites = factory.create_dynamic_voronoi_sites([])
			self.assertEqual(len(sites), 0)


if __name__ == '__main__':
	unittest.main()