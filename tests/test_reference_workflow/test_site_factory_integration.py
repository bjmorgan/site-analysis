"""Integration tests for the SiteFactory class.

These tests verify that SiteFactory correctly instantiates site objects
with the expected parameters. They do not test the behavior of the
site classes themselves, as that is covered in the site class tests.
"""

import unittest
from pymatgen.core import Structure, Lattice

from site_analysis.reference_workflow.site_factory import SiteFactory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site


class TestSiteFactoryIntegration(unittest.TestCase):
	"""Integration tests for the SiteFactory class."""
	
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
	
	def test_create_polyhedral_sites_integration(self):
		"""Test creation of PolyhedralSite objects."""
		factory = SiteFactory(self.structure)
		
		# Create sites with a single environment
		sites = factory.create_polyhedral_sites(self.tetrahedral_env)
		
		# Check that we got the right number of sites
		self.assertEqual(len(sites), 1)
		
		# Check that we got a PolyhedralSite
		self.assertIsInstance(sites[0], PolyhedralSite)
		
		# Check that the vertex indices are correct
		self.assertEqual(sites[0].vertex_indices, [1, 2, 3, 4])
		
		# Check that no label was set
		self.assertIsNone(sites[0].label)
		
		# Check that vertex_coords attribute exists
		self.assertIsNotNone(sites[0].vertex_coords)
	
	def test_create_dynamic_voronoi_sites_integration(self):
		"""Test creation of DynamicVoronoiSite objects."""
		factory = SiteFactory(self.structure)
		
		# Create sites with multiple environments and labels
		sites = factory.create_dynamic_voronoi_sites(
			self.multiple_envs, 
			labels=["site1", "site2"]
		)
		
		# Check that we got the right number of sites
		self.assertEqual(len(sites), 2)
		
		# Check that we got DynamicVoronoiSite objects
		for site in sites:
			self.assertIsInstance(site, DynamicVoronoiSite)
		
		# Check that the reference indices are correct
		self.assertEqual(sites[0].reference_indices, [1, 2, 4])
		self.assertEqual(sites[1].reference_indices, [0, 1, 3, 4])
		
		# Check that labels are set correctly
		self.assertEqual(sites[0].label, "site1")
		self.assertEqual(sites[1].label, "site2")
	
	def test_site_indices_incrementing_integration(self):
		"""Test that site indices are correctly incremented."""
		factory = SiteFactory(self.structure)
		
		# Create multiple sites of the same type
		polyhedral_sites_1 = factory.create_polyhedral_sites([[1, 2, 3]])
		polyhedral_sites_2 = factory.create_polyhedral_sites([[2, 3, 4]])
		polyhedral_sites_3 = factory.create_polyhedral_sites([[0, 1, 2]])
		
		# Check that site indices increment as expected
		self.assertEqual(polyhedral_sites_1[0].index, 0)
		self.assertEqual(polyhedral_sites_2[0].index, 1)
		self.assertEqual(polyhedral_sites_3[0].index, 2)
		
		# Reset the counter for the next test
		Site._newid = 0
		
		# Test with DynamicVoronoiSite
		dynamic_sites_1 = factory.create_dynamic_voronoi_sites([[1, 2, 3]])
		dynamic_sites_2 = factory.create_dynamic_voronoi_sites([[2, 3, 4]])
		
		# Check indices
		self.assertEqual(dynamic_sites_1[0].index, 0)
		self.assertEqual(dynamic_sites_2[0].index, 1)
	
	# Remove the mixed site creation test since we never mix site types in real use
	
	def test_empty_environment_integration(self):
		"""Test that empty environments list returns empty sites list."""
		factory = SiteFactory(self.structure)
		
		# Create sites with empty environments
		polyhedral_sites = factory.create_polyhedral_sites([])
		dynamic_sites = factory.create_dynamic_voronoi_sites([])
		
		# Check that we got empty lists
		self.assertEqual(len(polyhedral_sites), 0)
		self.assertEqual(len(dynamic_sites), 0)


if __name__ == '__main__':
	unittest.main()