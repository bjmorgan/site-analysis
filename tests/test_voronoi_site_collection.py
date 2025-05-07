import unittest
from unittest.mock import Mock, patch
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.voronoi_site_collection import VoronoiSiteCollection
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.atom import Atom
from site_analysis.site import Site


class VoronoiSiteCollectionTestCase(unittest.TestCase):
	
	def setUp(self):
		"""Set up test fixtures."""
		# Reset Site._newid counter
		Site._newid = 0
		
		# Create a real lattice (for proper method behavior)
		self.lattice = Lattice.cubic(10.0)
		
		# Create VoronoiSites with real objects for their core attributes
		self.site1 = VoronoiSite(
			frac_coords=np.array([0.1, 0.1, 0.1]),
			label="site1"
		)
		self.site2 = VoronoiSite(
			frac_coords=np.array([0.5, 0.5, 0.5]),
			label="site2"
		)
		
		# Create test atoms
		self.atom1 = Atom(index=0)
		self.atom1._frac_coords = np.array([0.15, 0.15, 0.15])
		
		self.atom2 = Atom(index=1)
		self.atom2._frac_coords = np.array([0.45, 0.45, 0.45])
		
		self.atoms = [self.atom1, self.atom2]
		
		# Create a test structure with a real lattice
		self.structure = Structure(
			lattice=self.lattice,
			species=["Na", "Na"],
			coords=[[0.15, 0.15, 0.15], [0.45, 0.45, 0.45]]
		)
		
		# Create the collection
		self.collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
	
	def test_initialization_type_checking(self):
		"""Test that initialization enforces VoronoiSite types."""
		# Valid initialization with VoronoiSites
		collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
		self.assertEqual(collection.sites, [self.site1, self.site2])
		
		# Invalid initialization with non-VoronoiSite
		non_voronoi_site = Mock()
		non_voronoi_site.index = 2
		
		with self.assertRaises(TypeError):
			VoronoiSiteCollection(sites=[self.site1, non_voronoi_site])
	
	def test_analyse_structure(self):
		"""Test that analyse_structure assigns coordinates and calls assign_site_occupations."""
		# Patch the methods we want to verify
		with patch.object(Atom, 'assign_coords') as mock_assign_coords, \
			 patch.object(self.collection, 'assign_site_occupations') as mock_assign:
			
			# Call the method being tested
			self.collection.analyse_structure(self.atoms, self.structure)
			
			# Verify assign_coords was called for each atom (twice)
			self.assertEqual(mock_assign_coords.call_count, 2)
			
			# Verify assign_site_occupations was called with atoms and structure
			mock_assign.assert_called_once_with(self.atoms, self.structure)
	
	def test_assign_site_occupations_distance_matrix(self):
		"""Test that assign_site_occupations uses distance matrix correctly."""
		# First reset the sites to ensure they're empty
		self.collection.reset_site_occupations()
		
		# Patch lattice.get_all_distances to return a known distance matrix
		# Row 0: distances from site1 to atoms [atom1, atom2]
		# Row 1: distances from site2 to atoms [atom1, atom2]
		distance_matrix = np.array([
			[1.0, 5.0],  # site1 is closer to atom1
			[5.0, 1.0]   # site2 is closer to atom2
		])
		
		with patch.object(self.structure.lattice, 'get_all_distances', 
						 return_value=distance_matrix) as mock_get_distances, \
			 patch.object(self.collection, 'update_occupation') as mock_update:
			
			# Call the method with non-empty atoms list
			self.collection.assign_site_occupations(self.atoms, self.structure)
			
			# Verify get_all_distances was called
			mock_get_distances.assert_called_once()
			
			# Verify update_occupation was called for each atom-site pair
			mock_update.assert_any_call(self.site1, self.atom1)
			mock_update.assert_any_call(self.site2, self.atom2)
			self.assertEqual(mock_update.call_count, 2)
	
	def test_empty_atoms_list(self):
		"""Test behavior with empty atoms list - verifies the early return after reset."""
		# Populate the sites with some atoms
		self.site1.contains_atoms = [1, 2]
		self.site2.contains_atoms = [3, 4]
		
		# Call assign_site_occupations with empty list
		self.collection.assign_site_occupations([], self.structure)
		
		# Verify sites were reset (contains_atoms should be empty)
		self.assertEqual(self.site1.contains_atoms, [])
		self.assertEqual(self.site2.contains_atoms, [])
	
	def test_reset_site_occupations(self):
		"""Test that reset_site_occupations clears the contains_atoms lists."""
		# Add some atoms to the sites
		self.site1.contains_atoms = [0, 1]
		self.site2.contains_atoms = [2, 3]
		
		# Reset the sites
		self.collection.reset_site_occupations()
		
		# Verify contains_atoms lists are empty
		self.assertEqual(self.site1.contains_atoms, [])
		self.assertEqual(self.site2.contains_atoms, [])
	
	def test_update_occupation(self):
		"""Test the update_occupation method."""
		# Create a test atom
		atom = Atom(index=3)
		atom._frac_coords = np.array([0.3, 0.3, 0.3])
		
		# Call update_occupation to assign the atom to a site
		self.collection.update_occupation(self.site1, atom)
		
		# Verify atom has been assigned to the site
		self.assertEqual(atom.in_site, self.site1.index)
		
		# Verify site contains the atom
		self.assertIn(atom.index, self.site1.contains_atoms)
	
	def test_integration(self):
		"""Integration test with actual behavior."""
		# Reset site occupations
		self.site1.contains_atoms = []
		self.site2.contains_atoms = []
		
		# Reset atom site assignments
		self.atom1.in_site = None
		self.atom2.in_site = None
		
		# Set up atoms with coordinates that make them clearly closer to specific sites
		self.atom1._frac_coords = np.array([0.1, 0.1, 0.11])  # Very close to site1
		self.atom2._frac_coords = np.array([0.5, 0.5, 0.51])  # Very close to site2
		
		# Call the method with real objects
		self.collection.analyse_structure(self.atoms, self.structure)
		
		# Atom1 should be assigned to site1 (closest) and atom2 to site2
		self.assertEqual(self.atom1.in_site, self.site1.index)
		self.assertEqual(self.atom2.in_site, self.site2.index)
		
		# Verify sites contain the correct atoms
		self.assertIn(self.atom1.index, self.site1.contains_atoms)
		self.assertIn(self.atom2.index, self.site2.contains_atoms)
		

if __name__ == '__main__':
	unittest.main()