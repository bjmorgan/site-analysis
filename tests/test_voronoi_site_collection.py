import unittest
from unittest.mock import Mock, patch
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.voronoi_site_collection import VoronoiSiteCollection
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.atom import Atom


class VoronoiSiteCollectionTestCase(unittest.TestCase):
	
	def setUp(self):
		"""Set up test fixtures."""
		# Create mock VoronoiSites - we don't test VoronoiSite functionality here
		self.site1 = Mock(spec=VoronoiSite)
		self.site1.index = 0
		self.site1.contains_atoms = []
		self.site1.frac_coords = np.array([0.1, 0.1, 0.1])
		
		self.site2 = Mock(spec=VoronoiSite)
		self.site2.index = 1
		self.site2.contains_atoms = []
		self.site2.frac_coords = np.array([0.5, 0.5, 0.5])
		
		# Create a non-VoronoiSite object for type checking tests
		self.non_voronoi_site = Mock()
		self.non_voronoi_site.index = 2
		
		# Create test atoms
		self.atom1 = Mock(spec=Atom)
		self.atom1.index = 0
		self.atom1.in_site = None
		self.atom1.frac_coords = np.array([0.15, 0.15, 0.15])
		
		self.atom2 = Mock(spec=Atom)
		self.atom2.index = 1
		self.atom2.in_site = None
		self.atom2.frac_coords = np.array([0.45, 0.45, 0.45])
		
		self.atoms = [self.atom1, self.atom2]
		
		# Create test structure
		self.structure = Mock(spec=Structure)
		self.structure.lattice = Mock()
	
	def test_initialization_type_checking(self):
		"""Test that initialization enforces VoronoiSite types."""
		# Valid initialization with VoronoiSites
		collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
		self.assertEqual(collection.sites, [self.site1, self.site2])
		
		# Invalid initialization with non-VoronoiSite
		with self.assertRaises(TypeError):
			VoronoiSiteCollection(sites=[self.site1, self.non_voronoi_site])
	
	def test_analyse_structure(self):
		"""Test that analyse_structure assigns coordinates and calls assign_site_occupations."""
		# Create collection
		collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
		
		# Mock methods
		with patch.object(collection, 'assign_site_occupations') as mock_assign:
			# Call the method being tested
			collection.analyse_structure(self.atoms, self.structure)
			
			# Verify each atom had assign_coords called with the structure
			self.atom1.assign_coords.assert_called_once_with(self.structure)
			self.atom2.assign_coords.assert_called_once_with(self.structure)
			
			# Verify assign_site_occupations was called with atoms and structure
			mock_assign.assert_called_once_with(self.atoms, self.structure)
	
	def test_assign_site_occupations_distance_matrix(self):
		"""Test that assign_site_occupations correctly uses the distance matrix."""
		# Create collection
		collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
		
		# Mock distance