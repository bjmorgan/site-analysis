import unittest
from unittest.mock import patch, Mock, PropertyMock
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.spherical_site_collection import SphericalSiteCollection
from site_analysis.spherical_site import SphericalSite
from site_analysis.atom import Atom
from site_analysis.site import Site


class SphericalSiteCollectionTestCase(unittest.TestCase):
	
	def setUp(self):
		"""Set up test fixtures with real objects rather than complex mocks."""
		# Reset Site ID counter
		Site._newid = 0
		
		# Create a real lattice
		self.lattice = Lattice.cubic(10.0)
		
		# Create real SphericalSite objects
		self.site1 = SphericalSite(
			frac_coords=np.array([0.1, 0.1, 0.1]),
			rcut=1.5,
			label="site1"
		)
		self.site2 = SphericalSite(
			frac_coords=np.array([0.5, 0.5, 0.5]),
			rcut=1.5,
			label="site2"
		)
		
		# Create collection with real sites
		self.sites = [self.site1, self.site2]
		self.collection = SphericalSiteCollection(sites=self.sites)
		
		# Create real atoms
		self.atom1 = Atom(index=0)
		self.atom2 = Atom(index=1)
		self.atoms = [self.atom1, self.atom2]
		
		# Create real structure
		self.structure = Structure(
			lattice=self.lattice,
			species=["Na", "Na"],
			coords=[[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]
		)
	
	def test_initialization(self):
		"""Test that SphericalSiteCollection initializes correctly."""
		self.assertEqual(self.collection.sites, self.sites)
	
	def test_analyse_structure(self):
		"""Test that analyse_structure calls assign_coords and assign_site_occupations."""
		# Patch the methods we want to verify
		with patch.object(Atom, 'assign_coords') as mock_assign_coords, \
			 patch.object(SphericalSiteCollection, 'assign_site_occupations') as mock_assign_occupations:
			
			# Call analyse_structure
			self.collection.analyse_structure(self.atoms, self.structure)
			
			# Verify that assign_coords was called for each atom
			self.assertEqual(mock_assign_coords.call_count, 2)
			
			# Verify that assign_site_occupations was called
			mock_assign_occupations.assert_called_once_with(self.atoms, self.structure)
	
	def test_site_occupation_reset(self):
		"""Test that site occupations are reset at the beginning of assign_site_occupations."""
		# Add atoms to sites
		self.site1.contains_atoms = [0]
		self.site2.contains_atoms = [1]
		
		# Patch update_occupation to prevent it from running
		with patch.object(self.collection, 'update_occupation'):
			# Call the method
			self.collection.assign_site_occupations([], self.structure)
			
			# Verify sites were reset
			self.assertEqual(self.site1.contains_atoms, [])
			self.assertEqual(self.site2.contains_atoms, [])
	
	def test_high_level_site_allocation(self):
		"""Test the high-level site allocation logic without mocking fine details."""
		# Set up atoms with coordinates that match the sites
		# This avoids mocking complex interactions with SphericalSite.contains_atom
		self.atom1._frac_coords = np.array([0.1, 0.1, 0.1])  # Near site1
		self.atom2._frac_coords = np.array([0.5, 0.5, 0.5])  # Near site2
		
		# Patch reset_site_occupations and update_occupation
		with patch.object(SphericalSiteCollection, 'reset_site_occupations'), \
			 patch.object(SphericalSiteCollection, 'update_occupation') as mock_update:
			
			# Call the method
			self.collection.assign_site_occupations(self.atoms, self.structure)
			
			# Verify update_occupation was called twice (once per atom)
			self.assertEqual(mock_update.call_count, 2)
	
	def test_integration(self):
		"""Integration test with actual behavior for a simple case."""
		# Set up atoms with coordinates
		self.atom1._frac_coords = np.array([0.1, 0.1, 0.1])  # Inside site1's radius
		self.atom2._frac_coords = np.array([0.5, 0.5, 0.5])  # Inside site2's radius
		
		# Reset sites
		self.site1.contains_atoms = []
		self.site2.contains_atoms = []
		
		# Call the method directly
		self.collection.analyse_structure(self.atoms, self.structure)
		
		# Verify atoms were assigned to the correct sites
		self.assertEqual(self.atom1.in_site, self.site1.index)
		self.assertEqual(self.atom2.in_site, self.site2.index)
		
		# Verify sites contain the correct atoms
		self.assertEqual(self.site1.contains_atoms, [self.atom1.index])
		self.assertEqual(self.site2.contains_atoms, [self.atom2.index])
	
	def test_update_occupation(self):
		"""Test the update_occupation method."""
		# Initialize an atom not in any site
		atom = Atom(index=5)
		atom.in_site = None
		atom._frac_coords = np.array([0.3, 0.3, 0.3])
		
		# Call update_occupation to assign the atom to a site
		self.collection.update_occupation(self.site1, atom)
		
		# Verify atom has been assigned to the site
		self.assertEqual(atom.in_site, self.site1.index)
		
		# Verify site contains the atom
		self.assertIn(atom.index, self.site1.contains_atoms)
		
		# Verify atom coords are added to site points
		np.testing.assert_array_equal(self.site1.points[-1], atom.frac_coords)
	
	def test_update_occupation_with_transition(self):
		"""Test the update_occupation method when an atom transitions between sites."""
		# Initialize an atom in site2
		atom = Atom(index=5)
		atom.in_site = self.site2.index
		atom.trajectory = [atom.in_site]
		atom._frac_coords = np.array([0.3, 0.3, 0.3])
		
		# Call update_occupation to move the atom to site1
		self.collection.update_occupation(self.site1, atom)
		
		# Verify atom has been assigned to the new site
		self.assertEqual(atom.in_site, self.site1.index)
		
		# Verify transition was recorded in site2
		self.assertEqual(self.site2.transitions[self.site1.index], 1)
		
	def test_checks_most_recent_site(self):
		"""Test that assign_site_occupations checks most_recent_site when in_site is None."""
		# Create mock lattice and structure
		mock_lattice = Mock(spec=Lattice)
		mock_structure = Mock(spec=Structure)
		mock_structure.lattice = mock_lattice
		
		# Create mock site and collection
		mock_site = Mock(spec=SphericalSite, index=5)
		collection = SphericalSiteCollection(sites=[mock_site])
		
		# Create mock atom with no current site but with history
		mock_atom = Mock(spec=Atom, index=42, in_site=None)
		
		# Mock the most_recent_site property
		most_recent_site_mock = PropertyMock(return_value=5)
		type(mock_atom).most_recent_site = most_recent_site_mock
		
		# Patch methods on the collection
		with patch.object(collection, 'update_occupation') as mock_update, \
			patch.object(collection, 'site_by_index') as mock_site_by_index:
			# Configure site_by_index to return our mock site
			mock_site_by_index.return_value = mock_site
			
			# Call the method we're testing
			collection.assign_site_occupations([mock_atom], mock_structure)
			
			# Verify most_recent_site property was accessed
			most_recent_site_mock.assert_called_once()
			
			# Verify site_by_index was called with the correct index
			mock_site_by_index.assert_called_with(5)
			
			# Verify contains_atom was called with the correct parameters
			mock_site.contains_atom.assert_called_with(mock_atom, mock_lattice)
			
			# Verify update_occupation was called with the right site and atom
			mock_update.assert_called_with(mock_site, mock_atom)


if __name__ == '__main__':
	unittest.main()