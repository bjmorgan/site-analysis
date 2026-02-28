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
		self.assertIsNotNone(self.collection._distance_ranked_sites)
		
	def test_init_with_empty_sites_list(self):
		"""Test that __init__ works with empty sites list."""
		collection = SphericalSiteCollection([])
		
		self.assertEqual(collection.sites, [])
		self.assertEqual(collection._site_lookup, {})
		
	def test_init_raises_type_error_with_non_spherical_sites(self):
		"""Test that initialisation raises TypeError with non-SphericalSite objects."""
		# Create a mix of site types
		non_spherical_site = Mock()
		mixed_sites = [self.site1, non_spherical_site]
		
		# Test initialisation with mixed site types
		with self.assertRaises(TypeError):
			SphericalSiteCollection(sites=mixed_sites)
		
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
		
	def test_full_optimised_assignment_integration(self):
			"""Integration test for the complete optimised site assignment algorithm."""
			# Simple setup: 3 sites, 2 atoms
			site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.5)
			site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.5)
			site3 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.5)
			
			collection = SphericalSiteCollection([site1, site2, site3])
			lattice = Lattice.cubic(10.0)
			structure = Structure(lattice, ["Li", "Li"], [[0.05, 0.05, 0.05], [0.15, 0.05, 0.05]])
			
			# Create atoms with some trajectory history
			atom1 = Atom(index=0)
			atom1.trajectory = [site1.index]
			atom1._recent_sites = [site1.index, None]
			atom1._frac_coords = np.array([0.05, 0.05, 0.05])  # Should stay in site1

			atom2 = Atom(index=1)
			atom2.trajectory = [site1.index]
			atom2._recent_sites = [site1.index, None]
			atom2._frac_coords = np.array([0.15, 0.05, 0.05])  # Should move to site2
			
			atoms = [atom1, atom2]
			
			# Run the optimised assignment
			collection.assign_site_occupations(atoms, structure)
			
			# Verify key integration points
			self.assertEqual(atom1.in_site, site1.index)           # Correct assignment
			self.assertEqual(atom2.in_site, site2.index)           # Correct assignment
			self.assertEqual(site1.contains_atoms, [atom1.index])  # Site updated
			self.assertEqual(site2.contains_atoms, [atom2.index])  # Site updated
			
			
class TestGetPrioritySites(unittest.TestCase):
	"""Test _get_priority_sites generator behavior for SphericalSiteCollection."""
	
	def setUp(self):
		Site._newid = 0
		self.lattice = Lattice.cubic(10.0)
		
		self.site1 = SphericalSite(frac_coords=np.array([0.1, 0.1, 0.1]), rcut=1.5, label="site1")
		self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.5, label="site2")
		self.site3 = SphericalSite(frac_coords=np.array([0.8, 0.8, 0.8]), rcut=1.5, label="site3")
		self.collection = SphericalSiteCollection([self.site1, self.site2, self.site3])
		
		self.atom = Atom(index=0)
		self.atom._frac_coords = np.array([0.2, 0.2, 0.2])
	
	def test_yields_most_recent_site_first(self):
		"""Test that generator yields most recent site as first site."""
		self.atom._recent_sites = [self.site2.index, None]
		priority_sites = list(self.collection._get_priority_sites(self.atom))
		self.assertEqual(priority_sites[0], self.site2)

	def test_yields_both_recent_sites(self):
		"""Test that generator yields both recent distinct sites."""
		self.atom._recent_sites = [self.site2.index, self.site1.index]
		priority_sites = list(self.collection._get_priority_sites(self.atom))
		self.assertEqual(priority_sites[0], self.site2)
		self.assertEqual(priority_sites[1], self.site1)
	
	def test_yields_all_sites_when_no_valid_trajectory(self):
		"""Test that generator yields all sites when no valid site history exists."""
		self.atom.trajectory = [None, None]
		priority_sites = list(self.collection._get_priority_sites(self.atom))
		self.assertEqual(len(priority_sites), 3)
		# All sites yielded, starting from nearest
		self.assertEqual(priority_sites[0], self.site1)  # nearest to atom at [0.2,0.2,0.2]
		
	def test_yields_transition_destinations_after_most_recent(self):
		"""Test that generator yields transition destinations after most recent site."""
		self.atom._recent_sites = [self.site1.index, None]
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			mock_transitions.return_value = [self.site3.index, self.site2.index]
			priority_site_indices = [site.index for site in self.collection._get_priority_sites(self.atom)]
			self.assertEqual(priority_site_indices, [self.site1.index, self.site3.index, self.site2.index])

	def test_yields_no_duplicates_when_all_sites_are_transitions(self):
		"""Test that generator doesn't yield duplicates when all sites appear as transitions."""
		self.atom._recent_sites = [self.site1.index, None]
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			mock_transitions.return_value = [self.site3.index, self.site2.index]
			priority_sites = list(self.collection._get_priority_sites(self.atom))
			self.assertEqual(len(priority_sites), 3)
			site_indices = [site.index for site in priority_sites]
			self.assertEqual(len(site_indices), len(set(site_indices)))
			self.assertEqual(site_indices, [self.site1.index, self.site3.index, self.site2.index])
	
	def test_no_history_uses_nearest_site_first(self):
		"""Test that nearest site is yielded first when atom has no history."""
		# atom at [0.2, 0.2, 0.2] -- nearest site is site1 at [0.1, 0.1, 0.1]
		priority_sites = list(self.collection._get_priority_sites(self.atom))
		self.assertEqual(len(priority_sites), 3)
		self.assertEqual(priority_sites[0], self.site1)
	
	def test_yields_remaining_sites_distance_ranked(self):
		"""Test that remaining sites are yielded in distance-ranked order."""
		self.atom._recent_sites = [self.site1.index, None]

		with patch.object(self.site1, 'most_frequent_transitions', return_value=[]):
			priority_sites = list(self.collection._get_priority_sites(self.atom))

			# site1 at [0.1,0.1,0.1], site2 at [0.5,0.5,0.5], site3 at [0.8,0.8,0.8]
			# From site1 with minimum-image convention:
			#   site3 is 0.3*sqrt(3) away via PBC, site2 is 0.4*sqrt(3) away
			self.assertEqual(priority_sites[0], self.site1)
			self.assertEqual(priority_sites[1], self.site3)
			self.assertEqual(priority_sites[2], self.site2)

	def test_yields_transitions_then_distance_ranked(self):
		"""Test that transitions come before distance-ranked remaining sites."""
		self.atom._recent_sites = [self.site1.index, None]

		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			mock_transitions.return_value = [self.site3.index]

			priority_sites = list(self.collection._get_priority_sites(self.atom))

			# site1 (recent), site3 (transition), site2 (distance-ranked remaining)
			self.assertEqual(priority_sites[0], self.site1)
			self.assertEqual(priority_sites[1], self.site3)
			self.assertEqual(priority_sites[2], self.site2)
	
	def test_yields_no_duplicates_with_transitions(self):
		"""Test that generator doesn't yield duplicates when transition overlaps with distance ranking."""
		self.atom._recent_sites = [self.site1.index, None]

		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			mock_transitions.return_value = [self.site2.index]

			priority_sites = list(self.collection._get_priority_sites(self.atom))

			# Should be exactly 3 sites, no duplicates
			self.assertEqual(len(priority_sites), 3)
			site_indices = [site.index for site in priority_sites]
			self.assertEqual(len(site_indices), len(set(site_indices)))

			# site2 appears as transition, then site3 from distance ranking
			self.assertEqual(site_indices, [self.site1.index, self.site2.index, self.site3.index])
				
class TestAssignSiteOccupationsInteraction(unittest.TestCase):
	"""Test interaction between assign_site_occupations and _get_priority_sites."""
	
	def setUp(self):
		Site._newid = 0
		self.lattice = Lattice.cubic(2.0)
		self.structure = Structure(self.lattice, ["Li"], [[0.1, 0.1, 0.1]])
		
		self.site1 = SphericalSite(frac_coords=np.array([0.1, 0.1, 0.1]), rcut=1.5)
		self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.5)
		self.collection = SphericalSiteCollection([self.site1, self.site2])
		
		self.atom = Atom(index=0)
		self.atom._frac_coords = np.array([0.2, 0.2, 0.2])
		self.atoms = [self.atom]
	
	def test_calls_generator_for_each_atom(self):
		"""Test that _get_priority_sites is called once per atom."""
		with patch.object(self.collection, '_get_priority_sites', return_value=[]):
			self.collection.assign_site_occupations(self.atoms, self.structure)
			self.collection._get_priority_sites.assert_called_once_with(self.atom)
	
	def test_calls_generator_for_multiple_atoms(self):
		"""Test that _get_priority_sites is called for each atom."""
		# Add second atom
		atom2 = Atom(index=1)
		atom2._frac_coords = np.array([0.3, 0.3, 0.3])
		atoms = [self.atom, atom2]
		
		with patch.object(self.collection, '_get_priority_sites', return_value=[]):
			self.collection.assign_site_occupations(atoms, self.structure)
			self.assertEqual(self.collection._get_priority_sites.call_count, 2)
	
	def test_checks_sites_in_generator_order(self):
		"""Test that sites are checked in the order returned by generator."""
		call_order = []
		self.site1.contains_atom = lambda atom, **kwargs: call_order.append(1) or False
		self.site2.contains_atom = lambda atom, **kwargs: call_order.append(2) or True
		
		with patch.object(self.collection, '_get_priority_sites') as mock_gen:
			mock_gen.return_value = [self.site2, self.site1]  # site2 first
			
			self.collection.assign_site_occupations(self.atoms, self.structure)
			
			self.assertEqual(call_order, [2])  # Only site2 checked (found there)
	
	def test_stops_checking_when_atom_found(self):
		"""Test that checking stops as soon as atom is found."""
		with patch.object(self.site1, 'contains_atom', return_value=True) as mock_contains1:
			with patch.object(self.site2, 'contains_atom', return_value=False) as mock_contains2:
				with patch.object(self.collection, '_get_priority_sites') as mock_gen:
					mock_gen.return_value = [self.site1, self.site2]
					
					self.collection.assign_site_occupations(self.atoms, self.structure)
					
					mock_contains1.assert_called_once()
					mock_contains2.assert_not_called()
	
	def test_calls_update_occupation_when_found(self):
		"""Test that update_occupation is called when atom found."""
		with patch.object(self.site1, 'contains_atom', return_value=True):
			with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
				with patch.object(self.collection, 'update_occupation') as mock_update:
					self.collection.assign_site_occupations(self.atoms, self.structure)
					mock_update.assert_called_once_with(self.site1, self.atom)
	
	def test_handles_atom_not_found(self):
		"""Test behavior when atom not found in any site."""
		with patch.object(self.site1, 'contains_atom', return_value=False):
			with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
				with patch.object(self.collection, 'update_occupation') as mock_update:
					self.collection.assign_site_occupations(self.atoms, self.structure)
					
					mock_update.assert_not_called()
					self.assertIsNone(self.atom.in_site)
	
	def test_resets_site_occupations(self):
		"""Test that reset_site_occupations is called at start."""
		with patch.object(self.collection, 'reset_site_occupations') as mock_reset:
			with patch.object(self.collection, '_get_priority_sites', return_value=[]):
				self.collection.assign_site_occupations(self.atoms, self.structure)
				mock_reset.assert_called_once()
	
	def test_resets_atom_in_site(self):
		"""Test that atom.in_site is reset to None."""
		self.atom.in_site = 999  # Set to some previous value
		
		with patch.object(self.collection, '_get_priority_sites', return_value=[]):
			self.collection.assign_site_occupations(self.atoms, self.structure)
			self.assertIsNone(self.atom.in_site)
	

if __name__ == '__main__':
	unittest.main()