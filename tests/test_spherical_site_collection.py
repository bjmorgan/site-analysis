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
		
	def test_init_initializes_neighbour_attributes(self):
		"""Test that __init__ initializes neighbour-related attributes to None."""
		sites = [self.site1, self.site2]
		collection = SphericalSiteCollection(sites)
		
		# Check that neighbour attributes are initialized to None
		self.assertIsNone(collection._neighbouring_sites)
		self.assertIsNone(collection._current_lattice)
	
	def test_init_with_empty_sites_list(self):
		"""Test that __init__ works with empty sites list."""
		collection = SphericalSiteCollection([])
		
		self.assertEqual(collection.sites, [])
		self.assertIsNone(collection._neighbouring_sites)
		self.assertIsNone(collection._current_lattice)
		self.assertEqual(collection._site_lookup, {})
		
	def test_init_sets_default_neighbour_cutoff(self):
		"""Test that __init__ sets default neighbour_cutoff to 10.0 Å."""
		sites = [self.site1, self.site2]
		collection = SphericalSiteCollection(sites)
		
		# Check that default cutoff is 10.0
		self.assertEqual(collection._neighbour_cutoff, 10.0)
	
	def test_init_sets_custom_neighbour_cutoff(self):
		"""Test that __init__ can set custom neighbour_cutoff."""
		sites = [self.site1, self.site2]
		collection = SphericalSiteCollection(sites, neighbour_cutoff=15.0)
		
		# Check that custom cutoff is set
		self.assertEqual(collection._neighbour_cutoff, 15.0)
		
	def test_init_raises_type_error_with_non_spherical_sites(self):
		"""Test that initialisation raises TypeError with non-SphericalSite objects."""
		# Create a mix of site types
		non_spherical_site = Mock()
		mixed_sites = [self.site1, non_spherical_site]
		
		# Test initialisation with mixed site types
		with self.assertRaises(TypeError):
			SphericalSiteCollection(sites=mixed_sites)
		
	def test_get_current_lattice_returns_stored_lattice(self):
		"""Test that _get_current_lattice returns the stored lattice when available."""
		collection = SphericalSiteCollection([self.site1])
		mock_lattice = Mock(spec=Lattice)
		
		# Set the current lattice
		collection._current_lattice = mock_lattice
		
		# Should return the stored lattice
		result = collection._get_current_lattice()
		self.assertIs(result, mock_lattice)
	
	def test_get_current_lattice_returns_none_when_none(self):
		"""Test that _get_current_lattice returns None when _current_lattice is None."""
		collection = SphericalSiteCollection([self.site1])
		
		# _current_lattice should be None from __init__
		result = collection._get_current_lattice()
		self.assertIsNone(result)
		
	def test_calculate_all_neighbouring_sites_basic_functionality(self):
		"""Test that _calculate_all_neighbouring_sites correctly identifies neighbours within default 10.0 Å cutoff."""
		# Create sites at known positions
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0, label="site1")
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0, label="site2")  
		site3 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0, label="site3")
		
		collection = SphericalSiteCollection([site1, site2, site3])  # Default 10.0 Å cutoff
		mock_lattice = Mock(spec=Lattice)
		
		# Mock distance matrix: site1-site2 close (5 Å < 10 Å), site1-site3 far (15 Å > 10 Å)
		mock_distances = np.array([
			[0.0, 5.0, 15.0],  # Distances from site1
			[5.0, 0.0, 12.0],  # Distances from site2  
			[15.0, 12.0, 0.0]  # Distances from site3
		])
		mock_lattice.get_all_distances.return_value = mock_distances
		
		# Calculate neighbours
		neighbours = collection._calculate_all_neighbouring_sites(mock_lattice)
		
		# Check results against 10.0 Å cutoff
		self.assertEqual(len(neighbours[site1.index]), 1)  # site1 neighbours: site2 (5 Å ≤ 10 Å)
		self.assertIs(neighbours[site1.index][0], site2)
		
		self.assertEqual(len(neighbours[site2.index]), 1)  # site2 neighbours: site1 (5 Å ≤ 10 Å)
		self.assertIs(neighbours[site2.index][0], site1)
		
		self.assertEqual(len(neighbours[site3.index]), 0)  # site3 neighbours: none (15, 12 Å > 10 Å)
	
	def test_calculate_all_neighbouring_sites_boundary_case(self):
		"""Test sites exactly at the default 10.0 Å cutoff distance."""
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0)
		
		collection = SphericalSiteCollection([site1, site2])  # Default 10.0 Å cutoff
		mock_lattice = Mock(spec=Lattice)
		
		# Mock distance matrix: exactly at the 10.0 Å cutoff
		mock_distances = np.array([
			[0.0, 10.0],
			[10.0, 0.0]
		])
		mock_lattice.get_all_distances.return_value = mock_distances
		
		# Calculate neighbours
		neighbours = collection._calculate_all_neighbouring_sites(mock_lattice)
		
		# Sites at exactly the cutoff distance should be neighbours (10.0 Å ≤ 10.0 Å)
		self.assertEqual(len(neighbours[site1.index]), 1)
		self.assertIs(neighbours[site1.index][0], site2)
		
	def test_calculate_all_neighbouring_sites_sorts_by_distance(self):
		"""Test that neighbours are sorted by increasing distance."""
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0)  # Closest
		site3 = SphericalSite(frac_coords=np.array([0.2, 0.0, 0.0]), rcut=1.0)  # Farthest
		
		collection = SphericalSiteCollection([site1, site2, site3])
		mock_lattice = Mock(spec=Lattice)
		
		# Mock distances: site1 to others = 2.0, 5.0 Å
		mock_distances = np.array([
			[0.0, 2.0, 5.0],
			[2.0, 0.0, 3.0],
			[5.0, 3.0, 0.0]
		])
		mock_lattice.get_all_distances.return_value = mock_distances
		
		neighbours = collection._calculate_all_neighbouring_sites(mock_lattice)
		
		# Check that site1's neighbours are sorted: closest (site2) first, then site3
		self.assertEqual(neighbours[site1.index], [site2, site3])
		
	def test_calculate_all_neighbouring_sites_uses_cutoff_attribute(self):
		"""Test that _calculate_all_neighbouring_sites uses the _neighbour_cutoff attribute."""
		# Create sites with custom cutoff
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0)
		site3 = SphericalSite(frac_coords=np.array([0.2, 0.0, 0.0]), rcut=1.0)
		
		# Create collection with 8.0 Å cutoff (instead of default 10.0 Å)
		collection = SphericalSiteCollection([site1, site2, site3], neighbour_cutoff=8.0)
		mock_lattice = Mock(spec=Lattice)
		
		# Mock distances: site1-site2 = 7.0 Å (within cutoff), site1-site3 = 9.0 Å (outside cutoff)
		mock_distances = np.array([
			[0.0, 7.0, 9.0],
			[7.0, 0.0, 2.0], 
			[9.0, 2.0, 0.0]
		])
		mock_lattice.get_all_distances.return_value = mock_distances
		
		# Calculate neighbours
		neighbours = collection._calculate_all_neighbouring_sites(mock_lattice)
		
		# site1 should only have site2 as neighbour (7.0 Å ≤ 8.0 Å cutoff)
		# site3 should not be a neighbour (9.0 Å > 8.0 Å cutoff)
		self.assertEqual(len(neighbours[site1.index]), 1)
		self.assertIs(neighbours[site1.index][0], site2)
		
		# site3 should have no neighbours from site1 (9.0 Å > 8.0 Å cutoff)
		# but should have site2 (2.0 Å ≤ 8.0 Å cutoff)
		self.assertEqual(len(neighbours[site3.index]), 1)
		self.assertIs(neighbours[site3.index][0], site2)
	
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
		
	def test_neighbouring_sites(self):
		"""Test that neighbouring_sites returns the correct neighbours."""
		# Create sites at known positions for distance testing
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0, label="site1")
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0, label="site2")  # Close to site1
		site3 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0, label="site3")  # Far from site1
		
		collection = SphericalSiteCollection([site1, site2, site3])
		
		# Mock the _neighbouring_sites to control the test
		test_neighbours = {
			site1.index: [site2],  # site1 neighbours site2
			site2.index: [site1],  # site2 neighbours site1
			site3.index: []        # site3 has no neighbours
		}
		collection._neighbouring_sites = test_neighbours
		
		# Test site1 neighbours
		neighbours = collection.neighbouring_sites(site1.index)
		self.assertEqual(len(neighbours), 1)
		self.assertIs(neighbours[0], site2)
		
		# Test site2 neighbours
		neighbours = collection.neighbouring_sites(site2.index)
		self.assertEqual(len(neighbours), 1)
		self.assertIs(neighbours[0], site1)
		
		# Test site3 neighbours (none)
		neighbours = collection.neighbouring_sites(site3.index)
		self.assertEqual(len(neighbours), 0)
	
	def test_neighbouring_sites_lazy_calculation_all_distances(self):
		"""Test that neighbouring_sites calculates all site-site distances on first access."""
		# Create sites at known positions
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0, label="site1")
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0, label="site2")  # Close to site1
		site3 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0, label="site3")  # Far from site1
		
		collection = SphericalSiteCollection([site1, site2, site3])
		
		# Mock the _calculate_all_neighbouring_sites method
		with patch.object(collection, '_calculate_all_neighbouring_sites') as mock_calculate:
			mock_neighbours = {
				site1.index: [site2],
				site2.index: [site1], 
				site3.index: []
			}
			mock_calculate.return_value = mock_neighbours
			
			# Create a mock structure with lattice for distance calculations
			mock_structure = Mock(spec=Structure)
			mock_lattice = Mock(spec=Lattice)
			mock_structure.lattice = mock_lattice
			
			# First call should trigger calculation
			with patch.object(collection, '_get_current_lattice', return_value=mock_lattice):
				neighbours = collection.neighbouring_sites(site1.index)
			
			# Verify calculation was called with the lattice
			mock_calculate.assert_called_once_with(mock_lattice)
			
			# Verify result
			self.assertEqual(len(neighbours), 1)
			self.assertIs(neighbours[0], site2)
	
	def test_neighbouring_sites_caches_calculation(self):
		"""Test that neighbouring_sites only calculates once and caches results."""
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)
		site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.0)
		
		collection = SphericalSiteCollection([site1, site2])
		
		with patch.object(collection, '_calculate_all_neighbouring_sites') as mock_calculate:
			mock_calculate.return_value = {site1.index: [site2], site2.index: [site1]}
			
			with patch.object(collection, '_get_current_lattice', return_value=Mock(spec=Lattice)):
				# Multiple calls to neighbouring_sites
				collection.neighbouring_sites(site1.index)
				collection.neighbouring_sites(site2.index)
				collection.neighbouring_sites(site1.index)  # Call again
			
			# Calculation should only be called once
			mock_calculate.assert_called_once()
	
	def test_neighbouring_sites_requires_lattice_for_calculation(self):
		"""Test that neighbouring_sites properly handles lattice requirement."""
		site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)
		collection = SphericalSiteCollection([site1])
		
		# Mock _get_current_lattice to return None (no lattice available)
		with patch.object(collection, '_get_current_lattice', return_value=None):
			with self.assertRaises(RuntimeError) as context:
				collection.neighbouring_sites(site1.index)
			
			self.assertIn("No lattice available", str(context.exception))
			
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
			atom1._frac_coords = np.array([0.05, 0.05, 0.05])  # Should stay in site1
			
			atom2 = Atom(index=1) 
			atom2.trajectory = [site1.index]
			atom2._frac_coords = np.array([0.15, 0.05, 0.05])  # Should move to site2
			
			atoms = [atom1, atom2]
			
			# Run the optimised assignment
			collection.assign_site_occupations(atoms, structure)
			
			# Verify key integration points
			self.assertEqual(collection._current_lattice, lattice)  # Lattice stored
			self.assertIsNotNone(collection._neighbouring_sites)    # Neighbors calculated
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
		self.atom.trajectory = [self.site2.index]
		with patch.object(self.collection, 'neighbouring_sites', return_value=[]):
			priority_sites = list(self.collection._get_priority_sites(self.atom))
			self.assertEqual(priority_sites[0], self.site2)
			
	def test_yields_most_recently_visited_when_most_recent_is_none(self):
		"""Test that generator yields most recently visited site when most recent is None."""
		self.atom.trajectory = [self.site1.index, self.site2.index, None]
		with patch.object(self.collection, 'neighbouring_sites', return_value=[]): 
			priority_sites = list(self.collection._get_priority_sites(self.atom))
			self.assertEqual(priority_sites[0], self.site2)
	
	def test_yields_all_sites_when_no_valid_trajectory(self):
		"""Test that generator yields all sites when no valid site history exists."""
		self.atom.trajectory = [None, None]
		priority_sites = list(self.collection._get_priority_sites(self.atom))
		self.assertEqual(len(priority_sites), 3)
		self.assertIn(self.site1, priority_sites)
		self.assertIn(self.site2, priority_sites)
		self.assertIn(self.site3, priority_sites)
		
	def test_yields_transition_destinations_after_most_recent(self):
		"""Test that generator yields transition destinations after most recent site."""
		self.atom.trajectory = [self.site1.index]
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			with patch.object(self.collection, 'neighbouring_sites', return_value=[]):
				mock_transitions.return_value = [self.site3.index, self.site2.index]
				priority_site_indices = [site.index for site in self.collection._get_priority_sites(self.atom)]
				self.assertEqual(priority_site_indices, [self.site1.index, self.site3.index, self.site2.index])   
	
	def test_yields_no_duplicates_when_all_sites_are_transitions(self):
		"""Test that generator doesn't yield duplicates when all sites appear as transitions."""
		self.atom.trajectory = [self.site1.index]
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			with patch.object(self.collection, 'neighbouring_sites', return_value=[]):
				mock_transitions.return_value = [self.site3.index, self.site2.index]
				priority_sites = list(self.collection._get_priority_sites(self.atom))
				self.assertEqual(len(priority_sites), 3)
				site_indices = [site.index for site in priority_sites]
				self.assertEqual(len(site_indices), len(set(site_indices)))
				self.assertEqual(site_indices, [self.site1.index, self.site3.index, self.site2.index])
	
	def test_skips_neighbour_checking_when_no_most_recent_site(self):
		"""Test that neighbour checking is skipped when atom has no most recent site."""
		self.atom.trajectory = []
		with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
			priority_sites = list(self.collection._get_priority_sites(self.atom))
			self.assertEqual(len(priority_sites), 3)
			mock_neighbours.assert_not_called()
	
	def test_yields_remaining_sites_after_neighbours(self):
		"""Test that generator yields remaining sites after neighbours."""
		# Set up atom with most recent site
		self.atom.trajectory = [self.site1.index]  # Most recent is site1
		
		# Mock transitions and neighbours
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
				mock_transitions.return_value = []  # No transitions
				mock_neighbours.return_value = [self.site2]  # One neighbour
				
				# Get priority sites
				priority_sites = list(self.collection._get_priority_sites(self.atom))
				
				# Should be: site1 (most recent), site2 (neighbour), site3 (remaining)
				self.assertEqual(priority_sites[0], self.site1)  # Most recent
				self.assertEqual(priority_sites[1], self.site2)  # Neighbour
				self.assertEqual(priority_sites[2], self.site3)  # Remaining
				
	def test_yields_neighbours_after_transitions(self):
		"""Test that generator yields neighbours after transition destinations."""
		# Set up atom with most recent site
		self.atom.trajectory = [self.site1.index]  # Most recent is site1
		
		# Mock transitions and neighbours
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
				mock_transitions.return_value = [self.site2.index]  # One transition
				mock_neighbours.return_value = [self.site3]  # One neighbour
				
				# Get priority sites
				priority_sites = list(self.collection._get_priority_sites(self.atom))
				
				# Should be: site1 (most recent), site2 (transition), site3 (neighbour)
				self.assertEqual(priority_sites[0], self.site1)  # Most recent
				self.assertEqual(priority_sites[1], self.site2)  # Transition
				self.assertEqual(priority_sites[2], self.site3)  # Neighbour
				
				# Verify neighbouring_sites was called with most recent site index
				mock_neighbours.assert_called_once_with(self.site1.index)
	
	def test_yields_no_duplicates_with_neighbours_and_transitions(self):
		"""Test that generator doesn't yield duplicates when neighbour appears as transition."""
		# Set up atom with most recent site
		self.atom.trajectory = [self.site1.index]  # Most recent is site1
		
		# Mock transitions and neighbours where site2 appears in both
		with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
			with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
				mock_transitions.return_value = [self.site2.index]  # site2 as transition
				mock_neighbours.return_value = [self.site2, self.site3]  # site2 also as neighbour
				
				# Get priority sites
				priority_sites = list(self.collection._get_priority_sites(self.atom))
				
				# Should be exactly 3 sites, no duplicates
				self.assertEqual(len(priority_sites), 3)
				
				# Convert to indices
				site_indices = [site.index for site in priority_sites]
				
				# Should have no duplicates
				self.assertEqual(len(site_indices), len(set(site_indices)))
				
				# site2 should only appear once (as transition, not again as neighbour)
				self.assertEqual(site_indices.count(self.site2.index), 1)
				
				# Order should be: site1, site2 (transition), site3 (neighbour)
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
	
	def test_stores_lattice_from_structure(self):
		"""Test that assign_site_occupations stores the lattice from structure."""
		with patch.object(self.collection, '_get_priority_sites', return_value=[]):
			self.collection.assign_site_occupations(self.atoms, self.structure)
			self.assertEqual(self.collection._current_lattice, self.structure.lattice)


if __name__ == '__main__':
	unittest.main()