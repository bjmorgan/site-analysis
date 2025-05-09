import unittest
from unittest.mock import Mock, patch, PropertyMock, call
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection
from site_analysis.atom import Atom
from site_analysis.site import Site


class DynamicVoronoiSiteCollectionTestCase(unittest.TestCase):

	def setUp(self):
		# Reset the Site._newid counter before each test
		Site._newid = 0
	
	def test_site_collection_is_initialised(self):
		"""Test that a DynamicVoronoiSiteCollection is correctly initialised."""
		sites = [Mock(spec=DynamicVoronoiSite, index=0),
				 Mock(spec=DynamicVoronoiSite, index=1)]
		site_collection = DynamicVoronoiSiteCollection(sites=sites)
		self.assertEqual(site_collection.sites, sites)
		
	def test_init_raises_error_for_non_dynamic_voronoi_sites(self):
		"""Test that initialization raises an error if sites are not DynamicVoronoiSite instances."""
		# Create a mix of site types
		sites = [Mock(spec=DynamicVoronoiSite), Mock()]
		
		# Check that initialization raises TypeError
		with self.assertRaises(TypeError):
			DynamicVoronoiSiteCollection(sites=sites)
			
	def test_analyse_structure(self):
		"""Test that analyse_structure correctly processes atoms and updates site centres."""
		# Create mock sites and atoms with proper index attributes
		sites = [
			Mock(spec=DynamicVoronoiSite, reference_indices=[0, 1], index=0),
			Mock(spec=DynamicVoronoiSite, reference_indices=[2, 3], index=1)
		]
		atoms = [Mock(spec=Atom) for _ in range(5)]
		structure = Mock(spec=Structure)
		
		# Initialize the collection
		site_collection = DynamicVoronoiSiteCollection(sites=sites)
		
		# Mock the assign_site_occupations method to avoid actually calling it
		site_collection.assign_site_occupations = Mock()
		
		# Call analyse_structure
		site_collection.analyse_structure(atoms, structure)
		
		# Check that each atom's coordinates are assigned
		for atom in atoms:
			atom.assign_coords.assert_called_with(structure)
		
		# Check that each site's centre is calculated
		for site in sites:
			site.calculate_centre.assert_called_with(structure)
		
		# Check that site occupations are assigned
		site_collection.assign_site_occupations.assert_called_with(atoms, structure)
			
	def test_assign_site_occupations(self):
		"""Test that atoms are correctly assigned to sites based on Voronoi tessellation."""
		# Create mock sites
		site1 = Mock(spec=DynamicVoronoiSite)
		site2 = Mock(spec=DynamicVoronoiSite)
		site1.index = 0
		site2.index = 1
		# Set up centre method to return fixed coordinates
		centre1 = PropertyMock(return_value=np.array([0.2, 0.2, 0.2]))
		centre2 = PropertyMock(return_value=np.array([0.8, 0.8, 0.8]))
		type(site1).centre = centre1
		type(site2).centre = centre2
		
		sites = [site1, site2]
		
		# Create mock atoms
		atoms = [Mock(spec=Atom) for _ in range(5)]
		for i, atom in enumerate(atoms):
			atom.index = i
			# Set up frac_coords attribute
			atom.frac_coords = np.array([0.1 + 0.2 * i,
										 0.1 + 0.2 * i,
										 0.1 + 0.2 * i])
		
		# Create mock structure and lattice
		lattice = Mock(spec=Lattice)
		structure = Mock(spec=Structure)
		structure.lattice = lattice
		
		# Mock the distance calculation
		# Return a matrix where:
		# - atoms 0-1 are closer to site1
		# - atoms 2-4 are closer to site2
		mock_distances = np.array([
			[2.0, 3.0, 8.0, 10.0, 7.0],  # Distances from site1 to atoms 0-4
			[8.0, 7.0, 2.0, 1.0, 3.0]    # Distances from site2 to atoms 0-4
		])
		lattice.get_all_distances = Mock(return_value=mock_distances)
		
		# Create collection with mocked sites
		site_collection = DynamicVoronoiSiteCollection(sites=sites)
		
		# Mock the reset_site_occupations method
		site_collection.reset_site_occupations = Mock()
		
		# Mock the update_occupation method to track assignments
		site_collection.update_occupation = Mock()
		
		# Call the method being tested
		site_collection.assign_site_occupations(atoms, structure)
		
		# Verify reset_site_occupations was called
		site_collection.reset_site_occupations.assert_called_once()
		
		# Verify centre property was accessed for each site
		centre1.assert_called()
		centre2.assert_called()
		
		# Verify get_all_distances was called with the correct parameters
		# We can't use site1.centre here because it's a PropertyMock, so get the value directly
		expected_site_coords = np.array([
			centre1.return_value,
			centre2.return_value
		])
		atom_coords = np.array([atom.frac_coords for atom in atoms])
		
		# We can't directly compare numpy arrays in the call args, so we need to extract them
		args, kwargs = lattice.get_all_distances.call_args
		np.testing.assert_array_equal(args[0], expected_site_coords)
		np.testing.assert_array_equal(args[1], atom_coords)
		
		# Verify update_occupation was called for each atom with the correct site
		expected_calls = [
			call(site1, atoms[0]),  # Atom 0 → Site 1
			call(site1, atoms[1]),  # Atom 1 → Site 1
			call(site2, atoms[2]),  # Atom 2 → Site 2
			call(site2, atoms[3]),  # Atom 3 → Site 2
			call(site2, atoms[4])   # Atom 4 → Site 2
		]
		
		# Verify each update_occupation call was made in the expected order
		site_collection.update_occupation.assert_has_calls(expected_calls)
		
		# Verify the number of calls matches the number of atoms
		self.assertEqual(site_collection.update_occupation.call_count, len(atoms))
		
	def test_empty_atoms_list(self):
		"""Test that assign_site_occupations correctly handles empty atom lists."""
		# Create sites with pre-populated contains_atoms
		site1 = DynamicVoronoiSite(reference_indices=[0, 1])
		site1.contains_atoms = [1, 2]
		
		site2 = DynamicVoronoiSite(reference_indices=[2, 3])
		site2.contains_atoms = [3, 4]
		
		# Set the _centre_coords to avoid needing to calculate them
		site1._centre_coords = np.array([0.3, 0.3, 0.3])
		site2._centre_coords = np.array([0.7, 0.7, 0.7])
		
		# Create a collection with these sites
		collection = DynamicVoronoiSiteCollection(sites=[site1, site2])
		
		# Create a test structure
		lattice = Lattice.cubic(10.0)
		structure = Structure(
			lattice=lattice,
			species=["Na", "Na", "Na", "Na"],
			coords=[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9]]
		)
		
		# Call the method with empty atom list
		collection.assign_site_occupations([], structure)
		
		# Verify that contains_atoms was reset for both sites
		self.assertEqual(site1.contains_atoms, [])
		self.assertEqual(site2.contains_atoms, [])
		
if __name__ == '__main__':
	unittest.main()