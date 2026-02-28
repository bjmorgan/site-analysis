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
		sites = [Mock(spec=DynamicVoronoiSite, index=0,
					  reference_indices=[0, 1], reference_center=None),
				 Mock(spec=DynamicVoronoiSite, index=1,
					  reference_indices=[2, 3], reference_center=None)]
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
		"""Test that analyse_structure computes centres and assigns occupations."""
		site1 = DynamicVoronoiSite(reference_indices=[0, 1])
		site2 = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site1, site2])

		lattice = Lattice.cubic(10.0)
		# Atoms 0,1 are reference for site1; 2,3 for site2; 4 is mobile
		coords = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
				  [0.7, 0.7, 0.7], [0.8, 0.8, 0.8],
				  [0.15, 0.15, 0.15]]
		structure = Structure(lattice, ["Na"] * 5, coords)

		atom = Atom(index=4)
		collection.analyse_structure([atom], structure)

		# Centres should be computed from reference atoms
		np.testing.assert_array_almost_equal(
			site1.centre, np.array([0.15, 0.15, 0.15]))
		np.testing.assert_array_almost_equal(
			site2.centre, np.array([0.75, 0.75, 0.75]))

		# Atom at [0.15, 0.15, 0.15] should be assigned to site1
		self.assertIn(atom.index, site1.contains_atoms)
			
	def test_assign_site_occupations(self):
		"""Test that atoms are correctly assigned to sites based on Voronoi tessellation."""
		# Create mock sites
		site1 = Mock(spec=DynamicVoronoiSite,
					 reference_indices=[0, 1], reference_center=None)
		site2 = Mock(spec=DynamicVoronoiSite,
					 reference_indices=[2, 3], reference_center=None)
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
		
class BatchCentreCalculationTestCase(unittest.TestCase):

	def setUp(self):
		Site._newid = 0

	def test_groups_formed_by_reference_count(self):
		"""Sites with different n_reference are placed in separate groups."""
		sites = [
			DynamicVoronoiSite(reference_indices=[0, 1]),
			DynamicVoronoiSite(reference_indices=[2, 3]),
			DynamicVoronoiSite(reference_indices=[4, 5, 6]),
		]
		collection = DynamicVoronoiSiteCollection(sites=sites)
		groups = collection._centre_groups
		self.assertEqual(len(groups), 2)
		group_sizes = sorted(len(g.site_positions) for g in groups)
		self.assertEqual(group_sizes, [1, 2])

	def test_batch_produces_same_centres_as_per_site(self):
		"""Batch path should produce identical centres to per-site calculation."""
		lattice = Lattice.cubic(10.0)
		coords = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
				  [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
		structure = Structure(lattice, ["Na"] * 4, coords)

		# Per-site path
		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		site_b = DynamicVoronoiSite(reference_indices=[2, 3])
		site_a.calculate_centre(structure)
		site_b.calculate_centre(structure)

		# Batch path
		site_c = DynamicVoronoiSite(reference_indices=[0, 1])
		site_d = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site_c, site_d])
		collection._batch_calculate_centres(structure.frac_coords, structure.lattice)

		np.testing.assert_array_almost_equal(site_a.centre, site_c.centre)
		np.testing.assert_array_almost_equal(site_b.centre, site_d.centre)

	def test_second_frame_uses_cached_batch_path(self):
		"""After first frame, batch path should not call _compute_corrected_coords."""
		lattice = Lattice.cubic(10.0)
		coords1 = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
				   [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
		coords2 = [[0.11, 0.11, 0.11], [0.21, 0.21, 0.21],
				   [0.71, 0.71, 0.71], [0.81, 0.81, 0.81]]
		struct1 = Structure(lattice, ["Na"] * 4, coords1)
		struct2 = Structure(lattice, ["Na"] * 4, coords2)

		site1 = DynamicVoronoiSite(reference_indices=[0, 1])
		site2 = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site1, site2])

		# First frame populates caches
		collection._batch_calculate_centres(struct1.frac_coords, struct1.lattice)
		self.assertTrue(collection._centre_groups[0].initialised)

		# Second frame should use vectorised path (not per-site)
		with patch.object(site1, '_compute_corrected_coords') as mock1, \
			 patch.object(site2, '_compute_corrected_coords') as mock2:
			collection._batch_calculate_centres(struct2.frac_coords, struct2.lattice)
			mock1.assert_not_called()
			mock2.assert_not_called()

		# Centres should still be updated
		self.assertIsNotNone(site1._centre_coords)
		self.assertIsNotNone(site2._centre_coords)

	def test_reset_centre_groups_clears_batch_state(self):
		"""reset_centre_groups should force full recomputation on next frame."""
		lattice = Lattice.cubic(10.0)
		coords = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
		structure = Structure(lattice, ["Na"] * 2, coords)

		site = DynamicVoronoiSite(reference_indices=[0, 1])
		collection = DynamicVoronoiSiteCollection(sites=[site])

		collection._batch_calculate_centres(structure.frac_coords, structure.lattice)
		self.assertTrue(collection._centre_groups[0].initialised)

		collection.reset_centre_groups()
		self.assertFalse(collection._centre_groups[0].initialised)

	def test_multi_frame_centres_match_per_site(self):
		"""Batch centres over multiple frames should match per-site computation."""
		lattice = Lattice.cubic(10.0)
		base_coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
								[0.7, 0.7, 0.7], [0.8, 0.8, 0.8]])

		# Per-site path
		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		site_b = DynamicVoronoiSite(reference_indices=[2, 3])
		# Batch path
		site_c = DynamicVoronoiSite(reference_indices=[0, 1])
		site_d = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site_c, site_d])

		for i in range(5):
			coords = base_coords + 0.01 * i
			structure = Structure(lattice, ["Na"] * 4, coords.tolist())
			frac = structure.frac_coords

			site_a._compute_corrected_coords(frac[[0, 1]], structure.lattice)
			site_b._compute_corrected_coords(frac[[2, 3]], structure.lattice)
			collection._batch_calculate_centres(frac, structure.lattice)

			np.testing.assert_array_almost_equal(site_a.centre, site_c.centre)
			np.testing.assert_array_almost_equal(site_b.centre, site_d.centre)


if __name__ == '__main__':
	unittest.main()