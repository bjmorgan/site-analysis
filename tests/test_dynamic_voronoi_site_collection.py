import unittest
from unittest.mock import Mock, patch
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
		"""Test that atoms are assigned to the nearest site centre."""
		site1 = DynamicVoronoiSite(reference_indices=[0, 1])
		site2 = DynamicVoronoiSite(reference_indices=[2, 3])
		site1._centre_coords = np.array([0.2, 0.2, 0.2])
		site2._centre_coords = np.array([0.8, 0.8, 0.8])

		collection = DynamicVoronoiSiteCollection(sites=[site1, site2])

		lattice = Lattice.cubic(10.0)
		# 4 reference atoms + 2 mobile atoms
		coords = [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3],
				  [0.7, 0.7, 0.7], [0.9, 0.9, 0.9],
				  [0.15, 0.15, 0.15], [0.85, 0.85, 0.85]]
		structure = Structure(lattice, ["Na"] * 6, coords)

		atom1 = Atom(index=4)
		atom2 = Atom(index=5)
		atom1.assign_coords(structure)
		atom2.assign_coords(structure)

		collection.assign_site_occupations([atom1, atom2], structure)

		self.assertIn(atom1.index, site1.contains_atoms)
		self.assertIn(atom2.index, site2.contains_atoms)
		
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

	def test_reset_clears_batch_state(self):
		"""reset should clear group initialised flag."""
		lattice = Lattice.cubic(10.0)
		coords = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
		structure = Structure(lattice, ["Na"] * 2, coords)

		site = DynamicVoronoiSite(reference_indices=[0, 1])
		collection = DynamicVoronoiSiteCollection(sites=[site])

		collection._batch_calculate_centres(structure.frac_coords, structure.lattice)
		self.assertTrue(collection._centre_groups[0].initialised)

		collection.reset()
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

	def test_site_reset_without_collection_reset_triggers_fallback(self):
		"""site.reset() without collection.reset() leaves stale group caches.
		A large coordinate change should trigger fallback and still produce
		correct centres."""
		lattice = Lattice.cubic(10.0)
		base_coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
								[0.7, 0.7, 0.7], [0.8, 0.8, 0.8]])

		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		site_b = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site_a, site_b])

		# Run several frames
		for i in range(5):
			coords = base_coords + 0.01 * i
			structure = Structure(lattice, ["Na"] * 4, coords.tolist())
			collection._batch_calculate_centres(structure.frac_coords, structure.lattice)

		# Only reset sites, not collection — group caches are stale
		for site in collection.sites:
			site.reset()
		self.assertTrue(collection._centre_groups[0].initialised)

		# New coords far from cached values (> 0.3 displacement triggers fallback)
		new_base = np.array([[0.6, 0.6, 0.6], [0.7, 0.7, 0.7],
							 [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
		ref_a = DynamicVoronoiSite(reference_indices=[0, 1])
		ref_b = DynamicVoronoiSite(reference_indices=[2, 3])

		for i in range(3):
			coords = new_base + 0.01 * i
			structure = Structure(lattice, ["Na"] * 4, coords.tolist())
			frac = structure.frac_coords

			ref_a._compute_corrected_coords(frac[[0, 1]], structure.lattice)
			ref_b._compute_corrected_coords(frac[[2, 3]], structure.lattice)
			collection._batch_calculate_centres(frac, structure.lattice)

			np.testing.assert_array_almost_equal(ref_a.centre, site_a.centre)
			np.testing.assert_array_almost_equal(ref_b.centre, site_b.centre)

	def test_batch_invalidation_falls_back_to_per_site(self):
		"""Large displacement should invalidate batch cache and recompute per-site."""
		lattice = Lattice.cubic(10.0)
		coords1 = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
				   [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
		# Displacement > 0.3 triggers invalidation
		coords2 = [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6],
				   [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
		struct1 = Structure(lattice, ["Na"] * 4, coords1)
		struct2 = Structure(lattice, ["Na"] * 4, coords2)

		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		site_b = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site_a, site_b])

		collection._batch_calculate_centres(struct1.frac_coords, struct1.lattice)

		# Fresh per-site reference for the post-invalidation frame
		ref_a = DynamicVoronoiSite(reference_indices=[0, 1])
		ref_b = DynamicVoronoiSite(reference_indices=[2, 3])
		ref_a.calculate_centre(struct2)
		ref_b.calculate_centre(struct2)

		collection._batch_calculate_centres(struct2.frac_coords, struct2.lattice)

		np.testing.assert_array_almost_equal(ref_a.centre, site_a.centre)
		np.testing.assert_array_almost_equal(ref_b.centre, site_b.centre)

	def test_batch_wrapping_across_periodic_boundary(self):
		"""Coordinates wrapping across boundary (0.99 -> 0.01) should be handled correctly."""
		lattice = Lattice.cubic(10.0)
		coords1 = [[0.99, 0.1, 0.1], [0.2, 0.1, 0.1],
				   [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
		coords2 = [[0.01, 0.1, 0.1], [0.2, 0.1, 0.1],  # atom 0 wraps
				   [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]]
		struct1 = Structure(lattice, ["Na"] * 4, coords1)
		struct2 = Structure(lattice, ["Na"] * 4, coords2)

		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		site_b = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site_a, site_b])

		# Fresh per-site reference
		ref_a = DynamicVoronoiSite(reference_indices=[0, 1])
		ref_b = DynamicVoronoiSite(reference_indices=[2, 3])

		for struct in [struct1, struct2]:
			frac = struct.frac_coords
			ref_a._compute_corrected_coords(frac[[0, 1]], struct.lattice)
			ref_b._compute_corrected_coords(frac[[2, 3]], struct.lattice)
			collection._batch_calculate_centres(frac, struct.lattice)

		np.testing.assert_array_almost_equal(ref_a.centre, site_a.centre)
		np.testing.assert_array_almost_equal(ref_b.centre, site_b.centre)

	def test_mixed_n_reference_groups_produce_correct_centres(self):
		"""Batch with multiple groups (different n_reference) should produce correct centres."""
		lattice = Lattice.cubic(10.0)
		# 7 atoms: sites use different subsets
		base_coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
								[0.5, 0.5, 0.5], [0.6, 0.6, 0.6],
								[0.8, 0.8, 0.8], [0.3, 0.3, 0.3],
								[0.4, 0.4, 0.4]])

		# 2-ref group
		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		# 3-ref group
		site_b = DynamicVoronoiSite(reference_indices=[2, 3, 4])
		# 2-ref group (same group as site_a)
		site_c = DynamicVoronoiSite(reference_indices=[5, 6])

		collection = DynamicVoronoiSiteCollection(sites=[site_a, site_b, site_c])
		self.assertEqual(len(collection._centre_groups), 2)

		# Fresh per-site references
		ref_a = DynamicVoronoiSite(reference_indices=[0, 1])
		ref_b = DynamicVoronoiSite(reference_indices=[2, 3, 4])
		ref_c = DynamicVoronoiSite(reference_indices=[5, 6])

		for i in range(5):
			coords = base_coords + 0.01 * i
			structure = Structure(lattice, ["Na"] * 7, coords.tolist())
			frac = structure.frac_coords

			ref_a._compute_corrected_coords(frac[[0, 1]], structure.lattice)
			ref_b._compute_corrected_coords(frac[[2, 3, 4]], structure.lattice)
			ref_c._compute_corrected_coords(frac[[5, 6]], structure.lattice)
			collection._batch_calculate_centres(frac, structure.lattice)

			np.testing.assert_array_almost_equal(ref_a.centre, site_a.centre)
			np.testing.assert_array_almost_equal(ref_b.centre, site_b.centre)
			np.testing.assert_array_almost_equal(ref_c.centre, site_c.centre)

	def test_reset_then_reuse_produces_correct_centres(self):
		"""After reset, batch centres on new data should match per-site computation."""
		lattice = Lattice.cubic(10.0)
		base_coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
								[0.7, 0.7, 0.7], [0.8, 0.8, 0.8]])

		site_a = DynamicVoronoiSite(reference_indices=[0, 1])
		site_b = DynamicVoronoiSite(reference_indices=[2, 3])
		collection = DynamicVoronoiSiteCollection(sites=[site_a, site_b])

		# Run several frames to build up cache state
		for i in range(5):
			coords = base_coords + 0.01 * i
			structure = Structure(lattice, ["Na"] * 4, coords.tolist())
			collection._batch_calculate_centres(structure.frac_coords, structure.lattice)

		collection.reset()

		# Run on different coordinates after reset
		new_base = np.array([[0.4, 0.4, 0.4], [0.5, 0.5, 0.5],
							 [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
		# Fresh per-site reference
		ref_a = DynamicVoronoiSite(reference_indices=[0, 1])
		ref_b = DynamicVoronoiSite(reference_indices=[2, 3])

		for i in range(3):
			coords = new_base + 0.01 * i
			structure = Structure(lattice, ["Na"] * 4, coords.tolist())
			frac = structure.frac_coords

			ref_a._compute_corrected_coords(frac[[0, 1]], structure.lattice)
			ref_b._compute_corrected_coords(frac[[2, 3]], structure.lattice)
			collection._batch_calculate_centres(frac, structure.lattice)

			np.testing.assert_array_almost_equal(ref_a.centre, site_a.centre)
			np.testing.assert_array_almost_equal(ref_b.centre, site_b.centre)


if __name__ == '__main__':
	unittest.main()