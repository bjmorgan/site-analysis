import unittest
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site


class TestCurrentPBCBehaviorRegression(unittest.TestCase):
	"""Regression tests to preserve correct current PBC behavior during refactoring."""

	def setUp(self):
		"""Set up test fixtures."""
		Site._newid = 0
		self.lattice = Lattice.cubic(10.0)

	def test_polyhedral_site_no_pbc_correction_needed(self):
		"""Test case where no PBC correction is needed - should remain unchanged."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.2],
			[0.3, 0.3, 0.3],
			[0.4, 0.4, 0.4]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)
		
		# Should remain unchanged (this works correctly)
		expected_coords = np.array(coords)
		np.testing.assert_array_almost_equal(site.vertex_coords, expected_coords, decimal=7)

	def test_dynamic_voronoi_site_no_pbc_correction_needed(self):
		"""Test case where no PBC correction is needed - should calculate centre correctly."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.2],
			[0.3, 0.3, 0.3],
			[0.4, 0.4, 0.4]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure)
		
		# Should calculate centre correctly (this works correctly)
		expected_centre = np.mean(coords, axis=0)
		np.testing.assert_array_almost_equal(site.centre, expected_centre, decimal=7)

	def test_polyhedral_site_working_pbc_case(self):
		"""Test a case where current PBC correction works correctly."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.9],   # Large z-coordinate, should trigger correction
			[0.3, 0.3, 0.1],
			[0.4, 0.4, 0.1]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)
		
		# Current algorithm should correctly shift small z-coordinates
		expected_coords = np.array([
			[0.1, 0.1, 1.1],   # z < 0.5, gets +1.0
			[0.2, 0.2, 0.9],   # z >= 0.5, unchanged  
			[0.3, 0.3, 1.1],   # z < 0.5, gets +1.0
			[0.4, 0.4, 1.1]    # z < 0.5, gets +1.0
		])
		np.testing.assert_array_almost_equal(site.vertex_coords, expected_coords, decimal=7)

	def test_dynamic_voronoi_site_working_pbc_case(self):
		"""Test a case where current PBC correction works correctly."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.9],   # Large z-coordinate, should trigger correction
			[0.3, 0.3, 0.1],
			[0.4, 0.4, 0.1]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure)
		
		# DynamicVoronoiSite applies correction AND wraps centre back to [0,1)
		corrected_coords = np.array([
			[0.1, 0.1, 1.1],
			[0.2, 0.2, 0.9],   
			[0.3, 0.3, 1.1],
			[0.4, 0.4, 1.1]
		])
		centre_before_wrap = np.mean(corrected_coords, axis=0)
		expected_centre = centre_before_wrap % 1.0  # DynamicVoronoiSite wraps back
		np.testing.assert_array_almost_equal(site.centre, expected_centre, decimal=7)

	def test_polyhedral_site_negative_coordinates_no_correction(self):
		"""Test negative coordinates that don't require PBC correction."""
		species = ["Re"] * 4
		coords = [
			[0.1, -0.1, 0.2],   # Negative y-coordinate
			[0.2, -0.05, 0.3],  # y-spread: 0.1 - (-0.1) = 0.2 < 0.5
			[0.3, 0.0, 0.4],
			[0.4, 0.1, 0.5]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)
		
		# Should remain unchanged (spread < 0.5, no correction needed)
		expected_coords = np.array(coords)
		np.testing.assert_array_almost_equal(site.vertex_coords, expected_coords, decimal=7)

	def test_dynamic_voronoi_site_negative_coordinates_no_correction(self):
		"""Test negative coordinates that don't require PBC correction."""
		species = ["Re"] * 4
		coords = [
			[0.1, -0.1, 0.2],   # Negative y-coordinate
			[0.2, -0.05, 0.3],  # y-spread: 0.1 - (-0.1) = 0.2 < 0.5
			[0.3, 0.0, 0.4],
			[0.4, 0.1, 0.5]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure)
		
		# Should calculate centre correctly (no correction needed)
		expected_centre = np.mean(coords, axis=0) % 1.0
		np.testing.assert_array_almost_equal(site.centre, expected_centre, decimal=7)


if __name__ == '__main__':
	unittest.main()