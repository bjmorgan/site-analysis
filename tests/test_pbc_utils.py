import unittest
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site
from site_analysis.pbc_utils import apply_legacy_pbc_correction, unwrap_vertices_to_reference_center


class TestLegacyPBCCorrection(unittest.TestCase):
	"""Tests for the isolated legacy PBC correction algorithm."""
	
	def test_apply_legacy_pbc_correction_no_correction_needed(self):
		"""Test that coordinates with spread < 0.5 remain unchanged."""
		# All coordinates well within [0, 1), no correction needed
		frac_coords = np.array([
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.2], 
			[0.3, 0.3, 0.3],
			[0.4, 0.4, 0.4]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		# Should remain unchanged
		np.testing.assert_array_equal(result, frac_coords)
		
	def test_apply_legacy_pbc_correction_with_correction_needed(self):
		"""Test coordinates with spread > 0.5 get corrected properly."""
		frac_coords = np.array([
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.9],
			[0.3, 0.3, 0.1],
			[0.4, 0.4, 0.1]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		expected = np.array([
			[0.1, 0.1, 1.1],
			[0.2, 0.2, 0.9],
			[0.3, 0.3, 1.1],
			[0.4, 0.4, 1.1]
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_apply_legacy_pbc_correction_mixed_dimensions(self):
		"""Test correction applied independently to each dimension."""
		frac_coords = np.array([
			[0.1, 0.1, 0.1],
			[0.4, 0.9, 0.2],
			[0.2, 0.1, 0.4]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		expected = np.array([
			[0.1, 1.1, 0.1],
			[0.4, 0.9, 0.2],
			[0.2, 1.1, 0.4]
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_apply_legacy_pbc_correction_negative_coordinates(self):
		"""Test that negative coordinates are handled correctly."""
		frac_coords = np.array([
			[0.1, -0.1, 0.2],
			[0.2, -0.05, 0.3],
			[0.3, 0.0, 0.4],
			[0.4, 0.1, 0.5]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		np.testing.assert_array_equal(result, frac_coords)
	
	def test_apply_legacy_pbc_correction_edge_case_coordinates(self):
		"""Test edge cases with coordinates at exactly 0.0, 0.5, 1.0."""
		frac_coords = np.array([
			[0.0, 0.0, 0.0],
			[0.5, 0.5, 0.5],
			[1.0, 1.0, 1.0]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		expected = np.array([
			[1.0, 1.0, 1.0],
			[0.5, 0.5, 0.5],
			[1.0, 1.0, 1.0]
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_apply_legacy_pbc_correction_all_coordinates_above_half(self):
		"""Test that no shifting occurs when all coordinates > 0.5 even with large spread."""
		frac_coords = np.array([
			[0.6, 0.6, 0.6],
			[0.7, 0.7, 0.7],
			[0.8, 0.8, 0.8],
			[0.9, 0.9, 0.9]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		np.testing.assert_array_equal(result, frac_coords)
	
	def test_apply_legacy_pbc_correction_large_spread_all_above_half(self):
		"""Test large spread with all coordinates > 0.5 - no shifting should occur."""
		frac_coords = np.array([
			[0.6, 0.6, 0.6],
			[0.7, 0.7, 0.7],
			[0.8, 0.8, 0.8],
			[1.2, 1.2, 1.2]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		np.testing.assert_array_equal(result, frac_coords)


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


class TestReferenceBasedUnwrapping(unittest.TestCase):
	"""Tests for reference centre-based vertex unwrapping."""
	
	def setUp(self):
		self.lattice = Lattice.cubic(10.0)
	
	def test_unwrap_vertices_no_unwrapping_needed(self):
		"""Test case where vertices form a tetrahedron around the reference centre."""
		# Simple tetrahedron centered around [0.5, 0.5, 0.5]
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6]
		])
		reference_center = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice)
		
		# Should remain unchanged since all vertices are already close
		np.testing.assert_array_almost_equal(result, vertex_coords, decimal=7)
		
	def test_unwrap_vertices_simple_case(self):
		"""Test unwrapping for an octahedron spanning the boundary at x=0."""
		# Octahedron centered at [0.05, 0.5, 0.5] spanning the x=0 boundary
		vertex_coords = np.array([
			[0.1, 0.5, 0.5],   # +x vertex
			[-0.05, 0.5, 0.5], # -x vertex
			[0.05, 0.6, 0.5],  # +y vertex  
			[0.05, 0.4, 0.5],  # -y vertex
			[0.05, 0.5, 0.6],  # +z vertex
			[0.05, 0.5, 0.4]   # -z vertex
		])
		reference_center = np.array([0.05, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice)
		
		# After unwrapping and shifting to ensure non-negative coordinates
		expected = np.array([
			[1.1, 0.5, 0.5],   # 0.1 + 1.0 (shifted)
			[0.95, 0.5, 0.5],  # -0.05 + 1.0 (unwrapped and shifted)
			[1.05, 0.6, 0.5],  # 0.05 + 1.0 (shifted)
			[1.05, 0.4, 0.5],  # 0.05 + 1.0 (shifted)
			[1.05, 0.5, 0.6],  # 0.05 + 1.0 (shifted)
			[1.05, 0.5, 0.4]   # 0.05 + 1.0 (shifted)
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_unwrap_vertices_pathological_case(self):
		"""Test octahedron that would break the legacy spread-based algorithm."""
		vertex_coords = np.array([
			[0.55, 0.25, 0.25],  
			[0.95, 0.25, 0.25],  
			[0.25, 0.95, 0.25],  
			[0.25, 0.55, 0.25],  
			[0.25, 0.25, 0.55],   
			[0.25, 0.25, 0.95]    
		])
		reference_center = np.array([0.25, 0.25, 0.25])
		expected_result_centre = np.array([1.25, 1.25, 1.25])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice)
		
		# The centre of the unwrapped vertices should be close to the reference centre
		result_centre = np.mean(result, axis=0)
		np.testing.assert_array_almost_equal(result_centre, expected_result_centre, decimal=2)
	
	def test_unwrap_vertices_tetrahedral_coordination(self):
		"""Test 4-coordinate tetrahedral case."""
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6]
		])
		reference_center = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice)
		
		# Should remain unchanged since all vertices are already close
		np.testing.assert_array_almost_equal(result, vertex_coords, decimal=7)
	
	def test_unwrap_vertices_empty_input(self):
		"""Test that empty input is handled correctly."""
		vertex_coords = np.array([]).reshape(0, 3)
		reference_center = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice)
		
		# Should return empty array with correct shape
		self.assertEqual(result.shape, (0, 3))

if __name__ == '__main__':
	unittest.main()