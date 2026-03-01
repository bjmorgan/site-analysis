import unittest
import numpy as np
import pytest
from pymatgen.core import Structure, Lattice

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis._compat import HAS_NUMBA
from site_analysis.site import Site
from unittest.mock import patch
from site_analysis.pbc_utils import apply_legacy_pbc_correction, unwrap_vertices_to_reference_center, correct_pbc


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
			[0.1, 0.3, 0.3],
			[0.3, 0.3, 0.1],
			[0.3, 0.1, 0.3]
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
			[0.3, 0.1, 0.9],   # Large z-coordinate, should trigger correction
			[0.1, 0.3, 0.1],
			[0.3, 0.3, 0.1]
		]
		structure = Structure(self.lattice, species, coords)

		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)

		# Current algorithm should correctly shift small z-coordinates
		expected_coords = np.array([
			[0.1, 0.1, 1.1],   # z < 0.5, gets +1.0
			[0.3, 0.1, 0.9],   # z >= 0.5, unchanged
			[0.1, 0.3, 1.1],   # z < 0.5, gets +1.0
			[0.3, 0.3, 1.1]    # z < 0.5, gets +1.0
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
			[0.2, -0.1, 0.2],    # Negative y-coordinate
			[0.2, 0.1, 0.4],     # y-spread: 0.1 - (-0.1) = 0.2 < 0.5
			[0.4, -0.1, 0.4],
			[0.3, 0.1, 0.2]
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
	
	def test_return_image_shifts_returns_tuple(self):
		"""Test that return_image_shifts=True returns (coords, shifts) tuple."""
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6]
		])
		reference_center = np.array([0.5, 0.5, 0.5])
		result = unwrap_vertices_to_reference_center(
			vertex_coords, reference_center, self.lattice, return_image_shifts=True)
		self.assertIsInstance(result, tuple)
		self.assertEqual(len(result), 2)
		coords, shifts = result
		self.assertEqual(coords.shape, (4, 3))
		self.assertEqual(shifts.shape, (4, 3))
		# Shifts should be integers
		np.testing.assert_array_equal(shifts, shifts.astype(int))

	def test_return_image_shifts_false_returns_array(self):
		"""Test that return_image_shifts=False returns just coordinates."""
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
		])
		reference_center = np.array([0.5, 0.5, 0.5])
		result = unwrap_vertices_to_reference_center(
			vertex_coords, reference_center, self.lattice, return_image_shifts=False)
		self.assertIsInstance(result, np.ndarray)
		self.assertEqual(result.shape, (2, 3))

	def test_return_image_shifts_consistent_with_coords(self):
		"""Image shifts applied to raw coords should match the returned coordinates."""
		vertex_coords = np.array([
			[0.05, 0.5, 0.5],
			[0.95, 0.5, 0.5],  # needs shift of -1 in x
		])
		reference_center = np.array([0.0, 0.5, 0.5])
		coords, shifts = unwrap_vertices_to_reference_center(
			vertex_coords, reference_center, self.lattice, return_image_shifts=True)
		# coords = raw + shifts + uniform, so (coords - uniform) - raw = shifts
		shifted_raw = vertex_coords + shifts
		min_coords = np.min(shifted_raw, axis=0)
		uniform = np.maximum(0, np.ceil(-min_coords))
		np.testing.assert_array_almost_equal(coords, shifted_raw + uniform)

	def test_unwrap_vertices_empty_input(self):
		"""Test that empty input is handled correctly."""
		vertex_coords = np.array([]).reshape(0, 3)
		reference_center = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice)
		
		# Should return empty array with correct shape
		self.assertEqual(result.shape, (0, 3))

class TestCorrectPbc(unittest.TestCase):
	"""Tests for the correct_pbc dispatch function."""

	def setUp(self):
		self.lattice = Lattice.cubic(10.0)

	def test_delegates_to_legacy_when_no_reference_center(self):
		"""With no reference centre, delegates to apply_legacy_pbc_correction."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		with patch('site_analysis.pbc_utils.apply_legacy_pbc_correction',
				   return_value=frac_coords.copy()) as mock_legacy:
			correct_pbc(frac_coords, None, self.lattice)
			mock_legacy.assert_called_once()

	def test_delegates_to_unwrap_when_reference_center_provided(self):
		"""With a reference centre, delegates to unwrap_vertices_to_reference_center."""
		frac_coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
		ref = np.array([0.15, 0.15, 0.15])
		with patch('site_analysis.pbc_utils.unwrap_vertices_to_reference_center',
				   return_value=(frac_coords.copy(), np.zeros((2, 3), dtype=np.int64))) as mock_unwrap:
			correct_pbc(frac_coords, ref, self.lattice)
			mock_unwrap.assert_called_once()
			self.assertTrue(mock_unwrap.call_args.kwargs.get('return_image_shifts'))

	def test_returns_int64_shifts_legacy_path(self):
		"""Image shifts have int64 dtype for the legacy path."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		_, shifts = correct_pbc(frac_coords, None, self.lattice)
		self.assertEqual(shifts.dtype, np.int64)

	def test_returns_int64_shifts_reference_centre_path(self):
		"""Image shifts have int64 dtype for the reference-centre path."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		ref = np.array([0.15, 0.15, 0.5])
		_, shifts = correct_pbc(frac_coords, ref, self.lattice)
		self.assertTrue(np.issubdtype(shifts.dtype, np.integer))

	def test_empty_input_returns_empty_arrays(self):
		"""Empty input should return empty arrays without error."""
		frac_coords = np.array([]).reshape(0, 3)
		corrected, shifts = correct_pbc(frac_coords, None, self.lattice)
		self.assertEqual(corrected.shape, (0, 3))
		self.assertEqual(shifts.shape, (0, 3))
		self.assertEqual(shifts.dtype, np.int64)

	def test_legacy_path_returns_consistent_shifts(self):
		"""Shifts from the legacy path satisfy corrected = original + shifts."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		corrected, shifts = correct_pbc(frac_coords, None, self.lattice)
		expected_shifts = np.round(corrected - frac_coords).astype(np.int64)
		np.testing.assert_array_equal(shifts, expected_shifts)


class TestNumpyUpdatePbcShifts(unittest.TestCase):
	"""Tests for _numpy_update_pbc_shifts."""

	def test_cache_hit_small_displacement(self):
		"""Small physical displacement returns valid cache."""
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached + 0.01  # small vibration
		shifts = np.array([[0, 0, 1],
						   [0, 1, 0]], dtype=np.int64)
		valid, coords, new_shifts = _numpy_update_pbc_shifts(new, cached, shifts)
		self.assertTrue(valid)
		np.testing.assert_array_equal(new_shifts, shifts)
		expected = new + shifts
		min_coords = np.min(expected, axis=0)
		uniform = np.maximum(0, np.ceil(-min_coords))
		np.testing.assert_array_almost_equal(coords, expected + uniform)

	def test_cache_hit_with_wrapping(self):
		"""Vertex wrapping from 0.99 to 0.01 is detected and shifts adjusted."""
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts
		cached = np.array([[0.99, 0.5, 0.5],
						   [0.5, 0.5, 0.5]])
		new = np.array([[0.01, 0.5, 0.5],   # wrapped across boundary
						[0.5, 0.5, 0.5]])
		shifts = np.array([[0, 0, 0],
						   [0, 0, 0]], dtype=np.int64)
		valid, coords, new_shifts = _numpy_update_pbc_shifts(new, cached, shifts)
		self.assertTrue(valid)
		# Wrapping of -0.98 rounds to -1, so shift gains +1
		np.testing.assert_array_equal(new_shifts[0], [1, 0, 0])
		np.testing.assert_array_equal(new_shifts[1], [0, 0, 0])

	def test_cache_miss_large_displacement(self):
		"""Large physical displacement invalidates cache."""
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached + 0.4  # too large
		shifts = np.zeros((2, 3), dtype=np.int64)
		valid, _, _ = _numpy_update_pbc_shifts(new, cached, shifts)
		self.assertFalse(valid)

	def test_non_negative_shift_applied(self):
		"""Uniform shift ensures all output coordinates are non-negative."""
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts
		cached = np.array([[0.05, 0.05, 0.05],
						   [0.1, 0.1, 0.1]])
		new = cached + 0.001
		shifts = np.array([[0, 0, -1],
						   [0, 0, -1]], dtype=np.int64)
		valid, coords, _ = _numpy_update_pbc_shifts(new, cached, shifts)
		self.assertTrue(valid)
		self.assertTrue(np.all(coords >= 0))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestNumbaUpdatePbcShifts(unittest.TestCase):
	"""Tests that numba and numpy PBC shift implementations agree."""

	def test_agrees_with_numpy_small_displacement(self):
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts, _numba_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached + 0.01
		shifts = np.array([[0, 0, 1],
						   [0, 1, 0]], dtype=np.int64)
		v_np, c_np, s_np = _numpy_update_pbc_shifts(new, cached, shifts)
		v_nb, c_nb, s_nb = _numba_update_pbc_shifts(new, cached, shifts)
		self.assertEqual(v_np, v_nb)
		np.testing.assert_array_almost_equal(c_np, c_nb)
		np.testing.assert_array_equal(s_np, s_nb)

	def test_agrees_with_numpy_wrapping(self):
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts, _numba_update_pbc_shifts
		cached = np.array([[0.99, 0.5, 0.5],
						   [0.5, 0.5, 0.5]])
		new = np.array([[0.01, 0.5, 0.5],
						[0.5, 0.5, 0.5]])
		shifts = np.zeros((2, 3), dtype=np.int64)
		v_np, c_np, s_np = _numpy_update_pbc_shifts(new, cached, shifts)
		v_nb, c_nb, s_nb = _numba_update_pbc_shifts(new, cached, shifts)
		self.assertEqual(v_np, v_nb)
		np.testing.assert_array_almost_equal(c_np, c_nb)
		np.testing.assert_array_equal(s_np, s_nb)

	def test_agrees_with_numpy_cache_miss(self):
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts, _numba_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached + 0.4
		shifts = np.zeros((2, 3), dtype=np.int64)
		v_np, _, _ = _numpy_update_pbc_shifts(new, cached, shifts)
		v_nb, _, _ = _numba_update_pbc_shifts(new, cached, shifts)
		self.assertEqual(v_np, v_nb)


if __name__ == '__main__':
	unittest.main()
