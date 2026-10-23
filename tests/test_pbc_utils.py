import unittest
import numpy as np
import pytest
from pymatgen.core import Structure, Lattice

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis._compat import HAS_NUMBA
from site_analysis.site import Site
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
	"""Regression tests to preserve correct current PBC behavior during refactoring.

	The sites have no reference centres (deprecated) and use the spread-based rule.
	"""

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

		with self.assertWarns(FutureWarning):
			site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)

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
		
		with self.assertWarns(FutureWarning):
			site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure.frac_coords, structure.lattice.matrix)
		
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

		with self.assertWarns(FutureWarning):
			site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)

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
		
		with self.assertWarns(FutureWarning):
			site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure.frac_coords, structure.lattice.matrix)
		
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

	def test_polyhedral_site_negative_coordinates_translated_into_unit_cell(self):
		"""Test negative coordinates that don't require spread correction are
		translated by whole cells into the unit cell."""
		species = ["Re"] * 4
		coords = [
			[0.2, -0.1, 0.2],    # Negative y-coordinate
			[0.2, 0.1, 0.4],     # y-spread: 0.1 - (-0.1) = 0.2 < 0.5
			[0.4, -0.1, 0.4],
			[0.3, 0.1, 0.2]
		]
		structure = Structure(self.lattice, species, coords)

		with self.assertWarns(FutureWarning):
			site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)

		# Spread < 0.5, so no spread correction; the vertices are only
		# translated by a whole cell in y so the lowest y coordinate is in [0, 1)
		expected_coords = np.array(coords) + np.array([0.0, 1.0, 0.0])
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
		
		with self.assertWarns(FutureWarning):
			site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure.frac_coords, structure.lattice.matrix)
		
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
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice.matrix)
		
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
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice.matrix)
		
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
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice.matrix)
		
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
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice.matrix)
		
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
			vertex_coords, reference_center, self.lattice.matrix, return_image_shifts=True)
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
			vertex_coords, reference_center, self.lattice.matrix, return_image_shifts=False)
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
			vertex_coords, reference_center, self.lattice.matrix, return_image_shifts=True)
		np.testing.assert_array_equal(shifts, [[0, 0, 0], [-1, 0, 0]])
		np.testing.assert_array_almost_equal(coords, [[1.05, 0.5, 0.5], [0.95, 0.5, 0.5]])

	def test_unwrap_vertices_empty_input(self):
		"""Test that empty input is handled correctly."""
		vertex_coords = np.array([]).reshape(0, 3)
		reference_center = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_center(vertex_coords, reference_center, self.lattice.matrix)
		
		# Should return empty array with correct shape
		self.assertEqual(result.shape, (0, 3))

	def test_reference_centre_several_cells_away(self):
		"""Moving the reference centre by whole cells moves the image shifts by the same amount."""
		vertex_coords = np.array([
			[0.1, 0.5, 0.5],
			[0.95, 0.5, 0.5],
			[0.05, 0.6, 0.5],
			[0.05, 0.4, 0.5],
		])
		reference_center = np.array([0.05, 0.5, 0.5])
		translation = np.array([2, -3, 1])
		_, shifts = unwrap_vertices_to_reference_center(
			vertex_coords, reference_center, self.lattice.matrix, return_image_shifts=True)
		_, shifts_translated = unwrap_vertices_to_reference_center(
			vertex_coords, reference_center + translation, self.lattice.matrix,
			return_image_shifts=True)
		np.testing.assert_array_equal(shifts_translated - shifts, np.tile(translation, (4, 1)))

	def test_vertices_several_cells_away(self):
		"""Image shifts undo whole-cell translations of the vertices relative to the centre."""
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6],
		])
		reference_center = np.array([0.5, 0.5, 0.5])
		translation = np.array([-2, 0, 3])
		coords, shifts = unwrap_vertices_to_reference_center(
			vertex_coords + translation, reference_center, self.lattice.matrix,
			return_image_shifts=True)
		np.testing.assert_array_equal(shifts, np.tile(-translation, (4, 1)))
		np.testing.assert_array_almost_equal(coords, vertex_coords)

	def test_minimum_coordinate_translated_into_the_unit_cell(self):
		"""Whole-cell translation puts the lowest coordinate along each axis in [0, 1)."""
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6],
		])
		reference_center = np.array([0.5, 0.5, 0.5])
		for translation in ([2, 0, -2], [-1, 3, 0]):
			with self.subTest(translation=translation):
				coords = unwrap_vertices_to_reference_center(
					vertex_coords, reference_center + translation, self.lattice.matrix)
				np.testing.assert_array_almost_equal(coords.min(axis=0), [0.4, 0.4, 0.4])


class TestExactUnwrapping(unittest.TestCase):
	"""Tests that unwrapping finds the nearest image exactly in an unreduced cell."""

	# Rows (1,0,0), (3,1,0), (0,0,1): an unreduced basis.
	UNREDUCED = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

	def test_unreduced_cell_picks_the_nearest_image(self):
		"""The nearest image of (0.6, 0.6, 0.5) to (0, 0, 0.5) is shifted by (1, -1, 0)."""
		_, shifts = unwrap_vertices_to_reference_center(
			np.array([[0.6, 0.6, 0.5]]), np.array([0.0, 0.0, 0.5]),
			self.UNREDUCED, return_image_shifts=True)
		np.testing.assert_array_equal(shifts, [[1, -1, 0]])

	def test_vertex_half_a_cell_from_the_centre_unwraps_the_same_in_any_image(self):
		"""A vertex exactly half a cell from the centre is unwrapped to the same image whichever whole-cell offset it is given with."""
		cubic = np.eye(3) * 4.0
		centre = np.array([0.25, 0.25, 0.25])
		vertices = np.array([[0.75, 0.25, 0.75], [0.35, 0.25, 0.25]])
		reference = unwrap_vertices_to_reference_center(vertices, centre, cubic)
		for offset in ([1, 0, 0], [-1, 0, 0], [3, 0, -1], [-1, 2, 1]):
			with self.subTest(offset=offset):
				shifted = vertices + np.array([offset, [0, 0, 0]])
				unwrapped = unwrap_vertices_to_reference_center(shifted, centre, cubic)
				np.testing.assert_array_equal(unwrapped, reference)

	def test_vertex_several_cells_away_in_an_unreduced_cell(self):
		"""Whole-cell offsets of the vertex change the shift by the opposite amount."""
		for offset in ([5, 0, 0], [-2, 5, 3], [0, -5, 0]):
			with self.subTest(offset=offset):
				_, shifts = unwrap_vertices_to_reference_center(
					np.array([[0.6, 0.6, 0.5]]) + offset, np.array([0.0, 0.0, 0.5]),
					self.UNREDUCED, return_image_shifts=True)
				np.testing.assert_array_equal(shifts, [np.array([1, -1, 0]) - offset])

	def test_non_finite_coordinates_raise(self):
		"""Non-finite vertex coordinates are rejected by name."""
		vertices = np.array([[0.1, 0.2, 0.3], [np.nan, 0.2, 0.3]])
		with self.assertRaisesRegex(ValueError, "^frac_coords must be finite$"):
			unwrap_vertices_to_reference_center(
				vertices, np.array([0.5, 0.5, 0.5]), Lattice.cubic(10.0).matrix)

	def test_non_finite_reference_center_raises(self):
		"""A non-finite reference centre is rejected by name."""
		with self.assertRaisesRegex(
				ValueError, "^reference_center must be three finite numbers, got "):
			unwrap_vertices_to_reference_center(
				np.array([[0.1, 0.2, 0.3]]), np.array([np.nan, 0.5, 0.5]),
				Lattice.cubic(10.0).matrix)

	def test_vertex_too_far_from_the_reference_centre_raises(self):
		"""A displacement too large for the search is rejected by the user's argument names."""
		cases = {
			"2**52 cells away": (np.array([[2.0 ** 52, 0.2, 0.3]]), np.array([0.0, 0.5, 0.5]),
				"^the displacement of each vertex from reference_center must be smaller than 2\\*\\*52 in magnitude$"),
			"overflowing": (np.array([[1e308, 0.2, 0.3]]), np.array([-1e308, 0.5, 0.5]),
				"^the displacement of each vertex from reference_center must be finite$"),
		}
		for name, (vertices, reference_center, message) in cases.items():
			with self.subTest(name), np.errstate(over="ignore"):
				with self.assertRaisesRegex(ValueError, message):
					unwrap_vertices_to_reference_center(
						vertices, reference_center, Lattice.cubic(10.0).matrix)

	def test_singular_lattice_raises(self):
		"""A singular lattice is rejected."""
		lattice_matrix = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
		with self.assertRaises(ValueError):
			unwrap_vertices_to_reference_center(
				np.array([[0.1, 0.2, 0.3]]), np.array([0.5, 0.5, 0.5]), lattice_matrix)


class TestCorrectPbc(unittest.TestCase):
	"""Tests for the correct_pbc dispatch function."""

	def setUp(self):
		self.lattice = Lattice.cubic(10.0)

	def test_delegates_to_legacy_when_no_reference_center(self):
		"""With no reference centre, the coordinates are those of apply_legacy_pbc_correction."""
		# The legacy correction takes 0.1 to 1.1, leaving the lowest z in [0, 1).
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		corrected, _ = correct_pbc(frac_coords, None, self.lattice.matrix)
		np.testing.assert_array_equal(corrected, apply_legacy_pbc_correction(frac_coords))

	def test_delegates_to_unwrap_when_reference_center_provided(self):
		"""With a reference centre, the coordinates and shifts are those of unwrap_vertices_to_reference_center."""
		frac_coords = np.array([[0.1, 0.1, 0.1], [0.9, 0.2, 0.2]])
		ref = np.array([0.15, 0.15, 0.15])
		corrected, shifts = correct_pbc(frac_coords, ref, self.lattice.matrix)
		expected_coords, expected_shifts = unwrap_vertices_to_reference_center(
			frac_coords, ref, self.lattice.matrix, return_image_shifts=True)
		np.testing.assert_array_equal(corrected, expected_coords)
		np.testing.assert_array_equal(shifts, expected_shifts)

	def test_returns_int64_shifts_legacy_path(self):
		"""Image shifts have int64 dtype for the legacy path."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		_, shifts = correct_pbc(frac_coords, None, self.lattice.matrix)
		self.assertEqual(shifts.dtype, np.int64)

	def test_returns_int64_shifts_reference_centre_path(self):
		"""Image shifts have int64 dtype for the reference-centre path."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		ref = np.array([0.15, 0.15, 0.5])
		_, shifts = correct_pbc(frac_coords, ref, self.lattice.matrix)
		self.assertTrue(np.issubdtype(shifts.dtype, np.integer))

	def test_empty_input_returns_empty_arrays(self):
		"""Empty input should return empty arrays without error."""
		frac_coords = np.array([]).reshape(0, 3)
		corrected, shifts = correct_pbc(frac_coords, None, self.lattice.matrix)
		self.assertEqual(corrected.shape, (0, 3))
		self.assertEqual(shifts.shape, (0, 3))
		self.assertEqual(shifts.dtype, np.int64)

	def test_non_finite_coordinates_raise_without_reference_centre(self):
		"""Non-finite coordinates are rejected by name on the spread-based path too."""
		for name, value in (("NaN", np.nan), ("infinite", np.inf)):
			with self.subTest(name):
				frame = np.array([[0.1, 0.1, 0.9], [value, 0.2, 0.1]])
				with self.assertRaisesRegex(ValueError, "^frac_coords must be finite$"):
					correct_pbc(frame, None, self.lattice.matrix)

	def test_legacy_path_returns_consistent_shifts(self):
		"""Shifts from the legacy path satisfy corrected = original + shifts."""
		frac_coords = np.array([[0.1, 0.1, 0.9], [0.2, 0.2, 0.1]])
		corrected, shifts = correct_pbc(frac_coords, None, self.lattice.matrix)
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
		np.testing.assert_array_almost_equal(coords, [[0.11, 0.21, 1.31], [0.41, 1.51, 0.61]])

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

	def test_minimum_coordinate_translated_into_the_unit_cell(self):
		"""Whole-cell translation puts the lowest coordinate along each axis in [0, 1)."""
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached + 0.01
		for image_shift in (2, -1):
			with self.subTest(image_shift=image_shift):
				shifts = np.full((2, 3), image_shift, dtype=np.int64)
				valid, coords, _ = _numpy_update_pbc_shifts(new, cached, shifts)
				self.assertTrue(valid)
				np.testing.assert_array_almost_equal(coords.min(axis=0), new.min(axis=0))


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

	def test_agrees_with_numpy_non_finite_coordinate(self):
		"""A NaN coordinate invalidates the cache, as it does without numba."""
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts, _numba_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached.copy()
		new[1, 2] = np.nan
		shifts = np.zeros((2, 3), dtype=np.int64)
		v_np, _, _ = _numpy_update_pbc_shifts(new, cached, shifts)
		v_nb, _, _ = _numba_update_pbc_shifts(new, cached, shifts)
		self.assertFalse(v_np)
		self.assertEqual(v_nb, v_np)

	def test_agrees_with_numpy_minimum_translated_into_the_unit_cell(self):
		from site_analysis.pbc_utils import _numpy_update_pbc_shifts, _numba_update_pbc_shifts
		cached = np.array([[0.1, 0.2, 0.3],
						   [0.4, 0.5, 0.6]])
		new = cached + 0.01
		for image_shift in (2, -1):
			with self.subTest(image_shift=image_shift):
				shifts = np.full((2, 3), image_shift, dtype=np.int64)
				_, c_np, _ = _numpy_update_pbc_shifts(new, cached, shifts)
				_, c_nb, _ = _numba_update_pbc_shifts(new, cached, shifts)
				np.testing.assert_array_almost_equal(c_nb, c_np)
				np.testing.assert_array_almost_equal(c_nb.min(axis=0), new.min(axis=0))


if __name__ == '__main__':
	unittest.main()
