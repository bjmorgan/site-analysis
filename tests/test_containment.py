"""Tests for site_analysis.containment module."""

import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from site_analysis.containment import (
    FaceTopologyCache,
    HAS_NUMBA,
    _numpy_update_pbc_shifts,
    update_pbc_shifts,
)

# Tetrahedron centred at [0.5, 0.5, 0.5]
TETRAHEDRON = np.array([
    [0.4, 0.4, 0.4],
    [0.4, 0.6, 0.6],
    [0.6, 0.6, 0.4],
    [0.6, 0.4, 0.6],
])

# Octahedron centred at [0.5, 0.5, 0.5]
OCTAHEDRON = np.array([
    [0.6, 0.5, 0.5],
    [0.4, 0.5, 0.5],
    [0.5, 0.6, 0.5],
    [0.5, 0.4, 0.5],
    [0.5, 0.5, 0.6],
    [0.5, 0.5, 0.4],
])

# PBC shifts (same as tools.x_pbc)
_SHIFTS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
])


def x_pbc(x: np.ndarray) -> np.ndarray:
    return _SHIFTS + x


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestFaceTopologyCache(unittest.TestCase):
    """Tests for FaceTopologyCache with numba acceleration."""

    def test_initialisation_computes_face_simplices(self):
        cache = FaceTopologyCache(TETRAHEDRON)
        # A tetrahedron has 4 triangular faces
        self.assertEqual(cache.face_simplices.shape[0], 4)
        self.assertEqual(cache.face_simplices.shape[1], 3)

    def test_initialisation_computes_face_simplices_octahedron(self):
        cache = FaceTopologyCache(OCTAHEDRON)
        # An octahedron has 8 triangular faces
        self.assertEqual(cache.face_simplices.shape[0], 8)
        self.assertEqual(cache.face_simplices.shape[1], 3)

    def test_update_preserves_topology(self):
        cache = FaceTopologyCache(TETRAHEDRON)
        original_simplices = cache.face_simplices.copy()
        # Slightly perturb vertex coordinates
        perturbed = TETRAHEDRON + 0.01
        cache.update(perturbed)
        assert_array_equal(cache.face_simplices, original_simplices)

    def test_contains_point_inside_tetrahedron(self):
        cache = FaceTopologyCache(TETRAHEDRON)
        point_inside = np.array([0.5, 0.5, 0.5])
        self.assertTrue(cache.contains_point(x_pbc(point_inside)))

    def test_contains_point_outside_tetrahedron(self):
        cache = FaceTopologyCache(TETRAHEDRON)
        point_outside = np.array([0.1, 0.1, 0.1])
        self.assertFalse(cache.contains_point(x_pbc(point_outside)))

    def test_contains_point_inside_octahedron(self):
        cache = FaceTopologyCache(OCTAHEDRON)
        point_inside = np.array([0.5, 0.5, 0.5])
        self.assertTrue(cache.contains_point(x_pbc(point_inside)))

    def test_contains_point_outside_octahedron(self):
        cache = FaceTopologyCache(OCTAHEDRON)
        point_outside = np.array([0.1, 0.1, 0.1])
        self.assertFalse(cache.contains_point(x_pbc(point_outside)))

    def test_contains_point_after_coordinate_update(self):
        """Containment still works after updating coordinates."""
        cache = FaceTopologyCache(TETRAHEDRON)
        # Shift all vertices by +0.1 in x
        shifted = TETRAHEDRON.copy()
        shifted[:, 0] += 0.1
        cache.update(shifted)
        # Centre of shifted tetrahedron is now at [0.6, 0.5, 0.5]
        point_inside = np.array([0.6, 0.5, 0.5])
        self.assertTrue(cache.contains_point(x_pbc(point_inside)))
        # Original centre is now outside
        point_outside = np.array([0.4, 0.5, 0.5])
        self.assertFalse(cache.contains_point(x_pbc(point_outside)))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestNumbaQuery(unittest.TestCase):
    """Tests for the low-level _numba_sn_query function."""

    def test_query_inside_point(self):
        from site_analysis.containment import _numba_sn_query

        cache = FaceTopologyCache(TETRAHEDRON)
        point_inside = np.array([[0.5, 0.5, 0.5]])
        result = _numba_sn_query(
            point_inside,
            cache._face_normals,
            cache._face_ref_points,
            cache._centre_signs,
        )
        self.assertTrue(result)

    def test_query_outside_point(self):
        from site_analysis.containment import _numba_sn_query

        cache = FaceTopologyCache(TETRAHEDRON)
        point_outside = np.array([[0.1, 0.1, 0.1]])
        result = _numba_sn_query(
            point_outside,
            cache._face_normals,
            cache._face_ref_points,
            cache._centre_signs,
        )
        self.assertFalse(result)

    def test_query_returns_true_if_any_image_inside(self):
        from site_analysis.containment import _numba_sn_query

        cache = FaceTopologyCache(TETRAHEDRON)
        # First point outside, second inside
        points = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
        result = _numba_sn_query(
            points,
            cache._face_normals,
            cache._face_ref_points,
            cache._centre_signs,
        )
        self.assertTrue(result)

    def test_query_returns_false_if_all_images_outside(self):
        from site_analysis.containment import _numba_sn_query

        cache = FaceTopologyCache(TETRAHEDRON)
        points = np.array([[0.1, 0.1, 0.1], [0.8, 0.8, 0.9]])
        result = _numba_sn_query(
            points,
            cache._face_normals,
            cache._face_ref_points,
            cache._centre_signs,
        )
        self.assertFalse(result)


class TestNumpyUpdatePbcShifts(unittest.TestCase):
    """Tests for _numpy_update_pbc_shifts."""

    def test_cache_hit_small_displacement(self):
        """Small physical displacement returns valid cache."""
        cached = np.array([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
        new = cached + 0.01  # small vibration
        shifts = np.array([[0, 0, 1],
                           [0, 1, 0]], dtype=np.int64)
        valid, coords, new_shifts = _numpy_update_pbc_shifts(new, cached, shifts)
        self.assertTrue(valid)
        assert_array_equal(new_shifts, shifts)
        expected = new + shifts
        min_coords = np.min(expected, axis=0)
        uniform = np.maximum(0, np.ceil(-min_coords))
        np.testing.assert_array_almost_equal(coords, expected + uniform)

    def test_cache_hit_with_wrapping(self):
        """Vertex wrapping from 0.99 to 0.01 is detected and shifts adjusted."""
        cached = np.array([[0.99, 0.5, 0.5],
                           [0.5, 0.5, 0.5]])
        new = np.array([[0.01, 0.5, 0.5],   # wrapped across boundary
                        [0.5, 0.5, 0.5]])
        shifts = np.array([[0, 0, 0],
                           [0, 0, 0]], dtype=np.int64)
        valid, coords, new_shifts = _numpy_update_pbc_shifts(new, cached, shifts)
        self.assertTrue(valid)
        # Wrapping of -0.98 rounds to -1, so shift gains +1
        assert_array_equal(new_shifts[0], [1, 0, 0])
        assert_array_equal(new_shifts[1], [0, 0, 0])

    def test_cache_miss_large_displacement(self):
        """Large physical displacement invalidates cache."""
        cached = np.array([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
        new = cached + 0.4  # too large
        shifts = np.zeros((2, 3), dtype=np.int64)
        valid, _, _ = _numpy_update_pbc_shifts(new, cached, shifts)
        self.assertFalse(valid)

    def test_non_negative_shift_applied(self):
        """Uniform shift ensures all output coordinates are non-negative."""
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
        from site_analysis.containment import _numba_update_pbc_shifts
        cached = np.array([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
        new = cached + 0.01
        shifts = np.array([[0, 0, 1],
                           [0, 1, 0]], dtype=np.int64)
        v_np, c_np, s_np = _numpy_update_pbc_shifts(new, cached, shifts)
        v_nb, c_nb, s_nb = _numba_update_pbc_shifts(new, cached, shifts)
        self.assertEqual(v_np, v_nb)
        np.testing.assert_array_almost_equal(c_np, c_nb)
        assert_array_equal(s_np, s_nb)

    def test_agrees_with_numpy_wrapping(self):
        from site_analysis.containment import _numba_update_pbc_shifts
        cached = np.array([[0.99, 0.5, 0.5],
                           [0.5, 0.5, 0.5]])
        new = np.array([[0.01, 0.5, 0.5],
                        [0.5, 0.5, 0.5]])
        shifts = np.zeros((2, 3), dtype=np.int64)
        v_np, c_np, s_np = _numpy_update_pbc_shifts(new, cached, shifts)
        v_nb, c_nb, s_nb = _numba_update_pbc_shifts(new, cached, shifts)
        self.assertEqual(v_np, v_nb)
        np.testing.assert_array_almost_equal(c_np, c_nb)
        assert_array_equal(s_np, s_nb)

    def test_agrees_with_numpy_cache_miss(self):
        from site_analysis.containment import _numba_update_pbc_shifts
        cached = np.array([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6]])
        new = cached + 0.4
        shifts = np.zeros((2, 3), dtype=np.int64)
        v_np, _, _ = _numpy_update_pbc_shifts(new, cached, shifts)
        v_nb, _, _ = _numba_update_pbc_shifts(new, cached, shifts)
        self.assertEqual(v_np, v_nb)


if __name__ == "__main__":
    unittest.main()
