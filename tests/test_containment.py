"""Tests for site_analysis.containment module."""

import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from site_analysis.containment import (
    FaceTopologyCache,
    HAS_NUMBA,
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


if __name__ == "__main__":
    unittest.main()
