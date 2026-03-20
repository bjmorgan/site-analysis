import unittest
import numpy as np
from pymatgen.core import Lattice
from site_analysis._compat import HAS_NUMBA


class TestFracToCart(unittest.TestCase):
    """Tests for fractional to Cartesian coordinate conversion."""

    def test_single_point_cubic(self):
        """Conversion matches pymatgen for a cubic lattice."""
        from site_analysis.distances import frac_to_cart
        lattice = Lattice.cubic(10.0)
        frac = np.array([0.1, 0.2, 0.3])
        expected = lattice.get_cartesian_coords(frac)
        result = frac_to_cart(frac, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_single_point_triclinic(self):
        """Conversion matches pymatgen for a triclinic lattice."""
        from site_analysis.distances import frac_to_cart
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac = np.array([0.3, 0.5, 0.7])
        expected = lattice.get_cartesian_coords(frac)
        result = frac_to_cart(frac, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_points(self):
        """Conversion works for an (N, 3) array of points."""
        from site_analysis.distances import frac_to_cart
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac = np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6],
                         [0.7, 0.8, 0.9]])
        expected = lattice.get_cartesian_coords(frac)
        result = frac_to_cart(frac, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestMicDistance(unittest.TestCase):
    """Tests for single-pair minimum-image distance."""

    def test_same_point_returns_zero(self):
        """Distance between a point and itself is zero."""
        from site_analysis.distances import mic_distance
        lattice = Lattice.cubic(10.0)
        frac = np.array([0.5, 0.5, 0.5])
        result = mic_distance(frac, frac, lattice.matrix)
        self.assertAlmostEqual(result, 0.0, places=12)

    def test_cubic_no_pbc(self):
        """Distance within cell matches pymatgen for cubic lattice."""
        from site_analysis.distances import mic_distance
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([0.1, 0.2, 0.3])
        frac2 = np.array([0.4, 0.5, 0.6])
        expected = lattice.get_distance_and_image(frac1, frac2)[0]
        result = mic_distance(frac1, frac2, lattice.matrix)
        self.assertAlmostEqual(result, expected, places=10)

    def test_cubic_across_boundary(self):
        """Distance across periodic boundary is shorter than direct path."""
        from site_analysis.distances import mic_distance
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([0.05, 0.5, 0.5])
        frac2 = np.array([0.95, 0.5, 0.5])
        expected = lattice.get_distance_and_image(frac1, frac2)[0]
        result = mic_distance(frac1, frac2, lattice.matrix)
        self.assertAlmostEqual(result, expected, places=10)
        # Verify it chose the short path (1.0 A) not direct (9.0 A)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_triclinic(self):
        """Distance matches pymatgen for a triclinic lattice."""
        from site_analysis.distances import mic_distance
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac1 = np.array([0.1, 0.9, 0.5])
        frac2 = np.array([0.9, 0.1, 0.5])
        expected = lattice.get_distance_and_image(frac1, frac2)[0]
        result = mic_distance(frac1, frac2, lattice.matrix)
        self.assertAlmostEqual(result, expected, places=10)

    def test_matches_pymatgen_random_points(self):
        """Distance matches pymatgen for many random point pairs."""
        from site_analysis.distances import mic_distance
        rng = np.random.default_rng(42)
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        for _ in range(100):
            frac1 = rng.random(3)
            frac2 = rng.random(3)
            expected = lattice.get_distance_and_image(frac1, frac2)[0]
            result = mic_distance(frac1, frac2, lattice.matrix)
            self.assertAlmostEqual(result, float(expected), places=10,
                msg=f"Mismatch for {frac1} -> {frac2}")

    def test_symmetry(self):
        """Distance is symmetric: d(a, b) == d(b, a)."""
        from site_analysis.distances import mic_distance
        rng = np.random.default_rng(99)
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        for _ in range(20):
            frac1 = rng.random(3)
            frac2 = rng.random(3)
            d_ab = mic_distance(frac1, frac2, lattice.matrix)
            d_ba = mic_distance(frac2, frac1, lattice.matrix)
            self.assertAlmostEqual(d_ab, d_ba, places=12)

    def test_coords_outside_unit_cell(self):
        """Coordinates outside [0, 1) produce correct distances."""
        from site_analysis.distances import mic_distance
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([1.1, 0.2, 0.3])
        frac2 = np.array([0.1, 0.2, 0.3])
        expected = lattice.get_distance_and_image(frac1, frac2)[0]
        result = mic_distance(frac1, frac2, lattice.matrix)
        self.assertAlmostEqual(result, float(expected), places=10)


class TestAllMicDistances(unittest.TestCase):
    """Tests for batch all-pairs minimum-image distance matrix."""

    def test_single_pair(self):
        """1x1 distance matrix matches mic_distance."""
        from site_analysis.distances import all_mic_distances, mic_distance
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([[0.1, 0.2, 0.3]])
        frac2 = np.array([[0.4, 0.5, 0.6]])
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        expected = mic_distance(frac1[0], frac2[0], lattice.matrix)
        self.assertEqual(result.shape, (1, 1))
        self.assertAlmostEqual(result[0, 0], expected, places=10)

    def test_shape(self):
        """Output shape is (N, M) for N and M input points."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        frac2 = np.array([[0.7, 0.8, 0.9], [0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        self.assertEqual(result.shape, (2, 3))

    def test_matches_pymatgen_cubic(self):
        """Full distance matrix matches pymatgen for cubic lattice."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 0.1]])
        frac2 = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.4]])
        expected = lattice.get_all_distances(frac1, frac2)
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matches_pymatgen_triclinic(self):
        """Full distance matrix matches pymatgen for triclinic lattice."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        frac1 = rng.random((5, 3))
        frac2 = rng.random((8, 3))
        expected = lattice.get_all_distances(frac1, frac2)
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_pbc_distances_shorter_than_direct(self):
        """Points near opposite boundaries have short PBC distances."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([[0.05, 0.5, 0.5]])
        frac2 = np.array([[0.95, 0.5, 0.5]])
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        self.assertAlmostEqual(result[0, 0], 1.0, places=10)

    def test_self_distance_diagonal_is_zero(self):
        """Distance matrix of a set with itself has zeros on the diagonal."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(77)
        coords = rng.random((5, 3))
        result = all_mic_distances(coords, coords, lattice.matrix)
        np.testing.assert_allclose(np.diag(result), 0.0, atol=1e-12)

    def test_empty_first_array(self):
        """Empty first array returns correctly shaped empty output."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.cubic(10.0)
        frac1 = np.empty((0, 3))
        frac2 = np.array([[0.1, 0.2, 0.3]])
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        self.assertEqual(result.shape, (0, 1))

    def test_empty_second_array(self):
        """Empty second array returns correctly shaped empty output."""
        from site_analysis.distances import all_mic_distances
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([[0.1, 0.2, 0.3]])
        frac2 = np.empty((0, 3))
        result = all_mic_distances(frac1, frac2, lattice.matrix)
        self.assertEqual(result.shape, (1, 0))


class TestNumpyFallback(unittest.TestCase):
    """Tests that numpy fallback paths are correct regardless of numba."""

    def test_mic_distance_numpy_fallback_matches_pymatgen(self):
        """Numpy mic_distance fallback produces correct results."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        with patch.object(dist_mod, 'HAS_NUMBA', False):
            for _ in range(100):
                frac1 = rng.random(3)
                frac2 = rng.random(3)
                result = dist_mod.mic_distance(frac1, frac2, lattice.matrix)
                expected = float(lattice.get_distance_and_image(frac1, frac2)[0])
                self.assertAlmostEqual(result, expected, places=10)

    def test_all_mic_distances_numpy_fallback_matches_pymatgen(self):
        """Numpy all_mic_distances fallback produces correct results."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        frac1 = rng.random((5, 3))
        frac2 = rng.random((8, 3))
        expected = lattice.get_all_distances(frac1, frac2)
        with patch.object(dist_mod, 'HAS_NUMBA', False):
            result = dist_mod.all_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestNumbaAcceleration(unittest.TestCase):
    """Tests for numba-accelerated distance functions."""

    def test_mic_distance_numba_matches_pymatgen(self):
        """Numba single-pair version produces same results as pymatgen."""
        from site_analysis.distances import _mic_distance_numba
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        for _ in range(100):
            frac1 = rng.random(3)
            frac2 = rng.random(3)
            expected = float(lattice.get_distance_and_image(frac1, frac2)[0])
            result = _mic_distance_numba(frac1, frac2, lattice.matrix)
            self.assertAlmostEqual(result, expected, places=10)

    def test_all_mic_distances_numba_matches_pymatgen(self):
        """Numba batch version produces same results as pymatgen."""
        from site_analysis.distances import _all_mic_distances_numba
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        frac1 = rng.random((5, 3))
        frac2 = rng.random((8, 3))
        expected = lattice.get_all_distances(frac1, frac2)
        result = _all_mic_distances_numba(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)
