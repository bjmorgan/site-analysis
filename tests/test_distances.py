import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice
from site_analysis._compat import HAS_NUMBA
import site_analysis.distances as dist_mod


# Run with and without numba where numba is installed.
BACKENDS = (False, True) if HAS_NUMBA else (False,)

# Cells in which the nearest image of a pair can lie several cells away
# from the image nearest in fractional coordinates (#84).
HARD_CELLS = {
    "thin hexagonal 1x10x1": (
        Lattice.hexagonal(3.0, 4.0).matrix * np.array([[1.0], [10.0], [1.0]])),
    "monoclinic 1x1x8, beta 125": (
        Lattice.monoclinic(4.0, 5.0, 6.0, 125).matrix * np.array([[1.0], [1.0], [8.0]])),
    "cubic lattice in a cell sheared 5x": (
        np.array([[1.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        @ Lattice.cubic(3.0).matrix),
}


def brute_force_distance(frac1, frac2, lattice_matrix, reach=12):
    """Minimum-image distance from every shift up to ``reach`` cells each way."""
    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)),
                      dtype=float)
    d = np.asarray(frac1, dtype=float) - np.asarray(frac2, dtype=float)
    d -= np.round(d)
    return float(np.linalg.norm((d + shifts) @ lattice_matrix, axis=1).min())


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

    def test_coords_many_cells_away(self):
        """Coordinates differing by many unit cells produce correct distances."""
        from site_analysis.distances import mic_distance
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac1 = np.array([50.3, -100.7, 25.1])
        frac2 = np.array([0.1, 0.2, 0.3])
        expected = lattice.get_distance_and_image(frac1, frac2)[0]
        result = mic_distance(frac1, frac2, lattice.matrix)
        self.assertAlmostEqual(result, float(expected), places=10)


class TestPairedMicDistances(unittest.TestCase):
    """Tests for minimum-image distances between paired points."""

    def test_matches_pymatgen_cubic(self):
        """Distances match pymatgen for a cubic lattice, across boundaries."""
        from site_analysis.distances import paired_mic_distances
        lattice = Lattice.cubic(10.0)
        frac1 = np.array([[0.1, 0.2, 0.3], [0.05, 0.5, 0.5]])
        frac2 = np.array([[0.9, 0.1, 0.5], [0.95, 0.5, 0.5]])
        expected = [lattice.get_distance_and_image(a, b)[0] for a, b in zip(frac1, frac2)]
        result = paired_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matches_pymatgen_triclinic(self):
        """Distances match pymatgen for a triclinic lattice."""
        from site_analysis.distances import paired_mic_distances
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        frac1 = rng.random((20, 3))
        frac2 = rng.random((20, 3))
        expected = [lattice.get_distance_and_image(a, b)[0] for a, b in zip(frac1, frac2)]
        result = paired_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_coords_many_cells_away(self):
        """Coordinates differing by many unit cells produce correct distances."""
        from site_analysis.distances import paired_mic_distances
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac1 = np.array([[50.3, -100.7, 25.1], [0.1, 0.2, 0.3]])
        frac2 = np.array([[0.1, 0.2, 0.3], [-50.4, 75.9, -10.8]])
        expected = [lattice.get_distance_and_image(a, b)[0] for a, b in zip(frac1, frac2)]
        result = paired_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_empty_input_returns_empty(self):
        """Empty inputs give an empty array of distances."""
        from site_analysis.distances import paired_mic_distances
        lattice = Lattice.cubic(10.0)
        result = paired_mic_distances(np.empty((0, 3)), np.empty((0, 3)), lattice.matrix)
        self.assertEqual(result.shape, (0,))

    def test_wrong_shapes_raise_value_error(self):
        """Coordinates not both shaped (K, 3), or a lattice not (3, 3), raise ValueError."""
        from site_analysis.distances import paired_mic_distances
        lattice_matrix = Lattice.cubic(10.0).matrix
        cases = [
            (np.zeros((2, 3)), np.zeros((3, 3)), lattice_matrix),
            (np.zeros((2, 2)), np.zeros((2, 2)), lattice_matrix),
            (np.zeros(3), np.zeros(3), lattice_matrix),
            (np.zeros((2, 3)), np.zeros((2, 3)), np.eye(2)),
        ]
        for frac1, frac2, lattice in cases:
            with self.subTest(frac1_shape=frac1.shape, lattice_shape=lattice.shape):
                with self.assertRaises(ValueError):
                    paired_mic_distances(frac1, frac2, lattice)

    def test_non_finite_input_raises_value_error(self):
        """NaN or inf in either coordinate array or the lattice raises ValueError."""
        for value in (np.nan, np.inf):
            for argument in range(3):
                args = [np.zeros((2, 3)), np.zeros((2, 3)), 10.0 * np.eye(3)]
                args[argument][0, 0] = value
                with self.subTest(value=value, argument=argument):
                    with self.assertRaises(ValueError):
                        dist_mod.paired_mic_distances(*args)


class TestExactMinimumImage(unittest.TestCase):
    """Distances are exact even where the nearest image is several cells away."""

    def test_issue_84_pair(self):
        """The #84 pair is 6 * sqrt(3) apart, not the 10.82 the nearest 27 images give."""
        lattice_matrix = HARD_CELLS["thin hexagonal 1x10x1"]
        frac1, frac2 = np.zeros(3), np.array([0.0, 0.4, 0.0])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                self.assertAlmostEqual(
                    dist_mod.mic_distance(frac1, frac2, lattice_matrix),
                    6 * math.sqrt(3), places=12)
                np.testing.assert_allclose(
                    dist_mod.paired_mic_distances(
                        frac1[np.newaxis], frac2[np.newaxis], lattice_matrix),
                    [6 * math.sqrt(3)], rtol=1e-14)

    def test_matches_brute_force_in_hard_cells(self):
        """Single and paired distances match a search over many images."""
        rng = np.random.default_rng(84)
        for name, lattice_matrix in HARD_CELLS.items():
            frac1 = rng.uniform(-1.0, 2.0, (100, 3))
            frac2 = rng.uniform(-1.0, 2.0, (100, 3))
            expected = [brute_force_distance(a, b, lattice_matrix)
                        for a, b in zip(frac1, frac2)]
            for has_numba in BACKENDS:
                with self.subTest(cell=name, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    np.testing.assert_allclose(
                        dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix),
                        expected, rtol=1e-12)
                    np.testing.assert_allclose(
                        [dist_mod.mic_distance(a, b, lattice_matrix)
                         for a, b in zip(frac1, frac2)],
                        expected, rtol=1e-12)

    def test_all_implementations_agree_exactly(self):
        """Single and paired distances, with and without numba, are identical."""
        rng = np.random.default_rng(85)
        cells = {"triclinic": Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix,
                 **HARD_CELLS}
        for name, lattice_matrix in cells.items():
            frac1 = rng.uniform(-2.0, 3.0, (200, 3))
            frac2 = rng.uniform(-2.0, 3.0, (200, 3))
            results = []
            for has_numba in BACKENDS:
                with patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    results.append(
                        dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix))
                    results.append(np.array(
                        [dist_mod.mic_distance(a, b, lattice_matrix)
                         for a, b in zip(frac1, frac2)]))
            with self.subTest(cell=name):
                for other in results[1:]:
                    np.testing.assert_array_equal(other, results[0])


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

    def test_paired_mic_distances_numpy_fallback_matches_pymatgen(self):
        """Numpy paired_mic_distances fallback produces correct results."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        frac1 = rng.random((20, 3))
        frac2 = rng.random((20, 3))
        expected = [lattice.get_distance_and_image(a, b)[0] for a, b in zip(frac1, frac2)]
        with patch.object(dist_mod, 'HAS_NUMBA', False):
            result = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_paired_mic_distances_numpy_fallback_does_not_depend_on_batch(self):
        """Without numba, a pair's distance is the same in batches of any size."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.from_parameters(6.0, 6.0, 6.0, 60, 60, 60)
        rng = np.random.default_rng(43)
        frac1 = rng.uniform(-2.0, 3.0, (1000, 3))
        frac2 = rng.uniform(-2.0, 3.0, (1000, 3))
        with patch.object(dist_mod, 'HAS_NUMBA', False):
            whole = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
            for size in (1, 7):
                with self.subTest(batch_size=size):
                    batches = [dist_mod.paired_mic_distances(
                        frac1[i:i + size], frac2[i:i + size], lattice.matrix)
                        for i in range(0, len(frac1), size)]
                    np.testing.assert_array_equal(np.concatenate(batches), whole)


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

    def test_paired_mic_distances_numba_matches_numpy(self):
        """Numba and numpy paired distances are identical, so results do not depend on numba."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        frac1 = rng.uniform(-2.0, 3.0, (200, 3))
        frac2 = rng.uniform(-2.0, 3.0, (200, 3))
        numba_result = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
        with patch.object(dist_mod, 'HAS_NUMBA', False):
            numpy_result = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_array_equal(numba_result, numpy_result)

    def test_small_and_large_batches_agree(self):
        """Small batches, on one thread, and large batches, in parallel, give identical distances."""
        import site_analysis.distances as dist_mod
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(7)
        n = 2 * dist_mod._PARALLEL_MIN_PAIRS
        frac1 = rng.uniform(-2.0, 3.0, (n, 3))
        frac2 = rng.uniform(-2.0, 3.0, (n, 3))
        small_n = dist_mod._PARALLEL_MIN_PAIRS - 1
        large = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
        small = dist_mod.paired_mic_distances(frac1[:small_n], frac2[:small_n], lattice.matrix)
        np.testing.assert_array_equal(small, large[:small_n])

    def test_small_batches_use_serial_kernel(self):
        """Batches below the parallel threshold use the single-threaded kernel."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.cubic(10.0)
        rng = np.random.default_rng(8)
        n = dist_mod._PARALLEL_MIN_PAIRS - 1
        with patch.object(dist_mod, '_paired_mic_distances_parallel',
                          side_effect=AssertionError("parallel kernel used")):
            dist_mod.paired_mic_distances(
                rng.random((n, 3)), rng.random((n, 3)), lattice.matrix)

    def test_large_batches_use_parallel_kernel(self):
        """Batches at or above the parallel threshold use the parallel kernel."""
        from unittest.mock import patch
        import site_analysis.distances as dist_mod
        lattice = Lattice.cubic(10.0)
        rng = np.random.default_rng(8)
        n = dist_mod._PARALLEL_MIN_PAIRS
        with patch.object(dist_mod, '_paired_mic_distances_serial',
                          side_effect=AssertionError("serial kernel used")):
            dist_mod.paired_mic_distances(
                rng.random((n, 3)), rng.random((n, 3)), lattice.matrix)
