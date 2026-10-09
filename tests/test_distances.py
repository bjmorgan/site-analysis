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

# The hard cells with their lattice vectors in each cyclic order, so that
# the thin or skewed direction falls on every axis in turn, and a moderately
# skewed cell in which some pairs need shifts of two cells along an axis.
HARD_CELLS_ON_EVERY_AXIS = {
    **{f"{name}, vectors rolled {k}": np.roll(lattice_matrix, k, axis=0)
       for name, lattice_matrix in HARD_CELLS.items() for k in range(3)},
    "skewed 2.7x3.9x8.4": Lattice.from_parameters(2.7, 3.9, 8.4, 100.7, 37.2, 81.3).matrix,
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
        for name, lattice_matrix in HARD_CELLS_ON_EVERY_AXIS.items():
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
                 **HARD_CELLS_ON_EVERY_AXIS}
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

    def test_numpy_mic_distance_follows_a_change_of_lattice(self):
        """Without numba, each distance uses its own lattice, however lattices alternate."""
        frac1, frac2 = np.array([0.1, 0.2, 0.3]), np.array([0.6, 0.9, 0.8])
        lattices = [Lattice.cubic(5.0).matrix,
                    Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix]
        expected = [brute_force_distance(frac1, frac2, m) for m in lattices]
        with patch.object(dist_mod, "HAS_NUMBA", False):
            for i in (0, 1, 0, 1):
                with self.subTest(call_with_lattice=i):
                    self.assertAlmostEqual(
                        dist_mod.mic_distance(frac1, frac2, lattices[i]),
                        expected[i], places=12)
            # The same array, changed in place, is a different lattice.
            lattice_matrix = lattices[1].copy()
            dist_mod.mic_distance(frac1, frac2, lattice_matrix)
            lattice_matrix *= 1.5
            with self.subTest("lattice changed in place"):
                self.assertAlmostEqual(
                    dist_mod.mic_distance(frac1, frac2, lattice_matrix),
                    brute_force_distance(frac1, frac2, lattice_matrix), places=12)

    def test_coordinates_beyond_64_bit_integers(self):
        """Coordinates too large for a 64-bit integer are whole numbers of cells."""
        lattice_matrix = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix
        far, near = np.array([1e19, 0.2, 0.3]), np.array([0.0, 0.2, 0.3])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                self.assertEqual(dist_mod.mic_distance(far, near, lattice_matrix), 0.0)
                np.testing.assert_array_equal(
                    dist_mod.paired_mic_distances(
                        far[np.newaxis], near[np.newaxis], lattice_matrix), [0.0])


class TestUndefinedDistances(unittest.TestCase):
    """Non-finite input and singular lattices give NaN or raise, never hang."""

    SINGULAR = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # Singular in exact arithmetic, as the third vector is the sum of the
    # first two, but rounding leaves a volume of about 1e-14 rather than 0.
    NEARLY_SINGULAR = (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
                       @ Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix)

    def test_nearly_singular_lattices_count_as_singular(self):
        """A volume that is only rounding error, or tiny next to the edges, gives no widths."""
        cases = {
            "volume from rounding": self.NEARLY_SINGULAR,
            # The third vector is only 2e-8 out of the plane of the other two.
            "flat cell": np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [3.0, 4.0, 2e-8]]),
        }
        for name, lattice_matrix in cases.items():
            with self.subTest(name):
                self.assertFalse(dist_mod._inverse_widths_are_finite(
                    dist_mod._inverse_widths(lattice_matrix.tolist())))

    def test_mic_distance_of_non_finite_input_is_nan(self):
        """NaN or inf coordinates give NaN, with and without numba."""
        for has_numba in BACKENDS:
            for value in (np.nan, np.inf):
                with self.subTest(numba=has_numba, value=value), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    self.assertTrue(math.isnan(dist_mod.mic_distance(
                        np.array([value, 0.2, 0.3]), np.zeros(3), 10.0 * np.eye(3))))

    def test_mic_distance_in_singular_lattice_is_nan(self):
        """A singular or nearly singular lattice gives NaN, with and without numba."""
        for lattice_matrix in (self.SINGULAR, self.NEARLY_SINGULAR):
            for has_numba in BACKENDS:
                with self.subTest(numba=has_numba, lattice=lattice_matrix.tolist()), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    self.assertTrue(math.isnan(dist_mod.mic_distance(
                        np.array([0.1, 0.2, 0.3]), np.zeros(3), lattice_matrix)))

    def test_paired_mic_distances_rejects_singular_lattice(self):
        """paired_mic_distances raises ValueError for a singular or nearly singular lattice."""
        for lattice_matrix in (self.SINGULAR, self.NEARLY_SINGULAR):
            with self.subTest(lattice=lattice_matrix.tolist()):
                with self.assertRaises(ValueError):
                    dist_mod.paired_mic_distances(
                        np.zeros((2, 3)), np.zeros((2, 3)), lattice_matrix)

    def test_unchecked_paired_distances_give_nan_for_non_finite_pairs(self):
        """A non-finite pair gives NaN and leaves the other pairs unchanged."""
        # In this cell the finite pairs need only their rounded image, so
        # without the guard on non-finite pairs the NaN pair still could not
        # widen the numpy shift search into an endless loop: the test would
        # fail rather than hang.
        lattice_matrix = 10.0 * np.eye(3)
        frac1 = np.array([[0.1, 0.2, 0.3], [np.nan, 0.2, 0.3], [0.4, 0.3, 0.2]])
        frac2 = np.zeros((3, 3))
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                result = dist_mod._paired_mic_distances(frac1, frac2, lattice_matrix)
                self.assertTrue(math.isnan(result[1]))
                np.testing.assert_array_equal(
                    result[[0, 2]],
                    dist_mod._paired_mic_distances(
                        frac1[[0, 2]], frac2[[0, 2]], lattice_matrix))


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

    def test_paired_mic_distances_numpy_fallback_does_not_depend_on_block_size(self):
        """Without numba, pairs searched a block at a time give the same distances in any block size."""
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(44)
        frac1 = rng.uniform(-2.0, 3.0, (1000, 3))
        frac2 = rng.uniform(-2.0, 3.0, (1000, 3))
        with patch.object(dist_mod, 'HAS_NUMBA', False):
            whole = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
            with patch.object(dist_mod, '_NEIGHBOUR_BLOCK', 7):
                blocked = dist_mod.paired_mic_distances(frac1, frac2, lattice.matrix)
        np.testing.assert_array_equal(blocked, whole)


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
