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
    # The same at a thousandth of the size, so that distances are below 1.
    "thin hexagonal 1x10x1, scaled by 1e-3": (
        Lattice.hexagonal(3.0e-3, 4.0e-3).matrix * np.array([[1.0], [10.0], [1.0]])),
    "monoclinic 1x1x8, beta 125": (
        Lattice.monoclinic(4.0, 5.0, 6.0, 125).matrix * np.array([[1.0], [1.0], [8.0]])),
    "cubic lattice in a cell sheared 5x": (
        np.array([[1.0, 0.0, 0.0], [5.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        @ Lattice.cubic(3.0).matrix),
    # The same lattice as an ordinary triclinic cell, described by vectors
    # that are sums of its reduced ones.
    "triclinic lattice in an unreduced cell": (
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        @ Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix),
}

# The hard cells with their lattice vectors in each cyclic order, so that
# the thin or skewed direction falls on every axis in turn; a moderately
# skewed cell in which some pairs need shifts of two cells along an axis;
# and a rhombohedral cell with 10 degree angles, flatter than real cells
# but far from singular.
HARD_CELLS_ON_EVERY_AXIS = {
    **{f"{name}, vectors rolled {k}": np.roll(lattice_matrix, k, axis=0)
       for name, lattice_matrix in HARD_CELLS.items() for k in range(3)},
    "skewed 2.7x3.9x8.4": Lattice.from_parameters(2.7, 3.9, 8.4, 100.7, 37.2, 81.3).matrix,
    "rhombohedral, 10 degree angles": Lattice.from_parameters(5.0, 5.0, 5.0, 10, 10, 10).matrix,
}

# Elongated cells, in which long pairs have boxes of many shifts that the
# ball search mostly skips.
ELONGATED_CELLS = {
    "orthorhombic 10x10x200": Lattice.orthorhombic(10.0, 10.0, 200.0).matrix,
    "hexagonal 12x12x200, long axis first": np.roll(Lattice.hexagonal(12.0, 200.0).matrix, 1, axis=0),
}


def brute_force_distances(frac1, frac2, lattice_matrix, reach=12):
    """Minimum-image distances between pairs, from every shift up to ``reach`` cells each way."""
    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)),
                      dtype=float)
    d = np.atleast_2d(np.asarray(frac1, dtype=float) - np.asarray(frac2, dtype=float))
    d -= np.round(d)
    return np.linalg.norm((d[:, np.newaxis, :] + shifts) @ lattice_matrix, axis=2).min(axis=1)


def brute_force_distance(frac1, frac2, lattice_matrix, reach=12):
    """Minimum-image distance between two points, as ``brute_force_distances``."""
    return float(brute_force_distances(frac1, frac2, lattice_matrix, reach)[0])


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
            expected = brute_force_distances(frac1, frac2, lattice_matrix)
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
                 **HARD_CELLS_ON_EVERY_AXIS, **ELONGATED_CELLS}
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

    def test_mic_distance_without_numba_follows_a_change_of_lattice(self):
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

    def test_distances_scale_with_the_cell(self):
        """Scaling a cell by a very large or small factor scales its distances."""
        lattice_matrix = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix
        frac1, frac2 = np.array([0.1, 0.9, 0.5]), np.array([0.9, 0.1, 0.4])
        for has_numba in BACKENDS:
            with patch.object(dist_mod, "HAS_NUMBA", has_numba):
                unscaled = dist_mod.mic_distance(frac1, frac2, lattice_matrix)
                for scale in (1e-105, 1e105):
                    with self.subTest(numba=has_numba, scale=scale):
                        self.assertAlmostEqual(
                            dist_mod.mic_distance(frac1, frac2, scale * lattice_matrix) / scale,
                            unscaled, places=12)
                        np.testing.assert_allclose(
                            dist_mod.paired_mic_distances(
                                frac1[np.newaxis], frac2[np.newaxis], scale * lattice_matrix) / scale,
                            [unscaled], rtol=1e-12)

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


class TestSearches(unittest.TestCase):
    """The box and ball searches find the same images, and each is used where it is cheaper."""

    @staticmethod
    def rounded_images(frac1, frac2, lattice_matrix):
        """Rounded displacements and their squared lengths, as _pair_distance starts."""
        rows = lattice_matrix.tolist()
        for a, b in zip(frac1, frac2):
            r = [dist_mod._wrap(float(x) - float(y)) for x, y in zip(a, b)]
            yield r, dist_mod._squared_length(*r, rows), rows

    def test_box_and_ball_searches_agree_exactly(self):
        """Both searches return the same smallest squared length for every pair."""
        rng = np.random.default_rng(86)
        cells = {"triclinic": Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix,
                 **HARD_CELLS_ON_EVERY_AXIS, **ELONGATED_CELLS}
        for name, lattice_matrix in cells.items():
            inverse_widths = dist_mod._inverse_widths(lattice_matrix.tolist())
            frac1 = rng.uniform(-1.0, 2.0, (40, 3))
            frac2 = rng.uniform(-1.0, 2.0, (40, 3))
            with self.subTest(cell=name):
                for r, best, rows in self.rounded_images(frac1, frac2, lattice_matrix):
                    self.assertEqual(
                        dist_mod._search_ball(*r, best, rows, inverse_widths),
                        dist_mod._search_box(*r, best, rows, inverse_widths))

    def test_large_boxes_use_the_ball_search(self):
        """A long pair in an elongated cell is searched by the ball, a short one by the box."""
        lattice_matrix = ELONGATED_CELLS["orthorhombic 10x10x200"]
        long_pair = (np.array([0.1, 0.2, 0.0]), np.array([0.3, 0.4, 0.4]))
        # 6.4 A apart across the short axes: a box of 2 x 2 x 1 shifts.
        near_pair = (np.array([0.55, 0.65, 0.0]), np.array([0.1, 0.2, 0.0]))
        cases = {"long pair": (long_pair, "_search_ball", "_search_box"),
                 "pair needing a small box": (near_pair, "_search_box", "_search_ball")}
        for name, ((frac1, frac2), used, unused) in cases.items():
            with self.subTest(name), patch.object(dist_mod, "HAS_NUMBA", False), \
                    patch.object(dist_mod, used, wraps=getattr(dist_mod, used)) as used_search, \
                    patch.object(dist_mod, unused, wraps=getattr(dist_mod, unused)) as unused_search:
                dist_mod.mic_distance(frac1, frac2, lattice_matrix)
                used_search.assert_called_once()
                unused_search.assert_not_called()

    def test_numpy_batches_use_the_ball_search_for_large_boxes(self):
        """Without numba, batches send pairs with large boxes to the ball search too."""
        lattice_matrix = ELONGATED_CELLS["orthorhombic 10x10x200"]
        frac1 = np.array([[0.1, 0.2, 0.0], [0.1, 0.2, 0.0]])
        frac2 = np.array([[0.3, 0.4, 0.4], [0.3, 0.4, 0.45]])
        with patch.object(dist_mod, "HAS_NUMBA", False), \
                patch.object(dist_mod, "_search_ball", wraps=dist_mod._search_ball) as ball:
            dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix)
        self.assertEqual(ball.call_count, 2)

    def test_huge_boxes_finish(self):
        """A long pair in a very elongated cell, with a box of 200 million shifts, is found at once."""
        lattice_matrix = np.diag([1.0, 1.0, 10000.0])
        frac1, frac2 = np.array([0.5, 0.5, 0.5]), np.zeros(3)
        # In an orthogonal cell the rounded image is the nearest.
        expected = math.sqrt(0.25 + 0.25 + 5000.0 ** 2)
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                self.assertEqual(dist_mod.mic_distance(frac1, frac2, lattice_matrix), expected)
                np.testing.assert_array_equal(
                    dist_mod.paired_mic_distances(
                        frac1[np.newaxis], frac2[np.newaxis], lattice_matrix), [expected])

    @unittest.skipUnless(HAS_NUMBA, "numba not installed")
    def test_box_too_large_to_count_in_64_bits_uses_the_ball(self):
        """With numba, a box of more shifts than a 64-bit integer holds is searched by the ball."""
        # A relative volume of 1.04e-8, just above the singular threshold:
        # this pair's box holds about 3e19 shifts.
        lattice_matrix = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 1.2e-8]])
        frac1, frac2 = np.array([0.02, 0.0, 0.001]), np.zeros(3)
        # The rounded image is the nearest.
        self.assertAlmostEqual(dist_mod.mic_distance(frac1, frac2, lattice_matrix),
                               float(np.linalg.norm(frac1 @ lattice_matrix)), places=12)


class TestUndefinedDistances(unittest.TestCase):
    """Non-finite input and singular lattices give NaN or raise."""

    SINGULAR_LATTICES = {
        "two parallel vectors": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        # Singular in exact arithmetic, but rounding leaves a volume of
        # about 1e-14 rather than 0.
        "third vector the sum of the others": (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            @ Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix),
        # The third vector is only 2e-8 out of the plane of the other two.
        "flat cell": np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [3.0, 4.0, 2e-8]]),
    }

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
        for name, lattice_matrix in self.SINGULAR_LATTICES.items():
            for has_numba in BACKENDS:
                with self.subTest(name, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    self.assertTrue(math.isnan(dist_mod.mic_distance(
                        np.array([0.1, 0.2, 0.3]), np.zeros(3), lattice_matrix)))

    def test_paired_mic_distances_rejects_singular_lattice(self):
        """paired_mic_distances raises ValueError for a singular or nearly singular lattice."""
        for name, lattice_matrix in self.SINGULAR_LATTICES.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    dist_mod.paired_mic_distances(
                        np.zeros((2, 3)), np.zeros((2, 3)), lattice_matrix)

    def test_unchecked_paired_distances_give_nan_for_non_finite_pairs(self):
        """A non-finite pair gives NaN and leaves the other pairs unchanged."""
        # Converting NaN to an integer gives a different number on each
        # platform, so numpy is made to raise on it: without the guard on
        # non-finite pairs, the test then fails everywhere. In this cell the
        # finite pairs need only their rounded image, so it cannot hang.
        lattice_matrix = 10.0 * np.eye(3)
        frac1 = np.array([[0.1, 0.2, 0.3], [np.nan, 0.2, 0.3], [0.4, 0.3, 0.2]])
        frac2 = np.zeros((3, 3))
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                with np.errstate(invalid="raise"):
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

    def test_numpy_batches_match_single_pairs_in_hard_cells(self):
        """Without numba, a pair's distance in a batch is its distance on its own, in every hard cell."""
        rng = np.random.default_rng(87)
        for name, lattice_matrix in HARD_CELLS_ON_EVERY_AXIS.items():
            frac1 = rng.uniform(-1.0, 2.0, (100, 3))
            frac2 = rng.uniform(-1.0, 2.0, (100, 3))
            with self.subTest(cell=name), patch.object(dist_mod, 'HAS_NUMBA', False):
                whole = dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix)
                single = np.concatenate([
                    dist_mod.paired_mic_distances(frac1[i:i + 1], frac2[i:i + 1], lattice_matrix)
                    for i in range(len(frac1))])
                np.testing.assert_array_equal(single, whole)


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
        # About half the pairs in this cell are searched by the ball.
        lattice_matrix = HARD_CELLS["cubic lattice in a cell sheared 5x"]
        rng = np.random.default_rng(7)
        n = 2 * dist_mod._PARALLEL_MIN_PAIRS
        frac1 = rng.uniform(-2.0, 3.0, (n, 3))
        frac2 = rng.uniform(-2.0, 3.0, (n, 3))
        small_n = dist_mod._PARALLEL_MIN_PAIRS - 1
        large = dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix)
        small = dist_mod.paired_mic_distances(frac1[:small_n], frac2[:small_n], lattice_matrix)
        np.testing.assert_array_equal(small, large[:small_n])

    def test_small_batches_use_serial_kernel(self):
        """Batches below the parallel threshold use the single-threaded kernel."""
        lattice = Lattice.cubic(10.0)
        rng = np.random.default_rng(8)
        n = dist_mod._PARALLEL_MIN_PAIRS - 1
        with patch.object(dist_mod, '_paired_mic_distances_parallel',
                          side_effect=AssertionError("parallel kernel used")):
            dist_mod.paired_mic_distances(
                rng.random((n, 3)), rng.random((n, 3)), lattice.matrix)

    def test_large_batches_use_parallel_kernel(self):
        """Batches at or above the parallel threshold use the parallel kernel."""
        lattice = Lattice.cubic(10.0)
        rng = np.random.default_rng(8)
        n = dist_mod._PARALLEL_MIN_PAIRS
        with patch.object(dist_mod, '_paired_mic_distances_serial',
                          side_effect=AssertionError("serial kernel used")):
            dist_mod.paired_mic_distances(
                rng.random((n, 3)), rng.random((n, 3)), lattice.matrix)
