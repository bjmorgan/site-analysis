import itertools
import math
import re
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

# A typical triclinic cell, then every hard and elongated cell.
VARIED_CELLS = {
    "triclinic": Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix,
    **HARD_CELLS_ON_EVERY_AXIS,
    **ELONGATED_CELLS,
}

# Every combination, on the three axes, of one ULP below, exactly at and
# one ULP above -0.5 and 0.5: the 216 displacements on which the rounding
# of each axis, and the tie-breaks between images, are decided.
HALF_INTEGER_GRID = np.array(list(itertools.product(
    [value for half in (-0.5, 0.5)
     for value in (np.nextafter(half, -math.inf), half, np.nextafter(half, math.inf))],
    repeat=3)))


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


def random_unimodular(rng):
    """A random integer matrix with entries from -3 to 3 and determinant 1 or -1."""
    while True:
        matrix = rng.integers(-3, 4, (3, 3))
        if abs(round(np.linalg.det(matrix))) == 1:
            return matrix


def random_lattices(rng):
    """Random near-cubic, strongly sheared, thin and unreduced cells."""
    cubic = Lattice.cubic(4.0).matrix
    triclinic = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix
    lattices = {}
    for k in range(3):
        lattices[f"near-cubic {k}"] = cubic + rng.uniform(-0.3, 0.3, (3, 3))
        shear = np.eye(3)
        i, j = rng.choice(3, size=2, replace=False)
        shear[i, j] = rng.integers(3, 8)
        lattices[f"sheared {k}"] = shear @ triclinic
        scale = np.ones(3)
        scale[rng.integers(3)] = 0.1
        lattices[f"thin {k}"] = scale[:, np.newaxis] * triclinic
        lattices[f"unreduced cubic {k}"] = random_unimodular(rng) @ cubic
        lattices[f"unreduced triclinic {k}"] = random_unimodular(rng) @ triclinic
    return lattices


def random_displacements(rng, n):
    """Random fractional displacements, up to five cells away from the unit cell."""
    return rng.uniform(-0.5, 0.5, (n, 3)) + rng.integers(-5, 6, (n, 3))


def brute_force_minimum(displacement, lattice_matrix, reach):
    """Smallest length of any image of a displacement, from every image within ``reach``.

    Every image within ``reach`` of the origin lies in the box of shifts
    searched, widened by a cell each way against rounding.
    """
    r = displacement - np.round(displacement)
    inverse_widths = np.linalg.norm(np.linalg.inv(lattice_matrix), axis=0)
    lo = np.floor(-reach * inverse_widths - r).astype(int) - 1
    hi = np.ceil(reach * inverse_widths - r).astype(int) + 1
    shifts = np.stack(np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(lo, hi)),
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    return float(np.linalg.norm((r + shifts) @ lattice_matrix, axis=1).min())


def nearest_image_shifts_and_distances(displacements, lattice_matrix):
    """Shifts from ``_nearest_image_shifts``, and the lengths of the images they give."""
    shifts = dist_mod._nearest_image_shifts(displacements, lattice_matrix)
    return shifts, np.linalg.norm((displacements + shifts) @ lattice_matrix, axis=1)


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


class TestMicDistanceInputChecks(unittest.TestCase):
    """mic_distance rejects malformed inputs, with and without numba."""

    def test_point_of_wrong_shape_raises_value_error(self):
        """A point that does not have shape (3,) raises ValueError naming the argument."""
        good = np.array([0.1, 0.2, 0.3])
        lattice_matrix = 10.0 * np.eye(3)
        for bad_shape in ((2,), (4,), (1, 3), (3, 1)):
            bad = np.zeros(bad_shape)
            for has_numba in BACKENDS:
                with patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    with self.subTest(argument="frac1", shape=bad_shape, numba=has_numba):
                        with self.assertRaisesRegex(
                                ValueError, r"frac1 must have shape \(3,\), got " + re.escape(str(bad_shape))):
                            dist_mod.mic_distance(bad, good, lattice_matrix)
                    with self.subTest(argument="frac2", shape=bad_shape, numba=has_numba):
                        with self.assertRaisesRegex(
                                ValueError, r"frac2 must have shape \(3,\), got " + re.escape(str(bad_shape))):
                            dist_mod.mic_distance(good, bad, lattice_matrix)

    def test_lattice_of_wrong_shape_raises_value_error(self):
        """A lattice matrix that does not have shape (3, 3) raises ValueError naming it."""
        point = np.array([0.1, 0.2, 0.3])
        for bad_shape in ((2, 3), (3, 2), (9,), (3, 3, 1)):
            for has_numba in BACKENDS:
                with self.subTest(shape=bad_shape, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    with self.assertRaisesRegex(
                            ValueError, r"lattice_matrix must have shape \(3, 3\), got " + re.escape(str(bad_shape))):
                        dist_mod.mic_distance(point, np.zeros(3), np.ones(bad_shape))


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

    def test_matches_brute_force(self):
        """Single and paired distances match a search over many images."""
        rng = np.random.default_rng(84)
        for name, lattice_matrix in VARIED_CELLS.items():
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
        """Single and paired distances, with and without numba, are identical, on the half-integer grid too."""
        rng = np.random.default_rng(85)
        for name, lattice_matrix in VARIED_CELLS.items():
            frac1 = np.concatenate([rng.uniform(-2.0, 3.0, (200, 3)), HALF_INTEGER_GRID])
            frac2 = np.concatenate([rng.uniform(-2.0, 3.0, (200, 3)), np.zeros((216, 3))])
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
    """The box and ball searches find the same images, however large the box."""

    def test_box_and_ball_searches_agree_exactly(self):
        """Both searches return the same smallest squared length, and a shift that gives it, for every pair."""
        rng = np.random.default_rng(86)
        for name, lattice_matrix in VARIED_CELLS.items():
            rows = lattice_matrix.tolist()
            inverse_widths = dist_mod._inverse_widths(rows)
            frac1 = rng.uniform(-1.0, 2.0, (40, 3))
            frac2 = rng.uniform(-1.0, 2.0, (40, 3))
            with self.subTest(cell=name):
                for a, b in zip(frac1, frac2):
                    # The rounded image, as _pair_distance starts.
                    r = [dist_mod._wrap(float(x) - float(y)) for x, y in zip(a, b)]
                    best = dist_mod._squared_length(*r, rows)
                    ball = dist_mod._search_ball(*r, best, rows, inverse_widths)
                    box = dist_mod._search_box(*r, best, rows, inverse_widths)
                    self.assertEqual(ball[0], box[0])
                    for squared_length, *shift in (ball, box):
                        self.assertEqual(
                            dist_mod._squared_length(*(x + n for x, n in zip(r, shift)), rows),
                            squared_length)

    def test_huge_boxes_finish(self):
        """Long pairs with boxes of millions of shifts, nearest thousands of cells from the rounded image, are found at once, alone and in a batch with a short pair."""
        # A 1 x 1 x 10000 cell whose long vector leans 5000.5 cells along
        # x: the nearest image of a pair half a cell apart along it lies
        # 2500 cells from the rounded image, and its box holds about 75
        # million shifts.
        lattice_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5000.5, 0.0, 8660.0]])
        frac1 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.25], [0.1, 0.0, 0.0]])
        frac2 = np.zeros((3, 3))
        expected = [math.sqrt(0.25 ** 2 + 0.5 ** 2 + 4330.0 ** 2),
                    math.sqrt(0.375 ** 2 + 0.5 ** 2 + 2165.0 ** 2), 0.1]
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                self.assertEqual(
                    [dist_mod.mic_distance(a, b, lattice_matrix) for a, b in zip(frac1, frac2)],
                    expected)
                np.testing.assert_array_equal(
                    dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix), expected)

    def test_long_pairs_in_strongly_unreduced_cells(self):
        """Long pairs in cells that are sheared or tilted by thousands of cells match the reduced cell."""
        rng = np.random.default_rng(87)
        # Each lattice as a reduced cell, and the integer matrix that gives
        # the unreduced cell's vectors in terms of the reduced ones.
        cells = {
            "cubic lattice in a cell sheared 1000x": (
                Lattice.cubic(3.0).matrix, np.array([[1, 0, 0], [1000, 1, 0], [0, 0, 1]])),
            "orthorhombic 1x1x8660 as a 1x1x10000 cell tilted to 60 degrees": (
                np.diag([1.0, 1.0, 5000.0 * math.sqrt(3.0)]),
                np.array([[1, 0, 0], [0, 1, 0], [5000, 0, 1]])),
        }
        for name, (reduced, transform) in cells.items():
            unreduced = transform @ reduced
            frac1 = rng.uniform(-1.0, 2.0, (20, 3))
            frac2 = rng.uniform(-1.0, 2.0, (20, 3))
            # The same pairs in the reduced cell, where the nearest images
            # are a cell or so away.
            expected = brute_force_distances(frac1 @ transform, frac2 @ transform, reduced, reach=2)
            for has_numba in BACKENDS:
                with self.subTest(cell=name, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    np.testing.assert_allclose(
                        dist_mod.paired_mic_distances(frac1, frac2, unreduced),
                        expected, rtol=1e-11)
                    np.testing.assert_allclose(
                        [dist_mod.mic_distance(a, b, unreduced) for a, b in zip(frac1, frac2)],
                        expected, rtol=1e-11)

    def test_box_too_large_to_count_in_64_bits(self):
        """A box of more shifts than a 64-bit integer holds is searched, on every path."""
        # A relative volume of 1.04e-8, just above the singular threshold:
        # this pair's box holds about 5e24 shifts. Its nearest image, 6e-4
        # away, is shifted by one cell along two axes from its rounded
        # image, 1.0 away. So short a distance keeps the test fast: the
        # search's outermost loop steps through the distance in units of
        # the cell's 1.2e-8 width.
        lattice_matrix = np.array([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 1.2e-8]])
        frac1, frac2 = np.array([0.4995, 0.4995, -0.4999]), np.zeros(3)
        expected = brute_force_distance(frac1, frac2, lattice_matrix)
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                self.assertAlmostEqual(
                    dist_mod.mic_distance(frac1, frac2, lattice_matrix), expected, places=12)
                np.testing.assert_allclose(
                    dist_mod.paired_mic_distances(
                        frac1[np.newaxis], frac2[np.newaxis], lattice_matrix),
                    [expected], rtol=1e-12)


class TestNearestImageShifts(unittest.TestCase):
    """The shift to the nearest periodic image of each displacement is exact in any cell."""

    def test_nearest_shift_in_an_unreduced_cell(self):
        """A vertex at (0.6, 0.6, 0.5) is nearest a centre at (0, 0, 0.5) when shifted by (1, -1, 0)."""
        # Rows (1,0,0), (3,1,0), (0,0,1): an unreduced basis.
        lattice_matrix = np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        displacements = np.array([[0.6, 0.6, 0.5]]) - np.array([0.0, 0.0, 0.5])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                np.testing.assert_array_equal(
                    dist_mod._nearest_image_shifts(displacements, lattice_matrix), [[1, -1, 0]])

    def test_matches_brute_force(self):
        """Each shift gives an image as near as any, in near-cubic, sheared, thin and unreduced cells."""
        rng = np.random.default_rng(0)
        for name, lattice_matrix in random_lattices(rng).items():
            displacements = random_displacements(rng, 30)
            for has_numba in BACKENDS:
                with self.subTest(cell=name, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    shifts, distances = nearest_image_shifts_and_distances(
                        displacements, lattice_matrix)
                    self.assertEqual(shifts.dtype, np.int64)
                    # Any nearer image would lie within the distance found.
                    expected = [brute_force_minimum(f, lattice_matrix, d)
                                for f, d in zip(displacements, distances)]
                    np.testing.assert_allclose(distances, expected, rtol=1e-12)

    def test_distances_match_paired_mic_distances(self):
        """The image each shift gives is at exactly the minimum-image distance."""
        rng = np.random.default_rng(1)
        for name, lattice_matrix in random_lattices(rng).items():
            displacements = random_displacements(rng, 30)
            rows = lattice_matrix.tolist()
            for has_numba in BACKENDS:
                with self.subTest(cell=name, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    images = displacements + dist_mod._nearest_image_shifts(
                        displacements, lattice_matrix)
                    np.testing.assert_array_equal(
                        np.sqrt(dist_mod._squared_length(
                            images[:, 0], images[:, 1], images[:, 2], rows)),
                        dist_mod.paired_mic_distances(
                            displacements, np.zeros_like(displacements), lattice_matrix))

    def test_ties_are_resolved_consistently(self):
        """A displacement halfway between two images gets the same shift with and without numba, on repeated calls."""
        displacements = np.array([[0.5, 0.0, 0.0]])
        lattice_matrix = Lattice.cubic(4.0).matrix
        results = []
        for has_numba in BACKENDS:
            with patch.object(dist_mod, "HAS_NUMBA", has_numba):
                results += [dist_mod._nearest_image_shifts(displacements, lattice_matrix)
                            for _ in range(2)]
        for other in results[1:]:
            np.testing.assert_array_equal(other, results[0])

    def test_ties_do_not_depend_on_the_periodic_image_given(self):
        """Half-cell displacements given in different images are taken to the same image."""
        lattice_matrix = Lattice.cubic(4.0).matrix
        cases = (
            np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [-0.5, 0.0, 0.0],
                      [-1.5, 0.0, 0.0], [2.5, 0.0, 0.0], [-2.5, 0.0, 0.0]]),
            np.array([[0.5, 0.5, 0.5], [1.5, -0.5, 2.5], [-0.5, 1.5, -1.5]]),
            np.array([[0.5, 0.25, -0.5], [-1.5, 2.25, 1.5], [3.5, -0.75, 0.5]]),
        )
        for has_numba in BACKENDS:
            for displacements in cases:
                with self.subTest(numba=has_numba, displacements=displacements[0]), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    images = displacements + dist_mod._nearest_image_shifts(
                        displacements, lattice_matrix)
                    for image in images[1:]:
                        np.testing.assert_array_equal(image, images[0])

    def test_displacements_an_ulp_from_a_half_take_the_nearer_image(self):
        """One ULP below k + 0.5 on any axis keeps shift -k, one ULP above takes -(k + 1), and the distance is exact."""
        lattice_matrix = Lattice.cubic(4.0).matrix
        zeros = np.zeros((2, 3))
        for axis in range(3):
            for k in (-1, 0, 1):
                below = np.nextafter(k + 0.5, -math.inf)
                above = np.nextafter(k + 0.5, math.inf)
                displacements = np.zeros((2, 3))
                displacements[:, axis] = (below, above)
                expected_shifts = np.zeros((2, 3), dtype=np.int64)
                expected_shifts[:, axis] = (-k, -(k + 1))
                # The subtractions are exact, and so is scaling by 4.
                expected_distances = 4.0 * np.array([below - k, (k + 1) - above])
                for has_numba in BACKENDS:
                    with self.subTest(axis=axis, k=k, numba=has_numba), \
                            patch.object(dist_mod, "HAS_NUMBA", has_numba):
                        np.testing.assert_array_equal(
                            dist_mod._nearest_image_shifts(displacements, lattice_matrix),
                            expected_shifts)
                        np.testing.assert_array_equal(
                            dist_mod.paired_mic_distances(displacements, zeros, lattice_matrix),
                            expected_distances)
                        np.testing.assert_array_equal(
                            [dist_mod.mic_distance(d, zeros[0], lattice_matrix)
                             for d in displacements],
                            expected_distances)

    def test_empty_input_returns_empty(self):
        """No displacements give an empty (0, 3) array of shifts."""
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch.object(dist_mod, "HAS_NUMBA", has_numba):
                shifts = dist_mod._nearest_image_shifts(np.empty((0, 3)), np.eye(3))
                self.assertEqual(shifts.shape, (0, 3))
                self.assertEqual(shifts.dtype, np.int64)

    def test_invalid_input_raises_value_error(self):
        """Non-finite, misshapen or huge displacements and rejected lattices raise ValueError."""
        cubic = Lattice.cubic(4.0).matrix
        singular = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        cases = {
            "NaN displacement": (np.array([[np.nan, 0.0, 0.0]]), cubic,
                                 "displacements must be finite"),
            "wrong shape": (np.zeros((2, 2)), cubic,
                            r"displacements must have shape \(N, 3\), got \(2, 2\)"),
            "displacement beyond 2**52": (np.array([[2.0 ** 52, 0.0, 0.0]]), cubic,
                                         r"displacements must be smaller than 2\*\*52"),
            "lattice of wrong shape": (np.zeros((1, 3)), np.eye(2),
                                       r"lattice_matrix must have shape \(3, 3\), got \(2, 2\)"),
            "NaN lattice": (np.zeros((1, 3)), np.full((3, 3), np.nan),
                            "lattice_matrix must be finite"),
            "singular lattice": (np.zeros((1, 3)), singular,
                                 "lattice_matrix must not be singular, nearly singular or extremely elongated"),
        }
        for name, (displacements, lattice_matrix, message) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, message):
                    dist_mod._nearest_image_shifts(displacements, lattice_matrix)


class TestUndefinedDistances(unittest.TestCase):
    """Non-finite input and rejected lattices raise, or give NaN where unchecked."""

    # Singular, nearly singular or extremely elongated lattices.
    REJECTED_LATTICES = {
        "two parallel vectors": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        # Singular in exact arithmetic, but rounding leaves a volume of
        # about 1e-14 rather than 0.
        "third vector the sum of the others": (
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            @ Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix),
        # The third vector is only 2e-8 out of the plane of the other two.
        "flat cell": np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [3.0, 4.0, 2e-8]]),
        # Far from flat (a relative volume of 0.71), but with an edge about
        # 2e20 times its narrowest perpendicular width.
        "extremely elongated cell": np.array([[1e-10, 0.0, 0.0], [1e10, 1e10, 0.0], [0.0, 0.0, 1.0]]),
    }

    def test_mic_distance_of_non_finite_input_raises_value_error(self):
        """NaN or inf coordinates raise ValueError naming the argument, with and without numba."""
        good = np.array([0.1, 0.2, 0.3])
        for has_numba in BACKENDS:
            for value in (np.nan, np.inf):
                bad = np.array([value, 0.2, 0.3])
                with patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    with self.subTest(argument="frac1", value=value, numba=has_numba):
                        with self.assertRaisesRegex(ValueError, "frac1 must be finite"):
                            dist_mod.mic_distance(bad, good, 10.0 * np.eye(3))
                    with self.subTest(argument="frac2", value=value, numba=has_numba):
                        with self.assertRaisesRegex(ValueError, "frac2 must be finite"):
                            dist_mod.mic_distance(good, bad, 10.0 * np.eye(3))

    def test_mic_distance_in_non_finite_lattice_raises_value_error(self):
        """A lattice matrix with a NaN or inf entry raises ValueError naming it, with and without numba."""
        for has_numba in BACKENDS:
            for value in (np.nan, np.inf):
                lattice_matrix = 10.0 * np.eye(3)
                lattice_matrix[1, 2] = value
                with self.subTest(value=value, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    with self.assertRaisesRegex(ValueError, "lattice_matrix must be finite"):
                        dist_mod.mic_distance(np.array([0.1, 0.2, 0.3]), np.zeros(3), lattice_matrix)

    def test_mic_distance_in_rejected_lattice_raises_value_error(self):
        """A singular, nearly singular or extremely elongated lattice raises ValueError, with and without numba."""
        for name, lattice_matrix in self.REJECTED_LATTICES.items():
            for has_numba in BACKENDS:
                with self.subTest(name, numba=has_numba), \
                        patch.object(dist_mod, "HAS_NUMBA", has_numba):
                    with self.assertRaisesRegex(
                            ValueError,
                            "lattice_matrix must not be singular, nearly singular or extremely elongated"):
                        dist_mod.mic_distance(np.array([0.1, 0.2, 0.3]), np.zeros(3), lattice_matrix)

    def test_paired_mic_distances_rejects_lattice(self):
        """paired_mic_distances raises ValueError for a singular, nearly singular or extremely elongated lattice."""
        for name, lattice_matrix in self.REJECTED_LATTICES.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, "lattice_matrix"):
                    dist_mod.paired_mic_distances(
                        np.zeros((2, 3)), np.zeros((2, 3)), lattice_matrix)

    def test_unchecked_paired_distances_give_nan_for_non_finite_pairs(self):
        """A non-finite pair gives NaN and leaves the other pairs unchanged."""
        # numpy is made to raise on invalid operations, such as converting
        # NaN to an integer, which gives a different number on each
        # platform. In this cell the finite pairs need only their rounded
        # image, so the test cannot hang.
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
        """The single-pair distance with numba matches pymatgen."""
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        rng = np.random.default_rng(42)
        for _ in range(100):
            frac1 = rng.random(3)
            frac2 = rng.random(3)
            expected = float(lattice.get_distance_and_image(frac1, frac2)[0])
            result = dist_mod.mic_distance(frac1, frac2, lattice.matrix)
            self.assertAlmostEqual(result, expected, places=10)

    def test_small_and_large_batches_agree(self):
        """Small batches, on one thread, and large batches, in parallel, give identical distances."""
        # A typical cell; a sheared cell in which about half the pairs are
        # searched by the ball; and an elongated cell in which long pairs
        # have boxes of many shifts.
        cells = ("triclinic", "cubic lattice in a cell sheared 5x, vectors rolled 0",
                 "orthorhombic 10x10x200")
        rng = np.random.default_rng(7)
        n = 2 * dist_mod._PARALLEL_MIN_PAIRS
        small_n = dist_mod._PARALLEL_MIN_PAIRS - 1
        for name in cells:
            lattice_matrix = VARIED_CELLS[name]
            frac1 = rng.uniform(-2.0, 3.0, (n, 3))
            frac2 = rng.uniform(-2.0, 3.0, (n, 3))
            with self.subTest(cell=name):
                large = dist_mod.paired_mic_distances(frac1, frac2, lattice_matrix)
                small = dist_mod.paired_mic_distances(
                    frac1[:small_n], frac2[:small_n], lattice_matrix)
                np.testing.assert_array_equal(small, large[:small_n])


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestNearestImageShiftsNumba(unittest.TestCase):
    """Nearest-image shifts are the same with and without numba."""

    def test_numba_and_numpy_shifts_are_identical(self):
        """Both paths give identical shifts in near-cubic, sheared, thin and unreduced cells, on the half-integer grid too."""
        rng = np.random.default_rng(0)
        for name, lattice_matrix in random_lattices(rng).items():
            displacements = np.concatenate([random_displacements(rng, 30), HALF_INTEGER_GRID])
            with self.subTest(cell=name):
                with patch.object(dist_mod, "HAS_NUMBA", False):
                    without_numba = dist_mod._nearest_image_shifts(displacements, lattice_matrix)
                np.testing.assert_array_equal(
                    dist_mod._nearest_image_shifts(displacements, lattice_matrix), without_numba)
