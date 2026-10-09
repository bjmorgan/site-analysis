"""Tests for the periodic KD-tree neighbour search."""

import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice

from site_analysis.distances import paired_mic_distances
from site_analysis.neighbour_search import PeriodicNeighbourIndex


# Orthogonal, monoclinic, hexagonal, triclinic and rhombohedral cells, and
# a thin hexagonal supercell in which the nearest image of a pair can lie
# several cells from the image nearest in fractional coordinates (#84).
CELLS = {
    "cubic": Lattice.cubic(10.0).matrix,
    "orthorhombic": Lattice.orthorhombic(4.0, 9.0, 6.0).matrix,
    "monoclinic": Lattice.monoclinic(5.0, 6.0, 7.0, 110).matrix,
    "hexagonal": Lattice.hexagonal(5.0, 8.0).matrix,
    "triclinic": Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix,
    "rhombohedral": Lattice.from_parameters(6.0, 6.0, 6.0, 60, 60, 60).matrix,
    "thin hexagonal 1x10x1": (
        Lattice.hexagonal(3.0, 4.0).matrix * np.array([[1.0], [10.0], [1.0]])),
}


def brute_force_distances(query, points, lattice_matrix):
    """Distance from every query point to every point, shape (M, N)."""
    q, p = np.meshgrid(np.arange(len(query)), np.arange(len(points)), indexing="ij")
    distances = paired_mic_distances(query[q.ravel()], points[p.ravel()], lattice_matrix)
    return distances.reshape(len(query), len(points))


def random_points(rng):
    """Random points and query points, including points on cell boundaries."""
    points = rng.uniform(-1.0, 2.0, (150, 3))
    query = rng.uniform(-1.0, 2.0, (100, 3))
    query[:10] = points[:10]  # coincident points
    points[10] = [-1e-18, 0.5, 0.5]  # wraps to exactly 1.0
    points[11] = [np.nextafter(1.0, 0.0), 0.2, 0.3]
    points[12] = [0.0, 0.0, 0.0]
    points[13] = points[3]  # a duplicate indexed point, at query point 3
    return points, query


def shuffled_grid(rng):
    """A 4 x 4 x 4 grid of fractional coordinates in random order."""
    grid = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)],
                    dtype=float) / 4
    return grid[rng.permutation(len(grid))]


class TestPeriodicNeighbourIndexConstruction(unittest.TestCase):
    """Tests for building a PeriodicNeighbourIndex."""

    def test_len_is_number_of_points(self):
        """The index reports the number of points it holds."""
        index = PeriodicNeighbourIndex(np.random.default_rng(0).random((5, 3)), np.eye(3) * 4.0)
        self.assertEqual(len(index), 5)

    def test_rejects_coords_with_wrong_shape(self):
        """Coordinates that are not shaped (N, 3) raise ValueError."""
        for coords in (np.zeros((4, 2)), np.zeros(3)):
            with self.subTest(shape=coords.shape):
                with self.assertRaisesRegex(ValueError, "frac_coords must have shape"):
                    PeriodicNeighbourIndex(coords, np.eye(3))

    def test_copies_its_inputs(self):
        """Changing the caller's arrays afterwards does not change the index."""
        frac_coords = np.array([[0.1, 0.1, 0.1]])
        lattice_matrix = np.eye(3) * 10.0
        index = PeriodicNeighbourIndex(frac_coords, lattice_matrix)
        frac_coords[0] = [0.6, 0.6, 0.6]
        lattice_matrix[2, 2] = 1.0
        _, distances = index.query_nearest(np.array([[0.1, 0.1, 0.2]]))
        self.assertAlmostEqual(distances[0], 1.0)

    def test_rejects_non_finite_coords(self):
        """Coordinates that are NaN or infinite raise ValueError."""
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "frac_coords must be finite"):
                    PeriodicNeighbourIndex(np.array([[0.1, value, 0.2]]), np.eye(3))

    def test_rejects_invalid_lattice(self):
        """A lattice matrix that is not (3, 3), not finite, or singular raises ValueError."""
        lattices = {
            "(2, 2)": np.eye(2),
            "(3, 4)": np.eye(3, 4),
            "zero-length vector": np.diag([1.0, 1.0, 0.0]),
            "coplanar vectors": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            "non-finite": np.diag([1.0, 1.0, np.nan]),
        }
        for name, lattice_matrix in lattices.items():
            with self.subTest(lattice=name):
                with self.assertRaisesRegex(ValueError, "lattice_matrix"):
                    PeriodicNeighbourIndex(np.zeros((2, 3)), lattice_matrix)

    def test_accepts_points_that_wrap_to_box_edge(self):
        """A tiny negative coordinate, which wraps to exactly 1.0, is accepted."""
        index = PeriodicNeighbourIndex(np.array([[-1e-18, 0.5, 0.5]]), np.eye(3) * 4.0)
        self.assertEqual(len(index), 1)


class TestQueryWithin(unittest.TestCase):
    """Tests for PeriodicNeighbourIndex.query_within."""

    def test_matches_brute_force(self):
        """Pairs, distances and order match a search over every pair."""
        rng = np.random.default_rng(1)
        for name, lattice_matrix in CELLS.items():
            points, query = random_points(rng)
            index = PeriodicNeighbourIndex(points, lattice_matrix)
            dense = brute_force_distances(query, points, lattice_matrix)
            half_cell = np.linalg.norm(lattice_matrix, axis=1).min() / 2
            for cutoff in (0.5, 1.5, 3.0, half_cell, np.inf):
                with self.subTest(cell=name, cutoff=cutoff):
                    query_idx, point_idx, distances = index.query_within(query, cutoff)
                    rows, cols = np.nonzero(dense <= cutoff)
                    order = np.lexsort((cols, dense[rows, cols], rows))
                    np.testing.assert_array_equal(query_idx, rows[order])
                    np.testing.assert_array_equal(point_idx, cols[order])
                    np.testing.assert_array_equal(distances, dense[rows, cols][order])

    def test_equidistant_neighbours_in_index_order(self):
        """Neighbours exactly at the cutoff are kept, in point-index order."""
        points = shuffled_grid(np.random.default_rng(2))
        lattice_matrix = np.eye(3) * 4.0
        index = PeriodicNeighbourIndex(points, lattice_matrix)
        # The centre of a grid cube is equidistant from its eight corners,
        # and the cutoff is exactly that distance.
        query = np.array([[0.125, 0.125, 0.125]])
        cutoff = brute_force_distances(query, points, lattice_matrix).min()
        query_idx, point_idx, distances = index.query_within(query, cutoff)
        corners = np.nonzero(np.all(np.isin(points, [0.0, 0.25]), axis=1))[0]
        np.testing.assert_array_equal(point_idx, np.sort(corners))
        self.assertEqual(len(set(distances.tolist())), 1)

    def test_pairs_exactly_at_cutoff_on_search_edge(self):
        """Pairs exactly at the cutoff are found where the search is tightest.

        Each pair is separated along the direction in which the Cartesian
        distance is smallest relative to the tree's scaled distance, so in
        a non-orthogonal cell it lies exactly on the edge of the candidate
        search.
        """
        rng = np.random.default_rng(9)
        for name, lattice_matrix in CELLS.items():
            with self.subTest(cell=name):
                lengths = np.linalg.norm(lattice_matrix, axis=1)
                left, _, _ = np.linalg.svd(lattice_matrix / lengths[:, np.newaxis])
                points = rng.random((20, 3))
                query = points + rng.uniform(0.5, 2.0, (20, 1)) * left[:, -1] / lengths
                cutoffs = paired_mic_distances(query, points, lattice_matrix)
                index = PeriodicNeighbourIndex(points, lattice_matrix)
                missed = [i for i, cutoff in enumerate(cutoffs)
                          if i not in index.query_within(query[i:i + 1], cutoff)[1]]
                self.assertEqual(missed, [])

    def test_coincident_points_far_outside_cell(self):
        """Copies of the points 1000 cells away are found with a zero cutoff."""
        lattice_matrix = np.eye(3) * 100.0
        points = np.random.default_rng(7).random((200, 3))
        query = points + 1000.0
        dense = brute_force_distances(query, points, lattice_matrix)
        rows, cols = np.nonzero(dense <= 0.0)
        self.assertEqual(len(rows), 200)
        query_idx, point_idx, _ = PeriodicNeighbourIndex(
            points, lattice_matrix).query_within(query, 0.0)
        np.testing.assert_array_equal(query_idx, rows)
        np.testing.assert_array_equal(point_idx, cols)

    def test_finds_pair_whose_nearest_image_is_several_cells_away(self):
        """The #84 pair, 6 * sqrt(3) apart, is found with a 10.5 cutoff."""
        index = PeriodicNeighbourIndex(np.zeros((1, 3)), CELLS["thin hexagonal 1x10x1"])
        _, point_idx, distances = index.query_within(np.array([[0.0, 0.4, 0.0]]), 10.5)
        np.testing.assert_array_equal(point_idx, [0])
        self.assertAlmostEqual(distances[0], 6 * np.sqrt(3), places=12)

    def test_invalid_cutoff_raises(self):
        """A negative or NaN cutoff raises ValueError."""
        index = PeriodicNeighbourIndex(np.zeros((1, 3)), np.eye(3))
        for cutoff in (-0.1, float("nan")):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    index.query_within(np.zeros((1, 3)), cutoff)

    def test_wrong_query_shape_raises(self):
        """Query coordinates that are not shaped (M, 3) raise ValueError."""
        index = PeriodicNeighbourIndex(np.zeros((1, 3)), np.eye(3))
        with self.assertRaises(ValueError):
            index.query_within(np.zeros((1, 2)), 1.0)

    def test_non_finite_query_raises(self):
        """Query coordinates that are NaN or infinite raise ValueError."""
        index = PeriodicNeighbourIndex(np.zeros((1, 3)), np.eye(3))
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "query_frac must be finite"):
                    index.query_within(np.array([[value, 0.0, 0.0]]), 1.0)

    def test_empty_queries_return_empty_arrays(self):
        """No query points give empty arrays of the right types."""
        index = PeriodicNeighbourIndex(np.zeros((3, 3)), np.eye(3))
        query_idx, point_idx, distances = index.query_within(np.empty((0, 3)), 1.0)
        self.assertEqual((query_idx.shape, point_idx.shape, distances.shape), ((0,), (0,), (0,)))
        self.assertEqual((query_idx.dtype, point_idx.dtype, distances.dtype),
                         (np.dtype(np.intp), np.dtype(np.intp), np.dtype(np.float64)))

    def test_empty_index_returns_empty_arrays(self):
        """An empty index finds no neighbours."""
        index = PeriodicNeighbourIndex(np.empty((0, 3)), np.eye(3))
        query_idx, point_idx, distances = index.query_within(np.zeros((2, 3)), 1.0)
        self.assertEqual((len(query_idx), len(point_idx), len(distances)), (0, 0, 0))


class TestQueryNearest(unittest.TestCase):
    """Tests for PeriodicNeighbourIndex.query_nearest."""

    def test_matches_brute_force(self):
        """Nearest points and distances match a search over every pair."""
        rng = np.random.default_rng(3)
        for name, lattice_matrix in CELLS.items():
            with self.subTest(cell=name):
                points, query = random_points(rng)
                dense = brute_force_distances(query, points, lattice_matrix)
                point_idx, distances = PeriodicNeighbourIndex(
                    points, lattice_matrix).query_nearest(query)
                np.testing.assert_array_equal(point_idx, dense.argmin(axis=1))
                np.testing.assert_array_equal(distances, dense.min(axis=1))

    def test_ties_resolve_to_lowest_index(self):
        """A query equidistant from several points gets the lowest index."""
        lattice_matrix = np.eye(3) * 4.0
        points = shuffled_grid(np.random.default_rng(4))
        # Each grid-cube centre is equidistant from its eight corners.
        query = points + 0.125
        dense = brute_force_distances(query, points, lattice_matrix)
        n_tied = (dense == dense.min(axis=1, keepdims=True)).sum(axis=1)
        np.testing.assert_array_equal(n_tied, np.full(len(query), 8))
        point_idx, _ = PeriodicNeighbourIndex(points, lattice_matrix).query_nearest(query)
        np.testing.assert_array_equal(point_idx, dense.argmin(axis=1))

    def test_queries_far_outside_cell(self):
        """Query points many cells away find the same nearest points."""
        rng = np.random.default_rng(5)
        triclinic_points = rng.random((50, 3))
        cubic_points = rng.random((200, 3))
        cases = {
            "random queries, triclinic": (
                CELLS["triclinic"], triclinic_points, rng.uniform(-50.0, 50.0, (40, 3))),
            # Rounding errors in the distances grow with the coordinates,
            # of both the query points and the indexed points, and with the
            # longest lattice vector.
            "copies 1000 cells away, cubic": (
                np.eye(3) * 100.0, cubic_points, cubic_points + 1000.0),
            "indexed points 1000 cells away, cubic": (
                np.eye(3) * 100.0, cubic_points + 1000.0, cubic_points),
            "copies 100 cells away, 1 x 1 x 1000 cell": (
                Lattice.orthorhombic(1.0, 1.0, 1000.0).matrix, cubic_points, cubic_points + 100.0),
        }
        for name, (lattice_matrix, points, query) in cases.items():
            with self.subTest(case=name):
                dense = brute_force_distances(query, points, lattice_matrix)
                point_idx, distances = PeriodicNeighbourIndex(
                    points, lattice_matrix).query_nearest(query)
                np.testing.assert_array_equal(point_idx, dense.argmin(axis=1))
                np.testing.assert_array_equal(distances, dense.min(axis=1))

    def test_single_point_index(self):
        """With one indexed point, every query finds it."""
        lattice_matrix = CELLS["monoclinic"]
        point = np.array([[0.3, 0.6, 0.9]])
        query = np.random.default_rng(6).random((10, 3))
        point_idx, distances = PeriodicNeighbourIndex(point, lattice_matrix).query_nearest(query)
        np.testing.assert_array_equal(point_idx, np.zeros(10, dtype=np.intp))
        np.testing.assert_array_equal(
            distances, paired_mic_distances(query, np.repeat(point, 10, axis=0), lattice_matrix))

    def test_query_point_without_candidate_raises(self):
        """A query point with no candidate raises instead of being dropped."""
        index = PeriodicNeighbourIndex(np.zeros((1, 3)), np.eye(3))
        cases = {
            # (query_idx, point_idx, distances) from the candidate search.
            "no query has a candidate": (
                np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)),
            "only the first query has a candidate": (
                np.array([0], dtype=np.intp), np.array([0], dtype=np.intp), np.array([0.0])),
        }
        for name, candidates in cases.items():
            with self.subTest(name):
                with patch.object(index, "_candidates", return_value=candidates):
                    with self.assertRaises(RuntimeError):
                        index.query_nearest(np.zeros((2, 3)))

    def test_empty_index_raises(self):
        """An empty index has no nearest point, so raises ValueError."""
        index = PeriodicNeighbourIndex(np.empty((0, 3)), np.eye(3))
        with self.assertRaises(ValueError):
            index.query_nearest(np.zeros((1, 3)))

    def test_non_finite_query_raises(self):
        """Query coordinates that are NaN or infinite raise ValueError."""
        index = PeriodicNeighbourIndex(np.zeros((1, 3)), np.eye(3))
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "query_frac must be finite"):
                    index.query_nearest(np.array([[value, 0.0, 0.0]]))

    def test_empty_queries_return_empty_arrays(self):
        """No query points give empty arrays of the right types."""
        index = PeriodicNeighbourIndex(np.zeros((3, 3)), np.eye(3))
        point_idx, distances = index.query_nearest(np.empty((0, 3)))
        self.assertEqual((point_idx.shape, distances.shape), ((0,), (0,)))
        self.assertEqual((point_idx.dtype, distances.dtype),
                         (np.dtype(np.intp), np.dtype(np.float64)))


if __name__ == "__main__":
    unittest.main()
