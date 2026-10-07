"""Tests for the periodic KD-tree neighbour search."""

import unittest

import numpy as np
from pymatgen.core import Lattice

from site_analysis.distances import paired_mic_distances
from site_analysis.neighbour_search import PeriodicNeighbourIndex


# Orthogonal, monoclinic, hexagonal, triclinic and rhombohedral cells.
CELLS = {
    "cubic": Lattice.cubic(10.0).matrix,
    "orthorhombic": Lattice.orthorhombic(4.0, 9.0, 6.0).matrix,
    "monoclinic": Lattice.monoclinic(5.0, 6.0, 7.0, 110).matrix,
    "hexagonal": Lattice.hexagonal(5.0, 8.0).matrix,
    "triclinic": Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60).matrix,
    "rhombohedral": Lattice.from_parameters(6.0, 6.0, 6.0, 60, 60, 60).matrix,
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

    def test_rejects_invalid_lattice(self):
        """A lattice matrix that is not (3, 3), or is singular, raises ValueError."""
        lattices = {
            "(2, 2)": np.eye(2),
            "(3, 4)": np.eye(3, 4),
            "zero-length vector": np.diag([1.0, 1.0, 0.0]),
            "coplanar vectors": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        }
        for name, lattice_matrix in lattices.items():
            with self.subTest(lattice=name):
                with self.assertRaisesRegex(ValueError, "lattice_matrix"):
                    PeriodicNeighbourIndex(np.zeros((2, 3)), lattice_matrix)

    def test_accepts_points_that_wrap_to_box_edge(self):
        """A tiny negative coordinate, which wraps to exactly 1.0, is accepted."""
        index = PeriodicNeighbourIndex(np.array([[-1e-18, 0.5, 0.5]]), np.eye(3) * 4.0)
        self.assertEqual(len(index), 1)


if __name__ == "__main__":
    unittest.main()
