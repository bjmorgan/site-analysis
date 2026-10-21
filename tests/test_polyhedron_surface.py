import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from site_analysis._compat import HAS_NUMBA
from site_analysis.polyhedron_surface import _SurfaceCache, _image_shifts, _surface_triangles

if HAS_NUMBA:
    from site_analysis.polyhedron_surface import (
        _numba_star_planes, _numba_star_query, _numpy_star_planes, _numpy_star_query)

# Containment with numba where it is installed, and with numpy.
BACKENDS = [True, False] if HAS_NUMBA else [False]

# Corners of a cube of edge 0.4 centred at (0.5, 0.5, 0.5), in fractional
# coordinates of a cubic cell; corner i has offsets (i >> 2, (i >> 1) & 1, i & 1).
_CUBE = 0.3 + 0.4 * np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
_CUBE_INDICES = list(range(10, 18))
# The top face (z = 0.7) as local indices, in cyclic order: a, b, c, d.
_TOP = [1, 3, 7, 5]


def _atom_triples(triangles: np.ndarray, vertex_indices: list[int]) -> set[tuple[int, ...]]:
    """The triangles as sorted triples of atom indices."""
    return {tuple(sorted(vertex_indices[i] for i in t)) for t in triangles}


def _cube_with_corner_moved(local: int, dz: float) -> np.ndarray:
    """The cube with one corner of its top face moved by dz along z."""
    coords = _CUBE.copy()
    coords[local, 2] += dz
    return coords


def _top_face_diagonal(coords: np.ndarray) -> frozenset[int]:
    """The diagonal of the top face that the convex hull of the coordinates splits it along."""
    on_top = [set(t) for t in ConvexHull(coords).simplices if set(t) <= set(_TOP)]
    return frozenset(on_top[0] & on_top[1])


class TestSurfaceTriangles(unittest.TestCase):

    def test_cube_surface_is_twelve_outward_triangles(self):
        """A cube with no shared atoms has 12 triangles, each wound to face away from the centre."""
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        a, b, c = (_CUBE[triangles[:, k]] for k in range(3))
        outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - _CUBE.mean(axis=0))
        self.assertEqual(triangles.shape, (12, 3))
        self.assertEqual(triangles.dtype, np.int64)
        self.assertTrue(np.all(outward > 0.0))

    def test_triangles_face_outward_in_a_left_handed_cell(self):
        """Triangles face away from the centre in the given coordinates for a left-handed lattice."""
        lattice_matrix = np.diag([10.0, 10.0, -10.0])
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, lattice_matrix, [])
        a, b, c = (_CUBE[triangles[:, k]] for k in range(3))
        outward = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - _CUBE.mean(axis=0))
        self.assertTrue(np.all(outward > 0.0))

    def test_shared_square_is_fanned_whichever_diagonal_the_hull_chose(self):
        """Two cubes whose hulls split a shared square differently give it the same triangles."""
        raised = _cube_with_corner_moved(_TOP[0], 1e-6)
        lowered = _cube_with_corner_moved(_TOP[0], -1e-6)
        self.assertNotEqual(_top_face_diagonal(raised), _top_face_diagonal(lowered))
        square = [_CUBE_INDICES[i] for i in _TOP]
        on_square = []
        for coords in (raised, lowered):
            triples = _atom_triples(
                _surface_triangles(_CUBE_INDICES, coords, np.eye(3), [square]), _CUBE_INDICES)
            on_square.append({t for t in triples if set(t) <= set(square)})
        self.assertEqual(len(on_square[0]), 2)
        self.assertEqual(on_square[0], on_square[1])

    def test_shared_triangle_splits_its_square_along_its_diagonal(self):
        """A shared half {a, b, c} of a square is kept with the leftover {a, c, d}, whichever diagonal the hull chose."""
        a, b, c, d = (_CUBE_INDICES[i] for i in _TOP)
        for dz in (1e-6, -1e-6):
            with self.subTest(dz=dz):
                coords = _cube_with_corner_moved(_TOP[0], dz)
                triples = _atom_triples(
                    _surface_triangles(_CUBE_INDICES, coords, np.eye(3), [{a, b, c}]), _CUBE_INDICES)
                self.assertEqual({t for t in triples if set(t) <= {a, b, c, d}},
                                 {tuple(sorted((a, b, c))), tuple(sorted((a, c, d)))})
                self.assertEqual(len(triples), 12)

    def test_both_halves_of_a_square_along_one_diagonal_are_kept(self):
        """Shared triangles {a, b, c} and {a, c, d} both become triangles of the surface."""
        a, b, c, d = (_CUBE_INDICES[i] for i in _TOP)
        coords = _cube_with_corner_moved(_TOP[1], 1e-6)
        triples = _atom_triples(
            _surface_triangles(_CUBE_INDICES, coords, np.eye(3), [{a, b, c}, {a, c, d}]),
            _CUBE_INDICES)
        self.assertLessEqual({tuple(sorted((a, b, c))), tuple(sorted((a, c, d)))}, triples)

    def test_halves_of_a_square_along_both_diagonals_cannot_be_triangulated_identically(self):
        """Shared triangles {a, b, c} and {a, b, d} of one square raise ValueError."""
        a, b, c, d = (_CUBE_INDICES[i] for i in _TOP)
        with self.assertRaisesRegex(ValueError, "cannot be triangulated identically"):
            _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [{a, b, c}, {a, b, d}])

    def test_part_of_a_hexagonal_face_cannot_be_triangulated_identically(self):
        """Four atoms of a hexagonal prism's hexagon, which its triangles do not cover, raise ValueError."""
        angles = np.arange(6) * np.pi / 3
        hexagon = np.stack([0.5 + 0.2 * np.cos(angles), 0.5 + 0.2 * np.sin(angles),
                            np.full(6, 0.3)], axis=1)
        # Denting alternate corners of the bottom hexagon fixes its triangles
        # as the central triangle {0, 2, 4} and three ears.
        bottom = hexagon.copy()
        bottom[1::2, 2] += 1e-6
        coords = np.vstack([bottom, hexagon + [0.0, 0.0, 0.4]])
        vertex_indices = list(range(20, 32))
        with self.assertRaisesRegex(ValueError, "cannot be triangulated identically"):
            _surface_triangles(vertex_indices, coords, np.eye(3), [{20, 21, 22, 23}])


class TestSurfaceCache(unittest.TestCase):

    def test_convex_cube_containment_matches_its_convex_hull(self):
        """For a cube, contains_point agrees with convex hull containment on 5,000 random points."""
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        points = np.random.default_rng(0).random((5000, 3))
        expected = Delaunay(_CUBE).find_simplex(points) >= 0
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch("site_analysis.polyhedron_surface.HAS_NUMBA", has_numba):
                cache = _SurfaceCache(triangles, _CUBE)
                np.testing.assert_array_equal(
                    [cache.contains_point(p) for p in points], expected)

    def test_bent_cube_is_bounded_by_its_fixed_triangles(self):
        """A point in the dent of a bent cube is outside, though inside its convex hull."""
        square = [_CUBE_INDICES[i] for i in _TOP]
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [square])
        # The fan from the lowest atom (local 1) splits the top face along 1-7;
        # moving both ends of that diagonal down by 5% of the edge dents it.
        bent = _CUBE.copy()
        bent[[1, 7], 2] -= 0.02
        in_dent = np.array([0.5, 0.5, 0.69])
        below_dent = np.array([0.5, 0.5, 0.67])
        self.assertGreaterEqual(Delaunay(bent).find_simplex(in_dent), 0)
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch("site_analysis.polyhedron_surface.HAS_NUMBA", has_numba):
                cache = _SurfaceCache(triangles, _CUBE)
                cache.update(bent)
                self.assertFalse(cache.contains_point(in_dent))
                self.assertTrue(cache.contains_point(below_dent))

    def test_contains_point_in_another_image(self):
        """A point given in another periodic image is contained."""
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch("site_analysis.polyhedron_surface.HAS_NUMBA", has_numba):
                cache = _SurfaceCache(triangles, _CUBE)
                self.assertTrue(cache.contains_point(np.array([1.5, -0.5, 2.5])))

    def test_vertex_moved_past_the_centre_is_too_distorted(self):
        """Moving a vertex past the centre makes update raise RuntimeError."""
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        distorted = _CUBE.copy()
        distorted[0] = [0.65, 0.65, 0.65]
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch("site_analysis.polyhedron_surface.HAS_NUMBA", has_numba):
                cache = _SurfaceCache(triangles, _CUBE)
                with self.assertRaisesRegex(RuntimeError, "too distorted"):
                    cache.update(distorted)

    def test_image_shifts_for_vertices_within_two_cells(self):
        """Vertices with every coordinate in [0, 2) give the eight shifts with components in {0, 1}."""
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch("site_analysis.polyhedron_surface.HAS_NUMBA", has_numba):
                cache = _SurfaceCache(triangles, _CUBE + 1.0)
                self.assertEqual({tuple(s) for s in cache.image_shifts},
                                 set(itertools.product((0, 1), repeat=3)))

    def test_image_shifts_for_vertices_beyond_two_cells(self):
        """Vertices with a coordinate below 0 give the shifts of every cell they meet."""
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch("site_analysis.polyhedron_surface.HAS_NUMBA", has_numba):
                cache = _SurfaceCache(triangles, _CUBE - [0.5, 0.0, 0.0])
                self.assertEqual({tuple(s) for s in cache.image_shifts}, {(-1, 0, 0), (0, 0, 0)})


@unittest.skipUnless(HAS_NUMBA, "numba is not installed")
class TestNumbaMatchesNumpy(unittest.TestCase):

    @staticmethod
    def _random_sites(n_sites: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Noisy cubes at random positions, with the triangles of the undistorted cube."""
        rng = np.random.default_rng(1)
        triangles = _surface_triangles(_CUBE_INDICES, _CUBE, np.eye(3), [])
        return [(_CUBE + rng.uniform(-1.0, 1.5, 3) + rng.normal(scale=0.03, size=_CUBE.shape),
                 triangles) for _ in range(n_sites)]

    def test_planes_are_identical(self):
        """Both backends give bit-identical planes, star-shaped flags and within-two-cells flags."""
        for coords, triangles in self._random_sites(50):
            numba_result = _numba_star_planes(coords, triangles)
            numpy_result = _numpy_star_planes(coords, triangles)
            self.assertEqual(numba_result[0].tobytes(), numpy_result[0].tobytes())
            self.assertEqual(numba_result[1].tobytes(), numpy_result[1].tobytes())
            self.assertEqual(numba_result[2:], numpy_result[2:])

    def test_queries_are_identical(self):
        """Both backends give the same containment for random points."""
        rng = np.random.default_rng(2)
        for coords, triangles in self._random_sites(20):
            normals, refs, _, _ = _numpy_star_planes(coords, triangles)
            shifts = _image_shifts(coords)
            for point in rng.uniform(-1.0, 2.0, (300, 3)):
                self.assertEqual(bool(_numba_star_query(point, shifts, normals, refs)),
                                 _numpy_star_query(point, shifts, normals, refs))


class TestImageShifts(unittest.TestCase):
    """Tests for the limit on the cells in which a site's images are tested."""

    def test_image_shifts_rejects_vertices_spanning_too_many_cells(self):
        """Vertices spanning 1001 cells along x are rejected, naming the cells spanned."""
        vertices = _CUBE[:4] + [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        with self.assertRaisesRegex(ValueError, r"span \[1001, 1, 1\] cells"):
            _image_shifts(vertices)

    def test_image_shifts_allows_vertices_spanning_the_limit(self):
        """Vertices spanning 10 cells along each axis give 1000 shifts."""
        vertices = _CUBE[:4] + [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]]
        self.assertEqual(_image_shifts(vertices).shape, (1000, 3))


if __name__ == "__main__":
    unittest.main()
