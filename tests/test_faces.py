import dataclasses
import itertools
import unittest

import numpy as np

from site_analysis import Face
from site_analysis.faces import _faces_and_keys


def regular_vertices(kind):
    """Cartesian vertices of a polyhedron centred at (5, 5, 5) Angstrom."""
    if kind == "tetrahedron":
        v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    elif kind == "octahedron":
        v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                      [0, 0, 1], [0, 0, -1]], float) * 1.5
    elif kind == "cube":
        v = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    elif kind == "trigonal prism":
        angles = np.arange(3) * 2 * np.pi / 3
        v = np.array([[np.cos(a), np.sin(a), z] for z in (-0.75, 0.75) for a in angles])
    elif kind == "hexagonal prism":
        angles = np.arange(6) * np.pi / 3
        v = np.array([[np.cos(a), np.sin(a), z] for z in (-0.75, 0.75) for a in angles])
    return v + 5.0


def face_sizes(kind, displacement=None):
    """Number of vertices of each face, from a 10 Angstrom cubic cell."""
    cartesian = regular_vertices(kind)
    if displacement is not None:
        cartesian[0] += displacement
    lattice = np.eye(3) * 10.0
    faces = _faces_and_keys(list(range(len(cartesian))), cartesian / 10.0,
                            np.array([0.5, 0.5, 0.5]), lattice)
    return sorted(len(face.vertex_indices) for face, _ in faces)


class FaceTestCase(unittest.TestCase):

    def test_faces_compare_and_hash_by_vertex_indices(self):
        self.assertEqual(Face((1, 2, 3)), Face((1, 2, 3)))
        self.assertNotEqual(Face((1, 2, 3)), Face((1, 2, 4)))
        self.assertEqual(len({Face((1, 2, 3)), Face((1, 2, 3))}), 1)

    def test_face_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Face((1, 2, 3)).vertex_indices = (4, 5, 6)


class FacesAndKeysTestCase(unittest.TestCase):

    def test_tetrahedron_has_four_triangles(self):
        self.assertEqual(face_sizes("tetrahedron"), [3, 3, 3, 3])

    def test_octahedron_has_eight_triangles(self):
        self.assertEqual(face_sizes("octahedron"), [3] * 8)

    def test_cube_has_six_squares(self):
        self.assertEqual(face_sizes("cube"), [4] * 6)

    def test_trigonal_prism_has_two_triangles_and_three_squares(self):
        self.assertEqual(face_sizes("trigonal prism"), [3, 3, 4, 4, 4])

    def test_hexagonal_faces_are_merged_from_several_triangles(self):
        self.assertEqual(face_sizes("hexagonal prism"), [4] * 6 + [6, 6])

    def test_distortion_within_the_tolerance_still_merges(self):
        self.assertEqual(face_sizes("cube", displacement=[0.003, 0.003, 0.003]), [4] * 6)

    def test_distortion_beyond_the_tolerance_does_not_merge(self):
        self.assertGreater(len(face_sizes("cube", displacement=[0.1, 0.1, 0.1])), 6)

    def test_vertex_indices_are_atom_indices_in_ascending_order(self):
        cartesian = regular_vertices("tetrahedron")
        faces = _faces_and_keys([40, 7, 12, 3], cartesian / 10.0,
                                np.array([0.5, 0.5, 0.5]), np.eye(3) * 10.0)
        self.assertEqual([face.vertex_indices for face, _ in faces],
                         [(3, 7, 12), (3, 7, 40), (3, 12, 40), (7, 12, 40)])

    def test_keys_do_not_depend_on_a_lattice_translation(self):
        cartesian = regular_vertices("tetrahedron")
        lattice = np.eye(3) * 10.0
        first = _faces_and_keys([0, 1, 2, 3], cartesian / 10.0, None, lattice)
        second = _faces_and_keys([0, 1, 2, 3], cartesian / 10.0 + [1.0, 0.0, -2.0], None, lattice)
        self.assertEqual([key for _, key in first], [key for _, key in second])

    def test_degenerate_vertices_raise(self):
        flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float) + 5.0
        with self.assertRaises(ValueError):
            _faces_and_keys([0, 1, 2, 3], flat / 10.0, None, np.eye(3) * 10.0)


if __name__ == "__main__":
    unittest.main()
