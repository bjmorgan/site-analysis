import dataclasses
import itertools
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from site_analysis import Face, TrajectoryBuilder
from site_analysis.atom import Atom
from site_analysis.faces import _faces_and_keys
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.site import Site
from site_analysis.spherical_site import SphericalSite
from site_analysis.trajectory import Trajectory


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


def simple_cubic_cubes(n):
    """Atoms on an n x n x n simple cubic grid (2 Angstrom spacing) and the cube sites between them."""
    points = list(itertools.product(range(n), repeat=3))
    index = {point: k for k, point in enumerate(points)}
    structure = Structure(Lattice.cubic(2.0 * n), ["O"] * len(points),
                          np.array(points, float) / n)
    Site._newid = 0
    sites = [PolyhedralSite(
        vertex_indices=[index[tuple((np.array(corner) + offset) % n)]
                        for offset in itertools.product((0, 1), repeat=3)],
        reference_center=(np.array(corner) + 0.5) / n)
        for corner in points]
    return structure, sites


def adjacent_pairs(trajectory):
    adjacency = trajectory.site_adjacency().tocoo()
    return {(int(i), int(j)) for i, j in zip(adjacency.row, adjacency.col) if i < j}


class SharedFacesTestCase(unittest.TestCase):

    def cube_trajectory(self, n):
        structure, sites = simple_cubic_cubes(n)
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        return structure, trajectory

    def test_cell_two_sites_wide_gives_only_face_neighbours(self):
        _, trajectory = self.cube_trajectory(2)
        pairs = adjacent_pairs(trajectory)
        self.assertEqual(len(pairs), 12)
        for i, j in pairs:
            self.assertEqual(len(trajectory.shared_faces(i, j)), 2)

    def test_cell_three_sites_wide_gives_one_face_per_pair(self):
        _, trajectory = self.cube_trajectory(3)
        pairs = adjacent_pairs(trajectory)
        self.assertEqual(len(pairs), 81)
        self.assertEqual({len(trajectory.shared_faces(i, j)) for i, j in pairs}, {1})

    def test_every_cube_has_six_faces(self):
        _, trajectory = self.cube_trajectory(2)
        for site in trajectory.sites:
            self.assertEqual(len(site.faces), 6)
            for face in site.faces:
                self.assertTrue(set(face.vertex_indices) <= set(site.vertex_indices))

    def test_wrapping_atoms_differently_changes_nothing(self):
        structure, trajectory = self.cube_trajectory(2)
        faces = [site.faces for site in trajectory.sites]
        pairs = adjacent_pairs(trajectory)
        shifted = structure.copy()
        coords = shifted.frac_coords.copy()
        coords[[0, 5]] += [1.0, 0.0, -1.0]
        shifted = Structure(shifted.lattice, shifted.species, coords)
        trajectory.assign_faces(shifted)
        self.assertEqual([site.faces for site in trajectory.sites], faces)
        self.assertEqual(adjacent_pairs(trajectory), pairs)

    def test_shared_faces_are_symmetric_and_empty_for_non_neighbours(self):
        _, trajectory = self.cube_trajectory(3)
        self.assertEqual(trajectory.shared_faces(0, 1), trajectory.shared_faces(1, 0))
        self.assertEqual(len(trajectory.shared_faces(0, 1)), 1)
        self.assertEqual(trajectory.shared_faces(0, 4), ())  # edge neighbours

    def test_differently_triangulated_shared_square_is_one_face(self):
        # Two cubes sharing the square x = 7 A; one corner of the square is
        # displaced out of the plane by less than the tolerance, so each
        # hull splits the square along a different diagonal.
        lattice = Lattice.cubic(20.0)
        corners = [(x, y, z) for x in (5.0, 7.0, 9.0) for y in (5.0, 7.0) for z in (5.0, 7.0)]
        cartesian = np.array(corners)
        cartesian[4] += [0.004, 0.0, 0.0]   # corner (7, 5, 5)
        structure = Structure(lattice, ["O"] * len(corners), cartesian,
                              coords_are_cartesian=True)
        Site._newid = 0
        left = PolyhedralSite(vertex_indices=list(range(8)),
                              reference_center=np.array([6.0, 6.0, 6.0]) / 20.0)
        right = PolyhedralSite(vertex_indices=list(range(4, 12)),
                               reference_center=np.array([8.0, 6.0, 6.0]) / 20.0)
        trajectory = Trajectory(sites=[left, right], atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        self.assertEqual(trajectory.shared_faces(0, 1), (Face((4, 5, 6, 7)),))

    def test_sites_sharing_only_an_edge_are_not_adjacent(self):
        structure, sites = simple_cubic_cubes(3)
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        self.assertFalse(trajectory.site_adjacency()[0, 4])

    def test_matrix_is_indexed_by_site_index(self):
        structure, sites = simple_cubic_cubes(3)
        Site._newid = 10
        renumbered = [PolyhedralSite(vertex_indices=site.vertex_indices,
                                     reference_center=site.reference_center)
                      for site in sites]
        trajectory = Trajectory(sites=renumbered, atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        adjacency = trajectory.site_adjacency()
        self.assertEqual(adjacency.shape, (37, 37))
        self.assertTrue(adjacency[10, 11])
        self.assertEqual(adjacency.nnz, 162)

    def test_assigning_again_replaces_everything(self):
        structure, trajectory = self.cube_trajectory(3)
        # Displacing one atom along x bends the squares through it, so the
        # four pairs of cubes that shared such a square stop sharing a face.
        coords = structure.frac_coords.copy()
        coords[0] += [0.3 / 6.0, 0.0, 0.0]
        bent = Structure(structure.lattice, structure.species, coords)
        self.assertEqual(len(trajectory.shared_faces(0, 18)), 1)
        trajectory.assign_faces(bent)
        self.assertEqual(len(adjacent_pairs(trajectory)), 77)
        self.assertEqual(trajectory.shared_faces(0, 18), ())
        trajectory.assign_faces(structure)
        self.assertEqual(len(adjacent_pairs(trajectory)), 81)
        self.assertEqual(len(trajectory.shared_faces(0, 18)), 1)

    def test_adjacency_copies_are_independent(self):
        _, trajectory = self.cube_trajectory(3)
        first = trajectory.site_adjacency()
        first.data[:] = False
        self.assertTrue(trajectory.site_adjacency()[0, 1])

    def test_faces_survive_reset(self):
        _, trajectory = self.cube_trajectory(2)
        trajectory.reset()
        self.assertEqual(len(trajectory.sites[0].faces), 6)
        self.assertEqual(len(adjacent_pairs(trajectory)), 12)

    def test_before_assigning_faces_raises(self):
        _, sites = simple_cubic_cubes(2)
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        self.assertIsNone(trajectory.sites[0].faces)
        with self.assertRaisesRegex(ValueError, "assign_faces"):
            trajectory.site_adjacency()
        with self.assertRaisesRegex(ValueError, "assign_faces"):
            trajectory.shared_faces(0, 1)

    def test_other_site_types_raise(self):
        trajectory = Trajectory(
            sites=[SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)],
            atoms=[Atom(index=0)])
        structure = Structure(Lattice.cubic(5.0), ["Li"], [[0.5, 0.5, 0.5]])
        for call in (lambda: trajectory.assign_faces(structure),
                     trajectory.site_adjacency,
                     lambda: trajectory.shared_faces(0, 0)):
            with self.assertRaisesRegex(TypeError, "only available for polyhedral sites"):
                call()


class TransitionsAcrossFacesTestCase(unittest.TestCase):
    """Joining transition events with the adjacency matrix."""

    def setUp(self):
        Site._newid = 0
        # Two tetrahedra sharing the face (v0, v1, v2) in the plane x = 5 A;
        # centres at x = 4.6 and 5.4 A.
        self.vertices = [(5.0, 5.0, 6.0), (5.0, 4.134, 4.5), (5.0, 5.866, 4.5),
                         (3.4, 5.0, 5.0), (6.6, 5.0, 5.0)]
        self.lattice = Lattice.cubic(10.0)
        self.trajectory = Trajectory(
            sites=[PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                   PolyhedralSite(vertex_indices=[0, 1, 2, 4])],
            atoms=[Atom(index=5)])
        self.trajectory.assign_faces(self.frame((4.6, 5.0, 5.0)))

    def frame(self, li):
        return Structure(self.lattice, ["S"] * 5 + ["Li"],
                         [*self.vertices, li], coords_are_cartesian=True)

    def between(self):
        events = self.trajectory.transition_events()
        adjacency = self.trajectory.site_adjacency()
        return adjacency[events.from_site[:, None], events.to_site[:, None]].toarray().ravel()

    def test_face_sharing_tetrahedra_share_one_face(self):
        self.assertEqual(self.trajectory.shared_faces(0, 1), (Face((0, 1, 2)),))

    def test_a_transition_between_them_is_between_adjacent_sites(self):
        self.trajectory.trajectory_from_structures(
            [self.frame((4.6, 5.0, 5.0)), self.frame((5.4, 5.0, 5.0))])
        np.testing.assert_array_equal(self.between(), [True])

    def test_no_transitions_gives_an_empty_boolean_array(self):
        self.trajectory.trajectory_from_structures(
            [self.frame((4.6, 5.0, 5.0)), self.frame((4.6, 5.0, 5.0))])
        between = self.between()
        self.assertIsInstance(between, np.ndarray)
        self.assertEqual(len(between), 0)


def interstitial_structure(supercell=(2, 2, 2), a=4.0):
    """An fcc O lattice with Na at every octahedral and Li at every tetrahedral hole."""
    o = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    na = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    li = [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
    unit = Structure(Lattice.cubic(a), ["O"] * 4 + ["Na"] * 4 + ["Li"] * 8, o + na + li)
    return unit * supercell


class BuilderFacesTestCase(unittest.TestCase):

    def build(self):
        structure = interstitial_structure()
        trajectory = (TrajectoryBuilder()
                      .with_structure(structure)
                      .with_reference_structure(structure)
                      .with_mobile_species("Li")
                      .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                             cutoff=2.0, n_vertices=4, label="tet")
                      .with_polyhedral_sites(centre_species="Na", vertex_species="O",
                                             cutoff=2.2, n_vertices=6, label="oct")
                      .build())
        return structure, trajectory

    def test_builder_assigns_faces_to_every_site(self):
        _, trajectory = self.build()
        for site in trajectory.sites:
            self.assertEqual(len(site.faces), 4 if site.label == "tet" else 8)
            for face in site.faces:
                self.assertTrue(set(face.vertex_indices) <= set(site.vertex_indices))

    def test_tetrahedra_share_faces_with_octahedra_only(self):
        _, trajectory = self.build()
        adjacency = trajectory.site_adjacency().tocoo()
        labels = {site.index: site.label for site in trajectory.sites}
        pairs = {(int(i), int(j)) for i, j in zip(adjacency.row, adjacency.col) if i < j}
        self.assertEqual({frozenset((labels[i], labels[j])) for i, j in pairs},
                         {frozenset(("tet", "oct"))})
        self.assertEqual(len(pairs), 64 * 4)

    def test_builder_faces_match_faces_from_the_reference(self):
        structure, trajectory = self.build()
        adjacency = trajectory.site_adjacency().toarray()
        faces = [site.faces for site in trajectory.sites]
        trajectory.assign_faces(structure)
        np.testing.assert_array_equal(trajectory.site_adjacency().toarray(), adjacency)
        self.assertEqual([site.faces for site in trajectory.sites], faces)


class BuilderReferenceGeometryFacesTestCase(unittest.TestCase):

    def build_cubes(self):
        unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])
        reference = unit * (3, 3, 3)
        target = reference.copy()
        target.translate_sites(list(range(len(target))), [0.13, 0.07, -0.21],
                               frac_coords=True, to_unit_cell=True)
        oxygen = next(i for i, site in enumerate(target) if site.species_string == "O")
        target.translate_sites([oxygen], [0.36, 0.0, 0.0],
                               frac_coords=False, to_unit_cell=True)
        target = Structure(Lattice.from_parameters(9.0 * 1.02, 9.0, 9.0, 90, 90, 90),
            target.species, target.frac_coords)
        trajectory = (TrajectoryBuilder()
                      .with_structure(target)
                      .with_reference_structure(reference)
                      .with_mobile_species("Li")
                      .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                             cutoff=2.8, n_vertices=8)
                      .build())
        return target, trajectory

    def test_builder_assigns_square_faces_to_every_cube(self):
        _, trajectory = self.build_cubes()
        for site in trajectory.sites:
            self.assertEqual([len(face.vertex_indices) for face in site.faces], [4] * 6)

    def test_builder_faces_come_from_the_reference_geometry(self):
        target, trajectory = self.build_cubes()
        self.assertEqual(trajectory.site_adjacency().nnz // 2, 81)
        trajectory.assign_faces(target)
        self.assertLess(trajectory.site_adjacency().nnz // 2, 81)


if __name__ == "__main__":
    unittest.main()
