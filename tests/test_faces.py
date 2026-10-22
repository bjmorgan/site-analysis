import dataclasses
import itertools
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from site_analysis import Face, TrajectoryBuilder
from site_analysis.atom import Atom
from site_analysis.faces import _find_shared_faces
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.site import Site
from site_analysis.spherical_site import SphericalSite
from site_analysis.trajectory import Trajectory
from tests.test_helpers import interstitial_structure, interstitial_structure_outside_the_cell


def simple_cubic_cubes(n):
    """Atoms on an n x n x n simple cubic grid (2 Angstrom spacing) and the cube sites between them."""
    points = list(itertools.product(range(n), repeat=3))
    index = {point: k for k, point in enumerate(points)}
    structure = Structure(Lattice.cubic(2.0 * n), ["O"] * len(points),
                          np.array(points, float) / n)
    Site.reset_index(0)
    sites = [PolyhedralSite(
        vertex_indices=[index[tuple((np.array(corner) + offset) % n)]
                        for offset in itertools.product((0, 1), repeat=3)],
        reference_center=(np.array(corner) + 0.5) / n)
        for corner in points]
    return structure, sites


def adjacent_pairs(trajectory):
    adjacency = trajectory.site_adjacency().tocoo()
    return {(int(i), int(j)) for i, j in zip(adjacency.row, adjacency.col) if i < j}


def build_cubes(supercell=(3, 3, 3)):
    """A trajectory of cube sites, built from a reference with a strained, shifted target.

    Returns:
        ``(target, trajectory)``.
    """
    n = supercell[0]
    unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])
    reference = unit * supercell
    target = reference.copy()
    target.translate_sites(list(range(len(target))), [0.13, 0.07, -0.21],
                           frac_coords=True, to_unit_cell=True)
    oxygen = next(i for i, site in enumerate(target) if site.species_string == "O")
    target.translate_sites([oxygen], [0.36, 0.0, 0.0],
                           frac_coords=False, to_unit_cell=True)
    target = Structure(
        Lattice.from_parameters(3.0 * n * 1.02, 3.0 * n, 3.0 * n, 90, 90, 90),
        target.species, target.frac_coords)
    trajectory = (TrajectoryBuilder()
                  .with_structure(target)
                  .with_reference_structure(reference)
                  .with_mobile_species("Li")
                  .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                         cutoff=2.8, n_vertices=8)
                  .build())
    return target, trajectory


def reversed_order_cubes():
    """The target structure and trajectory of cube sites whose target atoms are in the reverse order of the reference."""
    unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])
    reference = unit * (3, 3, 3)
    target = Structure(reference.lattice, reference.species[::-1],
                       reference.frac_coords[::-1])
    trajectory = (TrajectoryBuilder()
                  .with_structure(target)
                  .with_reference_structure(reference)
                  .with_mobile_species("Li")
                  .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                         cutoff=2.8, n_vertices=8)
                  .build())
    return target, trajectory


def shared_faces_of(structure, sites):
    return _find_shared_faces(sites, structure.frac_coords, structure.lattice.matrix)


def pairs(faces):
    return {face.site_indices for face in faces}


def displaced(structure, sigma, seed=0):
    """A copy of a structure with every atom displaced by Gaussian noise (sigma in Angstrom)."""
    rng = np.random.default_rng(seed)
    cartesian = structure.cart_coords + rng.normal(scale=sigma, size=(len(structure), 3))
    return Structure(structure.lattice, structure.species, cartesian,
                     coords_are_cartesian=True)


def tetrahedra_structure():
    """Five atoms in a 10 Angstrom cubic cell: three on a plane, and one on each side of it."""
    vertices = [(5.0, 5.0, 6.0), (5.0, 4.134, 4.5), (5.0, 5.866, 4.5),
                (3.4, 5.0, 5.0), (6.6, 5.0, 5.0)]
    return Structure(Lattice.cubic(10.0), ["S"] * 5, vertices, coords_are_cartesian=True)


def tetrahedra_faces(*vertex_indices):
    """Shared faces of tetrahedra on the atoms of `tetrahedra_structure`."""
    Site.reset_index(0)
    sites = [PolyhedralSite(vertex_indices=list(v)) for v in vertex_indices]
    return shared_faces_of(tetrahedra_structure(), sites)


class FaceTestCase(unittest.TestCase):

    def test_fields_are_sorted_on_construction(self):
        face = Face((5, 2), (7, 3, 4))
        self.assertEqual(face.site_indices, (2, 5))
        self.assertEqual(face.vertex_indices, (3, 4, 7))

    def test_numpy_integer_fields_are_stored_as_ints(self):
        face = Face((np.int64(5), np.int64(2)), (np.int64(7), np.int64(3), np.int64(4)))
        self.assertEqual(face, Face((2, 5), (3, 4, 7)))
        self.assertTrue(all(type(i) is int for i in face.site_indices + face.vertex_indices))

    def test_non_integer_indices_raise(self):
        cases = [
            ((1.0, 2), (0, 1, 2), "^site index must be an integer, got float 1.0$"),
            ((1, 2), (0, 1, 2.0), "^vertex index must be an integer, got float 2.0$"),
            ((True, 2), (0, 1, 2), "^site index must be an integer, got bool True$"),
            ((1, 2), (0, 1, True), "^vertex index must be an integer, got bool True$"),
        ]
        for site_indices, vertex_indices, message in cases:
            with self.subTest(site_indices=site_indices, vertex_indices=vertex_indices):
                with self.assertRaisesRegex(TypeError, message):
                    Face(site_indices, vertex_indices)

    def test_not_two_site_indices_raises(self):
        for site_indices, shown in [((1,), r"\(1,\)"), ((1, 2, 3), r"\(1, 2, 3\)")]:
            with self.subTest(site_indices=site_indices):
                with self.assertRaisesRegex(
                        ValueError, f"^A face is shared by two sites, got site_indices {shown}$"):
                    Face(site_indices, (0, 1, 2))

    def test_faces_sort_by_site_indices_then_vertex_indices(self):
        faces = [Face((1, 2), (0, 1, 2)), Face((0, 3), (4, 5, 6)),
                 Face((0, 3), (1, 2, 3)), Face((0, 1), (7, 8, 9))]
        self.assertEqual(sorted(faces),
                         [Face((0, 1), (7, 8, 9)), Face((0, 3), (1, 2, 3)),
                          Face((0, 3), (4, 5, 6)), Face((1, 2), (0, 1, 2))])

    def test_faces_compare_and_hash_by_both_fields(self):
        self.assertEqual(Face((0, 1), (2, 3, 4)), Face((1, 0), (4, 3, 2)))
        self.assertEqual(len({Face((0, 1), (2, 3, 4)), Face((1, 0), (4, 3, 2))}), 1)
        self.assertNotEqual(Face((0, 1), (2, 3, 4)), Face((0, 2), (2, 3, 4)))

    def test_face_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Face((0, 1), (2, 3, 4)).site_indices = (0, 2)


class FindSharedFacesTestCase(unittest.TestCase):

    def test_face_sharing_tetrahedra_share_one_triangle(self):
        self.assertEqual(tetrahedra_faces([0, 1, 2, 3], [0, 1, 2, 4]),
                         (Face((0, 1), (0, 1, 2)),))

    def test_sites_sharing_an_edge_or_a_vertex_share_no_face(self):
        structure = Structure(Lattice.cubic(10.0), ["S"] * 8,
                              [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0), (5.0, 6.0, 5.0),
                               (5.0, 5.0, 6.0), (4.0, 5.0, 5.0), (5.0, 4.0, 5.0),
                               (5.0, 5.0, 4.0), (6.0, 6.0, 6.0)],
                              coords_are_cartesian=True)
        for second in ([0, 1, 4, 5], [0, 4, 5, 6]):
            with self.subTest(shared=len({0, 1, 2, 3} & set(second))):
                Site.reset_index(0)
                sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                         PolyhedralSite(vertex_indices=second)]
                self.assertEqual(shared_faces_of(structure, sites), ())

    def test_cell_two_sites_wide_gives_two_faces_per_face_neighbour(self):
        structure, sites = simple_cubic_cubes(2)
        faces = shared_faces_of(structure, sites)
        self.assertEqual(len(faces), 24)
        self.assertEqual(len(pairs(faces)), 12)
        self.assertEqual({len(face.vertex_indices) for face in faces}, {4})
        # 28 pairs of cubes; the 16 edge and corner neighbours share no face.
        all_pairs = set(itertools.combinations(range(8), 2))
        self.assertEqual(len(all_pairs - pairs(faces)), 16)
        for pair in pairs(faces):
            self.assertEqual(sum(face.site_indices == pair for face in faces), 2)

    def test_cell_three_sites_wide_gives_one_face_per_pair(self):
        structure, sites = simple_cubic_cubes(3)
        faces = shared_faces_of(structure, sites)
        self.assertEqual(len(faces), 81)
        self.assertEqual(len(pairs(faces)), 81)
        for face in faces:
            self.assertEqual(len(face.vertex_indices), 4)
            for i in face.site_indices:
                self.assertTrue(set(face.vertex_indices) <= set(sites[i].vertex_indices))

    def test_order_of_the_sites_does_not_change_shared_faces(self):
        structure, sites = simple_cubic_cubes(3)
        self.assertEqual(shared_faces_of(structure, sites[::-1]),
                         shared_faces_of(structure, sites))

    def test_noise_does_not_change_shared_faces(self):
        for n in (2, 3):
            structure, sites = simple_cubic_cubes(n)
            expected = shared_faces_of(structure, sites)
            for sigma in (0.05, 0.1, 0.2):
                with self.subTest(n=n, sigma=sigma):
                    self.assertEqual(
                        shared_faces_of(displaced(structure, sigma), sites), expected)

    def test_non_orthogonal_lattice(self):
        structure, sites = simple_cubic_cubes(3)
        expected = shared_faces_of(structure, sites)
        skewed = Structure(Lattice.from_parameters(6.0, 6.3, 5.8, 80, 85, 95),
                           structure.species, structure.frac_coords)
        self.assertEqual(shared_faces_of(skewed, sites), expected)

    def test_wrapping_does_not_change_shared_faces(self):
        structure, sites = simple_cubic_cubes(3)
        expected = shared_faces_of(structure, sites)
        coords = structure.frac_coords.copy()
        coords[[0, 5]] += [1.0, 0.0, -1.0]
        self.assertEqual(coords[3, 0], 0.0)
        coords[3, 0] = 1.0
        wrapped = Structure(structure.lattice, structure.species, coords)
        self.assertEqual(shared_faces_of(wrapped, sites), expected)

    def test_sites_without_reference_centres_in_a_wide_cell(self):
        structure, sites = simple_cubic_cubes(3)
        for site in sites:
            site.reference_center = None
        self.assertEqual(len(pairs(shared_faces_of(structure, sites))), 81)

    def test_tetrahedron_on_half_a_square_shares_a_triangle(self):
        corners = np.array(list(itertools.product((-1, 1), repeat=3)), float)
        apex = [3.0, 0.0, 0.0]
        structure = Structure(Lattice.cubic(10.0), ["O"] * 9,
                              np.vstack([corners, apex]) + 5.0,
                              coords_are_cartesian=True)
        Site.reset_index(0)
        cube = PolyhedralSite(vertex_indices=list(range(8)),
                              reference_center=np.array([0.5, 0.5, 0.5]))
        # Corners 4, 5 and 6 are three of the four corners with x = +1.
        tetrahedron = PolyhedralSite(vertex_indices=[4, 5, 6, 8],
                                     reference_center=np.array([0.7, 0.5, 0.5]))
        self.assertEqual(shared_faces_of(structure, [cube, tetrahedron]),
                         (Face((0, 1), (4, 5, 6)),))

    def test_reference_centres_outside_the_cell(self):
        rng = np.random.default_rng(0)
        for n in (3, 4):
            structure, sites = simple_cubic_cubes(n)
            expected = shared_faces_of(structure, sites)
            self.assertEqual(len(expected), 3 * n ** 3)
            Site.reset_index(0)
            shifted = [PolyhedralSite(vertex_indices=site.vertex_indices,
                                      reference_center=site.reference_center
                                      + rng.integers(-3, 4, size=3))
                       for site in sites]
            self.assertEqual(shared_faces_of(structure, shifted), expected)


class FindSharedFacesErrorsTestCase(unittest.TestCase):

    @staticmethod
    def cube_structure():
        corners = np.array(list(itertools.product((-1, 1), repeat=3)), float)
        return Structure(Lattice.cubic(10.0), ["O"] * 8, corners + 5.0,
                         coords_are_cartesian=True)

    def test_coincident_sites_raise(self):
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=list(range(8)),
                                reference_center=np.array([0.5, 0.5, 0.5]))
                 for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "Sites 0 and 1 coincide") as context:
            shared_faces_of(self.cube_structure(), sites)
        self.assertNotIn("reference centre", str(context.exception))

    def test_coincident_sites_in_reverse_index_order_are_reported_in_index_order(self):
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=list(range(8)),
                                reference_center=np.array([0.5, 0.5, 0.5]))
                 for _ in range(2)]
        with self.assertRaisesRegex(ValueError, "Sites 0 and 1 coincide"):
            shared_faces_of(self.cube_structure(), sites[::-1])

    def test_nested_site_raises(self):
        Site.reset_index(0)
        centre = np.array([0.5, 0.5, 0.5])
        sites = [PolyhedralSite(vertex_indices=list(range(8)), reference_center=centre),
                 PolyhedralSite(vertex_indices=[0, 3, 5, 6], reference_center=centre)]
        with self.assertRaisesRegex(ValueError, "Site 1 lies inside site 0") as context:
            shared_faces_of(self.cube_structure(), sites)
        self.assertNotIn("reference centre", str(context.exception))

    def test_nested_site_without_reference_centres_raises_with_a_hint(self):
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=list(range(8))),
                 PolyhedralSite(vertex_indices=[0, 3, 5, 6])]
        with self.assertRaisesRegex(ValueError, "Site 1 lies inside site 0") as context:
            shared_faces_of(self.cube_structure(), sites)
        self.assertIn("neither site has a reference centre", str(context.exception))

    def test_coincident_sites_in_a_narrow_cell_without_reference_centres_raise(self):
        structure, sites = simple_cubic_cubes(2)
        for site in sites:
            site.reference_center = None
        with self.assertRaisesRegex(ValueError, "coincide") as context:
            shared_faces_of(structure, sites)
        self.assertIn("neither site has a reference centre", str(context.exception))

    def test_non_finite_vertex_coordinates_raise(self):
        structure = tetrahedra_structure()
        coords = structure.frac_coords.copy()
        coords[2, 1] = np.nan
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3])]
        with self.assertRaisesRegex(ValueError, "Site 0 has non-finite vertex coordinates"):
            _find_shared_faces(sites, coords, structure.lattice.matrix)


class SharedFacesApiTestCase(unittest.TestCase):

    @staticmethod
    def cube_trajectory(n=3):
        structure, sites = simple_cubic_cubes(n)
        return structure, Trajectory(sites=sites, atoms=[Atom(index=0)])

    def test_shared_faces_and_adjacency_agree(self):
        structure, trajectory = self.cube_trajectory()
        trajectory.find_shared_faces(structure)
        self.assertEqual(adjacent_pairs(trajectory),
                         {face.site_indices for face in trajectory.shared_faces()})
        self.assertEqual(len(adjacent_pairs(trajectory)), 81)
        matrix = trajectory.site_adjacency().toarray()
        np.testing.assert_array_equal(matrix, matrix.T)
        self.assertFalse(matrix.diagonal().any())

    def test_sites_sharing_two_faces_are_one_adjacent_pair(self):
        structure, trajectory = self.cube_trajectory(2)
        trajectory.find_shared_faces(structure)
        self.assertEqual(len(adjacent_pairs(trajectory)), 12)
        self.assertEqual(len(trajectory.shared_faces()), 2 * len(adjacent_pairs(trajectory)))
        self.assertTrue(trajectory.site_adjacency().data.all())

    def test_shared_faces_are_sorted(self):
        structure, trajectory = self.cube_trajectory()
        trajectory.find_shared_faces(structure)
        faces = trajectory.shared_faces()
        self.assertEqual(list(faces), sorted(faces))

    def test_adjacency_is_indexed_by_site_index(self):
        Site.reset_index(5)
        first = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        Site.reset_index(8)
        second = PolyhedralSite(vertex_indices=[0, 1, 2, 4])
        trajectory = Trajectory(sites=[first, second], atoms=[Atom(index=0)])
        trajectory.find_shared_faces(tetrahedra_structure())
        matrix = trajectory.site_adjacency().toarray()
        self.assertEqual(matrix.shape, (9, 9))
        expected = np.zeros((9, 9), dtype=bool)
        expected[5, 8] = expected[8, 5] = True
        np.testing.assert_array_equal(matrix, expected)

    def test_no_shared_faces_gives_an_empty_adjacency(self):
        Site.reset_index(0)
        structure = Structure(
            Lattice.cubic(10.0), ["S"] * 7,
            [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (4, 4, 4), (5, 4, 4), (4, 5, 4)],
            coords_are_cartesian=True)
        trajectory = Trajectory(
            sites=[PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                   PolyhedralSite(vertex_indices=[3, 4, 5, 6])],
            atoms=[Atom(index=0)])
        trajectory.find_shared_faces(structure)
        self.assertEqual(trajectory.shared_faces(), ())
        matrix = trajectory.site_adjacency().toarray()
        self.assertEqual(matrix.shape, (2, 2))
        self.assertFalse(matrix.any())

    def test_finding_again_replaces_the_result(self):
        Site.reset_index(0)
        structure = tetrahedra_structure()
        coords = structure.frac_coords.copy()
        trajectory = Trajectory(
            sites=[PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                                  reference_center=coords[[0, 1, 2, 3]].mean(axis=0)),
                   PolyhedralSite(vertex_indices=[0, 1, 2, 4],
                                  reference_center=coords[[0, 1, 2, 4]].mean(axis=0))],
            atoms=[Atom(index=0)])
        trajectory.find_shared_faces(structure)
        self.assertEqual(trajectory.shared_faces(), (Face((0, 1), (0, 1, 2)),))
        # Atom 0 moves half a cell, so each site unwraps it to a different image.
        coords[0, 0] += 0.5
        with self.assertWarnsRegex(UserWarning, "connectivity of the sites has changed"):
            trajectory.find_shared_faces(Structure(structure.lattice, structure.species, coords))
        self.assertEqual(trajectory.shared_faces(), ())

    def test_failed_finding_leaves_the_earlier_result(self):
        structure, trajectory = self.cube_trajectory()
        trajectory.find_shared_faces(structure)
        faces = trajectory.shared_faces()
        coords = structure.frac_coords.copy()
        coords[0] = np.nan
        broken = Structure(structure.lattice, structure.species, coords)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            trajectory.find_shared_faces(broken)
        self.assertEqual(trajectory.shared_faces(), faces)

    def test_adjacency_copies_are_independent(self):
        structure, trajectory = self.cube_trajectory()
        trajectory.find_shared_faces(structure)
        first = trajectory.site_adjacency()
        first.data[:] = False
        self.assertTrue(trajectory.site_adjacency().data.all())
        self.assertGreater(trajectory.site_adjacency().nnz, 0)

    def test_reset_keeps_shared_faces(self):
        structure, trajectory = self.cube_trajectory(2)
        trajectory.find_shared_faces(structure)
        faces = trajectory.shared_faces()
        trajectory.reset()
        self.assertEqual(trajectory.shared_faces(), faces)

    def test_before_finding_raises(self):
        _, trajectory = self.cube_trajectory(2)
        for call in (trajectory.shared_faces, trajectory.site_adjacency):
            with self.subTest(call=call.__name__):
                with self.assertRaisesRegex(RuntimeError, "find_shared_faces"):
                    call()

    def test_other_site_types_raise(self):
        trajectory = Trajectory(
            sites=[SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)],
            atoms=[Atom(index=0)])
        structure = Structure(Lattice.cubic(5.0), ["Li"], [[0.5, 0.5, 0.5]])
        for call in (lambda: trajectory.find_shared_faces(structure),
                     trajectory.shared_faces,
                     trajectory.site_adjacency):
            with self.assertRaisesRegex(TypeError, "not SphericalSiteCollection"):
                call()


class TransitionsAcrossFacesTestCase(unittest.TestCase):
    """Joining transition events with the site adjacency."""

    def setUp(self):
        Site.reset_index(0)
        # Two tetrahedra sharing the face (v0, v1, v2) in the plane x = 5 A;
        # centres at x = 4.6 and 5.4 A.
        self.vertices = [(5.0, 5.0, 6.0), (5.0, 4.134, 4.5), (5.0, 5.866, 4.5),
                         (3.4, 5.0, 5.0), (6.6, 5.0, 5.0)]
        self.lattice = Lattice.cubic(10.0)
        self.trajectory = Trajectory(
            sites=[PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                   PolyhedralSite(vertex_indices=[0, 1, 2, 4])],
            atoms=[Atom(index=5)])
        self.trajectory.find_shared_faces(self.frame((4.6, 5.0, 5.0)))

    def frame(self, li):
        return Structure(self.lattice, ["S"] * 5 + ["Li"],
                         [*self.vertices, li], coords_are_cartesian=True)

    def between(self):
        events = self.trajectory.transition_events()
        adjacency = self.trajectory.site_adjacency()
        return adjacency[events.from_site[:, None], events.to_site[:, None]].toarray().ravel()

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


class FccInterstitialsTestCase(unittest.TestCase):

    def test_tetrahedra_share_triangles_with_octahedra_only(self):
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
        faces = trajectory.shared_faces()
        labels = {site.index: site.label for site in trajectory.sites}
        self.assertEqual(len(faces), 256)
        self.assertTrue(all(len(face.vertex_indices) == 3 for face in faces))
        self.assertTrue(all(sorted(labels[i] for i in face.site_indices) == ["oct", "tet"]
                            for face in faces))


class AutomaticSharedFacesTestCase(unittest.TestCase):

    def test_first_analysis_finds_the_shared_faces_of_its_structure(self):
        structure, sites = simple_cubic_cubes(3)
        expected = shared_faces_of(structure, sites)
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        trajectory.analyse_structure(displaced(structure, 0.1))
        self.assertEqual(trajectory.shared_faces(), expected)

    def test_coincident_sites_raise_from_analysis(self):
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=list(range(8)),
                                reference_center=np.array([0.5, 0.5, 0.5]))
                 for _ in range(2)]
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        with self.assertRaisesRegex(ValueError, "Sites 0 and 1 coincide"):
            trajectory.analyse_structure(FindSharedFacesErrorsTestCase.cube_structure())


class BuilderSharedFacesTestCase(unittest.TestCase):

    def test_reference_with_coordinates_outside_the_cell_and_no_alignment(self):
        structure = interstitial_structure()
        reference = interstitial_structure_outside_the_cell()

        def build(reference_structure):
            return (TrajectoryBuilder()
                    .with_structure(structure)
                    .with_reference_structure(reference_structure)
                    .with_mobile_species("Li")
                    .with_structure_alignment(align=False)
                    .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                           cutoff=2.0, n_vertices=4)
                    .with_polyhedral_sites(centre_species="Na", vertex_species="O",
                                           cutoff=2.2, n_vertices=6)
                    .build())

        expected = build(structure).shared_faces()
        self.assertEqual(len(expected), 256)
        self.assertEqual(build(reference).shared_faces(), expected)

    def test_distorted_target(self):
        target, trajectory = build_cubes()
        from_reference = trajectory.shared_faces()
        self.assertEqual(len(pairs(from_reference)), 81)
        trajectory.find_shared_faces(target)
        self.assertEqual(trajectory.shared_faces(), from_reference)

    def test_target_atoms_in_a_different_order_from_the_reference(self):
        target, trajectory = reversed_order_cubes()
        faces = trajectory.shared_faces()
        self.assertEqual(len(pairs(faces)), 81)
        vertices = {site.index: set(site.vertex_indices) for site in trajectory.sites}
        for face in faces:
            for site_index in face.site_indices:
                self.assertTrue(set(face.vertex_indices) <= vertices[site_index])
        trajectory.find_shared_faces(target)
        self.assertEqual(trajectory.shared_faces(), faces)

    def test_cubes_in_a_cell_two_sites_wide_are_not_duplicates(self):
        _, trajectory = build_cubes((2, 2, 2))
        self.assertEqual(len(trajectory.sites), 8)
        self.assertEqual(len(trajectory.shared_faces()), 24)

    def test_builder_uses_reference_not_target_coordinates(self):
        unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])
        reference = unit * (2, 2, 2)
        target = reference.copy()
        oxygen = next(i for i, site in enumerate(target) if site.species_string == "O")
        target.translate_sites([oxygen], [1.6, 1.6, 0.0],
                               frac_coords=False, to_unit_cell=True)
        trajectory = (TrajectoryBuilder()
                      .with_structure(target)
                      .with_reference_structure(reference)
                      .with_mobile_species("Li")
                      .with_structure_alignment(align=False)
                      .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                             cutoff=2.8, n_vertices=8)
                      .build())
        self.assertEqual(len(trajectory.shared_faces()), 24)
        # The moved atom leaves the target's cubes unable to share their faces.
        with self.assertWarns(UserWarning), \
                self.assertRaisesRegex(ValueError, "the sites may overlap"):
            trajectory.find_shared_faces(target)

    def test_existing_site_on_vertices_of_generated_sites_shares_faces(self):
        unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])
        structure = unit * (3, 3, 3)
        # Leave the cube from (1, 0, 0) to (2, 1, 1) cell units empty for the existing site.
        empty = next(i for i, site in enumerate(structure)
                     if np.allclose(site.frac_coords * 3, [1.5, 0.5, 0.5]))
        structure.remove_sites([empty])
        oxygen = [i for i, site in enumerate(structure) if site.species_string == "O"]
        by_position = {tuple(np.round(structure[i].frac_coords * 3).astype(int) % 3): i
                       for i in oxygen}
        face = [by_position[(1, y, z)] for y in (0, 1) for z in (0, 1)]
        Site.reset_index(1000)
        # The empty cube as an existing site, sharing a square with each of its six neighbours.
        existing = PolyhedralSite(
            vertex_indices=[by_position[(x, y, z)] for x in (1, 2) for y in (0, 1) for z in (0, 1)],
            reference_center=np.array([1.5, 0.5, 0.5]) / 3)
        trajectory = (TrajectoryBuilder()
                      .with_structure(structure)
                      .with_reference_structure(structure)
                      .with_mobile_species("Li")
                      .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                             cutoff=2.8, n_vertices=8)
                      .with_existing_sites([existing])
                      .build())
        existing_faces = [f for f in trajectory.shared_faces() if 1000 in f.site_indices]
        self.assertEqual([len(f.vertex_indices) for f in existing_faces], [4] * 6)
        self.assertIn(tuple(sorted(face)), {f.vertex_indices for f in existing_faces})

    def test_mixing_with_existing_sites_finds_none(self):
        unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])
        structure = unit * (3, 3, 3)
        lithium = [i for i, site in enumerate(structure) if site.species_string == "Li"]
        Site.reset_index(1000)
        existing = PolyhedralSite(vertex_indices=lithium[:4])
        trajectory = (TrajectoryBuilder()
                      .with_structure(structure)
                      .with_reference_structure(structure)
                      .with_mobile_species("Li")
                      .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                             cutoff=2.8, n_vertices=8)
                      .with_existing_sites([existing])
                      .build())
        with self.assertRaisesRegex(RuntimeError, "find_shared_faces"):
            trajectory.shared_faces()


if __name__ == "__main__":
    unittest.main()
