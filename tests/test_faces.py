import dataclasses
import itertools
import unittest
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

import site_analysis.polyhedral_site_collection as collection_module
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

    def test_indices_are_sorted_on_construction(self):
        self.assertEqual(Face((6, 4, 5, 7)).vertex_indices, (4, 5, 6, 7))
        self.assertEqual(Face((6, 4, 5, 7)), Face((4, 5, 6, 7)))

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
        # The displaced corner's three squares split into triangles.
        self.assertEqual(face_sizes("cube", displacement=[0.1, 0.1, 0.1]),
                         [3, 3, 3, 3, 3, 3, 4, 4, 4])

    def test_vertex_indices_are_atom_indices_in_ascending_order(self):
        cartesian = regular_vertices("tetrahedron")
        faces = _faces_and_keys([40, 7, 12, 3], cartesian / 10.0,
                                np.array([0.5, 0.5, 0.5]), np.eye(3) * 10.0)
        self.assertEqual([face.vertex_indices for face, _ in faces],
                         [(3, 7, 12), (3, 7, 40), (3, 12, 40), (7, 12, 40)])

    def test_degenerate_vertices_raise(self):
        flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float) + 5.0
        with self.assertRaises(ValueError):
            _faces_and_keys([0, 1, 2, 3], flat / 10.0, None, np.eye(3) * 10.0)


def square_pyramid_faces(apex_z, raised):
    """Faces of a square pyramid whose base corner 0 is raised by `raised` Angstrom."""
    base = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]], float)
    base[0, 2] += raised
    cartesian = np.vstack([base, [0.0, 0.0, apex_z]]) + 5.0
    faces = _faces_and_keys(list(range(5)), cartesian / 10.0,
                            np.array([0.5, 0.5, 0.5 + apex_z / 30.0]), np.eye(3) * 10.0)
    return [face.vertex_indices for face, _ in faces]


class MergeRuleTestCase(unittest.TestCase):
    """Facets merge by the best-fit plane of their combined vertices."""

    def test_raised_corner_within_tolerance_merges_on_both_sides(self):
        # Raising one corner of a square by h gives best-fit residuals of h/4.
        for raised in (0.015, 0.03):
            with self.subTest(raised=raised):
                self.assertIn((0, 1, 2, 3), square_pyramid_faces(1.2, raised))
                self.assertIn((0, 1, 2, 3), square_pyramid_faces(-1.2, raised))

    def test_raised_corner_beyond_tolerance_merges_on_neither_side(self):
        for apex_z in (1.2, -1.2):
            with self.subTest(apex_z=apex_z):
                faces = square_pyramid_faces(apex_z, 0.05)
                self.assertNotIn((0, 1, 2, 3), faces)
                self.assertEqual(len(faces), 6)

    def test_every_merged_face_fits_its_plane_within_the_tolerance(self):
        rng = np.random.default_rng(0)
        prism = regular_vertices("hexagonal prism")
        for _ in range(50):
            cartesian = prism + rng.normal(scale=0.004, size=prism.shape)
            faces = _faces_and_keys(list(range(12)), cartesian / 10.0,
                                    np.array([0.5, 0.5, 0.5]), np.eye(3) * 10.0)
            for face, _ in faces:
                points = cartesian[list(face.vertex_indices)]
                centred = points - points.mean(axis=0)
                normal = np.linalg.svd(centred)[2][-1]
                self.assertLess(np.abs(centred @ normal).max(), 0.01)

    def test_vertex_on_a_face_raises(self):
        tetrahedron = regular_vertices("tetrahedron")
        on_face = tetrahedron[:3].mean(axis=0)
        cartesian = np.vstack([tetrahedron, on_face])
        with self.assertRaisesRegex(ValueError, "corner"):
            _faces_and_keys(list(range(5)), cartesian / 10.0, None, np.eye(3) * 10.0)

    def test_vertex_pushed_out_of_a_face_raises(self):
        tetrahedron = regular_vertices("tetrahedron")
        centroid = tetrahedron[:3].mean(axis=0)
        outward = centroid - tetrahedron.mean(axis=0)
        pushed = centroid + 0.002 * outward / np.linalg.norm(outward)
        cartesian = np.vstack([tetrahedron, pushed])
        with self.assertRaisesRegex(ValueError, "corner"):
            _faces_and_keys(list(range(5)), cartesian / 10.0, None, np.eye(3) * 10.0)

    def test_vertex_pushed_out_of_an_edge_raises(self):
        tetrahedron = regular_vertices("tetrahedron")
        midpoint = tetrahedron[:2].mean(axis=0)
        outward = midpoint - tetrahedron.mean(axis=0)
        pushed = midpoint + 0.001 * outward / np.linalg.norm(outward)
        cartesian = np.vstack([tetrahedron, pushed])
        with self.assertRaisesRegex(ValueError, "corner"):
            _faces_and_keys(list(range(5)), cartesian / 10.0, None, np.eye(3) * 10.0)

    def test_vertex_inside_raises(self):
        cartesian = np.vstack([regular_vertices("octahedron"), [5.0, 5.0, 5.0]])
        with self.assertRaisesRegex(ValueError, "corner"):
            _faces_and_keys(list(range(7)), cartesian / 10.0, None, np.eye(3) * 10.0)

    def test_polyhedron_flatter_than_the_tolerance_raises(self):
        flat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                         [0.5, 0.5, 0.005]], float) + 5.0
        with self.assertRaisesRegex(ValueError, "four faces"):
            _faces_and_keys(list(range(5)), flat / 10.0, None, np.eye(3) * 10.0)


def stacked_hexagonal_prisms(offsets):
    """Two hexagonal prisms stacked on z, sharing a hexagon with the given z offsets (Angstrom)."""
    hexagon = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.arange(6) * np.pi / 3]) * 1.5
    middle = hexagon.copy()
    middle[:, 2] += offsets
    cartesian = np.vstack([hexagon + [0, 0, -1.5], middle, hexagon + [0, 0, 1.5]]) + 10.0
    structure = Structure(Lattice.cubic(20.0), ["O"] * 18, cartesian,
                          coords_are_cartesian=True)
    Site.reset_index(0)
    below = PolyhedralSite(vertex_indices=list(range(12)),
                           reference_center=np.array([10.0, 10.0, 9.25]) / 20.0)
    above = PolyhedralSite(vertex_indices=list(range(6, 18)),
                           reference_center=np.array([10.0, 10.0, 10.75]) / 20.0)
    trajectory = Trajectory(sites=[below, above], atoms=[Atom(index=0)])
    trajectory.assign_faces(structure)
    return trajectory


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


def bend_atom_0(structure):
    """A copy of a simple cubic grid structure with atom 0 moved 0.3 Angstrom along x."""
    coords = structure.frac_coords.copy()
    coords[0] += [0.3 / structure.lattice.a, 0.0, 0.0]
    return Structure(structure.lattice, structure.species, coords)


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
        # 27 cubes with six face neighbours each: 27 * 6 / 2 = 81 pairs.
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
        Site.reset_index(0)
        left = PolyhedralSite(vertex_indices=list(range(8)),
                              reference_center=np.array([6.0, 6.0, 6.0]) / 20.0)
        right = PolyhedralSite(vertex_indices=list(range(4, 12)),
                               reference_center=np.array([8.0, 6.0, 6.0]) / 20.0)
        trajectory = Trajectory(sites=[left, right], atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        self.assertEqual(trajectory.shared_faces(0, 1), (Face((4, 5, 6, 7)),))

    def test_sites_agree_on_a_slightly_irregular_shared_hexagon(self):
        trajectory = stacked_hexagonal_prisms(
            [-0.0029, 0.0067, -0.0014, -0.0001, -0.0131, 0.0093])
        self.assertEqual(trajectory.shared_faces(0, 1), (Face((6, 7, 8, 9, 10, 11)),))

    def test_matrix_is_indexed_by_site_index(self):
        structure, sites = simple_cubic_cubes(3)
        Site.reset_index(10)
        renumbered = [PolyhedralSite(vertex_indices=site.vertex_indices,
                                     reference_center=site.reference_center)
                      for site in sites]
        trajectory = Trajectory(sites=renumbered, atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        adjacency = trajectory.site_adjacency()
        self.assertEqual(adjacency.shape, (37, 37))
        self.assertTrue(adjacency[10, 11])
        # 81 face-sharing pairs, each stored as (i, j) and (j, i).
        self.assertEqual(adjacency.nnz, 162)

    def test_squares_bent_beyond_the_tolerance_are_not_shared(self):
        structure, sites = simple_cubic_cubes(3)
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        # Displacing one atom along x bends the squares through it, so the
        # four pairs of cubes that would share such a square share no face.
        trajectory.assign_faces(bend_atom_0(structure))
        self.assertEqual(len(adjacent_pairs(trajectory)), 81 - 4)
        self.assertEqual(trajectory.shared_faces(0, 18), ())

    def test_assigning_again_replaces_everything(self):
        structure, trajectory = self.cube_trajectory(3)
        faces = [site.faces for site in trajectory.sites]
        self.assertEqual(len(trajectory.shared_faces(0, 18)), 1)
        trajectory.assign_faces(bend_atom_0(structure))
        self.assertEqual(trajectory.shared_faces(0, 18), ())
        self.assertNotEqual([site.faces for site in trajectory.sites], faces)
        trajectory.assign_faces(structure)
        # 27 cubes with six face neighbours each: 27 * 6 / 2 = 81 pairs.
        self.assertEqual(len(adjacent_pairs(trajectory)), 81)
        self.assertEqual(len(trajectory.shared_faces(0, 18)), 1)
        self.assertEqual([site.faces for site in trajectory.sites], faces)

    def test_unknown_site_index_raises(self):
        _, trajectory = self.cube_trajectory(3)
        for i, j in [(0, 999), (999, 0)]:
            with self.subTest(i=i, j=j):
                with self.assertRaisesRegex(ValueError, "No site with index"):
                    trajectory.shared_faces(i, j)

    def test_failed_assignment_leaves_earlier_faces_and_names_the_site(self):
        structure, trajectory = self.cube_trajectory(3)
        faces = [site.faces for site in trajectory.sites]
        adjacency = trajectory.site_adjacency().toarray()
        coords = structure.frac_coords.copy()
        # Atom 26 is a vertex of eight sites, the first of which is site 13;
        # at the centre of site 13 it is not a corner of that site.
        coords[26] = [0.5, 0.5, 0.5]
        moved = Structure(structure.lattice, structure.species, coords)
        with self.assertRaisesRegex(ValueError, "Site 13: "):
            trajectory.assign_faces(moved)
        self.assertEqual([site.faces for site in trajectory.sites], faces)
        np.testing.assert_array_equal(trajectory.site_adjacency().toarray(), adjacency)

    def test_no_shared_faces_gives_an_empty_matrix(self):
        Site.reset_index()
        lattice = Lattice.cubic(20.0)
        tetrahedron = regular_vertices("tetrahedron")
        cartesian = np.vstack([tetrahedron, tetrahedron + 5.0])
        structure = Structure(lattice, ["O"] * 8, cartesian, coords_are_cartesian=True)
        trajectory = Trajectory(sites=[PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                                       PolyhedralSite(vertex_indices=[4, 5, 6, 7])],
                                atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        self.assertEqual(trajectory.site_adjacency().nnz, 0)

    def test_sites_without_reference_centres_share_faces_across_the_boundary(self):
        structure, sites = simple_cubic_cubes(3)
        for site in sites:
            site.reference_center = None
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)])
        trajectory.assign_faces(structure)
        # 27 cubes with six face neighbours each: 27 * 6 / 2 = 81 pairs.
        self.assertEqual(len(adjacent_pairs(trajectory)), 81)

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


class PolyhedralSiteSerialisationTestCase(unittest.TestCase):

    def test_as_dict_does_not_include_faces(self):
        structure = interstitial_structure()
        trajectory = (TrajectoryBuilder()
                      .with_structure(structure)
                      .with_reference_structure(structure)
                      .with_mobile_species("Li")
                      .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                             cutoff=2.0, n_vertices=4, label="tet")
                      .build())
        trajectory.assign_faces(structure)
        site = trajectory.sites[0]
        self.assertIsNotNone(site.faces)
        self.assertNotIn("faces", site.as_dict())


class LazyFacesTestCase(unittest.TestCase):

    def build(self, extra_sites=None):
        structure = interstitial_structure()
        builder = (TrajectoryBuilder()
                   .with_structure(structure)
                   .with_reference_structure(structure)
                   .with_mobile_species("Li")
                   .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                          cutoff=2.0, n_vertices=4, label="tet"))
        if extra_sites is not None:
            builder = builder.with_existing_sites(extra_sites)
        return builder.build()

    def test_building_computes_no_faces(self):
        with patch.object(collection_module, "_faces_and_keys",
                          wraps=collection_module._faces_and_keys) as faces_and_keys:
            trajectory = self.build()
            self.assertEqual(faces_and_keys.call_count, 0)
            trajectory.site_adjacency()
            self.assertEqual(faces_and_keys.call_count, len(trajectory.sites))
            trajectory.site_adjacency()
            trajectory.sites[0].faces
            self.assertEqual(faces_and_keys.call_count, len(trajectory.sites))

    def test_reading_a_site_s_faces_computes_them_for_all_sites(self):
        trajectory = self.build()
        self.assertEqual(len(trajectory.sites[0].faces), 4)
        self.assertTrue(all(site._faces is not None for site in trajectory.sites))

    def test_mixing_with_existing_sites_sets_no_face_source(self):
        Site.reset_index(1000)
        extra = [PolyhedralSite(vertex_indices=[0, 1, 2, 3])]
        trajectory = self.build(extra_sites=extra)
        self.assertIsNone(trajectory.sites[0].faces)
        with self.assertRaisesRegex(ValueError, "assign_faces"):
            trajectory.site_adjacency()

    def test_assign_faces_replaces_a_pending_source(self):
        structure = interstitial_structure()
        trajectory = self.build()
        trajectory.assign_faces(structure)
        self.assertIsNone(trajectory.site_collection._pending_vertex_coords)

    def test_unknown_site_index_does_not_compute_pending_faces(self):
        trajectory = self.build()
        with patch.object(collection_module, "_faces_and_keys") as faces_and_keys:
            with self.assertRaisesRegex(ValueError, "No site with index"):
                trajectory.shared_faces(0, 999)
            self.assertEqual(faces_and_keys.call_count, 0)

    def test_failed_computation_stays_pending_and_raises_again(self):
        structure = interstitial_structure()
        trajectory = self.build()
        collection = trajectory.site_collection
        vertex_coords = [structure.frac_coords[site.vertex_indices]
                         for site in collection.sites]
        failing = collection.sites[1]
        vertex_coords[1] = vertex_coords[1].copy()
        vertex_coords[1][:, 2] = 0.5
        collection._set_face_source(vertex_coords, structure.lattice.matrix)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, f"Site {failing.index}"):
                trajectory.site_adjacency()
            self.assertIsNotNone(collection._pending_vertex_coords)
            self.assertTrue(all(site._faces is None for site in collection.sites))


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
        # 27 cubes with six face neighbours each: 27 * 6 / 2 = 81 pairs.
        self.assertEqual(trajectory.site_adjacency().nnz // 2, 81)
        trajectory.assign_faces(target)
        self.assertLess(trajectory.site_adjacency().nnz // 2, 81)

    def test_target_atoms_in_a_different_order_from_the_reference(self):
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
        for site in trajectory.sites:
            self.assertEqual([len(face.vertex_indices) for face in site.faces], [4] * 6)
            for face in site.faces:
                self.assertTrue(set(face.vertex_indices) <= set(site.vertex_indices))
        # 27 cubes with six face neighbours each: 27 * 6 / 2 = 81 pairs.
        self.assertEqual(len(adjacent_pairs(trajectory)), 81)


if __name__ == "__main__":
    unittest.main()
