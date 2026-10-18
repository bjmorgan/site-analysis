import dataclasses
import itertools
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

from site_analysis import Face, TrajectoryBuilder
from site_analysis.faces import _find_shared_faces
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.site import Site


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


def interstitial_structure(supercell=(2, 2, 2), a=4.0):
    """An fcc O lattice with Na at every octahedral and Li at every tetrahedral hole."""
    o = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
    na = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    li = [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
    unit = Structure(Lattice.cubic(a), ["O"] * 4 + ["Na"] * 4 + ["Li"] * 8, o + na + li)
    return unit * supercell


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
    """A trajectory of cube sites whose target atoms are in the reverse order of the reference."""
    unit = Structure(Lattice.cubic(3.0), ["O", "Li"],
                     [[0, 0, 0], [0.5, 0.5, 0.5]])
    reference = unit * (3, 3, 3)
    target = Structure(reference.lattice, reference.species[::-1],
                       reference.frac_coords[::-1])
    return (TrajectoryBuilder()
            .with_structure(target)
            .with_reference_structure(reference)
            .with_mobile_species("Li")
            .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                   cutoff=2.8, n_vertices=8)
            .build())


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

    def test_faces_are_sorted_by_sites_then_vertices(self):
        structure, sites = simple_cubic_cubes(2)
        faces = shared_faces_of(structure, sites)
        self.assertEqual(
            list(faces),
            sorted(faces, key=lambda face: (face.site_indices, face.vertex_indices)))

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

    def test_nested_site_raises(self):
        Site.reset_index(0)
        centre = np.array([0.5, 0.5, 0.5])
        sites = [PolyhedralSite(vertex_indices=list(range(8)), reference_center=centre),
                 PolyhedralSite(vertex_indices=[0, 3, 5, 6], reference_center=centre)]
        with self.assertRaisesRegex(ValueError, "Site 1 lies inside site 0"):
            shared_faces_of(self.cube_structure(), sites)

    def test_coincident_sites_in_a_narrow_cell_without_reference_centres_raise(self):
        structure, sites = simple_cubic_cubes(2)
        for site in sites:
            site.reference_center = None
        with self.assertRaisesRegex(ValueError, "coincide") as context:
            shared_faces_of(structure, sites)
        self.assertIn("neither site has a reference centre", str(context.exception))

    def test_vertex_index_beyond_the_atoms_raises(self):
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 5])]
        with self.assertRaisesRegex(ValueError, "Site 0 has vertex index 5, but the structure has 5 atoms"):
            shared_faces_of(tetrahedra_structure(), sites)

    def test_non_finite_vertex_coordinates_raise(self):
        structure = tetrahedra_structure()
        coords = structure.frac_coords.copy()
        coords[2, 1] = np.nan
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3])]
        with self.assertRaisesRegex(ValueError, "Site 0 has non-finite vertex coordinates"):
            _find_shared_faces(sites, coords, structure.lattice.matrix)

    def test_negative_site_index_raises(self):
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3])]
        sites[0].index = -1
        with self.assertRaisesRegex(ValueError, "Site -1 has a negative index"):
            shared_faces_of(tetrahedra_structure(), sites)


if __name__ == "__main__":
    unittest.main()
