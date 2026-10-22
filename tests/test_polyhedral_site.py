import json
import unittest
import warnings

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.polyhedron_surface import _containment_planes, _surface_triangles
from site_analysis._compat import HAS_NUMBA
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import patch, Mock, PropertyMock
import numpy as np
from collections import Counter
from pymatgen.core import Structure, Lattice
from tests.test_helpers import recorded_future_warnings

# Containment with numba where it is installed, and with numpy.
BACKENDS = [True, False] if HAS_NUMBA else [False]

# The reference centre of the sites in these tests that lie around the
# middle of the cell, such as _TETRAHEDRON.
CENTRE = np.array([0.5, 0.5, 0.5])


class PolyhedralSiteTestCase(unittest.TestCase):

    INVALID_REFERENCE_CENTRES = {
        "NaN": [np.nan, 0.5, 0.6],
        "infinite": [np.inf, 0.5, 0.6],
        "two numbers": [0.4, 0.5],
        "text": "abc",
        "ragged": [[0.1, 0.2], [0.3]],
        "complex": [1 + 1j, 0.0, 0.0],
        "integer too large for a float": [10 ** 400, 0.0, 0.0],
    }

    def setUp(self):
        Site._newid = 1
        vertex_indices = [0, 1, 3, 4]
        self.site = PolyhedralSite(vertex_indices=vertex_indices, reference_center=CENTRE)
        Site._newid = 1

    def test_polyhedral_site_is_initialised(self):
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_indices=vertex_indices, reference_center=CENTRE)
        self.assertEqual(site.vertex_indices, vertex_indices)
        self.assertEqual(site.vertex_coords, None)
        self.assertEqual(site._delaunay, None)
        self.assertEqual(site.index, 1)
        self.assertEqual(site.label, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.points, [])
        self.assertEqual(site.transitions, Counter())

    def test_polyhedral_site_is_initialised_with_a_label(self):
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_indices=vertex_indices, reference_center=CENTRE,
                              label='foo')
        self.assertEqual(site.label, 'foo')

    def test_reset(self):
        site = self.site
        site._delaunay = 'foo'
        site._vertex_coords = 'bar'
        site.contains_atoms = [1]
        site.trajectory = [2]
        site.transitions = Counter([4])
        site.reset()
        self.assertEqual(site._delaunay, None)
        self.assertEqual(site.vertex_coords, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.transitions, Counter())

    def test_reset_keeps_the_triangles(self):
        """reset() keeps the triangles fixed on the site."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        site.fix_triangles(_TETRAHEDRON, Lattice.cubic(10.0).matrix, [])
        site.reset()
        self.assertEqual(_triples(site._triangles), _triples(_TETRAHEDRON_TRIANGLES))

    def test_reset_clears_notified_coordinates(self):
        """reset() discards coordinates notified with notify_structure_changed."""
        site = self.site
        site.notify_structure_changed(np.full((5, 3), 0.5), Lattice.cubic(10.0).matrix)
        site.reset()
        self.assertIsNone(site.vertex_coords)

    def test_reset_clears_pbc_shift_caches(self):
        """reset() clears PBC image shift caches."""
        site = self.site
        site._pbc_image_shifts = np.zeros((4, 3), dtype=int)
        site._pbc_cached_raw_frac = np.zeros((4, 3))
        site.reset()
        self.assertIsNone(site._pbc_image_shifts)
        self.assertIsNone(site._pbc_cached_raw_frac)

    def test_delaunay_if_not_already_set(self):
        site = self.site
        vertex_coords = np.array([[0.0, 0.0, 0.0],
                                  [0.2, 0.3, 0.4],
                                  [0.8, 0.0, 0.8],
                                  [0.5, 0.5, 0.5]])
        site._vertex_coords = vertex_coords
        with patch('site_analysis.polyhedral_site.Delaunay', autospec=True) as mock_Delaunay:
            mock_Delaunay.return_value = 'mock Delaunay'
            d = site.delaunay
            self.assertEqual('mock Delaunay', d)
            mock_Delaunay.assert_called_with(vertex_coords)
   
    def test_delaunay_if_already_set(self):
        site = self.site
        vertex_coords = np.array([[0.0, 0.0, 0.0],
                                  [0.2, 0.3, 0.4],
                                  [0.8, 0.0, 0.8],
                                  [0.5, 0.5, 0.5]])
        site._vertex_coords = vertex_coords
        site._delaunay = 'foo'
        with patch('site_analysis.polyhedral_site.Delaunay', autospec=True) as mock_Delaunay:
            d = site.delaunay
            self.assertEqual('foo', d)
            mock_Delaunay.assert_not_called()
   
    def test_coordination_number(self):
        site = self.site
        self.assertEqual(site.coordination_number, 4)
   
    def test_cn(self):
        site = self.site
        with patch('site_analysis.polyhedral_site.PolyhedralSite.coordination_number',
                   new_callable=PropertyMock, return_value=12) as mock_coordination_number:
            self.assertEqual(site.cn, 12)

    def test_assign_vertex_coords_no_wrapping(self):
        """Test assign_vertex_coords when PBC correction doesn't cross boundaries."""
        # Create structure with coordinates that don't need PBC correction
        lattice = Lattice.cubic(10.0)
        coords = [[0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4]]
        structure = Structure(lattice, ["Li", "Li", "Li", "Li"], coords)

        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.25, 0.25, 0.25])
        site._delaunay = 'foo'

        with patch('site_analysis.polyhedral_site.correct_pbc') as mock_pbc:
            # Mock returns the same coordinates (no correction needed)
            expected_frac_coords = np.array(coords)
            mock_pbc.return_value = (expected_frac_coords,
                                     np.zeros((4, 3), dtype=np.int64))

            site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)

            # Verify PBC function was called with the raw coordinates
            mock_pbc.assert_called_once()
            np.testing.assert_array_almost_equal(mock_pbc.call_args[0][0], coords)

            # Verify site uses the coordinates and resets Delaunay
            np.testing.assert_array_almost_equal(site.vertex_coords, expected_frac_coords)
            self.assertEqual(site._delaunay, None)

    def test_assign_vertex_coords_with_wrapping(self):
        """Test assign_vertex_coords when PBC correction crosses boundaries."""
        # Create structure with coordinates that span boundaries (need PBC correction)
        lattice = Lattice.cubic(10.0)
        coords = [[0.1, 0.1, 0.1],
                  [0.9, 0.1, 0.1],
                  [0.1, 0.9, 0.9],
                  [0.9, 0.9, 0.9]]
        structure = Structure(lattice, ["Li", "Li", "Li", "Li"], coords)

        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[1.0, 1.0, 1.0])
        site._delaunay = 'foo'

        with patch('site_analysis.polyhedral_site.correct_pbc') as mock_pbc:
            # Mock returns coordinates that have been wrapped across boundaries
            expected_frac_coords = np.array([[1.1, 1.1, 1.1],
                                             [0.9, 1.1, 1.1],
                                             [1.1, 0.9, 0.9],
                                             [0.9, 0.9, 0.9]])
            mock_pbc.return_value = (expected_frac_coords,
                                     np.array([[1, 1, 1], [0, 1, 1],
                                               [1, 0, 0], [0, 0, 0]]))

            site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)

            # Verify PBC function was called with the raw coordinates
            mock_pbc.assert_called_once()
            np.testing.assert_array_almost_equal(mock_pbc.call_args[0][0], coords)

            # Verify site uses the PBC-corrected coordinates and resets Delaunay
            np.testing.assert_array_almost_equal(site.vertex_coords, expected_frac_coords)
            self.assertEqual(site._delaunay, None)

    def test_get_vertex_species(self):
        structure = example_structure(species=['S', 'P', 'O', 'I', 'Cl'])
        site = self.site
        self.assertEqual(site.get_vertex_species([s.species_string for s in structure]),
                         ['S', 'P', 'I', 'Cl'])

    def test_contains_point_raises_runtime_error_if_vertex_coords_are_none(self):
        site = self.site
        with self.assertRaises(RuntimeError):
            site.contains_point(np.array([0.0, 0.0, 0.0]))

    def test_contains_point_algo_parameter_emits_deprecation_warning(self):
        site = self.site
        _assign_tetrahedron(site)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site.contains_point(np.array([0.5, 0.5, 0.5]), algo='simplex')
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("algo", str(w[0].message))

    def test_contains_point_no_warning_without_algo(self):
        site = self.site
        _assign_tetrahedron(site)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site.contains_point(np.array([0.5, 0.5, 0.5]))
            self.assertEqual(len(w), 0)

    def test_contains_point_uses_the_triangles_fixed_on_the_site(self):
        """Containment is bounded by the triangles fixed on the site from the next test on."""
        # A square pyramid whose base corner 0 is raised, so the hull folds the
        # base along the diagonal 1-3; fixing the base as a shared face fans it
        # from its lowest atom, folding it along 0-2.
        vertices = np.array([[0.4, 0.4, 0.42], [0.6, 0.4, 0.4], [0.6, 0.6, 0.4],
                             [0.4, 0.6, 0.4], [0.5, 0.5, 0.6]])
        lattice_matrix = Lattice.cubic(10.0).matrix
        # Inside the hull, below the fold along 0-2 near corner 1.
        point = np.array([0.57, 0.45, 0.4015])
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=range(5), reference_center=CENTRE)
                site.assign_vertex_coords(vertices, lattice_matrix)
                self.assertTrue(site.contains_point(point))
                site.fix_triangles(vertices, lattice_matrix, [[0, 1, 2, 3]])
                self.assertFalse(site.contains_point(point))
                self.assertTrue(site.contains_point(np.array([0.5, 0.5, 0.45])))

    def test_fix_triangles_does_not_assign_the_vertex_coordinates(self):
        """fix_triangles fixes the triangles from a structure without assigning coordinates."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        site.fix_triangles(_TETRAHEDRON, Lattice.cubic(10.0).matrix, [])
        self.assertIsNone(site.vertex_coords)
        self.assertEqual(_triples(site._triangles), _triples(_TETRAHEDRON_TRIANGLES))

    def test_fix_triangles_raises_for_shared_atoms_that_are_not_a_face(self):
        """Shared atoms that are not a face of the site raise ValueError naming the atoms."""
        octahedron = 0.5 + 0.2 * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                           [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
        site = PolyhedralSite(vertex_indices=range(6), reference_center=CENTRE)
        with self.assertRaisesRegex(
                ValueError, r"^Shared atoms \[0, 1, 2\] cannot be triangulated identically"):
            site.fix_triangles(octahedron, Lattice.cubic(10.0).matrix, [[0, 1, 2]])

    def test_fix_triangles_raises_for_degenerate_vertices_naming_the_site(self):
        """Coplanar vertices raise RuntimeError naming the site."""
        Site._newid = 7
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        flat = np.array([[0.4, 0.4, 0.5], [0.6, 0.4, 0.5], [0.6, 0.6, 0.5], [0.4, 0.6, 0.5]])
        with self.assertRaisesRegex(
                RuntimeError,
                r"^Degenerate vertex geometry for polyhedral_site 7 \(vertices \[0, 1, 2, 3\]\)"):
            site.fix_triangles(flat, Lattice.cubic(10.0).matrix, [])

    def test_contains_point_accepts_a_list(self):
        """A point given as a list of three numbers is tested like an array."""
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
                site.assign_vertex_coords(_TETRAHEDRON, Lattice.cubic(10.0).matrix)
                self.assertTrue(site.contains_point(_TETRAHEDRON.mean(axis=0).tolist()))
                self.assertFalse(site.contains_point([0.9, 0.9, 0.9]))

    def test_contains_point_rejects_a_point_that_is_not_three_finite_numbers(self):
        """contains_point raises ValueError for a point that is not three finite numbers."""
        cases = {
            "NaN": [np.nan, 0.5, 0.5],
            "infinite": [np.inf, 0.5, 0.5],
            "two numbers": [0.5, 0.5],
            "four numbers": [0.5, 0.5, 0.5, 0.5],
        }
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
                site.assign_vertex_coords(_TETRAHEDRON, Lattice.cubic(10.0).matrix)
                for name, point in cases.items():
                    with self.subTest(name):
                        with self.assertRaisesRegex(ValueError, "three finite numbers"):
                            site.contains_point(np.array(point))

    def test_too_distorted_site_names_itself(self):
        """A site whose vertex has moved through its opposite face raises naming the site."""
        a, b, c, d = _TETRAHEDRON
        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        reflected = _TETRAHEDRON.copy()
        reflected[3] = d - 2.0 * np.dot(d - a, normal) * normal
        lattice_matrix = Lattice.cubic(10.0).matrix
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                Site._newid = 7
                site = PolyhedralSite(vertex_indices=[3, 5, 8, 9],
                                      reference_center=np.array([0.5, 0.5, 0.5]))
                frac_coords = np.zeros((10, 3))
                frac_coords[[3, 5, 8, 9]] = _TETRAHEDRON
                site.assign_vertex_coords(frac_coords, lattice_matrix)
                site.contains_point(np.array([0.5, 0.5, 0.5]))
                frac_coords[[3, 5, 8, 9]] = reflected
                site.assign_vertex_coords(frac_coords, lattice_matrix)
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"^Site 7 \(vertices \[3, 5, 8, 9\]\): polyhedron is too distorted"):
                    site.contains_point(np.array([0.5, 0.5, 0.5]))

    def test_degenerate_site_names_itself(self):
        """A flat site raises RuntimeError naming the site on its first test."""
        Site._newid = 4
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        flat = np.array([[0.4, 0.4, 0.5], [0.6, 0.4, 0.5], [0.6, 0.6, 0.5], [0.4, 0.6, 0.5]])
        site.assign_vertex_coords(flat, Lattice.cubic(10.0).matrix)
        with self.assertRaisesRegex(
                RuntimeError,
                r"^Degenerate vertex geometry for polyhedral_site 4 \(vertices \[0, 1, 2, 3\]\)$"):
            site.contains_point(np.array([0.5, 0.5, 0.5]))

    def test_contains_point_pbc_images_parameter_is_ignored_and_warns(self):
        """pbc_images emits a DeprecationWarning and the point's own images are tested."""
        site = self.site
        _assign_tetrahedron(site)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inside = site.contains_point(
                np.array([0.1, 0.1, 0.1]), pbc_images=np.array([[0.5, 0.5, 0.5]]))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("pbc_images", str(w[0].message))
        self.assertFalse(inside)

    def test_contains_atom_pbc_images_parameter_emits_deprecation_warning(self):
        """contains_atom warns for pbc_images and does not pass it on."""
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0.3, 0.4, 0.5])
        site = self.site
        site.contains_point = Mock(return_value=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site.contains_atom(atom, pbc_images=np.array([[0.3, 0.4, 0.5]]))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("pbc_images", str(w[0].message))
        self.assertEqual(site.contains_point.call_args.kwargs, {})

    def test_contains_atom_algo_parameter_emits_deprecation_warning(self):
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0.3, 0.4, 0.5])
        site = self.site
        site.contains_point = Mock(return_value=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site.contains_atom(atom, algo='simplex')
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("algo", str(w[0].message))

    def test_contains_atom_calls_contains_point(self):
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0.3, 0.4, 0.5])
        site = self.site
        site.contains_point = Mock(return_value=True)
        return_value = site.contains_atom(atom)
        self.assertTrue(return_value)
        call = site.contains_point.call_args
        np.testing.assert_array_equal(call[0][0], atom.frac_coords)

    def test_centre(self):
        site = self.site
        _assign_tetrahedron(site)
        expected_centre = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_equal(site.centre, expected_centre)
        
    def test_init_with_empty_vertex_indices(self):
        """Test that PolyhedralSite raises ValueError with empty vertex_indices."""
        with self.assertRaises(ValueError) as context:
            PolyhedralSite(vertex_indices=[])
        
        self.assertIn("vertex_indices cannot be empty", str(context.exception))
    
    def test_init_with_repeated_vertex_indices(self):
        """Test that PolyhedralSite raises ValueError with a repeated vertex index."""
        with self.assertRaises(ValueError) as context:
            PolyhedralSite(vertex_indices=[1, 2, 3, 2])

        self.assertIn("vertex_indices lists atom 2 more than once", str(context.exception))

    def test_init_with_negative_vertex_indices(self):
        """Test that PolyhedralSite raises ValueError with a negative vertex index."""
        with self.assertRaises(ValueError) as context:
            PolyhedralSite(vertex_indices=[1, -2, 3])

        self.assertIn("vertex_indices contains negative atom index -2", str(context.exception))

    def test_init_with_non_integer_vertex_indices(self):
        """Test that PolyhedralSite raises TypeError with non-integer vertex_indices."""
        with self.assertRaises(TypeError) as context:
            PolyhedralSite(vertex_indices=[1, 2, "three"])
        
        self.assertEqual(str(context.exception),
                         "vertex index must be an integer, got str 'three'")

    def test_init_with_bool_vertex_indices(self):
        """Test that PolyhedralSite raises TypeError with bool vertex indices."""
        with self.assertRaises(TypeError) as context:
            PolyhedralSite(vertex_indices=[1, 2, True])

        self.assertEqual(str(context.exception),
                         "vertex index must be an integer, got bool True")

    def test_init_with_numpy_integer_vertex_indices(self):
        """Test that numpy integers in a list are accepted and stored as Python ints."""
        site = PolyhedralSite(vertex_indices=[np.int64(1), np.int64(2), np.int64(3)], reference_center=CENTRE)

        self.assertEqual(site.vertex_indices, [1, 2, 3])
        self.assertTrue(all(type(idx) is int for idx in site.vertex_indices))
    
    def test_init_with_generator_vertex_indices(self):
        """Test that a generator of vertex indices is stored as a list."""
        site = PolyhedralSite(vertex_indices=(i for i in range(4)), reference_center=CENTRE)

        self.assertEqual(site.vertex_indices, [0, 1, 2, 3])

    def test_init_rejects_a_reference_centre_that_is_not_three_finite_numbers(self):
        """A reference centre that is not three finite numbers raises ValueError."""
        for name, value in self.INVALID_REFERENCE_CENTRES.items():
            with self.subTest(name):
                with self.assertRaisesRegex(
                        ValueError, "^reference_center must be three finite numbers, got "):
                    PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=value)

    def test_init_stores_the_reference_centre_as_a_float_array(self):
        """A reference centre given as a list is stored as a float64 array."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0, 0.5, 1])
        self.assertEqual(site.reference_center.dtype, np.float64)
        np.testing.assert_array_equal(site.reference_center, [0.0, 0.5, 1.0])

    def test_emptiness_check_with_non_empty_numpy_array(self):
        """Test that non-empty numpy array doesn't cause truthiness ambiguity error."""
        vertex_indices = np.array([0, 1, 2, 3])
        # This will fail with ValueError about ambiguous truth value if bug exists
        PolyhedralSite(vertex_indices, reference_center=CENTRE)
    
    def test_emptiness_check_with_empty_numpy_array(self):
        """Test that empty numpy array is correctly detected as empty."""
        vertex_indices = np.array([])
        with self.assertRaises(ValueError) as context:
            PolyhedralSite(vertex_indices)
        self.assertIn("vertex_indices cannot be empty", str(context.exception))


class TestContainmentInUnreducedCell(unittest.TestCase):
    """Tests that containment uses the exactly unwrapped vertices in an unreduced cell."""

    # Rows (1,0,0), (3,1,0), (0,0,1) scaled to Angstrom: an unreduced basis.
    LATTICE_MATRIX = 4.0 * np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    RAW_FRAC_COORDS = np.array([
        [0.6, 0.6, 0.5],
        [0.4, 0.0, 0.7],
        [0.5, 0.8, 0.4],
        [0.3, 0.0, 0.4],
    ])
    REFERENCE_CENTER = np.array([0.0, 0.0, 0.5])
    # The images nearest the reference centre, from a brute-force search.
    NEAREST_IMAGES = np.array([
        [1.6, -0.4, 0.5],
        [0.4, 0.0, 0.7],
        [0.5, -0.2, 0.4],
        [0.3, 0.0, 0.4],
    ])

    def test_site_contains_its_centroid(self):
        """Vertices unwrapped to their nearest images in an unreduced cell enclose their centroid."""
        centroid = self.NEAREST_IMAGES.mean(axis=0) % 1.0
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=self.REFERENCE_CENTER)
        site.assign_vertex_coords(self.RAW_FRAC_COORDS, self.LATTICE_MATRIX)
        self.assertTrue(site.contains_point(centroid))

    def test_site_spanning_more_than_two_cells_contains_a_point_near_its_far_vertex(self):
        """A site whose vertices span x from 0.9 to 2.2 contains a point near the vertex at x = 2.2."""
        translation = np.array([0.6, 0.0, 0.0])
        vertices = self.NEAREST_IMAGES + translation
        point = (0.8 * vertices[0] + 0.2 * vertices.mean(axis=0)) % 1.0
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                                      reference_center=self.REFERENCE_CENTER + translation)
                site.assign_vertex_coords(self.RAW_FRAC_COORDS + translation, self.LATTICE_MATRIX)
                self.assertTrue(site.contains_point(point))


class TestContainmentImageLimit(unittest.TestCase):
    """Tests that containment rejects a site whose vertices span more cells than it will test."""

    def test_contains_point_rejects_a_site_spanning_too_many_cells(self):
        """contains_point raises ValueError for a site spanning 1001 cells, with and without numba."""
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                site_dict = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE).as_dict()
                site_dict['vertex_coords'] = _TETRAHEDRON + [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0],
                                                             [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                site = PolyhedralSite.from_dict(site_dict)
                with self.assertRaisesRegex(ValueError, r"vertices span \[1001, 1, 1\] cells"):
                    site.contains_point(np.array([0.5, 0.5, 0.5]))

    def test_site_spanning_too_many_cells_names_itself(self):
        """The ValueError for a site spanning 1001 cells names the site and its vertices."""
        site_dict = PolyhedralSite(vertex_indices=[3, 5, 8, 9], reference_center=CENTRE).as_dict()
        site_dict['vertex_coords'] = _TETRAHEDRON + [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        Site._newid = 7
        site = PolyhedralSite.from_dict(site_dict)
        with self.assertRaisesRegex(
                ValueError, r"^Site 7 \(vertices \[3, 5, 8, 9\]\): vertices span \[1001, 1, 1\]"):
            site.contains_point(np.array([0.5, 0.5, 0.5]))

    def test_containment_planes_are_computed_once_per_vertex_assignment(self):
        """The planes are computed on the first test after each assignment only."""
        lattice_matrix = Lattice.cubic(10.0).matrix
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba), \
                    patch('site_analysis.polyhedral_site._containment_planes',
                          wraps=_containment_planes) as set_planes:
                site = PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                                      reference_center=np.array([0.5, 0.5, 0.5]))
                site.assign_vertex_coords(_TETRAHEDRON, lattice_matrix)
                self.assertEqual(set_planes.call_count, 0)
                for _ in range(3):
                    site.contains_point(np.array([0.5, 0.5, 0.5]))
                self.assertEqual(set_planes.call_count, 1)
                site.assign_vertex_coords(_TETRAHEDRON + 0.01, lattice_matrix)
                self.assertEqual(set_planes.call_count, 1)
                site.contains_point(np.array([0.5, 0.5, 0.5]))
                self.assertEqual(set_planes.call_count, 2)


class TestContainmentOfPointsInAnyImage(unittest.TestCase):
    """Tests that containment does not depend on the periodic image a point is given in."""

    def test_point_in_another_image_is_contained(self):
        """A point is contained in whichever periodic image it is given."""
        vertices = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.4, 0.5],
                             [0.5, 0.6, 0.5], [0.5, 0.5, 0.4], [0.5, 0.5, 0.6]])
        for has_numba in BACKENDS:
            with patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=range(6),
                                      reference_center=np.array([0.5, 0.5, 0.5]))
                site.assign_vertex_coords(vertices, Lattice.cubic(10.0).matrix)
                for point in ([0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [-0.5, 2.5, 0.5]):
                    with self.subTest(numba=has_numba, point=point):
                        self.assertTrue(site.contains_point(np.array(point)))


class TestUnwrappedVertexCoords(unittest.TestCase):
    """Tests of the unwrapping that fix_triangles and assign_vertex_coords share."""

    # A tetrahedron around the origin, each vertex given in a different image.
    STRADDLING = np.array([[0.95, 0.95, 0.95], [1.05, 0.95, 0.95],
                           [-0.05, 1.05, 0.95], [0.95, 0.95, 2.05]])
    LATTICE = Lattice.cubic(10.0).matrix

    def test_unwrapped_vertex_coords_are_those_assign_vertex_coords_stores(self):
        """The result is what a fresh site stores, for reference centres in different images."""
        for reference_center in (np.zeros(3), np.ones(3)):
            with self.subTest(reference_center=reference_center):
                unwrapped = PolyhedralSite(
                    vertex_indices=[0, 1, 2, 3], reference_center=reference_center,
                )._unwrapped_vertex_coords(self.STRADDLING, self.LATTICE)
                assigned = PolyhedralSite(
                    vertex_indices=[0, 1, 2, 3], reference_center=reference_center)
                assigned.assign_vertex_coords(self.STRADDLING, self.LATTICE)
                np.testing.assert_array_equal(unwrapped, assigned.vertex_coords)
                self.assertLess(np.ptp(unwrapped, axis=0).max(), 0.2)

    def test_unwrapped_vertex_coords_without_reference_centre_are_those_assign_vertex_coords_stores(self):
        """Without a reference centre (deprecated), the result is what a fresh site stores."""
        with self.assertWarns(FutureWarning):
            site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        unwrapped = site._unwrapped_vertex_coords(self.STRADDLING, self.LATTICE)
        site.assign_vertex_coords(self.STRADDLING, self.LATTICE)
        np.testing.assert_array_equal(unwrapped, site.vertex_coords)
        self.assertLess(np.ptp(unwrapped, axis=0).max(), 0.2)

    def test_unwrapped_vertex_coords_does_not_change_the_site(self):
        """The call stores nothing on a fresh site and keeps the coordinates of one that has them."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=np.zeros(3))
        site._unwrapped_vertex_coords(self.STRADDLING, self.LATTICE)
        self.assertIsNone(site.vertex_coords)
        site.assign_vertex_coords(self.STRADDLING, self.LATTICE)
        before = site.vertex_coords.copy()
        site._unwrapped_vertex_coords(self.STRADDLING + 1.0, self.LATTICE)
        np.testing.assert_array_equal(site.vertex_coords, before)


class TestNotifyStructureChanged(unittest.TestCase):
    """Tests for lazy coordinate assignment via notify_structure_changed."""

    def setUp(self):
        Site._newid = 0
        self.site = PolyhedralSite(vertex_indices=[1, 3], reference_center=CENTRE)

    def test_contains_point_tests_the_notified_frame(self):
        """contains_point tests against the coordinates most recently notified."""
        all_frac = np.array([[0.0, 0.0, 0.0],
                             [0.4, 0.4, 0.4],
                             [0.4, 0.6, 0.6],
                             [0.6, 0.6, 0.4],
                             [0.6, 0.4, 0.6]])
        lattice_matrix = Lattice.cubic(10.0).matrix
        site = PolyhedralSite(vertex_indices=[1, 2, 3, 4], reference_center=CENTRE)
        point = np.array([0.5, 0.5, 0.5])
        site.notify_structure_changed(all_frac, lattice_matrix)
        self.assertTrue(site.contains_point(point))
        site.notify_structure_changed(all_frac + [0.3, 0.0, 0.0], lattice_matrix)
        self.assertFalse(site.contains_point(point))

    def test_vertex_coords_after_notify_are_the_new_frame(self):
        """Reading vertex_coords after notify_structure_changed gives the notified frame's coordinates."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        lattice_matrix = Lattice.cubic(10.0).matrix
        site.assign_vertex_coords(_TETRAHEDRON, lattice_matrix)
        site.notify_structure_changed(_TETRAHEDRON + 0.01, lattice_matrix)
        np.testing.assert_allclose(site.vertex_coords, _TETRAHEDRON + 0.01)

    def test_as_dict_after_notify_contains_the_new_frame(self):
        """as_dict after notify_structure_changed, with no containment test, has the notified coordinates."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        site.notify_structure_changed(_TETRAHEDRON, Lattice.cubic(10.0).matrix)
        np.testing.assert_allclose(site.as_dict()['vertex_coords'], _TETRAHEDRON)

    def test_delaunay_after_notify_reflects_the_new_frame(self):
        """The Delaunay tessellation after notify_structure_changed is of the notified coordinates."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        lattice_matrix = Lattice.cubic(10.0).matrix
        site.assign_vertex_coords(_TETRAHEDRON, lattice_matrix)
        site.delaunay
        site.notify_structure_changed(_TETRAHEDRON + 0.01, lattice_matrix)
        np.testing.assert_allclose(site.delaunay.points, _TETRAHEDRON + 0.01)

    def test_centre_after_notify_is_the_new_frame(self):
        """The centre after notify_structure_changed is that of the notified coordinates."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        lattice_matrix = Lattice.cubic(10.0).matrix
        site.assign_vertex_coords(_TETRAHEDRON, lattice_matrix)
        site.notify_structure_changed(_TETRAHEDRON + 0.01, lattice_matrix)
        np.testing.assert_allclose(site.centre, [0.51, 0.51, 0.51])

    def test_non_finite_notified_frame_raises_on_every_test(self):
        """A non-finite notified frame raises on each containment test, never using the previous frame."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        lattice_matrix = Lattice.cubic(10.0).matrix
        site.assign_vertex_coords(_TETRAHEDRON, lattice_matrix)
        site.contains_point(np.array([0.5, 0.5, 0.5]))
        non_finite = _TETRAHEDRON.copy()
        non_finite[0, 0] = np.nan
        site.notify_structure_changed(non_finite, lattice_matrix)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "^frac_coords must be finite$"):
                site.contains_point(np.array([0.5, 0.5, 0.5]))
        with self.assertRaisesRegex(ValueError, "^frac_coords must be finite$"):
            site.vertex_coords

    def test_assigned_coordinates_replace_notified_coordinates(self):
        """Coordinates assigned after notify_structure_changed replace the notified ones."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        lattice_matrix = Lattice.cubic(10.0).matrix
        site.notify_structure_changed(_TETRAHEDRON + 0.01, lattice_matrix)
        site.assign_vertex_coords(_TETRAHEDRON, lattice_matrix)
        np.testing.assert_allclose(site.vertex_coords, _TETRAHEDRON)

    def test_vertex_coords_cannot_be_assigned(self):
        """vertex_coords is read-only: coordinates are assigned with a lattice."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        with self.assertRaises(AttributeError):
            site.vertex_coords = _TETRAHEDRON

    def test_assign_from_pending_uses_vertex_indices(self):
        """_assign_from_pending extracts the correct rows via vertex_indices."""
        all_frac = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.2, 0.3],
                             [0.5, 0.5, 0.5],
                             [0.4, 0.4, 0.4]])
        lattice_matrix = Lattice.cubic(10.0).matrix
        with patch.object(self.site, '_store_vertex_coords') as mock_store:
            self.site._assign_from_pending(all_frac, lattice_matrix)
            called_frac = mock_store.call_args[0][0]
            expected = all_frac[[1, 3]]
            np.testing.assert_array_equal(called_frac, expected)


class TestStoreVertexCoords(unittest.TestCase):
    """Tests for _store_vertex_coords PBC shift caching."""

    def setUp(self):
        Site._newid = 0
        self.lattice_matrix = Lattice.cubic(10.0).matrix

    def test_first_call_computes_full_shifts(self):
        """First call with no cached shifts uses full computation."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.25, 0.25, 0.25])
        frac = np.array([[0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2],
                         [0.3, 0.3, 0.3],
                         [0.4, 0.4, 0.4]])
        self.assertIsNone(site._pbc_image_shifts)
        site._store_vertex_coords(frac, self.lattice_matrix)
        self.assertIsNotNone(site._pbc_image_shifts)
        self.assertIsNotNone(site.vertex_coords)

    def test_second_call_uses_cached_path(self):
        """Second call with small displacement uses cached shifts."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.25, 0.25, 0.25])
        frac1 = np.array([[0.1, 0.1, 0.1],
                          [0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.3],
                          [0.4, 0.4, 0.4]])
        site._store_vertex_coords(frac1, self.lattice_matrix)
        frac2 = frac1 + 0.005  # small vibration
        with patch('site_analysis.polyhedral_site.correct_pbc') as mock_pbc:
            site._store_vertex_coords(frac2, self.lattice_matrix)
            mock_pbc.assert_not_called()

    def test_large_displacement_falls_through_to_full_computation(self):
        """Large displacement invalidates cache and recomputes."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.25, 0.25, 0.25])
        frac1 = np.array([[0.1, 0.1, 0.1],
                          [0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.3],
                          [0.4, 0.4, 0.4]])
        site._store_vertex_coords(frac1, self.lattice_matrix)
        frac2 = frac1 + 0.4  # large displacement
        with patch('site_analysis.polyhedral_site.correct_pbc',
                   return_value=(frac2, np.zeros((4, 3), dtype=np.int64))) as mock_pbc:
            site._store_vertex_coords(frac2, self.lattice_matrix)
            mock_pbc.assert_called_once()

    def test_wrapping_adjusts_shifts_without_recomputation(self):
        """Coordinate wrapping (e.g. 0.99 -> 0.01) adjusts cached shifts."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.6225, 0.5, 0.5])
        frac1 = np.array([[0.99, 0.5, 0.5],
                          [0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5]])
        site._store_vertex_coords(frac1, self.lattice_matrix)
        original_shifts = site._pbc_image_shifts.copy()
        frac2 = np.array([[0.01, 0.5, 0.5],  # wrapped
                          [0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5],
                          [0.5, 0.5, 0.5]])
        with patch('site_analysis.polyhedral_site.correct_pbc') as mock_pbc:
            site._store_vertex_coords(frac2, self.lattice_matrix)
            mock_pbc.assert_not_called()
        # Shift for vertex 0 should have changed to account for wrapping
        self.assertEqual(site._pbc_image_shifts[0, 0], original_shifts[0, 0] + 1)

    def test_legacy_site_contains_centre_with_vertices_outside_the_cell(self):
        """A site without a reference centre contains its centre on the
        first frame and on later frames when its vertices lie whole cells
        outside the unit cell."""
        tetrahedron = np.array([[0.4, 0.4, 0.4],
                                [0.4, 0.6, 0.6],
                                [0.6, 0.6, 0.4],
                                [0.6, 0.4, 0.6]])
        centre = np.array([0.5, 0.5, 0.5])
        for offset in (2, -2):
            for frame in (1, 2):
                with self.subTest(offset=offset, frame=frame):
                    with self.assertWarns(FutureWarning):
                        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
                    site._store_vertex_coords(
                        tetrahedron + offset, self.lattice_matrix)
                    if frame == 2:
                        site._store_vertex_coords(
                            tetrahedron + offset + 0.005, self.lattice_matrix)
                    self.assertTrue(site.contains_point(centre))

    def test_non_finite_coordinates_raise_without_reference_centre(self):
        """A site without a reference centre rejects a NaN vertex coordinate."""
        with self.assertWarns(FutureWarning):
            site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        frac = np.array([[0.1, 0.1, 0.1],
                         [0.2, 0.2, 0.2],
                         [0.3, np.nan, 0.3],
                         [0.4, 0.4, 0.4]])
        with self.assertRaisesRegex(ValueError, "^frac_coords must be finite$"):
            site._store_vertex_coords(frac, self.lattice_matrix)

    def test_non_finite_coordinates_raise_on_a_later_frame(self):
        """A NaN vertex coordinate is rejected when cached shifts exist from an earlier frame."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.25, 0.25, 0.25])
        frac1 = np.array([[0.1, 0.1, 0.1],
                          [0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.3],
                          [0.4, 0.4, 0.4]])
        site._store_vertex_coords(frac1, self.lattice_matrix)
        frac2 = frac1.copy()
        frac2[2, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "^frac_coords must be finite$"):
            site._store_vertex_coords(frac2, self.lattice_matrix)


def example_structure(species=None):
    if not species:
        species = ['S']*5
    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    cartesian_coords = np.array([[1.0, 1.0, 1.0],
                                 [9.0, 1.0, 1.0],
                                 [5.0, 5.0, 5.0],
                                 [1.0, 9.0, 9.0],
                                 [9.0, 9.0, 9.0]])
    structure = Structure(coords=cartesian_coords,
                          lattice=lattice,
                          species=species,
                          coords_are_cartesian=True)
    return structure

class PolyhedralSiteSerialisationTestCase(unittest.TestCase):
    """Simple unit tests for PolyhedralSite serialisation."""

    def setUp(self):
        Site._newid = 0

    def test_as_dict_includes_vertex_indices(self):
        """Test as_dict includes vertex_indices."""
        site = PolyhedralSite(vertex_indices=[1, 2, 3, 4], reference_center=CENTRE)

        site_dict = site.as_dict()

        self.assertEqual(site_dict['vertex_indices'], [1, 2, 3, 4])
        self.assertIn('vertex_coords', site_dict)

    def test_as_dict_includes_vertex_coords_when_set(self):
        """Test as_dict includes vertex_coords when they exist."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        _assign_tetrahedron(site)

        site_dict = site.as_dict()

        np.testing.assert_array_equal(site_dict['vertex_coords'], _TETRAHEDRON)

    def test_as_dict_includes_none_vertex_coords_when_not_set(self):
        """Test as_dict includes None for vertex_coords when not set."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)

        site_dict = site.as_dict()

        self.assertIsNone(site_dict['vertex_coords'])

    def test_from_dict_creates_site_with_vertex_indices(self):
        """Test from_dict creates site with correct vertex_indices."""
        site_dict = {
            'vertex_indices': [5, 6, 7, 8],
            'vertex_coords': None,
            'contains_atoms': [],
            'reference_center': CENTRE,
        }

        site = PolyhedralSite.from_dict(site_dict)

        self.assertEqual(site.vertex_indices, [5, 6, 7, 8])
        self.assertIsNone(site.vertex_coords)

    def test_from_dict_creates_site_with_vertex_coords(self):
        """Test from_dict creates site with vertex coordinates."""
        vertex_coords = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        site_dict = {
            'vertex_indices': [1, 2],
            'vertex_coords': vertex_coords,
            'contains_atoms': [],
            'label': 'test_site',
            'reference_center': CENTRE,
        }

        site = PolyhedralSite.from_dict(site_dict)

        self.assertEqual(site.vertex_indices, [1, 2])
        np.testing.assert_array_equal(site.vertex_coords, vertex_coords)
        self.assertEqual(site.label, 'test_site')

    def test_from_dict_handles_missing_label(self):
        """Test from_dict handles missing label field."""
        site_dict = {
            'vertex_indices': [1, 2, 3, 4],
            'vertex_coords': None,
            'contains_atoms': [],
            'reference_center': CENTRE,
        }

        site = PolyhedralSite.from_dict(site_dict)

        self.assertIsNone(site.label)

    def test_round_trip_serialisation(self):
        """Test as_dict -> from_dict preserves site data."""
        original = PolyhedralSite(vertex_indices=[10, 11, 12, 13], label="original", reference_center=CENTRE)
        _assign_tetrahedron(original)

        site_dict = original.as_dict()
        reconstructed = PolyhedralSite.from_dict(site_dict)

        self.assertEqual(reconstructed.vertex_indices, original.vertex_indices)
        self.assertEqual(reconstructed.label, original.label)
        np.testing.assert_array_equal(reconstructed.vertex_coords, original.vertex_coords)

    def test_round_trip_through_json_gives_the_same_containment(self):
        """A site restored from a dict that has been through JSON tests points as the original does."""
        points = np.random.default_rng(0).uniform(0.3, 0.7, size=(200, 3))
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                original = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
                original.assign_vertex_coords(_TETRAHEDRON, Lattice.cubic(10.0).matrix)
                site_dict = original.as_dict()
                site_dict['vertex_coords'] = site_dict['vertex_coords'].tolist()
                restored = PolyhedralSite.from_dict(json.loads(json.dumps(site_dict)))

                expected = [original.contains_point(point) for point in points]
                self.assertTrue(any(expected) and not all(expected))
                self.assertEqual([restored.contains_point(point) for point in points], expected)

    def test_restored_site_is_bounded_by_the_hull_of_its_coordinates(self):
        """A dict holds no triangles; a restored site takes the hull of its coordinates."""
        site_dict = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE).as_dict()
        self.assertNotIn('triangles', site_dict)
        site_dict['vertex_coords'] = _TETRAHEDRON

        site = PolyhedralSite.from_dict(site_dict)

        self.assertTrue(site.contains_point(_TETRAHEDRON.mean(axis=0)))
        self.assertFalse(site.contains_point(np.array([0.9, 0.9, 0.9])))

    def test_round_trip_preserves_reference_center(self):
        """A reference centre survives as_dict -> from_dict."""
        original = PolyhedralSite(
            vertex_indices=[0, 1, 2, 3], reference_center=np.array([0.25, 0.5, 0.75]))

        reconstructed = PolyhedralSite.from_dict(original.as_dict())

        np.testing.assert_array_equal(
            reconstructed.reference_center, original.reference_center)

    def test_round_trip_without_reference_center_gives_none(self):
        """A site with no reference centre reloads with none."""
        with self.assertWarns(FutureWarning):
            original = PolyhedralSite(vertex_indices=[0, 1, 2, 3])

        with self.assertWarns(FutureWarning):
            reconstructed = PolyhedralSite.from_dict(original.as_dict())

        self.assertIsNone(reconstructed.reference_center)

    def test_from_dict_without_reference_center_key_gives_none(self):
        """A dict written before reference centres were saved still loads."""
        site_dict = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE).as_dict()
        del site_dict['reference_center']

        with self.assertWarns(FutureWarning):
            site = PolyhedralSite.from_dict(site_dict)

        self.assertIsNone(site.reference_center)
        
    def test_polyhedral_site_init_with_reference_center(self):
        """Test PolyhedralSite can be initialised with a reference centre."""
        vertex_indices = [0, 1, 2, 3]
        reference_center = np.array([0.5, 0.5, 0.5])
        
        site = PolyhedralSite(vertex_indices=vertex_indices, reference_center=reference_center)
        
        np.testing.assert_array_equal(site.reference_center, reference_center)
        self.assertEqual(site.vertex_indices, vertex_indices)
        
    def test_assign_vertex_coords_passes_none_reference_center(self):
        """Test that correct_pbc is called with reference_center=None when not set."""
        structure = example_structure()
        with self.assertWarns(FutureWarning):
            site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])

        with patch('site_analysis.polyhedral_site.correct_pbc') as mock_pbc:
            mock_pbc.return_value = (
                np.array([[0.1, 0.1, 0.1]]),
                np.zeros((1, 3), dtype=np.int64),
            )
            site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)
            mock_pbc.assert_called_once()
            self.assertIsNone(mock_pbc.call_args[0][1])

    def test_assign_vertex_coords_passes_reference_center(self):
        """Test that correct_pbc is called with the reference centre when set."""
        structure = example_structure()
        reference_center = np.array([0.5, 0.5, 0.5])
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=reference_center)

        with patch('site_analysis.polyhedral_site.correct_pbc') as mock_pbc:
            mock_pbc.return_value = (
                np.array([[0.1, 0.1, 0.1]]),
                np.array([[0, 0, 0]]),
            )
            site.assign_vertex_coords(structure.frac_coords, structure.lattice.matrix)
            mock_pbc.assert_called_once()
            np.testing.assert_array_equal(mock_pbc.call_args[0][1], reference_center)


class PolyhedralSiteWithoutReferenceCentreDeprecationTestCase(unittest.TestCase):
    """Sites without a reference centre are deprecated, warning at the caller's line."""

    SITE_DICT = {'vertex_indices': [0, 1, 2, 3], 'vertex_coords': None, 'contains_atoms': []}

    def test_site_without_reference_center_warns(self):
        """A site created without a reference centre issues a FutureWarning pointing to reference_center."""
        with self.assertWarnsRegex(
                FutureWarning,
                r"PolyhedralSite without a reference_center is deprecated and will be "
                r"removed in version 2\.0; use reference_center") as context:
            PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        self.assertEqual(context.filename, __file__)

    def test_site_with_reference_center_does_not_warn(self):
        """A site created with a reference centre issues no warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.5, 0.5, 0.5])

    def test_from_dict_without_reference_center_warns_once(self):
        """from_dict warns once, at the caller's line, for a dict with no reference centre or None."""
        for name, site_dict in (("missing", self.SITE_DICT),
                                ("None", {**self.SITE_DICT, 'reference_center': None})):
            with self.subTest(name):
                with recorded_future_warnings() as caught:
                    PolyhedralSite.from_dict(site_dict)
                self.assertEqual(len(caught), 1)
                self.assertRegex(
                    str(caught[0].message),
                    r"PolyhedralSite\.from_dict for a dict without a reference_center is "
                    r"deprecated and will be removed in version 2\.0; use a dict with a "
                    r"reference_center")
                self.assertEqual(caught[0].filename, __file__)

    def test_from_dict_with_reference_center_does_not_warn(self):
        """from_dict issues no warning for a dict with a reference centre."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PolyhedralSite.from_dict({**self.SITE_DICT, 'reference_center': [0.5, 0.5, 0.5]})

    def test_sites_from_vertex_indices_warns_once_per_call(self):
        """sites_from_vertex_indices warns once, at the caller's line, however many sites it creates."""
        with recorded_future_warnings() as caught:
            PolyhedralSite.sites_from_vertex_indices([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        self.assertEqual(len(caught), 1)
        self.assertRegex(
            str(caught[0].message),
            r"PolyhedralSite\.sites_from_vertex_indices is deprecated and will be removed "
            r"in version 2\.0; use TrajectoryBuilder")
        self.assertEqual(caught[0].filename, __file__)


# Vertex fixtures for containment tests

_TETRAHEDRON = np.array([
    [0.4, 0.4, 0.4],
    [0.4, 0.6, 0.6],
    [0.6, 0.6, 0.4],
    [0.6, 0.4, 0.6],
])

# The outward-wound triangles of _TETRAHEDRON.
_TETRAHEDRON_TRIANGLES = _surface_triangles(range(4), _TETRAHEDRON, np.eye(3), ())


def _triples(triangles: np.ndarray) -> set[tuple[int, ...]]:
    """The triangles as a set of sorted vertex triples."""
    return {tuple(sorted(t)) for t in triangles.tolist()}


def _assign_tetrahedron(site):
    """Assign the coordinates of _TETRAHEDRON to a site's vertices, in a 10 Angstrom cubic cell."""
    frac_coords = np.zeros((max(site.vertex_indices) + 1, 3))
    frac_coords[site.vertex_indices] = _TETRAHEDRON
    site.assign_vertex_coords(frac_coords, Lattice.cubic(10.0).matrix)

if __name__ == '__main__':
    unittest.main()

