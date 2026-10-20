import unittest
import warnings

import pytest

from site_analysis.polyhedral_site import PolyhedralSite, HullTriangleCache
from site_analysis._compat import HAS_NUMBA
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import patch, Mock, PropertyMock
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay
from pymatgen.core import Structure, Lattice

# Containment with numba where it is installed, and with Delaunay tessellation.
BACKENDS = [True, False] if HAS_NUMBA else [False]


class PolyhedralSiteTestCase(unittest.TestCase):

    INVALID_REFERENCE_CENTRES = {
        "NaN": [np.nan, 0.5, 0.6],
        "infinite": [np.inf, 0.5, 0.6],
        "two numbers": [0.4, 0.5],
        "text": "abc",
        "ragged": [[0.1, 0.2], [0.3]],
        "complex": [1 + 1j, 0.0, 0.0],
    }

    def setUp(self):
        Site._newid = 1
        vertex_indices = [0, 1, 3, 4]
        self.site = PolyhedralSite(vertex_indices=vertex_indices)
        Site._newid = 1

    def test_polyhedral_site_is_initialised(self):
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_indices=vertex_indices)
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
        site = PolyhedralSite(vertex_indices=vertex_indices,
                              label='foo')
        self.assertEqual(site.label, 'foo')

    def test_reset(self):
        site = self.site
        site._delaunay = 'foo'
        site.vertex_coords = 'bar'
        site.contains_atoms = [1]
        site.trajectory = [2]
        site.transitions = Counter([4])
        site.reset()
        self.assertEqual(site._delaunay, None)
        self.assertEqual(site.vertex_coords, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.transitions, Counter())

    def test_reset_preserves_hull_triangle_cache(self):
        site = self.site
        mock_cache = Mock()
        site._hull_triangle_cache = mock_cache
        site.reset()
        self.assertIs(site._hull_triangle_cache, mock_cache)

    def test_reset_clears_pending_state(self):
        """reset() clears pending fractional coords and lattice matrix."""
        site = self.site
        site._pending_frac_coords = np.zeros((4, 3))
        site._pending_lattice_matrix = Lattice.cubic(10.0).matrix
        site.reset()
        self.assertIsNone(site._pending_frac_coords)
        self.assertIsNone(site._pending_lattice_matrix)

    def test_reset_clears_pbc_shift_caches(self):
        """reset() clears PBC image shift caches."""
        site = self.site
        site._pbc_image_shifts = np.zeros((4, 3), dtype=int)
        site._pbc_cached_raw_frac = np.zeros((4, 3))
        site.reset()
        self.assertIsNone(site._pbc_image_shifts)
        self.assertIsNone(site._pbc_cached_raw_frac)

    def test_reset_sets_cache_stale(self):
        """reset() marks the cached hull triangle normals as stale."""
        site = self.site
        site._cache_stale = False
        site.reset()
        self.assertTrue(site._cache_stale)

    def test_delaunay_if_not_already_set(self):
        site = self.site
        vertex_coords = np.array([[0.0, 0.0, 0.0],
                                  [0.2, 0.3, 0.4],
                                  [0.8, 0.0, 0.8],
                                  [0.5, 0.5, 0.5]])
        site.vertex_coords = vertex_coords
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
        site.vertex_coords = vertex_coords
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

        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
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

        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
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
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site.contains_point(np.array([0.5, 0.5, 0.5]), algo='simplex')
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("algo", str(w[0].message))

    def test_contains_point_no_warning_without_algo(self):
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            site.contains_point(np.array([0.5, 0.5, 0.5]))
            self.assertEqual(len(w), 0)

    @patch('site_analysis.polyhedral_site.HAS_NUMBA', False)
    def test_contains_point_uses_delaunay_when_numba_unavailable(self):
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        site._hull_triangle_cache = None
        site._contains_point_delaunay = Mock(return_value=True)
        result = site.contains_point(np.array([0.5, 0.5, 0.5]))
        site._contains_point_delaunay.assert_called_once()
        self.assertTrue(result)

    @unittest.skipUnless(HAS_NUMBA, "numba not available")
    def test_contains_point_uses_cache_when_available(self):
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        mock_cache = Mock()
        mock_cache.image_shifts = np.zeros((1, 3), dtype=int)
        mock_cache.contains_point.return_value = True
        site._hull_triangle_cache = mock_cache
        result = site.contains_point(np.array([0.5, 0.5, 0.5]))
        mock_cache.contains_point.assert_called_once()
        self.assertTrue(result)

    @unittest.skipUnless(HAS_NUMBA, "numba not available")
    def test_contains_point_creates_cache_lazily(self):
        """Test that the hull triangles are cached on first contains_point call."""
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        self.assertIsNone(site._hull_triangle_cache)
        site.contains_point(np.array([0.5, 0.5, 0.5]))
        self.assertIsNotNone(site._hull_triangle_cache)

    def test__contains_point_delaunay_returns_true_if_point_inside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site._contains_point_delaunay(np.array([0.5, 0.5, 0.5]))
            self.assertTrue(in_site)

    def test__contains_point_delaunay_returns_false_if_point_outside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site._contains_point_delaunay(np.array([0.1, 0.1, 0.1]))
            self.assertFalse(in_site)

    def test__contains_point_delaunay_returns_false_if_all_points_outside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site._contains_point_delaunay(np.array([[0.1, 0.1, 0.1],
                                                            [0.8, 0.8, 0.9]]))
            self.assertFalse(in_site)

    def test__contains_point_delaunay_returns_true_if_one_point_inside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site._contains_point_delaunay(np.array([[0.1, 0.1, 0.1],
                                                            [0.5, 0.5, 0.5]]))
            self.assertTrue(in_site)

    def test_contains_point_tests_provided_pbc_images(self):
        """For a site within two cells, the provided images are tested in place of the point's own."""
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        self.assertTrue(site.contains_point(
            np.array([0.1, 0.1, 0.1]), pbc_images=np.array([[0.5, 0.5, 0.5]])))

    def test_contains_atom_forwards_pbc_images(self):
        """contains_atom passes pbc_images through to contains_point."""
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0.3, 0.4, 0.5])
        site = self.site
        site.contains_point = Mock(return_value=True)
        precomputed = np.array([[0.3, 0.4, 0.5]])
        site.contains_atom(atom, pbc_images=precomputed)
        call_kwargs = site.contains_point.call_args[1]
        np.testing.assert_array_equal(call_kwargs['pbc_images'], precomputed)

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
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
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
        site = PolyhedralSite(vertex_indices=[np.int64(1), np.int64(2), np.int64(3)])

        self.assertEqual(site.vertex_indices, [1, 2, 3])
        self.assertTrue(all(type(idx) is int for idx in site.vertex_indices))
    
    def test_init_with_generator_vertex_indices(self):
        """Test that a generator of vertex indices is stored as a list."""
        site = PolyhedralSite(vertex_indices=(i for i in range(4)))

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
        PolyhedralSite(vertex_indices)
    
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
                    patch('site_analysis.polyhedral_site.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                                      reference_center=self.REFERENCE_CENTER + translation)
                site.assign_vertex_coords(self.RAW_FRAC_COORDS + translation, self.LATTICE_MATRIX)
                self.assertTrue(site.contains_point(point))


class TestContainmentOfPointsInAnyImage(unittest.TestCase):
    """Tests that containment does not depend on the periodic image a point is given in."""

    def test_point_in_another_image_is_contained(self):
        """A point is contained in whichever periodic image it is given."""
        vertices = np.array([[0.4, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.4, 0.5],
                             [0.5, 0.6, 0.5], [0.5, 0.5, 0.4], [0.5, 0.5, 0.6]])
        for has_numba in BACKENDS:
            with patch('site_analysis.polyhedral_site.HAS_NUMBA', has_numba):
                site = PolyhedralSite(vertex_indices=range(6),
                                      reference_center=np.array([0.5, 0.5, 0.5]))
                site.assign_vertex_coords(vertices, Lattice.cubic(10.0).matrix)
                for point in ([0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [-0.5, 2.5, 0.5]):
                    with self.subTest(numba=has_numba, point=point):
                        self.assertTrue(site.contains_point(np.array(point)))


class TestNotifyStructureChanged(unittest.TestCase):
    """Tests for lazy coordinate assignment via notify_structure_changed."""

    def setUp(self):
        Site._newid = 0
        self.site = PolyhedralSite(vertex_indices=[1, 3])

    def test_stores_pending_data(self):
        all_frac = np.array([[0.0, 0.0, 0.0],
                             [0.1, 0.2, 0.3],
                             [0.5, 0.5, 0.5],
                             [0.4, 0.4, 0.4]])
        lattice_matrix = Lattice.cubic(10.0).matrix
        self.site.notify_structure_changed(all_frac, lattice_matrix)
        np.testing.assert_array_equal(self.site._pending_frac_coords, all_frac)
        self.assertIs(self.site._pending_lattice_matrix, lattice_matrix)

    def test_contains_point_triggers_assign_from_pending(self):
        """contains_point flushes pending data before testing."""
        all_frac = np.array([[0.0, 0.0, 0.0],
                             [0.4, 0.4, 0.4],
                             [0.4, 0.6, 0.6],
                             [0.6, 0.6, 0.4],
                             [0.6, 0.4, 0.6]])
        lattice_matrix = Lattice.cubic(10.0).matrix
        site = PolyhedralSite(vertex_indices=[1, 2, 3, 4])
        site.notify_structure_changed(all_frac, lattice_matrix)
        # After contains_point, pending data should be cleared
        site.contains_point(np.array([0.5, 0.5, 0.5]))
        self.assertIsNone(site._pending_frac_coords)
        self.assertIsNone(site._pending_lattice_matrix)
        self.assertIsNotNone(site.vertex_coords)

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
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
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
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
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
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
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
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
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
                    site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
                    site._store_vertex_coords(
                        tetrahedron + offset, self.lattice_matrix)
                    if frame == 2:
                        site._store_vertex_coords(
                            tetrahedron + offset + 0.005, self.lattice_matrix)
                    self.assertTrue(site.contains_point(centre))

    @unittest.skipUnless(HAS_NUMBA, "numba not available")
    def test_cache_stale_triggers_update_after_store_vertex_coords(self):
        """After _store_vertex_coords sets _cache_stale, the next
        contains_point call should update the cached hull triangles' normals."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        frac = np.array([[0.4, 0.4, 0.4],
                         [0.4, 0.6, 0.6],
                         [0.6, 0.6, 0.4],
                         [0.6, 0.4, 0.6]])
        site._store_vertex_coords(frac, self.lattice_matrix)
        self.assertTrue(site._cache_stale)
        site.contains_point(np.array([0.5, 0.5, 0.5]))
        self.assertFalse(site._cache_stale)


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
        site = PolyhedralSite(vertex_indices=[1, 2, 3, 4])

        site_dict = site.as_dict()

        self.assertEqual(site_dict['vertex_indices'], [1, 2, 3, 4])
        self.assertIn('vertex_coords', site_dict)

    def test_as_dict_includes_vertex_coords_when_set(self):
        """Test as_dict includes vertex_coords when they exist."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        site.vertex_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        site_dict = site.as_dict()

        np.testing.assert_array_equal(site_dict['vertex_coords'], site.vertex_coords)

    def test_as_dict_includes_none_vertex_coords_when_not_set(self):
        """Test as_dict includes None for vertex_coords when not set."""
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])

        site_dict = site.as_dict()

        self.assertIsNone(site_dict['vertex_coords'])

    def test_from_dict_creates_site_with_vertex_indices(self):
        """Test from_dict creates site with correct vertex_indices."""
        site_dict = {
            'vertex_indices': [5, 6, 7, 8],
            'vertex_coords': None,
            'contains_atoms': []
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
            'label': 'test_site'
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
            'contains_atoms': []
        }

        site = PolyhedralSite.from_dict(site_dict)

        self.assertIsNone(site.label)

    def test_round_trip_serialisation(self):
        """Test as_dict -> from_dict preserves site data."""
        original = PolyhedralSite(vertex_indices=[10, 11, 12, 13], label="original")
        original.vertex_coords = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])

        site_dict = original.as_dict()
        reconstructed = PolyhedralSite.from_dict(site_dict)

        self.assertEqual(reconstructed.vertex_indices, original.vertex_indices)
        self.assertEqual(reconstructed.label, original.label)
        np.testing.assert_array_equal(reconstructed.vertex_coords, original.vertex_coords)
        
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


# Vertex fixtures for hull triangle tests

_TETRAHEDRON = np.array([
    [0.4, 0.4, 0.4],
    [0.4, 0.6, 0.6],
    [0.6, 0.6, 0.4],
    [0.6, 0.4, 0.6],
])

_OCTAHEDRON = np.array([
    [0.6, 0.5, 0.5],
    [0.4, 0.5, 0.5],
    [0.5, 0.6, 0.5],
    [0.5, 0.4, 0.5],
    [0.5, 0.5, 0.6],
    [0.5, 0.5, 0.4],
])

_PBC_SHIFTS = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
])


def _x_pbc(x: np.ndarray) -> np.ndarray:
    return _PBC_SHIFTS + x


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestHullTriangleCache(unittest.TestCase):
    """Tests for HullTriangleCache with numba acceleration."""

    def test_initialisation_computes_triangles(self):
        cache = HullTriangleCache(_TETRAHEDRON)
        # A tetrahedron has 4 triangular faces
        self.assertEqual(cache.triangles.shape[0], 4)
        self.assertEqual(cache.triangles.shape[1], 3)

    def test_initialisation_computes_triangles_octahedron(self):
        cache = HullTriangleCache(_OCTAHEDRON)
        # An octahedron has 8 triangular faces
        self.assertEqual(cache.triangles.shape[0], 8)
        self.assertEqual(cache.triangles.shape[1], 3)

    def test_update_preserves_topology(self):
        cache = HullTriangleCache(_TETRAHEDRON)
        original_simplices = cache.triangles.copy()
        # Slightly perturb vertex coordinates
        perturbed = _TETRAHEDRON + 0.01
        cache.update(perturbed)
        np.testing.assert_array_equal(cache.triangles, original_simplices)

    def test_image_shifts_cover_every_cell_the_vertices_span(self):
        """The image shifts run from 0 to 2 along x for vertices spanning x from 0.4 to 2.6."""
        cache = HullTriangleCache(_TETRAHEDRON + [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.assertEqual(sorted(set(cache.image_shifts[:, 0])), [0, 1, 2])

    def test_contains_point_inside_tetrahedron(self):
        cache = HullTriangleCache(_TETRAHEDRON)
        point_inside = np.array([0.5, 0.5, 0.5])
        self.assertTrue(cache.contains_point(_x_pbc(point_inside)))

    def test_contains_point_outside_tetrahedron(self):
        cache = HullTriangleCache(_TETRAHEDRON)
        point_outside = np.array([0.1, 0.1, 0.1])
        self.assertFalse(cache.contains_point(_x_pbc(point_outside)))

    def test_contains_point_inside_octahedron(self):
        cache = HullTriangleCache(_OCTAHEDRON)
        point_inside = np.array([0.5, 0.5, 0.5])
        self.assertTrue(cache.contains_point(_x_pbc(point_inside)))

    def test_contains_point_outside_octahedron(self):
        cache = HullTriangleCache(_OCTAHEDRON)
        point_outside = np.array([0.1, 0.1, 0.1])
        self.assertFalse(cache.contains_point(_x_pbc(point_outside)))

    def test_contains_point_after_coordinate_update(self):
        """Containment still works after updating coordinates."""
        cache = HullTriangleCache(_TETRAHEDRON)
        # Shift all vertices by +0.1 in x
        shifted = _TETRAHEDRON.copy()
        shifted[:, 0] += 0.1
        cache.update(shifted)
        # Centre of shifted tetrahedron is now at [0.6, 0.5, 0.5]
        point_inside = np.array([0.6, 0.5, 0.5])
        self.assertTrue(cache.contains_point(_x_pbc(point_inside)))
        # Original centre is now outside
        point_outside = np.array([0.4, 0.5, 0.5])
        self.assertFalse(cache.contains_point(_x_pbc(point_outside)))


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
class TestNumbaQuery(unittest.TestCase):
    """Tests for the low-level _numba_sn_query function."""

    def test_query_inside_point(self):
        from site_analysis.polyhedral_site import _numba_sn_query

        cache = HullTriangleCache(_TETRAHEDRON)
        point_inside = np.array([[0.5, 0.5, 0.5]])
        result = _numba_sn_query(
            point_inside,
            cache._triangle_normals,
            cache._triangle_ref_points,
            cache._centre_signs,
        )
        self.assertTrue(result)

    def test_query_outside_point(self):
        from site_analysis.polyhedral_site import _numba_sn_query

        cache = HullTriangleCache(_TETRAHEDRON)
        point_outside = np.array([[0.1, 0.1, 0.1]])
        result = _numba_sn_query(
            point_outside,
            cache._triangle_normals,
            cache._triangle_ref_points,
            cache._centre_signs,
        )
        self.assertFalse(result)

    def test_query_returns_true_if_any_image_inside(self):
        from site_analysis.polyhedral_site import _numba_sn_query

        cache = HullTriangleCache(_TETRAHEDRON)
        # First point outside, second inside
        points = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
        result = _numba_sn_query(
            points,
            cache._triangle_normals,
            cache._triangle_ref_points,
            cache._centre_signs,
        )
        self.assertTrue(result)

    def test_query_returns_false_if_all_images_outside(self):
        from site_analysis.polyhedral_site import _numba_sn_query

        cache = HullTriangleCache(_TETRAHEDRON)
        points = np.array([[0.1, 0.1, 0.1], [0.8, 0.8, 0.9]])
        result = _numba_sn_query(
            points,
            cache._triangle_normals,
            cache._triangle_ref_points,
            cache._centre_signs,
        )
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()

