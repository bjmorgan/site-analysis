import unittest
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import patch, Mock, PropertyMock
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay, ConvexHull
from pymatgen.core import Structure, Lattice

class PolyhedralSiteTestCase(unittest.TestCase):

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

    def test_assign_vertex_coords(self):
        structure = example_structure()
        site = self.site
        site._delaunay = 'foo'
        site.assign_vertex_coords(structure)
        expected_frac_coords = np.array([[1.1, 1.1, 1.1],
                                         [0.9, 1.1, 1.1],
                                         [1.1, 0.9, 0.9],
                                         [0.9, 0.9, 0.9]])
        np.testing.assert_array_almost_equal(site.vertex_coords,
            expected_frac_coords)
        self.assertEqual(site._delaunay, None)
                                    
    def test_assign_vertex_coords_across_periodic_boundary(self):
        structure = example_structure()
        site = self.site
        site._delaunay = 'foo'
        site.assign_vertex_coords(structure)
        expected_fractional_coords = np.array([[1.1, 1.1, 1.1],
                                               [0.9, 1.1, 1.1],
                                               [1.1, 0.9, 0.9],
                                               [0.9, 0.9, 0.9]])
        np.testing.assert_array_almost_equal(site.vertex_coords,
            expected_fractional_coords)
        self.assertEqual(site._delaunay, None)

    def test_get_vertex_species(self):
        structure = example_structure(species=['S', 'P', 'O', 'I', 'Cl'])
        site = self.site
        self.assertEqual(site.get_vertex_species(structure),
                         ['S', 'P', 'I', 'Cl'])

    def test_contains_point_raises_runtime_error_if_vertex_coords_are_none(self):
        site = self.site
        with self.assertRaises(RuntimeError):
            site.contains_point(np.array([0.0, 0.0, 0.0]))

    def test_contains_point_raises_value_error_if_algo_is_not_valid(self):
        site = self.site
        with self.assertRaises(ValueError):
            site.contains_point(np.array([0.0, 0.0, 0.0]), algo='foo')

    def test_contains_point_assigns_vertex_coords_if_called_with_structure(self):
        site = self.site
        structure = example_structure()
        site.assign_vertex_coords = Mock()
        site.vertex_coords = np.array([[0.0, 0.0, 0.0]])
        site.contains_point_simplex = Mock()
        x = np.array([0.0, 0.0, 0.0])
        with patch('site_analysis.polyhedral_site.x_pbc', autospec=True) as mock_x_pbc:
            site.contains_point(x, structure=structure, algo='simplex')
            site.assign_vertex_coords.assert_called_with(structure)
            site.contains_point_simplex.assert_called_once()
            mock_x_pbc.assert_called_once_with(x)
    
    def test_contains_point_with_algo_simplex(self):
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        site.contains_point_simplex = Mock()
        x = np.random.random(3)
        x_pbc = np.array([x])
        with patch('site_analysis.polyhedral_site.x_pbc', autospec=True) as mock_x_pbc:
            mock_x_pbc.return_value = x_pbc
            site.contains_point(x, algo='simplex')
            mock_x_pbc.assert_called_with(x)
            site.contains_point_simplex.assert_called_with(x_pbc)

    def test_contains_point_with_algo_sn(self):
        site = self.site
        site.vertex_coords = np.array([[0.4, 0.4, 0.4],
                                       [0.4, 0.6, 0.6],
                                       [0.6, 0.6, 0.4],
                                       [0.6, 0.4, 0.6]])
        site.contains_point_sn = Mock()
        x = np.random.random(3)
        x_pbc = np.array([x])
        with patch('site_analysis.polyhedral_site.x_pbc', autospec=True) as mock_x_pbc:
            mock_x_pbc.return_value = x_pbc
            site.contains_point(x, algo='sn')
            mock_x_pbc.assert_called_with(x)
            site.contains_point_sn.assert_called_with(x_pbc)

    def test_contains_point_simplex_returns_true_if_point_inside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site.contains_point_simplex(np.array([0.5, 0.5, 0.5]))
            self.assertTrue(in_site)

    def test_contains_point_simplex_returns_false_if_point_outside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site.contains_point_simplex(np.array([0.1, 0.1, 0.1]))
            self.assertFalse(in_site)

    def test_contains_point_simplex_returns_false_if_all_points_outside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site.contains_point_simplex(np.array([[0.1, 0.1, 0.1],
                                                            [0.8, 0.8, 0.9]]))
            self.assertFalse(in_site)

    def test_contains_point_simplex_returns_true_if_one_point_inside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.delaunay',
                   new_callable=PropertyMock) as mock_delaunay:
            mock_delaunay.return_value = Delaunay(points)
            in_site = site.contains_point_simplex(np.array([[0.1, 0.1, 0.1],
                                                            [0.5, 0.5, 0.5]]))
            self.assertTrue(in_site)

    def test_contains_point_sn_returns_true_if_point_inside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.centre', 
            new_callable=PropertyMock) as mock_centre:
            mock_centre.return_value = np.array([0.5, 0.5, 0.5])
            with patch('site_analysis.polyhedral_site.ConvexHull', 
                    autospec=True) as mock_ConvexHull:
                mock_ConvexHull.return_value = ConvexHull(points)
                in_site = site.contains_point_sn(np.array([0.5, 0.5, 0.5]))
                self.assertTrue(in_site)

    def test_contains_point_sn_returns_false_if_point_outside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.centre', 
            new_callable=PropertyMock) as mock_centre:
            mock_centre.return_value = np.array([0.5, 0.5, 0.5])
            with patch('site_analysis.polyhedral_site.ConvexHull',
                    autospec=True) as mock_ConvexHull:
                mock_ConvexHull.return_value = ConvexHull(points)
                in_site = site.contains_point_sn(np.array([0.1, 0.1, 0.1]))
                self.assertFalse(in_site)

    def test_contains_point_sn_returns_false_if_all_points_outside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.centre', 
            new_callable=PropertyMock) as mock_centre:
            mock_centre.return_value = np.array([0.5, 0.5, 0.5])
            with patch('site_analysis.polyhedral_site.ConvexHull',
                    autospec=True) as mock_ConvexHull:
                mock_ConvexHull.return_value = ConvexHull(points)
                in_site = site.contains_point_sn(np.array([[0.1, 0.1, 0.1],
                                                        [0.8, 0.8, 0.9]]))
                self.assertFalse(in_site)

    def test_contains_point_sn_returns_true_if_one_point_inside_polyhedron(self):
        site = self.site
        points = np.array([[0.4, 0.4, 0.4],
                           [0.4, 0.6, 0.6],
                           [0.6, 0.6, 0.4],
                           [0.6, 0.4, 0.6]]) 
        with patch('site_analysis.polyhedral_site.PolyhedralSite.centre', 
            new_callable=PropertyMock) as mock_centre:
            mock_centre.return_value = np.array([0.5, 0.5, 0.5])
            with patch('site_analysis.polyhedral_site.ConvexHull',
                    autospec=True) as mock_ConvexHull:
                mock_ConvexHull.return_value = ConvexHull(points)
                in_site = site.contains_point_sn(np.array([[0.1, 0.1, 0.1],
                                                        [0.5, 0.5, 0.5]]))
                self.assertTrue(in_site)

    def test_contains_atom_raises_value_error_if_algo_is_invalid(self):
        atom = Mock(spec=Atom)
        site = self.site
        with self.assertRaises(ValueError):
            site.contains_atom(atom, algo='foo')

    def test_contains_atom_calls_contains_point_if_algo_is_simplex(self):
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0.3, 0.4, 0.5])
        site = self.site
        site.contains_point = Mock(return_value='foo')
        return_value = site.contains_atom(atom, algo='simplex')
        self.assertEqual(return_value, 'foo')
        call = site.contains_point.call_args
        np.testing.assert_array_equal(call[0][0], atom.frac_coords)
        self.assertEqual(call[1], {'algo': 'simplex'})

    def test_contains_atom_calls_contains_point_if_algo_is_sn(self):
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0.3, 0.4, 0.5])
        site = self.site
        site.contains_point = Mock(return_value='foo')
        return_value = site.contains_atom(atom, algo='sn')
        self.assertEqual(return_value, 'foo')
        call = site.contains_point.call_args
        np.testing.assert_array_equal(call[0][0], atom.frac_coords)
        self.assertEqual(call[1], {'algo': 'sn'})

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
    
    def test_init_with_non_integer_vertex_indices(self):
        """Test that PolyhedralSite raises TypeError with non-integer vertex_indices."""
        with self.assertRaises(TypeError) as context:
            PolyhedralSite(vertex_indices=[1, 2, "three"])
        
        self.assertIn("All vertex indices must be integers", str(context.exception))
  
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


if __name__ == '__main__':
    unittest.main()
    
