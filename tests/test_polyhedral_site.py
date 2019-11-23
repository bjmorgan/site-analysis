import unittest
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.site import Site
from unittest.mock import patch, Mock, PropertyMock
import numpy as np
from collections import Counter
from scipy.spatial import Delaunay
from pymatgen import Structure, Lattice

class PolyhedralSiteTestCase(unittest.TestCase):

    def setUp(self):
        Site._newid = 1

    def test_polyhedral_site_is_initialised(self):
        vertex_species = ['S', 'I']
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        self.assertEqual(site.vertex_species, vertex_species)
        self.assertEqual(site.vertex_indices, vertex_indices)
        self.assertEqual(site.vertex_coords, None)
        self.assertEqual(site._delaunay, None)
        self.assertEqual(site.index, 1)
        self.assertEqual(site.label, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.points, [])
        self.assertEqual(site.transitions, Counter())

    def test_polyhedral_site_is_initialised_with_a_single_vertex_species(self):
        vertex_species = 'S'
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        self.assertEqual(site.vertex_species, [vertex_species])
        
    def test_polyhedral_site_is_initialised_with_a_label(self):
        vertex_species = 'S'
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices,
                              label='foo')
        self.assertEqual(site.label, 'foo')

    def test_reset(self):
        vertex_species = ['S', 'I']
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
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
        vertex_species = ['S', 'I']
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        vertex_coords = np.array([[0.0, 0.0, 0.0],
                                  [0.2, 0.3, 0.4],
                                  [0.8, 0.0, 0.8],
                                  [0.5, 0.5, 0.5]])
        site.vertex_coords = vertex_coords
        with patch('site_analysis.polyhedral_site.Delaunay', spec=Delaunay) as mock_Delaunay:
            mock_Delaunay.return_value = 'mock Delaunay'
            d = site.delaunay
            self.assertEqual('mock Delaunay', d)
            mock_Delaunay.assert_called_with(vertex_coords)
   
    def test_delaunay_if_already_set(self):
        vertex_species = ['S', 'I']
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        vertex_coords = np.array([[0.0, 0.0, 0.0],
                                  [0.2, 0.3, 0.4],
                                  [0.8, 0.0, 0.8],
                                  [0.5, 0.5, 0.5]])
        site.vertex_coords = vertex_coords
        site._delaunay = 'foo'
        with patch('site_analysis.polyhedral_site.Delaunay', spec=Delaunay) as mock_Delaunay:
            d = site.delaunay
            self.assertEqual('foo', d)
            mock_Delaunay.assert_not_called()
   
    def test_coordination_number(self):
        vertex_species = ['S', 'I']
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        self.assertEqual(site.coordination_number, 4)
   
    def test_cn(self):
        vertex_species = 'S'
        vertex_indices = [1, 2, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        with patch('site_analysis.polyhedral_site.PolyhedralSite.coordination_number',
                   new_callable=PropertyMock, return_value=12) as mock_coordination_number:
            self.assertEqual(site.cn, 12)

    def test_assign_vertex_coords(self):
        lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
        cartesian_coords = np.array([[1.0, 1.0, 1.0],
                                     [3.0, 1.0, 1.0],
                                     [2.0, 2.0, 2.0],
                                     [1.0, 3.0, 3.0],
                                     [3.0, 3.0, 3.0]])
        structure = Structure(coords=cartesian_coords,
                              lattice=lattice,
                              species=['S']*5,
                              coords_are_cartesian=True)
        vertex_species = 'S'
        vertex_indices = [0, 1, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        site._delaunay = 'foo'
        site.assign_vertex_coords(structure)
        np.testing.assert_array_almost_equal(site.vertex_coords,
            cartesian_coords[[0,1,3,4]]/10.0)
        self.assertEqual(site._delaunay, None)
                                    
    def test_assign_vertex_coords_across_periodic_boundary(self):
        lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
        cartesian_coords = np.array([[1.0, 1.0, 1.0],
                                     [9.0, 1.0, 1.0],
                                     [5.0, 5.0, 5.0],
                                     [1.0, 9.0, 9.0],
                                     [9.0, 9.0, 9.0]])
        structure = Structure(coords=cartesian_coords,
                              lattice=lattice,
                              species=['S']*5,
                              coords_are_cartesian=True)
        vertex_species = 'S'
        vertex_indices = [0, 1, 3, 4]
        site = PolyhedralSite(vertex_species=vertex_species,
                              vertex_indices=vertex_indices)
        site._delaunay = 'foo'
        site.assign_vertex_coords(structure)
        expected_fractional_coords = np.array([[1.1, 1.1, 1.1],
                                               [0.9, 1.1, 1.1],
                                               [1.1, 0.9, 0.9],
                                               [0.9, 0.9, 0.9]])
        np.testing.assert_array_almost_equal(site.vertex_coords,
            expected_fractional_coords)
        self.assertEqual(site._delaunay, None)

if __name__ == '__main__':
    unittest.main()
    
