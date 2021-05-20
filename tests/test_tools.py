import unittest
from site_analysis.tools import get_vertex_indices, x_pbc, site_index_mapping
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from pymatgen.core import Lattice, Structure, PeriodicSite
from collections import Counter

class ToolsTestCase(unittest.TestCase):

    def test_get_vertex_indices(self):
        # Create a 2x2x2 NaCl supercell
        lattice = Lattice.from_parameters(a=5.0, b=5.0, c=5.0, 
                      alpha=90, beta=90, gamma=90)
        structure = Structure.from_spacegroup(sg='Fm-3m', lattice=lattice, 
                                              species=['Na','Cl'],
                                              coords=[[0.0, 0.0, 0.0],
                                                      [0.5, 0.0, 0.0]])*[2,2,2]
        vertex_indices = get_vertex_indices(structure=structure, centre_species='Na',
                                            vertex_species='Cl', cutoff=3.0,
                                            n_vertices=6)
        c = Counter()
        for vi in vertex_indices:
            self.assertEqual(len(vi), 6)
            c += Counter(vi)
            for i in vi:
                self.assertEqual(structure[i].species_string, 'Cl')
        for i in range(33,64):
            self.assertEqual(c[i], 6)

    def test_x_pbc(self):
        pbc_coords = x_pbc(np.array([0.1, 0.2, 0.3]))
        expected_coords = np.array( [[0.1, 0.2, 0.3],
                                     [1.1, 0.2, 0.3],
                                     [0.1, 1.2, 0.3],
                                     [0.1, 0.2, 1.3],
                                     [1.1, 1.2, 0.3],
                                     [1.1, 0.2, 1.3], 
                                     [0.1, 1.2, 1.3],
                                     [1.1, 1.2, 1.3]] )
        for c in expected_coords:
            self.assertEqual(c in pbc_coords, True)
            
 
    def test_site_index_mapping_one(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.1, 0.1, 0.1],
                            [0.4, 0.6, 0.4]])
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords2]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2)
        np.testing.assert_array_equal(mapping, np.array([0, 1]))

    def test_site_index_mapping_two(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords2]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2)
        np.testing.assert_array_equal(mapping, np.array([1, 0]))
        mapping = site_index_mapping(structure1, structure2)
        np.testing.assert_array_equal(mapping, np.array([1, 0]))
        
    def test_site_index_mapping_with_species_1_as_string(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species1 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species1, coords1)]
        sites2 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords2]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2, species1='Na')
        np.testing.assert_array_equal(mapping, np.array([1]))
        
    def test_site_index_mapping_with_species_1_as_list(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species1 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species1, coords1)]
        sites2 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords2]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2, species1=['Na'])
        np.testing.assert_array_equal(mapping, np.array([1]))
                
    def test_site_index_mapping_with_species_2_as_string(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species2 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species2, coords2)]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2, species2='Na', one_to_one_mapping=False)
        np.testing.assert_array_equal(mapping, np.array([0, 0]))

    def test_site_index_mapping_with_species_2_as_list(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species2 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species2, coords2)]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2, species2=['Na'], one_to_one_mapping=False)
        np.testing.assert_array_equal(mapping, np.array([0, 0]))
                
    def test_site_index_mapping_with_one_to_one_mapping_raises_ValueError_one(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species2 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species2, coords2)]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        with self.assertRaises(ValueError):
            site_index_mapping(structure1, structure2, species2='Na')
            
    def test_site_index_mapping_with_one_to_one_mapping_raises_ValueError_two(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species2 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species2, coords2)]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        with self.assertRaises(ValueError):
            site_index_mapping(structure1, structure2, species2='Na', one_to_one_mapping=True)
        
    def test_site_index_mapping_with_species_1_and_species_2_as_strings_one(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species1 = ['Na', 'Cl']
        species2 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species1, coords1)]
        sites2 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species2, coords2)]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2, species1='Na', species2='Na')
        np.testing.assert_array_equal(mapping, np.array([0]))
        
    def test_site_index_mapping_with_species_1_and_species_2_as_strings_two(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.4, 0.6, 0.4],
                            [0.1, 0.1, 0.1]])
        species1 = ['Na', 'Cl']
        species2 = ['Na', 'Cl']
        sites1 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species1, coords1)]
        sites2 = [PeriodicSite(species=s, coords=c, lattice=lattice) for s, c in zip(species2, coords2)]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping = site_index_mapping(structure1, structure2, species1='Na', species2='Cl')
        np.testing.assert_array_equal(mapping, np.array([1]))
        
    def test_site_index_mapping_with_return_mapping_distances(self):
        a = 6.19399
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        coords1 = np.array([[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5]])
        coords2 = np.array([[0.1, 0.1, 0.1],
                            [0.4, 0.6, 0.4]])
        sites1 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords1]
        sites2 = [PeriodicSite(species='Na', coords=c, lattice=lattice) for c in coords2]
        structure1 = Structure.from_sites(sites1)
        structure2 = Structure.from_sites(sites2)
        mapping, distances = site_index_mapping(structure1, structure2, return_mapping_distances=True)
        np.testing.assert_array_equal(mapping, np.array([0, 1]))
        expected_distance = np.sqrt(3*((0.1*a)**2))
        np.testing.assert_array_almost_equal(distances, np.array([expected_distance, expected_distance]))
              
if __name__ == '__main__':
    unittest.main()
    
