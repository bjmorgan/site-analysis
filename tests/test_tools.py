import unittest
from site_analysis.tools import get_vertex_indices, x_pbc, site_index_mapping
from site_analysis.tools import get_coordination_indices
from site_analysis.tools import get_nearest_neighbour_indices
from site_analysis.tools import calculate_species_distances
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from pymatgen.core import Lattice, Structure, PeriodicSite
from collections import Counter
import warnings

class ToolsTestCase(unittest.TestCase):

    def test_get_vertex_indices(self):
        # Suppress the deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            
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
    
    def test_get_vertex_indices_deprecation(self):
        """Test that get_vertex_indices raises a deprecation warning."""
        # Create a simple structure
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        
        # Check for the deprecation warning
        with warnings.catch_warnings(record=True) as w:
            # Trigger all warnings
            warnings.simplefilter("always")
            
            # Call the deprecated function
            get_vertex_indices(structure=structure, centre_species='Na',
                            vertex_species='Cl', cutoff=3.0, n_vertices=1)
            
            # Verify that we got a deprecation warning
            self.assertTrue(len(w) > 0, "No warning was raised")
            self.assertTrue(any(issubclass(warning.category, DeprecationWarning) for warning in w),
                            "No DeprecationWarning was raised")
            self.assertTrue(any("deprecated" in str(warning.message) for warning in w),
                            "Warning message does not mention 'deprecated'")

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
        
        
        
class GetCoordinationIndicesTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test structure with varying coordination environments."""
        # Create a test structure with 3 Na atoms having different Cl coordination
        self.test_lattice = Lattice.cubic(10.0)
        self.test_structure = Structure(
            lattice=self.test_lattice,
            species=["Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl", "Cl"],
            coords=[
                [0.0, 0.0, 0.0],    # Na1 (idx 0): 4 Cl neighbors within 2.0 Å
                [0.5, 0.5, 0.5],    # Na2 (idx 1): 6 Cl neighbors within 2.0 Å
                [0.8, 0.8, 0.8],    # Na3 (idx 2): 0 Cl neighbors within 2.0 Å
                [0.15, 0.0, 0.0],   # Cl1 (idx 3) - near Na1
                [0.0, 0.15, 0.0],   # Cl2 (idx 4) - near Na1
                [0.0, 0.0, 0.15],   # Cl3 (idx 5) - near Na1
                [0.15, 0.15, 0.15], # Cl4 (idx 6) - near Na1
                [0.4, 0.5, 0.5],    # Cl5 (idx 7) - near Na2
                [0.5, 0.4, 0.5],    # Cl6 (idx 8) - near Na2
                [0.5, 0.5, 0.4],    # Cl7 (idx 9) - near Na2
                [0.6, 0.5, 0.5],    # Cl8 (idx 10) - near Na2
                [0.5, 0.6, 0.5],    # Cl9 (idx 11) - near Na2
                [0.5, 0.5, 0.6]     # Cl10 (idx 12) - near Na2
            ]
        )
        
        # Expected coordinating indices for Na1 and Na2
        self.na1_expected_coords = {3, 4, 5, 6}  # Cl1-4 indices
        self.na2_expected_coords = {7, 8, 9, 10, 11, 12}  # Cl5-10 indices
        
        # Create a structure with multiple species for coordination
        self.multi_species_structure = Structure(
            lattice=Lattice.cubic(5.0),
            species=["Mg", "O", "O", "F", "F"],
            coords=[
                [0.0, 0.0, 0.0],  # Mg (idx 0) with 2 O and 2 F neighbors
                [0.2, 0.0, 0.0],  # O1 (idx 1)
                [0.0, 0.2, 0.0],  # O2 (idx 2)
                [0.0, 0.0, 0.2],  # F1 (idx 3)
                [0.2, 0.2, 0.0]   # F2 (idx 4)
            ]
        )
        
        # Expected coordinating indices for Mg
        self.mg_o_expected_coords = {1, 2}  # O indices
        self.mg_f_expected_coords = {3, 4}  # F indices
        
        # Create a distance gradient structure
        self.distance_structure = Structure(
            lattice=Lattice.cubic(10.0),
            species=["Na", "Cl", "Cl", "Cl", "Cl", "Cl"],
            coords=[
                [0.0, 0.0, 0.0],  # Na
                [0.1, 0.0, 0.0],  # Cl - 1.0 Å away
                [0.0, 0.2, 0.0],  # Cl - 2.0 Å away
                [0.0, 0.0, 0.3],  # Cl - 3.0 Å away
                [0.4, 0.0, 0.0],  # Cl - 4.0 Å away
                [0.0, 0.5, 0.0]   # Cl - 5.0 Å away
            ]
        )

    def test_exact_coordination_matching(self):
        """Test that only environments with exactly matching coordination are returned."""
        # Test exact match for environments with 4 Cl neighbors
        environments_4 = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=3.0  , 
            n_coord=4
        )
        
        # Should find exactly one environment (Na1)
        self.assertEqual(len(environments_4), 1)
        self.assertEqual(len(environments_4[0]), 4)
        
        # Verify correct atoms were identified
        neighbor_indices = set(environments_4[0])
        self.assertEqual(neighbor_indices, self.na1_expected_coords)
        
        # Test exact match for environments with 6 Cl neighbors
        environments_6 = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=2.0, 
            n_coord=6
        )
        
        # Should find exactly one environment (Na2)
        self.assertEqual(len(environments_6), 1)
        self.assertEqual(len(environments_6[0]), 6)
        
        # Verify correct atoms were identified
        neighbor_indices = set(environments_6[0])
        self.assertEqual(neighbor_indices, self.na2_expected_coords)
        
        # Test no environments with 5 Cl neighbors
        environments_5 = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=2.0, 
            n_coord=5
        )
        
        # Should find no environments
        self.assertEqual(len(environments_5), 0)

    def test_multiple_species(self):
        """Test coordination with multiple species types."""
        # Test Mg coordinated by both O and F atoms
        environments = get_coordination_indices(
            self.multi_species_structure, 
            centre_species="Mg", 
            coordination_species=["O", "F"], 
            cutoff=2.0, 
            n_coord=4
        )
        
        # Should find one environment with exactly 4 coordinating atoms
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(environments[0]), 4)
        
        # Verify all expected coordinating atoms were found
        neighbor_indices = set(environments[0])
        expected_indices = self.mg_o_expected_coords | self.mg_f_expected_coords
        self.assertEqual(neighbor_indices, expected_indices)
        
        # Test Mg coordinated by only O atoms
        environments = get_coordination_indices(
            self.multi_species_structure, 
            centre_species="Mg", 
            coordination_species="O", 
            cutoff=2.0, 
            n_coord=2
        )
        
        # Should find one environment with exactly 2 O atoms
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(environments[0]), 2)
        
        # Verify the correct O atoms were identified
        neighbor_indices = set(environments[0])
        self.assertEqual(neighbor_indices, self.mg_o_expected_coords)

    def test_variable_coordination(self):
        """Test with different coordination numbers for different centers."""
        # Request customized coordination for each Na atom
        environments = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=2.7, 
            n_coord=[4, 6, 0]  # Exact coordination for each Na
        )
        
        # Should find 3 environments matching the requirements
        self.assertEqual(len(environments), 3)
        self.assertEqual(len(environments[0]), 4)  # Na1 with 4 Cl
        self.assertEqual(len(environments[1]), 6)  # Na2 with 6 Cl
        self.assertEqual(len(environments[2]), 0)  # Na3 with 0 Cl
        
        # Test with incorrect coordination requirements
        environments = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=2.0, 
            n_coord=[5, 5, 5]  # No Na has exactly 5 Cl neighbors
        )
        
        # Should find no environments
        self.assertEqual(len(environments), 0)

    def test_edge_cases_and_errors(self):
        """Test handling of edge cases."""
        # Test with non-existent centre species
        with self.assertRaises(ValueError):
            get_coordination_indices(
                self.test_structure, 
                centre_species="K",  # Not in the structure
                coordination_species="Cl", 
                cutoff=2.0, 
                n_coord=4
            )
        
        # Test with non-existent coordinating species
        environments = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species="K",  # Not in the structure
            cutoff=2.0, 
            n_coord=0
        )
        # Should return environments with 0 neighbors (for Na3)
        self.assertEqual(len(environments), 3)
        self.assertEqual(len(environments[0]), 0)
        
        # Test with both non-existent centre and coordinating species
        with self.assertRaises(ValueError):
            get_coordination_indices(
                self.test_structure, 
                centre_species="Ca",  # Not in the structure
                coordination_species="O",   # Not in the structure
                cutoff=2.0, 
                n_coord=4
            )
        
        # Test with mismatched n_coord list length
        with self.assertRaises(ValueError):
            get_coordination_indices(
                self.test_structure, 
                centre_species="Na", 
                coordination_species="Cl", 
                cutoff=2.0, 
                n_coord=[4, 6]  # Too short (3 Na atoms in structure)
            )

    def test_list_of_coordination_species(self):
        """Test coordination with lists of coordinating species, including missing species."""
        # Test with a list where one species doesn't exist in the structure
        environments = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species=["Cl", "K"],  # K doesn't exist in the structure
            cutoff=2.7, 
            n_coord=4
        )
        
        # Should still find Na1 with 4 Cl atoms
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(environments[0]), 4)
        
        # Verify correct atoms were identified (all Cl, no K)
        neighbor_indices = set(environments[0])
        self.assertEqual(neighbor_indices, self.na1_expected_coords)
        
        # Test with list of only missing species
        environments = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species=["K", "Ca"],  # Neither exists in the structure
            cutoff=2.7, 
            n_coord=0
        )
        
        # Should find all Na atoms with 0 coordinating atoms
        self.assertEqual(len(environments), 3)
        self.assertEqual(len(environments[0]), 0)
        
        # Test with an empty list of coordinating species
        environments = get_coordination_indices(
            self.test_structure, 
            centre_species="Na", 
            coordination_species=[],  # Empty list
            cutoff=2.7, 
            n_coord=0
        )
        
        # Should find all Na atoms with 0 coordinating atoms
        self.assertEqual(len(environments), 3)
        for env in environments:
            self.assertEqual(len(env), 0)

    def test_cutoff_sensitivity(self):
        """Test sensitivity to cutoff distance."""
        # Test with distance gradient structure at different cutoffs
        
        # With 1.5 Å cutoff, should find exactly 1 neighbor
        environments = get_coordination_indices(
            self.distance_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=1.5, 
            n_coord=1
        )
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(environments[0]), 1)
        self.assertEqual(environments[0][0], 1)  # First Cl atom
        
        # With 2.5 Å cutoff, should find exactly 2 neighbors
        environments = get_coordination_indices(
            self.distance_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=2.5, 
            n_coord=2
        )
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(environments[0]), 2)
        self.assertEqual(set(environments[0]), {1, 2})  # First and second Cl atoms
        
        # With 3.5 Å cutoff, should find exactly 3 neighbors
        environments = get_coordination_indices(
            self.distance_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=3.5, 
            n_coord=3
        )
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(environments[0]), 3)
        self.assertEqual(set(environments[0]), {1, 2, 3})  # First, second, and third Cl atoms
        
        # With 3.5 Å cutoff but requesting 2 neighbors, should return nothing
        environments = get_coordination_indices(
            self.distance_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=3.5, 
            n_coord=2
        )
        self.assertEqual(len(environments), 0)  # No matching environments
        
        # With 3.5 Å cutoff but requesting 4 neighbors, should return nothing
        environments = get_coordination_indices(
            self.distance_structure, 
            centre_species="Na", 
            coordination_species="Cl", 
            cutoff=3.5, 
            n_coord=4
        )
        self.assertEqual(len(environments), 0)  # No matching environments
        
    def test_calculate_species_distances_identical_structures(self):
        """Test distance calculation between identical structures."""
        # Create a simple structure
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Cl", "Na", "Cl"]
        coords = [
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6],
            [0.3, 0.3, 0.3],
            [0.8, 0.8, 0.8]
        ]
        structure1 = Structure(lattice, species, coords)
        structure2 = Structure(lattice, species, coords)
        
        # Calculate distances
        species_distances, all_distances = calculate_species_distances(structure1, structure2)
        
        # Check results
        self.assertEqual(len(species_distances), 2)  # Na and Cl
        self.assertIn("Na", species_distances)
        self.assertIn("Cl", species_distances)
        
        # Distances should be zero for identical structures
        self.assertEqual(len(species_distances["Na"]), 2)  # Two Na atoms
        self.assertEqual(len(species_distances["Cl"]), 2)  # Two Cl atoms
        for dist in all_distances:
            self.assertAlmostEqual(dist, 0.0, places=5)
    
    def test_calculate_species_distances_translated_structures(self):
        """Test distance calculation between translated structures."""
        # Create a simple structure
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Cl", "Na", "Cl"]
        coords1 = [
            [0.1, 0.1, 0.1],
            [0.6, 0.6, 0.6],
            [0.3, 0.3, 0.3],
            [0.8, 0.8, 0.8]
        ]
        structure1 = Structure(lattice, species, coords1)
        
        # Create a translated version (shift by [0.1, 0.1, 0.1])
        coords2 = [
            [0.2, 0.2, 0.2],  # +0.1 in each direction
            [0.7, 0.7, 0.7],
            [0.4, 0.4, 0.4],
            [0.9, 0.9, 0.9]
        ]
        structure2 = Structure(lattice, species, coords2)
        
        # Calculate distances
        species_distances, all_distances = calculate_species_distances(structure1, structure2)
        
        # Expected distance for a [0.1, 0.1, 0.1] shift in a 5.0 Å unit cell
        expected_dist = 5.0 * np.sqrt(3 * (0.1**2))
        
        # Check results
        for sp, distances in species_distances.items():
            for dist in distances:
                self.assertAlmostEqual(dist, expected_dist, places=5)
    
    def test_calculate_species_distances_different_compositions(self):
        """Test distance calculation between structures with different compositions."""
        # Create structures with different compositions
        lattice = Lattice.cubic(5.0)
        
        # Structure 1: Na2Cl2
        species1 = ["Na", "Na", "Cl", "Cl"]
        coords1 = [
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8]
        ]
        structure1 = Structure(lattice, species1, coords1)
        
        # Structure 2: Na1Cl3
        species2 = ["Na", "Cl", "Cl", "Cl"]
        coords2 = [
            [0.1, 0.1, 0.1],  # Same Na position
            [0.6, 0.6, 0.6],  # Same Cl position
            [0.7, 0.7, 0.7],  # Extra Cl
            [0.8, 0.8, 0.8]   # Same Cl position
        ]
        structure2 = Structure(lattice, species2, coords2)
        
        # Calculate distances with auto-detection of common species
        species_distances, all_distances = calculate_species_distances(structure1, structure2)
        
        # Should include Na and Cl (common to both)
        self.assertEqual(len(species_distances), 2)
        self.assertIn("Na", species_distances)
        self.assertIn("Cl", species_distances)
        
        # Should include 2 Na atoms (the number in structure1)
        self.assertEqual(len(species_distances["Na"]), 2)
        
        # Should include 2 Cl atoms (the number in structure1)
        self.assertEqual(len(species_distances["Cl"]), 2)
        
        # First Na atom should have zero distance (same position in both structures)
        self.assertAlmostEqual(species_distances["Na"][0], 0.0, places=5)
        
        # Second Na atom should have non-zero distance (different position)
        self.assertGreater(species_distances["Na"][1], 0.0)
    
    def test_calculate_species_distances_species_filtering(self):
        """Test distance calculation with explicit species filtering."""
        # Create a multi-species structure
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Cl", "K", "F"]
        coords = [
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.6, 0.6, 0.6],
            [0.8, 0.8, 0.8]
        ]
        structure1 = Structure(lattice, species, coords)
        structure2 = Structure(lattice, species, coords)
        
        # Calculate distances with only Na and K
        species_distances, all_distances = calculate_species_distances(
            structure1, structure2, species=["Na", "K"])
        
        # Should only include Na and K
        self.assertEqual(len(species_distances), 2)
        self.assertIn("Na", species_distances)
        self.assertIn("K", species_distances)
        self.assertNotIn("Cl", species_distances)
        self.assertNotIn("F", species_distances)
        
        # Each should have exactly one atom
        self.assertEqual(len(species_distances["Na"]), 1)
        self.assertEqual(len(species_distances["K"]), 1)
        
        # All distances should be 0 (identical structures)
        self.assertEqual(len(all_distances), 2)
        for dist in all_distances:
            self.assertAlmostEqual(dist, 0.0, places=5)
    
    def test_calculate_species_distances_edge_cases(self):
        """Test edge cases for calculate_species_distances."""
        # Create a minimal structure
        lattice = Lattice.cubic(5.0)
        species = ["Na"]
        coords = [[0.0, 0.0, 0.0]]
        structure = Structure(lattice, species, coords)
        
        # Empty second structure
        empty_structure = Structure(lattice, [], [])
        
        # Calculate distances between structure and empty structure
        species_distances, all_distances = calculate_species_distances(structure, empty_structure)
        
        # No common species, should return empty results
        self.assertEqual(len(species_distances), 0)
        self.assertEqual(len(all_distances), 0)
        
        # Non-existent species
        species_distances, all_distances = calculate_species_distances(
            structure, structure, species=["Cl"])
        
        # No matches for Cl, should return empty results
        self.assertEqual(len(species_distances), 0)
        self.assertEqual(len(all_distances), 0)
        
        
class ToolsValidationTestCase(unittest.TestCase):
    
    def setUp(self):
        """Create test structures for validation tests."""
        self.lattice = Lattice.cubic(5.0)
        self.structure = Structure(
            lattice=self.lattice,
            species=["Na", "Cl", "Na", "Cl"],
            coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]]
        )
        
        self.ref_structure = Structure(
            lattice=self.lattice,
            species=["O", "O"],
            coords=[[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
        )
    
    def test_empty_structure(self):
        """Test that get_nearest_neighbour_indices raises ValueError with empty structure."""
        empty_structure = Structure(self.lattice, [], [])
        
        with self.assertRaises(ValueError) as context:
            get_nearest_neighbour_indices(
                empty_structure, 
                self.ref_structure, 
                vertex_species=["Na"], 
                n_coord=2
            )
        
        self.assertIn("Empty structure provided", str(context.exception))
    
    def test_empty_reference_structure(self):
        """Test that get_nearest_neighbour_indices raises ValueError with empty reference structure."""
        empty_ref = Structure(self.lattice, [], [])
        
        with self.assertRaises(ValueError) as context:
            get_nearest_neighbour_indices(
                self.structure, 
                empty_ref, 
                vertex_species=["Na"], 
                n_coord=2
            )
        
        self.assertIn("Empty reference structure", str(context.exception))
    
    def test_empty_vertex_species(self):
        """Test that get_nearest_neighbour_indices raises ValueError with empty vertex_species."""
        with self.assertRaises(ValueError) as context:
            get_nearest_neighbour_indices(
                self.structure, 
                self.ref_structure, 
                vertex_species=[], 
                n_coord=2
            )
        
        self.assertIn("No vertex species specified", str(context.exception))
    
    def test_non_positive_n_coord(self):
        """Test that get_nearest_neighbour_indices raises ValueError with n_coord <= 0."""
        with self.assertRaises(ValueError) as context:
            get_nearest_neighbour_indices(
                self.structure, 
                self.ref_structure, 
                vertex_species=["Na"], 
                n_coord=0
            )
        
        self.assertIn("n_coord must be positive", str(context.exception))
    
    def test_no_matching_atoms(self):
        """Test that get_nearest_neighbour_indices raises ValueError when no atoms match vertex_species."""
        with self.assertRaises(ValueError) as context:
            get_nearest_neighbour_indices(
                self.structure, 
                self.ref_structure, 
                vertex_species=["K"],  # No K atoms in the structure
                n_coord=2
            )
        
        self.assertIn("No atoms of species", str(context.exception))
    
    def test_too_few_matching_atoms(self):
        """Test that get_nearest_neighbour_indices raises ValueError when fewer matching atoms than n_coord."""
        with self.assertRaises(ValueError) as context:
            get_nearest_neighbour_indices(
                self.structure, 
                self.ref_structure, 
                vertex_species=["Na"],  # 2 Na atoms in the structure
                n_coord=3  # Requesting 3 neighbors
            )
        
        self.assertIn("Requested 3 neighbors but only 2 matching atoms found", str(context.exception))

              
if __name__ == '__main__':
    unittest.main()
    
