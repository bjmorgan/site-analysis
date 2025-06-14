import unittest
import numpy as np
from pymatgen.core import Structure, Lattice
from site_analysis.reference_workflow.coord_finder import CoordinationEnvironmentFinder


class CoordinationEnvironmentFinderTestCase(unittest.TestCase):
    
    def test_index_atoms_by_species(self):
        """Test that atoms are correctly indexed by species."""
        # Create a simple structure
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, 
                              species=["Si", "O", "Si", "O", "Na"], 
                              coords=[[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.25]])
        
        finder = CoordinationEnvironmentFinder(structure)
        indices = finder._index_atoms_by_species()
        
        # Check that we get the correct indices for each species
        self.assertIn("Si", indices)
        self.assertIn("O", indices)
        self.assertIn("Na", indices)
        self.assertEqual(indices["Si"], [0, 2])
        self.assertEqual(indices["O"], [1, 3])
        self.assertEqual(indices["Na"], [4])

    def test_find_diamond_coordination(self):
        """Test finding tetrahedral coordination in diamond structure."""
        a = 5.43  # Silicon lattice constant in Angstroms
    
        # FCC lattice for diamond structure
        lattice = Lattice([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    
        # Two atoms in the diamond basis
        sites = [("Si", [0, 0, 0]), ("Si", [0.25, 0.25, 0.25])]
        structure = Structure(lattice, *zip(*sites))
    
        # Create a small supercell to have more atoms
        structure = structure * [2, 2, 2]
    
        finder = CoordinationEnvironmentFinder(structure)
        environments = finder.find_environments(
            center_species="Si",
            coordination_species="Si",
            n_coord=4,
            cutoff=3.0  # This should be sufficient for Si-Si bonds (~2.35 Å)
        )
    
        # Each Si should have 4 Si neighbors in diamond structure
        for center_idx, coordinating in environments.items():
            self.assertEqual(len(coordinating), 4, 
                             f"Si {center_idx} has {len(coordinating)} neighbors, expected 4")
    
    def test_find_cspbi3_coordination(self):
        """Test finding Pb coordination around I in cubic perovskite."""
        # Create cubic perovskite CsPbI3
        lattice = Lattice.cubic(6.0)
        species = ["Cs", "Pb", "I", "I", "I"]
        coords = [
            [0, 0, 0],       # Cs
            [0.5, 0.5, 0.5], # Pb
            [0.5, 0.5, 0],   # I
            [0.5, 0, 0.5],   # I
            [0, 0.5, 0.5]    # I
        ]
        structure = Structure(lattice, species, coords)
        
        finder = CoordinationEnvironmentFinder(structure)
        environments = finder.find_environments(
            center_species="I",
            coordination_species="Pb",
            n_coord=2,
            cutoff=4.0
        )
        
        # Each I should have 2 Pb neighbors in cubic perovskite
        self.assertEqual(len(environments), 3)  # 3 I atoms
        for center_idx, vertices in environments.items():
            self.assertEqual(len(vertices), 2)

    def test_error_for_missing_species(self):
        """Test that appropriate errors are raised for missing species."""
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, species=["Si"], coords=[[0, 0, 0]])
        
        finder = CoordinationEnvironmentFinder(structure)
        
        # Should raise error if center species is missing
        with self.assertRaises(ValueError) as cm:
            finder.find_environments("O", "Si", 4, 3.0)
        self.assertIn("Center species 'O' not found", str(cm.exception))
        
        # Should raise error if vertex species is missing
        with self.assertRaises(ValueError) as cm:
            finder.find_environments("Si", "O", 4, 3.0)
        self.assertIn("Coordinating species 'O' not found", str(cm.exception))

    def test_empty_structure(self):
        """Test behavior with empty structure."""
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, species=[], coords=[])
        
        finder = CoordinationEnvironmentFinder(structure)
        indices = finder._index_atoms_by_species()
        
        self.assertEqual(indices, {})

    def test_find_environments_with_single_coordination_species(self):
        """Test find_environments with single coordination species as string."""
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, 
                              species=["Na", "Cl", "Cl"], 
                              coords=[[0, 0, 0], [0.5, 0.0, 0.0], [0.0, 0.5, 0]])
        
        finder = CoordinationEnvironmentFinder(structure)
        environments = finder.find_environments(
            center_species="Na",
            coordination_species="Cl",  # Single string, not list
            n_coord=4,
            cutoff=3.0
        )
        
        self.assertEqual(len(environments), 1)
        self.assertIn(0, environments)
        self.assertEqual(len(environments[0]), 4)

    def test_find_environments_with_multiple_coordination_species(self):
        """Test find_environments with multiple coordination species."""
        lattice = Lattice.cubic(5.0)
        structure = Structure(lattice, 
                              species=["Na", "Cl", "O", "Cl"], 
                              coords=[[0, 0, 0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        
        finder = CoordinationEnvironmentFinder(structure)
        environments = finder.find_environments(
            center_species="Na",
            coordination_species=["Cl", "O"],  # Multiple species as list
            n_coord=6,
            cutoff=3.0
        )
        
        self.assertEqual(len(environments), 1)
        self.assertIn(0, environments)
        self.assertEqual(len(environments[0]), 6)
        
    def test_find_environments_maps_to_correct_center_index_minimal(self):
        """Test that find_environments correctly maps environment to the right center atom."""
        lattice = Lattice.cubic(5.0)
        
        # Only 2 Li atoms - one isolated, one with S coordination
        species = ["Li", "Li", "S", "S", "S", "S"]
        coords = [
            [0.1, 0.1, 0.1],  # Li index 0 - isolated, no S neighbors
            [0.5, 0.5, 0.5],  # Li index 1 - has tetrahedral S coordination
            [0.4, 0.4, 0.4],  # S index 2 - coordinates Li index 1
            [0.6, 0.4, 0.4],  # S index 3 - coordinates Li index 1  
            [0.4, 0.6, 0.4],  # S index 4 - coordinates Li index 1
            [0.4, 0.4, 0.6],  # S index 5 - coordinates Li index 1
        ]
        
        structure = Structure(lattice, species, coords)
        finder = CoordinationEnvironmentFinder(structure)
        
        environments = finder.find_environments(
            center_species="Li",
            coordination_species="S", 
            n_coord=4,
            cutoff=2.0
        )
        
        # Should correctly map to Li index 1 (which actually has the coordination)
        self.assertEqual(len(environments), 1)
        self.assertIn(1, environments)  # Li at index 1
        self.assertEqual(len(environments[1]), 4)
        
        # Verify it contains the correct S atom indices
        expected_s_indices = {2, 3, 4, 5}
        actual_s_indices = set(environments[1])
        self.assertEqual(actual_s_indices, expected_s_indices)


if __name__ == '__main__':
    unittest.main()
