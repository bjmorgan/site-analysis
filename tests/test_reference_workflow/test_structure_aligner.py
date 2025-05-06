"""Unit tests for the StructureAligner class.

These tests verify that StructureAligner correctly aligns reference structures to target 
structures by finding optimal translation vectors.
"""

import unittest
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.reference_workflow.structure_aligner import StructureAligner


class TestStructureAligner(unittest.TestCase):
    """Test cases for the StructureAligner class."""
    
    def test_basic_translation(self):
        """Test alignment with a simple translation."""
        # Create a simple cubic reference structure
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, 
                            species=["Na", "Cl"], 
                            coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        
        # Create target with a known translation
        translation = [0.1, 0.2, 0.3]
        target_coords = [[coord[0] + translation[0], 
                         coord[1] + translation[1], 
                         coord[2] + translation[2]] for coord in reference.frac_coords]
        target = Structure(lattice, reference.species, target_coords)
        
        # Align structures
        aligner = StructureAligner()
        aligned_structure, trans_vector, metrics = aligner.align(reference, target)
        
        # Check that the translation vector is correct (within numerical precision)
        np.testing.assert_allclose(trans_vector, translation, rtol=1e-3, atol=1e-4)
        
        # Check that the aligned structure has the correct coordinates
        np.testing.assert_allclose(
            aligned_structure.frac_coords, target.frac_coords, rtol=1e-3, atol=1e-4)
            
        # Check metrics - should be near zero for perfect alignment
        self.assertLess(metrics['rmsd'], 1e-3)

    def test_species_based_alignment(self):
        """Test alignment based only on selected species."""
        # Create a reference structure with multiple species
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, 
                            species=["Na", "Cl", "K"], 
                            coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
        
        # Create target with different translations for different species
        # Na and K shifted by [0.1, 0.1, 0.1], Cl shifted by [0.2, 0.2, 0.2]
        target_coords = [
            [0.1, 0.1, 0.1],  # Na - shifted by [0.1, 0.1, 0.1]
            [0.7, 0.7, 0.7],  # Cl - shifted by [0.2, 0.2, 0.2]
            [0.35, 0.35, 0.35]  # K - shifted by [0.1, 0.1, 0.1]
        ]
        target = Structure(lattice, reference.species, target_coords)
        
        # Align structures using only Na and K
        aligner = StructureAligner()
        aligned_structure, trans_vector, metrics = aligner.align(reference, target, species=["Na", "K"])
        
        # Check that the translation matches the Na/K shift
        np.testing.assert_allclose(trans_vector, [0.1, 0.1, 0.1], rtol=1e-3, atol=1e-4)
        
        # Check that Na and K are well-aligned, but Cl is not
        na_idx = reference.indices_from_symbol("Na")[0]
        k_idx = reference.indices_from_symbol("K")[0]
        cl_idx = reference.indices_from_symbol("Cl")[0]
        
        # Na and K should align well
        np.testing.assert_allclose(
            aligned_structure[na_idx].frac_coords, 
            target[na_idx].frac_coords,
            rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(
            aligned_structure[k_idx].frac_coords, 
            target[k_idx].frac_coords,
            rtol=1e-3, atol=1e-4)
            
        # Cl should not align perfectly (check that coordinates are different)
        self.assertFalse(np.allclose(
            aligned_structure[cl_idx].frac_coords, 
            target[cl_idx].frac_coords,
            rtol=1e-3, atol=1e-4))

    def test_different_number_of_atoms(self):
        """Test alignment fails with different numbers of atoms."""
        # Create a reference structure
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, 
                            species=["Na", "Cl"], 
                            coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        
        # Create target with an extra atom
        target = Structure(lattice, 
                         species=["Na", "Cl", "Na"], 
                         coords=[[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3]])
        
        # Alignment should fail with a ValueError
        aligner = StructureAligner()
        with self.assertRaises(ValueError):
            aligner.align(reference, target)
            
        # But should work if we specify a species that has the same count
        aligned_structure, trans_vector, metrics = aligner.align(reference, target, species=["Cl"])
        # Check that only Cl atoms are aligned
        cl_idx_ref = reference.indices_from_symbol("Cl")[0]
        cl_idx_target = target.indices_from_symbol("Cl")[0]
        np.testing.assert_allclose(
            aligned_structure[cl_idx_ref].frac_coords, 
            target[cl_idx_target].frac_coords,
            rtol=1e-3, atol=1e-4)

    def test_different_compositions(self):
        """Test alignment fails with different compositions."""
        # Create a reference structure
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, 
                            species=["Na", "Cl"], 
                            coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        
        # Create target with different species
        target = Structure(lattice, 
                         species=["K", "Br"], 
                         coords=[[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]])
        
        # Alignment should fail with a ValueError
        aligner = StructureAligner()
        with self.assertRaises(ValueError):
            aligner.align(reference, target)

    def test_metric_selection(self):
        """Test different metrics for alignment."""
        # Create a reference structure
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, 
                            species=["Na", "Cl", "Na", "Cl"], 
                            coords=[
                                [0.0, 0.0, 0.0], 
                                [0.5, 0.5, 0.5],
                                [0.2, 0.2, 0.2],
                                [0.7, 0.7, 0.7]
                            ])
        
        # Create target with a non-uniform translation
        target_coords = [
            [0.1, 0.1, 0.1],  # +[0.1, 0.1, 0.1]
            [0.65, 0.65, 0.65],  # +[0.15, 0.15, 0.15]
            [0.35, 0.35, 0.35],  # +[0.15, 0.15, 0.15]
            [0.85, 0.85, 0.85],  # +[0.15, 0.15, 0.15]
        ]
        target = Structure(lattice, reference.species, target_coords)
        
        aligner = StructureAligner()
        
        # Test RMSD metric
        aligned_rmsd, trans_rmsd, metrics_rmsd = aligner.align(
            reference, target, metric='rmsd')
            
        # Test max_dist metric
        aligned_max, trans_max, metrics_max = aligner.align(
            reference, target, metric='max_dist')
            
        # Test mean_dist metric
        aligned_mean, trans_mean, metrics_mean = aligner.align(
            reference, target, metric='mean_dist')
            
        # The translations may differ slightly due to the different metrics
        # but they should all be close to the average translation
        avg_translation = [0.14, 0.14, 0.14]
        self.assertLess(np.linalg.norm(trans_rmsd - avg_translation), 0.05)
        self.assertLess(np.linalg.norm(trans_max - avg_translation), 0.05)
        self.assertLess(np.linalg.norm(trans_mean - avg_translation), 0.05)
        
        # Check that metrics are properly returned
        self.assertIn('rmsd', metrics_rmsd)
        self.assertIn('max_dist', metrics_rmsd)
        self.assertIn('mean_dist', metrics_rmsd)

    def test_pbc_handling(self):
        """Test alignment with periodic boundary conditions."""
        # Create a reference structure
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, 
                            species=["Na", "Cl"], 
                            coords=[[0.05, 0.05, 0.05], [0.55, 0.55, 0.55]])
        
        # Create target with a translation that wraps around PBC
        # 0.05 -> 0.95 is a shift of 0.9 or -0.1
        target = Structure(lattice, 
                         species=["Na", "Cl"], 
                         coords=[[0.95, 0.95, 0.95], [0.45, 0.45, 0.45]])
        
        # Align structures
        aligner = StructureAligner()
        aligned_structure, trans_vector, metrics = aligner.align(reference, target)
        
        # Check that the aligned structure coordinates match the target when considering PBC
        for i in range(len(reference)):
            # Compute minimum distance considering PBC
            dist = target.lattice.get_distance_and_image(
                aligned_structure[i].frac_coords, target[i].frac_coords)[0]
            self.assertLess(dist, 0.01)  # Should be very close
    
    def test_half_cell_offset(self):
        """Test alignment with exactly half a unit cell offset in a highly symmetric structure.
        
        This represents a case where a gradient-based optimizer might struggle,
        as the starting point is a local symetric maximum.
        """
        # Create a reference structure with high symmetry
        # Using rock salt (NaCl) structure which has full cubic symmetry
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
        # Place atoms at the corners of a cube
        coords = np.array([
            [0.0, 0.0, 0.0],  # Na at origin
            [0.0, 0.5, 0.5],  # Na at face center
            [0.5, 0.0, 0.5],  # Na at face center
            [0.5, 0.5, 0.0],  # Na at face center
            [0.5, 0.0, 0.0],  # Cl at edge center
            [0.0, 0.5, 0.0],  # Cl at edge center
            [0.0, 0.0, 0.5],  # Cl at edge center
            [0.5, 0.5, 0.5],  # Cl at body center
        ])
        reference = Structure(lattice, species, coords)
        
        # Create target with exactly half a unit cell offset [0.5, 0.5, 0.5]
        # This creates a perfectly symmetric case with multiple equivalent minima
        half_cell_offset = np.array([0.5, 0.5, 0.5])
        target_coords = np.mod(coords + half_cell_offset, 1.0)
        target = Structure(lattice, species, target_coords)
        
        # Align structures
        aligner = StructureAligner()
        aligned_structure, trans_vector, metrics = aligner.align(reference, target)
        
        # The translation vector should be close to [0.5, 0.5, 0.5] or an equivalent vector
        # Due to the periodicity, we check the absolute values of each vector component
        expected_translation_magnitude = [0.5, 0.5, 0.5]
        np.allclose(np.abs(trans_vector), expected_translation_magnitude, rtol=1e-3, atol=1e-4)
        
        # Check that the aligned structure matches the target
        for i in range(len(reference)):
            # Compute minimum distance considering PBC
            dist = target.lattice.get_distance_and_image(
                aligned_structure[i].frac_coords, target[i].frac_coords)[0]
            self.assertLess(dist, 0.01)  # Should be very close
            
    def test_align_with_permuted_atoms(self):
        """Test that alignment works correctly with permuted atom ordering."""
        # Create a simple reference structure
        lattice = Lattice.cubic(5.0)
        species1 = ["Na", "Cl", "Na", "Cl"]
        coords1 = [
            [0.1, 0.1, 0.1],  # Na1
            [0.6, 0.6, 0.6],  # Cl1
            [0.3, 0.3, 0.3],  # Na2
            [0.8, 0.8, 0.8]   # Cl2
        ]
        reference = Structure(lattice, species1, coords1)
        
        # Create target with permuted atom order (reversed)
        permuted_species = species1[::-1]  # Reverse the order
        permuted_coords = coords1[::-1]    # Reverse the order
        
        target = Structure(lattice, permuted_species, permuted_coords)
        
        # Align the structures
        aligner = StructureAligner()
        aligned_structure, translation_vector, metrics = aligner.align(reference, target)
        
        # Check that alignment succeeded without translation
        self.assertLess(metrics['rmsd'], 0.01)
        np.testing.assert_array_almost_equal(translation_vector, [0, 0, 0], decimal=2)
    
    def test_align_with_permuted_atoms_and_translation(self):
        """Test that alignment works correctly with permuted atoms and translation."""
        # Create a simple reference structure
        lattice = Lattice.cubic(5.0)
        species1 = ["Na", "Cl", "Na", "Cl"]
        coords1 = [
            [0.1, 0.1, 0.1],  # Na1
            [0.6, 0.6, 0.6],  # Cl1
            [0.3, 0.3, 0.3],  # Na2
            [0.8, 0.8, 0.8]   # Cl2
        ]
        reference = Structure(lattice, species1, coords1)
        
        # Apply a translation of [0.1, 0.1, 0.1]
        translation = [0.1, 0.1, 0.1]
        translated_coords = []
        for coord in coords1:
            translated = [(c + t) % 1.0 for c, t in zip(coord, translation)]
            translated_coords.append(translated)
        
        # Permute the atom order
        permuted_species = species1[::-1]  # Reverse the order
        permuted_coords = translated_coords[::-1]  # Reverse the order
        
        target = Structure(lattice, permuted_species, permuted_coords)
        
        # Align the structures
        aligner = StructureAligner()
        aligned_structure, found_translation, metrics = aligner.align(reference, target)
        
        # Check that alignment succeeded and found the correct translation
        self.assertLess(metrics['rmsd'], 0.01)
        np.testing.assert_array_almost_equal(found_translation, translation, decimal=2)
        
    def test_validate_structures(self):
        """Test the _validate_structures method with various scenarios."""
        # Create test structures with different compositions
        lattice = Lattice.cubic(5.0)
        
        # Structure A: Na2Cl2
        species_a = ["Na", "Na", "Cl", "Cl"]
        coords_a = [
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5],
            [0.7, 0.7, 0.7]
        ]
        structure_a = Structure(lattice, species_a, coords_a)
        
        # Structure B: Na2Cl2 with different atom ordering
        species_b = ["Cl", "Na", "Cl", "Na"]
        coords_b = [
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7],
            [0.1, 0.1, 0.1]
        ]
        structure_b = Structure(lattice, species_b, coords_b)
        
        # Structure C: Na3Cl2 (different Na count)
        species_c = ["Na", "Na", "Na", "Cl", "Cl"]
        coords_c = [
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5],
            [0.7, 0.7, 0.7]
        ]
        structure_c = Structure(lattice, species_c, coords_c)
        
        # Structure D: Na2Cl2F1 (extra species)
        species_d = ["Na", "Na", "Cl", "Cl", "F"]
        coords_d = [
            [0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3],
            [0.5, 0.5, 0.5],
            [0.7, 0.7, 0.7],
            [0.9, 0.9, 0.9]
        ]
        structure_d = Structure(lattice, species_d, coords_d)
        
        # Create StructureAligner
        aligner = StructureAligner()
        
        # Case 1: Both structures have identical composition, species=None
        species_list = aligner._validate_structures(structure_a, structure_b, None)
        self.assertCountEqual(species_list, ["Na", "Cl"])
        
        # Case 2: Both structures have different composition, species=None
        with self.assertRaises(ValueError) as context:
            aligner._validate_structures(structure_a, structure_c, None)
        self.assertIn("different compositions", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            aligner._validate_structures(structure_a, structure_d, None)
        self.assertIn("different compositions", str(context.exception))
        
        # Case 3: Species is explicitly specified and exists in both structures
        species_list = aligner._validate_structures(structure_a, structure_b, ["Na"])
        self.assertEqual(species_list, ["Na"])
        
        # Case 4: Species is explicitly specified but doesn't exist in reference
        with self.assertRaises(ValueError) as context:
            aligner._validate_structures(structure_a, structure_d, ["F"])
        self.assertIn("not found in reference structure", str(context.exception))
        
        # Case 5: Species is explicitly specified but doesn't exist in target
        with self.assertRaises(ValueError) as context:
            aligner._validate_structures(structure_d, structure_a, ["F"])
        self.assertIn("not found in target structure", str(context.exception))
        
        # Case 6: Species is explicitly specified but has different counts
        with self.assertRaises(ValueError) as context:
            aligner._validate_structures(structure_a, structure_c, ["Na"])
        self.assertIn("Different number of Na atoms", str(context.exception))

if __name__ == '__main__':
    unittest.main()
