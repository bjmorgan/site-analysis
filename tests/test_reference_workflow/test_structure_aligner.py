"""Unit tests for the StructureAligner class.

These tests verify that StructureAligner correctly aligns reference structures to target 
structures by finding optimal translation vectors.
"""

import unittest
import numpy as np
from pymatgen.core import Structure, Lattice
from unittest.mock import patch, Mock, MagicMock

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
            
        # The translations may differ slightly due to the different metrics
        # but they should all be close to the average translation
        print(trans_rmsd, trans_max)
        avg_translation = [0.14, 0.14, 0.14]
        self.assertLess(np.linalg.norm(trans_rmsd - avg_translation), 0.05)
        self.assertLess(np.linalg.norm(trans_max - avg_translation), 0.05)
        
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
        as the starting point is a local symmetric maximum.
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
        
        # Use the helper function to verify alignment quality by species
        from site_analysis.tools import calculate_species_distances
        species_distances, all_distances = calculate_species_distances(
            aligned_structure, target)
        
        # Verify overall alignment quality
        rmsd = np.sqrt(np.mean(np.array(all_distances)**2)) if all_distances else float('inf')
        self.assertLess(rmsd, 0.01, "Overall RMSD is too large")
        
        # Check that each species is well-aligned
        for sp, distances in species_distances.items():
            for i, dist in enumerate(distances):
                self.assertLess(dist, 0.01, 
                    f"{sp} atom {i} not aligned properly (distance: {dist})")
        
        # Check that the translation vector is one of the valid forms
        trans_components = np.mod(trans_vector, 1.0)  # Ensure in [0,1)
        
        # Convert components close to 1.0 to be close to 0.0 for easier checking
        trans_components = np.abs(trans_components - np.rint(trans_components))
        
        # Count components close to 0.5
        half_shift_count = sum(np.isclose(component, 0.5, atol=0.05) 
                            for component in trans_components)
        
        # Count components close to 0.0
        zero_shift_count = sum(np.isclose(component, 0.0, atol=0.05) 
                            for component in trans_components)
        
        # Valid translations must have either:
        # 1. Exactly one component close to 0.5 and two close to 0.0, OR
        # 2. All three components close to 0.5
        is_valid_translation = (
            (half_shift_count == 1 and zero_shift_count == 2) or  # Single-axis half shift
            (half_shift_count == 3)                               # Diagonal half shift
        )
        
        self.assertTrue(is_valid_translation, 
                    f"Translation vector {trans_vector} is not a valid half-cell shift pattern")

            
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
        
    def test_tolerance_passed_to_minimizer(self):
        """Test that the tolerance parameter is correctly passed to the minimizer."""
        # Mock structures - simple mocks are sufficient
        reference = Mock(spec=Structure)
        target = Mock(spec=Structure)
        
        # Create aligner
        aligner = StructureAligner()
        
        # Mock _validate_structures to avoid structure validation
        aligner._validate_structures = Mock(return_value=["Na"])
        
        # Mock _create_objective_function to return a simple objective function
        mock_objective = Mock(return_value=0.1)
        aligner._create_objective_function = Mock(return_value=mock_objective)
        
        # Custom tolerance value
        custom_tolerance = 0.05
        
        # Mock _run_nelder_mead to check if it receives the correct tolerance
        with patch.object(aligner, '_run_nelder_mead') as mock_run_nelder_mead:
            # Configure mock to return a valid translation vector
            mock_run_nelder_mead.return_value = np.array([0.1, 0.1, 0.1])
            
            # Mock _apply_translation to return a mock structure
            aligned_structure = Mock(spec=Structure)
            aligner._apply_translation = Mock(return_value=aligned_structure)
            
            # Also mock calculate_species_distances for the metrics calculation
            with patch('site_analysis.reference_workflow.structure_aligner.calculate_species_distances') as mock_calc_distances:
                mock_calc_distances.return_value = ({}, [0.1])
                
                # Call align with custom tolerance
                aligner.align(reference, target, tolerance=custom_tolerance)
                
                # Check that _run_nelder_mead was called with the correct tolerance value
                mock_run_nelder_mead.assert_called_once()
                args, kwargs = mock_run_nelder_mead.call_args
                self.assertEqual(args[0], mock_objective)  # First arg is objective function
                self.assertEqual(args[1], custom_tolerance)  # Second arg is tolerance
                
    def test_create_objective_function(self):
        """Test that _create_objective_function properly creates an objective function."""
        # Create aligner
        aligner = StructureAligner()
        
        # Use MagicMock which handles magic methods better
        reference = MagicMock(spec=Structure)
        target = MagicMock(spec=Structure)
        
        # Set up frac_coords and length behaviour
        frac_coords = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]])
        reference.frac_coords = frac_coords
        reference.__len__.return_value = 2
        
        # Create site mocks
        site_mocks = []
        for i in range(2):
            site_mock = MagicMock()
            site_mock.species = f"species_{i}"
            site_mocks.append(site_mock)
        
        # Configure indexing behaviour
        reference.__getitem__.side_effect = lambda i: site_mocks[i]
        
        # Create a copy mock with the same behaviour
        copy_mock = MagicMock(spec=Structure)
        copy_mock.frac_coords = frac_coords.copy()
        copy_mock.__len__.return_value = 2
        copy_mock.__getitem__.side_effect = lambda i: site_mocks[i]
        reference.copy.return_value = copy_mock
        
        # Mock calculate_species_distances
        with patch('site_analysis.reference_workflow.structure_aligner.calculate_species_distances') as mock_calc_distances:
            # Configure mock to return a known value
            mock_calc_distances.return_value = ({}, [0.1, 0.2])
            
            # Create the objective function
            objective_function = aligner._create_objective_function(
                reference, target, valid_species=["Na"], metric='rmsd')
            
            # Verify the objective function is callable
            self.assertTrue(callable(objective_function))
            
            # Test the objective function with a translation vector
            result = objective_function(np.array([0.1, 0.1, 0.1]))
            
            # Verify calculate_species_distances was called
            mock_calc_distances.assert_called_once()
            
            # Verify result is a float (the RMSD value)
            self.assertIsInstance(result, float)
            
    def test_run_nelder_mead(self):
        """Test that _run_nelder_mead properly runs the Nelder-Mead algorithm."""
        # Create aligner
        aligner = StructureAligner()
        
        # Create a mock objective function
        objective_function = Mock(return_value=0.1)
        
        # Mock minimize to check if it receives the correct parameters
        with patch('scipy.optimize.minimize') as mock_minimize:
            # Configure mock to return a valid result
            mock_result = Mock()
            mock_result.success = True
            mock_result.x = np.array([0.1, 0.1, 0.1])
            mock_minimize.return_value = mock_result
            
            # Call the method
            result = aligner._run_nelder_mead(objective_function, tolerance=0.05)
            
            # Verify minimize was called with the correct parameters
            mock_minimize.assert_called_once()
            args, kwargs = mock_minimize.call_args
            self.assertEqual(args[0], objective_function)
            self.assertEqual(kwargs['method'], 'Nelder-Mead')
            self.assertEqual(kwargs['x0'].tolist(), [0, 0, 0])
            self.assertEqual(kwargs['options']['xatol'], 0.05)
            self.assertEqual(kwargs['options']['fatol'], 0.05)
            
            # Verify the result
            np.testing.assert_array_equal(result, [0.1, 0.1, 0.1])
            
    def test_run_differential_evolution_basic_functionality(self):
        """Test that _run_differential_evolution calls the SciPy function correctly."""
        # Create aligner
        aligner = StructureAligner()
        
        # Create a simple mock objective function
        objective_function = Mock()
        
        # Mock differential_evolution to avoid actual optimization
        with patch('scipy.optimize.differential_evolution') as mock_de:
            # Configure mock to return a simple result
            mock_result = Mock(success=True, x=np.array([0.1, 0.1, 0.1]))
            mock_de.return_value = mock_result
            
            # Call the method with minimal parameters
            aligner._run_differential_evolution(objective_function, tolerance=0.05)
            
            # Simply verify that differential_evolution was called with the right function
            mock_de.assert_called_once()
            args, _ = mock_de.call_args
            self.assertEqual(args[0], objective_function)
    
    def test_run_differential_evolution_returns_valid_translation(self):
        """Test that _run_differential_evolution returns a valid translation vector."""
        aligner = StructureAligner()
        expected_translation = np.array([0.1, 0.2, 0.3])
        
        with patch('scipy.optimize.differential_evolution') as mock_de:
            # Return a known vector
            mock_de.return_value = Mock(success=True, x=expected_translation)
            
            # Call the method
            result = aligner._run_differential_evolution(Mock(), tolerance=0.05)
            
            # Verify the result is the expected translation
            np.testing.assert_array_equal(result, expected_translation)
    
    def test_run_differential_evolution_handles_optimization_failure(self):
        """Test that _run_differential_evolution handles optimization failures."""
        aligner = StructureAligner()
        
        with patch('scipy.optimize.differential_evolution') as mock_de:
            # Mock a failed optimization
            mock_de.return_value = Mock(success=False, message="Test failure")
            
            # Check that it raises the expected error
            with self.assertRaises(ValueError):
                aligner._run_differential_evolution(Mock(), tolerance=0.05)
                
    def test_run_differential_evolution_passes_options(self):
        """Test that _run_differential_evolution correctly passes options to the optimizer."""
        aligner = StructureAligner()
        
        with patch('scipy.optimize.differential_evolution') as mock_de:
            mock_de.return_value = Mock(success=True, x=np.array([0.1, 0.1, 0.1]))
            
            # Call with custom options
            aligner._run_differential_evolution(
                Mock(), 
                tolerance=0.05,
                minimizer_options={'popsize': 20, 'strategy': 'rand1bin'}
            )
            
            # Check that the options were passed correctly
            args, kwargs = mock_de.call_args
            self.assertEqual(kwargs['tol'], 0.05)
            self.assertEqual(kwargs['popsize'], 20)
            self.assertEqual(kwargs['strategy'], 'rand1bin')
            
    def test_align_calls_nelder_mead_by_default(self):
        """Test that align uses Nelder-Mead by default."""
        aligner = StructureAligner()
        
        # Mock all dependencies to isolate just the algorithm selection
        aligner._validate_structures = Mock(return_value=["Na"])
        aligner._create_objective_function = Mock(return_value=Mock())
        aligner._run_nelder_mead = Mock(return_value=np.array([0, 0, 0]))
        aligner._run_differential_evolution = Mock()
        aligner._apply_translation = Mock(return_value=Mock())
        
        # Mock calculate_species_distances to avoid dependencies
        with patch('site_analysis.reference_workflow.structure_aligner.calculate_species_distances') as mock_calc:
            mock_calc.return_value = ({}, [0.1])
            
            # Call align with no algorithm specified
            aligner.align(Mock(), Mock())
            
            # Verify Nelder-Mead was called and differential_evolution was not
            aligner._run_nelder_mead.assert_called_once()
            aligner._run_differential_evolution.assert_not_called()
            
    def test_align_calls_differential_evolution_when_specified(self):
        """Test that align uses differential_evolution when specified."""
        aligner = StructureAligner()
        
        # Mock all dependencies to isolate just the algorithm selection
        aligner._validate_structures = Mock(return_value=["Na"])
        aligner._create_objective_function = Mock(return_value=Mock())
        aligner._run_nelder_mead = Mock()
        aligner._run_differential_evolution = Mock(return_value=np.array([0, 0, 0]))
        aligner._apply_translation = Mock(return_value=Mock())
        
        # Mock calculate_species_distances to avoid dependencies
        with patch('site_analysis.reference_workflow.structure_aligner.calculate_species_distances') as mock_calc:
            mock_calc.return_value = ({}, [0.1])
            
            # Call align with differential_evolution specified
            aligner.align(Mock(), Mock(), algorithm='differential_evolution')
            
            # Verify differential_evolution was called and Nelder-Mead was not
            aligner._run_differential_evolution.assert_called_once()
            aligner._run_nelder_mead.assert_not_called()
    
    def test_align_raises_error_for_unknown_algorithm(self):
        """Test that align raises an error for unknown algorithms."""
        aligner = StructureAligner()
        
        # Mock minimum dependencies needed for the test
        aligner._validate_structures = Mock(return_value=["Na"])
        aligner._create_objective_function = Mock()
        
        # Test with an invalid algorithm
        with self.assertRaises(ValueError) as context:
            aligner.align(Mock(), Mock(), algorithm='invalid_algorithm')
        
        # Verify error message mentions the unknown algorithm
        self.assertIn("invalid_algorithm", str(context.exception))
        
    def test_algorithm_registry(self):
        """Test that the algorithm registry correctly maps algorithms to their implementations."""
        # Create aligner
        aligner = StructureAligner()
        
        # Get the algorithm registry
        registry = aligner._get_algorithm_registry()
        
        # Verify the registry has the expected keys
        self.assertIn('Nelder-Mead', registry)
        self.assertIn('differential_evolution', registry)
        
        # Verify the registry contains callable functions
        self.assertTrue(callable(registry['Nelder-Mead']))
        self.assertTrue(callable(registry['differential_evolution']))
        
        # Verify the registry functions are the correct methods
        self.assertEqual(registry['Nelder-Mead'], aligner._run_nelder_mead)
        self.assertEqual(registry['differential_evolution'], aligner._run_differential_evolution)
        
    def test_run_minimizer(self):
        """Test that _run_minimizer correctly dispatches to the appropriate algorithm."""
        aligner = StructureAligner()
        
        # Mock the registry methods
        aligner._run_nelder_mead = Mock(return_value=np.array([0.1, 0.1, 0.1]))
        aligner._run_differential_evolution = Mock(return_value=np.array([0.2, 0.2, 0.2]))
        
        objective_function = Mock()
        
        # Test Nelder-Mead dispatching
        result = aligner._run_minimizer('Nelder-Mead', objective_function, 0.05)
        aligner._run_nelder_mead.assert_called_once_with(objective_function, 0.05, None)
        np.testing.assert_array_equal(result, np.array([0.1, 0.1, 0.1]))
        
        # Reset mocks
        aligner._run_nelder_mead.reset_mock()
        
        # Test differential_evolution dispatching
        result = aligner._run_minimizer('differential_evolution', objective_function, 0.05)
        aligner._run_differential_evolution.assert_called_once_with(objective_function, 0.05, None)
        np.testing.assert_array_equal(result, np.array([0.2, 0.2, 0.2]))
        
        # Test error for unknown algorithm
        with self.assertRaises(ValueError) as context:
            aligner._run_minimizer('unknown_algorithm', objective_function, 0.05)
        self.assertIn("unknown_algorithm", str(context.exception))
        self.assertIn("Supported algorithms", str(context.exception))

if __name__ == '__main__':
    unittest.main()
