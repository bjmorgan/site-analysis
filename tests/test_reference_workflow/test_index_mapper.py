"""Unit tests for the IndexMapper class.

These tests verify that IndexMapper correctly maps coordinating atom indices
from reference structures to target structures while handling various edge cases.
"""

import unittest
import numpy as np
from pymatgen.core import Structure, Lattice
from typing import List

from site_analysis.reference_workflow.index_mapper import IndexMapper


class TestIndexMapper(unittest.TestCase):
    """Test cases for IndexMapper class."""
    
    def test_identical_structures_single_coordination(self):
        """Test that identical structures map perfectly.
        
        When reference and target structures are identical, the mapping
        should preserve all coordinating atom indices.
        """
        # Create identical Na-Cl structure
        lattice = Lattice.cubic(4.0)
        coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        species = ["Na", "Cl"]
        reference = Structure(lattice, species, coords)
        target = Structure(lattice, species, coords)
        
        # Define coordinating atoms (each site has one coordinating atom)
        ref_coordinating = [[1]]  # One site with Cl at index 1
        
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        
        # Should map identically
        self.assertEqual(mapped_coordinating, ref_coordinating)
    
    def test_permuted_atoms_within_species(self):
        """Test mapping when coordinating atoms of same species are permuted.
        
        Tests that IndexMapper correctly identifies and maps permuted
        coordinating atoms of the same species based on closest distance.
        """
        lattice = Lattice.cubic(5.0)
        
        # Reference structure: Na atoms with Cl coordination
        ref_coords = [
            [0.25, 0.5, 0.5],  # Na
            [0.75, 0.5, 0.5],  # Na  
            [0.0, 0.5, 0.5],   # Cl1 - closer to first Na
            [1.0, 0.5, 0.5],   # Cl2 - closer to second Na
        ]
        ref_species = ["Na", "Na", "Cl", "Cl"]
        reference = Structure(lattice, ref_species, ref_coords, coords_are_cartesian=True)
        
        # target structure: Cl atoms permuted in ordering
        target_coords = [
            [0.25, 0.5, 0.5],  # Na - same positions
            [0.75, 0.5, 0.5],  # Na
            [1.0, 0.5, 0.5],   # Cl2 (permuted position)
            [0.0, 0.5, 0.5],   # Cl1 (permuted position)
        ]
        target_species = ["Na", "Na", "Cl", "Cl"]
        target = Structure(lattice, target_species, target_coords, coords_are_cartesian=True)
        
        # Coordinating atoms (each Na coordinated by nearest Cl)
        ref_coordinating = [
            [2],     # First Na coordinated by Cl1 at index 2
            [3]      # Second Na coordinated by Cl2 at index 3  
        ]
        
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        
        # Should map to permuted Cl positions based on closest atoms
        expected = [
            [3],     # Cl1 now at index 3
            [2]      # Cl2 now at index 2
        ]
        self.assertEqual(mapped_coordinating, expected)
    
    def test_permuted_atoms_across_species(self):
        """Test mapping with atoms permuted across species.
        
        Verifies that coordination environments are correctly mapped
        when different species are reordered in the structure.
        """
        lattice = Lattice.cubic(4.0)
        
        # Reference: Na first, then Cl
        ref_coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        ref_species = ["Na", "Cl"]
        reference = Structure(lattice, ref_species, ref_coords)
        
        # target: Cl first, then Na (reversed order)
        target_coords = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
        target_species = ["Cl", "Na"]
        target = Structure(lattice, target_species, target_coords)
        
        # Coordinating atoms
        ref_coordinating = [[1]]  # One site with Cl at index 1
        
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        
        # Indices should be updated based on new ordering
        expected = [[0]]  # Cl now at index 0
        self.assertEqual(mapped_coordinating, expected)
    
    def test_structural_distortion_closest_mapping(self):
        """Test mapping with small structural distortions.
        
        Ensures that slightly distorted structures are mapped correctly
        using closest-atom distances.
        """
        lattice = Lattice.cubic(5.0)
        
        # Reference structure (ideal)
        ref_coords = [
            [0.5, 0.5, 0.5],  # A (central)
            [0.0, 0.5, 0.5],  # B1
            [1.0, 0.5, 0.5],  # B2
            [0.5, 0.0, 0.5],  # B3
        ]
        ref_species = ["A", "B", "B", "B"]
        reference = Structure(lattice, ref_species, ref_coords, coords_are_cartesian=True)
        
        # target structure (slightly distorted)
        target_coords = [
            [0.52, 0.48, 0.5],  # A (distorted position)
            [0.02, 0.48, 0.5],  # B1 (distorted position)
            [0.98, 0.52, 0.5],  # B2 (distorted position)
            [0.48, 0.02, 0.5],  # B3 (distorted position)
        ]
        target_species = ["A", "B", "B", "B"]
        target = Structure(lattice, target_species, target_coords, coords_are_cartesian=True)
        
        # Coordinating atoms (one site with three coordinating B atoms)
        ref_coordinating = [[1, 2, 3]]  # A coordinated by all three B atoms
        
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        
        # Should map to closest atoms despite distortion
        expected = [[1, 2, 3]]  # Same indices as positions are closest
        self.assertEqual(mapped_coordinating, expected)
    
    def test_multiple_coordination_atoms(self):
        """Test mapping multiple coordinating atoms per site.
        
        Verifies correct mapping when sites have multiple coordinating atoms.
        """
        lattice = Lattice.cubic(6.0)
        
        # Perovskite-like structure
        ref_coords = [
            [0.0, 0.0, 0.0],  # A (central)
            [0.5, 0.0, 0.0],  # X
            [0.0, 0.5, 0.0],  # X
            [0.0, 0.0, 0.5],  # X
        ]
        ref_species = ["A", "X", "X", "X"]
        reference = Structure(lattice, ref_species, ref_coords)
        target = Structure(lattice, ref_species, ref_coords)
        
        # Multiple coordinating atoms
        ref_coordinating = [[1, 2, 3]]  # One site coordinated by all X atoms
        
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        
        # Should preserve coordination
        self.assertEqual(mapped_coordinating, ref_coordinating)
    
    def test_missing_coordinating_atom_raises_error(self):
        """Test that 1:1 mapping violations raise appropriate errors.
        
        When multiple reference coordinating atoms map to the same atom
        in the target structure, mapping should fail. This test covers both
        missing atoms and ambiguous distance cases.
        """
        lattice = Lattice.cubic(5.0)
        
        # Reference structure - two distinct B atoms
        ref_coords = [
            [0.0, 0.0, 0.0],  # A 
            [0.2, 0.0, 0.0],  # B1
            [0.8, 0.0, 0.0],  # B2
        ]
        ref_species = ["A", "B", "B"]
        reference = Structure(lattice, ref_species, ref_coords)
        
        # target structure - only one B atom, positioned where it's closest to both
        target_coords = [
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.0, 0.0],  # B (closest to both ref B1 and B2)
        ]
        target_species = ["A", "B"]
        target = Structure(lattice, target_species, target_coords)
        
        # Coordinating atoms reference two distinct atoms
        ref_coordinating = [[1], [2]]  # Two sites, each with different B atom
        
        mapper = IndexMapper()
        
        # Should raise error - multiple reference atoms map to same target atom
        with self.assertRaises(ValueError):
            mapper.map_coordinating_atoms(reference, target, ref_coordinating)
    
    def test_periodic_boundary_conditions(self):
        """Test correct mapping across periodic boundaries.
    
        Ensures atoms that are closest across PBC are mapped correctly,
        verifies both species-blind and species-aware mapping.
        """
        lattice = Lattice.cubic(3.0)

        # Reference structure
        ref_coords = [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]]
        ref_species = ["A", "B"]
        reference = Structure(lattice, ref_species, ref_coords)

        # target structure with multiple options
        target_coords = [
            [0.0, 0.0, 0.0],   # A atom - closest to ref B with PBC
            [0.05, 0.0, 0.0],  # B atom - closest B atom to ref B with PBC
            [0.2, 0.0, 0.0]    # B atom - further away
        ]
        target_species = ["A", "B", "B"]
        target = Structure(lattice, target_species, target_coords)

        # Coordinating atoms
        ref_coordinating = [[1]]  # One site uses B at index 1

        # Test 1: Without species filtering - should map to closest atom (A)
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        expected = [[0]]  # A atom at index 0
        self.assertEqual(mapped_coordinating, expected)
    
        # Test 2: With species filtering - should map to closest B atom
        mapped_with_filter = mapper.map_coordinating_atoms(
            reference, target, ref_coordinating, target_species="B"
        )
        expected_with_filter = [[1]]  # B atom at index 1 (not the B at index 2)
        self.assertEqual(mapped_with_filter, expected_with_filter)

    def test_complex_coordination_environment(self):
        """Test mapping complex coordination (e.g., octahedral).
        
        Tests correct mapping for sites with multiple coordinating atoms.
        """
        lattice = Lattice.cubic(4.0)
        
        # Reference structure: octahedral coordination
        ref_coords = [
            [0.5, 0.5, 0.5],  # A (central)
            [0.0, 0.5, 0.5],  # X (-x direction)
            [1.0, 0.5, 0.5],  # X (+x direction)  
            [0.5, 0.0, 0.5],  # X (-y direction)
            [0.5, 1.0, 0.5],  # X (+y direction)
            [0.5, 0.5, 0.0],  # X (-z direction)
            [0.5, 0.5, 1.0],  # X (+z direction)
            [1.0, 1.0, 1.0],  # X (not coordinated)
            [0.0, 0.5, 1.0],  # X (not coordinated)
        ]
        ref_species = ["A"] + ["X"] * 8
        reference = Structure(lattice, ref_species, ref_coords, coords_are_cartesian=True)
        
        # target structure: identical but with atoms permuted
        target_coords = [
            [0.5, 0.5, 0.5],  # A (central)
            [1.0, 0.5, 0.5],  # X (+x direction) - permuted
            [0.0, 0.5, 0.5],  # X (-x direction) - permuted
            [1.0, 1.0, 1.0],  # X (not coordinated)
            [0.5, 1.0, 0.5],  # X (+y direction) - permuted
            [0.5, 0.0, 0.5],  # X (-y direction) - permuted
            [0.5, 0.5, 1.0],  # X (+z direction) - permuted
            [0.0, 0.5, 1.0],  # X (not coordinated)
            [0.5, 0.5, 0.0],  # X (-z direction) - permuted
        ]
        target_species = ["A"] + ["X"] * 8
        target = Structure(lattice, target_species, target_coords, coords_are_cartesian=True)
        
        # Coordinating atoms
        ref_coordinating = [[1, 2, 3, 4, 5, 6]]  # All 6 X atoms
        
        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(reference, target, ref_coordinating)
        
        # Should correctly map all coordinating atoms
        expected = [[2, 1, 5, 4, 8, 6]]  # Permuted indices
        self.assertEqual(mapped_coordinating, expected)
        
    def test_detailed_error_message(self):
        """Test that 1:1 mapping violation produces detailed error message.
        
        Ensures the error message contains information about which specific
        target atoms were mapped to by multiple reference atoms.
        """
        lattice = Lattice.cubic(5.0)
        
        # Reference structure - 3 reference atoms
        ref_coords = [
            [0.1, 0.5, 0.5],  # Reference atom 0
            [0.3, 0.5, 0.5],  # Reference atom 1
            [0.5, 0.5, 0.5],  # Reference atom 2
        ]
        ref_species = ["A", "A", "A"]
        reference = Structure(lattice, ref_species, ref_coords, coords_are_cartesian=True)
        
        # Target structure - only 2 target atoms, forcing a violation
        target_coords = [
            [0.2, 0.5, 0.5],  # Target atom 0 - will map to ref atoms 0 and 1
            [0.5, 0.5, 0.5],  # Target atom 1 - will map to ref atom 2
        ]
        target_species = ["B", "B"]
        target = Structure(lattice, target_species, target_coords, coords_are_cartesian=True)
        
        # Coordinating atoms - need to map all 3 reference atoms
        ref_coordinating = [[0, 1, 2]]
        
        mapper = IndexMapper()
        
        # Catch the error and check its content
        try:
            mapper.map_coordinating_atoms(reference, target, ref_coordinating)
            self.fail("Expected ValueError was not raised")
        except ValueError as e:
            # Error message should mention the duplicate target atom (index 0)
            error_msg = str(e)
            self.assertIn("1:1 mapping violation", error_msg)
            self.assertIn("indices [0]", error_msg)  # Index 0 should be mentioned as duplicate


"""Unit tests for IndexMapper species filtering functionality.

These tests verify that species filtering works correctly by ensuring atoms
of filtered species are preferred even when other species are closer.
"""

class TestIndexMapperSpeciesFiltering(unittest.TestCase):
    """Test cases for IndexMapper species filtering functionality."""

    def test_single_species_filtering_with_closer_non_target(self):
        """Test single species filtering when non-target atoms are closer.

        Ensures that atoms of target species are mapped even when other species
        are geometrically closer (verifying that filtering is applied correctly).
        """
        lattice = Lattice.cubic(5.0)

        # Reference structure
        ref_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.0, 0.5, 0.5],  # X1 - we want to map this to Cl in target
            [1.0, 0.5, 0.5],  # X2 - we want to map this to Cl in target
        ]
        ref_species = ["A", "X", "X"]
        reference = Structure(lattice, ref_species, ref_coords, coords_are_cartesian=True)

        # target structure - Cl is farther than Br
        target_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.1, 0.5, 0.5],  # Br - CLOSER to ref X1 at [0.0, 0.5, 0.5]
            [0.3, 0.5, 0.5],  # Cl - FARTHER from ref X1, should still be selected
            [0.8, 0.5, 0.5],  # Br - CLOSER to ref X2 at [1.0, 0.5, 0.5]
            [0.7, 0.5, 0.5],  # Cl - FARTHER from ref X2, should still be selected
        ]
        target_species = ["A", "Br", "Cl", "Br", "Cl"]
        target = Structure(lattice, target_species, target_coords, coords_are_cartesian=True)

        # Coordinating atoms
        ref_coordinating = [[1, 2]]  # Both X atoms

        mapper = IndexMapper()

        # Filter to only Cl species - should map to Cl even though Br is closer
        mapped_coordinating = mapper.map_coordinating_atoms(
            reference, target, ref_coordinating, target_species="Cl"
        )

        # Should map to Cl atoms (indices 2 and 4) despite Br being closer
        expected = [[2, 4]]
        self.assertEqual(mapped_coordinating, expected)

    def test_multiple_species_filtering_with_closer_non_target(self):
        """Test multiple species filtering when non-target atoms are closer.

        Ensures that atoms of any target species are preferred over
        geometrically closer atoms of non-target species.
        """
        lattice = Lattice.cubic(6.0)

        # Reference structure
        ref_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.2, 0.5, 0.5],  # B1
            [0.8, 0.5, 0.5],  # B2
            [0.5, 0.2, 0.5],  # C1
            [0.5, 0.8, 0.5],  # C2
        ]
        ref_species = ["A", "B", "B", "C", "C"]
        reference = Structure(lattice, ref_species, ref_coords)

        # target structure - D atoms are closest but not in target species
        target_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.15, 0.5, 0.5], # D - CLOSEST to ref B1
            [0.25, 0.5, 0.5], # B - FARTHER from ref B1
            [0.75, 0.5, 0.5], # D - CLOSEST to ref B2
            [0.85, 0.5, 0.5], # B - FARTHER from ref B2
            [0.5, 0.15, 0.5], # D - CLOSEST to ref C1
            [0.5, 0.25, 0.5], # C - FARTHER from ref C1
            [0.5, 0.75, 0.5], # D - CLOSEST to ref C2
            [0.5, 0.85, 0.5], # C - FARTHER from ref C2
        ]
        target_species = ["A", "D", "B", "D", "B", "D", "C", "D", "C"]
        target = Structure(lattice, target_species, target_coords)

        # Coordinating atoms
        ref_coordinating = [[1, 2, 3, 4]]  # All B and C atoms

        mapper = IndexMapper()

        # Filter to B and C species - should map to B and C even though D is closer
        mapped_coordinating = mapper.map_coordinating_atoms(
            reference, target, ref_coordinating, target_species=["B", "C"]
        )

        # Should map to B and C atoms (indices 2, 4, 6, 8) despite D being closer
        expected = [[2, 4, 6, 8]]
        self.assertEqual(mapped_coordinating, expected)

    def test_mixed_distance_species_filtering(self):
        """Test that filtering works correctly with mixed distance scenarios.

        Some target species are closest while others have closer non-target species.
        """
        lattice = Lattice.cubic(4.0)

        # Reference structure
        ref_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.0, 0.5, 0.5],  # X1 - for mapping
            [1.0, 0.5, 0.5],  # X2 - for mapping
            [0.5, 0.0, 0.5],  # X3 - for mapping
        ]
        ref_species = ["A", "X", "X", "X"]
        reference = Structure(lattice, ref_species, ref_coords, coords_are_cartesian=True)

        # target structure with mixed distances
        target_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.05, 0.5, 0.5], # Cl - CLOSEST to ref X1, should map here
            [0.15, 0.5, 0.5], # I - farther from ref X1
            [0.9, 0.5, 0.5],  # I - CLOSER to ref X2 than any Cl
            [0.95, 0.5, 0.5], # Cl - FARTHER to ref X2, but should still map here
            [0.5, 0.1, 0.5],  # I - CLOSER to ref X3 than any Cl
            [0.5, 0.05, 0.5], # Cl - CLOSEST to ref X3, should map here
        ]
        target_species = ["A", "Cl", "I", "I", "Cl", "I", "Cl"]
        target = Structure(lattice, target_species, target_coords, coords_are_cartesian=True)

        # Coordinating atoms
        ref_coordinating = [[1, 2, 3]]  # All three X atoms

        mapper = IndexMapper()

        # Filter to only Cl species
        mapped_coordinating = mapper.map_coordinating_atoms(
            reference, target, ref_coordinating, target_species="Cl"
        )

        # Should map to closest Cl atoms (indices 1, 4, 6)
        expected = [[1, 4, 6]]
        self.assertEqual(mapped_coordinating, expected)

    def test_no_valid_mapping_targets_raises_error(self):
        """Test that having no atoms of target species raises appropriate error.

        When the target structure has no atoms of the filtered species,
        appropriate error should be raised (likely from 1:1 mapping constraint).
        """
        lattice = Lattice.cubic(5.0)

        # Reference structure
        ref_coords = [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5]]
        ref_species = ["A", "X"]
        reference = Structure(lattice, ref_species, ref_coords)

        # target structure with no target species
        target_coords = [[0.5, 0.5, 0.5], [0.1, 0.5, 0.5]]
        target_species = ["A", "Y"]  # No X atoms
        target = Structure(lattice, target_species, target_coords)

        # Coordinating atoms
        ref_coordinating = [[1]]  # X atom

        mapper = IndexMapper()

        # Filter to X species which doesn't exist
        with self.assertRaises(ValueError):
            mapper.map_coordinating_atoms(
                reference, target, ref_coordinating, target_species="X"
            )

    def test_insufficient_filtered_atoms_raises_error(self):
        """Test that insufficient filtered atoms causes 1:1 mapping violation.

        When there are fewer filtered atoms than needed for 1:1 mapping.
        """
        lattice = Lattice.cubic(5.0)

        # Reference structure with 3 X atoms
        ref_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.2, 0.5, 0.5],  # X1
            [0.5, 0.5, 0.5],  # X2
            [0.8, 0.5, 0.5],  # X3
        ]
        ref_species = ["A", "X", "X", "X"]
        reference = Structure(lattice, ref_species, ref_coords)

        # target structure with only 2 Cl atoms
        target_coords = [
            [0.5, 0.5, 0.5],  # Center
            [0.25, 0.5, 0.5], # Cl
            [0.75, 0.5, 0.5], # Cl
            [0.5, 0.5, 0.5],  # I - not Cl
        ]
        target_species = ["A", "Cl", "Cl", "I"]
        target = Structure(lattice, target_species, target_coords)

        # Coordinating atoms - need 3 X atoms mapped
        ref_coordinating = [[1, 2, 3]]  # All three X atoms

        mapper = IndexMapper()

        # Filter to Cl species - need 3 but only 2 available
        with self.assertRaises(ValueError):
            mapper.map_coordinating_atoms(
                reference, target, ref_coordinating, target_species="Cl"
            )

if __name__ == '__main__':
    unittest.main()
