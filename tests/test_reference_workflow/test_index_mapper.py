"""Unit tests for the IndexMapper class.

These tests verify that IndexMapper correctly maps coordinating atom indices
from reference structures to target structures while handling various edge cases.
"""

import unittest
import numpy as np

from site_analysis.reference_workflow.index_mapper import IndexMapper


def _cubic_matrix(a: float) -> np.ndarray:
    """Return a cubic lattice matrix with parameter a."""
    return np.array([[a, 0, 0], [0, a, 0], [0, 0, a]], dtype=float)


class TestIndexMapper(unittest.TestCase):
    """Test cases for IndexMapper class."""

    def test_identical_structures_single_coordination(self):
        """Test that identical structures map perfectly.

        When reference and target structures are identical, the mapping
        should preserve all coordinating atom indices.
        """
        lattice_matrix = _cubic_matrix(4.0)
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        ref_coordinating = [[1]]  # One site with atom at index 1

        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=coords,
            target_frac_coords=coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )

        self.assertEqual(mapped_coordinating, ref_coordinating)

    def test_permuted_atoms_within_species(self):
        """Test mapping when coordinating atoms of same species are permuted.

        Tests that IndexMapper correctly identifies and maps permuted
        coordinating atoms of the same species based on closest distance.
        """
        lattice_matrix = _cubic_matrix(5.0)

        # Reference: Cartesian -> fractional for a=5.0
        ref_frac_coords = np.array([
            [0.05, 0.1, 0.1],  # Na
            [0.15, 0.1, 0.1],  # Na
            [0.0, 0.1, 0.1],   # Cl1 - closer to first Na
            [0.2, 0.1, 0.1],   # Cl2 - closer to second Na
        ])

        # Target: Cl atoms permuted in ordering
        target_frac_coords = np.array([
            [0.05, 0.1, 0.1],  # Na - same positions
            [0.15, 0.1, 0.1],  # Na
            [0.2, 0.1, 0.1],   # Cl2 (permuted position)
            [0.0, 0.1, 0.1],   # Cl1 (permuted position)
        ])

        ref_coordinating = [
            [2],  # First Na coordinated by Cl1 at index 2
            [3],  # Second Na coordinated by Cl2 at index 3
        ]

        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )

        expected = [
            [3],  # Cl1 now at index 3
            [2],  # Cl2 now at index 2
        ]
        self.assertEqual(mapped_coordinating, expected)

    def test_permuted_atoms_across_species(self):
        """Test mapping with atoms permuted across species.

        Verifies that coordination environments are correctly mapped
        when different species are reordered in the structure.
        """
        lattice_matrix = _cubic_matrix(4.0)

        ref_frac_coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

        # Target: reversed order
        target_frac_coords = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])

        ref_coordinating = [[1]]  # One site with atom at index 1

        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )

        expected = [[0]]  # Atom now at index 0
        self.assertEqual(mapped_coordinating, expected)

    def test_structural_distortion_closest_mapping(self):
        """Test mapping with small structural distortions.

        Ensures that slightly distorted structures are mapped correctly
        using closest-atom distances.
        """
        lattice_matrix = _cubic_matrix(5.0)

        # Reference structure (ideal) - Cartesian / 5.0 = fractional
        ref_frac_coords = np.array([
            [0.1, 0.1, 0.1],  # A (central)
            [0.0, 0.1, 0.1],  # B1
            [0.2, 0.1, 0.1],  # B2
            [0.1, 0.0, 0.1],  # B3
        ])

        # Target structure (slightly distorted)
        target_frac_coords = np.array([
            [0.104, 0.096, 0.1],  # A (distorted)
            [0.004, 0.096, 0.1],  # B1 (distorted)
            [0.196, 0.104, 0.1],  # B2 (distorted)
            [0.096, 0.004, 0.1],  # B3 (distorted)
        ])

        ref_coordinating = [[1, 2, 3]]  # A coordinated by all three B atoms

        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )

        expected = [[1, 2, 3]]
        self.assertEqual(mapped_coordinating, expected)

    def test_multiple_coordination_atoms(self):
        """Test mapping multiple coordinating atoms per site.

        Verifies correct mapping when sites have multiple coordinating atoms.
        """
        lattice_matrix = _cubic_matrix(6.0)

        # Perovskite-like structure
        frac_coords = np.array([
            [0.0, 0.0, 0.0],  # A (central)
            [0.5, 0.0, 0.0],  # X
            [0.0, 0.5, 0.0],  # X
            [0.0, 0.0, 0.5],  # X
        ])

        ref_coordinating = [[1, 2, 3]]  # One site coordinated by all X atoms

        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=frac_coords,
            target_frac_coords=frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )

        self.assertEqual(mapped_coordinating, ref_coordinating)

    def test_missing_coordinating_atom_raises_error(self):
        """Test that 1:1 mapping violations raise appropriate errors.

        When multiple reference coordinating atoms map to the same atom
        in the target structure, mapping should fail.
        """
        lattice_matrix = _cubic_matrix(5.0)

        # Reference structure - two distinct B atoms
        ref_frac_coords = np.array([
            [0.0, 0.0, 0.0],  # A
            [0.2, 0.0, 0.0],  # B1
            [0.8, 0.0, 0.0],  # B2
        ])

        # Target structure - only one B atom, closest to both
        target_frac_coords = np.array([
            [0.0, 0.0, 0.0],  # A
            [0.5, 0.0, 0.0],  # B (closest to both ref B1 and B2)
        ])

        ref_coordinating = [[1], [2]]

        mapper = IndexMapper()

        with self.assertRaises(ValueError):
            mapper.map_coordinating_atoms(
                ref_frac_coords=ref_frac_coords,
                target_frac_coords=target_frac_coords,
                lattice_matrix=lattice_matrix,
                ref_coordinating=ref_coordinating,
            )

    def test_periodic_boundary_conditions(self):
        """Test correct mapping across periodic boundaries.

        Ensures atoms that are closest across PBC are mapped correctly,
        verifies both species-blind and species-aware mapping.
        """
        lattice_matrix = _cubic_matrix(3.0)

        ref_frac_coords = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]])

        # Target structure with multiple options
        target_frac_coords = np.array([
            [0.0, 0.0, 0.0],    # A atom - closest to ref B with PBC
            [0.05, 0.0, 0.0],   # B atom - closest B to ref B with PBC
            [0.2, 0.0, 0.0],    # B atom - further away
        ])
        target_species = ["A", "B", "B"]

        ref_coordinating = [[1]]

        mapper = IndexMapper()

        # Test 1: Without species filtering - should map to closest atom (A)
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )
        expected = [[0]]  # A atom at index 0
        self.assertEqual(mapped_coordinating, expected)

        # Test 2: With species filtering - should map to closest B atom
        mapped_with_filter = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
            target_species=target_species,
            species_filter="B",
        )
        expected_with_filter = [[1]]  # B atom at index 1
        self.assertEqual(mapped_with_filter, expected_with_filter)

    def test_complex_coordination_environment(self):
        """Test mapping complex coordination (e.g., octahedral).

        Tests correct mapping for sites with multiple coordinating atoms.
        """
        lattice_matrix = _cubic_matrix(4.0)

        # Reference: octahedral coordination (Cartesian / 4.0 = fractional)
        ref_frac_coords = np.array([
            [0.125, 0.125, 0.125],  # A (central)
            [0.0, 0.125, 0.125],    # X (-x)
            [0.25, 0.125, 0.125],   # X (+x)
            [0.125, 0.0, 0.125],    # X (-y)
            [0.125, 0.25, 0.125],   # X (+y)
            [0.125, 0.125, 0.0],    # X (-z)
            [0.125, 0.125, 0.25],   # X (+z)
            [0.25, 0.25, 0.25],     # X (not coordinated)
            [0.0, 0.125, 0.25],     # X (not coordinated)
        ])

        # Target: atoms permuted
        target_frac_coords = np.array([
            [0.125, 0.125, 0.125],  # A (central)
            [0.25, 0.125, 0.125],   # X (+x) - permuted
            [0.0, 0.125, 0.125],    # X (-x) - permuted
            [0.25, 0.25, 0.25],     # X (not coordinated)
            [0.125, 0.25, 0.125],   # X (+y) - permuted
            [0.125, 0.0, 0.125],    # X (-y) - permuted
            [0.125, 0.125, 0.25],   # X (+z) - permuted
            [0.0, 0.125, 0.25],     # X (not coordinated)
            [0.125, 0.125, 0.0],    # X (-z) - permuted
        ])

        ref_coordinating = [[1, 2, 3, 4, 5, 6]]

        mapper = IndexMapper()
        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
        )

        expected = [[2, 1, 5, 4, 8, 6]]
        self.assertEqual(mapped_coordinating, expected)

    def test_detailed_error_message(self):
        """Test that 1:1 mapping violation produces detailed error message.

        Ensures the error message contains information about which specific
        target atoms were mapped to by multiple reference atoms.
        """
        lattice_matrix = _cubic_matrix(5.0)

        # Reference: 3 atoms (Cartesian / 5.0 = fractional)
        ref_frac_coords = np.array([
            [0.02, 0.1, 0.1],  # Reference atom 0
            [0.06, 0.1, 0.1],  # Reference atom 1
            [0.1, 0.1, 0.1],   # Reference atom 2
        ])

        # Target: only 2 atoms, forcing violation
        target_frac_coords = np.array([
            [0.04, 0.1, 0.1],  # Target atom 0 - maps to ref 0 and 1
            [0.1, 0.1, 0.1],   # Target atom 1 - maps to ref 2
        ])

        ref_coordinating = [[0, 1, 2]]

        mapper = IndexMapper()

        try:
            mapper.map_coordinating_atoms(
                ref_frac_coords=ref_frac_coords,
                target_frac_coords=target_frac_coords,
                lattice_matrix=lattice_matrix,
                ref_coordinating=ref_coordinating,
            )
            self.fail("Expected ValueError was not raised")
        except ValueError as e:
            error_msg = str(e)
            self.assertIn("1:1 mapping violation", error_msg)
            self.assertIn("indices [0]", error_msg)


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
        lattice_matrix = _cubic_matrix(5.0)

        # Reference structure (Cartesian / 5.0 = fractional)
        ref_frac_coords = np.array([
            [0.1, 0.1, 0.1],   # Centre
            [0.0, 0.1, 0.1],   # X1
            [0.2, 0.1, 0.1],   # X2
        ])

        # Target structure - Cl is farther than Br
        target_frac_coords = np.array([
            [0.1, 0.1, 0.1],   # Centre
            [0.02, 0.1, 0.1],  # Br - CLOSER to ref X1
            [0.06, 0.1, 0.1],  # Cl - FARTHER from ref X1
            [0.16, 0.1, 0.1],  # Br - CLOSER to ref X2
            [0.14, 0.1, 0.1],  # Cl - FARTHER from ref X2
        ])
        target_species = ["A", "Br", "Cl", "Br", "Cl"]

        ref_coordinating = [[1, 2]]

        mapper = IndexMapper()

        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
            target_species=target_species,
            species_filter="Cl",
        )

        expected = [[2, 4]]
        self.assertEqual(mapped_coordinating, expected)

    def test_multiple_species_filtering_with_closer_non_target(self):
        """Test multiple species filtering when non-target atoms are closer.

        Ensures that atoms of any target species are preferred over
        geometrically closer atoms of non-target species.
        """
        lattice_matrix = _cubic_matrix(6.0)

        ref_frac_coords = np.array([
            [0.5, 0.5, 0.5],  # Centre
            [0.2, 0.5, 0.5],  # B1
            [0.8, 0.5, 0.5],  # B2
            [0.5, 0.2, 0.5],  # C1
            [0.5, 0.8, 0.5],  # C2
        ])

        # Target: D atoms closest but not in species filter
        target_frac_coords = np.array([
            [0.5, 0.5, 0.5],   # Centre
            [0.15, 0.5, 0.5],  # D - CLOSEST to ref B1
            [0.25, 0.5, 0.5],  # B - FARTHER from ref B1
            [0.75, 0.5, 0.5],  # D - CLOSEST to ref B2
            [0.85, 0.5, 0.5],  # B - FARTHER from ref B2
            [0.5, 0.15, 0.5],  # D - CLOSEST to ref C1
            [0.5, 0.25, 0.5],  # C - FARTHER from ref C1
            [0.5, 0.75, 0.5],  # D - CLOSEST to ref C2
            [0.5, 0.85, 0.5],  # C - FARTHER from ref C2
        ])
        target_species = ["A", "D", "B", "D", "B", "D", "C", "D", "C"]

        ref_coordinating = [[1, 2, 3, 4]]

        mapper = IndexMapper()

        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
            target_species=target_species,
            species_filter=["B", "C"],
        )

        expected = [[2, 4, 6, 8]]
        self.assertEqual(mapped_coordinating, expected)

    def test_mixed_distance_species_filtering(self):
        """Test that filtering works correctly with mixed distance scenarios.

        Some target species are closest while others have closer non-target species.
        """
        lattice_matrix = _cubic_matrix(4.0)

        # Reference structure (Cartesian / 4.0 = fractional)
        ref_frac_coords = np.array([
            [0.125, 0.125, 0.125],  # Centre
            [0.0, 0.125, 0.125],    # X1
            [0.25, 0.125, 0.125],   # X2
            [0.125, 0.0, 0.125],    # X3
        ])

        # Target structure with mixed distances
        target_frac_coords = np.array([
            [0.125, 0.125, 0.125],    # Centre
            [0.0125, 0.125, 0.125],   # Cl - CLOSEST to ref X1
            [0.0375, 0.125, 0.125],   # I - farther from ref X1
            [0.225, 0.125, 0.125],    # I - CLOSER to ref X2
            [0.2375, 0.125, 0.125],   # Cl - FARTHER to ref X2
            [0.125, 0.025, 0.125],    # I - CLOSER to ref X3
            [0.125, 0.0125, 0.125],   # Cl - CLOSEST to ref X3
        ])
        target_species = ["A", "Cl", "I", "I", "Cl", "I", "Cl"]

        ref_coordinating = [[1, 2, 3]]

        mapper = IndexMapper()

        mapped_coordinating = mapper.map_coordinating_atoms(
            ref_frac_coords=ref_frac_coords,
            target_frac_coords=target_frac_coords,
            lattice_matrix=lattice_matrix,
            ref_coordinating=ref_coordinating,
            target_species=target_species,
            species_filter="Cl",
        )

        expected = [[1, 4, 6]]
        self.assertEqual(mapped_coordinating, expected)

    def test_no_valid_mapping_targets_raises_error(self):
        """Test that having no atoms of target species raises appropriate error."""
        lattice_matrix = _cubic_matrix(5.0)

        ref_frac_coords = np.array([[0.5, 0.5, 0.5], [0.0, 0.5, 0.5]])

        target_frac_coords = np.array([[0.5, 0.5, 0.5], [0.1, 0.5, 0.5]])
        target_species = ["A", "Y"]  # No X atoms

        ref_coordinating = [[1]]

        mapper = IndexMapper()

        with self.assertRaises(ValueError):
            mapper.map_coordinating_atoms(
                ref_frac_coords=ref_frac_coords,
                target_frac_coords=target_frac_coords,
                lattice_matrix=lattice_matrix,
                ref_coordinating=ref_coordinating,
                target_species=target_species,
                species_filter="X",
            )

    def test_insufficient_filtered_atoms_raises_error(self):
        """Test that insufficient filtered atoms causes 1:1 mapping violation."""
        lattice_matrix = _cubic_matrix(5.0)

        ref_frac_coords = np.array([
            [0.5, 0.5, 0.5],  # Centre
            [0.2, 0.5, 0.5],  # X1
            [0.5, 0.5, 0.5],  # X2
            [0.8, 0.5, 0.5],  # X3
        ])

        target_frac_coords = np.array([
            [0.5, 0.5, 0.5],   # Centre
            [0.25, 0.5, 0.5],  # Cl
            [0.75, 0.5, 0.5],  # Cl
            [0.5, 0.5, 0.5],   # I - not Cl
        ])
        target_species = ["A", "Cl", "Cl", "I"]

        ref_coordinating = [[1, 2, 3]]

        mapper = IndexMapper()

        with self.assertRaises(ValueError):
            mapper.map_coordinating_atoms(
                ref_frac_coords=ref_frac_coords,
                target_frac_coords=target_frac_coords,
                lattice_matrix=lattice_matrix,
                ref_coordinating=ref_coordinating,
                target_species=target_species,
                species_filter="Cl",
            )


if __name__ == '__main__':
    unittest.main()
