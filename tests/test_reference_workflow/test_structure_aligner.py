"""Unit tests for the StructureAligner class.

These tests verify that StructureAligner finds a translation that
superimposes a reference structure on a target structure.
"""

import math
import unittest

import numpy as np
from pymatgen.core import Lattice, Structure

import site_analysis.reference_workflow.structure_aligner as aligner_module
from site_analysis.distances import paired_mic_distances
from site_analysis.reference_workflow.structure_aligner import StructureAligner


def align_structures(reference, target, species=None):
    """Align two pymatgen structures through the array interface."""
    return StructureAligner().align(
        reference.frac_coords, [site.species_string for site in reference],
        target.frac_coords, [site.species_string for site in target],
        reference.lattice.matrix, species=species)


def aligned_coords(reference, translation):
    """Reference fractional coordinates after the translation."""
    return (reference.frac_coords + translation) % 1.0


def nearest_same_species_distances(frac_coords, species, target):
    """Distance from each point to the nearest target atom of its species."""
    target_species = np.array([site.species_string for site in target])
    distances = []
    for coords, name in zip(frac_coords, species):
        others = target.frac_coords[target_species == name]
        distances.append(paired_mic_distances(
            np.repeat(coords[np.newaxis], len(others), axis=0), others,
            target.lattice.matrix).min())
    return np.array(distances)


def with_noise(structure, scale, seed, shift=(0.0, 0.0, 0.0)):
    """The structure shifted (fractional) and with Gaussian noise (A) added."""
    rng = np.random.default_rng(seed)
    cartesian = (((structure.frac_coords + shift) % 1.0) @ structure.lattice.matrix
                 + rng.normal(scale=scale, size=(len(structure), 3)))
    return Structure(structure.lattice, structure.species, cartesian,
                     coords_are_cartesian=True)


def rock_salt(supercell=(3, 3, 3), a=5.64):
    """A rock-salt NaCl supercell."""
    unit = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return unit * supercell


def spinel(supercell=(1, 1, 1)):
    """A LiMn2O4 spinel supercell (origin choice 1, oxygen x = 0.388).

    The oxygen sublattice is a slightly distorted cubic close packing, so
    some translations that are not symmetries superimpose it to within
    about 0.3 A.
    """
    unit = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(8.25), ["Li", "Mn", "O"],
        [[0.0, 0.0, 0.0], [0.625, 0.625, 0.625], [0.388, 0.388, 0.388]])
    return unit * supercell


class TestStructureAligner(unittest.TestCase):
    """Test cases for the StructureAligner class."""

    def test_basic_translation(self):
        """A uniform translation is recovered exactly."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        translation = [0.1, 0.2, 0.3]
        target = Structure(lattice, reference.species, reference.frac_coords + translation)
        found, metrics = align_structures(reference, target)
        np.testing.assert_allclose(found, translation, atol=1e-10)
        self.assertLess(metrics['rmsd'], 1e-10)

    def test_species_based_alignment(self):
        """Only the alignment species decide the translation."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl", "K"],
                              [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
        # Na and K shifted by 0.1, Cl by 0.2.
        target = Structure(lattice, reference.species,
                           [[0.1, 0.1, 0.1], [0.7, 0.7, 0.7], [0.35, 0.35, 0.35]])
        found, _ = align_structures(reference, target, species=["Na", "K"])
        np.testing.assert_allclose(found, [0.1, 0.1, 0.1], atol=1e-10)

    def test_different_number_of_atoms(self):
        """Different counts fail for all species, but not for a species whose counts match."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        target = Structure(lattice, ["Na", "Cl", "Na"],
                           [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3]])
        with self.assertRaises(ValueError):
            align_structures(reference, target)
        found, _ = align_structures(reference, target, species=["Cl"])
        np.testing.assert_allclose(found, [0.1, 0.1, 0.1], atol=1e-10)

    def test_different_compositions(self):
        """Structures with different species cannot be aligned on all species."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        target = Structure(lattice, ["K", "Br"], [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]])
        with self.assertRaises(ValueError):
            align_structures(reference, target)

    def test_minimises_rmsd_for_a_non_uniform_shift(self):
        """With atoms shifted by different amounts, the translation is their mean shift."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl", "Na", "Cl"],
                              [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
                               [0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
        # Shifted by 0.1, 0.15, 0.15 and 0.15 along each axis.
        target = Structure(lattice, reference.species,
                           [[0.1, 0.1, 0.1], [0.65, 0.65, 0.65],
                            [0.35, 0.35, 0.35], [0.85, 0.85, 0.85]])
        found, metrics = align_structures(reference, target)
        np.testing.assert_allclose(found, [0.1375] * 3, atol=1e-12)
        self.assertEqual(set(metrics), {'rmsd', 'max_dist', 'mean_dist'})

    def test_pbc_handling(self):
        """A translation across the cell boundary is found."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl"], [[0.05, 0.05, 0.05], [0.55, 0.55, 0.55]])
        target = Structure(lattice, ["Na", "Cl"], [[0.95, 0.95, 0.95], [0.45, 0.45, 0.45]])
        found, _ = align_structures(reference, target)
        species = [site.species_string for site in reference]
        distances = nearest_same_species_distances(
            aligned_coords(reference, found), species, target)
        self.assertLess(distances.max(), 1e-10)

    def test_half_cell_offset(self):
        """A half-cell offset of a symmetric structure is superimposed exactly."""
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"]
        coords = np.array([
            [0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5],
        ])
        reference = Structure(lattice, species, coords)
        target = Structure(lattice, species, (coords + 0.5) % 1.0)
        found, metrics = align_structures(reference, target)
        distances = nearest_same_species_distances(
            aligned_coords(reference, found), species, target)
        self.assertLess(distances.max(), 1e-10)
        self.assertLess(metrics['rmsd'], 1e-10)

    def test_align_with_permuted_atoms(self):
        """Atom order does not matter."""
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Cl", "Na", "Cl"]
        coords = [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3], [0.8, 0.8, 0.8]]
        reference = Structure(lattice, species, coords)
        target = Structure(lattice, species[::-1], coords[::-1])
        found, metrics = align_structures(reference, target)
        self.assertLess(metrics['rmsd'], 1e-10)
        np.testing.assert_allclose(found, [0.0, 0.0, 0.0], atol=1e-10)

    def test_align_with_permuted_atoms_and_translation(self):
        """Atom order does not matter when the target is also translated."""
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Cl", "Na", "Cl"]
        coords = np.array([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3], [0.8, 0.8, 0.8]])
        reference = Structure(lattice, species, coords)
        target = Structure(lattice, species[::-1], ((coords + 0.1) % 1.0)[::-1])
        found, metrics = align_structures(reference, target)
        self.assertLess(metrics['rmsd'], 1e-10)
        np.testing.assert_allclose(found, [0.1, 0.1, 0.1], atol=1e-10)

    def test_validate_structures(self):
        """Test the _validate_structures method with various scenarios."""
        species_a = ["Na", "Na", "Cl", "Cl"]
        species_b = ["Cl", "Na", "Cl", "Na"]
        species_c = ["Na", "Na", "Na", "Cl", "Cl"]
        species_d = ["Na", "Na", "Cl", "Cl", "F"]
        aligner = StructureAligner()
        self.assertCountEqual(aligner._validate_structures(species_a, species_b, None), ["Cl", "Na"])
        with self.assertRaisesRegex(ValueError, "different compositions"):
            aligner._validate_structures(species_a, species_c, None)
        with self.assertRaisesRegex(ValueError, "different compositions"):
            aligner._validate_structures(species_a, species_d, None)
        self.assertEqual(aligner._validate_structures(species_a, species_b, ["Na"]), ["Na"])
        with self.assertRaisesRegex(ValueError, "not found in reference"):
            aligner._validate_structures(species_a, species_d, ["F"])
        with self.assertRaisesRegex(ValueError, "not found in target"):
            aligner._validate_structures(species_d, species_a, ["F"])
        with self.assertRaisesRegex(ValueError, "Different number of Na atoms"):
            aligner._validate_structures(species_a, species_c, ["Na"])
        # A species named twice is used once
        self.assertEqual(aligner._validate_structures(species_a, species_b, ["Na", "Na"]), ["Na"])

    def test_mismatched_coordinates_and_species_raise(self):
        """Each structure needs one species per atom."""
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "reference has 2 coordinates but 1 species"):
            StructureAligner().align(coords, ["Na"], coords, ["Na", "Cl"], np.eye(3) * 5.0)
        with self.assertRaisesRegex(ValueError, "target has 2 coordinates but 3 species"):
            StructureAligner().align(coords, ["Na", "Cl"], coords, ["Na", "Cl", "Cl"], np.eye(3) * 5.0)

    def test_empty_species_raises(self):
        """An empty species list is rejected."""
        structure = Structure(Lattice.cubic(5.0), ["Na", "Cl"],
                              [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "No species to align on"):
            align_structures(structure, structure.copy(), species=[])


class TestCandidateSearch(unittest.TestCase):
    """The search finds origin offsets of any size."""

    def assert_superimposed(self, reference, target, translation, tolerance):
        """Every translated reference atom is near a target atom of its species."""
        species = [site.species_string for site in reference]
        distances = nearest_same_species_distances(
            aligned_coords(reference, translation), species, target)
        self.assertLess(distances.max(), tolerance)

    def test_half_cell_offset_in_a_supercell(self):
        """A half-cell offset in a rock-salt supercell is found."""
        reference = rock_salt()
        target = with_noise(reference, scale=0.0, seed=0, shift=(0.5 / 3, 0.0, 0.0))
        found, metrics = align_structures(reference, target)
        self.assert_superimposed(reference, target, found, 1e-6)
        self.assertLess(metrics['rmsd'], 1e-6)

    def test_arbitrary_offset_in_a_triclinic_cell(self):
        """An arbitrary offset in a triclinic cell with thermal noise is recovered."""
        unit = Structure(Lattice.from_parameters(4.0, 4.5, 5.0, 80, 95, 105),
                         ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        reference = unit * (3, 3, 3)
        target = with_noise(reference, scale=0.05, seed=1, shift=(0.37, 0.12, 0.81))
        found, metrics = align_structures(reference, target)
        self.assert_superimposed(reference, target, found, 0.3)
        self.assertLess(metrics['rmsd'], 0.1)

    def test_arbitrary_offset_in_spinel(self):
        """An arbitrary offset in spinel is found (Nelder-Mead from zero gets stuck)."""
        reference = spinel()
        target = with_noise(reference, scale=0.05, seed=5, shift=(0.3, 0.1, 0.2))
        found, metrics = align_structures(reference, target)
        self.assertLess(metrics['rmsd'], 0.15)

    def test_arbitrary_offset_of_a_random_framework(self):
        """An offset of an irregular framework, with many false minima, is found."""
        rng = np.random.default_rng(6)
        points = []
        while len(points) < 60:
            point = rng.random(3)
            if all(np.linalg.norm(((point - other + 0.5) % 1.0 - 0.5) * 10.0) > 2.0
                   for other in points):
                points.append(point)
        reference = Structure(Lattice.cubic(10.0), ["O"] * 60, points)
        target = with_noise(reference, scale=0.05, seed=7, shift=(0.37, 0.12, 0.29))
        found, metrics = align_structures(reference, target)
        self.assertLess(metrics['rmsd'], 0.15)

    def test_near_symmetry_is_not_mistaken_for_the_alignment(self):
        """The best fit wins over a translation that only nearly superimposes the atoms.

        Spinel's oxygens are nearly close-packed: some translations that
        are not symmetries superimpose them to within about 0.3 A RMSD,
        well within the threshold, but move every lithium site onto an
        empty tetrahedron.
        """
        reference = spinel((2, 2, 2))
        target = with_noise(reference, scale=0.05, seed=5, shift=(0.0625, 0.0625, 0.0625))
        found, metrics = align_structures(reference, target, species=["O"])
        self.assertLess(metrics['rmsd'], 0.15)
        lithium = [i for i, site in enumerate(reference) if site.species_string == "Li"]
        distances = nearest_same_species_distances(
            aligned_coords(reference, found)[lithium], ["Li"] * len(lithium), target)
        self.assertLess(distances.max(), 0.3)

    def test_a_displaced_anchor_does_not_prevent_alignment(self):
        """The anchor's own displacement does not count against the candidates."""
        reference = rock_salt((2, 2, 2))
        sodium = [i for i, site in enumerate(reference) if site.species_string == "Na"]
        threshold = aligner_module._threshold(
            reference.frac_coords[sodium], reference.lattice.matrix)
        # The anchor is the first target Na. Displacing it and a sampled Na
        # in opposite directions puts that Na 1.2 thresholds from its
        # partner at every candidate from the anchor.
        sampled = aligner_module._sample_indices(len(sodium))[1]
        cartesian = reference.cart_coords.copy()
        cartesian[sodium[0]] += [0.6 * threshold, 0.0, 0.0]
        cartesian[sodium[sampled]] -= [0.6 * threshold, 0.0, 0.0]
        target = Structure(reference.lattice, reference.species, cartesian,
                           coords_are_cartesian=True)
        _, metrics = align_structures(reference, target, species=["Na"])
        self.assertLess(metrics['max_dist'], threshold)

    def test_no_superimposing_translation_raises(self):
        """A target atom far from every site leaves no translation that superimposes the structures."""
        reference = rock_salt((2, 2, 2))
        target = reference.copy()
        sodium = next(i for i, site in enumerate(target) if site.species_string == "Na")
        target.replace(sodium, "Na", (target[sodium].frac_coords + [0.25, 0.0, 0.0]) % 1.0)
        with self.assertRaisesRegex(
                ValueError, r"\(Na: .*\)\. Align on species whose atoms stay on their sites"):
            align_structures(reference, target, species=["Na"])

    def test_same_input_gives_the_same_result(self):
        """Alignment is deterministic."""
        reference = rock_salt((2, 2, 2))
        target = with_noise(reference, scale=0.1, seed=3)
        first, _ = align_structures(reference, target)
        second, _ = align_structures(reference, target)
        np.testing.assert_array_equal(first, second)


class TestDeprecatedAlignmentOptions(unittest.TestCase):
    """The deprecated options are named together in one warning."""

    def test_several_options_are_named_together(self):
        """Two options give one warning naming both."""
        with self.assertWarnsRegex(
                FutureWarning, r"align_algorithm, align_tolerance have no effect .* delete them"):
            aligner_module._check_deprecated_alignment_options(
                align_algorithm="Nelder-Mead", align_tolerance=1e-4)


class TestThreshold(unittest.TestCase):
    """The per-species threshold is half the shortest same-species spacing."""

    def test_half_the_shortest_spacing(self):
        """In rock salt, half the Na-Na spacing, a / (2 sqrt 2)."""
        reference = rock_salt((2, 2, 2), a=4.0)
        sodium = [i for i, site in enumerate(reference) if site.species_string == "Na"]
        self.assertAlmostEqual(
            aligner_module._threshold(reference.frac_coords[sodium], reference.lattice.matrix),
            4.0 / (2 * math.sqrt(2)), places=12)

    def test_one_atom_has_no_threshold(self):
        """A single atom maps one to one under any translation."""
        self.assertEqual(aligner_module._threshold(np.zeros((1, 3)), np.eye(3)), math.inf)

    def test_an_atom_nearer_its_own_image_than_the_others(self):
        """In a thin cell the search widens past the atom's own images."""
        self.assertAlmostEqual(aligner_module._threshold(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]), np.diag([1.0, 1.0, 20.0])),
            5.0, places=12)


if __name__ == "__main__":
    unittest.main()
