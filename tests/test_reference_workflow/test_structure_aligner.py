"""Unit tests for the StructureAligner class.

These tests verify that StructureAligner finds a translation that
superimposes a reference structure on a target structure.
"""

import math
import unittest
from unittest.mock import patch

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


def assert_same_translation(found, expected):
    """The translations agree up to whole lattice vectors."""
    difference = np.asarray(found) - np.asarray(expected)
    np.testing.assert_allclose(difference - np.round(difference), 0.0, atol=1e-10)


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
    """The structure shifted (fractional), with Gaussian noise (A) added and its atoms shuffled."""
    rng = np.random.default_rng(seed)
    cartesian = (((structure.frac_coords + shift) % 1.0) @ structure.lattice.matrix
                 + rng.normal(scale=scale, size=(len(structure), 3)))
    order = rng.permutation(len(structure))
    return Structure(structure.lattice, [structure.species[i] for i in order],
                     cartesian[order], coords_are_cartesian=True)


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


def random_framework(seed=6, n_atoms=60, a=10.0, min_separation=2.0):
    """O atoms at random positions at least min_separation apart: no translational symmetry."""
    rng = np.random.default_rng(seed)
    points = []
    while len(points) < n_atoms:
        point = rng.random(3)
        if all(np.linalg.norm(((point - other + 0.5) % 1.0 - 0.5) * a) > min_separation
               for other in points):
            points.append(point)
    return Structure(Lattice.cubic(a), ["O"] * n_atoms, points)


class TestStructureAligner(unittest.TestCase):
    """Test cases for the StructureAligner class."""

    def test_basic_translation(self):
        """A uniform translation is recovered exactly."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        translation = [0.1, 0.2, 0.3]
        target = Structure(lattice, reference.species, reference.frac_coords + translation)
        found, metrics = align_structures(reference, target)
        assert_same_translation(found, translation)
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
        assert_same_translation(found, [0.1, 0.1, 0.1])

    def test_different_number_of_atoms(self):
        """Different counts fail for all species, but not for a species whose counts match."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Cl"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        target = Structure(lattice, ["Na", "Cl", "Na"],
                           [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3]])
        with self.assertRaises(ValueError):
            align_structures(reference, target)
        found, _ = align_structures(reference, target, species=["Cl"])
        assert_same_translation(found, [0.1, 0.1, 0.1])

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
        assert_same_translation(found, [0.1375] * 3)
        self.assertEqual(set(metrics), {'rmsd', 'max_dist', 'mean_dist'})
        distances = np.array([0.0375, 0.0125, 0.0125, 0.0125]) * 5.0 * math.sqrt(3.0)
        self.assertAlmostEqual(metrics['rmsd'], math.sqrt(np.mean(distances ** 2)), places=12)
        self.assertAlmostEqual(metrics['max_dist'], distances.max(), places=12)
        self.assertAlmostEqual(metrics['mean_dist'], distances.mean(), places=12)

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
        assert_same_translation(found, [0.0, 0.0, 0.0])

    def test_align_with_permuted_atoms_and_translation(self):
        """Atom order does not matter when the target is also translated."""
        lattice = Lattice.cubic(5.0)
        species = ["Na", "Cl", "Na", "Cl"]
        coords = np.array([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6], [0.3, 0.3, 0.3], [0.8, 0.8, 0.8]])
        reference = Structure(lattice, species, coords)
        target = Structure(lattice, species[::-1], ((coords + 0.1) % 1.0)[::-1])
        found, metrics = align_structures(reference, target)
        self.assertLess(metrics['rmsd'], 1e-10)
        assert_same_translation(found, [0.1, 0.1, 0.1])

    def test_validate_structures(self):
        """Test the _validate_structures method with various scenarios."""
        species_a = ["Na", "Na", "Cl", "Cl"]
        species_b = ["Cl", "Na", "Cl", "Na"]
        species_c = ["Na", "Na", "Na", "Cl", "Cl"]
        species_d = ["Na", "Na", "Cl", "Cl", "F"]
        aligner = StructureAligner()
        self.assertCountEqual(aligner._validate_structures(species_a, species_b, None), ["Cl", "Na"])
        with self.assertRaisesRegex(ValueError, r"different compositions.*Pass the alignment species"):
            aligner._validate_structures(species_a, species_c, None)
        with self.assertRaisesRegex(ValueError, r"different compositions.*Pass the alignment species"):
            aligner._validate_structures(species_a, species_d, None)
        self.assertEqual(aligner._validate_structures(species_a, species_b, ["Na"]), ["Na"])
        with self.assertRaisesRegex(ValueError, r"Species 'F' not found in reference structure \(it has Cl, Na\)"):
            aligner._validate_structures(species_a, species_d, ["F"])
        with self.assertRaisesRegex(ValueError, r"Species 'F' not found in target structure \(it has Cl, Na\)"):
            aligner._validate_structures(species_d, species_a, ["F"])
        with self.assertRaisesRegex(ValueError, r"Different number of Na atoms.*Align on species present in equal numbers"):
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

    def test_coincident_reference_atoms_raise(self):
        """Two reference atoms of an alignment species at the same position cannot be matched one to one."""
        lattice = Lattice.cubic(5.0)
        reference = Structure(lattice, ["Na", "Na", "Cl"],
                              [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        with self.assertRaisesRegex(ValueError, "coincident Na atoms"):
            align_structures(reference, reference.copy())

    def test_species_as_a_string(self):
        """A single species may be given as a string."""
        reference = rock_salt((2, 2, 2))
        target = with_noise(reference, scale=0.05, seed=9, shift=(0.1, 0.0, 0.0))
        as_string, _ = align_structures(reference, target, species="Na")
        as_list, _ = align_structures(reference, target, species=["Na"])
        np.testing.assert_array_equal(as_string, as_list)


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
        reference = random_framework()
        target = with_noise(reference, scale=0.05, seed=7, shift=(0.37, 0.12, 0.29))
        found, metrics = align_structures(reference, target)
        self.assertLess(metrics['rmsd'], 0.15)

    def test_refinement_finds_the_least_squares_translation(self):
        """With every atom paired with its own image, the result is the shift plus the mean noise."""
        reference = random_framework()
        rng = np.random.default_rng(11)
        noise = rng.normal(scale=0.05, size=(len(reference), 3))
        shift = np.array([0.37, 0.12, 0.29])
        target = Structure(reference.lattice, reference.species,
                           ((reference.frac_coords + shift) % 1.0) @ reference.lattice.matrix + noise,
                           coords_are_cartesian=True)
        found, _ = align_structures(reference, target)
        expected = shift + reference.lattice.get_fractional_coords(noise).mean(axis=0)
        assert_same_translation(found, expected)

    def test_translation_is_wrapped_into_the_unit_cell(self):
        """Every component of the translation is in [0, 1)."""
        reference = rock_salt((2, 2, 2))
        for seed in range(10):
            with self.subTest(seed=seed):
                found, _ = align_structures(reference, with_noise(reference, scale=0.1, seed=seed))
                self.assertTrue(((found >= 0.0) & (found < 1.0)).all(), found)

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
        # This relies on the anchor being the first target Na, and on
        # _sample_indices to pick a sampled atom other than the anchor (with
        # rock_salt((2, 2, 2)) all 32 Na are sampled), so a change to either
        # needs this test updated. Displacing the anchor and a sampled Na in
        # opposite directions puts that Na 1.2 thresholds from its partner
        # at every candidate from the anchor.
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
                ValueError, r"Could not superimpose the reference on the target using Na\. Try aligning only on immobile framework species \(align_species\), and check that both structures use the same cell and orientation\. See \"Alignment Errors\" in the reference workflow guide\.$"):
            align_structures(reference, target, species=["Na"])

    def test_a_misplaced_atom_outside_the_sample_raises(self):
        """The full check catches a misplaced atom that the screen's sample does not include."""
        reference = rock_salt()
        sodium = [i for i, site in enumerate(reference) if site.species_string == "Na"]
        sampled = set(aligner_module._sample_indices(len(sodium)).tolist())
        unsampled = next(k for k in range(len(sodium)) if k not in sampled)
        target = reference.copy()
        index = sodium[unsampled]
        # Onto a Cl site: a / 2 from its nearest Na sites, beyond the Na threshold a / (2 sqrt 2).
        target.replace(index, "Na", (target[index].frac_coords + [0.5 / 3, 0.0, 0.0]) % 1.0)
        with self.assertRaisesRegex(
                ValueError, r"Could not superimpose the reference on the target using Na\. Try aligning only on immobile framework species"):
            align_structures(reference, target, species=["Na"])

    def test_each_species_is_held_to_its_own_threshold(self):
        """An atom beyond its own species' threshold fails, even within another species' threshold."""
        # O on a simple cubic grid 3 A apart (threshold 1.5 A); Ti on a grid 6 A apart (threshold 3 A).
        lattice = Lattice.cubic(12.0)
        oxygen = [[i / 4, j / 4, k / 4] for i in range(4) for j in range(4) for k in range(4)]
        titanium = [[(i + 0.25) / 2, (j + 0.25) / 2, (k + 0.25) / 2]
                    for i in range(2) for j in range(2) for k in range(2)]
        reference = Structure(lattice, ["O"] * 64 + ["Ti"] * 8, oxygen + titanium)
        target = reference.copy()
        # One O moved 2 A along a body diagonal: beyond 1.5 A, within 3 A.
        target.translate_sites([0], np.full(3, 2.0 / math.sqrt(3)),
                               frac_coords=False, to_unit_cell=True)
        with self.assertRaisesRegex(ValueError, r"Could not superimpose the reference on the target using O, Ti\."):
            align_structures(reference, target)

    def test_screening_in_small_blocks_gives_the_same_result(self):
        """The block size bounds memory only."""
        reference = rock_salt()
        target = with_noise(reference, scale=0.05, seed=10, shift=(0.5 / 3, 0.0, 0.0))
        expected, _ = align_structures(reference, target)
        with patch.object(aligner_module, "_SCREEN_BLOCK", 3):
            found, _ = align_structures(reference, target)
        np.testing.assert_array_equal(found, expected)

    def test_one_candidate_per_set_of_equivalent_alignments(self):
        """In a supercell every candidate is equivalent, so only one is screened."""
        reference = rock_salt()
        target = with_noise(reference, scale=0.05, seed=12, shift=(0.37, 0.12, 0.29))
        with patch.object(aligner_module, "_screen", wraps=aligner_module._screen) as screen:
            found, metrics = align_structures(reference, target)
        self.assertEqual(len(screen.call_args.args[0]), 1)
        self.assert_superimposed(reference, target, found, 0.3)
        self.assertLess(metrics['rmsd'], 0.15)

    def test_a_reference_without_symmetry_screens_every_candidate(self):
        """Noise in the reference above the symmetry tolerance leaves no candidates equivalent."""
        reference = with_noise(rock_salt(), scale=0.05, seed=13)
        target = with_noise(reference, scale=0.05, seed=14, shift=(0.37, 0.12, 0.29))
        with patch.object(aligner_module, "_screen", wraps=aligner_module._screen) as screen:
            _, metrics = align_structures(reference, target)
        self.assertEqual(len(screen.call_args.args[0]), 108)
        self.assertLess(metrics['rmsd'], 0.15)

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
                align_metric=None, align_algorithm="Nelder-Mead",
                align_minimizer_options=None, align_tolerance=1e-4)

    def test_max_dist_is_rejected_in_any_case(self):
        """'MAX_DIST' is rejected like 'max_dist'."""
        with self.assertRaisesRegex(ValueError, "max_dist"):
            aligner_module._check_deprecated_alignment_options(
                align_metric="MAX_DIST", align_algorithm=None,
                align_minimizer_options=None, align_tolerance=None)


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
