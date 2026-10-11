"""Structure alignment by translation.

This module provides the StructureAligner class, which finds the
translation that superimposes a reference structure on a target structure
in the same periodic cell. The reference-based workflow uses it to place
sites defined in a reference structure onto the target.

At the right translation, every reference atom of an alignment species
lies near a target atom of that species. So the right translation lies
near one of the translations that put a reference atom exactly on a
chosen target atom (the anchor). The aligner screens all of these
candidates on a sample of atoms, then refines them in order of fit, each
to minimise the RMSD of same-species nearest-neighbour distances, and
returns the first that brings every alignment atom within its species'
threshold. Translations that differ by a translational symmetry of the
alignment atoms fit equally well and only relabel the sites; any one of
them is returned. The search is deterministic and finds origin offsets
of any size, without symmetry information.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from site_analysis.neighbour_search import PeriodicNeighbourIndex

# Candidates are screened on up to this many reference atoms of each
# alignment species.
_SAMPLE_SIZE = 32

# Seed for choosing the sample, so that alignment is deterministic.
_SEED = 0

# Candidate translations screened at a time, to bound memory.
_SCREEN_BLOCK = 4096

# Refinement stops after this many steps if the pairing still changes.
_MAX_REFINEMENT_STEPS = 100

# The densest sphere packing (density pi / sqrt(18)) bounds the closest
# distance between two points of a periodic structure with N points in
# volume V by (sqrt(2) * V / N) ** (1/3) = 1.1225 * (V / N) ** (1/3);
# the factor is rounded up for a margin.
_CLOSEST_PAIR_FACTOR = 1.13


@dataclass(frozen=True, eq=False)
class _SpeciesGroup:
    """The atoms of one alignment species.

    Attributes:
        name: Species string.
        reference: Fractional coordinates of the reference atoms, (N, 3).
        target: Fractional coordinates of the target atoms, (N, 3).
        index: Neighbour index over the target atoms.
        threshold: Half the shortest distance between two different
            reference atoms (infinite for a single atom). If every
            reference atom is within it of a target atom, no two share a
            nearest target atom, so the pairing is one to one.
    """

    name: str
    reference: np.ndarray
    target: np.ndarray
    index: PeriodicNeighbourIndex
    threshold: float


def _threshold(frac_coords: np.ndarray, lattice_matrix: np.ndarray) -> float:
    """Half the shortest minimum-image distance between two different points.

    Args:
        frac_coords: Fractional coordinates, shape (N, 3).
        lattice_matrix: (3, 3) lattice matrix, rows as lattice vectors.

    Returns:
        Half the shortest distance, or infinity for fewer than two points.
    """
    n_points = len(frac_coords)
    if n_points < 2:
        return math.inf
    index = PeriodicNeighbourIndex(frac_coords, lattice_matrix)
    volume = abs(float(np.linalg.det(lattice_matrix)))
    # The packing bound covers every pair of points of the periodic
    # structure, including a point and its own image. If an image is the
    # nearer, the search widens until two different points are found.
    cutoff = _CLOSEST_PAIR_FACTOR * (volume / n_points) ** (1 / 3)
    while True:
        query_idx, point_idx, distances = index.query_within(frac_coords, cutoff)
        different = query_idx != point_idx
        if different.any():
            return 0.5 * float(distances[different].min())
        cutoff *= 2.0


def _sample_indices(n_atoms: int) -> np.ndarray:
    """Reference atoms of one species on which candidates are screened.

    Args:
        n_atoms: Number of reference atoms of the species.

    Returns:
        Up to ``_SAMPLE_SIZE`` distinct indices, in increasing order.
    """
    rng = np.random.default_rng(_SEED)
    return np.sort(rng.choice(n_atoms, size=min(_SAMPLE_SIZE, n_atoms), replace=False))


def _screen(candidates: np.ndarray,
        groups: list[_SpeciesGroup],
        samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Screen candidate translations on a sample of atoms.

    Each candidate first takes one refinement step on the sample: it moves
    by the mean displacement from the sampled atoms to their nearest
    target atoms. That removes the anchor's own displacement from its
    site, which every candidate from that anchor shares, so each sampled
    atom is then judged on its own displacement, as in the final check.
    The step has not converged, so the screen is slightly stricter than
    the final check.

    Args:
        candidates: Candidate translations, fractional, shape (M, 3).
        groups: The alignment species.
        samples: For each species, the sampled reference coordinates.

    Returns:
        ``(passed, moved, rmsd)``: an (M,) boolean array, true where every
        sampled atom of every species is within that species' threshold
        of a target atom of the species after the step; the moved
        candidates, shape (M, 3); and the RMSD of the sampled distances
        after the step, shape (M,).
    """
    passed = np.ones(len(candidates), dtype=bool)
    moved = np.empty_like(candidates)
    squares = np.zeros(len(candidates))
    n_sampled = sum(len(sample) for sample in samples)
    for start in range(0, len(candidates), _SCREEN_BLOCK):
        block = candidates[start:start + _SCREEN_BLOCK]
        rows = slice(start, start + len(block))
        total = np.zeros_like(block)
        for group, sample in zip(groups, samples):
            points = (sample[np.newaxis, :, :] + block[:, np.newaxis, :]) % 1.0
            nearest, _ = group.index.query_nearest(points.reshape(-1, 3))
            displacement = group.target[nearest].reshape(points.shape) - points
            total += (displacement - np.round(displacement)).sum(axis=1)
        block = block + total / n_sampled
        moved[rows] = block
        for group, sample in zip(groups, samples):
            points = (sample[np.newaxis, :, :] + block[:, np.newaxis, :]) % 1.0
            _, distances = group.index.query_nearest(points.reshape(-1, 3))
            distances = distances.reshape(len(block), len(sample))
            passed[rows] &= (distances < group.threshold).all(axis=1)
            squares[rows] += (distances ** 2).sum(axis=1)
    return passed, moved, np.sqrt(squares / n_sampled)


def _refine(translation: np.ndarray, groups: list[_SpeciesGroup]) -> np.ndarray:
    """Refine a translation to minimise the RMSD for its pairing.

    Pairs each reference atom with its nearest target atom of the same
    species and moves the translation by the mean displacement, which
    minimises the RMSD for that pairing, until the pairing stops changing.
    Displacements are wrapped by rounding, which gives the nearest image
    for displacements under half the cell's narrowest width.

    Args:
        translation: Starting translation, fractional, shape (3,).
        groups: The alignment species.

    Returns:
        The refined translation, wrapped into the unit cell.
    """
    pairing: list[np.ndarray] | None = None
    for _ in range(_MAX_REFINEMENT_STEPS):
        new_pairing = []
        displacements = []
        for group in groups:
            shifted = (group.reference + translation) % 1.0
            nearest, _ = group.index.query_nearest(shifted)
            displacement = group.target[nearest] - shifted
            displacements.append(displacement - np.round(displacement))
            new_pairing.append(nearest)
        if pairing is not None and all(
                np.array_equal(old, new) for old, new in zip(pairing, new_pairing)):
            break
        pairing = new_pairing
        translation = translation + np.concatenate(displacements).mean(axis=0)
    return translation % 1.0


def _nearest_distances(translation: np.ndarray,
        groups: list[_SpeciesGroup]) -> list[np.ndarray]:
    """Distance from each translated reference atom to its nearest target atom.

    Args:
        translation: Translation, fractional, shape (3,).
        groups: The alignment species.

    Returns:
        For each species, the exact minimum-image distances.
    """
    return [group.index.query_nearest((group.reference + translation) % 1.0)[1]
            for group in groups]


def _alignment(candidate: np.ndarray,
        groups: list[_SpeciesGroup]) -> tuple[np.ndarray, np.ndarray] | None:
    """Refine a candidate translation and check that it superimposes the structures.

    Args:
        candidate: Candidate translation, fractional, shape (3,).
        groups: The alignment species.

    Returns:
        (translation, distances), the refined translation and the
        distance from each reference atom of the alignment species to its
        nearest target atom of that species, if every distance is within
        its species' threshold; otherwise None.
    """
    translation = _refine(candidate, groups)
    distances = _nearest_distances(translation, groups)
    if all((d < group.threshold).all() for d, group in zip(distances, groups)):
        return translation, np.concatenate(distances)
    return None


def _rmsd(distances: np.ndarray) -> float:
    """Root-mean-square of the distances."""
    return float(np.sqrt(np.mean(distances ** 2)))


def _metrics(distances: np.ndarray) -> dict[str, float]:
    """The alignment metrics reported by StructureAligner.align."""
    return {
        'rmsd': _rmsd(distances),
        'max_dist': float(distances.max()),
        'mean_dist': float(distances.mean()),
    }


def _check_deprecated_alignment_options(**options: Any) -> None:
    """Warn about deprecated alignment options, which have no effect.

    Call this directly from the public function or method that takes the
    options, so that the warning points at the user's call.

    Args:
        **options: The deprecated options by name; ``None`` means not passed.

    Raises:
        ValueError: If ``align_metric`` is ``'max_dist'``: alignment now
            always minimises the RMSD, so honouring it is impossible and
            ignoring it would change the result.
    """
    if options.get("align_metric") == "max_dist":
        raise ValueError(
            "align_metric='max_dist' is no longer supported: alignment now always "
            "minimises the RMSD. Delete align_metric from this call.")
    passed = [name for name, value in options.items() if value is not None]
    if passed:
        names = ", ".join(passed)
        verb = "has" if len(passed) == 1 else "have"
        warnings.warn(
            f"{names} {verb} no effect and will be removed in version 2.0; delete "
            f"{'it' if len(passed) == 1 else 'them'} from this call. Alignment now "
            "always finds the best-fitting translation, whatever the origin offset.",
            FutureWarning, stacklevel=3)


class StructureAligner:
    """Finds the translation that superimposes a reference structure on a target."""

    def align(self,
            reference_frac_coords: np.ndarray,
            reference_species: list[str],
            target_frac_coords: np.ndarray,
            target_species: list[str],
            lattice_matrix: np.ndarray,
            species: list[str] | None = None) -> tuple[np.ndarray, dict[str, float]]:
        """Find the translation that superimposes the reference on the target.

        Screens candidate translations on a sample of atoms and refines
        them in order of fit, returning the first that brings every
        reference atom of the alignment species within half the shortest
        same-species spacing of a target atom of that species, refined to
        minimise the RMSD of those distances. Translations that differ by a
        translational symmetry of the alignment atoms fit equally well; any
        one of them is returned. Atom order does not matter.

        The screen is a filter: it applies the final threshold after a
        single refinement step, so an alignment that only just passes the
        full check (within about a percent of the threshold) can be
        rejected.

        Args:
            reference_frac_coords: Fractional coordinates of the reference
                structure, shape (N, 3).
            reference_species: Species string of each reference atom.
            target_frac_coords: Fractional coordinates of the target
                structure, shape (M, 3).
            target_species: Species string of each target atom.
            lattice_matrix: (3, 3) lattice matrix shared by both
                structures, rows as lattice vectors.
            species: Species to align on. If None, all species are used
                and the compositions must match.

        Returns:
            ``(translation, metrics)``: the fractional translation to add
            to the reference coordinates, and a dictionary with the
            ``'rmsd'``, ``'max_dist'`` and ``'mean_dist'`` of the distances
            from each reference atom of the alignment species to its
            nearest target atom of the same species.

        Raises:
            ValueError: If the structures cannot be compared on these
                species (see ``_validate_structures``), or the coordinates
                and species of a structure differ in length, or no
                candidate passes the screen and the full check.
        """
        reference_frac_coords = np.asarray(reference_frac_coords, dtype=float)
        target_frac_coords = np.asarray(target_frac_coords, dtype=float)
        for name, coords, names in (("reference", reference_frac_coords, reference_species),
                                    ("target", target_frac_coords, target_species)):
            if len(coords) != len(names):
                raise ValueError(
                    f"The {name} has {len(coords)} coordinates but {len(names)} species")
        valid_species = self._validate_structures(
            list(reference_species), list(target_species), species)
        reference_species_array = np.asarray(reference_species)
        target_species_array = np.asarray(target_species)
        groups = []
        for name in valid_species:
            reference = reference_frac_coords[reference_species_array == name]
            target = target_frac_coords[target_species_array == name]
            groups.append(_SpeciesGroup(
                name=name, reference=reference, target=target,
                index=PeriodicNeighbourIndex(target, lattice_matrix),
                threshold=_threshold(reference, lattice_matrix)))
        samples = [group.reference[_sample_indices(len(group.reference))]
                   for group in groups]
        # The candidates put each reference atom of the species with the
        # fewest atoms on the first target atom of that species.
        anchor_group = min(groups, key=lambda group: (len(group.reference), group.name))
        candidates = (anchor_group.target[0] - anchor_group.reference) % 1.0
        passed, moved, rmsd = _screen(candidates, groups, samples)
        survivors = moved[passed]
        # Best fit first: translations that only nearly superimpose the
        # alignment atoms, such as near-symmetries of a slightly distorted
        # sublattice, fit measurably worse than the right one.
        best: tuple[np.ndarray, np.ndarray] | None = None
        for candidate in survivors[np.argsort(rmsd[passed], kind="stable")]:
            best = _alignment(candidate, groups)
            if best is not None:
                break
        if best is None:
            thresholds = ", ".join(
                f"{group.name}: {group.threshold:.3g} A" for group in groups)
            raise ValueError(
                "No translation brings every alignment atom within half the "
                f"shortest same-species spacing of a target atom ({thresholds}). "
                "Align on species whose atoms stay on their sites.")
        return best[0], _metrics(best[1])

    def _validate_structures(self,
                            ref_species: list[str],
                            target_species: list[str],
                            species: list[str] | None = None) -> list[str]:
        """Validate that structures can be aligned and determine species to use.

        Args:
            ref_species: List of species strings for each site in the reference.
            target_species: List of species strings for each site in the target.
            species: Optional list of species to use for alignment. If ``None``,
                all species are used and compositions must match exactly.

        Returns:
            List of species to use for alignment.

        Raises:
            ValueError: If structures cannot be aligned.
        """
        if species is None:
            ref_counts: dict[str, int] = {}
            for s in ref_species:
                ref_counts[s] = ref_counts.get(s, 0) + 1
            target_counts: dict[str, int] = {}
            for s in target_species:
                target_counts[s] = target_counts.get(s, 0) + 1
            if ref_counts != target_counts:
                raise ValueError(
                    f"Structures have different compositions: "
                    f"{ref_counts} vs {target_counts}"
                )
            species_to_use = sorted(ref_counts.keys())
        else:
            species_to_use = list(dict.fromkeys(species))
        if not species_to_use:
            raise ValueError("No species to align on")

        for sp in species_to_use:
            ref_count = sum(1 for s in ref_species if s == sp)
            target_count = sum(1 for s in target_species if s == sp)
            if ref_count == 0:
                raise ValueError(f"Species {sp} not found in reference structure")
            if target_count == 0:
                raise ValueError(f"Species {sp} not found in target structure")
            if ref_count != target_count:
                raise ValueError(
                    f"Different number of {sp} atoms: "
                    f"{ref_count} in reference vs {target_count} in target"
                )
        return species_to_use
