"""Faces shared by polyhedral sites.

This module provides ``Face``, a face of polyhedral sites, and finds the faces
that polyhedral sites share: the vertex atoms two sites have in common under
one lattice translation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .pbc_utils import correct_pbc

if TYPE_CHECKING:
    from .polyhedral_site import PolyhedralSite


@dataclass(frozen=True, slots=True)
class Face:
    """A face of polyhedral sites.

    Attributes:
        site_indices: Indices of the sites the face bounds (two for a shared
            face), in ascending order (sorted on construction).
        vertex_indices: Indices of the atoms at the face's vertices, in
            ascending order (sorted on construction).
    """

    site_indices: tuple[int, ...]
    vertex_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Store the site indices and vertex indices in ascending order."""
        object.__setattr__(self, "site_indices", tuple(sorted(self.site_indices)))
        object.__setattr__(self, "vertex_indices", tuple(sorted(self.vertex_indices)))


def _find_shared_faces(
        sites: Sequence[PolyhedralSite],
        frac_coords: np.ndarray,
        lattice_matrix: np.ndarray) -> tuple[Face, ...]:
    """Find the faces that pairs of sites share.

    Each site's vertex coordinates are wrapped into the unit cell, then
    unwrapped around the site with ``correct_pbc``, which gives each atom an
    integer image shift. Two sites share the atoms they
    have in common that differ by the same lattice translation, that is,
    whose image shifts differ by the same amount. When three or more atoms
    share a translation, they form a shared face of the two sites.

    Args:
        sites: The sites to compare.
        frac_coords: Fractional coordinates of the atoms, shape
            ``(n_atoms, 3)``, indexed by atom index.
        lattice_matrix: (3, 3) lattice matrix, rows as lattice vectors.

    Returns:
        Every shared face, sorted by site indices then vertex indices.

    Raises:
        ValueError: If a site has a negative index, a vertex index that is
            not an atom of the structure, or non-finite vertex coordinates;
            if two sites coincide (have the same vertex atoms at the same
            positions); or if the vertex atoms of one site are all vertex
            atoms of another site at the same positions, so that it lies
            inside it.
    """
    sites_by_index: dict[int, PolyhedralSite] = {}
    shifts_by_atom: defaultdict[int, list[tuple[PolyhedralSite, tuple[int, ...]]]] = (
        defaultdict(list))
    for site in sites:
        if site.index < 0:
            raise ValueError(f"Site {site.index} has a negative index")
        if max(site.vertex_indices) >= len(frac_coords):
            raise ValueError(
                f"Site {site.index} has vertex index {max(site.vertex_indices)}, "
                f"but the structure has {len(frac_coords)} atoms")
        coords = frac_coords[site.vertex_indices]
        if not np.isfinite(coords).all():
            raise ValueError(f"Site {site.index} has non-finite vertex coordinates")
        _, image_shifts = correct_pbc(
            coords - np.floor(coords), site.reference_center, lattice_matrix)
        sites_by_index[site.index] = site
        for atom, shift in zip(site.vertex_indices, image_shifts.tolist(), strict=True):
            shifts_by_atom[atom].append((site, tuple(shift)))

    atoms_by_translation: defaultdict[tuple[int, int, tuple[int, ...]], set[int]] = (
        defaultdict(set))
    for atom, entries in shifts_by_atom.items():
        for position, (site_a, shift_a) in enumerate(entries):
            for site_b, shift_b in entries[position + 1:]:
                if site_a.index > site_b.index:
                    first, second, shift_first, shift_second = (
                        site_b, site_a, shift_b, shift_a)
                else:
                    first, second, shift_first, shift_second = (
                        site_a, site_b, shift_a, shift_b)
                translation = tuple(
                    p - q for p, q in zip(shift_first, shift_second, strict=True))
                atoms_by_translation[(first.index, second.index, translation)].add(atom)

    faces = []
    for (i, j, _), atoms in atoms_by_translation.items():
        first, second = sites_by_index[i], sites_by_index[j]
        in_first = atoms == set(first.vertex_indices)
        in_second = atoms == set(second.vertex_indices)
        if in_first and in_second:
            message = (f"Sites {i} and {j} coincide: they have the same vertex "
                       f"atoms at the same positions")
            if first.reference_center is None and second.reference_center is None:
                message += ("; neither site has a reference centre, so this can come "
                            "from ambiguous unwrapping in a narrow cell: "
                            "set reference centres")
            raise ValueError(message)
        if in_first or in_second:
            inner, outer = (i, j) if in_first else (j, i)
            raise ValueError(
                f"Site {inner} lies inside site {outer}: all its vertex atoms "
                f"are vertices of site {outer}")
        if len(atoms) >= 3:
            faces.append(Face((i, j), tuple(atoms)))
    return tuple(sorted(faces, key=lambda face: (face.site_indices, face.vertex_indices)))
