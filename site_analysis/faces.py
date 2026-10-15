"""Faces of polyhedral sites.

This module provides ``Face``, one face of a polyhedral site, and the
construction of a site's faces from the coordinates of its vertex atoms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .pbc_utils import correct_pbc

# Neighbouring hull facets are one face if every vertex of each lies within
# this distance (Angstroms) of the other's plane.
_FACE_PLANE_TOLERANCE = 0.01

# A face's atoms with their periodic image shifts, relative to the first
# atom's: (atom index, a, b, c) for each atom, sorted by atom index.
FaceKey = tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class Face:
    """One face of a polyhedral site.

    Attributes:
        vertex_indices: Indices of the atoms at the face's vertices, in
            ascending order.
    """

    vertex_indices: tuple[int, ...]


def _faces_and_keys(
        vertex_indices: Sequence[int],
        vertex_frac_coords: np.ndarray,
        reference_center: np.ndarray | None,
        lattice_matrix: np.ndarray) -> list[tuple[Face, FaceKey]]:
    """The faces of one polyhedral site, each with its key.

    The vertices are unwrapped with ``correct_pbc``. Neighbouring facets of
    their convex hull are merged into one face when every vertex of each
    lies within ``_FACE_PLANE_TOLERANCE`` of the other's plane.

    Args:
        vertex_indices: Atom index of each vertex.
        vertex_frac_coords: Fractional coordinates of the vertices, in the
            order of ``vertex_indices``.
        reference_center: Reference centre for unwrapping, or None.
        lattice_matrix: (3, 3) lattice matrix, rows as lattice vectors.

    Returns:
        ``(face, key)`` for each face, sorted by vertex indices then key.

    Raises:
        ValueError: If the vertices do not form a polyhedron.
    """
    unwrapped, image_shifts = correct_pbc(
        np.asarray(vertex_frac_coords, dtype=float), reference_center, lattice_matrix)
    cartesian = unwrapped @ lattice_matrix
    try:
        hull = ConvexHull(cartesian)
    except (QhullError, ValueError) as error:
        raise ValueError(
            f"Vertices {list(vertex_indices)} do not form a polyhedron") from error

    def within_plane(points: np.ndarray, facet: int) -> bool:
        normal, offset = hull.equations[facet][:3], hull.equations[facet][3]
        return bool((np.abs(cartesian[points] @ normal + offset) < _FACE_PLANE_TOLERANCE).all())

    group = list(range(len(hull.simplices)))

    def root(facet: int) -> int:
        while group[facet] != facet:
            group[facet] = group[group[facet]]
            facet = group[facet]
        return facet

    for facet, neighbours in enumerate(hull.neighbors):
        for neighbour in neighbours:
            if (neighbour > facet
                    and within_plane(hull.simplices[neighbour], facet)
                    and within_plane(hull.simplices[facet], neighbour)):
                group[root(neighbour)] = root(facet)

    vertices_by_face: defaultdict[int, set[int]] = defaultdict(set)
    for facet, simplex in enumerate(hull.simplices):
        vertices_by_face[root(facet)].update(int(k) for k in simplex)

    faces = []
    for vertices in vertices_by_face.values():
        pairs = sorted((int(vertex_indices[k]), *(int(s) for s in image_shifts[k]))
                       for k in vertices)
        _, a0, b0, c0 = pairs[0]
        key = tuple((atom, a - a0, b - b0, c - c0) for atom, a, b, c in pairs)
        faces.append((Face(tuple(atom for atom, *_ in pairs)), key))
    return sorted(faces, key=lambda face_and_key: (face_and_key[0].vertex_indices,
                                                    face_and_key[1]))
