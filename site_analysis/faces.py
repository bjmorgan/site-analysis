"""Faces of polyhedral sites.

This module provides ``Face``, one face of a polyhedral site, and builds the
faces of a polyhedron from the coordinates of its vertex atoms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .pbc_utils import correct_pbc

# Facets whose combined vertices all lie within this distance (Angstrom) of
# their best-fit plane form one face.
_FACE_PLANE_TOLERANCE = 0.01

# A face's atoms with their periodic image shifts, relative to the first
# atom's: (atom index, a, b, c) for each atom, sorted by atom index.
FaceKey = tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True, slots=True)
class Face:
    """One face of a polyhedral site.

    Attributes:
        vertex_indices: Indices of the atoms at the face's vertices, in
            ascending order (sorted on construction).
    """

    vertex_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Store the vertex indices in ascending order."""
        object.__setattr__(self, "vertex_indices", tuple(sorted(self.vertex_indices)))


def _faces_and_keys(
        vertex_indices: Sequence[int],
        vertex_frac_coords: np.ndarray,
        reference_center: np.ndarray | None,
        lattice_matrix: np.ndarray) -> list[tuple[Face, FaceKey]]:
    """The faces of one polyhedral site, each with its key.

    The vertices are unwrapped with ``correct_pbc``. Neighbouring facets of
    their convex hull are merged into one face when all their combined
    vertices lie within ``_FACE_PLANE_TOLERANCE`` of their best-fit plane.

    Args:
        vertex_indices: Atom index of each vertex.
        vertex_frac_coords: Fractional coordinates of the vertices, in the
            order of ``vertex_indices``.
        reference_center: Reference centre for unwrapping, or None.
        lattice_matrix: (3, 3) lattice matrix, rows as lattice vectors.

    Returns:
        ``(face, key)`` for each face, sorted by vertex indices then key.

    Raises:
        ValueError: If the vertices do not form a polyhedron, if any vertex
            is not a corner of their convex hull, or if the hull has fewer
            than four faces.
    """
    unwrapped, image_shifts = correct_pbc(
        np.asarray(vertex_frac_coords, dtype=float), reference_center, lattice_matrix)
    cartesian = unwrapped @ lattice_matrix
    try:
        hull = ConvexHull(cartesian)
    except (QhullError, ValueError) as error:
        raise ValueError(
            f"Vertices {list(vertex_indices)} do not form a polyhedron") from error

    if len(hull.vertices) < len(cartesian):
        corners = set(int(k) for k in hull.vertices)
        missing = [int(vertex_indices[k]) for k in range(len(cartesian)) if k not in corners]
        raise ValueError(
            f"Vertices {missing} are not corners of the polyhedron "
            f"with vertices {list(vertex_indices)}")

    def fits_one_plane(points: np.ndarray) -> bool:
        """Whether the points lie within the tolerance of their best-fit plane."""
        centred = points - points.mean(axis=0)
        normal = np.linalg.svd(centred)[2][-1]
        return bool((np.abs(centred @ normal) < _FACE_PLANE_TOLERANCE).all())

    group = list(range(len(hull.simplices)))
    group_vertices = {facet: set(int(k) for k in simplex)
                      for facet, simplex in enumerate(hull.simplices)}

    def root(facet: int) -> int:
        while group[facet] != facet:
            group[facet] = group[group[facet]]
            facet = group[facet]
        return facet

    for facet, neighbours in enumerate(hull.neighbors):
        for neighbour in neighbours:
            a, b = root(facet), root(int(neighbour))
            if a != b:
                combined = group_vertices[a] | group_vertices[b]
                if fits_one_plane(cartesian[sorted(combined)]):
                    group[b] = a
                    group_vertices[a] = combined

    roots = {root(facet) for facet in range(len(group))}
    faces = []
    for vertices in (group_vertices[r] for r in roots):
        pairs = sorted((int(vertex_indices[k]), *(int(s) for s in image_shifts[k]))
                       for k in vertices)
        _, a0, b0, c0 = pairs[0]
        key = tuple((atom, a - a0, b - b0, c - c0) for atom, a, b, c in pairs)
        faces.append((Face(tuple(atom for atom, *_ in pairs)), key))
    if len(faces) < 4:
        raise ValueError(
            f"Vertices {list(vertex_indices)} form fewer than four faces")
    return sorted(faces, key=lambda face_and_key: (face_and_key[0].vertex_indices,
                                                    face_and_key[1]))
