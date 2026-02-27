"""Optional numba acceleration for polyhedral site analysis.

Provides JIT-compiled routines for containment testing and PBC shift
updates, with pure-numpy fallbacks when numba is not installed.

When numba is available, ``update_pbc_shifts`` and
``FaceTopologyCache`` use JIT-compiled implementations. Otherwise,
``update_pbc_shifts`` falls back to numpy and the caller should use
Delaunay tessellation for containment (handled in ``PolyhedralSite``).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull  # type: ignore

try:
    import numba  # type: ignore
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @numba.njit(cache=True)  # type: ignore[misc]
    def _numba_update_faces(
        vertex_coords: np.ndarray,
        face_simplices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """JIT-compiled face normal and centre sign computation.

        Args:
            vertex_coords: (N_vertices, 3) array of vertex positions.
            face_simplices: (N_faces, 3) array of vertex indices per face.

        Returns:
            Tuple of (face_normals, face_ref_points, centre_signs).
        """
        n_faces = face_simplices.shape[0]
        n_vertices = vertex_coords.shape[0]
        face_normals = np.empty((n_faces, 3))
        face_ref_points = np.empty((n_faces, 3))
        centre_signs = np.empty(n_faces)

        centre = np.zeros(3)
        for i in range(n_vertices):
            for k in range(3):
                centre[k] += vertex_coords[i, k]
        for k in range(3):
            centre[k] /= n_vertices

        for j in range(n_faces):
            i0 = face_simplices[j, 0]
            i1 = face_simplices[j, 1]
            i2 = face_simplices[j, 2]

            # Edge vectors from vertex 2
            e0x = vertex_coords[i0, 0] - vertex_coords[i2, 0]
            e0y = vertex_coords[i0, 1] - vertex_coords[i2, 1]
            e0z = vertex_coords[i0, 2] - vertex_coords[i2, 2]
            e1x = vertex_coords[i1, 0] - vertex_coords[i2, 0]
            e1y = vertex_coords[i1, 1] - vertex_coords[i2, 1]
            e1z = vertex_coords[i1, 2] - vertex_coords[i2, 2]

            # Cross product
            face_normals[j, 0] = e0y * e1z - e0z * e1y
            face_normals[j, 1] = e0z * e1x - e0x * e1z
            face_normals[j, 2] = e0x * e1y - e0y * e1x

            # Reference point (first vertex of face)
            for k in range(3):
                face_ref_points[j, k] = vertex_coords[i0, k]

            # Centre sign
            dot = 0.0
            for k in range(3):
                dot += face_normals[j, k] * (centre[k] - face_ref_points[j, k])
            centre_signs[j] = np.sign(dot)

        return face_normals, face_ref_points, centre_signs

    @numba.njit(cache=True)  # type: ignore[misc]
    def _numba_sn_query(
        x_pbc_points: np.ndarray,
        face_normals: np.ndarray,
        face_ref_points: np.ndarray,
        centre_signs: np.ndarray,
    ) -> bool:
        """JIT-compiled containment check using surface normals.

        For each PBC image point, checks whether the point lies on the
        same side as the polyhedron centre for every face. Uses early
        exit on first failing face for efficiency.

        Args:
            x_pbc_points: (N, 3) array of periodic boundary images.
            face_normals: (N_faces, 3) precomputed outward face normals.
            face_ref_points: (N_faces, 3) reference vertex per face.
            centre_signs: (N_faces,) sign of centre dot product per face.

        Returns:
            True if any PBC image is inside the polyhedron.
        """
        n_points = x_pbc_points.shape[0]
        n_faces = face_normals.shape[0]
        for i in range(n_points):
            all_match = True
            for j in range(n_faces):
                dot = 0.0
                for k in range(3):
                    dot += ((x_pbc_points[i, k] - face_ref_points[j, k])
                            * face_normals[j, k])
                if dot != 0.0 and np.sign(dot) != centre_signs[j]:
                    all_match = False
                    break
            if all_match:
                return True
        return False


def _numpy_update_pbc_shifts(
    frac_coords: np.ndarray,
    cached_raw_frac: np.ndarray,
    image_shifts: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Update cached PBC image shifts using numpy.

    Args:
        frac_coords: (n, 3) new raw fractional coordinates.
        cached_raw_frac: (n, 3) previous raw fractional coordinates.
        image_shifts: (n, 3) int, current cached image shifts.

    Returns:
        Tuple of (cache_valid, new_vertex_coords, new_image_shifts).
        If cache_valid is False, the other values are undefined.
    """
    diff = frac_coords - cached_raw_frac
    wrapping = np.round(diff).astype(np.int64)
    physical_diff = diff - wrapping
    if not np.all(np.abs(physical_diff) < 0.3):
        return False, frac_coords, image_shifts
    new_shifts = image_shifts - wrapping
    shifted = frac_coords + new_shifts
    min_coords = np.min(shifted, axis=0)
    uniform = np.maximum(0, np.ceil(-min_coords))
    return True, shifted + uniform, new_shifts


if HAS_NUMBA:
    @numba.njit(cache=True)  # type: ignore[misc]
    def _numba_update_pbc_shifts(
        frac_coords: np.ndarray,
        cached_raw_frac: np.ndarray,
        image_shifts: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """JIT-compiled PBC image shift update with early exit.

        Args:
            frac_coords: (n, 3) new raw fractional coordinates.
            cached_raw_frac: (n, 3) previous raw fractional coordinates.
            image_shifts: (n, 3) int, current cached image shifts.

        Returns:
            Tuple of (cache_valid, new_vertex_coords, new_image_shifts).
            If cache_valid is False, the other values are undefined.
        """
        n = frac_coords.shape[0]
        new_shifts = np.empty((n, 3), dtype=np.int64)
        for i in range(n):
            for k in range(3):
                diff = frac_coords[i, k] - cached_raw_frac[i, k]
                w = int(np.round(diff))
                physical = diff - w
                if physical >= 0.3 or physical <= -0.3:
                    return False, frac_coords, image_shifts
                new_shifts[i, k] = image_shifts[i, k] - w
        shifted = np.empty((n, 3))
        for i in range(n):
            for k in range(3):
                shifted[i, k] = frac_coords[i, k] + new_shifts[i, k]
        for k in range(3):
            min_val = shifted[0, k]
            for i in range(1, n):
                if shifted[i, k] < min_val:
                    min_val = shifted[i, k]
            u = 0.0
            if min_val < 0.0:
                u = np.ceil(-min_val)
            for i in range(n):
                shifted[i, k] += u
        return True, shifted, new_shifts

    update_pbc_shifts = _numba_update_pbc_shifts
else:
    update_pbc_shifts = _numpy_update_pbc_shifts


class FaceTopologyCache:
    """Cached face topology and surface normal data for polyhedral containment.

    The face topology (which vertex triples form each face) is computed
    once from an initial ``ConvexHull`` and cached permanently -- it depends
    only on vertex connectivity, not on coordinates. Per-timestep, the
    face normals, reference points, and centre signs are recomputed from
    new vertex coordinates using the cached topology.

    Attributes:
        face_simplices: (N_faces, 3) array of vertex indices per face.
    """

    def __init__(self, vertex_coords: np.ndarray) -> None:
        """Compute face topology from initial vertex coordinates.

        Args:
            vertex_coords: (N_vertices, 3) array of vertex positions.
                Used to build a ConvexHull and extract face connectivity.
        """
        hull = ConvexHull(vertex_coords)
        self.face_simplices: np.ndarray = hull.simplices
        self._face_normals: np.ndarray | None = None
        self._face_ref_points: np.ndarray | None = None
        self._centre_signs: np.ndarray | None = None
        self.update(vertex_coords)

    def update(self, vertex_coords: np.ndarray) -> None:
        """Recompute face normals from cached topology and new coordinates.

        Called once per timestep after vertex coordinates are assigned.

        Args:
            vertex_coords: (N_vertices, 3) array of current vertex positions.
        """
        self._face_normals, self._face_ref_points, self._centre_signs = (
            _numba_update_faces(vertex_coords, self.face_simplices)
        )

    def contains_point(self, x_pbc_points: np.ndarray) -> bool:
        """Test whether any PBC image point is inside the polyhedron.

        Args:
            x_pbc_points: (N, 3) array of periodic boundary images.

        Returns:
            True if any point is inside the polyhedron.
        """
        return bool(_numba_sn_query(
            x_pbc_points,
            self._face_normals,
            self._face_ref_points,
            self._centre_signs,
        ))
