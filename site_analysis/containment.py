"""Polyhedral containment algorithms with optional numba acceleration.

Provides a cached surface normal method for point-in-polyhedron testing.
Face topology (which vertex triples form each face) is computed once from
an initial ``ConvexHull`` and reused across timesteps. Per-timestep, only
face normals and reference signs are recomputed from updated vertex
coordinates.

When numba is available, the containment query is JIT-compiled with
early-exit logic for maximum throughput. Otherwise, the caller should
fall back to Delaunay tessellation (handled in ``PolyhedralSite``).
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
