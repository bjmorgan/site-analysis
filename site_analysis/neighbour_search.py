"""Periodic neighbour search with a KD-tree.

Provides ``PeriodicNeighbourIndex``, which finds the indexed points
within a cutoff of each query point, or the nearest indexed point, in a
periodic cell of any shape. Results equal those of computing the
minimum-image distance from every query point to every indexed point,
with memory that scales with the number of points and pairs found.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# The rows of the normalised lattice are unit vectors, so a smallest
# singular value below this means the lattice vectors are (almost)
# coplanar.
_MIN_SINGULAR_VALUE = 1e-8


def _as_coords(coords: np.ndarray, name: str) -> np.ndarray:
    """Return coordinates as a float64 array of shape (N, 3).

    Args:
        coords: Coordinates to check.
        name: Argument name, for the error message.

    Returns:
        The coordinates as a contiguous float64 array.

    Raises:
        ValueError: If ``coords`` does not have shape (N, 3).
    """
    array = np.ascontiguousarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    return array


class PeriodicNeighbourIndex:
    """KD-tree neighbour search for points in a periodic cell.

    The tree holds fractional coordinates scaled by the lattice lengths,
    in which the cell is an axis-aligned box. Where the cell angles are
    not 90 degrees, separations in these coordinates differ from Cartesian
    separations, but a Cartesian separation is never shorter than
    ``sigma_min`` times the scaled separation, where ``sigma_min`` is the
    smallest singular value of the lattice matrix with each row scaled to
    unit length. A tree search with radius ``r / sigma_min`` therefore
    finds every pair within Cartesian distance ``r``. The distance of each
    candidate pair is then computed over 27 periodic images by
    ``paired_mic_distances``.
    """

    def __init__(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Create a PeriodicNeighbourIndex.

        Args:
            frac_coords: Fractional coordinates of the points to index,
                shape (N, 3).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention: ``lattice.matrix``).

        Raises:
            ValueError: If ``frac_coords`` does not have shape (N, 3), or
                ``lattice_matrix`` does not have shape (3, 3) or is
                singular.
        """
        self._frac_coords = _as_coords(frac_coords, "frac_coords").copy()
        self._lattice_matrix = np.array(lattice_matrix, dtype=np.float64, order="C")
        if self._lattice_matrix.shape != (3, 3):
            raise ValueError(
                f"lattice_matrix must have shape (3, 3), got {self._lattice_matrix.shape}"
            )
        self._lengths = np.linalg.norm(self._lattice_matrix, axis=1)
        if np.any(self._lengths == 0.0):
            raise ValueError("lattice_matrix must be non-singular, but has a zero-length row")
        unit_rows = self._lattice_matrix / self._lengths[:, np.newaxis]
        self._sigma_min = float(np.linalg.svd(unit_rows, compute_uv=False).min())
        if self._sigma_min < _MIN_SINGULAR_VALUE:
            raise ValueError("lattice_matrix must be non-singular, but its rows are coplanar")
        self._tree = cKDTree(self._scaled(self._frac_coords), boxsize=self._lengths)

    def __len__(self) -> int:
        """Return the number of indexed points."""
        return int(self._frac_coords.shape[0])

    def _scaled(self, frac_coords: np.ndarray) -> np.ndarray:
        """Map fractional coordinates into the tree's box.

        Args:
            frac_coords: Fractional coordinates, shape (N, 3).

        Returns:
            Coordinates wrapped into [0, 1) and scaled by the lattice
            lengths.
        """
        scaled = np.mod(frac_coords, 1.0) * self._lengths
        # np.mod returns exactly 1.0 for tiny negative inputs, and cKDTree
        # rejects points on the far edge of the box.
        return np.where(scaled >= self._lengths, 0.0, scaled)
