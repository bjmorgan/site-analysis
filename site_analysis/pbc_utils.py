"""Utilities for handling periodic boundary conditions."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from pymatgen.core import Lattice

def apply_legacy_pbc_correction(frac_coords: np.ndarray) -> np.ndarray:
    """Apply the legacy spread-based periodic boundary condition handling.
    
    If the range of fractional coordinates along x, y, or z exceeds 0.5, 
    assume that the site wraps around the periodic boundary in that 
    dimension. Fractional coordinates for that dimension that are less 
    than 0.5 will be incremented by 1.0.
    
    Args:
        frac_coords: Array of fractional coordinates with shape (n, 3).
        
    Returns:
        Adjusted fractional coordinates with the same shape.
        
    Warning:
        This algorithm can produce incorrect results for sites spanning
        periodic boundaries in small unit cells. Consider using reference
        centre-based approaches for robust PBC handling.
    """
    corrected_coords: np.ndarray = frac_coords.copy()
    for dim in range(3):
        spread = np.max(corrected_coords[:, dim]) - np.min(corrected_coords[:, dim])
        if spread > 0.5:
            corrected_coords[corrected_coords[:, dim] < 0.5, dim] += 1.0
    
    return corrected_coords

# Generate all 27 possible shifts: [-1,0,1] for each dimension
_PERIODIC_SHIFTS = np.array([[dx, dy, dz] for dx in [-1, 0, 1] 
                                          for dy in [-1, 0, 1] 
                                          for dz in [-1, 0, 1]])
                                        
@overload
def unwrap_vertices_to_reference_center(
    frac_coords: np.ndarray,
    reference_center: np.ndarray,
    lattice: Lattice,
    return_image_shifts: Literal[False] = ...,
) -> np.ndarray: ...

@overload
def unwrap_vertices_to_reference_center(
    frac_coords: np.ndarray,
    reference_center: np.ndarray,
    lattice: Lattice,
    return_image_shifts: Literal[True] = ...,
) -> tuple[np.ndarray, np.ndarray]: ...

def unwrap_vertices_to_reference_center(
    frac_coords: np.ndarray,
    reference_center: np.ndarray,
    lattice: Lattice,
    return_image_shifts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Vectorised unwrapping of vertices to their closest periodic images relative to a reference centre.

    Args:
        frac_coords: Array of fractional coordinates with shape (n, 3).
        reference_center: Reference centre position for unwrapping.
        lattice: Lattice object for distance calculations.
        return_image_shifts: If True, also return the per-vertex integer
            image shifts (from ``_PERIODIC_SHIFTS``), separate from the
            uniform non-negative shift.

    Returns:
        Unwrapped fractional coordinates with the same shape, shifted to
        ensure all coordinates >= 0. If ``return_image_shifts`` is True,
        returns a tuple of (unwrapped_coords, image_shifts).
    """
    if frac_coords.size == 0:
        if return_image_shifts:
            return frac_coords, np.zeros((0, 3), dtype=int)
        return frac_coords

    n_vertices = len(frac_coords)
    vertex_images = frac_coords[:, np.newaxis, :] + _PERIODIC_SHIFTS[np.newaxis, :, :]

    ref_cart = lattice.get_cartesian_coords(reference_center)
    vertex_images_cart = lattice.get_cartesian_coords(
        vertex_images.reshape(n_vertices * 27, 3))
    distances = np.linalg.norm(
        vertex_images_cart - ref_cart, axis=1).reshape(n_vertices, 27)

    best_indices = np.argmin(distances, axis=1)
    image_shifts: np.ndarray = _PERIODIC_SHIFTS[best_indices]
    result = frac_coords + image_shifts

    min_coords = np.min(result, axis=0)
    uniform_shift = np.maximum(0, np.ceil(-min_coords))
    result = result + uniform_shift

    if return_image_shifts:
        return result, image_shifts
    return np.asarray(result)  # no-op; satisfies mypy no-any-return


def correct_pbc(
    frac_coords: np.ndarray,
    reference_center: np.ndarray | None,
    lattice: Lattice,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PBC correction to fractional coordinates.

    Selects the appropriate unwrapping strategy based on whether a
    reference centre is provided. When a reference centre is given,
    vertices are unwrapped to their closest periodic images relative
    to that centre. Otherwise, the legacy spread-based correction is
    applied.

    Args:
        frac_coords: Fractional coordinates, shape ``(n, 3)``.
        reference_center: Reference centre for unwrapping, or ``None``
            for legacy spread-based correction.
        lattice: Lattice for Cartesian distance calculations.
            Passed to the reference-centre unwrapping path; unused by
            the legacy spread-based path.

    Returns:
        Tuple of ``(corrected_coords, image_shifts)`` where both have
        shape ``(n, 3)`` and ``image_shifts`` has ``int64`` dtype.
    """
    if frac_coords.size == 0:
        return frac_coords, np.zeros((0, 3), dtype=np.int64)
    if reference_center is not None:
        return unwrap_vertices_to_reference_center(
            frac_coords, reference_center, lattice,
            return_image_shifts=True)
    corrected = apply_legacy_pbc_correction(frac_coords)
    image_shifts = np.round(corrected - frac_coords).astype(np.int64)
    return corrected, image_shifts
