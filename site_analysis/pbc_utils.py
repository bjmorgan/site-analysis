"""Utilities for handling periodic boundary conditions."""

import numpy as np

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
	# Work with a copy to avoid modifying the input
	corrected_coords = frac_coords.copy()
	
	# Handle periodic boundary conditions for each dimension
	for dim in range(3):
		spread = np.max(corrected_coords[:, dim]) - np.min(corrected_coords[:, dim])
		if spread > 0.5:
			corrected_coords[corrected_coords[:, dim] < 0.5, dim] += 1.0
	
	return corrected_coords
	
def unwrap_vertices_to_reference_centre(frac_coords: np.ndarray, reference_centre: np.ndarray, lattice) -> np.ndarray:
	"""Unwrap vertices to their closest periodic images relative to a reference centre.
	
	Args:
		frac_coords: Array of fractional coordinates with shape (n, 3).
		reference_centre: Reference centre position for unwrapping.
		lattice: Lattice object for distance calculations.
		
	Returns:
		Unwrapped fractional coordinates with the same shape, shifted to ensure all coordinates >= 0.
	"""
	# Handle empty input
	if len(frac_coords) == 0:
		return frac_coords.copy()
	
	result = np.zeros_like(frac_coords)
	
	for i, vertex in enumerate(frac_coords):
		_, image = lattice.get_distance_and_image(reference_centre, vertex)
		result[i] = vertex + image
	
	# Apply uniform shift to ensure all coordinates are non-negative
	min_coords = np.min(result, axis=0)
	shift = np.maximum(0, np.ceil(-min_coords))
	result += shift
	
	return result