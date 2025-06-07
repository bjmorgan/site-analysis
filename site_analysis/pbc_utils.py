"""Utilities for handling periodic boundary conditions."""

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
	# Work with a copy to avoid modifying the input
	corrected_coords: np.ndarray = frac_coords.copy()
	
	# Handle periodic boundary conditions for each dimension
	for dim in range(3):
		spread = np.max(corrected_coords[:, dim]) - np.min(corrected_coords[:, dim])
		if spread > 0.5:
			corrected_coords[corrected_coords[:, dim] < 0.5, dim] += 1.0
	
	return corrected_coords

# Generate all 27 possible shifts: [-1,0,1] for each dimension
_PERIODIC_SHIFTS = np.array([[dx, dy, dz] for dx in [-1, 0, 1] 
  										  for dy in [-1, 0, 1] 
  										  for dz in [-1, 0, 1]])
										
def unwrap_vertices_to_reference_center(
	frac_coords: np.ndarray,
	reference_center: np.ndarray,
	lattice: Lattice
) -> np.ndarray:
	"""Vectorised unwrapping of vertices to their closest periodic images relative to a reference centre.
	
	Args:
		frac_coords: Array of fractional coordinates with shape (n, 3).
		reference_center: Reference centre position for unwrapping.
		lattice: Lattice object for distance calculations.
		
	Returns:
		Unwrapped fractional coordinates with the same shape, shifted to ensure all coordinates >= 0.
	"""
	if frac_coords.size == 0:
		return frac_coords
		
	# Apply all shifts using broadcasting: (n_vertices, 27, 3)
	vertex_images = frac_coords[:, np.newaxis, :] + _PERIODIC_SHIFTS[np.newaxis, :, :]
	
	# Convert to Cartesian coordinates for true distance calculation
	ref_cart = lattice.get_cartesian_coords(reference_center)  # (3,)
	n_vertices = len(frac_coords)
	vertex_images_flat = vertex_images.reshape(n_vertices * 27, 3)  # (n_vertices * 27, 3)
	vertex_images_cart = lattice.get_cartesian_coords(vertex_images_flat)  # (n_vertices * 27, 3)
	
	# Calculate Euclidean distances from reference centre to all images
	distances_flat = np.linalg.norm(vertex_images_cart - ref_cart, axis=1)  # (n_vertices * 27,)
	
	# Reshape distances back to (n_vertices, 27)
	distances = distances_flat.reshape(n_vertices, 27)
	
	# Select closest image for each vertex
	best_indices = np.argmin(distances, axis=1)
	result: np.ndarray = vertex_images[np.arange(n_vertices), best_indices]
	
	# Apply uniform shift to ensure all coordinates are non-negative
	min_coords = np.min(result, axis=0)
	shift = np.maximum(0, np.ceil(-min_coords))
	result += shift
	
	return result