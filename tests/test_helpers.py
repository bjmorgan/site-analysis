"""Helper functions for testing."""

import numpy as np
from pymatgen.core import Structure


def apply_random_displacement(structure, magnitude=0.1, seed=None):
	"""Apply random displacements to atoms in a structure.
	
	Args:
		structure (Structure): The pymatgen Structure to modify
		magnitude (float): Maximum displacement magnitude in Angstroms
		seed (int, optional): Random seed for reproducibility
	
	Returns:
		Structure: A new structure with atoms displaced
	"""
	if seed is not None:
		np.random.seed(seed)
	
	# Create a copy to avoid modifying the original
	displaced = structure.copy()
	
	# Apply displacement to each site
	for i in range(len(displaced)):
		# Generate random displacement vector
		displacement = np.random.uniform(-magnitude, magnitude, 3)
		displaced.translate_sites(i, displacement, frac_coords=False)
	
	return displaced