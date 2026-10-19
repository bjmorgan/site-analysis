"""Helper functions for testing."""

import numpy as np
from pymatgen.core import Lattice, Structure


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


def interstitial_structure(supercell=(2, 2, 2), a=4.0):
	"""An fcc O lattice with Na at every octahedral and Li at every tetrahedral hole."""
	o = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
	na = [[0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
	li = [[x, y, z] for x in (0.25, 0.75) for y in (0.25, 0.75) for z in (0.25, 0.75)]
	unit = Structure(Lattice.cubic(a), ["O"] * 4 + ["Na"] * 4 + ["Li"] * 8, o + na + li)
	return unit * supercell
