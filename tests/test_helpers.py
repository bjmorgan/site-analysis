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


def interstitial_structure_outside_the_cell(seed=3):
	"""`interstitial_structure()` with each atom moved by a random whole number of cells along each axis."""
	structure = interstitial_structure()
	coords = structure.frac_coords + np.random.default_rng(seed).integers(
		-2, 3, size=structure.frac_coords.shape)
	return Structure(structure.lattice, structure.species, coords)


def li_na_structure():
	"""A Li atom and a Na atom in a 10 Angstrom cubic cell."""
	return Structure(Lattice.cubic(10.0), ["Li", "Na"], [[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]])


def one_target_atom_for_two_reference_atoms():
	"""A Li and a Na tetrahedron of O atoms whose vertices map to one target atom.

	In the reference, O atoms 2-5 surround Li atom 0 and O atoms 6-9
	surround Na atom 1. In the target, atom 2 moves away and atom 6 moves
	to between reference atoms 2 and 6, so both are nearest to target
	atom 6.

	Returns:
		tuple: The reference and target Structures.
	"""
	li, na = np.array([3.0, 5.0, 5.0]), np.array([7.0, 5.0, 5.0])
	tetrahedron = 0.6 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
	coords = [li, na, *(li + tetrahedron), *(na + tetrahedron * [-1, 1, 1])]
	reference = Structure(Lattice.cubic(10.0), ["Li", "Na"] + ["O"] * 8, coords,
		coords_are_cartesian=True)
	coords[2] = np.array([0.5, 0.5, 0.5])
	coords[6] = np.array([5.0, 5.6, 5.6])
	target = Structure(reference.lattice, reference.species, coords,
		coords_are_cartesian=True)
	return reference, target
