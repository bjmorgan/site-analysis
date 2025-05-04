"""StructureAligner for aligning crystal structures via translation optimization.

This module provides functionality to align reference structures to target structures
by finding the optimal translation vector that minimizes distances between corresponding atoms.
"""

import numpy as np
from pymatgen.core import Structure
from scipy.optimize import minimize # type: ignore
from typing import List, Dict, Tuple, Optional, Union, Any


class StructureAligner:
	"""Aligns crystal structures via translation optimization.
	
	This class provides methods to align a reference structure to a target structure
	by finding the optimal translation vector that minimizes distances between
	corresponding atoms, considering periodic boundary conditions.
	"""
	
	def align(self, 
			  reference: Structure, 
			  target: Structure, 
			  species: Optional[List[str]] = None, 
			  metric: str = 'rmsd', 
			  tolerance: float = 0.1) -> Tuple[Structure, np.ndarray, Dict[str, float]]:
		"""Align reference structure to target structure via translation.
		
		Args:
			reference: Reference structure to be aligned.
			target: Target structure to align to.
			species: List of species to use for alignment.
				If None, all common species will be used.
			metric: Metric to optimize. Options are:
				'rmsd': Root mean square deviation
				'max_dist': Maximum distance between any atom pair
				'mean_dist': Mean distance between atom pairs
			tolerance: Tolerance for considering atoms as aligned.
				Default is 0.1 Angstroms.
				
		Returns:
			tuple: (aligned_structure, translation_vector, metrics)
				aligned_structure: The aligned reference structure.
				translation_vector: The translation vector used for alignment.
				metrics: Dict with alignment quality metrics.
				
		Raises:
			ValueError: If the structures cannot be aligned due to different
				composition or if no valid alignment is found.
		"""
		# Validate inputs and create atom mappings
		ref_indices, target_indices = self._validate_and_map_structures(reference, target, species)
		
		# Define objective function for optimization
		def objective_function(translation_vector):
			# Apply translation to reference coordinates
			translated_coords = self._translate_coords(
				reference.frac_coords[ref_indices], translation_vector)
			
			# Calculate distances considering PBC
			distances = self._calculate_distances(
				translated_coords, target.frac_coords[target_indices], target.lattice)
			
			# Return the metric value
			if metric == 'rmsd':
				return np.sqrt(np.mean(distances**2))
			elif metric == 'max_dist':
				return np.max(distances)
			elif metric == 'mean_dist':
				return np.mean(distances)
			else:
				raise ValueError(f"Unknown metric: {metric}")
		
		# Perform optimization
		result = minimize(
			objective_function,
			x0=[0, 0, 0],  # Start with zero translation
			method='Nelder-Mead',  # More robust method without gradient requirements
			options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
		)
		
		if not result.success:
			raise ValueError(f"Optimization failed: {result.message}")
		
		# Get the optimal translation vector
		translation_vector = result.x
		
		# Apply the translation to get the aligned structure
		aligned_structure = self._apply_translation(reference, translation_vector)
		
		# Calculate final metrics
		metrics = self._calculate_metrics(
			aligned_structure.frac_coords[ref_indices],
			target.frac_coords[target_indices],
			target.lattice
		)
		
		return aligned_structure, translation_vector, metrics
	
	def _validate_and_map_structures(self, 
									 reference: Structure, 
									 target: Structure, 
									 species: Optional[List[str]] = None) -> Tuple[List[int], List[int]]:
		"""Validate structures and create atom mappings for alignment.
		
		Args:
			reference: Reference structure
			target: Target structure
			species: List of species to use for alignment
			
		Returns:
			Tuple of lists containing indices of atoms to align in each structure
			
		Raises:
			ValueError: If structures cannot be aligned
		"""
		if species is None:
			# Find common species
			ref_species = set(site.species_string for site in reference)
			target_species = set(site.species_string for site in target)
			common_species = ref_species.intersection(target_species)
			
			if not common_species:
				raise ValueError("No common species found between reference and target structures")
			
			species = list(common_species)
		
		# Extract indices for each species in both structures
		ref_indices: List[int] = []
		target_indices: List[int] = []
		
		for sp in species:
			ref_sp_indices = reference.indices_from_symbol(sp)
			target_sp_indices = target.indices_from_symbol(sp)
			
			if not ref_sp_indices:
				raise ValueError(f"Species {sp} not found in reference structure")
			if not target_sp_indices:
				raise ValueError(f"Species {sp} not found in target structure")
			
			# Check if we have the same number of atoms for this species
			if len(ref_sp_indices) != len(target_sp_indices):
				raise ValueError(
					f"Different number of {sp} atoms: "
					f"{len(ref_sp_indices)} in reference vs "
					f"{len(target_sp_indices)} in target"
				)
			
			# Add indices to our lists
			ref_indices.extend(ref_sp_indices)
			target_indices.extend(target_sp_indices)
		
		return ref_indices, target_indices
	
	def _translate_coords(self, 
						 coords: np.ndarray, 
						 translation_vector: np.ndarray) -> np.ndarray:
		"""Apply translation to coordinates.
		
		Args:
			coords: Fractional coordinates to translate
			translation_vector: Translation vector to apply
			
		Returns:
			Translated coordinates
		"""
		translated = coords + translation_vector
		# Ensure coordinates are within [0, 1)
		return translated % 1.0
	
	def _calculate_distances(self, 
							coords1: np.ndarray, 
							coords2: np.ndarray, 
							lattice: Any) -> np.ndarray:
		"""Calculate distances between pairs of coordinates.
		
		Args:
			coords1: First set of coordinates
			coords2: Second set of coordinates
			lattice: Lattice to use for distance calculations
			
		Returns:
			Array of distances between corresponding coordinate pairs
		"""
		distances = []
		for i in range(len(coords1)):
			# Use pymatgen's method that handles PBC correctly
			dist = lattice.get_distance_and_image(coords1[i], coords2[i])[0]
			distances.append(dist)
		return np.array(distances)
	
	def _apply_translation(self, 
						  structure: Structure, 
						  translation_vector: np.ndarray) -> Structure:
		"""Apply translation to entire structure.
		
		Args:
			structure: Structure to translate
			translation_vector: Translation vector to apply
			
		Returns:
			Translated structure
		"""
		# Create a copy of the structure
		new_structure = structure.copy()
		
		# Apply translation to all sites
		for i, site in enumerate(new_structure):
			frac_coords = site.frac_coords + translation_vector
			# Ensure coordinates are within [0, 1)
			frac_coords = frac_coords % 1.0
			new_structure[i] = site.species, frac_coords
		
		return new_structure
	
	def _calculate_metrics(self, 
						  coords1: np.ndarray, 
						  coords2: np.ndarray, 
						  lattice: Any) -> Dict[str, float]:
		"""Calculate alignment quality metrics.
		
		Args:
			coords1: First set of coordinates
			coords2: Second set of coordinates
			lattice: Lattice to use for distance calculations
			
		Returns:
			Dictionary of metrics
		"""
		distances = self._calculate_distances(coords1, coords2, lattice)
		
		metrics = {
			'rmsd': np.sqrt(np.mean(distances**2)),
			'max_dist': np.max(distances),
			'mean_dist': np.mean(distances)
		}
		
		return metrics