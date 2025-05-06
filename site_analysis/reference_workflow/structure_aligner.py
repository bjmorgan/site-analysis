"""StructureAligner for aligning crystal structures via translation optimization.

This module provides functionality to align reference structures to target structures
by finding the optimal translation vector that minimizes distances between corresponding atoms.
"""

import numpy as np
from pymatgen.core import Structure
from scipy.optimize import minimize # type: ignore
from typing import List, Dict, Tuple, Optional, Union, Any
from site_analysis.tools import calculate_species_distances

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
		# Validate structures and get species to use
		valid_species = self._validate_structures(reference, target, species)
		
		# Define objective function for optimization
		def objective_function(translation_vector):
			# Apply translation to reference coordinates
			translated_coords = reference.frac_coords + translation_vector
			translated_coords = translated_coords % 1.0  # Apply PBC
			
			# Create a temporary translated structure for distance calculation
			temp_structure = reference.copy()
			for i in range(len(temp_structure)):
				temp_structure[i] = temp_structure[i].species, translated_coords[i]
			
			# Calculate distances using our helper function
			_, all_distances = calculate_species_distances(temp_structure, target, species=valid_species)
			
			# Calculate the desired metric
			if not all_distances:  # Handle empty distance list
				return float('inf')
				
			if metric == 'rmsd':
				return np.sqrt(np.mean(np.array(all_distances)**2))
			elif metric == 'max_dist':
				return np.max(all_distances)
			else:
				raise ValueError(f"Unknown metric: {metric}")
		
		# Perform optimization
		from scipy.optimize import minimize
		result = minimize(
			objective_function,
			x0=[0, 0, 0],  # Start with zero translation
			method='Nelder-Mead',
			options={'xatol': 1e-4, 'fatol': 1e-4}
		)
		
		if not result.success:
			raise ValueError(f"Optimization failed: {result.message}")
		
		# Get the optimal translation vector
		translation_vector = result.x % 1.0  # Ensure in [0,1) range
		
		# Apply the translation to get the aligned structure
		aligned_structure = self._apply_translation(reference, translation_vector)
		
		# Calculate final metrics
		species_distances, all_distances = calculate_species_distances(
			aligned_structure, target, species=valid_species)
		
		metrics = {
			'rmsd': np.sqrt(np.mean(np.array(all_distances)**2)) if all_distances else float('inf'),
			'max_dist': np.max(all_distances) if all_distances else float('inf'),
			'mean_dist': np.mean(all_distances) if all_distances else float('inf')
		}
		
		return aligned_structure, translation_vector, metrics
	
	def _validate_structures(self, 
							reference: Structure, 
							target: Structure, 
							species: Optional[List[str]] = None) -> List[str]:
		"""Validate that structures can be aligned and determine species to use.
		
		Args:
			reference: Reference structure
			target: Target structure
			species: List of species to use for alignment
			
		Returns:
			List of species to use for alignment
			
		Raises:
			ValueError: If structures cannot be aligned
		"""
		# Check if species is provided
		if species is None:
			# No specific species provided - get all species from reference
			ref_species_counts = reference.composition.as_dict()
			target_species_counts = target.composition.as_dict()
			
			# Verify compositions match exactly
			if ref_species_counts != target_species_counts:
				raise ValueError(
					f"Structures have different compositions: "
					f"{reference.composition.formula} vs {target.composition.formula}"
				)
			
			# Use all species from reference
			species_to_use = list(ref_species_counts.keys())
		else:
			species_to_use = species
		
		# Validate each species has matching counts
		for sp in species_to_use:
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
		
		return species_to_use
	
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