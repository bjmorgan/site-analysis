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
		# Validate structures and get species to use
		valid_species = self._validate_structures(reference, target, species)
		
		# Map atoms between structures
		ref_indices, target_indices = self._map_atoms_by_species(reference, target, valid_species)
		
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
	
	def _map_atoms_by_species(self,
							 reference: Structure,
							 target: Structure,
							 species: List[str]) -> Tuple[List[int], List[int]]:
		"""Map atoms between structures by species and proximity.
		
		For each species, finds the optimal assignment of atoms between 
		reference and target structures based on minimum distances.
		
		Args:
			reference: Reference structure
			target: Target structure
			species: List of species to map
			
		Returns:
			Tuple of (ref_indices, target_indices) mapping atoms between structures
		"""
		from scipy.optimize import linear_sum_assignment
		
		ref_indices = []
		target_indices = []
		
		for sp in species:
			ref_sp_indices = reference.indices_from_symbol(sp)
			target_sp_indices = target.indices_from_symbol(sp)
			
			# Get coordinates for this species
			ref_coords = reference.frac_coords[ref_sp_indices]
			target_coords = target.frac_coords[target_sp_indices]
			
			# Calculate distance matrix between all pairs of atoms of this species
			distance_matrix = np.zeros((len(ref_sp_indices), len(target_sp_indices)))
			for i, ref_idx in enumerate(ref_sp_indices):
				for j, target_idx in enumerate(target_sp_indices):
					# Calculate minimum distance considering PBC
					dist = target.lattice.get_distance_and_image(
						reference[ref_idx].frac_coords, 
						target[target_idx].frac_coords)[0]
					distance_matrix[i, j] = dist
			
			# Use the Hungarian algorithm to find the optimal assignment
			row_ind, col_ind = linear_sum_assignment(distance_matrix)
			
			# Add the mapped indices
			for i, j in zip(row_ind, col_ind):
				ref_indices.append(ref_sp_indices[i])
				target_indices.append(target_sp_indices[j])
		
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