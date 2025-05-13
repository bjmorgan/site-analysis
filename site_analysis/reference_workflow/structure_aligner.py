"""Structure alignment tools for comparing and superimposing crystal structures.

This module provides the StructureAligner class, which finds the optimal
translation vector to superimpose one crystal structure onto another. This
alignment is important for:

1. Comparing structures from different sources with different coordinate origins
2. Analyzing structural changes while accounting for rigid translations
3. Preparing structures for site mapping in reference-based workflows

The alignment algorithm optimizes a translation vector to minimise distances
between corresponding atoms in the two structures, considering periodic
boundary conditions. It supports different optimisation metrics (RMSD or
maximum atom distance) and can align based on specific atom species.

This module is a key component of the reference-based workflow for defining
sites in one structure based on a template from another structure.
"""

import numpy as np
from pymatgen.core import Structure
from scipy.optimize import minimize
from typing import Optional, Union, Any, Callable
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
		species: Optional[list[str]] = None, 
		metric: str = 'rmsd', 
		tolerance: float = 1e-4,
		algorithm: str = 'Nelder-Mead',
		minimizer_options: Optional[dict[str, Any]] = None) -> tuple[Structure, np.ndarray, dict[str, float]]:
		"""Align reference structure to target structure via translation."""
		# Validate structures and get species to use
		valid_species = self._validate_structures(reference, target, species)
		
		# Create objective function
		objective_function = self._create_objective_function(reference, target, valid_species, metric)
		
		# Run the appropriate optimizer using the dispatcher
		translation_vector = self._run_minimizer(algorithm, objective_function, tolerance, minimizer_options)
		
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
		
	def _create_objective_function(self, 
		reference: Structure,
		target: Structure,
		valid_species: list[str],
		metric: str) -> Callable[[np.ndarray], float]:
		"""Create the objective function for optimization.
		
		Args:
			reference: Reference structure
			target: Target structure
			valid_species: List of species to include in alignment
			metric: Metric to optimize ('rmsd' or 'max_dist')
			
		Returns:
			function: The objective function that takes a translation vector and
					returns the distance metric value
		"""
		def objective_function(
			translation_vector: np.ndarray) -> float:
			# Ensure translation is in [0,1) range
			translation_vector = translation_vector % 1.0
			
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
				return float(np.sqrt(np.mean(np.array(all_distances)**2)))
			elif metric == 'max_dist':
				return float(np.max(all_distances))
			else:
				raise ValueError(f"Unknown metric: {metric}")
		
		return objective_function
	
	def _validate_structures(self, 
							reference: Structure, 
							target: Structure, 
							species: Optional[list[str]] = None) -> list[str]:
		"""Validate that structures can be aligned and determine species to use.
		
		Args:
			reference: Reference structure
			target: Target structure
			species: list of species to use for alignment
			
		Returns:
			list of species to use for alignment
			
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
		return np.array(translated % 1.0)
	
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
	
	def _run_minimizer(self,
		algorithm: str, 
		objective_function: Callable[[np.ndarray], float],
		tolerance: float,
		minimizer_options: Optional[dict[str, Any]] = None) -> np.ndarray:
		"""Run the selected minimization algorithm.
		
		Args:
			algorithm: Name of the algorithm to run
			objective_function: Function to minimize
			tolerance: Convergence tolerance
			minimizer_options: Additional options for the minimizer
			
		Returns:
			np.ndarray: Optimal translation vector
			
		Raises:
			ValueError: If the algorithm is not supported
		"""
		# Get the algorithm registry
		algorithm_registry = self._get_algorithm_registry()
		
		# Check if algorithm is supported
		if algorithm not in algorithm_registry:
			raise ValueError(f"Unsupported algorithm: {algorithm}. "
							f"Supported algorithms: {', '.join(algorithm_registry.keys())}")
		
		# Get the appropriate implementation method
		run_algorithm = algorithm_registry[algorithm]
		
		# Call the selected algorithm implementation
		return run_algorithm(objective_function, tolerance, minimizer_options)
		
	def _get_algorithm_registry(self) -> dict[str, 
												Callable[
													[Callable[[np.ndarray], float],
													float,
													Optional[dict[str, Any]]
												], np.ndarray]]:
		"""Get the registry of supported optimization algorithms.
		
		Returns:
			dict: Dictionary mapping algorithm names to implementation methods
		"""
		return {
			'Nelder-Mead': self._run_nelder_mead,
			'differential_evolution': self._run_differential_evolution
		}
		
	def _run_nelder_mead(self, 
					objective_function: Callable[[np.ndarray], float], 
					tolerance: float, 
					minimizer_options: Optional[dict[str, Any]] = None) -> np.ndarray:
		"""Run Nelder-Mead optimization.
		
		Args:
			objective_function: Function to minimize
			tolerance: Convergence tolerance
			minimizer_options: Additional options for the minimizer
			
		Returns:
			np.ndarray: Optimised translation vector
			
		Raises:
			ValueError: If optimization fails
		"""
		from scipy.optimize import minimize
		
		# Ensure minimizer_options is a dictionary
		minimizer_options = minimizer_options or {}
		
		# Default options - ensure they exactly match the original implementation
		options: dict[str, Any] = {
			'xatol': tolerance,
			'fatol': tolerance
		}
		
		# Update with user-provided options
		options.update(minimizer_options)
		
		# Run optimisation
		result = minimize(
			objective_function,
			x0=np.array([0, 0, 0]),  # Start with zero translation
			method='Nelder-Mead',
			options=options
		)
		
		if not result.success:
			raise ValueError(f"Optimization failed: {result.message}")
		
		# Ensure in [0,1) range
		return np.array(result.x) % 1.0
		
	def _run_differential_evolution(self,
			objective_function: Callable[[np.ndarray], float],
			tolerance: float,
			minimizer_options: Optional[dict[str, Any]] = None) -> np.ndarray:
		"""Run differential evolution optimization.
		
		Args:
			objective_function: Function to minimize
			tolerance: Convergence tolerance
			minimizer_options: Additional options for the minimizer
			
		Returns:
			np.ndarray: Optimal translation vector
		"""
		from scipy.optimize import differential_evolution
		
		# Default options for differential evolution
		options = {
			'tol': tolerance,
			'popsize': 15,
			'maxiter': 1000,
			'strategy': 'best1bin',
			'updating': 'immediate',
			'workers': 1  # Default to single process for compatibility
		}
		
		# Bounds for translation vector (all components in [0,1))
		bounds = [(0, 1), (0, 1), (0, 1)]
		
		# Update with user-provided options
		if minimizer_options:
			options.update(minimizer_options)
		
		# Extract bounds if provided in options
		if minimizer_options and 'bounds' in minimizer_options:
			bounds = minimizer_options.pop('bounds')
			
		# Run optimization
		result = differential_evolution(
			objective_function,
			bounds=bounds,
			**options
		)
		
		if not result.success:
			raise ValueError(f"Differential evolution optimization failed: {result.message}")
		
		return np.array(result.x) % 1.0  # Ensure in [0,1) range
		
	