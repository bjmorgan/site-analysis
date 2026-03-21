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
from typing import Any, Callable
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
        species: list[str] | None = None, 
        metric: str = 'rmsd', 
        tolerance: float = 1e-4,
        algorithm: str = 'Nelder-Mead',
        minimizer_options: dict[str, Any] | None = None) -> tuple[Structure, np.ndarray, dict[str, float]]:
        """Align reference structure to target structure via translation.

        Finds the optimal translation vector that minimises distances
        between corresponding atoms in the two structures.

        Args:
            reference: Reference structure to translate.
            target: Target structure to align to.
            species: Species to include in alignment. If None, all
                species present in both structures are used.
            metric: Distance metric to optimise ('rmsd' or 'max_dist').
            tolerance: Convergence tolerance for the optimiser.
            algorithm: Optimisation algorithm ('Nelder-Mead' or
                'differential_evolution').
            minimizer_options: Additional options passed to the optimiser.

        Returns:
            A tuple of (aligned_structure, translation_vector, metrics)
            where aligned_structure is the translated reference,
            translation_vector is the applied translation in fractional
            coordinates, and metrics is a dictionary of alignment quality
            measures.

        Raises:
            ValueError: If structures have incompatible compositions or
                if optimisation fails.
        """
        # Extract arrays from Structure at the public boundary
        ref_frac_coords = reference.frac_coords
        target_frac_coords = target.frac_coords
        lattice_matrix = reference.lattice.matrix
        ref_species = [s.species_string for s in reference]
        target_species_list = [s.species_string for s in target]

        # Validate structures and get species to use
        valid_species = self._validate_structures(
            ref_species, target_species_list, species)

        # Create objective function
        objective_function = self._create_objective_function(
            ref_frac_coords, target_frac_coords, lattice_matrix,
            ref_species, target_species_list, valid_species, metric)

        # Run the appropriate optimiser using the dispatcher
        translation_vector = self._run_minimizer(
            algorithm, objective_function, tolerance, minimizer_options)

        # Apply the translation to get the aligned structure
        aligned_structure = self._apply_translation(reference, translation_vector)

        # Calculate final metrics using arrays
        aligned_coords = (ref_frac_coords + translation_vector) % 1.0
        species_distances, all_distances = calculate_species_distances(
            aligned_coords, target_frac_coords, lattice_matrix,
            ref_species, target_species_list, species=valid_species)

        metrics = {
            'rmsd': float(np.sqrt(np.mean(np.array(all_distances)**2))) if all_distances else float('inf'),
            'max_dist': float(np.max(all_distances)) if all_distances else float('inf'),
            'mean_dist': float(np.mean(all_distances)) if all_distances else float('inf'),
        }

        return aligned_structure, translation_vector, metrics
        
    def _create_objective_function(self,
        ref_frac_coords: np.ndarray,
        target_frac_coords: np.ndarray,
        lattice_matrix: np.ndarray,
        ref_species: list[str],
        target_species: list[str],
        valid_species: list[str],
        metric: str) -> Callable[[np.ndarray], float]:
        """Create the objective function for optimisation.

        Args:
            ref_frac_coords: Fractional coordinates of the reference structure.
            target_frac_coords: Fractional coordinates of the target structure.
            lattice_matrix: Lattice matrix for distance calculations.
            ref_species: Species strings for each site in the reference.
            target_species: Species strings for each site in the target.
            valid_species: List of species to include in alignment.
            metric: Metric to optimise (``'rmsd'`` or ``'max_dist'``).

        Returns:
            Objective function that takes a translation vector and returns
            the distance metric value.
        """
        def objective_function(
            translation_vector: np.ndarray) -> float:
            translation_vector = translation_vector % 1.0
            translated_coords = (ref_frac_coords + translation_vector) % 1.0

            _, all_distances = calculate_species_distances(
                translated_coords, target_frac_coords, lattice_matrix,
                ref_species, target_species, species=valid_species)

            if not all_distances:
                return float('inf')

            if metric == 'rmsd':
                return float(np.sqrt(np.mean(np.array(all_distances)**2)))
            elif metric == 'max_dist':
                return float(np.max(all_distances))
            else:
                raise ValueError(f"Unknown metric: {metric}")

        return objective_function
    
    def _validate_structures(self,
                            ref_species: list[str],
                            target_species: list[str],
                            species: list[str] | None = None) -> list[str]:
        """Validate that structures can be aligned and determine species to use.

        Args:
            ref_species: List of species strings for each site in the reference.
            target_species: List of species strings for each site in the target.
            species: Optional list of species to use for alignment. If ``None``,
                all species are used and compositions must match exactly.

        Returns:
            List of species to use for alignment.

        Raises:
            ValueError: If structures cannot be aligned.
        """
        if species is None:
            ref_counts: dict[str, int] = {}
            for s in ref_species:
                ref_counts[s] = ref_counts.get(s, 0) + 1
            target_counts: dict[str, int] = {}
            for s in target_species:
                target_counts[s] = target_counts.get(s, 0) + 1
            if ref_counts != target_counts:
                raise ValueError(
                    f"Structures have different compositions: "
                    f"{ref_counts} vs {target_counts}"
                )
            species_to_use = sorted(ref_counts.keys())
        else:
            species_to_use = species

        for sp in species_to_use:
            ref_count = sum(1 for s in ref_species if s == sp)
            target_count = sum(1 for s in target_species if s == sp)
            if ref_count == 0:
                raise ValueError(f"Species {sp} not found in reference structure")
            if target_count == 0:
                raise ValueError(f"Species {sp} not found in target structure")
            if ref_count != target_count:
                raise ValueError(
                    f"Different number of {sp} atoms: "
                    f"{ref_count} in reference vs {target_count} in target"
                )
        return species_to_use
    
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
        minimizer_options: dict[str, Any] | None = None) -> np.ndarray:
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
                                                    dict[str, Any] | None
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
                    minimizer_options: dict[str, Any] | None = None) -> np.ndarray:
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
        result = minimize(  # type: ignore[call-overload]
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
            minimizer_options: dict[str, Any] | None = None) -> np.ndarray:
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
            bounds = minimizer_options['bounds']
            options.pop('bounds')
            
        # Run optimization
        result = differential_evolution(
            objective_function,
            bounds=bounds,  # type: ignore[arg-type]
            **options  # type: ignore[arg-type]
        )
        
        if not result.success:
            raise ValueError(f"Differential evolution optimization failed: {result.message}")
        
        return np.array(result.x) % 1.0  # Ensure in [0,1) range
        
    