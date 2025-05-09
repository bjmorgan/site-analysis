"""Index mapping between reference and target crystal structures.

This module provides the IndexMapper class, which maps coordinating atom indices
from a reference structure to corresponding atoms in a target structure. This mapping
is essential when transferring site definitions from one structure to another,
particularly when atom orderings differ or when structures have been distorted.

The IndexMapper uses a distance-based approach to find the closest corresponding atoms
in the target structure for each reference atom, considering periodic boundary conditions.
It enforces a 1:1 mapping constraint to ensure that each reference atom maps to a distinct
target atom, which is necessary for preserving the topology of coordination environments.

The mapping can be filtered by atom species to ensure that atoms map only to atoms
of the same species, which is important for maintaining chemical validity when
mapping between structures with mixed compositions.

This module is a core component of the reference-based workflow, enabling the transfer
of site definitions between different structures or timesteps in a simulation.
"""

import numpy as np
from typing import List, Dict, Optional, Union
from pymatgen.core import Structure


class IndexMapper:
	"""Maps coordinating atom indices between reference and target crystal structures.
	
	Used to translate coordination environments defined in an ideal reference
	structure to corresponding atoms in a target structure, handling permutations
	and structural distortions via distance-based matching.
	
	The mapper verifies 1:1 correspondence between reference and target atoms,
	ensuring each coordinating position maps to exactly one target atom. If
	this constraint is violated, a ValueError is raised.
	"""
	
	def map_coordinating_atoms(
		self,
		reference: Structure,
		target: Structure,
		ref_coordinating: List[List[int]],
		target_species: Optional[Union[str, List[str]]] = None
	) -> List[List[int]]:
		"""Map coordinating atom indices from reference to target structure.
		
		Args:
			reference: Reference structure containing ideal coordination environments.
			target: target structure with potentially distorted or permuted atoms.
			ref_coordinating: List of coordinating atom index lists from reference.
				Each sublist contains indices of atoms that define a site.
			target_species: Optional filter for target atom species to map to.
				If specified, only maps to atoms of these species in the target structure.
				
		Returns:
			List of coordinating atom index lists mapped to the target structure.
			Maintains the same structure as input but with updated indices.
			
		Raises:
			ValueError: If 1:1 mapping cannot be achieved (e.g., missing atoms,
				ambiguous distances, or insufficient target atoms in target structure).
		"""
		# Extract all unique coordinating atoms from reference structure
		unique_indices = self._extract_unique_coordinating_atoms(ref_coordinating)
		
		# Create mapping from reference to target structure
		index_mapping = self._find_closest_atom_mapping(
			reference, target, unique_indices, target_species
		)
		
		# Map the coordination lists using the established mapping
		mapped_coordinating = self._apply_mapping(ref_coordinating, index_mapping)
		
		return mapped_coordinating
	
	def _extract_unique_coordinating_atoms(
		self,
		ref_coordinating: List[List[int]]
	) -> List[int]:
		"""Extract unique coordinating atom indices from coordination lists.
		
		Args:
			ref_coordinating: List of coordinating atom index lists.
			
		Returns:
			Sorted list of unique coordinating atom indices.
		"""
		unique_indices = set()
		for coord_list in ref_coordinating:
			unique_indices.update(coord_list)
		return sorted(list(unique_indices))
	
	def _find_closest_atom_mapping(
		self,
		reference: Structure,
		target: Structure,
		ref_indices: List[int],
		target_species: Optional[Union[str, List[str]]]
	) -> Dict[int, int]:
		"""Find closest atom in target structure for each reference atom.
		
		Args:
			reference: Reference structure.
			target: Target structure.
			ref_indices: List of reference atom indices to map.
			target_species: Optional species filter for the target structure.
			
		Returns:
			Dictionary mapping reference indices to target indices.
			
		Raises:
			ValueError: If 1:1 mapping cannot be achieved.
		"""
		# Ensure that target_species is a list if it is specified
		if isinstance(target_species, str):
			target_species = [target_species]
			
		# Create a filtered list of target atoms in the target structure
		if target_species is not None:
			target_mask = np.array([site.species_string in target_species for site in target])
			if not np.any(target_mask):
				raise ValueError(f"No atoms of species {target_species} found in target structure")
		else:
			target_mask = np.ones(len(target), dtype=bool)
			
		# Get coordinates of reference atoms to map
		ref_coords = np.array([reference[i].frac_coords for i in ref_indices])
		
		# Get coordinates of target atoms in the target structure
		target_indices = np.where(target_mask)[0]
		target_coords = np.array([target[i].frac_coords for i in target_indices])
		
		# Calculate distances between reference and target atoms (with PBC)
		dr_ij = reference.lattice.get_all_distances(ref_coords, target_coords)
		
		# Find closest target atom for each reference atom
		closest_indices = np.argmin(dr_ij, axis=1)
		mapped_indices = target_indices[closest_indices]
		
		# Check for 1:1 mapping violations
		if len(mapped_indices) != len(np.unique(mapped_indices)):
			# Find the duplicates
			seen = set()
			duplicates = []
			for idx in mapped_indices:
				if idx in seen:
					duplicates.append(int(idx))
				seen.add(idx)
			
			raise ValueError(
				f"1:1 mapping violation: Multiple reference atoms map to "
				f"the same target atom(s) at indices {duplicates}"
			)
		
		# Create mapping dictionary
		mapping = {ref_indices[i]: int(mapped_indices[i]) for i in range(len(ref_indices))}
		
		return mapping
	
	def _apply_mapping(
		self,
		ref_coordinating: List[List[int]],
		index_mapping: Dict[int, int]
	) -> List[List[int]]:
		"""Apply the index mapping to coordination lists.
		
		Args:
			ref_coordinating: Original coordination lists from reference.
			index_mapping: Mapping from reference to target indices.
			
		Returns:
			Coordination lists with mapped indices.
		"""
		mapped_coordinating = []
		for coord_list in ref_coordinating:
			# Convert each index to Python int to ensure consistent types
			mapped_list = [int(index_mapping[ref_idx]) for ref_idx in coord_list]
			mapped_coordinating.append(mapped_list)
		
		return mapped_coordinating