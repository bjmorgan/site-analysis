from typing import Dict, List, Union
from pymatgen.core import Structure
from site_analysis.tools import get_vertex_indices


class CoordinationEnvironmentFinder:
    """Finds coordination environments in a structure."""
    
    def __init__(self, structure: Structure):
        self.structure = structure
        self._atom_indices = self._index_atoms_by_species()
    
    def _index_atoms_by_species(self) -> Dict[str, List[int]]:
        """Organize atoms by species."""
        indices = {}
        for i, site in enumerate(self.structure):
            species = site.species_string
            if species not in indices:
                indices[species] = []
            indices[species].append(i)
        return indices
    
    def find_environments(self, 
                          center_species: str, 
                          vertex_species: Union[str, List[str]], 
                          n_vertices: int, 
                          cutoff: float) -> Dict[int, List[int]]:
        """Find coordination environments."""
        # Check if we have the required species
        if center_species not in self._atom_indices:
            raise ValueError(f"Center species '{center_species}' not found in structure")
        
        if isinstance(vertex_species, str):
            vertex_species = [vertex_species]
        
        for species in vertex_species:
            if species not in self._atom_indices:
                raise ValueError(f"Vertex species '{species}' not found in structure")
        
        # Use get_vertex_indices to find coordination environments
        environments = get_vertex_indices(
            structure=self.structure,
            centre_species=center_species,
            vertex_species=vertex_species,
            cutoff=cutoff,
            n_vertices=n_vertices
        )
        
        # Convert to dictionary format
        result = {}
        center_indices = self._atom_indices[center_species]
        for i, vertices in enumerate(environments):
            center_idx = center_indices[i]
            result[center_idx] = vertices
        
        return result
