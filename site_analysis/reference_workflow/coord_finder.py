"""Coordination environment finder for crystal structures.

This module provides the CoordinationEnvironmentFinder class, which identifies
coordination environments in crystal structures based on species, distance cutoffs,
and coordination numbers. A coordination environment consists of a central atom
surrounded by a specific number of coordinating atoms within a given distance.

The CoordinationEnvironmentFinder is used to identify specific site types in a
reference structure, such as tetrahedral, octahedral, or other coordination
environments. These environments can then be mapped to a target structure,
enabling the creation of corresponding sites.

Key features include:
- Finding atoms with exactly the specified coordination environment
- Supporting both single and multiple coordinating species
- Filtering by centre atom species and coordinating atom species
- Customising coordination requirements per centre atom

This module is a key component of the reference-based workflow, providing the
initial identification of coordination environments that will be used to define
sites in both reference and target structures.
"""

from pymatgen.core import Structure
from site_analysis.tools import get_coordination_indices


class CoordinationEnvironmentFinder:
    """Finds coordination environments in a structure."""

    def __init__(self, structure: Structure):
        """Initialise with a structure. Extracts arrays for internal use.

        Args:
            structure: The pymatgen Structure to analyse.
        """
        self._frac_coords = structure.frac_coords
        self._lattice_matrix = structure.lattice.matrix
        self._species = [site.species_string for site in structure]
        self._atom_indices = self._index_atoms_by_species()

    def _index_atoms_by_species(self) -> dict[str, list[int]]:
        """Create a mapping of species to atom indices.

        Indexes all atoms by their species string, creating a dictionary
        where keys are species strings and values are lists of atom indices
        having that species.

        Returns:
            A dictionary mapping species strings to lists of atom indices.
        """
        indices: dict[str, list[int]] = {}
        for i, sp in enumerate(self._species):
            if sp not in indices:
                indices[sp] = []
            indices[sp].append(i)
        return indices

    def find_environments(self,
                          center_species: str,
                          coordination_species: str | list[str],
                          n_coord: int,
                          cutoff: float) -> dict[int, list[int]]:
        """Find coordination environments in the structure.

        Locates atoms of center_species and finds environments where these
        atoms are coordinated by exactly n_coord atoms of
        coordination_species within the specified cutoff distance.

        Args:
            center_species: Species at the centre of the coordination
                environment.
            coordination_species: Species of the coordinating atoms (can
                be a string or a list of strings).
            n_coord: Number of coordinating atoms required for each
                environment.
            cutoff: Maximum distance (in Angstroms) for atoms to be
                considered coordinating.

        Returns:
            A dictionary mapping centre atom indices to lists of
            coordinating atom indices.

        Raises:
            ValueError: If center_species or coordination_species are not
                found in the structure.
        """
        if center_species not in self._atom_indices:
            raise ValueError(f"Center species '{center_species}' not found in structure")

        if isinstance(coordination_species, str):
            coordination_species = [coordination_species]

        for species in coordination_species:
            if species not in self._atom_indices:
                raise ValueError(f"Coordinating species '{species}' not found in structure")

        return get_coordination_indices(
            frac_coords=self._frac_coords,
            lattice_matrix=self._lattice_matrix,
            species=self._species,
            centre_species=center_species,
            coordination_species=coordination_species,
            cutoff=cutoff,
            n_coord=n_coord,
        )
