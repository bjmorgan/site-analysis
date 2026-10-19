"""Reference-based workflow for defining sites in crystal structures.

This module provides the ReferenceBasedSites class, which is the main orchestrator
for defining crystallographic sites in a target structure based on coordination
environments identified in a reference structure. This approach is particularly
useful for:

1. Analysing structures with distortions relative to an ideal reference
2. Tracking specific site types across different structures or simulation frames
3. Creating consistent site definitions across diverse structures

The ReferenceBasedSites class integrates several components to accomplish this workflow:
- StructureAligner: Finds the translation that superimposes the reference on the target
- CoordinationEnvironmentFinder: Identifies coordination environments in the reference
- IndexMapper: Maps atom indices between reference and target structures
- SiteFactory: Creates appropriate site objects in the target structure

This approach lets users define sites based on ideal coordination environments in
a reference structure, then create corresponding sites in real or distorted structures
where those same environments might be harder to identify directly.
"""

import numpy as np
from typing import Any, Callable, Literal, TypeVar

from pymatgen.core import Structure

from site_analysis.reference_workflow.structure_aligner import (
    StructureAligner, _check_deprecated_alignment_options)
from site_analysis.reference_workflow.coord_finder import CoordinationEnvironmentFinder
from site_analysis.reference_workflow.index_mapper import IndexMapper, _check_one_to_one
from site_analysis.reference_workflow.site_factory import SiteFactory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site

SiteT = TypeVar('SiteT', bound=Site)


def _environment_error(
        verb: Literal["find", "map"],
        centre_species: str,
        coordination_species: str | list[str],
        error: Exception) -> ValueError:
    """Create the error for coordination environments that could not be found or mapped.

    Args:
        verb: What failed, ``"find"`` or ``"map"``.
        centre_species: The species at the centres of the environments.
        coordination_species: The coordinating species, a string or a list
            of strings (joined with ", " in the message).
        error: The error that caused the failure.

    Returns:
        A ValueError whose message names the verb, the species and
        ``error``.
    """
    species = (coordination_species if isinstance(coordination_species, str)
               else ", ".join(coordination_species))
    return ValueError(
        f"Failed to {verb} coordination environments for {centre_species} centres "
        f"and {species} coordinating atoms: {error}")


class ReferenceBasedSites:
    """Main orchestrator for defining sites using a reference structure approach.
    
    This class ties together all components needed to define sites in crystal structures
    using a reference structure as a template:
    
    1. StructureAligner - to find the translation that superimposes the reference structure on the target
    2. CoordinationEnvironmentFinder - to find coordination environments in the reference
    3. IndexMapper - to map environments from reference to target structure
    4. SiteFactory - to create appropriate site objects
    
    All ``create_*`` calls on an instance share one reference-to-target
    mapping, which each call checks and extends.

    Attributes:
        reference_structure: Reference structure defining ideal coordination environments.
        target_structure: Target structure where sites will be created.
        aligned_structure: Reference structure translated to match the target
            (None if align=False).
        translation_vector: Fractional translation added to the reference
            coordinates to align them with the target, wrapped into the unit cell
            (None if align=False).
        alignment_metrics: Alignment quality, with keys 'rmsd', 'max_dist'
            and 'mean_dist' in Angstrom (None if align=False).
    """
    
    def __init__(self, 
            reference_structure: Structure, 
            target_structure: Structure, 
            align: bool = True, 
            align_species: str | list[str] | None = None, 
            align_metric: str | None = None,
            align_algorithm: str | None = None,
            align_minimizer_options: dict[str, Any] | None = None,
            align_tolerance: float | None = None) -> None:
        """Initialise ReferenceBasedSites with reference and target structures.
        
        Args:
            reference_structure: Reference structure defining ideal coordination environments
            target_structure: Target structure where sites will be created
            align: Whether to perform structure alignment. Default is True.
            align_species: Species to use for alignment, as a list or a single
                string. Default is all species, in which case both structures
                must have the same composition.
            align_metric: Deprecated; has no effect, except that 'max_dist'
                raises ValueError. Will be removed in version 2.0.
            align_algorithm: Deprecated; has no effect. Will be removed in
                version 2.0.
            align_minimizer_options: Deprecated; has no effect. Will be
                removed in version 2.0.
            align_tolerance: Deprecated; has no effect. Will be removed in
                version 2.0.

        Raises:
            ValueError: If align_metric is 'max_dist', which is no longer
                supported, or if alignment fails.
        """
        _check_deprecated_alignment_options(
            align_metric=align_metric, align_algorithm=align_algorithm,
            align_minimizer_options=align_minimizer_options, align_tolerance=align_tolerance)
        if isinstance(align_species, str):
            align_species = [align_species]

        self.reference_structure = reference_structure
        self.target_structure = target_structure

        # Eagerly extract arrays from structures
        self._ref_frac_coords: np.ndarray = reference_structure.frac_coords
        self._ref_lattice_matrix: np.ndarray = reference_structure.lattice.matrix
        self._ref_species = [s.species_string for s in reference_structure]
        self._target_frac_coords = target_structure.frac_coords
        self._target_species = [s.species_string for s in target_structure]

        # Initialise alignment attributes
        self.aligned_structure: Structure | None = None
        self.translation_vector: np.ndarray | None = None
        self.alignment_metrics: dict[str, float] | None = None
        self._aligned_frac_coords: np.ndarray | None = None

        # Perform alignment if requested
        if align:
            self._align_structures(align_species)

        # These will be initialised on first use
        self._coord_finder: CoordinationEnvironmentFinder | None = None
        self._index_mapper: IndexMapper | None = None
        self._site_factory: SiteFactory | None = None

        # The mapping between the atoms of the sites created by the
        # create_* methods: the target atom index of each reference atom,
        # or -1 if unmapped
        self._target_of_ref = np.full(len(self._ref_frac_coords), -1)
        
    @property
    def _effective_ref_coords(self) -> np.ndarray:
        """Reference coordinates to use, preferring aligned if available."""
        if self._aligned_frac_coords is not None:
            return self._aligned_frac_coords
        return self._ref_frac_coords

    def create_polyhedral_sites(self,
                            center_species: str, 
                            vertex_species: str | list[str], 
                            cutoff: float, 
                            n_vertices: int,
                            label: str | None = None, 
                            labels: list[str] | None = None, 
                            target_species: str | list[str] | None = None,
                            use_reference_centers: bool = True) -> list[PolyhedralSite]:
        """Create PolyhedralSite objects based on coordination environments in the reference structure.
        
        Args:
            center_species: Species at the centre of coordination environments
            vertex_species: Species at vertices of coordination environments
            cutoff: Cutoff distance for coordination environment (required)
            n_vertices: Number of vertices per environment (required)
            label: Label to apply to all created sites. Default is None.
            labels: List of labels for each site. Default is None.
            target_species: Species to map to in the target structure. Default is None.
            use_reference_centers: Whether to use reference centres for 
                PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
                Default is True.
            
        Returns:
            List of PolyhedralSite objects
            
        Raises:
            ValueError: If coordination environments cannot be found or mapped,
                if a target atom is mapped from two different reference
                atoms, or a reference atom to two different target atoms,
                counting the mapping held from earlier calls (a 1:1 mapping
                violation), or if both label and labels are provided.
        """
        return self._create_sites(
            self._initialise_site_factory().create_polyhedral_sites,
            center_species=center_species,
            coordination_species=vertex_species,
            cutoff=cutoff,
            n_coord=n_vertices,
            label=label,
            labels=labels,
            target_species=target_species,
            use_reference_centers=use_reference_centers)

    def create_dynamic_voronoi_sites(self, 
                                center_species: str, 
                                reference_species: str | list[str], 
                                cutoff: float, 
                                n_reference: int,
                                label: str | None = None, 
                                labels: list[str] | None = None, 
                                target_species: str | list[str] | None = None,
                                use_reference_centers: bool = True) -> list[DynamicVoronoiSite]:
        """Create DynamicVoronoiSite objects based on coordination environments in the reference structure.
        
        Args:
            center_species: Species at the centre of coordination environments
            reference_species: Species of reference atoms used to define the dynamic site centres
            cutoff: Cutoff distance for finding reference atoms (required)
            n_reference: Number of reference atoms per site (required)
            label: Label to apply to all created sites. Default is None.
            labels: List of labels for each site. Default is None.
            target_species: Species to map to in the target structure. Default is None.
            use_reference_centers: Whether to use reference centres for 
                PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
                Default is True.
            
        Returns:
            List of DynamicVoronoiSite objects
            
        Raises:
            ValueError: If coordination environments cannot be found or mapped,
                if a target atom is mapped from two different reference
                atoms, or a reference atom to two different target atoms,
                counting the mapping held from earlier calls (a 1:1 mapping
                violation), or if both label and labels are provided.
        """
        return self._create_sites(
            self._initialise_site_factory().create_dynamic_voronoi_sites,
            center_species=center_species,
            coordination_species=reference_species,
            cutoff=cutoff,
            n_coord=n_reference,
            label=label,
            labels=labels,
            target_species=target_species,
            use_reference_centers=use_reference_centers)

    def _create_sites(self,
            create: Callable[..., list[SiteT]],
            center_species: str,
            coordination_species: str | list[str],
            cutoff: float,
            n_coord: int,
            label: str | None,
            labels: list[str] | None,
            target_species: str | list[str] | None,
            use_reference_centers: bool) -> list[SiteT]:
        """Create sites from coordination environments in the reference structure.

        Finds the environments, maps them to the target structure, checks
        that their atom pairs and the held mapping are one-to-one, creates
        the sites with ``create``, then adds the pairs to the held mapping.

        Args:
            create: The SiteFactory method that creates the sites.
            center_species: Species at the centre of coordination environments.
            coordination_species: Coordination atom species.
            cutoff: Cutoff distance for coordination environment.
            n_coord: Number of coordination atoms per environment.
            label: Label to apply to all created sites.
            labels: List of labels for each site.
            target_species: Species to map to in the target structure.
            use_reference_centers: Whether to pass reference centres to
                ``create``.

        Returns:
            The sites returned by ``create``.

        Raises:
            ValueError: If coordination environments cannot be found or
                mapped, if the atom pairs and the held mapping are not
                one-to-one, or if ``create`` raises.
        """
        ref_environments = self._find_coordination_environments(
            center_species, coordination_species, cutoff, n_coord)
        self._validate_unique_environments(ref_environments)
        if use_reference_centers:
            reference_centers = self._calculate_reference_centers_from_indices(
                list(ref_environments))
        else:
            reference_centers = None
        ref_environment_list = list(ref_environments.values())
        ref_atoms = np.array(ref_environment_list, dtype=int).reshape(-1)
        try:
            mapped_environments = self._map_environments(ref_environment_list, target_species)
            target_atoms = np.array(mapped_environments, dtype=int).reshape(-1)
            self._check_pairs(ref_atoms, target_atoms)
        except ValueError as error:
            raise _environment_error(
                "map", center_species, coordination_species, error) from error
        sites = create(
            mapped_environments,
            reference_centers=reference_centers,
            label=label,
            labels=labels
        )
        self._target_of_ref[ref_atoms] = target_atoms
        return sites

    def mapped_reference_frac_coords(self) -> np.ndarray:
        """Return the fractional coordinates of the reference atom mapped to each target atom.

        The mapping holds the coordinating atoms of the sites created by
        ``create_polyhedral_sites`` and ``create_dynamic_voronoi_sites``
        (the centre atoms are not mapped, so their rows are NaN); a call
        that raises adds none.

        Returns:
            A new array of shape ``(n_target_atoms, 3)``. Row ``t`` holds the
            coordinates of the reference atom mapped to target atom ``t``
            (aligned to the target when alignment is on), or NaN if target
            atom ``t`` is not mapped.
        """
        coords = np.full((len(self._target_frac_coords), 3), np.nan)
        refs, targets = self._held_pairs()
        coords[targets] = self._effective_ref_coords[refs]
        return coords

    def _held_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """The reference-target atom pairs of the held mapping.

        Returns:
            The mapped reference atom indices, in ascending order, and the
            target atom index paired with each.
        """
        refs = np.flatnonzero(self._target_of_ref != -1)
        return refs, self._target_of_ref[refs]

    def _align_structures(self, align_species: list[str] | None = None) -> None:
        """Align the reference structure to the target structure.

        Sets ``translation_vector``, ``alignment_metrics`` and
        ``aligned_structure``, the reference translated onto the target.

        Alignment works in fractional coordinates: the reference's lattice
        scales the distances used for the thresholds and metrics, and the
        target's lattice is not used.

        Args:
            align_species: Species to align on. Default is all species.

        Raises:
            ValueError: If alignment fails.
        """
        try:
            translation_vector, metrics = StructureAligner().align(
                self._ref_frac_coords, self._ref_species,
                self._target_frac_coords, self._target_species,
                self._ref_lattice_matrix, species=align_species)
        except ValueError as e:
            raise ValueError(f"Structure alignment failed: {str(e)}") from e
        self.translation_vector = translation_vector
        self.alignment_metrics = metrics
        self._aligned_frac_coords = (self._ref_frac_coords + translation_vector) % 1.0
        aligned_structure = self.reference_structure.copy()
        for i, site in enumerate(aligned_structure):
            aligned_structure[i] = site.species, self._aligned_frac_coords[i]
        self.aligned_structure = aligned_structure

    def _find_coordination_environments(self, 
                                      center_species: str, 
                                      coordination_species: str | list[str], 
                                      cutoff: float, 
                                      n_coord: int) -> dict[int, list[int]]:
        """Find coordination environments in the reference structure.
        
        Args:
            center_species: Species at the centre of coordination environments
            coordination_species: Coordination atom species
            cutoff: Cutoff distance for coordination environment
            n_coord: Number of coordination atoms per environment
            
        Returns:
            Dictionary mapping centre atom indices to lists of coordinating atom indices.
            Keys are indices of centre atoms, values are lists of coordinating atom indices.
            
        Raises:
            ValueError: If coordination environments cannot be found.
        """
        try:
            # Find coordination environments
            coord_finder = self._initialise_coord_finder()
            environments_dict = coord_finder.find_environments(
                center_species=center_species,
                coordination_species=coordination_species,
                n_coord=n_coord,
                cutoff=cutoff
            )
            
            return environments_dict
            
        except ValueError as e:
            raise _environment_error(
                "find", center_species, coordination_species, e) from e
    
    def _map_environments(self, 
        ref_environments: list[list[int]], 
        target_species: str | list[str] | None = None) -> list[list[int]]:
        """Map coordination environments from reference to target structure.
        
        Args:
            ref_environments: List of environments from reference structure
            target_species: Species to map to in the target structure. Default is None.
            
        Returns:
            List of mapped environments for the target structure
            
        Raises:
            ValueError: If environments cannot be mapped between structures.
        """
        # If no environments were found, return an empty list immediately
        if not ref_environments:
            return []
        
        # Create index mapper if not already initialised
        if self._index_mapper is None:
            self._index_mapper = IndexMapper()

        return self._index_mapper.map_coordinating_atoms(
            ref_frac_coords=self._effective_ref_coords,
            target_frac_coords=self._target_frac_coords,
            lattice_matrix=self._ref_lattice_matrix,
            ref_coordinating=ref_environments,
            target_species=self._target_species,
            species_filter=target_species,
        )

    def _check_pairs(self, ref_atoms: np.ndarray, target_atoms: np.ndarray) -> None:
        """Check that reference-target atom pairs and the held mapping are one-to-one.

        Args:
            ref_atoms: Reference atom indices.
            target_atoms: The target atom index paired with each reference
                atom.

        Raises:
            ValueError: If the pairs and the held mapping together pair a
                reference atom with two target atoms, or a target atom with
                two reference atoms (a 1:1 mapping violation).
        """
        held_refs, held_targets = self._held_pairs()
        _check_one_to_one(
            np.concatenate([held_refs, ref_atoms]),
            np.concatenate([held_targets, target_atoms]))

    def _initialise_site_factory(self) -> SiteFactory:
        """Initialise the site factory if not already done.
        
        Note: This method exists primarily for testing purposes.
        """
        if self._site_factory is None:
            self._site_factory = SiteFactory(self.target_structure)
        return self._site_factory

    def _initialise_coord_finder(self) -> CoordinationEnvironmentFinder:
        """Initialise the coordination environment finder if not already done."""
        if self._coord_finder is None:
            self._coord_finder = CoordinationEnvironmentFinder(self.reference_structure)
        return self._coord_finder

    def _validate_unique_environments(self, environments: dict[int, list[int]]) -> None:
        """Validate that each environment contains unique atom indices.
        
        Args:
            environments: Dict of environments, where keys are centre atom indices 
                and values are lists of coordinating atom indices.
            
        Raises:
            ValueError: If any environment contains duplicate atom indices.
        """
        for center_idx, env in environments.items():
            if len(env) != len(set(env)):
                # Find the duplicates
                counts: dict[int, int] = {}
                for idx in env:
                    counts[idx] = counts.get(idx, 0) + 1
                duplicates = [idx for idx, count in counts.items() if count > 1]
                
                raise ValueError(
                    f"Environment for centre atom {center_idx} contains duplicate atom indices {duplicates}. "
                    f"This typically occurs in small unit cells where the same atom "
                    f"appears as a neighbour in multiple periodic images. "
                    f"Please use a larger supercell for your analysis."
                )
    def _calculate_reference_centers_from_indices(self, center_indices: list[int]) -> list[np.ndarray]:
        """Calculate reference centres from centre atom indices."""
        coords = self._effective_ref_coords
        return [coords[i].copy() for i in center_indices]
