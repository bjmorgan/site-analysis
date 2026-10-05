"""Trajectory analysis for tracking site occupations over time.

This module provides the Trajectory class, which is responsible for analyzing
and tracking atom movements through crystallographic sites in a simulation
trajectory.

The Trajectory class manages the relationship between atoms and sites, analyzes
structures to assign atoms to sites, and records the movement history of atoms
between sites over time.

Key functionality includes:
- Assigning atoms to sites based on their positions in a structure
- Tracking atom migrations between sites over a sequence of structures
- Recording site occupation and transition data
- Supporting different site definitions via appropriate SiteCollection types

Note:
    Trajectory objects should typically be created using the TrajectoryBuilder class
    rather than directly instantiated. The builder provides an interface for
    configuring all aspects of the trajectory:

    >>> from site_analysis.builders import TrajectoryBuilder
    >>> trajectory = (TrajectoryBuilder()
    ...              .with_structure(structure)
    ...              .with_mobile_species("Li")
    ...              .with_spherical_sites(centres=[[0.5, 0.5, 0.5]], radii=[2.0])
    ...              .build())
"""

import warnings
from collections import Counter
from collections.abc import Iterable
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from pymatgen.core import Structure

from .transition_table import TransitionTable

from .atom import Atom
from .distances import mic_distance
from .dynamic_voronoi_site import DynamicVoronoiSite
from .dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection
from .polyhedral_site import PolyhedralSite
from .polyhedral_site_collection import PolyhedralSiteCollection
from .site import Site
from .site_collection import SiteCollection
from .spherical_site import SphericalSite
from .spherical_site_collection import SphericalSiteCollection
from .voronoi_site import VoronoiSite
from .voronoi_site_collection import VoronoiSiteCollection

class Trajectory:
    """Class for performing sites analysis on simulation trajectories."""

    def __init__(self,
            sites: Sequence[Site],
            atoms: list[Atom],
            commitment_radius: float | dict[str, float] | None = None) -> None:
        """Initialize a Trajectory object for site analysis of simulation trajectories.
        
        This constructor ensures all sites are of the same type and initializes the
        appropriate site collection based on the type of sites provided.
        
        Args:
            sites: list of Site objects (must all be of the same type).
            atoms: list of Atom objects to track during the trajectory analysis.
            commitment_radius: Turns on spatial commitment and sets the
                commitment radius in Å, either one value for every site or a
                dict mapping site labels to radii. Default is None
                (commitment off). See the commitment guide.

        Raises:
            ValueError: If sites or atoms list is empty, or if
                commitment_radius contains a radius that is not positive,
                lacks a radius for a site label, or is a dict while some
                sites have no label.
            TypeError: If sites contains mixed site types or an unrecognised site type.
        """
        # Validate sites is not empty
        if not sites:
            raise ValueError("Cannot initialize Trajectory with empty sites list")
        
        # Validate atoms is not empty
        if not atoms:
            raise ValueError("Cannot initialize Trajectory with empty atoms list")
        
        # ensure that all sites are of the same type
        if len(set([type(s) for s in sites])) > 1:
            raise TypeError("A Trajectory cannot be initialised with mixed Site types")
        
        # Map site types to their corresponding collection classes
        site_collection_map: dict[type[Site], type[SiteCollection]] = {
            PolyhedralSite: PolyhedralSiteCollection,
            VoronoiSite: VoronoiSiteCollection,
            SphericalSite: SphericalSiteCollection,
            DynamicVoronoiSite: DynamicVoronoiSiteCollection
        }
        
        # Find the appropriate site collection class
        site_type = type(sites[0])
        try:
            collection_class = site_collection_map[site_type]
        except KeyError:
            raise TypeError(f"Site type {site_type} not recognised for Trajectory initialisation")
        self.site_collection = collection_class(sites)
        
        self.sites = sites
        self.atoms = atoms
        self.timesteps: list[int] = []
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        self.site_lookup = {s.index: i for i, s in enumerate(sites)}
        self._commitment_radius = (dict(commitment_radius)
                                   if isinstance(commitment_radius, dict)
                                   else commitment_radius)
        self._commitment_radii = self._resolve_commitment_radii(
            sites, self._commitment_radius)

    @staticmethod
    def _resolve_commitment_radii(
            sites: Sequence[Site],
            commitment_radius: float | dict[str, float] | None,
    ) -> dict[int, float] | None:
        """Return the commitment radius for each site index.

        Args:
            sites: The sites in this trajectory.
            commitment_radius: One radius for every site, a dict mapping
                site labels to radii, or None.

        Returns:
            A dict mapping each site index to its commitment radius, or
            None if commitment_radius is None.

        Raises:
            ValueError: If any radius is not positive, if a dict lacks a
                radius for a site label, or if a dict is given while some
                sites have no label.
        """
        if commitment_radius is None:
            return None
        if isinstance(commitment_radius, dict):
            values = list(commitment_radius.values())
        else:
            values = [commitment_radius]
        if not all(value > 0 for value in values):
            raise ValueError(
                f"commitment radii must be positive, got {commitment_radius}"
            )
        if not isinstance(commitment_radius, dict):
            return {site.index: commitment_radius for site in sites}
        radii: dict[int, float] = {}
        missing: set[str] = set()
        for site in sites:
            if site.label is None:
                raise ValueError(
                    "commitment_radius can only be a dict of radii per "
                    "label if every site has a label"
                )
            if site.label in commitment_radius:
                radii[site.index] = commitment_radius[site.label]
            else:
                missing.add(site.label)
        if missing:
            raise ValueError(
                f"commitment_radius has no radius for site labels {sorted(missing)}"
            )
        return radii

    @property
    def commitment_radius(self) -> float | dict[str, float] | None:
        """The commitment radius given when this trajectory was created, or None."""
        return self._commitment_radius

    def atom_by_index(self,
            i: int) -> Atom:
        """Return the atom with the specified index.
        
        Args:
            i: Index of the atom to return.
            
        Returns:
            The Atom object with the specified index.
        """
        return self.atoms[self.atom_lookup[i]] 

    def site_by_index(self,
            i: int) -> Site:
        """Return the site with the specified index.
        
        Args:
            i: Index of the site to return.
            
        Returns:
            The Site object with the specified index.
        """
        return self.sites[self.site_lookup[i]] 

    def analyse_structure(self,
            structure: Structure) -> None:
        """Analyse a structure to assign atoms to sites.
        
        This delegates the analysis to the site collection's analyse_structure method.
        
        Args:
            structure: A pymatgen Structure object to be analysed.
        """
        self.site_collection.analyse_structure(self.atoms, structure)
        
    def assign_site_occupations(self,
            structure: Structure) -> None:
        """Assign atoms to sites for a specific structure.

        This delegates the assignment to the site collection's assign_site_occupations method.

        Args:
            structure: A pymatgen Structure object to be analysed.
        """
        self.site_collection.assign_site_occupations(self.atoms, structure.lattice.matrix)
                    
    def site_coordination_numbers(self) -> Counter:
        """Return the coordination numbers of all sites.
        
        Returns:
            A Counter object mapping coordination numbers to their frequencies.
        """
        return Counter([s.coordination_number for s in self.sites])

    def site_labels(self) -> list[str | None]:
        """Return the labels of all sites.
        
        Returns:
            A list of site labels (or None for sites without labels).
        """
        return [s.label for s in self.sites]

    @staticmethod
    def _validate_destination(site_index: int, dest: int, valid_indices: set[int]) -> None:
        """Raise ValueError if dest is not in valid_indices."""
        if dest not in valid_indices:
            raise ValueError(
                f"Site {site_index} has a transition to unknown "
                f"site index {dest}."
            )

    def transition_counts_by_site(
        self,
        *,
        keys: Sequence[int] | None = None,
    ) -> TransitionTable[int]:
        """Return per-site transition counts as a :class:`TransitionTable`.

        Args:
            keys: Optional key ordering for rows and columns. If ``None``,
                keys are sorted. Must be a permutation of the default keys.

        Returns:
            A :class:`TransitionTable` of integer counts keyed by site index.

        Raises:
            ValueError: If *keys* does not match the default key set, or
                if a site has a transition to an unknown site index.
        """
        site_keys = tuple(sorted(s.index for s in self.sites))
        index_set = set(site_keys)
        idx_lookup = {k: i for i, k in enumerate(site_keys)}
        n = len(site_keys)
        matrix = np.zeros((n, n), dtype=int)
        for site in self.sites:
            row = idx_lookup[site.index]
            for dest, count in site.transitions.items():
                self._validate_destination(site.index, dest, index_set)
                matrix[row, idx_lookup[dest]] = count
        table = TransitionTable(keys=site_keys, matrix=matrix)
        if keys is not None:
            return table.reorder(keys)
        return table

    def transition_counts_by_label(
        self,
        *,
        keys: Sequence[str] | None = None,
    ) -> TransitionTable[str]:
        """Return label-aggregated transition counts as a :class:`TransitionTable`.

        Sites without labels are skipped. A warning is emitted if any
        transitions are dropped as a result.

        Args:
            keys: Optional key ordering for rows and columns. If ``None``,
                keys are sorted. Must be a permutation of the default keys.

        Returns:
            A :class:`TransitionTable` of integer counts keyed by site label.

        Raises:
            ValueError: If *keys* does not match the default key set, or
                if a site has a transition to an unknown site index.
        """
        all_site_indices = {s.index for s in self.sites}
        index_to_label = {
            s.index: s.label for s in self.sites if s.label is not None
        }
        label_keys = tuple(sorted(set(index_to_label.values())))
        label_lookup = {k: i for i, k in enumerate(label_keys)}
        n = len(label_keys)
        matrix = np.zeros((n, n), dtype=int)
        dropped = 0
        for site in self.sites:
            src_label = index_to_label.get(site.index)
            if src_label is None:
                for dest in site.transitions:
                    self._validate_destination(site.index, dest, all_site_indices)
                dropped += sum(site.transitions.values())
                continue
            for dest, count in site.transitions.items():
                self._validate_destination(site.index, dest, all_site_indices)
                dest_label = index_to_label.get(dest)
                if dest_label is None:
                    dropped += count
                    continue
                matrix[label_lookup[src_label], label_lookup[dest_label]] += count
        if dropped > 0:
            warnings.warn(
                f"{dropped} transition(s) involving unlabelled sites were "
                f"excluded from the label-aggregated counts.",
                stacklevel=2,
            )
        table = TransitionTable(keys=label_keys, matrix=matrix)
        if keys is not None:
            return table.reorder(keys)
        return table

    def transition_probabilities_by_site(
        self,
        *,
        keys: Sequence[int] | None = None,
    ) -> TransitionTable[int]:
        """Return per-site row-normalised transition probabilities.

        Each row is normalised so that its values sum to 1.0. Rows with no
        outgoing transitions remain as all zeros.

        Args:
            keys: Optional key ordering for rows and columns. If ``None``,
                keys are sorted. Must be a permutation of the default keys.

        Returns:
            A :class:`TransitionTable` of float probabilities keyed by site index.

        Raises:
            ValueError: If *keys* does not match the default key set, or
                if a site has a transition to an unknown site index.
        """
        return self.transition_counts_by_site(keys=keys).normalise_rows()

    def transition_probabilities_by_label(
        self,
        *,
        keys: Sequence[str] | None = None,
    ) -> TransitionTable[str]:
        """Return label-aggregated row-normalised transition probabilities.

        Each row is normalised so that its values sum to 1.0. Rows with no
        outgoing transitions remain as all zeros. Sites without labels are
        skipped. A warning is emitted if any transitions are dropped.

        Args:
            keys: Optional key ordering for rows and columns. If ``None``,
                keys are sorted. Must be a permutation of the default keys.

        Returns:
            A :class:`TransitionTable` of float probabilities keyed by site label.

        Raises:
            ValueError: If *keys* does not match the default key set, or
                if a site has a transition to an unknown site index.
        """
        return self.transition_counts_by_label(keys=keys).normalise_rows()

    @property
    def atom_sites(self) -> list[int | None]:
        """Return the sites that each atom currently occupies.
        
        Returns:
            A list of site indices (or None for unoccupied atoms), one for each atom.
        """
        return [atom.in_site for atom in self.atoms]
        
    @property
    def site_occupations(self) -> list[list[int]]:
        """Return the atoms occupying each site.
        
        Returns:
            A list of lists, where each inner list contains the indices of atoms
            occupying a site.
        """
        return [s.contains_atoms for s in self.sites]

    def append_timestep(self,
        structure: Structure,
        t: int | None=None) -> None:
        """Append a new timestep to the trajectory.

        This method:

        1. Analyses the structure to assign atoms to sites
        2. Records transitions and updates each assigned atom's recent-site
           history. Without commitment, a transition is recorded for each
           atom assigned to a different site from the last site it was
           assigned to in an appended timestep. With commitment, each atom's
           site is replaced by its committed site, and a transition is
           recorded when an atom's committed site changes from one site to
           another.
        3. Updates the trajectory information for atoms and sites
        4. Adds the timestep to the list of timesteps if provided

        Args:
            structure: A pymatgen Structure object for this timestep.
            t: Optional timestep index to record. If None, no timestep is recorded.
        """
        self.analyse_structure(structure)
        if self._commitment_radii is None:
            self._record_transitions()
        else:
            self._apply_commitment(self._commitment_radii, structure.lattice.matrix)
        for atom in self.atoms:
            atom.trajectory.append(atom.in_site)
        for site in self.sites:
            site.trajectory.append(site.contains_atoms)
        if t is not None:
            self.timesteps.append(t)

    def _record_transitions(self) -> None:
        """Record a transition for each atom that has moved to a new site.

        A transition is recorded from the last site an atom was assigned to in
        an appended timestep to its current site, and the atom's recent-site
        history is updated.
        """
        for atom in self.atoms:
            if atom.in_site is not None:
                previous_site_index = atom.most_recent_site
                if previous_site_index is not None and previous_site_index != atom.in_site:
                    self.site_by_index(previous_site_index).transitions[atom.in_site] += 1
                atom.update_recent_site(atom.in_site)

    def _apply_commitment(self,
            radii: dict[int, float],
            lattice_matrix: np.ndarray) -> None:
        """Replace each atom's assigned site with its committed site.

        An atom with no committed site commits to the site it is assigned to.
        An atom assigned to a different site from its committed site commits
        to that site when it is within the site's commitment radius of the
        site centre, and a transition is recorded. Otherwise the atom keeps
        its committed site. Site occupations are then rebuilt from the
        committed sites. Each atom's recent-site history follows the site it
        is assigned to.

        Args:
            radii: Commitment radius for each site index.
            lattice_matrix: (3, 3) lattice matrix of the analysed structure.
        """
        for atom in self.atoms:
            assigned = atom.in_site
            if assigned is not None:
                atom.update_recent_site(assigned)
            committed = atom.committed_site
            if committed is None:
                committed = assigned
            elif assigned is not None and assigned != committed:
                site = self.site_by_index(assigned)
                distance = mic_distance(atom.frac_coords, site.centre, lattice_matrix)
                if distance <= radii[assigned]:
                    self.site_by_index(committed).transitions[assigned] += 1
                    committed = assigned
            atom.committed_site = committed
            atom.in_site = committed
        self.site_collection.reset_site_occupations()
        for atom in self.atoms:
            if atom.in_site is not None:
                self.site_by_index(atom.in_site).contains_atoms.append(atom.index)

    def reset(self) -> None:
        """Reset the trajectory.
        
        This clears all trajectory information for atoms and sites and
        resets the timesteps list.
        """
        for atom in self.atoms:
            atom.reset()
        self.site_collection.reset()
        self.timesteps = []

    @property
    def atoms_trajectory(self):
        """Return the trajectory of all atoms.
        
        Returns:
            A list of lists, where each inner list represents a timestep and
            contains the site indices occupied by each atom at that timestep.
        """
        return list(map(list, zip(*[atom.trajectory for atom in self.atoms])))

    @property
    def sites_trajectory(self):
        """Return the trajectory of all sites.
        
        Returns:
            A list of lists, where each inner list represents a timestep and
            contains the atom indices occupying each site at that timestep.
        """
        return list(map(list, zip(*[site.trajectory for site in self.sites])))

    @property
    def at(self):
        """Shorthand for atoms_trajectory.
        
        Returns:
            The atoms_trajectory property.
        """
        return self.atoms_trajectory

    @property
    def st(self):
        """Shorthand for sites_trajectory.
        
        Returns:
            The sites_trajectory property.
        """
        return self.sites_trajectory

    def trajectory_from_structures(
        self,
        structures: Sequence[Structure],
        progress: bool = False,
    ) -> None:
        """Generate a trajectory from a list of structures.

        Args:
            structures: The structures to analyse, one per timestep.
            progress: Show a progress bar. Automatically selects the
                appropriate widget for terminal or notebook environments.
        """
        generator: Iterable[tuple[int, Structure]] = enumerate(structures, 1)
        if progress:
            generator = tqdm(generator,
                             total=len(structures),
                             unit=' steps',
                             desc='Analysing trajectory')
        for timestep, s in generator:
            self.append_timestep(s, t=timestep)
   
    def __len__(self):
        """Return the length of the trajectory.
        
        Returns:
            The number of timesteps in the trajectory.
        """
        return len(self.timesteps)
        
    def site_summaries(self,
        metrics: list[str] | None = None) -> list[dict]:
        """Generate summary statistics for all sites in this trajectory.
        
        Args:
            metrics: List of metrics to include for each site. None returns 
                default metrics for each site.
                
        Returns:
            List of summary dicts, one per site, in site order.
        """
        return self.site_collection.summaries(metrics=metrics)
        
    def write_site_summaries(self,
        filename: str,
        metrics: list[str] | None = None) -> None:
        """Write site summaries to a JSON file.
        
        Args:
            filename: Path to output JSON file.
            metrics: List of metrics to include for each site. None returns 
                default metrics for each site.
        """
        import json
        
        summaries = self.site_summaries(metrics=metrics)
        
        with open(filename, 'w') as f:
            json.dump(summaries, f, indent=2)
