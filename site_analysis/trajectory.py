"""Trajectory analysis for tracking site occupations over time.

This module provides the Trajectory class, which is responsible for analysing
and tracking atom movements through crystallographic sites in a simulation
trajectory.

The Trajectory class manages the relationship between atoms and sites, analyses
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
from numbers import Real
from typing import Sequence

import numpy as np
from scipy.sparse import csr_array
from tqdm.auto import tqdm

from pymatgen.core import Structure

from .transition_events import TransitionEvents
from .transition_table import TransitionTable

from .atom import Atom
from .distances import mic_distance
from .faces import Face
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
        """Initialise a Trajectory object for site analysis of simulation trajectories.
        
        This constructor ensures all sites are of the same type and initialises the
        appropriate site collection based on the type of sites provided. The atoms
        and sites are reset, so the analysis starts afresh: any trajectories,
        recent-site histories, committed sites and transition counts they hold
        from an earlier analysis are cleared, and each atom's current site and
        coordinates are unset until the first structure is analysed.
        
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
            TypeError: If sites contains mixed site types or an unrecognised
                site type, or if any commitment radius is not a number.
        """
        # Validate sites is not empty
        if not sites:
            raise ValueError("Cannot initialise Trajectory with empty sites list")
        
        # Validate atoms is not empty
        if not atoms:
            raise ValueError("Cannot initialise Trajectory with empty atoms list")
        
        # ensure that all sites are of the same type
        site_types = {type(site) for site in sites}
        if len(site_types) > 1:
            raise TypeError(
                "A Trajectory cannot be initialised with mixed Site types: "
                f"{', '.join(sorted(site_type.__name__ for site_type in site_types))}")
        
        # Map site types to their corresponding collection classes
        site_collection_map: dict[type[Site], type[SiteCollection]] = {
            PolyhedralSite: PolyhedralSiteCollection,
            VoronoiSite: VoronoiSiteCollection,
            SphericalSite: SphericalSiteCollection,
            DynamicVoronoiSite: DynamicVoronoiSiteCollection
        }
        
        # Find the appropriate site collection class
        (site_type,) = site_types
        try:
            collection_class = site_collection_map[site_type]
        except KeyError:
            raise TypeError(f"Site type {site_type} not recognised for Trajectory initialisation")
        self.site_collection = collection_class(sites)
        
        self.sites = self.site_collection.sites
        self.atoms = atoms
        self.timesteps: list[int] = []
        self.atom_lookup = {a.index: i for i, a in enumerate(atoms)}
        self.site_lookup = {s.index: i for i, s in enumerate(self.sites)}
        self._commitment_radius = (dict(commitment_radius)
                                   if isinstance(commitment_radius, dict)
                                   else commitment_radius)
        self._commitment_radii = self._resolve_commitment_radii(
            sites, self._commitment_radius)
        self.reset()

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
            TypeError: If any radius is not a number.
        """
        if commitment_radius is None:
            return None
        if isinstance(commitment_radius, dict):
            values = list(commitment_radius.values())
        else:
            values = [commitment_radius]
        if not all(isinstance(value, Real) and not isinstance(value, bool)
                   for value in values):
            raise TypeError(
                "commitment_radius must be a number or a dict mapping site "
                f"labels to numbers, got {commitment_radius!r}"
            )
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
        if isinstance(self._commitment_radius, dict):
            return dict(self._commitment_radius)
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

        Raises:
            ValueError: If the sites are polyhedral and a site has a vertex
                index that is not an atom of the structure, or no shared
                faces have been found and finding them from the structure
                fails (see ``find_shared_faces``).
            RuntimeError: If the sites are polyhedral and finding the shared
                faces fails on degenerate vertex geometry, or a site tested
                for an atom is too distorted for its triangles to enclose
                its centre.
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
                    
    def _polyhedral_site_collection(self) -> PolyhedralSiteCollection:
        """Return the site collection, which must be of polyhedral sites.

        Returns:
            The site collection.

        Raises:
            TypeError: If the sites are not polyhedral.
        """
        if not isinstance(self.site_collection, PolyhedralSiteCollection):
            raise TypeError(
                "Shared faces are only available for polyhedral sites, "
                f"not {type(self.site_collection).__name__}")
        return self.site_collection

    def find_shared_faces(self,
            structure: Structure) -> None:
        """Find the faces shared by pairs of polyhedral sites, and fix each site's triangles.

        Two sites share a face when they share three or more vertex atoms
        under one lattice translation. Each site's triangles are fixed from
        the structure, with each shared face triangulated identically by the
        sites sharing it (see ``PolyhedralSiteCollection.find_shared_faces``).
        The structure's atoms must be in the trajectory's atom order, so
        that its atom indices are the sites' vertex indices. The result
        replaces any earlier one, and is kept by ``reset()``. If no shared
        faces have been found when a structure is first analysed, they are
        found from that structure.

        Args:
            structure: A pymatgen Structure object.

        Raises:
            TypeError: If the sites are not polyhedral.
            ValueError: If a site has a vertex index that is not an atom of
                the structure; the vertex coordinates are not finite; two
                sites coincide or one lies inside another; the atoms two
                sites share cannot be triangulated identically on both; or
                two sites, or a site and a periodic image of itself,
                overlap. The earlier result is then kept.
            RuntimeError: If a site's vertices are degenerate. The earlier
                result is then kept.
        """
        self._polyhedral_site_collection().find_shared_faces(
            structure.frac_coords, structure.lattice.matrix)

    def shared_faces(self) -> tuple[Face, ...]:
        """Return the faces shared by pairs of polyhedral sites.

        The faces are those found most recently by ``TrajectoryBuilder.build()``
        or a successful call to ``find_shared_faces``. A failed call to
        ``find_shared_faces`` keeps the earlier result.

        Returns:
            The shared faces, sorted by site indices then vertex indices.

        Raises:
            TypeError: If the sites are not polyhedral.
            RuntimeError: If the shared faces have not been found.
        """
        return self._polyhedral_site_collection().shared_faces()

    def site_adjacency(self) -> csr_array:
        """Return which polyhedral sites share a face.

        The faces are those found most recently by ``TrajectoryBuilder.build()``
        or a successful call to ``find_shared_faces``. A failed call to
        ``find_shared_faces`` keeps the earlier result.

        Returns:
            A new symmetric boolean sparse array of shape ``(n, n)``, where
            ``n`` is one more than the largest site index. Entry ``(i, j)``
            is True when the sites with indices ``i`` and ``j`` share a face.

        Raises:
            TypeError: If the sites are not polyhedral.
            RuntimeError: If the shared faces have not been found.
        """
        return self._polyhedral_site_collection().site_adjacency()

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
    def geometric_sites(self) -> list[int | None]:
        """Return the site each atom is inside.

        This is the site each atom is inside in the most recently analysed
        structure (None if it is in no site), one for each atom in the order
        of ``atoms``. With commitment it can differ from
        ``committed_sites``.

        Returns:
            A list of site indices (or None), one for each atom.
        """
        return [atom.in_site for atom in self.atoms]

    @property
    def committed_sites(self) -> list[int | None]:
        """Return the committed site of each atom.

        This is each atom's committed site as of the last appended
        timestep, one for each atom in the order of ``atoms``. It is None
        for every atom without commitment, and for an atom not yet
        assigned.

        Returns:
            A list of site indices (or None), one for each atom.
        """
        return [atom.committed_site for atom in self.atoms]

    @property
    def atom_sites(self) -> list[int | None]:
        """Return the site each atom is counted in.

        Deprecated: use ``geometric_sites`` or ``committed_sites``. This gives
        the committed sites with commitment and the geometric sites without,
        as in version 1.10.0 after ``append_timestep()``.

        Returns:
            The same list as ``committed_sites`` with commitment, and as
            ``geometric_sites`` without.
        """
        warnings.warn(
            "Trajectory.atom_sites is deprecated and will be removed in "
            "version 2.0; use Trajectory.geometric_sites (or "
            "Trajectory.committed_sites).",
            FutureWarning,
            stacklevel=2)
        if self._commitment_radii is None:
            return self.geometric_sites
        return self.committed_sites

    @property
    def site_occupations(self) -> list[list[int]]:
        """Return the atoms occupying each site.

        With commitment, after ``append_timestep()`` these are the atoms
        committed to each site; after ``analyse_structure()`` alone they are
        the atoms inside each site.

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
           committed site is updated, and a transition is recorded when it
           changes from one site to another. Site occupations are then
           rebuilt from the committed sites, and each atom's recent-site
           history follows the site it is inside.
        3. Updates the trajectory information for atoms and sites. With
           commitment, each atom's trajectory records its committed site and
           its geometric trajectory the site it is inside.
        4. Adds the timestep to the list of timesteps if provided

        Args:
            structure: A pymatgen Structure object for this timestep.
            t: Optional timestep index to record. If None, no timestep is recorded.

        Raises:
            ValueError: If the sites are polyhedral and a site has a vertex
                index that is not an atom of the structure, or no shared
                faces have been found and finding them from the structure
                fails (see ``find_shared_faces``).
            RuntimeError: If the sites are polyhedral and finding the shared
                faces fails on degenerate vertex geometry, or a site tested
                for an atom is too distorted for its triangles to enclose
                its centre.
        """
        self.analyse_structure(structure)
        if self._commitment_radii is None:
            self._record_transitions()
            for atom in self.atoms:
                atom.trajectory.append(atom.in_site)
        else:
            self._apply_commitment(self._commitment_radii, structure.lattice.matrix)
            for atom in self.atoms:
                atom.trajectory.append(atom.committed_site)
                atom.geometric_trajectory.append(atom.in_site)
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
        """Update each atom's committed site.

        An atom with no committed site commits to the site it is inside, if
        any. An atom inside a different site from its committed site commits
        to that site when it is within the site's commitment radius of the
        site centre, and a transition is recorded. Otherwise the atom keeps
        its committed site. Site occupations are then rebuilt from the
        committed sites. Each atom's recent-site history follows the site it
        is inside. ``atom.in_site`` remains the site the atom is inside.

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
        self.site_collection.reset_site_occupations()
        for atom in self.atoms:
            if atom.committed_site is not None:
                self.site_by_index(atom.committed_site).contains_atoms.append(atom.index)

    def reset(self) -> None:
        """Reset the trajectory.
        
        This clears all trajectory information for atoms and sites and
        resets the timesteps list. The shared faces of polyhedral sites are
        kept.
        """
        for atom in self.atoms:
            atom.reset()
        self.site_collection.reset()
        self.timesteps = []

    @property
    def atoms_trajectory(self) -> list[list[int | None]]:
        """Return the trajectory of all atoms.
        
        Returns:
            A list of lists, where each inner list represents a timestep and
            contains the site index of each atom at that timestep: its
            committed site with commitment, otherwise the site it is inside.
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
    def geometric_trajectory(self) -> list[list[int | None]]:
        """Return the site each atom was inside at each timestep.

        With commitment this can differ from ``atoms_trajectory``, which
        holds each atom's committed site; without commitment the two are
        the same.

        Returns:
            A list of lists, where each inner list represents a timestep and
            contains the index of the site each atom was inside (or None),
            in the order of ``atoms``.
        """
        if self._commitment_radii is None:
            return self.atoms_trajectory
        return list(map(list, zip(*[atom.geometric_trajectory for atom in self.atoms])))

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

    def transition_events(self) -> TransitionEvents:
        """Return every transition of an atom from one site to another.

        A transition is a change of site between consecutive frames in
        which the atom has a site; frames in which it has none are skipped.
        With commitment, the trajectory holds each atom's committed site, so
        a transition is a change of committed site. These are the
        transitions counted in ``site.transitions``: the number of events
        from site i to site j is ``self.site_by_index(i).transitions[j]``.
        The events are derived from the stored trajectory each time this is
        called. Each event's entry frame is the earliest frame from which
        the atom stayed inside its new site, without leaving, up to the
        event's frame; with commitment it is derived from
        ``geometric_trajectory``, and without commitment it equals the
        event's frame.

        Returns:
            The transitions, sorted by frame and then by atom index. Each
            event's frame is the frame, from 0, in which the atom is first
            counted in its new site. Timesteps and entry timesteps are
            the labels in ``timesteps`` if every frame has one, and None
            otherwise.
        """
        # Atoms in index order, so that events within a frame come out sorted.
        atoms = sorted(self.atoms, key=lambda atom: atom.index)
        n_frames = len(atoms[0].trajectory)
        # Site of each atom in each frame, NaN where it has none: (frames, atoms).
        sites = np.array([atom.trajectory for atom in atoms], dtype=float).T
        # Carry each atom's last site forward over frames in which it has none.
        assigned = ~np.isnan(sites)
        last_assigned = np.where(assigned, np.arange(n_frames)[:, np.newaxis], 0)
        np.maximum.accumulate(last_assigned, axis=0, out=last_assigned)
        carried = np.take_along_axis(sites, last_assigned, axis=0)
        # NaN compares unequal to everything, even NaN, so the two isnan
        # tests are needed to exclude frames with no site and atoms with no
        # earlier site.
        changed = (assigned[1:] & ~np.isnan(carried[:-1])
                   & (sites[1:] != carried[:-1]))
        previous_frame, column = np.nonzero(changed)
        atom_index = np.array([atom.index for atom in atoms], dtype=np.int64)[column]
        frame = previous_frame + 1
        if self._commitment_radii is None:
            entry_frame = frame
        else:
            # Site each atom is inside in each frame. A frame starts a new
            # run where the site differs from the previous frame's; NaN
            # compares unequal to everything, so frames with no site break
            # runs. The entry frame is the start of the run at the event.
            inside = np.array([atom.geometric_trajectory for atom in atoms], dtype=float).T
            new_run = np.ones(inside.shape, dtype=bool)
            new_run[1:] = inside[1:] != inside[:-1]
            run_start = np.where(new_run, np.arange(n_frames)[:, np.newaxis], 0)
            np.maximum.accumulate(run_start, axis=0, out=run_start)
            entry_frame = run_start[frame, column]
        labels = (np.asarray(self.timesteps, dtype=np.int64)
                  if len(self.timesteps) == n_frames else None)
        return TransitionEvents(
            frame=frame,
            timestep=None if labels is None else labels[frame],
            entry_frame=entry_frame,
            entry_timestep=None if labels is None else labels[entry_frame],
            atom=atom_index,
            from_site=carried[previous_frame, column].astype(np.int64),
            to_site=sites[frame, column].astype(np.int64))

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

        Raises:
            ValueError: If the sites are polyhedral and a site has a vertex
                index that is not an atom of a structure, or no shared
                faces have been found and finding them from the first
                structure fails (see ``find_shared_faces``).
            RuntimeError: If the sites are polyhedral and finding the shared
                faces fails on degenerate vertex geometry, or a site tested
                for an atom is too distorted for its triangles to enclose
                its centre.
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
