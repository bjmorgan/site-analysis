"""Collection manager for dynamic Voronoi sites in crystal structures.

This module provides the DynamicVoronoiSiteCollection class, which manages a
collection of DynamicVoronoiSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The DynamicVoronoiSiteCollection extends the base SiteCollection class with
specific functionality for dynamic Voronoi sites, including:
1. Calculating the dynamic centres of sites based on reference atom positions
2. Assigning atoms to sites using Voronoi tessellation principles

For atom assignment, the collection:
1. First updates each site's centre by calculating the mean position of its
   reference atoms, with special handling for periodic boundary conditions
2. Calculates distances from each (dynamically determined) site centre to each atom
3. Assigns each atom to the site with the nearest centre
4. Uses the structure's lattice to correctly handle distances across
   periodic boundaries

This collection is particularly useful for tracking sites in frameworks
that deform during simulation, as the site centres adapt to the changing
positions of the reference atoms.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pymatgen.core import Lattice, Structure
from site_analysis.site_collection import SiteCollection
from site_analysis.site import Site
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.pbc_utils import correct_pbc
from site_analysis.atom import Atom


@dataclass
class _CentreGroup:
    """Batch arrays for a group of sites sharing the same n_reference.

    Owns the cached PBC shift state and the vectorised fast-path
    computation.  The collection orchestrates fallback (per-site)
    computation and distributes computed centres back to individual
    sites.

    Attributes:
        site_positions: Indices into the parent
            ``DynamicVoronoiSiteCollection.sites`` list for this group.
        ref_indices: ``(n_sites, n_ref)`` int array of reference atom
            indices.
        pbc_shifts: ``(n_sites, n_ref, 3)`` int, cached image shifts.
        cached_raw_frac: ``(n_sites, n_ref, 3)`` float, previous raw
            coords.
        initialised: Whether the PBC caches have been populated.
    """
    site_positions: list[int]
    ref_indices: np.ndarray
    pbc_shifts: np.ndarray = field(init=False, repr=False)
    cached_raw_frac: np.ndarray = field(init=False, repr=False)
    initialised: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        n_sites, n_ref = self.ref_indices.shape
        self.pbc_shifts = np.zeros((n_sites, n_ref, 3), dtype=np.int64)
        self.cached_raw_frac = np.zeros((n_sites, n_ref, 3))

    def try_fast_update(self, batch_ref: np.ndarray) -> np.ndarray | None:
        """Try the vectorised incremental shift update.

        If the group is initialised and all coordinate displacements
        since the last frame are below 0.3 fractional units, updates
        the cached shifts and returns the computed centres.  Otherwise
        returns ``None`` to signal that the caller should fall back to
        per-site full PBC computation.

        Args:
            batch_ref: Raw fractional coordinates for this group,
                shape ``(n_sites, n_ref, 3)``.

        Returns:
            Site centres as ``(n_sites, 3)`` array, or ``None`` if the
            fast path cannot be used.
        """
        if not self.initialised:
            return None
        diff = batch_ref - self.cached_raw_frac
        wrapping = np.round(diff).astype(np.int64)
        physical_diff = diff - wrapping
        if not np.all(np.abs(physical_diff) < 0.3):
            return None
        new_shifts = self.pbc_shifts - wrapping
        corrected = batch_ref + new_shifts
        # Shift each site's coords so all values are >= 0
        min_coords = np.min(corrected, axis=1)  # (n_sites, 3)
        non_neg = np.maximum(0, np.ceil(-min_coords))  # (n_sites, 3)
        corrected = corrected + non_neg[:, np.newaxis, :]
        self.pbc_shifts = new_shifts
        self.cached_raw_frac = batch_ref.copy()
        centres: np.ndarray = np.mean(corrected, axis=1) % 1.0
        return centres

    def initialise(self, batch_ref: np.ndarray) -> None:
        """Mark the group as initialised after fallback computation.

        Called once all ``pbc_shifts`` entries have been populated by
        per-site full PBC computation.

        Args:
            batch_ref: Raw fractional coordinates for this group,
                shape ``(n_sites, n_ref, 3)``.
        """
        self.cached_raw_frac = batch_ref.copy()
        self.initialised = True


class DynamicVoronoiSiteCollection(SiteCollection):
    """A collection of DynamicVoronoiSite objects.
    
    This collection manages a set of dynamic Voronoi sites and handles
    the assignment of atoms to sites based on their dynamically calculated centres.
    
    Attributes:
        sites (list[DynamicVoronoiSite]): list of DynamicVoronoiSite objects.
    """
    
    def __init__(self,
                 sites: list[Site]) -> None:
        """Create a DynamicVoronoiSiteCollection instance.
        
        Args:
            sites (list[DynamicVoronoiSite]): list of DynamicVoronoiSite objects.
            
        Returns:
            None
            
        Raises:
            TypeError: If any of the sites is not a DynamicVoronoiSite.
        """
        for s in sites:
            if not isinstance(s, DynamicVoronoiSite):
                raise TypeError("All sites must be DynamicVoronoiSite instances")
        super(DynamicVoronoiSiteCollection, self).__init__(sites)
        self.sites: list[DynamicVoronoiSite]
        self._centre_groups: list[_CentreGroup] = self._build_centre_groups()
        
    def _build_centre_groups(self) -> list[_CentreGroup]:
        """Group sites by reference count for batch centre calculation."""
        by_nref: dict[int, list[int]] = defaultdict(list)
        for i, site in enumerate(self.sites):
            by_nref[len(site.reference_indices)].append(i)
        groups: list[_CentreGroup] = []
        for positions in by_nref.values():
            ref_indices = np.array(
                [self.sites[i].reference_indices for i in positions])
            groups.append(_CentreGroup(
                site_positions=positions,
                ref_indices=ref_indices,
            ))
        return groups

    def _batch_calculate_centres(self,
                                  all_frac_coords: np.ndarray,
                                  lattice: Lattice) -> None:
        """Compute all site centres in batch, grouped by reference count.

        For each group, tries the vectorised fast path first.  If that
        fails (first frame, after reset, or large displacement), falls
        back to per-site full PBC computation.

        Args:
            all_frac_coords: Full fractional coordinate array from the
                structure, shape ``(n_atoms, 3)``.
            lattice: Lattice for PBC distance calculations.
        """
        for group in self._centre_groups:
            batch_ref = all_frac_coords[group.ref_indices]  # (n_sites, n_ref, 3)
            centres = group.try_fast_update(batch_ref)
            if centres is not None:
                for idx, pos in enumerate(group.site_positions):
                    self.sites[pos]._centre_coords = centres[idx]
                continue
            # Fallback — per-site full PBC computation.
            for idx, pos in enumerate(group.site_positions):
                site = self.sites[pos]
                corrected, image_shifts = correct_pbc(
                    batch_ref[idx], site.reference_center, lattice)
                site._centre_coords = np.mean(corrected, axis=0) % 1.0
                group.pbc_shifts[idx] = image_shifts
            group.initialise(batch_ref)

    def reset(self) -> None:
        """Reset all sites and batch PBC caches for a fresh analysis run."""
        super().reset()
        for group in self._centre_groups:
            group.initialised = False

    def analyse_structure(self,
                          atoms: list[Atom],
                          structure: Structure) -> None:
        """Analyse a structure to assign atoms to sites.
        
        This method:
        1. Assigns coordinates to atoms
        2. Calculates the centres of all dynamic Voronoi sites
        3. Assigns atoms to sites based on these centres
        
        Args:
            atoms (list[Atom]): list of atoms to be assigned to sites.
            structure (Structure): Pymatgen Structure containing atom positions.
            
        Returns:
            None
        """
        for atom in atoms:
            atom.assign_coords(structure)
        all_frac_coords = structure.frac_coords
        lattice = structure.lattice
        self._batch_calculate_centres(all_frac_coords, lattice)
        self.assign_site_occupations(atoms, structure)
        
    def assign_site_occupations(self,
                                atoms: list[Atom],
                                structure: Structure) -> None:
        """Assign atoms to sites based on Voronoi tessellation.
        
        This method assigns each atom to the nearest site centre,
        taking into account periodic boundary conditions.
        
        Args:
            atoms (list[Atom]): list of atoms to be assigned to sites.
            structure (Structure): Pymatgen Structure containing atom positions.
            
        Returns:
            None
        """
        self.reset_site_occupations()
        if not atoms:
            return
        lattice = structure.lattice
        site_coords = np.array([site.centre for site in self.sites])
        atom_coords = np.array([atom.frac_coords for atom in atoms])
        dist_matrix = lattice.get_all_distances(site_coords, atom_coords)
        site_list_indices = np.argmin(dist_matrix, axis=0)
        for atom, site_list_index in zip( atoms, site_list_indices):
            site = self.sites[site_list_index]
            self.update_occupation(site, atom)