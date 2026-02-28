"""Collection manager for dynamic Voronoi sites in crystal structures.

This module provides the DynamicVoronoiSiteCollection class, which manages a
collection of DynamicVoronoiSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The DynamicVoronoiSiteCollection extends the base SiteCollection class with
specific functionality for dynamic Voronoi sites, including:
1. Calculating the dynamic centers of sites based on reference atom positions
2. Assigning atoms to sites using Voronoi tessellation principles

For atom assignment, the collection:
1. First updates each site's center by calculating the mean position of its
   reference atoms, with special handling for periodic boundary conditions
2. Calculates distances from each (dynamically determined) site center to each atom
3. Assigns each atom to the site with the nearest center
4. Uses the structure's lattice to correctly handle distances across
   periodic boundaries

This collection is particularly useful for tracking sites in frameworks
that deform during simulation, as the site centers adapt to the changing
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
from site_analysis.atom import Atom


@dataclass
class _CentreGroup:
    """Batch arrays for a group of sites sharing the same n_reference.

    Attributes:
        site_positions: Indices into ``self.sites`` for this group.
        ref_indices: ``(n_sites, n_ref)`` int array of reference atom indices.
        pbc_shifts: ``(n_sites, n_ref, 3)`` int, cached image shifts.
        cached_raw_frac: ``(n_sites, n_ref, 3)`` float, previous raw coords.
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

    def reset(self) -> None:
        """Clear cached PBC state so the next frame does a full computation."""
        self.initialised = False


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

        On the first frame (or after reset), falls back to per-site full
        PBC computation to populate the caches.  On subsequent frames,
        uses vectorised incremental shift updates across all sites in
        each group simultaneously.

        Args:
            all_frac_coords: Full fractional coordinate array from the
                structure, shape ``(n_atoms, 3)``.
            lattice: Lattice for PBC distance calculations.
        """
        for group in self._centre_groups:
            # (n_sites, n_ref, 3)
            batch_ref = all_frac_coords[group.ref_indices]
            if group.initialised:
                diff = batch_ref - group.cached_raw_frac
                wrapping = np.round(diff).astype(np.int64)
                physical_diff = diff - wrapping
                if np.all(np.abs(physical_diff) < 0.3):
                    new_shifts = group.pbc_shifts - wrapping
                    corrected = batch_ref + new_shifts
                    # Per-site uniform non-negative shift
                    min_coords = np.min(corrected, axis=1)  # (n_sites, 3)
                    uniform = np.maximum(0, np.ceil(-min_coords))  # (n_sites, 3)
                    corrected = corrected + uniform[:, np.newaxis, :]
                    group.pbc_shifts = new_shifts
                    group.cached_raw_frac = batch_ref.copy()
                    centres = np.mean(corrected, axis=1) % 1.0
                    for idx, pos in enumerate(group.site_positions):
                        self.sites[pos]._centre_coords = centres[idx]
                    continue
            # First frame or cache invalidation — per-site full computation
            for idx, pos in enumerate(group.site_positions):
                site = self.sites[pos]
                ref_coords = batch_ref[idx]
                site._compute_corrected_coords(ref_coords, lattice)
                group.pbc_shifts[idx] = site._pbc_image_shifts
                group.cached_raw_frac[idx] = site._pbc_cached_raw_frac
            group.initialised = True

    def reset_centre_groups(self) -> None:
        """Reset batch and per-site PBC caches so the next frame does full computation."""
        for group in self._centre_groups:
            group.reset()
            for pos in group.site_positions:
                self.sites[pos]._pbc_image_shifts = None
                self.sites[pos]._pbc_cached_raw_frac = None

    def analyse_structure(self,
                          atoms: list[Atom],
                          structure: Structure) -> None:
        """Analyze a structure to assign atoms to sites.
        
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
        
        This method assigns each atom to the nearest site center,
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