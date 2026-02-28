"""Collection manager for spherical sites in crystal structures.

This module provides the SphericalSiteCollection class, which manages a
collection of SphericalSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The SphericalSiteCollection extends the base SiteCollection class with
specific functionality for spherical sites. Optimised atom assignment is
provided by the PriorityAssignmentMixin, which leverages recent site
history, learned transition patterns, and precomputed distance-ranked site
ordering.

This handles overlapping spherical sites in a consistent way -- if an atom
is in a region where multiple sites overlap, it will remain assigned to its
original site as long as it stays within that site's volume. This
persistence can be useful for tracking atoms through small oscillations
without generating spurious site transitions.
"""

import numpy as np
from pymatgen.core import Structure
from site_analysis.atom import Atom
from site_analysis.spherical_site import SphericalSite
from site_analysis.site_collection import SiteCollection, PriorityAssignmentMixin

class SphericalSiteCollection(PriorityAssignmentMixin, SiteCollection):


    def __init__(self,
        sites: list[SphericalSite]) -> None:
        """A collection of SphericalSite objects with optimised atom assignment.

        Extends the base SiteCollection class with specific functionality for
        spherical sites, using precomputed distance-ranked site ordering for
        optimised atom assignment via the PriorityAssignmentMixin.

        Args:
            sites (list): List of ``SphericalSite`` objects.

        Attributes:
            sites (list): List of ``SphericalSite`` objects.

        """
        for s in sites:
            if not isinstance(s, SphericalSite):
                raise TypeError(f"Expected SphericalSite, got {type(s).__name__}")
        super().__init__(sites)
        self.sites: list[SphericalSite]
        centres = np.array([s.frac_coords for s in self.sites])
        site_indices = [s.index for s in self.sites]
        self._init_priority_ranking(centres, site_indices)

    def analyse_structure(self,
            atoms: list[Atom],
            structure: Structure) -> None:
        """Analyse a structure to assign atoms to spherical sites.

        This method:
        1. Assigns fractional coordinates to each atom based on the structure
        2. Delegates to assign_site_occupations to determine which atoms
           belong in which sites

        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions

        Returns:
            None
        """
        for a in atoms:
            a.assign_coords(structure)
        self.assign_site_occupations(atoms, structure)

    def assign_site_occupations(self,
        atoms: list[Atom],
        structure: Structure) -> None:
        """Assign atoms to spherical sites based on their positions.

        Uses the priority-based site checking approach from
        PriorityAssignmentMixin to check sites in an optimised order.

        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
        """
        self.reset_site_occupations()
        for atom in atoms:
            atom.in_site = None

            # Check sites in priority order until found
            for site in self._get_priority_sites(atom):
                if site.contains_atom(atom, lattice=structure.lattice):
                    self.update_occupation(site, atom)
                    break
