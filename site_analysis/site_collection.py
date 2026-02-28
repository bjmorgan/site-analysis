"""Base classes for collections of sites in crystal structures.

This module defines:

- ``SiteCollection``: abstract base class that all site collection types
  must inherit from. Provides the interface for site-atom assignment and
  common functionality for managing site occupations.
- ``PriorityAssignmentMixin``: mixin providing priority-based site
  assignment ordering. Used by collection types that check sites one at
  a time (polyhedral, spherical) but not by those that use global
  distance-matrix assignment (Voronoi, dynamic Voronoi).
- ``_NearestSiteLookup``: precomputed lookup for finding the nearest
  site to a given position.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import NamedTuple, Sequence, TYPE_CHECKING

import numpy as np
from pymatgen.core import Structure # type: ignore
from .atom import Atom
from .site import Site


class _NearestSiteLookup(NamedTuple):
    """Precomputed lookup for finding the nearest site to a given position.

    Uses minimum-image convention in fractional space, which is only
    geometrically exact for orthogonal cells.
    """
    centres: np.ndarray
    site_indices: list[int]

    def nearest_site_index(self, frac_coords: np.ndarray) -> int:
        """Return the site index nearest to the given fractional coordinates.

        Uses minimum-image convention in fractional space.

        Args:
            frac_coords: Fractional coordinates to find the nearest site for.

        Returns:
            The site index of the nearest site.
        """
        diffs = self.centres - frac_coords
        diffs -= np.round(diffs)
        dists = np.linalg.norm(diffs, axis=1)
        return self.site_indices[int(np.argmin(dists))]

class PriorityAssignmentMixin:
    """Mixin providing priority-based site assignment ordering.

    Provides ``_get_priority_sites(atom)``, a generator that yields sites
    in an optimised order based on recent site history, learned transitions,
    and precomputed distance ranking.

    Subclasses call ``_init_priority_ranking(centres, site_indices)`` from
    their ``__init__`` to enable distance-ranked ordering. If not called,
    the generator falls back to neighbours then arbitrary order.

    Note: distance ranking uses minimum-image convention in fractional
    space, which is only geometrically exact for orthogonal cells. For
    non-orthogonal cells the ranking is approximate, but correctness is
    unaffected since all sites are eventually checked.

    Expects to be mixed with ``SiteCollection`` which provides
    ``site_by_index``, ``neighbouring_sites``, and ``sites``.
    """

    # Type stubs for the SiteCollection interface this mixin requires.
    # These are provided by SiteCollection at runtime via MRO.
    if TYPE_CHECKING:
        sites: Sequence[Site]
        def site_by_index(self, index: int) -> Site: ...
        def neighbouring_sites(self, site_index: int) -> Sequence[Site]: ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distance_ranked_sites: dict[int, list[int]] | None = None
        self._nearest_site_lookup: _NearestSiteLookup | None = None

    def _init_priority_ranking(self, centres: np.ndarray, site_indices: list[int]) -> None:
        """Precompute distance-ranked site ordering from the given centres.

        Args:
            centres: (N, 3) array of fractional coordinates for each site.
            site_indices: Corresponding site indices.
        """
        ranked: dict[int, list[int]] = {}
        for i, idx in enumerate(site_indices):
            diffs = centres - centres[i]
            diffs -= np.round(diffs)
            dists = np.linalg.norm(diffs, axis=1)
            order = np.argsort(dists)
            ranked[idx] = [site_indices[j] for j in order if j != i]
        self._distance_ranked_sites = ranked
        self._nearest_site_lookup = _NearestSiteLookup(
            centres=centres, site_indices=site_indices
        )

    def _get_priority_sites(self, atom: Atom) -> Generator[Site, None, None]:
        """Generator that yields sites in priority order for optimised atom assignment.

        The checking sequence depends on available information:

        When trajectory history exists:
            1. Most recently visited site, then previously visited site
            2. Learned transition destinations in frequency order
            3. Remaining sites by distance from anchor (if distance ranking
               available), otherwise neighbours then arbitrary order

        When no trajectory history exists:
            - If distance ranking is available: nearest site centre first,
              then learned transitions, then distance-ranked outward
            - Otherwise: all sites in arbitrary order

        Each site is yielded at most once.

        Args:
            atom: Atom object with recent site history used to determine
                site priorities.

        Yields:
            Site: Sites in optimal checking order.
        """
        checked_indices: set[int] = set()
        anchor_index = None

        recent = [s for s in atom._recent_sites if s is not None]
        if recent:
            anchor_index = recent[0]
            for index in recent:
                yield self.site_by_index(index)
                checked_indices.add(index)
        elif self._nearest_site_lookup is not None:
            anchor_index = self._nearest_site_lookup.nearest_site_index(atom.frac_coords)
            yield self.site_by_index(anchor_index)
            checked_indices.add(anchor_index)

        if anchor_index is not None:
            # Learned transitions in frequency order
            anchor_site = self.site_by_index(anchor_index)
            for dest_index in anchor_site.most_frequent_transitions():
                if dest_index not in checked_indices:
                    yield self.site_by_index(dest_index)
                    checked_indices.add(dest_index)

            # Remaining sites
            if self._distance_ranked_sites is not None:
                for index in self._distance_ranked_sites[anchor_index]:
                    if index not in checked_indices:
                        yield self.site_by_index(index)
                        checked_indices.add(index)
            else:
                for neighbour_site in self.neighbouring_sites(anchor_index):
                    if neighbour_site.index not in checked_indices:
                        yield neighbour_site
                        checked_indices.add(neighbour_site.index)
                for site in self.sites:
                    if site.index not in checked_indices:
                        yield site
        else:
            for site in self.sites:
                yield site


class SiteCollection(ABC):
    """Parent class for collections of sites.

    Collections of specific site types should inherit from this class.

    Attributes:
        sites (list): List of ``Site``-like objects.

    """

    def __init__(self, sites: Sequence[Site]) -> None:
        """Create a SiteCollection object.
        
        Args:
            sites (list): List of ``Site`` objects.
            
        Raises:
            ValueError: If there are duplicate site indices.
        
        """
        self.sites = sites
        
        # Create lookup dictionary for efficient site access by index
        self._site_lookup: dict[int, Site] = {}
        for site in sites:
            if site.index in self._site_lookup:
                raise ValueError(f"Duplicate site index detected: {site.index}. Site indices must be unique.")
            self._site_lookup[site.index] = site

    @abstractmethod
    def assign_site_occupations(self, atoms, structure):
        """Assigns atoms to sites for a specific structure.

        This method should be implemented in the derived subclass

        Args:
            atoms (list(Atom)): List of Atom objects to be assigned to sites.
            struture (pymatgen.Structure): Pymatgen Structure object used to specificy
                the atomic coordinates.

        Returns:
            None

        Notes:
            The atom coordinates should already be consistent with the coordinates
            in `structure`. Recommended usage is via the ``analyse_structure()`` method.

        """
        raise NotImplementedError('assign_site_occupations should be implemented in'
            ' the derived class')

    @abstractmethod
    def analyse_structure(self, atoms, structure):
        """Perform a site analysis for a set of atoms on a specific structure.

        This method should be implemented in the derived subclass.

        Args:
            atoms (list(Atom)): List of Atom objects to be assigned to sites.
            struture (pymatgen.Structure): Pymatgen Structure object used to specificy
                the atomic coordinates.

        Returns:
            None

        """
        raise NotImplementedError('analyse_structure should be implemented in the derived class')

    def neighbouring_sites(self, site_index):
        """If implemented, returns a list of sites that neighbour
        a given site.

        This method should be implemented in the derived subclass.
        
        Args:
            site_index (int): Index of the site to return a list of neighbours for.

        """
        raise NotImplementedError('neighbouring_sites should be implemented'
            'in the derived class')

    def site_by_index(self, index):
        """Returns the site with a specific index.
        
        Args:
            index (int): index for the site to be returned.
        
        Returns:
            (Site)
        
        Raises:
            ValueError: If a site with the specified index is not contained
                in this SiteCollection.
        
        """
        site = self._site_lookup.get(index)
        if site is None:
            raise ValueError(f'No site with index {index} found')
        return site

    def update_occupation(self, site, atom):
        """Updates site and atom attributes for this atom occupying this site.

        Args:
            site (Site): The site to be updated.
            atom (Atom): The atom to be updated.

        Returns:
            None

        Notes:

            This method does the following:

            1. If the atom has moved to a new site, record a old_site --> new_site transition.
            2. Add this atom's index to the list of atoms occupying this site.
            3. Add this atom's fractional coordinates to the list of
               coordinates observed occupying this site.
            4. Assign this atom this site index.

        """
        previous_site_index = None
        if atom.trajectory:
            previous_site_index = atom.trajectory[-1]
        if previous_site_index is not None:
            if previous_site_index != site.index: # this atom has moved
                previous_site = self.site_by_index(previous_site_index)
                previous_site.transitions[site.index] += 1
        site.contains_atoms.append(atom.index)
        site.points.append(atom.frac_coords)
        atom.in_site = site.index
        atom.update_recent_site(site.index)

    def reset(self) -> None:
        """Reset the collection and all its sites for a fresh analysis run.

        Resets per-site state (occupations, trajectories, caches) via
        ``Site.reset()``. Subclasses may override to also clear
        collection-level caches, but should call ``super().reset()``.
        """
        for site in self.sites:
            site.reset()

    def reset_site_occupations(self):
        """Occupations of all sites in this site collection are set as empty.

        Args:
            None

        Returns:
            None

        """
        for s in self.sites:
            s.contains_atoms = []

    def sites_contain_points(self,
                             points: np.ndarray,
                             structure: Structure | None=None) -> bool:
        """If implemented, Checks whether the set of sites contain
        a corresponding set of fractional coordinates.
        Args:
            points (np.array): 3xN numpy array of fractional coordinates.
                There should be one coordinate for each site being checked.

        Returns:
            (bool)
            
        Notes:
            Specific SiteCollection subclass implementations may require
            additional arguments to be passed.
        """
        raise NotImplementedError('sites_contain_points() should be'
            ' implemented in the derived class')
            
    def summaries(self, metrics: list[str] | None = None) -> list[dict]:
        """Generate summary statistics for all sites in the collection.
        
        Args:
            metrics: List of metrics to include for each site. None returns 
                default metrics for each site.
                
        Returns:
            List of summary dicts, one per site, in site order.
        """
        return [site.summary(metrics=metrics) for site in self.sites]
