"""Base classes for collections of sites in crystal structures.

This module defines:

- ``SiteCollection``: abstract base class that all site collection types
  must inherit from. Provides the interface for site-atom assignment and
  common functionality for managing site occupations.
- ``PriorityAssignmentMixin``: mixin providing priority-based site
  assignment ordering. Used by collection types that check sites one at
  a time (polyhedral, spherical) but not by those that assign each atom
  to its nearest site centre (Voronoi, dynamic Voronoi).
- ``_SiteCentreIndex``: site centres, ranked by distance from a point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from typing import Generic, TypeVar, TYPE_CHECKING

import numpy as np
from .atom import Atom
from .site import Site
from .neighbour_search import PeriodicNeighbourIndex

# Sites within this relative margin beyond the reach are also ranked, so
# that rounding cannot leave out a site whose containment test would
# accept the point.
_REACH_TOLERANCE = 1e-9


class _SiteCentreIndex:
    """Site centres, ranked by Cartesian minimum-image distance from a point.

    Holds a ``PeriodicNeighbourIndex`` over the centres, built on first
    use and rebuilt whenever the lattice changes. With a reach, only the
    sites whose centres are within the reach of the point are ranked.
    """

    def __init__(self,
            centres: np.ndarray,
            site_indices: Sequence[int],
            reach: float | None = None) -> None:
        """Create a _SiteCentreIndex.

        Args:
            centres: Fractional coordinates of the site centres, shape
                (N, 3), with N at least 1.
            site_indices: The site index of each centre.
            reach: If given, the largest distance from its centre at which
                any site can contain a point. Only the sites whose centres
                are within the reach of the point are then ranked.

        Raises:
            ValueError: If ``centres`` is not shaped (N, 3) with N at least
                1, or is not finite, or ``site_indices`` does not have one
                entry per centre, or ``reach`` is negative or NaN.
        """
        self._centres = np.array(centres, dtype=np.float64)
        if self._centres.ndim != 2 or self._centres.shape[1] != 3 or len(self._centres) == 0:
            raise ValueError(
                f"centres must have shape (N, 3) with N at least 1, got {self._centres.shape}")
        if not np.isfinite(self._centres).all():
            raise ValueError("centres must be finite")
        self._site_indices = np.array(site_indices)
        if self._site_indices.shape != (len(self._centres),):
            raise ValueError(
                f"need one site index per centre, got shape {self._site_indices.shape} "
                f"for {len(self._centres)} centres")
        if reach is not None and not reach >= 0:
            raise ValueError(f"reach must be non-negative, got {reach}")
        self._reach = reach
        # The lattice the index was built for, set when it is built.
        self._lattice_matrix = np.zeros((3, 3))
        self._index: PeriodicNeighbourIndex | None = None
        self._nearby_radius = 0.0

    def _index_for(self, lattice_matrix: np.ndarray) -> PeriodicNeighbourIndex:
        """Return the neighbour index for a lattice, building it if needed.

        Args:
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Returns:
            The neighbour index over the centres in this lattice.
        """
        if self._index is None or not np.array_equal(lattice_matrix, self._lattice_matrix):
            self._index = PeriodicNeighbourIndex(self._centres, lattice_matrix)
            self._lattice_matrix = np.array(lattice_matrix, dtype=np.float64)
            # The edge of a cube holding one site's share of the cell volume.
            # A sphere of this radius holds about four sites on average, so
            # the first list is short but rarely empty.
            volume = abs(np.linalg.det(self._lattice_matrix))
            self._nearby_radius = (volume / len(self._centres)) ** (1 / 3)
        return self._index

    def ranked_site_indices(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> Iterator[list[int]]:
        """Yield lists of site indices in order of distance from a point.

        Sites are ordered by the Cartesian minimum-image distance of their
        centres from the point, with equal distances in the order of the
        centres. Without a reach, the lists joined together hold every
        site once: the first list holds the sites near the point, and the
        rest are ranked only if the caller asks for another list. With a
        reach, a single list holds the sites whose centres are within the
        reach of the point.

        Args:
            frac_coords: Fractional coordinates of the point, shape (3,).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Yields:
            Lists of site indices.

        Raises:
            ValueError: If the lattice matrix is singular or non-finite, or
                ``frac_coords`` is non-finite. As this is a generator, the
                error is raised when the first list is requested.
        """
        neighbour_index = self._index_for(lattice_matrix)
        query = np.reshape(frac_coords, (1, 3))
        if self._reach is not None:
            _, within, _ = neighbour_index.query_within(
                query, self._reach * (1.0 + _REACH_TOLERANCE))
            yield self._site_indices[within].tolist()
            return
        _, nearby, _ = neighbour_index.query_within(query, self._nearby_radius)
        yield self._site_indices[nearby].tolist()
        if len(nearby) < len(self._site_indices):
            _, ranked, _ = neighbour_index.query_within(query, np.inf)
            # Leave out the nearby sites by position, so that each site is
            # yielded once even if the two queries round a distance near the
            # nearby radius differently.
            remaining = np.ones(len(self._site_indices), dtype=bool)
            remaining[nearby] = False
            yield self._site_indices[ranked[remaining[ranked]]].tolist()


SiteT = TypeVar('SiteT', bound=Site)


class PriorityAssignmentMixin(Generic[SiteT]):
    """Mixin providing priority-based site assignment ordering.

    Provides ``_get_priority_sites(atom, lattice_matrix)``, a generator
    that yields sites in an optimised order based on recent site history,
    learned transitions, and distance from the atom.

    Subclasses call ``_init_priority_ranking(centres, site_indices)`` from
    their ``__init__`` to enable distance-ranked ordering, passing a
    ``reach`` if no site can contain a point beyond some distance from its
    centre (as for spherical sites). If not called, the generator falls
    back to ``neighbouring_sites`` then list order (used by
    ``PolyhedralSiteCollection`` when reference centres are unavailable).

    Expects to be mixed with ``SiteCollection`` which provides
    ``site_by_index``, ``neighbouring_sites``, and ``sites``.
    """

    # Type stubs for the SiteCollection interface this mixin requires.
    # These are provided by SiteCollection at runtime via MRO.
    if TYPE_CHECKING:
        sites: Sequence[SiteT]
        def site_by_index(self, index: int) -> SiteT: ...
        def neighbouring_sites(self, site_index: int) -> Sequence[SiteT]: ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._site_centres: _SiteCentreIndex | None = None

    def _init_priority_ranking(self,
            centres: np.ndarray,
            site_indices: list[int],
            reach: float | None = None) -> None:
        """Set up distance-ranked site ordering from the given centres.

        Does nothing if ``centres`` is empty (zero sites).

        Args:
            centres: (N, 3) array of fractional coordinates for each site.
            site_indices: Corresponding site indices.
            reach: If given, no site contains a point further than this
                from its centre, so sites further than this from an atom
                are left out of the distance ranking.
        """
        if len(centres) == 0:
            return
        self._site_centres = _SiteCentreIndex(centres, site_indices, reach)

    def _get_priority_sites(self,
            atom: Atom,
            lattice_matrix: np.ndarray) -> Generator[SiteT, None, None]:
        """Generator that yields sites in priority order for optimised atom assignment.

        The checking sequence is:

            1. Most recently visited site, then previously visited site
            2. Learned transition destinations from the most recent site
               in frequency order
            3. Remaining sites by the distance of their centres from the
               atom (if site centres are available; only those within
               reach, if a reach was given), otherwise neighbours of the
               most recent site then list order

        An atom with no recent site starts at step 3. Without site centres
        it gets all sites in list order.

        Each site is yielded at most once.

        Args:
            atom: Atom object with recent site history used to determine
                site priorities.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Yields:
            Site: Sites in search order.
        """
        checked_indices: set[int] = set()
        most_recent_index: int | None = None

        recent = [s for s in atom._recent_sites if s is not None]
        if recent:
            most_recent_index = recent[0]
            for index in recent:
                yield self.site_by_index(index)
                checked_indices.add(index)

            # Learned transitions in frequency order
            most_recent_site = self.site_by_index(most_recent_index)
            for dest_index in most_recent_site.most_frequent_transitions():
                if dest_index not in checked_indices:
                    yield self.site_by_index(dest_index)
                    checked_indices.add(dest_index)

        # Remaining sites
        if self._site_centres is not None:
            for ranked in self._site_centres.ranked_site_indices(atom.frac_coords, lattice_matrix):
                for index in ranked:
                    if index not in checked_indices:
                        yield self.site_by_index(index)
        elif most_recent_index is not None:
            for neighbour_site in self.neighbouring_sites(most_recent_index):
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
        # A copy, so the collection does not share the caller's list. Typed
        # as a Sequence so that subclasses can narrow the type of site.
        self.sites: Sequence[Site] = list(sites)
        
        # Create lookup dictionary for efficient site access by index
        self._site_lookup: dict[int, Site] = {}
        for site in self.sites:
            if site.index in self._site_lookup:
                raise ValueError(f"Duplicate site index detected: {site.index}. Site indices must be unique.")
            self._site_lookup[site.index] = site

    @abstractmethod
    def assign_site_occupations(self, atoms, lattice_matrix):
        """Assign atoms to sites.

        Args:
            atoms: List of Atom objects to be assigned to sites.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Note:
            The atom coordinates should already be consistent with the
            structure. Recommended usage is via ``analyse_structure()``.
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

            1. Add this atom's index to the list of atoms occupying this site.
            2. Add this atom's fractional coordinates to the list of
               coordinates observed occupying this site.
            3. Assign this atom this site index.

        """
        site.contains_atoms.append(atom.index)
        site.points.append(atom.frac_coords)
        atom.in_site = site.index

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
                             all_frac_coords: np.ndarray,
                             lattice_matrix: np.ndarray) -> bool:
        """Check whether the set of sites contain corresponding points.

        Args:
            points: (N, 3) array of fractional coordinates.
                One coordinate per site being checked.
            all_frac_coords: Full fractional coordinate array, shape
                ``(n_atoms, 3)``.
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Returns:
            True if every point is contained by its corresponding site.
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
