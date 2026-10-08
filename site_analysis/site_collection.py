"""Base classes for collections of sites in crystal structures.

This module defines:

- ``SiteCollection``: abstract base class that all site collection types
  must inherit from. Provides the interface for site-atom assignment and
  common functionality for managing site occupations.
- ``PriorityAssignmentMixin``: mixin providing priority-based site
  assignment ordering. Used by collection types that check sites one at
  a time (polyhedral, spherical) but not by those that assign each atom
  to its nearest site centre (Voronoi, dynamic Voronoi).
- ``_NearestSiteLookup``: precomputed lookup for finding the nearest
  site to a given position.
- ``_DistanceRanking``: site centres for ranking sites by distance from
  an anchor site.
- ``_SiteCentreIndex``: site centres, ranked by distance from a point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from typing import Generic, NamedTuple, TypeVar, TYPE_CHECKING

import numpy as np
from .atom import Atom
from .site import Site
from .neighbour_search import PeriodicNeighbourIndex


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


class _DistanceRanking(NamedTuple):
    """Site centres for ranking sites by distance from an anchor site.

    Each ranking is computed when requested. Uses minimum-image
    convention in fractional space, which is only geometrically exact for
    orthogonal cells.
    """
    centres: np.ndarray
    site_indices: np.ndarray
    positions: dict[int, int]

    def ranked_site_indices(self, anchor_index: int) -> list[int]:
        """Return every other site index, nearest to the anchor site first.

        Args:
            anchor_index: Index of the site to rank from.

        Returns:
            Indices of all sites except the anchor, ordered by the
            distance between their centres and the anchor's centre.
        """
        i = self.positions[anchor_index]
        diffs = self.centres - self.centres[i]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        ranked: list[int] = self.site_indices[order[order != i]].tolist()
        return ranked


class _SiteCentreIndex:
    """Site centres, ranked by Cartesian minimum-image distance from a point.

    Holds a ``PeriodicNeighbourIndex`` over the centres, built on first
    use and rebuilt whenever the lattice changes.
    """

    def __init__(self,
            centres: np.ndarray,
            site_indices: Sequence[int]) -> None:
        """Create a _SiteCentreIndex.

        Args:
            centres: Fractional coordinates of the site centres, shape
                (N, 3), with N at least 1.
            site_indices: The site index of each centre.
        """
        self._centres = np.array(centres, dtype=np.float64)
        self._site_indices = np.asarray(site_indices)
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
            volume = abs(np.linalg.det(self._lattice_matrix))
            self._nearby_radius = (volume / len(self._centres)) ** (1 / 3)
        return self._index

    def ranked_site_indices(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> Iterator[list[int]]:
        """Yield site indices in order of distance from a point.

        Joined, the lists hold every site once, ordered by the Cartesian
        minimum-image distance of its centre from the point, with equal
        distances in the order of the centres. The first list holds the
        sites near the point. The rest are ranked only if the caller asks
        for another list.

        Args:
            frac_coords: Fractional coordinates of the point, shape (3,).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors.

        Yields:
            Lists of site indices.

        Raises:
            ValueError: If the lattice matrix is singular or non-finite.
                As this is a generator, the error is raised when the first
                list is requested.
        """
        neighbour_index = self._index_for(lattice_matrix)
        query = np.reshape(frac_coords, (1, 3))
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

    Provides ``_get_priority_sites(atom)``, a generator that yields sites
    in an optimised order based on recent site history, learned transitions,
    and distance ranking.

    Subclasses call ``_init_priority_ranking(centres, site_indices)`` from
    their ``__init__`` to enable distance-ranked ordering. If not called,
    the generator falls back to ``neighbouring_sites`` then arbitrary
    order (used by ``PolyhedralSiteCollection`` when reference centres
    are unavailable).

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
        sites: Sequence[SiteT]
        def site_by_index(self, index: int) -> SiteT: ...
        def neighbouring_sites(self, site_index: int) -> Sequence[SiteT]: ...

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._distance_ranking: _DistanceRanking | None = None
        self._nearest_site_lookup: _NearestSiteLookup | None = None

    def _init_priority_ranking(self, centres: np.ndarray, site_indices: list[int]) -> None:
        """Set up distance-ranked site ordering from the given centres.

        Does nothing if ``centres`` is empty (zero sites).

        Args:
            centres: (N, 3) array of fractional coordinates for each site.
            site_indices: Corresponding site indices.
        """
        if len(centres) == 0:
            return
        self._distance_ranking = _DistanceRanking(
            centres=centres,
            site_indices=np.asarray(site_indices),
            positions={idx: i for i, idx in enumerate(site_indices)},
        )
        self._nearest_site_lookup = _NearestSiteLookup(
            centres=centres, site_indices=site_indices
        )

    def _get_priority_sites(self, atom: Atom) -> Generator[SiteT, None, None]:
        """Generator that yields sites in priority order for optimised atom assignment.

        The generator picks an *anchor site* — the most recent site from
        the atom's history, or the nearest site centre if no history
        exists — and uses it to order the remaining sites.

        The checking sequence depends on available information:

        When trajectory history exists:
            1. Most recently visited site, then previously visited site
            2. Learned transition destinations from the anchor in
               frequency order
            3. Remaining sites by distance from anchor (if distance ranking
               available), otherwise neighbours then arbitrary order

        When no trajectory history exists:
            - If distance ranking is available: nearest site centre
              (anchor) first, then learned transitions, then
              distance-ranked outward
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
            if self._distance_ranking is not None:
                for index in self._distance_ranking.ranked_site_indices(anchor_index):
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
