"""Collection manager for polyhedral sites in crystal structures.

This module provides the PolyhedralSiteCollection class, which manages a
collection of PolyhedralSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The PolyhedralSiteCollection extends the base SiteCollection class with
specific functionality for polyhedral sites, including:
- Maintaining a map of neighbouring polyhedral sites that share faces
- Optimised atom assignment using priority-based site checking
- Using observed transition patterns between sites for performance optimisation
- Precomputed distance-ranked site ordering when reference centres are available

The module also includes utility functions:
- construct_neighbouring_sites: analyses polyhedral sites to determine which
  ones are face-sharing neighbours (sharing three or more vertices).
- _compute_distance_ranked_sites: precomputes distance-ranked site lists
  from reference centres for efficient fallback ordering.

For atom assignment, the collection implements a priority-based optimisation
that reduces average search complexity from O(N) to O(k):

When trajectory history exists:
    1. Check the most recently visited site, then the previously visited
       site (covers staying put and bouncing back)
    2. Check learned transition destinations in frequency order
    3. Check remaining sites by distance from anchor site (if reference
       centres are available), otherwise face-sharing neighbours then
       arbitrary order

When no trajectory history exists:
    - If reference centres are available: check the nearest site centre
      first, then learned transitions, then distance-ranked outward
    - Otherwise: check all sites in arbitrary order

Note: distance ranking uses minimum-image convention in fractional space,
which is only geometrically exact for orthogonal cells. For non-orthogonal
cells the ranking is approximate, but correctness is unaffected since all
sites are eventually checked.

This approach leverages spatial relationships, learned transition patterns,
and distance-based ordering to reduce the number of containment checks.
"""

from collections.abc import Generator
from typing import NamedTuple

from .site_collection import SiteCollection
from .polyhedral_site import PolyhedralSite
from .atom import Atom
from .site import Site
from .tools import x_pbc
from pymatgen.core import Structure # type: ignore
import numpy as np


class _ReferenceData(NamedTuple):
    """Precomputed reference centre data for nearest-site lookups."""
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

class PolyhedralSiteCollection(SiteCollection):
    """A collection of PolyhedralSite objects.
    
    Extends the base SiteCollection class with specific functionality for
    polyhedral sites, including maintaining a map of neighbouring polyhedral
    sites that share faces and implementing optimised atom assignment based
    on spatial relationships and learned transition patterns.
    
    Attributes:
        sites (list): List of ``PolyhedralSite`` objects.
    
    """

    def __init__(self,
            sites: list[Site]) -> None:
        """Create a PolyhedralSiteCollection instance.

        Args:
            sites (list(PolyhedralSite)): List of PolyhedralSite objects.

        Returns:
            None

        """
        for s in sites:
            if not isinstance(s, PolyhedralSite):
                raise TypeError
        super(PolyhedralSiteCollection, self).__init__(sites)
        self.sites: list[PolyhedralSite]
        self._neighbouring_sites = construct_neighbouring_sites(self.sites)
        self._distance_ranked_sites, self._reference_data = (
            _compute_distance_ranked_sites(self.sites)
        )

    def analyse_structure(self,
            atoms: list[Atom],
            structure: Structure):
        for a in atoms:
            a.assign_coords(structure)
        all_frac_coords = structure.frac_coords
        lattice = structure.lattice
        for s in self.sites:
            s.notify_structure_changed(all_frac_coords, lattice)
        self.assign_site_occupations(atoms, structure)
                    
    def assign_site_occupations(self, atoms, structure) -> None:
        """Assign atoms to polyhedral sites based on their positions.
        
        This method implements an optimised assignment logic using a priority-based
        site checking approach.
        
        Args:
            atoms: List of Atom objects to be assigned to sites
            structure: Pymatgen Structure containing the atom positions
        """
        self.reset_site_occupations()
        for atom in atoms:
            atom.in_site = None
            pbc_images = x_pbc(atom.frac_coords)

            # Check sites in priority order until found
            for site in self._get_priority_sites(atom):
                if site.contains_atom(atom, pbc_images=pbc_images):
                    self.update_occupation(site, atom)
                    break
    
    def _get_priority_sites(self, atom: Atom) -> Generator[PolyhedralSite, None, None]:
        """Generator that yields sites in priority order for optimised atom assignment.

        The checking sequence depends on available information:

        When trajectory history exists:
            1. Most recently visited site, then previously visited site
            2. Learned transition destinations in frequency order
            3. Remaining sites by distance from anchor (if reference centres
               available), otherwise face-sharing neighbours then arbitrary order

        When no trajectory history exists:
            - If reference centres are available: nearest site centre first,
              then learned transitions, then distance-ranked outward
            - Otherwise: all sites in arbitrary order

        Each site is yielded at most once.

        Args:
            atom: Atom object with trajectory history used to determine
                site priorities.

        Yields:
            PolyhedralSite: Sites in optimal checking order.
        """
        checked_indices: set[int] = set()
        anchor_index = None

        recent = [s for s in atom._recent_sites if s is not None]
        if recent:
            anchor_index = recent[0]
            for index in recent:
                yield self.site_by_index(index)
                checked_indices.add(index)
        elif self._reference_data is not None:
            anchor_index = self._reference_data.nearest_site_index(atom.frac_coords)
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

    def neighbouring_sites(self,
            index: int) -> list[PolyhedralSite]:
        return self._neighbouring_sites[index] 

    def sites_contain_points(self,
            points: np.ndarray,
            structure: Structure | None=None) -> bool:
        """Checks whether the set of sites contain 
        a corresponding set of fractional coordinates.

        Args:
            points (np.array): 3xN numpy array of fractional coordinates.
                There should be one coordinate for each site being checked.
            structure (Structure): Pymatgen Structure used to define the
                vertex coordinates of each polyhedral site.
        
        Returns:
            (bool)

        """
        if not isinstance(structure, Structure):
            raise TypeError(f"Expected a Structure, got {type(structure).__name__}")
        check = all([s.contains_point(p,structure) for s, p in zip(self.sites, points)])
        return check

def _compute_distance_ranked_sites(
        sites: list[PolyhedralSite],
) -> tuple[dict[int, list[int]] | None, _ReferenceData | None]:
    """Precompute distance-ranked site lists from reference centres.

    For each site, produces a list of all other site indices sorted by
    distance from that site's reference centre, using minimum-image
    convention in fractional space.

    Args:
        sites: List of PolyhedralSite objects.

    Returns:
        A tuple of (ranked_dict, reference_data) where:
        - ranked_dict maps site index to a list of other site indices
          sorted by distance, or None if any site lacks a reference centre.
        - reference_data contains the centres array and site indices for
          nearest-site lookups, or None.
    """
    centres = []
    for s in sites:
        if s.reference_center is None:
            return None, None
        centres.append(s.reference_center)
    centres_array = np.array(centres)
    site_indices = [s.index for s in sites]

    ranked: dict[int, list[int]] = {}
    for i, site in enumerate(sites):
        diffs = centres_array - centres_array[i]
        diffs -= np.round(diffs)
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)
        ranked[site.index] = [site_indices[j] for j in order if j != i]
    return ranked, _ReferenceData(centres=centres_array, site_indices=site_indices)


def construct_neighbouring_sites(
        sites: list[PolyhedralSite]) -> dict[int, list[PolyhedralSite]]:
    """
    Find all polyhedral sites that are face-sharing neighbours.

    Any polyhedral sites that share 3 or more vertices are considered
    to share a face.

    Args:
        sites: List of PolyhedralSite objects to check for shared faces.

    Returns:
        (dict): Dictionary of `int`: `list` entries.
            Keys are site indices. Values are lists of ``PolyhedralSite`` objects.

    """
    neighbours: dict[int, list[PolyhedralSite]] = {}
    for site_i in sites:
        neighbours[site_i.index] = []
        for site_j in sites:
            if site_i is site_j:
                continue
            # 3 or more common vertices indicated a shared face.
            n_shared_vertices = len(set(site_i.vertex_indices) & set(site_j.vertex_indices))
            if n_shared_vertices >= 3:
                neighbours[site_i.index].append(site_j)
    return neighbours
 
