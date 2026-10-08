import itertools
import unittest
from site_analysis.site_collection import (
    SiteCollection, PriorityAssignmentMixin, _SiteCentreIndex,
)
from site_analysis.site import Site
from site_analysis.atom import Atom
from pymatgen.core import Lattice, Structure
from unittest.mock import patch, Mock, MagicMock
import numpy as np
from collections import Counter

class ConcreteSiteCollection(SiteCollection):

    def assign_site_occupations(self,
                                atoms,
                                structure):
        raise NotImplementedError

    def analyse_structure(self,
                          atoms,
                          structure):
        raise NotImplementedError


class SiteCollectionTestCase(unittest.TestCase):

    def test_site_collection_is_initialised(self):
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        site_collection = ConcreteSiteCollection(sites=sites)
        self.assertEqual(site_collection.sites, sites)

    def test_site_collection_keeps_its_own_list_of_sites(self):
        """Appending to the caller's list afterwards does not change the collection."""
        site = Mock(spec=Site, index=0)
        sites = [site]
        site_collection = ConcreteSiteCollection(sites=sites)
        sites.append(Mock(spec=Site, index=1))
        self.assertEqual(site_collection.sites, [site])

    def test_site_collection_finds_every_site_given_as_an_iterator(self):
        """Sites given as an iterator can all be found by index."""
        sites = [Mock(spec=Site, index=12), Mock(spec=Site, index=42)]
        site_collection = ConcreteSiteCollection(sites=iter(sites))
        self.assertEqual(
            [site_collection.site_by_index(12), site_collection.site_by_index(42)], sites)

    def test_assign_site_occupations_raises_not_implemented_error(self):
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        atoms = [Mock(spec=Atom), 
                 Mock(spec=Atom)]
        lattice_matrix = np.eye(3) * 10.0
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.assign_site_occupations(atoms, lattice_matrix)

    def test_analyse_structure_raises_not_implemented_error(self):
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        atoms = [Mock(spec=Atom),
                 Mock(spec=Atom)]
        structure = Mock(spec=Structure)
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.analyse_structure(atoms, structure)

    def test_neighbouring_sites_raises_not_implemented_error(self):
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.neighbouring_sites(site_index=27)
   
    def test_site_by_index(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        site_collection = ConcreteSiteCollection(sites=sites)
        self.assertEqual(site_collection.site_by_index(12), sites[0])
        self.assertEqual(site_collection.site_by_index(42), sites[1])
        with self.assertRaises(ValueError):
            site_collection.site_by_index(93)

    def test_update_occupation(self):
        sites = [Mock(spec=Site)]
        sites[0].index = 12
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].contains_atoms, [atom.index])
        np.testing.assert_array_equal(sites[0].points, [atom.frac_coords])
        self.assertEqual(atom.in_site, sites[0].index)

    def test_update_occupation_does_not_record_transitions(self):
        """Transitions are recorded by Trajectory.append_timestep, not here."""
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[0].contains_atoms = []
        sites[0].points = []
        sites[0].transitions = Counter()
        sites[1].index = 42
        sites[1].transitions = Counter()
        site_collection = ConcreteSiteCollection(sites=sites)
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.most_recent_site = 42
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].transitions, Counter())
        self.assertEqual(sites[1].transitions, Counter())

    def test_update_occupation_does_not_update_recent_site(self):
        """The recent-site memory is updated by Trajectory.append_timestep, not here."""
        sites = [Mock(spec=Site)]
        sites[0].index = 5
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        atom = Mock(spec=Atom)
        atom.index = 1
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        atom.update_recent_site.assert_not_called()

    def test_reset_calls_reset_on_all_sites(self):
        """reset() should call site.reset() for every site in the collection."""
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        site_collection = ConcreteSiteCollection(sites=sites)
        site_collection.reset()
        sites[0].reset.assert_called_once()
        sites[1].reset.assert_called_once()

    def test_reset_site_occupations(self):
        sites = [Mock(spec=Site, index=0), 
                 Mock(spec=Site, index=1)]
        sites[0].contains_atoms = [12]
        sites[1].contains_atoms = [42]
        site_collection = ConcreteSiteCollection(sites=sites)
        site_collection.reset_site_occupations()
        self.assertEqual(sites[0].contains_atoms, [])
        self.assertEqual(sites[1].contains_atoms, [])
   
    def test_sites_contain_points_raises_not_implemented_error(self):
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        site_collection = ConcreteSiteCollection(sites=sites)
        points = np.array([[0.0, 0.0, 0.0],
                           [0.5, 0.5, 0.5]])
        with self.assertRaises(NotImplementedError):
            site_collection.sites_contain_points(
                points=points,
                all_frac_coords=np.eye(3),
                lattice_matrix=np.eye(3) * 10.0)
            
    def test_site_lookup_dict_creation(self):
        """Test that a site lookup dictionary is created during initialization."""
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        site_collection = ConcreteSiteCollection(sites=sites)
        
        # Check that the lookup dictionary exists
        self.assertTrue(hasattr(site_collection, '_site_lookup'))
        
        # Check that it contains the correct mappings
        self.assertEqual(site_collection._site_lookup[12], sites[0])
        self.assertEqual(site_collection._site_lookup[42], sites[1])
    
    def test_site_by_index_uses_lookup(self):
        """Test that site_by_index uses the lookup dictionary."""
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        site_collection = ConcreteSiteCollection(sites=sites)
        
        # Replace the lookup dictionary with a Mock that we can track
        original_lookup = site_collection._site_lookup
        mock_lookup = MagicMock()
        # Configure the mock to return values like the original dict
        mock_lookup.get.side_effect = lambda k, default=None: original_lookup.get(k, default)
        site_collection._site_lookup = mock_lookup
        
        # Call site_by_index
        result = site_collection.site_by_index(12)
        
        # Verify the result is correct
        self.assertEqual(result, sites[0])
        
        # Verify the lookup dictionary was used
        mock_lookup.get.assert_called_once_with(12)
        
        # Restore the original lookup dict to avoid affecting other tests
        site_collection._site_lookup = original_lookup
    
    def test_site_by_index_raises_value_error(self):
        """Test that site_by_index raises ValueError for non-existent indices."""
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        site_collection = ConcreteSiteCollection(sites=sites)
        
        with self.assertRaises(ValueError):
            site_collection.site_by_index(99)  # Non-existent index
    
    def test_duplicate_site_indices_error(self):
        """Test that an error is raised if sites have duplicate indices."""
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 12  # Same index as sites[0]
        
        with self.assertRaises(ValueError) as context:
            site_collection = ConcreteSiteCollection(sites=sites)
        
        # Verify error message mentions duplicate indices
        self.assertIn('duplicate', str(context.exception).lower())
        self.assertIn('12', str(context.exception))
        
    def test_summaries_default(self):
        """Test that summaries returns list of summaries for all sites."""
        # Create sites with index attribute set
        site1 = Mock(spec=Site, index=0)
        site1.summary.return_value = {'index': 0, 'site_type': 'MockSite'}
        
        site2 = Mock(spec=Site, index=1)
        site2.summary.return_value = {'index': 1, 'site_type': 'MockSite'}
        
        collection = ConcreteSiteCollection([site1, site2])
        
        summaries = collection.summaries()
        
        # Should call summary() on each site with default metrics
        site1.summary.assert_called_once_with(metrics=None)
        site2.summary.assert_called_once_with(metrics=None)
        
        # Should return list of summaries
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]['index'], 0)
        self.assertEqual(summaries[1]['index'], 1)
    
    def test_summaries_with_metrics(self):
        """Test that summaries passes metrics parameter to each site."""
        site1 = Mock(spec=Site, index=0)
        site1.summary.return_value = {'index': 0}
        
        site2 = Mock(spec=Site, index=1)
        site2.summary.return_value = {'index': 1}
        
        collection = ConcreteSiteCollection([site1, site2])
        
        summaries = collection.summaries(metrics=['index'])
        
        # Should pass metrics to each site
        site1.summary.assert_called_once_with(metrics=['index'])
        site2.summary.assert_called_once_with(metrics=['index'])
        
        self.assertEqual(summaries, [{'index': 0}, {'index': 1}])
    
    def test_summaries_empty_collection(self):
        """Test that summaries returns empty list for empty collection."""
        collection = ConcreteSiteCollection([])
        
        summaries = collection.summaries()
        
        self.assertEqual(summaries, [])
    
    def test_summaries_preserves_order(self):
        """Test that summaries preserves site order."""
        sites = []
        for i in range(3):
            site = Mock(spec=Site, index=i)
            site.summary.return_value = {'index': i}
            sites.append(site)
        
        collection = ConcreteSiteCollection(sites)
        
        summaries = collection.summaries()
        
        # Should preserve order
        self.assertEqual([s['index'] for s in summaries], [0, 1, 2])

class ConcretePriorityCollection(PriorityAssignmentMixin, SiteCollection):
    """Concrete class for testing PriorityAssignmentMixin."""

    def assign_site_occupations(self, atoms, structure):
        raise NotImplementedError

    def analyse_structure(self, atoms, structure):
        raise NotImplementedError


def _brute_force_distances(centres, point, lattice_matrix):
    """Minimum-image Cartesian distances from the point to each centre, over 27 images."""
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    diffs = centres - point
    diffs -= np.round(diffs)
    cartesian = (diffs[:, np.newaxis, :] + shifts) @ lattice_matrix
    return np.linalg.norm(cartesian, axis=2).min(axis=1)


def _brute_force_ranking(centres, point, lattice_matrix):
    """Positions of the centres, nearest to the point first.

    Equal distances keep the order of the centres.
    """
    return np.argsort(_brute_force_distances(centres, point, lattice_matrix), kind="stable")


def _joined_ranking(site_centres, point, lattice_matrix):
    """All site indices from ranked_site_indices, as one list."""
    return [index
            for ranked in site_centres.ranked_site_indices(point, lattice_matrix)
            for index in ranked]


class TestSiteCentreIndex(unittest.TestCase):
    """Tests for _SiteCentreIndex."""

    def test_rejects_centres_not_shaped_n_by_3(self):
        """Centres must be shaped (N, 3), with N at least 1."""
        for centres in (np.zeros((0, 3)), np.zeros((2, 2)), np.zeros(3)):
            with self.subTest(shape=centres.shape):
                with self.assertRaisesRegex(ValueError, r"centres must have shape \(N, 3\)"):
                    _SiteCentreIndex(centres, list(range(len(centres))))

    def test_rejects_non_finite_centres(self):
        """Centres must be finite."""
        with self.assertRaisesRegex(ValueError, "centres must be finite"):
            _SiteCentreIndex(np.array([[0.1, np.nan, 0.2]]), [0])

    def test_rejects_site_indices_not_one_per_centre(self):
        """There must be one site index per centre."""
        with self.assertRaisesRegex(ValueError, "one site index per centre"):
            _SiteCentreIndex(np.zeros((2, 3)), [0])

    def test_rejects_negative_or_nan_reach(self):
        """A reach, if given, must be non-negative."""
        for reach in (-1.0, np.nan):
            with self.subTest(reach=reach):
                with self.assertRaisesRegex(ValueError, "reach must be non-negative"):
                    _SiteCentreIndex(np.zeros((1, 3)), [0], reach=reach)

    def test_keeps_its_own_copy_of_the_site_indices(self):
        """Changing the caller's site indices afterwards does not change the ranking."""
        # An array, which np.asarray would share rather than copy.
        site_indices = np.array([7, 9])
        site_centres = _SiteCentreIndex(np.array([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]), site_indices)
        site_indices[0] = 99
        self.assertEqual(_joined_ranking(site_centres, np.zeros(3), np.eye(3) * 10.0), [7, 9])

    def test_ranks_every_site_by_minimum_image_distance(self):
        """Joined, the lists hold every site once, nearest to the point first."""
        rng = np.random.default_rng(0)
        lattice_matrix = Lattice.from_parameters(9.0, 10.0, 11.0, 75, 100, 110).matrix
        centres = rng.random((30, 3))
        site_indices = list(range(100, 160, 2))
        site_centres = _SiteCentreIndex(centres, site_indices)
        for point in rng.uniform(-0.5, 1.5, (20, 3)):
            expected = [site_indices[position]
                        for position in _brute_force_ranking(centres, point, lattice_matrix)]
            self.assertEqual(_joined_ranking(site_centres, point, lattice_matrix), expected)

    def test_first_list_holds_only_nearby_sites(self):
        """The first list leaves distant sites for later lists."""
        grid = np.arange(4) / 4
        centres = np.array([[x, y, z] for x in grid for y in grid for z in grid])
        site_centres = _SiteCentreIndex(centres, list(range(64)))
        first = next(site_centres.ranked_site_indices(np.array([0.5, 0.5, 0.5]), np.eye(3) * 8.0))
        self.assertLess(len(first), 64)
        # The site at the point itself comes first.
        self.assertEqual(first[0], 42)

    def test_ranking_is_cartesian_in_a_skewed_cell(self):
        """Sites are ranked by Cartesian, not fractional, distance."""
        lattice_matrix = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 30).matrix
        # (0.3, 0, 0) is 3.0 A from the origin. (0.3, -0.3, 0) is further
        # in fractional units but only 1.55 A away.
        centres = np.array([[0.3, 0.0, 0.0], [0.3, -0.3, 0.0]])
        site_centres = _SiteCentreIndex(centres, [0, 1])
        self.assertEqual(_joined_ranking(site_centres, np.zeros(3), lattice_matrix), [1, 0])

    def test_equal_distances_rank_lower_position_first(self):
        """Sites at equal distances are ranked in the order of their centres."""
        # From the point, the first and third centres are both 2 A away and
        # the second is 4 A away.
        centres = np.array([[0.75, 0.5, 0.5], [0.0, 0.5, 0.5], [0.25, 0.5, 0.5]])
        site_centres = _SiteCentreIndex(centres, [7, 9, 3])
        self.assertEqual(
            _joined_ranking(site_centres, np.array([0.5, 0.5, 0.5]), np.eye(3) * 8.0),
            [7, 3, 9])

    def test_ranking_follows_lattice_changes(self):
        """Changing the lattice, even in place, changes the ranking."""
        centres = np.array([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0]])
        site_centres = _SiteCentreIndex(centres, [0, 1])
        lattice_matrix = np.eye(3) * 10.0
        # 2 A and 3 A away in a 10 A cubic cell; 4 A and 3 A away once a is doubled.
        self.assertEqual(_joined_ranking(site_centres, np.zeros(3), lattice_matrix), [0, 1])
        lattice_matrix[0, 0] = 20.0
        self.assertEqual(_joined_ranking(site_centres, np.zeros(3), lattice_matrix), [1, 0])

    def test_reach_limits_ranking_to_sites_within_reach(self):
        """With a reach, one list holds the sites within it, including one exactly at it."""
        # 2 A, 1 A, 2.002 A and 5.7 A from the origin.
        centres = np.array([[0.0, 0.25, 0.0], [0.125, 0.0, 0.0],
                            [0.0, 0.0, 0.25025], [0.5, 0.5, 0.0]])
        site_centres = _SiteCentreIndex(centres, [0, 1, 2, 3], reach=2.0)
        self.assertEqual(
            list(site_centres.ranked_site_indices(np.zeros(3), np.eye(3) * 8.0)), [[1, 0]])

    def test_reach_includes_sites_just_beyond_it_within_the_tolerance(self):
        """A site just beyond the reach, within the rounding tolerance, is ranked."""
        # 2 A from the origin, just beyond the reach.
        site_centres = _SiteCentreIndex(np.array([[0.25, 0.0, 0.0]]), [5], reach=2.0 * (1 - 1e-12))
        self.assertEqual(
            list(site_centres.ranked_site_indices(np.zeros(3), np.eye(3) * 8.0)), [[5]])

    def test_reach_ranking_matches_brute_force(self):
        """With a reach, the list is the brute-force ranking cut at the reach."""
        rng = np.random.default_rng(1)
        lattice_matrix = Lattice.from_parameters(9.0, 10.0, 11.0, 75, 100, 110).matrix
        centres = rng.random((30, 3))
        site_indices = list(range(100, 160, 2))
        site_centres = _SiteCentreIndex(centres, site_indices, reach=4.0)
        for point in rng.uniform(-0.5, 1.5, (20, 3)):
            distances = _brute_force_distances(centres, point, lattice_matrix)
            expected = [site_indices[position]
                        for position in np.argsort(distances, kind="stable")
                        if distances[position] <= 4.0]
            self.assertEqual(
                list(site_centres.ranked_site_indices(point, lattice_matrix)), [expected])


class TestInitPriorityRanking(unittest.TestCase):
    """Tests for PriorityAssignmentMixin._init_priority_ranking."""

    def test_empty_sites_is_noop(self):
        """Calling _init_priority_ranking with empty centres does nothing."""
        collection = ConcretePriorityCollection([])
        collection._init_priority_ranking(np.empty((0, 3)), [])
        self.assertIsNone(collection._site_centres)


class TestGetPrioritySitesWithSiteCentres(unittest.TestCase):
    """Tests for the site search order when site centres are set up."""

    def setUp(self):
        # Four sites along x in a 10 A cubic cell, at x = 1, 3, 5 and 7.5 A,
        # with site indices that differ from their positions in the list.
        self.sites = [Mock(spec=Site, index=i, frac_coords=np.array([x, 0.0, 0.0]))
                      for i, x in zip([5, 3, 0, 7], [0.1, 0.3, 0.5, 0.75])]
        for site in self.sites:
            site.most_frequent_transitions.return_value = []
        self.collection = ConcretePriorityCollection(self.sites)
        self.centres = np.array([s.frac_coords for s in self.sites])
        self.site_indices = [s.index for s in self.sites]
        self.collection._init_priority_ranking(self.centres, self.site_indices)
        self.lattice_matrix = np.eye(3) * 10.0
        # 0.7 A from site 7, 1.8 A from site 0, 3.8 A from site 3 and
        # 4.2 A from site 5.
        self.atom = Atom(index=0)
        self.atom._frac_coords = np.array([0.68, 0.0, 0.0])

    def priority_indices(self):
        """Site indices in the order _get_priority_sites yields them."""
        return [site.index
                for site in self.collection._get_priority_sites(self.atom, self.lattice_matrix)]

    def test_recent_sites_come_first_most_recent_first(self):
        """Both recent sites come first, most recent first, before the ranking."""
        self.atom._recent_sites = [3, 5]
        self.assertEqual(self.priority_indices(), [3, 5, 7, 0])

    def test_remaining_sites_ranked_by_distance_from_atom(self):
        """After the recent site, sites are ranked by distance from the atom."""
        self.atom._recent_sites = [5, None]
        # Ranked from site 5's centre instead, site 3 would come second.
        self.assertEqual(self.priority_indices(), [5, 7, 0, 3])

    def test_transitions_come_before_distance_ranking(self):
        """Learned transitions from the recent site come before the ranking."""
        self.atom._recent_sites = [5, None]
        self.sites[0].most_frequent_transitions.return_value = [0]
        self.assertEqual(self.priority_indices(), [5, 0, 7, 3])

    def test_atom_without_history_gets_distance_order(self):
        """An atom with no recent site gets distance order, without transitions."""
        self.sites[3].most_frequent_transitions.return_value = [3]
        # Ranked from the nearest site's centre (site 7's) instead, site 5
        # would come before site 3; with that site's transitions, site 3
        # would come second.
        self.assertEqual(self.priority_indices(), [7, 0, 3, 5])

    def test_transitions_keep_frequency_order(self):
        """Learned transitions come in frequency order, not distance order."""
        self.atom._recent_sites = [5, None]
        # Site 3 is further from the atom than site 0 but the more frequent
        # destination, so it comes first.
        self.sites[0].most_frequent_transitions.return_value = [3, 0]
        self.assertEqual(self.priority_indices(), [5, 3, 0, 7])

    def test_transition_back_to_previous_site_not_repeated(self):
        """A transition back to the previous recent site is not yielded twice."""
        self.atom._recent_sites = [3, 5]
        self.sites[1].most_frequent_transitions.return_value = [5, 0]
        self.assertEqual(self.priority_indices(), [3, 5, 0, 7])

    def test_distance_ranking_stops_at_reach(self):
        """With a reach, the ranking holds only the sites within it of the atom."""
        self.collection._init_priority_ranking(self.centres, self.site_indices, reach=2.0)
        self.assertEqual(self.priority_indices(), [7, 0])

    def test_recent_sites_and_transitions_yielded_beyond_reach(self):
        """Recent sites and learned transitions are yielded even when out of reach."""
        self.collection._init_priority_ranking(self.centres, self.site_indices, reach=2.0)
        self.atom._recent_sites = [5, None]
        self.sites[0].most_frequent_transitions.return_value = [3]
        self.assertEqual(self.priority_indices(), [5, 3, 7, 0])


class TestGetPrioritySitesBeyondNearbyRadius(unittest.TestCase):
    """Tests for sites further from the atom than the nearby radius."""

    def test_site_beyond_nearby_radius_is_yielded(self):
        """A site further from the atom than the nearby radius is still yielded."""
        # Two sites in a 10 A cubic cell, with site indices that differ
        # from their positions in the list. The nearby radius is
        # (1000 / 2) ** (1 / 3) = 7.94 A, and the second centre is 8.66 A
        # from the atom, so it is only in the second list of the ranking.
        sites = [Mock(spec=Site, index=i, frac_coords=np.array(centre))
                 for i, centre in [(4, [0.0, 0.0, 0.0]), (9, [0.5, 0.5, 0.5])]]
        collection = ConcretePriorityCollection(sites)
        collection._init_priority_ranking(np.array([s.frac_coords for s in sites]), [4, 9])
        atom = Atom(index=0)
        atom._frac_coords = np.zeros(3)
        order = [site.index for site in collection._get_priority_sites(atom, np.eye(3) * 10.0)]
        self.assertEqual(order, [4, 9])


if __name__ == '__main__':
    unittest.main()

