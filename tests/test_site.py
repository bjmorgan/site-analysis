import unittest
from site_analysis.site import Site
from site_analysis.atom import Atom
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from collections import Counter
from typing import Any

class ConcreteSite(Site):

    def centre(self) -> np.ndarray:
        raise NotImplementedError

    def contains_point(self,
                       x: np.ndarray,
                       *args: Any,
                       **kwargs: Any) -> bool:
        raise NotImplementedError
        
    def __repr__(self) -> str:
        raise NotImplementedError

class SiteTestCase(unittest.TestCase):

    def setUp(self):
        Site._newid = 0

    def test_site_is_initialised(self):
        site = ConcreteSite()
        self.assertEqual(site.index, 0)
        self.assertEqual(site.label, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.points, [])
        self.assertEqual(site.transitions, {})

    def test_site_is_initialised_with_label(self):
        site = ConcreteSite(label='foo')
        self.assertEqual(site.label, 'foo')

    def test_site_index_autoincrements(self):
        site1 = ConcreteSite()
        site2 = ConcreteSite()
        self.assertEqual(site2.index, site1.index + 1)

    def test_reset(self):
        site = ConcreteSite()
        site.contains_atoms = ['foo']
        site.trajectory = ['bar']
        site.transitions = Counter([4])
        site.reset()
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.transitions, {})

    def test_contains_point_raises_not_implemented_error(self):
        site = ConcreteSite()
        with self.assertRaises(NotImplementedError):
            site.contains_point(np.array([0,0,0]))

    def test_centre_raises_not_implemented_error(self):
        site = ConcreteSite()
        with self.assertRaises(NotImplementedError):
            site.centre()

    def test_contains_atom(self):
        site = ConcreteSite()
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0,0,0])
        site.contains_point = Mock(return_value=True)
        self.assertTrue(site.contains_atom(atom))
        site.contains_point = Mock(return_value=False)
        self.assertFalse(site.contains_atom(atom))

    def test_as_dict(self):
        site = ConcreteSite()
        site.index = 7
        site.contains_atoms = [3]
        site.trajectory = [10,11,12]
        site.points = [np.array([0,0,0])]
        site.label = 'foo'
        site.transitions = Counter([3,3,2])
        site_dict = site.as_dict()
        expected_dict = {'index': 7,
                         'contains_atoms': [3],
                         'trajectory': [10,11,12],
                         'points': [np.array([0,0,0])],
                         'label': 'foo',
                         'transitions': Counter([3,3,2])}
        self.assertEqual(site.index, expected_dict['index'])
        self.assertEqual(site.contains_atoms, expected_dict['contains_atoms'])
        self.assertEqual(site.trajectory, expected_dict['trajectory'])
        np.testing.assert_array_equal(site.points, expected_dict['points'])
        self.assertEqual(site.label, expected_dict['label'])
        self.assertEqual(site.transitions, expected_dict['transitions'])

    def test_reset_index(self):
        Site._newid = 7
        site = ConcreteSite()
        self.assertEqual(site.index, 7)
        Site.reset_index()
        site = ConcreteSite()
        self.assertEqual(site.index, 0)
        
    def test_reset_index_to_defined_index(self):
        Site._newid = 7
        site = ConcreteSite()
        self.assertEqual(site.index, 7)
        Site.reset_index(newid=12)
        site = ConcreteSite()
        self.assertEqual(site.index, 12)
   
    def test_from_dict(self):
        site_dict = {'index': 7,
                     'contains_atoms': [3],
                     'trajectory': [10,11,12],
                     'points': [np.array([0,0,0])],
                     'label': 'foo',
                     'transitions': Counter([3,3,2])}
        site = ConcreteSite.from_dict(site_dict)
        self.assertEqual(site.index, site_dict['index'])
        self.assertEqual(site.contains_atoms, site_dict['contains_atoms'])
        self.assertEqual(site.trajectory, site_dict['trajectory'])
        np.testing.assert_array_equal(site.points, site_dict['points'])
        self.assertEqual(site.label, site_dict['label'])
        self.assertEqual(site.transitions, site_dict['transitions'])
        
    def test_repr_is_abstract_method(self):
        """Test that __repr__ is an abstract method in Site class."""
        abstract_methods = getattr(Site, "__abstractmethods__", set())
        self.assertIn("__repr__", abstract_methods)
   
    def test_coordination_number_raises_not_implemented_error(self):
        """Test coordination_number raises NotImplementedError in abstract Site."""
        site = ConcreteSite()

        with self.assertRaises(NotImplementedError):
            _ = site.coordination_number
            
    def test_most_frequent_transitions_ordered_by_frequency(self):
        """Test that most_frequent_transitions returns site indices ordered by frequency."""
        site = ConcreteSite()  # Using the existing test concrete class
        site.transitions[1] = 5  # Most frequent
        site.transitions[3] = 2  # Least frequent  
        site.transitions[2] = 3  # Middle
        
        result = site.most_frequent_transitions()
        self.assertEqual(result, [1, 2, 3])  # Ordered by frequency: 5, 3, 2
    
    def test_most_frequent_transitions_empty_transitions(self):
        """Test that most_frequent_transitions returns empty list when no transitions."""
        site = ConcreteSite()
        
        result = site.most_frequent_transitions()
        self.assertEqual(result, [])
        
    def test_average_occupation_empty_trajectory(self):
        """Test that average_occupation returns None when trajectory is empty."""
        site = ConcreteSite()
        self.assertIsNone(site.average_occupation)
    
    def test_average_occupation_all_empty_timesteps(self):
        """Test that average_occupation returns 0.0 when all timesteps are empty."""
        site = ConcreteSite()
        site.trajectory = [[], [], [], []]
        self.assertEqual(site.average_occupation, 0.0)
    
    def test_average_occupation_all_occupied_timesteps(self):
        """Test that average_occupation returns 1.0 when all timesteps are occupied."""
        site = ConcreteSite()
        site.trajectory = [[1], [2], [3], [4]]
        self.assertEqual(site.average_occupation, 1.0)
    
    def test_average_occupation_mixed_timesteps(self):
        """Test that average_occupation returns correct fraction for mixed occupation."""
        site = ConcreteSite()
        site.trajectory = [[1], [], [2], [], [3]]  # 3 occupied out of 5
        self.assertAlmostEqual(site.average_occupation, 0.6)
    
    def test_average_occupation_multiple_atoms_per_timestep(self):
        """Test that timesteps with multiple atoms still count as occupied."""
        site = ConcreteSite()
        site.trajectory = [[1, 2, 3], [], [4, 5], []]  # 2 occupied out of 4
        self.assertEqual(site.average_occupation, 0.5)
        
    def test_summary_default(self):
        """Test that summary returns default metrics."""
        site = ConcreteSite()
        site.index = 7
        site.trajectory = [[1]]
        site.transitions = Counter({2: 3, 5: 1})
        
        summary = site.summary()
        
        # Should include all default metrics
        self.assertEqual(set(summary.keys()), {'index', 'site_type', 'average_occupation', 'transitions'})
        self.assertEqual(summary['site_type'], 'ConcreteSite')
        self.assertEqual(summary['transitions'], {2: 3, 5: 1})
    
    def test_summary_includes_label_when_present(self):
        """Test that summary includes label when set."""
        site = ConcreteSite(label='test_site')
        site.trajectory = [[1]]
        
        summary = site.summary()
        
        self.assertIn('label', summary)
        self.assertEqual(summary['label'], 'test_site')
    
    def test_summary_with_specific_metrics(self):
        """Test that summary returns only requested metrics."""
        site = ConcreteSite()
        site.trajectory = [[1]]
        site.transitions = Counter({2: 3})
        
        summary = site.summary(metrics=['index', 'site_type'])
        
        self.assertEqual(set(summary.keys()), {'index', 'site_type'})
    
    def test_summary_exclude_transitions(self):
        """Test that summary can exclude transitions when requested."""
        site = ConcreteSite()
        site.trajectory = [[1]]
        site.transitions = Counter({2: 3, 5: 1})
        
        summary = site.summary(metrics=['index', 'average_occupation'])
        
        self.assertNotIn('transitions', summary)
        
    def test_summary_empty_metrics_list(self):
        """Test that summary returns empty dict for empty metrics list."""
        site = ConcreteSite()
        site.transitions = Counter({2: 3})
        
        summary = site.summary(metrics=[])
        
        self.assertEqual(summary, {})
    
    def test_summary_invalid_metric(self):
        """Test that summary raises ValueError for invalid metric names."""
        site = ConcreteSite()
        
        with self.assertRaises(ValueError) as context:
            site.summary(metrics=['invalid_metric'])
        
        self.assertIn('invalid_metric', str(context.exception))
    
    def test_summary_multiple_invalid_metrics(self):
        """Test that summary reports all invalid metrics."""
        site = ConcreteSite()
        
        with self.assertRaises(ValueError) as context:
            site.summary(metrics=['invalid1', 'index', 'invalid2'])
        
        # Should mention both invalid metrics
        self.assertIn('invalid1', str(context.exception))
        self.assertIn('invalid2', str(context.exception))
    
    def test_summary_excludes_none_from_defaults(self):
        """Test that summary excludes None values from default output."""
        site = ConcreteSite()
        site.trajectory = []  # Empty trajectory means average_occupation is None
        
        summary = site.summary()  # Using defaults
        
        # Should not include average_occupation when it's None in default mode
        self.assertNotIn('average_occupation', summary)
    
    def test_summary_includes_none_when_explicitly_requested(self):
        """Test that summary includes None values when explicitly requested."""
        site = ConcreteSite()
        site.trajectory = []  # Empty trajectory means average_occupation is None
        
        summary = site.summary(metrics=['index', 'average_occupation'])
        
        # Should include average_occupation even though it's None
        self.assertIn('average_occupation', summary)
        self.assertIsNone(summary['average_occupation'])

    # --- residence_times: core run extraction (include_edge_runs=True) ---

    def test_residence_times_empty_trajectory(self):
        """Test that residence_times returns empty tuple for empty trajectory."""
        site = ConcreteSite()
        self.assertEqual(site.residence_times(), ())

    def test_residence_times_never_occupied(self):
        """Test that residence_times returns empty tuple when site is never occupied."""
        site = ConcreteSite()
        site.trajectory = [[], [], []]
        self.assertEqual(site.residence_times(), ())

    def test_residence_times_single_continuous_run(self):
        """Test single atom occupying site for all timesteps."""
        site = ConcreteSite()
        site.trajectory = [[1], [1], [1]]
        self.assertEqual(site.residence_times(include_edge_runs=True), (3,))

    def test_residence_times_gap_in_middle(self):
        """Test single atom with a gap produces two separate runs."""
        site = ConcreteSite()
        site.trajectory = [[1], [], [1]]
        self.assertEqual(site.residence_times(include_edge_runs=True), (1, 1))

    def test_residence_times_gap_at_start(self):
        """Test that leading empty timesteps are not counted."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [1]]
        self.assertEqual(site.residence_times(include_edge_runs=True), (2,))

    def test_residence_times_gap_at_end(self):
        """Test that trailing empty timesteps are not counted."""
        site = ConcreteSite()
        site.trajectory = [[1], [1], []]
        self.assertEqual(site.residence_times(include_edge_runs=True), (2,))

    def test_residence_times_multiple_atoms_independent_runs(self):
        """Test multiple atoms produce independent run lengths."""
        site = ConcreteSite()
        site.trajectory = [[1], [1], [2], [2], [2]]
        result = site.residence_times(include_edge_runs=True)
        self.assertCountEqual(result, (2, 3))

    def test_residence_times_multiple_atoms_same_timestep(self):
        """Test atoms co-occupying a site produce correct per-atom runs."""
        site = ConcreteSite()
        site.trajectory = [[1, 2], [1, 2], [1]]
        result = site.residence_times(include_edge_runs=True)
        self.assertCountEqual(result, (3, 2))

    def test_residence_times_single_timestep(self):
        """Test single occupied timestep returns run of length 1."""
        site = ConcreteSite()
        site.trajectory = [[1]]
        self.assertEqual(site.residence_times(include_edge_runs=True), (1,))

    # --- residence_times: edge run exclusion (default) ---

    def test_residence_times_excludes_run_at_trajectory_start(self):
        """Test that a run touching the first timestep is excluded by default."""
        site = ConcreteSite()
        site.trajectory = [[1], [1], [], [1], [1], [1], []]
        self.assertEqual(site.residence_times(), (3,))

    def test_residence_times_excludes_run_at_trajectory_end(self):
        """Test that a run touching the last timestep is excluded by default."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [1], [1], [], [1], [1]]
        self.assertEqual(site.residence_times(), (3,))

    def test_residence_times_excludes_both_edge_runs(self):
        """Test that runs at both edges are excluded, keeping interior runs."""
        site = ConcreteSite()
        site.trajectory = [[1], [], [1], [1], [1], [], [1]]
        self.assertEqual(site.residence_times(), (3,))

    def test_residence_times_all_edge_runs_returns_empty(self):
        """Test that a single run spanning the whole trajectory is excluded."""
        site = ConcreteSite()
        site.trajectory = [[1], [1], [1]]
        self.assertEqual(site.residence_times(), ())

    def test_residence_times_only_interior_runs_kept(self):
        """Test that only fully interior runs are returned by default."""
        site = ConcreteSite()
        # atom 1: edge run at start (2), interior run (1), edge run at end (2)
        site.trajectory = [[1], [1], [], [1], [], [1], [1]]
        self.assertEqual(site.residence_times(), (1,))

    def test_residence_times_multiple_atoms_edge_exclusion(self):
        """Test edge exclusion applied independently per atom."""
        site = ConcreteSite()
        # atom 1: runs at t0-1 (edge), t4-5 (interior), t8-9 (edge)
        # atom 2: runs at t2-3 (interior), t6-7 (interior)
        site.trajectory = [
            [1], [1], [2], [2], [1], [1], [2], [2], [1], [1],
        ]
        result = site.residence_times()
        self.assertCountEqual(result, (2, 2, 2))

    # --- residence_times: filtering (interior gaps only) ---

    def test_residence_times_filter_fills_interior_gap(self):
        """Test that filter_length=1 fills a single-frame interior gap."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [1], []]
        self.assertEqual(
            site.residence_times(filter_length=1, include_edge_runs=True), (3,))

    def test_residence_times_filter_does_not_fill_longer_gap(self):
        """Test that filter_length=1 does not fill a two-frame gap."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [], [1], []]
        self.assertEqual(
            site.residence_times(filter_length=1, include_edge_runs=True), (1, 1))

    def test_residence_times_filter_length_2_fills_two_frame_gap(self):
        """Test that filter_length=2 fills a two-frame gap."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [], [1], []]
        self.assertEqual(
            site.residence_times(filter_length=2, include_edge_runs=True), (4,))

    def test_residence_times_filter_does_not_fill_edge_gap_at_start(self):
        """Test that filtering does not fill gaps at the trajectory start."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [1]]
        self.assertEqual(
            site.residence_times(filter_length=1, include_edge_runs=True), (2,))

    def test_residence_times_filter_does_not_fill_edge_gap_at_end(self):
        """Test that filtering does not fill gaps at the trajectory end."""
        site = ConcreteSite()
        site.trajectory = [[1], [1], []]
        self.assertEqual(
            site.residence_times(filter_length=1, include_edge_runs=True), (2,))

    def test_residence_times_filter_does_not_fill_gap_between_different_atoms(self):
        """Test that gaps between different atoms are not filled."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [2], []]
        self.assertEqual(
            site.residence_times(filter_length=1, include_edge_runs=True), (1, 1))

    def test_residence_times_filter_zero_same_as_default(self):
        """Test that filter_length=0 produces same result as default."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [1], []]
        self.assertEqual(
            site.residence_times(filter_length=0),
            site.residence_times(),
        )

    def test_residence_times_negative_filter_length_raises(self):
        """Test that negative filter_length raises ValueError."""
        site = ConcreteSite()
        site.trajectory = [[1], [], [1]]
        with self.assertRaises(ValueError):
            site.residence_times(filter_length=-1)

    def test_residence_times_non_integer_filter_length_raises(self):
        """Test that non-integer filter_length raises TypeError."""
        site = ConcreteSite()
        site.trajectory = [[1], [], [1]]
        with self.assertRaises(TypeError):
            site.residence_times(filter_length=1.5)

    def test_residence_times_filter_and_edge_exclusion_combined(self):
        """Test that filtering merges interior runs, then edge exclusion applies."""
        site = ConcreteSite()
        # atom 1: edge, gap(1), interior(2), gap(2), edge
        site.trajectory = [[1], [], [1], [1], [], [], [1]]
        # filter_length=1 fills the first gap -> merged edge run of 4, then gap(2), edge run
        # edge exclusion removes both -> empty
        self.assertEqual(site.residence_times(filter_length=1), ())

    def test_residence_times_filter_creates_interior_run(self):
        """Test that filtering can merge two interior runs into one."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [1], []]
        # Without filter: two interior runs of 1
        self.assertEqual(site.residence_times(), (1, 1))
        # With filter: one interior run of 3
        self.assertEqual(site.residence_times(filter_length=1), (3,))

    # --- residence_times: additional edge cases ---

    def test_residence_times_same_atom_revisits(self):
        """Test same atom visiting site in two separate interior runs."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [1], []]
        self.assertEqual(site.residence_times(), (1, 1))

    def test_residence_times_large_gap_not_filtered(self):
        """Test that a gap larger than filter_length is not filled."""
        site = ConcreteSite()
        site.trajectory = [[], [1], [], [], [], [1], []]
        self.assertEqual(site.residence_times(filter_length=2), (1, 1))

    def test_residence_times_multiple_gaps_partial_filter(self):
        """Test that only gaps within filter_length are filled."""
        site = ConcreteSite()
        # Gap of 1 (filled) then gap of 2 (not filled)
        site.trajectory = [[], [1], [], [1], [], [], [1], []]
        self.assertEqual(site.residence_times(filter_length=1), (3, 1))

    def test_residence_times_filter_with_both_edge_gaps(self):
        """Test filtering when both edges are gaps in a single atom's sequence."""
        site = ConcreteSite()
        site.trajectory = [[], [1], []]
        # Both gaps are at edges; neither should be filled
        self.assertEqual(
            site.residence_times(filter_length=1, include_edge_runs=True), (1,))

if __name__ == '__main__':
    unittest.main()
    
