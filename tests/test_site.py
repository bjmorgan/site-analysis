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

if __name__ == '__main__':
    unittest.main()
    
