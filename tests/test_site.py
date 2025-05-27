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

if __name__ == '__main__':
    unittest.main()
    
