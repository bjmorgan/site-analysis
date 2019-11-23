import unittest
from site_analysis.site import Site
from site_analysis.atom import Atom
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from collections import Counter

class SiteTestCase(unittest.TestCase):

    def setUp(self):
        Site._newid = 1

    def test_site_is_initialised(self):
        site = Site()
        self.assertEqual(site.index, 1)
        self.assertEqual(site.label, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.points, [])
        self.assertEqual(site.transitions, {})

    def test_site_is_initialised_with_label(self):
        site = Site(label='foo')
        self.assertEqual(site.label, 'foo')

    def test_site_index_autoincrements(self):
        site1 = Site()
        site2 = Site()
        self.assertEqual(site2.index, site1.index + 1)

    def test_reset(self):
        site = Site()
        site.contains_atoms = ['foo']
        site.trajectory = ['bar']
        site.transitions = Counter([4])
        site.reset()
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.transitions, {})

    def test_contains_point_raises_not_implemented_error(self):
        site = Site()
        with self.assertRaises(NotImplementedError):
            site.contains_point(np.array([0,0,0]))

    def test_centre_raises_not_implemented_error(self):
        site = Site()
        with self.assertRaises(NotImplementedError):
            site.centre()

    def test_contains_atom(self):
        site = Site()
        atom = Mock(spec=Atom)
        atom.frac_coords = np.array([0,0,0])
        site.contains_point = Mock(return_value=True)
        self.assertTrue(site.contains_atom(atom))
        site.contains_point = Mock(return_value=False)
        self.assertFalse(site.contains_atom(atom))

    def test_as_dict(self):
        site = Site()
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
        self.assertEqual(site_dict['index'], expected_dict['index'])
        self.assertEqual(site_dict['contains_atoms'], expected_dict['contains_atoms'])
        self.assertEqual(site_dict['trajectory'], expected_dict['trajectory'])
        np.testing.assert_array_equal(site_dict['points'], expected_dict['points'])
        self.assertEqual(site_dict['label'], expected_dict['label'])
        self.assertEqual(site_dict['transitions'], expected_dict['transitions'])

if __name__ == '__main__':
    unittest.main()
    
