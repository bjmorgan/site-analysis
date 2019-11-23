import unittest
from site_analysis.site import Site
from unittest.mock import patch, MagicMock, Mock
import numpy as np

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
        site.reset()
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.contains_atoms, [])

if __name__ == '__main__':
    unittest.main()
    
