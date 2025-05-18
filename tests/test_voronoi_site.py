import unittest
from site_analysis.voronoi_site import VoronoiSite
from unittest.mock import patch, MagicMock, Mock
import numpy as np

class VoronoiSiteTestCase(unittest.TestCase):

    def test_voronoi_site_is_initialised(self):
        frac_coords = np.array([0.1, 0.2, 0.3])
        site = VoronoiSite(frac_coords=frac_coords)
        np.testing.assert_array_equal(site.frac_coords, frac_coords)
        self.assertEqual(site.label, None)

    def test_voronoi_site_is_initialised_with_label(self):
        frac_coords = np.array([0.1, 0.2, 0.3])
        site = VoronoiSite(frac_coords=frac_coords, label='foo')
        np.testing.assert_array_equal(site.frac_coords, frac_coords)
        self.assertEqual(site.label, 'foo')

    def test_as_dict(self):
        frac_coords = np.array([0.1, 0.2, 0.3])
        site = VoronoiSite(frac_coords=frac_coords, label='foo')
        site_dict = site.as_dict()
        np.testing.assert_array_equal(site_dict['frac_coords'], frac_coords)
        self.assertEqual(site_dict['label'], 'foo')
   
    def test_from_dict(self):
        site_dict = {'frac_coords': np.array([0.1, 0.2, 0.3]),
                     'label': 'foo'}
        with patch('site_analysis.site.Site.from_dict') as mock_from_dict:
            mock_from_dict.return_value = {}
            site = VoronoiSite.from_dict(site_dict)
            self.assertEqual(site.label, 'foo')
            np.testing.assert_array_equal(site.frac_coords, site_dict['frac_coords'])
            
    def test_contains_point_raises_not_implemented_error(self):
        site = VoronoiSite(frac_coords=np.array([0.1, 0.1, 0.1]))
        with self.assertRaises(NotImplementedError):
            site.contains_point(np.array([0.5, 0.5, 0.5]))
            
    def test_repr(self):
        """Test the __repr__ method for VoronoiSite."""
        frac_coords = np.array([0.1, 0.2, 0.3])
        label = "voronoi_site"
        site = VoronoiSite(frac_coords=frac_coords, label=label)
        site.index = 3  # Set index explicitly for testing
        site.contains_atoms = [4, 5]  # Add some atoms for testing
        
        # Generate the string representation
        repr_str = repr(site)
        
        # Verify all key attributes are included
        self.assertIn("site_analysis.VoronoiSite", repr_str)
        self.assertIn(f"index={site.index}", repr_str)
        self.assertIn(f"label={site.label}", repr_str)
        self.assertIn(f"contains_atoms={site.contains_atoms}", repr_str)
        # Don't check for exact frac_coords representation as it varies
        self.assertIn("frac_coords=", repr_str)

if __name__ == '__main__':
    unittest.main()
    
