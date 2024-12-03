import unittest
from typing import List, Dict
from unittest.mock import patch, Mock
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection

class PolyhedralSiteCollectionTestCase(unittest.TestCase):

    def setUp(self):
        self.mock_site1 = Mock(spec=PolyhedralSite)
        self.mock_site2 = Mock(spec=PolyhedralSite)
        self.mock_site1.cn = 4
        self.mock_site2.cn = 4
        self.sites = [self.mock_site1, self.mock_site2]

    def test_site_collection_initialization(self):
        """Test that PolyhedralSiteCollection is properly initialized"""
        site_collection = PolyhedralSiteCollection(sites=self.sites)

        # Test that sites are properly stored
        self.assertEqual(site_collection.sites, self.sites)

        # Test that _neighbouring_sites starts as None
        self.assertIsNone(site_collection._neighbouring_sites)

    def test_neighbouring_sites_lazy_loading(self):
        """Test that neighbouring_sites is lazily loaded"""
        site_collection = PolyhedralSiteCollection(sites=self.sites)

        with patch.object(site_collection, '_construct_neighbouring_sites') as mock_construct:
            mock_construct.return_value = {'mock': 'neighbours'}

            # First access should trigger construction
            neighbours = site_collection.neighbouring_sites
            mock_construct.assert_called_once_with(site_collection.sites)

            # Second access should use cached value
            neighbours = site_collection.neighbouring_sites
            mock_construct.assert_called_once()  # Should still only be called once

            self.assertEqual(neighbours, {'mock': 'neighbours'})

    def test_face_sharing_neighbours_computation(self):
        """Test the actual computation of face-sharing neighbours"""
        
        self.mock_site1.index = 0
        self.mock_site1.vertex_indices = [0, 1, 2, 3]
        self.mock_site2.index = 1
        self.mock_site2.vertex_indices = [1, 2, 3, 4]

        site_collection = PolyhedralSiteCollection(sites=self.sites)
        neighbours = site_collection.neighbouring_sites

        # Sites share vertices [1, 2, 3], so they should be neighbours
        self.assertEqual(neighbours[0], [self.mock_site2])
        self.assertEqual(neighbours[1], [self.mock_site1])

if __name__ == '__main__':
    unittest.main()