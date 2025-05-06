import unittest
from pymatgen.core import Lattice, Structure
from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection
from site_analysis.polyhedral_site import PolyhedralSite
from unittest.mock import patch, Mock

class PolyhedralSiteCollectionTestCase(unittest.TestCase):

    def test_site_collection_is_initialised(self):
        sites = [Mock(spec=PolyhedralSite), Mock(spec=PolyhedralSite)]
        with patch('site_analysis.polyhedral_site_collection'
                   '.construct_neighbouring_sites') as mock_construct_neighbouring_sites:
            mock_construct_neighbouring_sites.return_value = 'foo'
            site_collection = PolyhedralSiteCollection(sites=sites)
            self.assertEqual(site_collection.sites, sites)
            mock_construct_neighbouring_sites.assert_called_with(site_collection.sites)
            self.assertEqual(site_collection._neighbouring_sites, 'foo')
            
    def test_empty_atoms_list_polyhedral(self):
        """Test PolyhedralSiteCollection handles empty atom lists correctly."""
        site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        site1.contains_atoms = [1, 2]
        
        site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7])
        site2.contains_atoms = [3, 4]
        
        # Create the collection
        collection = PolyhedralSiteCollection(sites=[site1, site2])
        
        # Create a structure to pass to the method
        lattice = Lattice.cubic(10.0)
        structure = Structure(
            lattice=lattice,
            species=["Na"] * 8,  # Need 8 atoms for the vertex indices
            coords=[
                [0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3],
                [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]
            ]
        )
        
        # Call the method with empty atom list
        collection.assign_site_occupations([], structure)
        
        # Verify that contains_atoms was reset for both sites
        self.assertEqual(site1.contains_atoms, [])
        self.assertEqual(site2.contains_atoms, [])
       
if __name__ == '__main__':
    unittest.main()
    
