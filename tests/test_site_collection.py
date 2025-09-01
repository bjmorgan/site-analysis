import unittest
from site_analysis.site_collection import SiteCollection
from site_analysis.site import Site
from site_analysis.atom import Atom
from pymatgen.core import Structure
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

    def test_assign_site_occupations_raises_not_implemented_error(self):
        sites = [Mock(spec=Site, index=0),
                 Mock(spec=Site, index=1)]
        atoms = [Mock(spec=Atom), 
                 Mock(spec=Atom)]
        structure = Mock(spec=Structure)
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.assign_site_occupations(atoms, structure)

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

    def test_update_occupation_if_atom_has_not_moved(self):
        sites = [Mock(spec=Site)]
        sites[0].index = 12
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.trajectory = [12]
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].contains_atoms, [atom.index])
        np.testing.assert_array_equal(sites[0].points, [atom.frac_coords])
        self.assertEqual(atom.in_site, sites[0].index)

    def test_update_occupation_if_atom_has_moved(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        sites[1].transitions = Counter()
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        site_collection.site_by_index = Mock(return_value=sites[1])
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.trajectory = [42]
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].contains_atoms, [atom.index])
        np.testing.assert_array_equal(sites[0].points, [atom.frac_coords])
        self.assertEqual(atom.in_site, sites[0].index)
        self.assertEqual(sites[1].transitions, {12: 1})
  
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
            site_collection.sites_contain_points(points=points)  
            
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

if __name__ == '__main__':
    unittest.main()
    
