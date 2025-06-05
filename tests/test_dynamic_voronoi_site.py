import unittest
from unittest.mock import patch
import numpy as np
from collections import Counter
from pymatgen.core import Structure, Lattice

from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site
from site_analysis.atom import Atom

class DynamicVoronoiSiteTestCase(unittest.TestCase):
    
    def setUp(self):
        Site._newid = 1
        
    def test_dynamic_voronoi_site_is_initialised(self):
        reference_indices = [1, 2, 3, 4]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        
        # Check that reference indices are stored correctly
        self.assertEqual(site.reference_indices, reference_indices)
        
        # Check that the centre is initially None (calculated on demand)
        self.assertEqual(site._centre_coords, None)
        
        # Check inherited attributes from Site
        self.assertEqual(site.index, 1)
        self.assertEqual(site.label, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.points, [])
        self.assertEqual(site.transitions, Counter())
        
    def test_dynamic_voronoi_site_is_initialised_with_label(self):
        """Test that a DynamicVoronoiSite can be initialised with a label."""
        reference_indices = [1, 2, 3, 4]
        label = "tetrahedral_site"
        site = DynamicVoronoiSite(reference_indices=reference_indices, label=label)
        
        self.assertEqual(site.reference_indices, reference_indices)
        self.assertEqual(site.label, label)
        
    def test_reset(self):
        """Test that reset clears the calculated centre and site occupation data."""
        reference_indices = [1, 2, 3, 4]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        
        # Mock some data that should be cleared on reset
        site._centre_coords = np.array([0.5, 0.5, 0.5])
        site.contains_atoms = [10, 11]
        site.trajectory = [[10], [10, 11]]
        site.transitions = Counter([3, 3, 2])
        
        # Reset the site
        site.reset()
        
        # Check that all appropriate attributes are reset
        self.assertEqual(site._centre_coords, None)
        self.assertEqual(site.contains_atoms, [])
        self.assertEqual(site.trajectory, [])
        self.assertEqual(site.transitions, Counter())
        
    def test_calculate_centre_no_wrapping(self):
        """Test centre calculation when result doesn't need wrapping."""
        reference_indices = [0, 1, 2, 3]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        
        lattice = Lattice.cubic(10.0)
        coords = [[0.1, 0.1, 0.1],
                  [0.2, 0.1, 0.1],
                  [0.1, 0.2, 0.1],
                  [0.1, 0.1, 0.2]]
        structure = Structure(lattice, ["Li", "Li", "Li", "Li"], coords)
        
        with patch('site_analysis.dynamic_voronoi_site.apply_legacy_pbc_correction') as mock_pbc:
            # Mock returns coordinates that produce centre < 1.0
            corrected_coords = np.array([[0.1, 0.1, 0.1],
                                         [0.2, 0.1, 0.1],
                                         [0.1, 0.2, 0.1],
                                         [0.1, 0.1, 0.2]])
            mock_pbc.return_value = corrected_coords
            
            site.calculate_centre(structure)
            
            mock_pbc.assert_called_once()
            np.testing.assert_array_almost_equal(mock_pbc.call_args[0][0], coords)
            
            # Expected centre: mean = [0.125, 0.125, 0.125], no wrapping needed
            expected_centre = np.mean(corrected_coords, axis=0)
            np.testing.assert_array_almost_equal(site._centre_coords, expected_centre)
    
    def test_calculate_centre_with_wrapping(self):
        """Test centre calculation when result needs wrapping."""
        reference_indices = [0, 1, 2, 3]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        
        lattice = Lattice.cubic(10.0)
        coords = [[0.1, 0.1, 0.1],
                  [0.2, 0.1, 0.1],
                  [0.1, 0.2, 0.1],
                  [0.1, 0.1, 0.2]]
        structure = Structure(lattice, ["Li", "Li", "Li", "Li"], coords)
        
        with patch('site_analysis.dynamic_voronoi_site.apply_legacy_pbc_correction') as mock_pbc:
            # Mock returns coordinates that produce centre > 1.0 to test wrapping
            corrected_coords = np.array([[1.1, 1.1, 1.1],
                                         [1.2, 1.1, 1.1],
                                         [1.1, 1.2, 1.1],
                                         [1.1, 1.1, 1.2]])
            mock_pbc.return_value = corrected_coords
            
            site.calculate_centre(structure)
            
            mock_pbc.assert_called_once()
            np.testing.assert_array_almost_equal(mock_pbc.call_args[0][0], coords)
            
            # Expected centre with wrapping: mean([1.1, 1.2, 1.1, 1.1]) = 1.125 -> 0.125
            centre_before_wrap = np.mean(corrected_coords, axis=0)
            expected_centre = centre_before_wrap % 1.0
            np.testing.assert_array_almost_equal(site._centre_coords, expected_centre)
        
    def test_centre_method(self):
        """Test that the centre method returns the calculated centre coordinates."""
        reference_indices = [0, 1, 2, 3]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        
        # Set a predefined centre
        centre_coords = np.array([0.5, 0.5, 0.5])
        site._centre_coords = centre_coords
        
        # Check that centre returns the expected coordinates
        np.testing.assert_array_equal(site.centre, centre_coords)
        
        # Check that centre raises an error if the centre has not been calculated yet
        site._centre_coords = None
        with self.assertRaises(RuntimeError):
            site.centre
            
    def test_update_centre_when_reference_atoms_move(self):
        """Test that the centre updates when reference atoms move."""
        reference_indices = [0, 1, 2, 3]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        
        # Create initial structure
        lattice = Lattice.cubic(10.0)
        coords1 = [[0.1, 0.1, 0.1],
                [0.2, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.2]]
        structure1 = Structure(lattice, ["Li", "Li", "Li", "Li"], coords1)
        
        # Calculate initial centre
        site.calculate_centre(structure1)
        initial_centre = site._centre_coords.copy()
        
        # Create a new structure with moved atoms
        coords2 = [[0.2, 0.2, 0.2],
                [0.3, 0.2, 0.2],
                [0.2, 0.3, 0.2],
                [0.2, 0.2, 0.3]]
        structure2 = Structure(lattice, ["Li", "Li", "Li", "Li"], coords2)
        
        # Calculate new centre
        site.calculate_centre(structure2)
        new_centre = site._centre_coords
        
        # Check that the centre has changed
        self.assertFalse(np.array_equal(initial_centre, new_centre))
        
        # Expected new centre
        expected_new_centre = np.mean(np.array(coords2), axis=0)
        np.testing.assert_array_almost_equal(new_centre, expected_new_centre)
        
    def test_as_dict_and_from_dict(self):
        """Test serialization and deserialization of DynamicVoronoiSite."""
        reference_indices = [1, 2, 3, 4]
        label = "tetrahedral_site"
        site = DynamicVoronoiSite(reference_indices=reference_indices, label=label)
        
        # Set some attributes for testing
        site._centre_coords = np.array([0.5, 0.5, 0.5])
        site.contains_atoms = [10, 11]
        site.points = [np.array([0.6, 0.6, 0.6])]
        
        # Convert to dict
        site_dict = site.as_dict()
        
        # Check that dict contains the expected keys
        self.assertIn('reference_indices', site_dict)
        self.assertIn('label', site_dict)
        self.assertIn('centre_coords', site_dict)
        
        # Create a new site from the dict
        new_site = DynamicVoronoiSite.from_dict(site_dict)
        
        # Check that the new site has the expected attributes
        self.assertEqual(new_site.reference_indices, reference_indices)
        self.assertEqual(new_site.label, label)
        np.testing.assert_array_equal(new_site._centre_coords, site._centre_coords)
        self.assertEqual(new_site.contains_atoms, site.contains_atoms)
        for i, point in enumerate(new_site.points):
            np.testing.assert_array_equal(point, site.points[i])
            
    def test_coordination_number(self):
        """Test that coordination_number returns the number of reference atoms."""
        # Test with 4 reference atoms
        reference_indices = [1, 2, 3, 4]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        self.assertEqual(site.coordination_number, 4)
        
        # Test with a different number of reference atoms
        reference_indices = [1, 2, 3, 4, 5, 6]
        site = DynamicVoronoiSite(reference_indices=reference_indices)
        self.assertEqual(site.coordination_number, 6)
        
        # Test the cn property (convenience alias for coordination_number)
        self.assertEqual(site.cn, 6)

    def test_sites_from_reference_indices(self):
        """Test that sites_from_reference_indices correctly creates sites from reference indices."""
        reference_indices_list = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        label = "tetrahedral_site"
        
        # Create sites using the class method
        sites = DynamicVoronoiSite.sites_from_reference_indices(reference_indices_list, label=label)
        
        # Check that the correct number of sites were created
        self.assertEqual(len(sites), len(reference_indices_list))
        
        # Check that each site has the correct reference indices and label
        for site, ref_indices in zip(sites, reference_indices_list):
            self.assertEqual(site.reference_indices, ref_indices)
            self.assertEqual(site.label, label)
        
    def test_sites_from_reference_indices_without_label(self):
        """Test that sites_from_reference_indices works correctly without a label."""
        reference_indices_list = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        
        # Create sites using the class method
        sites = DynamicVoronoiSite.sites_from_reference_indices(reference_indices_list)
        
        # Check that the correct number of sites were created
        self.assertEqual(len(sites), len(reference_indices_list))
        
        # Check that each site has the correct reference indices and no label
        for site, ref_indices in zip(sites, reference_indices_list):
            self.assertEqual(site.reference_indices, ref_indices)
            self.assertEqual(site.label, None)
    
    def test_sites_from_reference_indices_empty_list(self):
        """Test that sites_from_reference_indices returns an empty list when given an empty list."""
        reference_indices_list = []
        
        # Create sites using the class method
        sites = DynamicVoronoiSite.sites_from_reference_indices(reference_indices_list)
        
        # Check that no sites were created
        self.assertEqual(len(sites), 0)

class DynamicVoronoiSiteCentreSerialisationTestCase(unittest.TestCase):
    """Test centre coordinate serialisation for DynamicVoronoiSite."""

    def setUp(self):
        Site._newid = 0

    def test_as_dict_excludes_centre_when_not_calculated(self):
        """Test as_dict excludes centre_coords when centre not calculated."""
        site = DynamicVoronoiSite(reference_indices=[1, 2, 3])
        # _centre_coords remains None

        site_dict = site.as_dict()

        self.assertNotIn('centre_coords', site_dict)

    def test_as_dict_includes_centre_when_calculated(self):
        """Test as_dict includes centre_coords when centre calculated."""
        site = DynamicVoronoiSite(reference_indices=[1, 2, 3])
        site._centre_coords = np.array([0.25, 0.5, 0.75])

        site_dict = site.as_dict()

        self.assertIn('centre_coords', site_dict)
        self.assertEqual(site_dict['centre_coords'], [0.25, 0.5, 0.75])

    def test_from_dict_restores_centre_when_present(self):
        """Test from_dict restores centre coordinates when present in dict."""
        site_dict = {
            'reference_indices': [4, 5, 6],
            'centre_coords': [0.1, 0.2, 0.3]
        }

        site = DynamicVoronoiSite.from_dict(site_dict)

        np.testing.assert_array_equal(site._centre_coords, [0.1, 0.2, 0.3])

    def test_from_dict_leaves_centre_none_when_absent(self):
        """Test from_dict leaves centre as None when not in dict."""
        site_dict = {
            'reference_indices': [4, 5, 6]
        }

        site = DynamicVoronoiSite.from_dict(site_dict)

        self.assertIsNone(site._centre_coords)

    def test_centre_property_works_after_deserialisation(self):
        """Test centre property works correctly after from_dict."""
        site_dict = {
            'reference_indices': [1, 2],
            'centre_coords': [0.4, 0.5, 0.6]
        }

        site = DynamicVoronoiSite.from_dict(site_dict)

        np.testing.assert_array_equal(site.centre, [0.4, 0.5, 0.6])

    def test_centre_property_raises_error_when_not_calculated_after_deserialisation(self):
        """Test centre property raises error when centre not calculated after deserialisation."""
        site_dict = {
            'reference_indices': [1, 2]
            # No centre_coords
        }

        site = DynamicVoronoiSite.from_dict(site_dict)

        with self.assertRaises(RuntimeError):
            _ = site.centre
            
    def test_dynamic_voronoi_site_init_with_reference_center(self):
        """Test DynamicVoronoiSite can be initialised with a reference centre."""
        reference_indices = [0, 1, 2, 3]
        reference_center = np.array([0.25, 0.5, 0.75])
        site = DynamicVoronoiSite(reference_indices=reference_indices, reference_center=reference_center)
        np.testing.assert_array_equal(site.reference_center, reference_center)
        self.assertEqual(site.reference_indices, reference_indices)
        
    def test_calculate_centre_uses_legacy_when_no_reference_center(self):
        """Test that legacy PBC correction is used when reference_center is None."""
        reference_indices = [0, 1, 2, 3]
        site = DynamicVoronoiSite(reference_indices=reference_indices)  # No reference_center
        
        lattice = Lattice.cubic(10.0)
        coords = [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.2]]
        structure = Structure(lattice, ["Li", "Li", "Li", "Li"], coords)
        
        with patch('site_analysis.dynamic_voronoi_site.apply_legacy_pbc_correction') as mock_legacy:
            mock_legacy.return_value = np.array(coords)
            
            site.calculate_centre(structure)
            
            mock_legacy.assert_called_once()
    
    def test_calculate_centre_uses_reference_center_when_provided(self):
        """Test that reference centre-based unwrapping is used when reference_center is provided."""
        reference_indices = [0, 1, 2, 3]
        reference_center = np.array([0.5, 0.5, 0.5])
        site = DynamicVoronoiSite(reference_indices=reference_indices, reference_center=reference_center)
        
        lattice = Lattice.cubic(10.0)
        coords = [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.2]]
        structure = Structure(lattice, ["Li", "Li", "Li", "Li"], coords)
        
        with patch('site_analysis.dynamic_voronoi_site.unwrap_vertices_to_reference_center') as mock_unwrap:
            mock_unwrap.return_value = np.array(coords)
            
            site.calculate_centre(structure)
            
            mock_unwrap.assert_called_once()
            
if __name__ == '__main__':
    unittest.main()
