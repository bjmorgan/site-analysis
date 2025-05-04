"""Unit tests for the ReferenceBasedSites class.

These tests verify that ReferenceBasedSites correctly orchestrates the workflow
for defining sites using a reference structure approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pymatgen.core import Structure, Lattice

# Import the class (assuming it will be defined in this module)
from site_analysis.reference_workflow.reference_based_sites import ReferenceBasedSites


class TestReferenceBasedSites(unittest.TestCase):
    """Test cases for the ReferenceBasedSites class."""
    
    def setUp(self):
        """Set up test structures and mocks."""
        # Create simple test structures
        lattice = Lattice.cubic(5.0)
        species1 = ["Na", "Cl", "Na", "Cl"]
        coords1 = [
            [0.0, 0.0, 0.0],  # Na
            [0.5, 0.5, 0.5],  # Cl
            [0.5, 0.0, 0.0],  # Na
            [0.0, 0.5, 0.5],  # Cl
        ]
        self.reference = Structure(lattice, species1, coords1)
        
        # Target structure (slightly shifted)
        species2 = ["Na", "Cl", "Na", "Cl"]
        coords2 = [
            [0.1, 0.1, 0.1],  # Na
            [0.6, 0.6, 0.6],  # Cl
            [0.6, 0.1, 0.1],  # Na
            [0.1, 0.6, 0.6],  # Cl
        ]
        self.target = Structure(lattice, species2, coords2)
        
        # Sample coordination environments
        self.ref_environments = [[1, 3]]  # One environment with Cl atoms at indices 1 and 3
        self.mapped_environments = [[1, 3]]  # Same indices after mapping (for simplicity in tests)
        
        # Set up mock objects
        self.mock_structure_aligner = self._setup_mock_structure_aligner()
        self.mock_coord_finder = self._setup_mock_coord_finder()
        self.mock_index_mapper = self._setup_mock_index_mapper()
        self.mock_site_factory = self._setup_mock_site_factory()
        
        # Set up patch objects
        self.structure_aligner_patch = patch(
            'site_analysis.reference_workflow.reference_based_sites.StructureAligner',
            return_value=self.mock_structure_aligner
        )
        self.coord_finder_patch = patch(
            'site_analysis.reference_workflow.reference_based_sites.CoordinationEnvironmentFinder',
            return_value=self.mock_coord_finder
        )
        self.index_mapper_patch = patch(
            'site_analysis.reference_workflow.reference_based_sites.IndexMapper',
            return_value=self.mock_index_mapper
        )
        self.site_factory_patch = patch(
            'site_analysis.reference_workflow.reference_based_sites.SiteFactory',
            return_value=self.mock_site_factory
        )
    
    def _setup_mock_structure_aligner(self):
        """Set up a mock StructureAligner."""
        mock = Mock()
        
        # Mock align method
        aligned_structure = self.target.copy()
        translation_vector = np.array([0.1, 0.1, 0.1])
        metrics = {'rmsd': 0.1, 'max_dist': 0.2, 'mean_dist': 0.15}
        mock.align.return_value = (aligned_structure, translation_vector, metrics)
        
        return mock
    
    def _setup_mock_coord_finder(self):
        """Set up a mock CoordinationEnvironmentFinder."""
        mock = Mock()
        
        # Mock find_environments method
        mock.find_environments.return_value = {0: [1, 3]}  # Na at index 0 coordinated by Cl at indices 1 and 3
        
        return mock
    
    def _setup_mock_index_mapper(self):
        """Set up a mock IndexMapper."""
        mock = Mock()
        
        # Mock map_coordinating_atoms method
        mock.map_coordinating_atoms.return_value = self.mapped_environments
        
        return mock
    
    def _setup_mock_site_factory(self):
        """Set up a mock SiteFactory."""
        mock = Mock()
        
        # Mock create methods
        mock_polyhedral_sites = [Mock(name='PolyhedralSite')]
        mock_dynamic_voronoi_sites = [Mock(name='DynamicVoronoiSite')]
        
        mock.create_polyhedral_sites.return_value = mock_polyhedral_sites
        mock.create_dynamic_voronoi_sites.return_value = mock_dynamic_voronoi_sites
        
        return mock
    
    def test_init_with_alignment(self):
        """Test initialization with structure alignment."""
        with self.structure_aligner_patch:
            # initialise with alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=True)
            
            # Check attributes
            self.assertEqual(rbs.reference_structure, self.reference)
            self.assertEqual(rbs.target_structure, self.target)
            self.assertEqual(rbs.aligned_structure, self.target)  # Mock returns target as aligned
            np.testing.assert_array_equal(rbs.translation_vector, np.array([0.1, 0.1, 0.1]))
            self.assertEqual(rbs.alignment_metrics, {'rmsd': 0.1, 'max_dist': 0.2, 'mean_dist': 0.15})
    
    def test_init_without_alignment(self):
        """Test initialization without structure alignment."""
        with self.structure_aligner_patch:
            # initialise without alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Check attributes
            self.assertEqual(rbs.reference_structure, self.reference)
            self.assertEqual(rbs.target_structure, self.target)
            self.assertEqual(rbs.aligned_structure, self.target)  # Without alignment, target is used directly
            self.assertIsNone(rbs.translation_vector)
            self.assertIsNone(rbs.alignment_metrics)
    
    def test_align_structures(self):
        """Test the _align_structures method."""
        with self.structure_aligner_patch:
            # initialise without alignment initially
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Now perform alignment
            rbs._align_structures(align_species=['Na'], align_metric='max_dist')
            
            # Check that align was called with correct parameters
            self.mock_structure_aligner.align.assert_called_with(
                self.reference, self.target, species=['Na'], metric='max_dist'
            )
            
            # Check that attributes were updated
            self.assertEqual(rbs.aligned_structure, self.target)  # Mock returns target as aligned
            np.testing.assert_array_equal(rbs.translation_vector, np.array([0.1, 0.1, 0.1]))
            self.assertEqual(rbs.alignment_metrics, {'rmsd': 0.1, 'max_dist': 0.2, 'mean_dist': 0.15})
    
    def test_find_coordination_environments(self):
        """Test the _find_coordination_environments method."""
        with self.coord_finder_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Find coordination environments
            environments = rbs._find_coordination_environments(
                center_species='Na',
                vertex_species='Cl',
                cutoff=3.0,
                n_vertices=2
            )
            
            # Check that find_environments was called with correct parameters
            self.mock_coord_finder.find_environments.assert_called_with(
                center_species='Na',
                vertex_species='Cl',
                n_vertices=2,
                cutoff=3.0
            )
            
            # Check returned environments (converted from dict to list of lists)
            self.assertEqual(environments, self.ref_environments)
    
    def test_map_environments(self):
        """Test the _map_environments method."""
        with self.index_mapper_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Map environments
            mapped = rbs._map_environments(
                ref_environments=self.ref_environments,
                target_species='Cl'
            )
            
            # Check that map_coordinating_atoms was called with correct parameters
            self.mock_index_mapper.map_coordinating_atoms.assert_called_with(
                self.reference,
                self.target,  # Should use target directly since no alignment
                self.ref_environments,
                target_species='Cl'
            )
            
            # Check returned environments
            self.assertEqual(mapped, self.mapped_environments)
    
    def test_map_environments_with_aligned_structure(self):
        """Test the _map_environments method with an aligned structure."""
        with self.structure_aligner_patch, self.index_mapper_patch:
            # initialise with alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=True)
            
            # Map environments
            mapped = rbs._map_environments(
                ref_environments=self.ref_environments,
                target_species='Cl'
            )
            
            # Check that map_coordinating_atoms was called with correct parameters
            # Should use aligned_structure instead of target_structure
            self.mock_index_mapper.map_coordinating_atoms.assert_called_with(
                self.reference,
                self.target,  # Mock aligned structure is same as target in our setup
                self.ref_environments,
                target_species='Cl'
            )
            
            # Check returned environments
            self.assertEqual(mapped, self.mapped_environments)
    
    def test_create_polyhedral_sites(self):
        """Test the create_polyhedral_sites method."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Create polyhedral sites
            sites = rbs.create_polyhedral_sites(
                center_species='Na',
                vertex_species='Cl',
                cutoff=3.0,
                n_vertices=2,
                label='test_label',
                target_species='Cl'
            )
            
            # Check that find_coordination_environments was called
            self.mock_coord_finder.find_environments.assert_called_with(
                center_species='Na',
                vertex_species='Cl',
                n_vertices=2,
                cutoff=3.0
            )
            
            # Check that map_coordinating_atoms was called
            self.mock_index_mapper.map_coordinating_atoms.assert_called_with(
                self.reference,
                self.target,
                self.ref_environments,
                target_species='Cl'
            )
            
            # Check that create_polyhedral_sites was called
            self.mock_site_factory.create_polyhedral_sites.assert_called_with(
                self.mapped_environments,
                label='test_label',
                labels=None
            )
            
            # Check returned sites
            self.assertEqual(sites, self.mock_site_factory.create_polyhedral_sites.return_value)
    
    def test_create_dynamic_voronoi_sites(self):
        """Test the create_dynamic_voronoi_sites method."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Create dynamic voronoi sites
            sites = rbs.create_dynamic_voronoi_sites(
                center_species='Na',
                reference_species='Cl',
                cutoff=3.0,
                n_reference=2,
                labels=['site1'],
                target_species='Cl'
            )
            
            # Check that find_coordination_environments was called
            self.mock_coord_finder.find_environments.assert_called_with(
                center_species='Na',
                vertex_species='Cl',  # vertex_species is used internally
                n_vertices=2,  # n_vertices is used internally
                cutoff=3.0
            )
            
            # Check that map_coordinating_atoms was called
            self.mock_index_mapper.map_coordinating_atoms.assert_called_with(
                self.reference,
                self.target,
                self.ref_environments,
                target_species='Cl'
            )
            
            # Check that create_dynamic_voronoi_sites was called
            self.mock_site_factory.create_dynamic_voronoi_sites.assert_called_with(
                self.mapped_environments,
                label=None,
                labels=['site1']
            )
            
            # Check returned sites
            self.assertEqual(sites, self.mock_site_factory.create_dynamic_voronoi_sites.return_value)
    
    def test_error_handling_in_find_coordination_environments(self):
        """Test error handling in _find_coordination_environments method."""
        with self.coord_finder_patch:
            # Make find_environments raise an error
            self.mock_coord_finder.find_environments.side_effect = ValueError("Species not found")
            
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Check that the error is propagated with additional context
            with self.assertRaises(ValueError) as context:
                rbs._find_coordination_environments(
                    center_species='Mg',  # Not in the structure
                    vertex_species='Cl',
                    cutoff=3.0,
                    n_vertices=6
                )
            
            # Check error message
            self.assertIn("Species not found", str(context.exception))
    
    def test_error_handling_in_map_environments(self):
        """Test error handling in _map_environments method."""
        with self.index_mapper_patch:
            # Make map_coordinating_atoms raise an error
            self.mock_index_mapper.map_coordinating_atoms.side_effect = ValueError("Mapping violation")
            
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Check that the error is propagated with additional context
            with self.assertRaises(ValueError) as context:
                rbs._map_environments(
                    ref_environments=self.ref_environments,
                    target_species='O'  # Not in the structure
                )
            
            # Check error message
            self.assertIn("Mapping violation", str(context.exception))
    
    def test_site_factory_initialisation(self):
        """Test that SiteFactory is initialised with the correct structure."""
        with patch('site_analysis.reference_workflow.reference_based_sites.SiteFactory') as mock_factory:
            # Create the ReferenceBasedSites instance
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Make sure _site_factory is None
            rbs._site_factory = None
            
            # Now call our initialization method
            rbs._initialise_site_factory()
            
            # This will verify the mock was called with the target structure
            mock_factory.assert_called_once_with(self.target)


if __name__ == '__main__':
    unittest.main()
