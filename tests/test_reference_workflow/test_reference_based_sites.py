"""Unit tests for the ReferenceBasedSites class.

These tests verify that ReferenceBasedSites correctly orchestrates the workflow
for defining sites using a reference structure approach.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pymatgen.core import Structure, Lattice
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
        self.ref_environments = {0: [1, 3]} # One environment with Cl atoms at indices 1 and 3
        self.ref_environments_list = [[1, 3]] 
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
        aligned_reference = Mock(spec=Structure)
        translation_vector = np.array([0.1, 0.1, 0.1])
        metrics = {'rmsd': 0.1, 'max_dist': 0.2, 'mean_dist': 0.15}
        
        # Mock align method to return the aligned reference structure
        mock.align.return_value = (aligned_reference, translation_vector, metrics)
        
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
            # Configure the mock to return aligned reference
            aligned_reference, _, _ = self.mock_structure_aligner.align.return_value
            translation_vector = np.array([0.1, 0.1, 0.1])
            metrics = {'rmsd': 0.1, 'max_dist': 0.2, 'mean_dist': 0.15}
            self.mock_structure_aligner.align.return_value = (aligned_reference, translation_vector, metrics)
            
            # Initialize with alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=True)
            
            # Check attributes
            self.assertEqual(rbs.reference_structure, self.reference)
            self.assertEqual(rbs.target_structure, self.target)
            self.assertEqual(rbs.aligned_structure, aligned_reference)
            np.testing.assert_array_equal(rbs.translation_vector, translation_vector)
            self.assertEqual(rbs.alignment_metrics, metrics)
        
    def test_init_without_alignment(self):
        """Test initialization without structure alignment."""
        with self.structure_aligner_patch:
            # Initialize without alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Check attributes
            self.assertEqual(rbs.reference_structure, self.reference)
            self.assertEqual(rbs.target_structure, self.target)
            self.assertIsNone(rbs.aligned_structure)
            self.assertIsNone(rbs.translation_vector)
            self.assertIsNone(rbs.alignment_metrics)
    
    def test_align_structures(self):
        """Test the _align_structures method."""
        with self.structure_aligner_patch:
            # Get the aligned reference that will be returned by the mock
            aligned_reference, translation_vector, metrics = self.mock_structure_aligner.align.return_value
            
            # Initialize without alignment initially
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Now perform alignment
            rbs._align_structures(align_species=['Na'], align_metric='max_dist')
            
            # Check that align was called with correct parameters
            self.mock_structure_aligner.align.assert_called_with(
                self.reference,
                self.target,
                species=['Na'],
                metric='max_dist',
                tolerance=0.0001,
                algorithm='Nelder-Mead',
                minimizer_options=None
            )
            
            # Check that attributes were updated correctly
            self.assertIs(rbs.aligned_structure, aligned_reference)
            np.testing.assert_array_equal(rbs.translation_vector, translation_vector)
            self.assertEqual(rbs.alignment_metrics, metrics)
    
    def test_find_coordination_environments(self):
        """Test the _find_coordination_environments method."""
        with self.coord_finder_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Find coordination environments
            environments = rbs._find_coordination_environments(
                center_species='Na',
                coordination_species='Cl',
                cutoff=3.0,
                n_coord=2
            )
            
            # Check that find_environments was called with correct parameters
            self.mock_coord_finder.find_environments.assert_called_with(
                center_species='Na',
                coordination_species='Cl',
                n_coord=2,
                cutoff=3.0
            )
            
            # Check returned environments (should be dictionary format)
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
            # Get the aligned reference mock from setup
            aligned_reference, _, _ = self.mock_structure_aligner.align.return_value
            
            # Initialize with alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=True)
            
            # Map environments
            mapped = rbs._map_environments(
                ref_environments=self.ref_environments,
                target_species='Cl'
            )
            
            # Check map_coordinating_atoms call - use individual argument checks
            args, kwargs = self.mock_index_mapper.map_coordinating_atoms.call_args
            
            # First arg should be the aligned reference
            self.assertIs(args[0], aligned_reference)
            
            # Second arg should be the target
            self.assertIs(args[1], self.target)
            
            # Verify other arguments
            self.assertEqual(args[2], self.ref_environments)
            self.assertEqual(kwargs['target_species'], 'Cl')
            
            # Check returned environments
            self.assertEqual(mapped, self.mapped_environments)
    
    def test_create_polyhedral_sites(self):
        """Test the create_polyhedral_sites method."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Mock the _calculate_reference_centers_from_indices method
            mock_reference_centers = [np.array([0.1, 0.1, 0.1])]
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                mock_calc_centers.return_value = mock_reference_centers
                
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
                    coordination_species='Cl',
                    n_coord=2,
                    cutoff=3.0
                )
                
                # Check that _calculate_reference_centers_from_indices was called
                mock_calc_centers.assert_called_once_with([0])  # center indices from ref_environments.keys()
                
                # Check that map_coordinating_atoms was called
                self.mock_index_mapper.map_coordinating_atoms.assert_called_with(
                    self.reference,
                    self.target,
                    self.mapped_environments,
                    target_species='Cl'
                )
                
                # Check that create_polyhedral_sites was called with reference_centers
                self.mock_site_factory.create_polyhedral_sites.assert_called_with(
                    self.mapped_environments,
                    reference_centers=mock_reference_centers,
                    label='test_label',
                    labels=None
                )
                
                # Check returned sites
                self.assertEqual(sites, self.mock_site_factory.create_polyhedral_sites.return_value)
    
    def test_create_dynamic_voronoi_sites(self):
        """Test the create_dynamic_voronoi_sites method."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Mock the _calculate_reference_centers_from_indices method
            mock_reference_centers = [np.array([0.1, 0.1, 0.1])]
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                mock_calc_centers.return_value = mock_reference_centers
                
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
                    coordination_species='Cl',  # vertex_species is used internally
                    n_coord=2,  # n_vertices is used internally
                    cutoff=3.0
                )
                
                # Check that _calculate_reference_centers_from_indices was called
                mock_calc_centers.assert_called_once_with([0])  # center indices from ref_environments.keys()
                
                # Check that map_coordinating_atoms was called
                self.mock_index_mapper.map_coordinating_atoms.assert_called_with(
                    self.reference,
                    self.target,
                    self.ref_environments_list,
                    target_species='Cl'
                )
                
                # Check that create_dynamic_voronoi_sites was called with reference_centers
                self.mock_site_factory.create_dynamic_voronoi_sites.assert_called_with(
                    self.mapped_environments,
                    reference_centers=mock_reference_centers,
                    label=None,
                    labels=['site1']
                )
                
                # Check returned sites
                self.assertEqual(sites, self.mock_site_factory.create_dynamic_voronoi_sites.return_value)
                
    def test_create_dynamic_voronoi_sites_with_alignment(self):
        """Test the create_dynamic_voronoi_sites method when alignment has been performed."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch, self.structure_aligner_patch:
            # Get the aligned reference mock from setup
            aligned_reference, _, _ = self.mock_structure_aligner.align.return_value
            
            # Initialize with alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=True)
            
            # Mock the _calculate_reference_centers_from_indices method
            mock_reference_centers = [np.array([0.1, 0.1, 0.1])]
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                mock_calc_centers.return_value = mock_reference_centers
                
                # Create dynamic voronoi sites
                sites = rbs.create_dynamic_voronoi_sites(
                    center_species='Na',
                    reference_species='Cl',
                    cutoff=3.0,
                    n_reference=2,
                    labels=['site1'],
                    target_species='Cl'
                )
                
                # Check that _calculate_reference_centers_from_indices was called
                mock_calc_centers.assert_called_once_with([0])
                
                # Check map_coordinating_atoms call
                args, kwargs = self.mock_index_mapper.map_coordinating_atoms.call_args
                
                # First arg should be the aligned reference
                self.assertIs(args[0], aligned_reference)
                
                # Second arg should be the target
                self.assertIs(args[1], self.target)
                
                # Verify other arguments
                self.assertEqual(args[2], self.ref_environments_list)
                self.assertEqual(kwargs['target_species'], 'Cl')
                
                # Check that create_dynamic_voronoi_sites was called with reference_centers
                self.mock_site_factory.create_dynamic_voronoi_sites.assert_called_with(
                    self.mapped_environments,
                    reference_centers=mock_reference_centers,
                    label=None,
                    labels=['site1']
                )
                
                # Verify expected sites are returned
                self.assertEqual(sites, self.mock_site_factory.create_dynamic_voronoi_sites.return_value)
    
    def test_create_polyhedral_sites_with_alignment_using_setup_mock(self):
        """Test the create_polyhedral_sites method when alignment has been performed using the setup mock."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch, self.structure_aligner_patch:
            # Get the aligned reference mock from setup
            aligned_reference, _, _ = self.mock_structure_aligner.align.return_value
            
            # Initialize with alignment
            rbs = ReferenceBasedSites(self.reference, self.target, align=True)
            
            # Mock the _calculate_reference_centers_from_indices method
            mock_reference_centers = [np.array([0.1, 0.1, 0.1])]
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                mock_calc_centers.return_value = mock_reference_centers
                
                # Create polyhedral sites
                sites = rbs.create_polyhedral_sites(
                    center_species='Na',
                    vertex_species='Cl',
                    cutoff=3.0,
                    n_vertices=2,
                    label='test_label',
                    target_species='Cl'
                )
                
                # Check that _calculate_reference_centers_from_indices was called
                mock_calc_centers.assert_called_once_with([0])
                
                # Check map_coordinating_atoms call
                args, kwargs = self.mock_index_mapper.map_coordinating_atoms.call_args
                
                # First arg should be the aligned reference
                self.assertIs(args[0], aligned_reference)
                
                # Second arg should be the target
                self.assertIs(args[1], self.target)
                
                # Verify other arguments
                self.assertEqual(args[2], self.ref_environments_list)
                self.assertEqual(kwargs['target_species'], 'Cl')
                
                # Check that create_polyhedral_sites was called with reference_centers
                self.mock_site_factory.create_polyhedral_sites.assert_called_with(
                    self.mapped_environments,
                    reference_centers=mock_reference_centers,
                    label='test_label',
                    labels=None
                )
                
                # Verify expected sites are returned
                self.assertEqual(sites, self.mock_site_factory.create_polyhedral_sites.return_value)

    
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
                    coordination_species='Cl',
                    cutoff=3.0,
                    n_coord=6
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
            
    def test_map_environments_with_empty_list(self):
        """Test that _map_environments correctly handles an empty list of environments.
        
        This directly tests the internal method that's failing when no environments are found.
        """
        with self.index_mapper_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Call _map_environments directly with an empty list
            empty_environments = []
            result = rbs._map_environments(empty_environments)
            
            # Verify that an empty list is returned
            self.assertEqual(result, [])
            
            # Verify that index_mapper was not called (which would cause the error)
            self.mock_index_mapper.map_coordinating_atoms.assert_not_called()
            
    def test_validate_unique_environments_valid(self):
        """Test that environments with unique indices pass validation."""
        with self.coord_finder_patch, self.index_mapper_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Valid environments with unique indices
            valid_environments = {
                0: [1, 2, 3, 4],       # All unique
                1: [5, 6, 7],          # All unique
                2: [8]                 # Single index
            }
            
            # Should not raise any exception
            rbs._validate_unique_environments(valid_environments)
    
    def test_validate_unique_environments_empty(self):
        """Test that empty environments pass validation."""
        with self.coord_finder_patch, self.index_mapper_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Empty environments
            empty_environments = {
                0: [],                 # Empty list
                1: []                  # Another empty list
            }
            
            # Should not raise any exception
            rbs._validate_unique_environments(empty_environments)
    
    def test_validate_unique_environments_duplicates(self):
        """Test that environments with duplicate indices raise ValueError."""
        with self.coord_finder_patch, self.index_mapper_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Environments with duplicate indices
            duplicate_environments = {
                0: [1, 2, 2, 3],       # Duplicate 2
                1: [4, 5, 6]           # All unique (but earlier list has duplicates)
            }
            
            # Should raise ValueError due to duplicates in the first environment
            with self.assertRaises(ValueError) as context:
                rbs._validate_unique_environments(duplicate_environments)
            
            # Check error message
            self.assertIn("Environment for center atom 0 contains duplicate atom indices", str(context.exception))
            self.assertIn("2", str(context.exception))  # Should mention the duplicated index
    
    def test_validate_unique_environments_multiple_duplicates(self):
        """Test validation with multiple duplicate indices in an environment."""
        with self.coord_finder_patch, self.index_mapper_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Environment with multiple duplicates
            multi_duplicate_environments = {
                0: [1, 2, 3, 1, 2, 4]  # Duplicates 1 and 2
            }
            
            # Should raise ValueError
            with self.assertRaises(ValueError) as context:
                rbs._validate_unique_environments(multi_duplicate_environments)
            
            # Check error message contains both duplicated indices
            error_msg = str(context.exception)
            self.assertIn("1", error_msg)
            self.assertIn("2", error_msg)
            
            # Check suggestion about supercell is included
            self.assertIn("Please use a larger supercell", error_msg)
            
    def test_integration_with_find_environments(self):
        """Test integration with _find_coordination_environments method."""
        # Create a mock that returns environments with duplicates
        with self.coord_finder_patch, self.index_mapper_patch:
            # Configure the mock to return environments with duplicates
            duplicate_envs = {0: [1, 2, 3, 1]}  # Na at index 0 with duplicated Cl at index 1
            self.mock_coord_finder.find_environments.return_value = duplicate_envs
            
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Attempting to create sites should raise the duplicate error
            with self.assertRaises(ValueError) as context:
                rbs.create_polyhedral_sites(
                    center_species='Na',
                    vertex_species='Cl',
                    cutoff=3.0,
                    n_vertices=4
                )
                
            # Check error message
            self.assertIn("contains duplicate atom indices", str(context.exception))
            
    def test_map_environments_uses_correct_structures(self):
        """Test that _map_environments uses the appropriate structures based on alignment status."""
        ref_environments = [[1, 2, 3]]
        
        # Case 1: When alignment has been performed
        # ---------------------------------------
        with patch('site_analysis.reference_workflow.reference_based_sites.IndexMapper') as mock_mapper_class:
            mock_mapper = mock_mapper_class.return_value
            mock_mapper.map_coordinating_atoms.return_value = []
            
            # Create ReferenceBasedSites with mocked but compatible objects
            rbs1 = ReferenceBasedSites(self.reference, self.target, align=False)
            rbs1._index_mapper = mock_mapper
            
            # Set an aligned structure
            aligned_reference = Mock(spec=Structure)
            rbs1.aligned_structure = aligned_reference
            
            # Call the method
            rbs1._map_environments(ref_environments, target_species="Na")
            
            # Check that the method was called once
            mock_mapper.map_coordinating_atoms.assert_called_once()
            
            # Get the call arguments
            args, kwargs = mock_mapper.map_coordinating_atoms.call_args
            
            # Check arguments individually
            self.assertIs(args[0], aligned_reference)  # First arg should be aligned reference
            self.assertIs(args[1], self.target)        # Second arg should be target structure
            self.assertEqual(args[2], ref_environments) # Third arg should be environments list
            self.assertEqual(kwargs['target_species'], "Na")
        
        # Case 2: When no alignment has been performed
        # -----------------------------------------
        with patch('site_analysis.reference_workflow.reference_based_sites.IndexMapper') as mock_mapper_class:
            mock_mapper = mock_mapper_class.return_value
            mock_mapper.map_coordinating_atoms.return_value = []
            
            rbs2 = ReferenceBasedSites(self.reference, self.target, align=False)
            rbs2._index_mapper = mock_mapper
            rbs2.aligned_structure = None  # No aligned structure
            
            # Call the method
            rbs2._map_environments(ref_environments)
            
            # Check that the method was called once
            mock_mapper.map_coordinating_atoms.assert_called_once()
            
            # Get the call arguments
            args, kwargs = mock_mapper.map_coordinating_atoms.call_args
            
            # Check arguments individually
            self.assertIs(args[0], self.reference)     # First arg should be original reference
            self.assertIs(args[1], self.target)        # Second arg should be target structure
            self.assertEqual(args[2], ref_environments) # Third arg should be environments list
            self.assertEqual(kwargs.get('target_species'), None)
            
    def test_align_structures_unit(self):
        """Unit test for _align_structures - verifies parameters are correctly passed to StructureAligner."""
        # Create mock structures
        mock_reference = Mock(spec=Structure)
        mock_target = Mock(spec=Structure)
        
        # Create test parameters
        align_species = ["Na"]
        align_metric = "max_dist"
        align_algorithm = "differential_evolution"
        align_minimizer_options = {"popsize": 20, "maxiter": 500}
        
        # Create ReferenceBasedSites with mocked structures but don't call _align_structures yet
        with patch('site_analysis.reference_workflow.reference_based_sites.StructureAligner') as MockAligner:
            # Configure mock
            mock_aligner = Mock()
            MockAligner.return_value = mock_aligner
            mock_aligner.align.return_value = (Mock(), np.array([0.1, 0.1, 0.1]), {"rmsd": 0.1})
            
            # Create instance with align=False to avoid immediate call to _align_structures
            rbs = ReferenceBasedSites(
                reference_structure=mock_reference,
                target_structure=mock_target,
                align=False  # Important: don't align in the constructor
            )
            
            # Now call _align_structures directly with our parameters
            rbs._align_structures(
                align_species=align_species,
                align_metric=align_metric,
                align_algorithm=align_algorithm,
                align_minimizer_options=align_minimizer_options
            )
            
            # Verify StructureAligner was created
            MockAligner.assert_called_once()
            
            # Verify align was called with correct parameters
            mock_aligner.align.assert_called_once()
            _, kwargs = mock_aligner.align.call_args
            
            # Check each parameter was correctly passed
            self.assertEqual(kwargs["species"], align_species)
            self.assertEqual(kwargs["metric"], align_metric)
            self.assertEqual(kwargs["algorithm"], align_algorithm)
            self.assertEqual(kwargs["minimizer_options"], align_minimizer_options)
            
    def test_tolerance_passed_to_structure_aligner(self):
        """Test that align_tolerance is correctly passed to StructureAligner.align()."""
        # Create mock structures
        mock_reference = Mock(spec=Structure)
        mock_target = Mock(spec=Structure)
        
        # Custom tolerance value to test
        align_tolerance = 1e-5
        
        # Create ReferenceBasedSites with mocked structures
        with patch('site_analysis.reference_workflow.reference_based_sites.StructureAligner') as MockAligner:
            # Configure mock
            mock_aligner = Mock()
            MockAligner.return_value = mock_aligner
            mock_aligner.align.return_value = (Mock(), np.array([0]), {"rmsd": 0})
            
            # Create instance with align=True to trigger call to aligner
            rbs = ReferenceBasedSites(
                reference_structure=mock_reference,
                target_structure=mock_target,
                align=True,
                align_tolerance=align_tolerance
            )
            
            # Verify align was called with correct tolerance
            mock_aligner.align.assert_called_once()
            _, kwargs = mock_aligner.align.call_args
            self.assertEqual(kwargs["tolerance"], align_tolerance)
            
    def test_align_structures_passes_tolerance(self):
        """Test that _align_structures correctly passes tolerance to the StructureAligner."""
        # Mock structures
        mock_reference = Mock(spec=Structure)
        mock_target = Mock(spec=Structure)
        
        # Custom tolerance value to test
        align_tolerance = 1e-5
        
        # Create ReferenceBasedSites with mocked structures but don't align immediately
        with patch('site_analysis.reference_workflow.reference_based_sites.StructureAligner') as MockAligner:
            # Configure mock aligner
            mock_aligner = Mock()
            MockAligner.return_value = mock_aligner
            mock_aligner.align.return_value = (Mock(), np.array([0]), {"rmsd": 0})
            
            # Create instance without alignment
            rbs = ReferenceBasedSites(
                reference_structure=mock_reference,
                target_structure=mock_target,
                align=False  # Don't align in constructor
            )
            
            # Now directly call _align_structures with our custom tolerance
            rbs._align_structures(align_tolerance=align_tolerance)
            
            # Verify align was called with correct tolerance
            mock_aligner.align.assert_called_once()
            _, kwargs = mock_aligner.align.call_args
            self.assertEqual(kwargs["tolerance"], align_tolerance)
            
    def test_create_polyhedral_sites_passes_calculated_reference_centers_to_site_factory(self):
        """Test that create_polyhedral_sites passes calculated reference centers to site factory."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Mock calculated reference centers
            mock_reference_centers = [np.array([0.1, 0.1, 0.1])]
            
            # Mock the _calculate_reference_centers_from_indices method
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                mock_calc_centers.return_value = mock_reference_centers
                
                # Call create_polyhedral_sites
                rbs.create_polyhedral_sites(
                    center_species='Na',
                    vertex_species='Cl',
                    cutoff=3.0,
                    n_vertices=2,
                    label='test_label'
                )
                
                # Verify that create_polyhedral_sites was called with reference_centers
                self.mock_site_factory.create_polyhedral_sites.assert_called_once_with(
                    self.mapped_environments,
                    reference_centers=mock_reference_centers,
                    label='test_label',
                    labels=None
                )
    
    def test_create_polyhedral_sites_passes_none_reference_centers_when_disabled(self):
        """Test that create_polyhedral_sites passes None for reference_centers when use_reference_centers=False."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Mock the _calculate_reference_centers_from_indices method
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                
                # Call create_polyhedral_sites with use_reference_centers=False
                rbs.create_polyhedral_sites(
                    center_species='Na',
                    vertex_species='Cl',
                    cutoff=3.0,
                    n_vertices=2,
                    use_reference_centers=False,
                    label='test_label'
                )
                
                # Verify that create_polyhedral_sites was called with reference_centers=None
                self.mock_site_factory.create_polyhedral_sites.assert_called_once_with(
                    self.mapped_environments,
                    reference_centers=None,
                    label='test_label',
                    labels=None
                )
    
    def test_create_dynamic_voronoi_sites_passes_calculated_reference_centers_to_site_factory(self):
        """Test that create_dynamic_voronoi_sites passes calculated reference centers to site factory."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Mock calculated reference centers
            mock_reference_centers = [np.array([0.2, 0.2, 0.2])]
            
            # Mock the _calculate_reference_centers_from_indices method
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                mock_calc_centers.return_value = mock_reference_centers
                
                # Call create_dynamic_voronoi_sites
                rbs.create_dynamic_voronoi_sites(
                    center_species='Na',
                    reference_species='Cl',
                    cutoff=3.0,
                    n_reference=2,
                    labels=['test_site']
                )
                
                # Verify that create_dynamic_voronoi_sites was called with reference_centers
                self.mock_site_factory.create_dynamic_voronoi_sites.assert_called_once_with(
                    self.mapped_environments,
                    reference_centers=mock_reference_centers,
                    label=None,
                    labels=['test_site']
                )
    
    def test_create_dynamic_voronoi_sites_passes_none_reference_centers_when_disabled(self):
        """Test that create_dynamic_voronoi_sites passes None for reference_centers when use_reference_centers=False."""
        with self.coord_finder_patch, self.index_mapper_patch, self.site_factory_patch:
            rbs = ReferenceBasedSites(self.reference, self.target, align=False)
            
            # Mock the _calculate_reference_centers_from_indices method
            with patch.object(rbs, '_calculate_reference_centers_from_indices') as mock_calc_centers:
                
                # Call create_dynamic_voronoi_sites with use_reference_centers=False
                rbs.create_dynamic_voronoi_sites(
                    center_species='Na',
                    reference_species='Cl',
                    cutoff=3.0,
                    n_reference=2,
                    use_reference_centers=False,
                    labels=['test_site']
                )
                
                # Verify that create_dynamic_voronoi_sites was called with reference_centers=None
                self.mock_site_factory.create_dynamic_voronoi_sites.assert_called_once_with(
                    self.mapped_environments,
                    reference_centers=None,
                    label=None,
                    labels=['test_site']
                )
            
class TestReferenceBasedSitesCalculateReferenceCentres(unittest.TestCase):
    """Unit tests for _calculate_reference_centers_from_indices method."""
    
    def setUp(self):
        """Set up simple test structures."""
        self.lattice = Lattice.cubic(5.0)
        
        # Reference structure with known coordinates
        self.reference_structure = Structure(
            lattice=self.lattice,
            species=["Li", "O", "Li", "O", "Li"],
            coords=[
                [0.0, 0.0, 0.0],    # Li at index 0
                [0.1, 0.0, 0.0],    # O at index 1
                [0.2, 0.2, 0.2],    # Li at index 2
                [0.3, 0.3, 0.3],    # O at index 3
                [0.5, 0.5, 0.5]     # Li at index 4
            ]
        )
        
        # Target structure (same for these unit tests)
        self.target_structure = self.reference_structure.copy()
        
    def test_calculate_reference_centers_single_index(self):
        """Test calculating reference centre for a single center index."""
        rbs = ReferenceBasedSites(
            reference_structure=self.reference_structure,
            target_structure=self.target_structure,
            align=False
        )
        
        center_indices = [0]  # Li at [0.0, 0.0, 0.0]
        reference_centers = rbs._calculate_reference_centers_from_indices(center_indices)
        
        # Should return one centre
        self.assertEqual(len(reference_centers), 1)
        
        # Centre should match the coordinate of atom 0
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(reference_centers[0], expected)
        
    def test_calculate_reference_centers_multiple_indices(self):
        """Test calculating reference centres for multiple center indices."""
        rbs = ReferenceBasedSites(
            reference_structure=self.reference_structure,
            target_structure=self.target_structure,
            align=False
        )
        
        center_indices = [0, 2, 4]  # Three Li atoms
        reference_centers = rbs._calculate_reference_centers_from_indices(center_indices)
        
        # Should return three centres
        self.assertEqual(len(reference_centers), 3)
        
        # Centres should match the coordinates of atoms 0, 2, 4
        expected_centres = [
            np.array([0.0, 0.0, 0.0]),  # Li at index 0
            np.array([0.2, 0.2, 0.2]),  # Li at index 2
            np.array([0.5, 0.5, 0.5])   # Li at index 4
        ]
        
        for i, expected in enumerate(expected_centres):
            np.testing.assert_array_equal(reference_centers[i], expected)
            
    def test_calculate_reference_centers_uses_reference_structure_when_no_alignment(self):
        """Test that method uses reference_structure when no alignment was performed."""
        rbs = ReferenceBasedSites(
            reference_structure=self.reference_structure,
            target_structure=self.target_structure,
            align=False  # No alignment
        )
        
        # Verify no aligned structure exists
        self.assertIsNone(rbs.aligned_structure)
        
        center_indices = [2]  # Li at [0.2, 0.2, 0.2]
        reference_centers = rbs._calculate_reference_centers_from_indices(center_indices)
        
        # Should use reference_structure coordinates
        expected = self.reference_structure[2].frac_coords
        np.testing.assert_array_equal(reference_centers[0], expected)
        
    def test_calculate_reference_centers_uses_aligned_structure_when_available(self):
        """Test that method uses aligned_structure when alignment was performed."""
        rbs = ReferenceBasedSites(
            reference_structure=self.reference_structure,
            target_structure=self.target_structure,
            align=True  # Enable alignment
        )
        
        # Manually set aligned_structure to simulate alignment
        aligned_structure = Structure(
            lattice=self.lattice,
            species=["Li", "O", "Li", "O", "Li"],
            coords=[
                [0.1, 0.1, 0.1],    # Li at index 0 (shifted)
                [0.2, 0.1, 0.1],    # O at index 1 (shifted)
                [0.3, 0.3, 0.3],    # Li at index 2 (shifted)
                [0.4, 0.4, 0.4],    # O at index 3 (shifted)
                [0.6, 0.6, 0.6]     # Li at index 4 (shifted)
            ]
        )
        rbs.aligned_structure = aligned_structure
        
        center_indices = [2]  # Li atom
        reference_centers = rbs._calculate_reference_centers_from_indices(center_indices)
        
        # Should use aligned_structure coordinates, not reference_structure
        expected = aligned_structure[2].frac_coords  # [0.3, 0.3, 0.3]
        np.testing.assert_array_equal(reference_centers[0], expected)
        
        # Verify it's different from reference_structure
        reference_coord = self.reference_structure[2].frac_coords  # [0.2, 0.2, 0.2]
        self.assertFalse(np.array_equal(reference_centers[0], reference_coord))
        
    def test_calculate_reference_centers_empty_list(self):
        """Test calculating reference centres for empty list of indices."""
        rbs = ReferenceBasedSites(
            reference_structure=self.reference_structure,
            target_structure=self.target_structure,
            align=False
        )
        
        center_indices = []
        reference_centers = rbs._calculate_reference_centers_from_indices(center_indices)
        
        # Should return empty list
        self.assertEqual(len(reference_centers), 0)
        self.assertEqual(reference_centers, [])
        
    def test_calculate_reference_centers_copies_coordinates(self):
        """Test that returned coordinates are copies, not references."""
        rbs = ReferenceBasedSites(
            reference_structure=self.reference_structure,
            target_structure=self.target_structure,
            align=False
        )
        
        center_indices = [0]
        reference_centers = rbs._calculate_reference_centers_from_indices(center_indices)
        
        # Modify the returned coordinate
        original_value = reference_centers[0][0]
        reference_centers[0][0] = 0.9
        
        # Original structure should be unchanged
        structure_value = self.reference_structure[0].frac_coords[0]
        self.assertEqual(structure_value, original_value)
        self.assertNotEqual(structure_value, 0.9)    

class ReferenceWorkflowImportTestCase(unittest.TestCase):
    """Test that reference workflow classes can be imported at package level."""

    def test_can_import_reference_based_sites(self):
        """Test that ReferenceBasedSites can be imported from reference_workflow package."""
        try:
            from site_analysis.reference_workflow import ReferenceBasedSites
        except ImportError as e:
            self.fail(f"Could not import ReferenceBasedSites from reference_workflow: {e}")



if __name__ == '__main__':
    unittest.main()
