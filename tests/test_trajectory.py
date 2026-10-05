import unittest
from collections import Counter
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.trajectory import Trajectory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.spherical_site import SphericalSite
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site_collection import SiteCollection
from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection
from site_analysis.voronoi_site_collection import VoronoiSiteCollection
from site_analysis.spherical_site_collection import SphericalSiteCollection
from site_analysis.dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import Mock, patch, PropertyMock

import tempfile
import os
import json


class TrajectoryInitializationTestCase(unittest.TestCase):
    """Tests for Trajectory initialization with different site types."""

    def setUp(self):
        Site._newid = 0

    def test_initialisation_with_polyhedral_sites(self):
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3])]
        trajectory = Trajectory(atoms=[Atom(index=0)], sites=sites)
        self.assertIsInstance(trajectory.site_collection, PolyhedralSiteCollection)

    def test_initialisation_with_voronoi_sites(self):
        sites = [VoronoiSite(frac_coords=np.array([0.5, 0.5, 0.5]))]
        trajectory = Trajectory(atoms=[Atom(index=0)], sites=sites)
        self.assertIsInstance(trajectory.site_collection, VoronoiSiteCollection)

    def test_initialisation_with_spherical_sites(self):
        sites = [SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)]
        trajectory = Trajectory(atoms=[Atom(index=0)], sites=sites)
        self.assertIsInstance(trajectory.site_collection, SphericalSiteCollection)

    def test_initialisation_with_dynamic_voronoi_sites(self):
        sites = [DynamicVoronoiSite(reference_indices=[0, 1])]
        trajectory = Trajectory(atoms=[Atom(index=0)], sites=sites)
        self.assertIsInstance(trajectory.site_collection, DynamicVoronoiSiteCollection)

    def test___len___returns_zero_for_empty_trajectory(self):
        sites = [SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)]
        trajectory = Trajectory(atoms=[Atom(index=0)], sites=sites)
        self.assertEqual(len(trajectory), 0)

    def test___len___returns_number_of_timesteps(self):
        trajectory = Trajectory.__new__(Trajectory)
        trajectory.timesteps = ['foo', 'bar']
        self.assertEqual(len(trajectory), 2)

    def test_init_raises_type_error_if_passed_mixed_site_types(self):
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                 VoronoiSite(frac_coords=np.array([0.5, 0.5, 0.5]))]
        with self.assertRaises(TypeError):
            Trajectory(atoms=[Atom(index=0)], sites=sites)

    def test_init_raises_type_error_if_passed_invalid_site_type(self):
        with self.assertRaises(TypeError):
            Trajectory(atoms=[Atom(index=0)], sites=["foo"])


class TrajectoryFunctionalityTestCase(unittest.TestCase):
    """Tests for Trajectory functionality using real pymatgen objects."""
    
    def setUp(self):
        """Set up test fixtures with simple objects."""
        # Reset Site._newid counter
        Site._newid = 0
        
        # Create simple lattice
        self.lattice = Lattice.cubic(5.0)
        
        # Create a structure with two atoms
        species = ["Na", "Na"]
        coords = [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]
        self.structure = Structure(self.lattice, species, coords)
        
        # Create two slightly different structures for trajectory testing
        coords2 = [[0.11, 0.1, 0.1], [0.51, 0.5, 0.5]]
        self.structure2 = Structure(self.lattice, species, coords2)
        
        # Create atoms
        self.atom1 = Atom(index=0)
        self.atom2 = Atom(index=1)
        self.atoms = [self.atom1, self.atom2]
        
        # Create sites (spherical sites are simple to create)
        self.site1 = SphericalSite(frac_coords=np.array([0.1, 0.1, 0.1]), rcut=0.3, label="site1")
        self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=0.3, label="site2")
        self.sites = [self.site1, self.site2]
        
        # Create trajectory object
        self.trajectory = Trajectory(sites=self.sites, atoms=self.atoms)
    
    def test_initialization(self):
        """Test that Trajectory initializes correctly."""
        # Check that the sites and atoms were stored
        self.assertEqual(self.trajectory.sites, self.sites)
        self.assertEqual(self.trajectory.atoms, self.atoms)
        
        # Check initial state
        self.assertEqual(len(self.trajectory.timesteps), 0)
        
        # Check lookup dictionaries
        self.assertEqual(self.trajectory.atom_lookup[0], 0)  # atom with index 0 is at position 0
        self.assertEqual(self.trajectory.atom_lookup[1], 1)  # atom with index 1 is at position 1
        self.assertEqual(self.trajectory.site_lookup[self.site1.index], 0)
        self.assertEqual(self.trajectory.site_lookup[self.site2.index], 1)
    
    def test_atom_by_index(self):
        """Test retrieving an atom by index."""
        self.assertIs(self.trajectory.atom_by_index(0), self.atom1)
        self.assertIs(self.trajectory.atom_by_index(1), self.atom2)
    
    def test_site_by_index(self):
        """Test retrieving a site by index."""
        self.assertIs(self.trajectory.site_by_index(self.site1.index), self.site1)
        self.assertIs(self.trajectory.site_by_index(self.site2.index), self.site2)
    
    def test_analyse_structure_delegates_to_site_collection(self):
        """Test that analyse_structure delegates to site_collection.analyse_structure."""
        # Create minimal real sites (required for Trajectory initialization)
        from site_analysis.spherical_site import SphericalSite
        import numpy as np
        
        real_sites = [SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)]
        
        # Create mock atoms with required attributes
        mock_atom1 = Mock(spec=Atom)
        mock_atom1.index = 0
        mock_atom2 = Mock(spec=Atom) 
        mock_atom2.index = 1
        mock_atoms = [mock_atom1, mock_atom2]
        
        mock_structure = Mock(spec=Structure)
        
        # Create trajectory (this will create a real site_collection)
        trajectory = Trajectory(sites=real_sites, atoms=mock_atoms)
        
        # Mock the site_collection that was created
        trajectory.site_collection = Mock()
        
        # Call the method
        trajectory.analyse_structure(mock_structure)
        
        # Verify it delegates correctly
        trajectory.site_collection.analyse_structure.assert_called_once_with(mock_atoms, mock_structure)
    
    def test_append_timestep(self):
        """Test that append_timestep correctly adds a timestep."""
        # Pre-assign atoms to sites for simplicity
        self.atom1.in_site = self.site1.index
        self.atom2.in_site = self.site2.index
        self.site1.contains_atoms = [0]
        self.site2.contains_atoms = [1]
        
        # Append a timestep
        self.trajectory.append_timestep(self.structure, t=42)
        
        # Check timestep was recorded
        self.assertEqual(len(self.trajectory.timesteps), 1)
        self.assertEqual(self.trajectory.timesteps[0], 42)
        
        # Check atom trajectories were updated
        self.assertEqual(len(self.atom1.trajectory), 1)
        self.assertEqual(self.atom1.trajectory[0], self.site1.index)
        
        self.assertEqual(len(self.atom2.trajectory), 1)
        self.assertEqual(self.atom2.trajectory[0], self.site2.index)
        
        # Check site trajectories were updated
        self.assertEqual(len(self.site1.trajectory), 1)
        self.assertEqual(self.site1.trajectory[0], [0])  # Contains atom 0
        
        self.assertEqual(len(self.site2.trajectory), 1)
        self.assertEqual(self.site2.trajectory[0], [1])  # Contains atom 1
    
    def test_append_timestep_with_t_zero(self):
        """Test that append_timestep records timestep t=0."""
        self.trajectory.append_timestep(self.structure, t=0)
        self.assertEqual(self.trajectory.timesteps, [0])

    def test_reset(self):
        """Test that reset correctly clears the trajectory."""
        # Setup with a timestep
        self.test_append_timestep()  # Reuse the previous test
        
        # Reset the trajectory
        self.trajectory.reset()
        
        # Check timesteps are cleared
        self.assertEqual(len(self.trajectory.timesteps), 0)
        
        # Check atoms are reset
        self.assertIsNone(self.atom1.in_site)
        self.assertIsNone(self.atom2.in_site)
        self.assertEqual(len(self.atom1.trajectory), 0)
        self.assertEqual(len(self.atom2.trajectory), 0)
        self.assertIsNone(self.atom1.most_recent_site)
        self.assertIsNone(self.atom2.most_recent_site)
        
        # Check sites are reset
        self.assertEqual(len(self.site1.contains_atoms), 0)
        self.assertEqual(len(self.site2.contains_atoms), 0)
        self.assertEqual(len(self.site1.trajectory), 0)
        self.assertEqual(len(self.site2.trajectory), 0)
    
    def test_trajectory_from_structures(self):
        """Test generating a trajectory from a list of structures."""
        # Create list of structures
        structures = [self.structure, self.structure2]
        
        # Generate trajectory
        self.trajectory.trajectory_from_structures(structures)
        
        # Check two timesteps were recorded
        self.assertEqual(len(self.trajectory.timesteps), 2)
        self.assertEqual(self.trajectory.timesteps, [1, 2])  # Should be 1-indexed
        
        # Check atom trajectories length
        self.assertEqual(len(self.atom1.trajectory), 2)
        self.assertEqual(len(self.atom2.trajectory), 2)
        
        # Check site trajectories length
        self.assertEqual(len(self.site1.trajectory), 2)
        self.assertEqual(len(self.site2.trajectory), 2)
    
    def test_atom_sites_property(self):
        """Test the atom_sites property."""
        # Assign atoms to sites
        self.atom1.in_site = self.site1.index
        self.atom2.in_site = self.site2.index
        
        # Check the property
        atom_sites = self.trajectory.atom_sites
        self.assertEqual(len(atom_sites), 2)
        self.assertEqual(atom_sites[0], self.site1.index)
        self.assertEqual(atom_sites[1], self.site2.index)
    
    def test_site_occupations_property(self):
        """Test the site_occupations property."""
        # Assign atoms to sites
        self.site1.contains_atoms = [0]
        self.site2.contains_atoms = [1]
        
        # Check the property
        occupations = self.trajectory.site_occupations
        self.assertEqual(len(occupations), 2)
        self.assertEqual(occupations[0], [0])  # site1 contains atom0
        self.assertEqual(occupations[1], [1])  # site2 contains atom1
    
    def test_trajectory_properties(self):
        """Test atoms_trajectory and sites_trajectory properties."""
        # Setup with two timesteps
        self.atom1.trajectory = [self.site1.index, self.site1.index]
        self.atom2.trajectory = [self.site2.index, self.site2.index]
        
        self.site1.trajectory = [[0], [0]]
        self.site2.trajectory = [[1], [1]]
        
        # Check atoms_trajectory
        at = self.trajectory.atoms_trajectory
        self.assertEqual(len(at), 2)  # 2 timesteps
        self.assertEqual(at[0], [self.site1.index, self.site2.index])  # First timestep
        self.assertEqual(at[1], [self.site1.index, self.site2.index])  # Second timestep
        
        # Check sites_trajectory
        st = self.trajectory.sites_trajectory
        self.assertEqual(len(st), 2)  # 2 timesteps
        self.assertEqual(st[0], [[0], [1]])  # First timestep
        self.assertEqual(st[1], [[0], [1]])  # Second timestep
        
        # Check shortcuts
        self.assertEqual(self.trajectory.at, at)
        self.assertEqual(self.trajectory.st, st)
    
    def test_site_coordination_numbers(self):
        """Test the site_coordination_numbers method."""
        # Use patch to mock the coordination_number property
        with patch.object(SphericalSite, 'coordination_number', 
                         new_callable=PropertyMock) as mock_coord_number:
            # Configure the mock to return different values for different sites
            mock_coord_number.side_effect = [4, 6]
            
            # Get coordination numbers
            coordination = self.trajectory.site_coordination_numbers()
            
            # Check the counter
            self.assertEqual(coordination[4], 1)  # 1 site with coordination 4
            self.assertEqual(coordination[6], 1)  # 1 site with coordination 6
    
    def test_site_labels(self):
        """Test the site_labels method."""
        # Sites already have labels from setUp
        labels = self.trajectory.site_labels()
        
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0], "site1")
        self.assertEqual(labels[1], "site2")
        
    def test_assign_site_occupations(self):
        """Test that assign_site_occupations extracts lattice_matrix and delegates."""
        self.trajectory.site_collection = Mock()

        self.trajectory.assign_site_occupations(self.structure)

        self.trajectory.site_collection.assign_site_occupations.assert_called_once()
        args = self.trajectory.site_collection.assign_site_occupations.call_args[0]
        self.assertIs(args[0], self.atoms)
        np.testing.assert_array_equal(args[1], self.structure.lattice.matrix)
    
    def test_trajectory_from_structures_with_progress(self):
        """Test trajectory_from_structures wraps iterator with tqdm."""
        structures = [self.structure, self.structure2]

        with patch.object(self.trajectory, 'append_timestep') as mock_append, \
             patch('site_analysis.trajectory.tqdm') as mock_tqdm:
            mock_tqdm.return_value = enumerate(structures, 1)

            self.trajectory.trajectory_from_structures(structures, progress=True)

            mock_tqdm.assert_called_once()
            _, kwargs = mock_tqdm.call_args
            self.assertEqual(kwargs['total'], 2)
            self.assertEqual(mock_append.call_count, 2)

    def test_init_with_empty_sites(self):
        """Test that Trajectory raises ValueError with empty sites list."""
        with self.assertRaises(ValueError) as context:
            Trajectory(sites=[], atoms=[Atom(index=0)])

        self.assertIn("empty sites list", str(context.exception))

    def test_init_with_empty_atoms(self):
        """Test that Trajectory raises ValueError with empty atoms list."""
        sites = [SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)]

        with self.assertRaises(ValueError) as context:
            Trajectory(sites=sites, atoms=[])

        self.assertIn("empty atoms list", str(context.exception))
        
    def test_site_summaries_default(self):
        """Test that site_summaries delegates to site_collection."""
        # Create real sites
        sites = [
            SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0),
            SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)
        ]
        mock_atoms = [Mock(spec=Atom, index=0)]
        
        # Create trajectory
        trajectory = Trajectory(sites=sites, atoms=mock_atoms)
        
        # Replace site_collection with mock
        trajectory.site_collection = Mock(spec=SiteCollection)
        trajectory.site_collection.summaries.return_value = [
            {'index': 0, 'site_type': 'SphericalSite'},
            {'index': 1, 'site_type': 'SphericalSite'}
        ]
        
        result = trajectory.site_summaries()
        
        # Should delegate to site_collection
        trajectory.site_collection.summaries.assert_called_once_with(metrics=None)
        
        # Should return the same result
        self.assertEqual(result, [
            {'index': 0, 'site_type': 'SphericalSite'},
            {'index': 1, 'site_type': 'SphericalSite'}
        ])
    
    def test_site_summaries_with_metrics(self):
        """Test that site_summaries passes metrics to site_collection."""
        # Create real site
        sites = [SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)]
        mock_atoms = [Mock(spec=Atom, index=0)]
        
        # Create trajectory
        trajectory = Trajectory(sites=sites, atoms=mock_atoms)
        
        # Replace site_collection with mock
        trajectory.site_collection = Mock(spec=SiteCollection)
        trajectory.site_collection.summaries.return_value = [
            {'index': 0},
            {'index': 1}
        ]
        
        result = trajectory.site_summaries(metrics=['index'])
        
        # Should pass metrics through
        trajectory.site_collection.summaries.assert_called_once_with(metrics=['index'])
        
        # Should return the same result
        self.assertEqual(result, [{'index': 0}, {'index': 1}])
        
    def test_write_site_summaries_default(self):
        """Test that write_site_summaries writes summaries to JSON file."""
        # Create trajectory
        sites = [SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)]
        mock_atoms = [Mock(spec=Atom, index=0)]
        trajectory = Trajectory(sites=sites, atoms=mock_atoms)
        
        # Mock site_summaries to return known data
        expected_data = [{'index': 0, 'site_type': 'SphericalSite'}]
        trajectory.site_summaries = Mock(return_value=expected_data)
        
        # Mock file operations
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        
        with patch('builtins.open', return_value=mock_file) as mock_open:
            with patch('json.dump') as mock_json_dump:
                trajectory.write_site_summaries('output.json')
                
                # Should call site_summaries with default metrics
                trajectory.site_summaries.assert_called_once_with(metrics=None)
                
                # Should open file for writing
                mock_open.assert_called_once_with('output.json', 'w')
                
                # Should dump data to file with nice formatting
                mock_json_dump.assert_called_once_with(expected_data, mock_file, indent=2)
    
    def test_write_site_summaries_with_metrics(self):
        """Test that write_site_summaries passes metrics parameter."""
        # Create trajectory
        sites = [SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0)]
        mock_atoms = [Mock(spec=Atom, index=0)]
        trajectory = Trajectory(sites=sites, atoms=mock_atoms)
        
        # Mock site_summaries
        expected_data = [{'index': 0}]
        trajectory.site_summaries = Mock(return_value=expected_data)
        
        # Mock file operations
        mock_file = Mock()
        mock_file.__enter__ = Mock(return_value=mock_file)
        mock_file.__exit__ = Mock(return_value=None)
        
        with patch('builtins.open', return_value=mock_file) as mock_open:
            with patch('json.dump') as mock_json_dump:
                trajectory.write_site_summaries('output.json', metrics=['index'])
                
                # Should pass metrics to site_summaries
                trajectory.site_summaries.assert_called_once_with(metrics=['index'])
                
                # Should write to file
                mock_json_dump.assert_called_once_with(expected_data, mock_file, indent=2)
    
    def test_reset_clears_dynamic_voronoi_batch_caches(self):
        """trajectory.reset() should clear DynamicVoronoiSiteCollection group caches."""
        Site._newid = 0
        lattice = Lattice.cubic(10.0)
        # 4 reference atoms + 1 mobile atom
        coords = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
                  [0.7, 0.7, 0.7], [0.8, 0.8, 0.8],
                  [0.15, 0.15, 0.15]]
        structure = Structure(lattice, ["Na"] * 5, coords)

        site1 = DynamicVoronoiSite(reference_indices=[0, 1])
        site2 = DynamicVoronoiSite(reference_indices=[2, 3])
        atom = Atom(index=4)
        trajectory = Trajectory(sites=[site1, site2], atoms=[atom])

        trajectory.append_timestep(structure, t=1)
        self.assertTrue(trajectory.site_collection._centre_groups[0].initialised)

        trajectory.reset()
        self.assertFalse(trajectory.site_collection._centre_groups[0].initialised)
        self.assertIsNone(site1._centre_coords)
        self.assertIsNone(site2._centre_coords)

    def test_write_site_summaries_integration(self):
        """Integration test that actually writes a file."""
        
        # Create trajectory with real data
        Site.reset_index()  # Reset to get predictable indices
        sites = [
            SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.0),
            SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)
        ]
        mock_atoms = [Mock(spec=Atom, index=0)]
        trajectory = Trajectory(sites=sites, atoms=mock_atoms)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_filename = f.name
        
        try:
            trajectory.write_site_summaries(temp_filename, metrics=['index', 'site_type'])
            
            # Read back and verify
            with open(temp_filename, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]['index'], 0)
            self.assertEqual(data[0]['site_type'], 'SphericalSite')
            self.assertEqual(data[1]['index'], 1)
            self.assertEqual(data[1]['site_type'], 'SphericalSite')
        finally:
            # Clean up
            os.unlink(temp_filename)


class TransitionCountsBySiteTestCase(unittest.TestCase):
    """Tests for Trajectory.transition_counts_by_site()."""

    def setUp(self):
        Site._newid = 0
        self.site0 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=0.3, label="A")
        self.site1 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=0.3, label="B")
        self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.0, 0.5]), rcut=0.3)
        self.sites = [self.site0, self.site1, self.site2]
        self.atoms = [Atom(index=0)]
        self.trajectory = Trajectory(sites=self.sites, atoms=self.atoms)

    def test_all_sites_have_transitions(self):
        """Test correct counts with transitions on every site."""
        self.site0.transitions = Counter({1: 3, 2: 1})
        self.site1.transitions = Counter({0: 2})
        self.site2.transitions = Counter({0: 1, 1: 4})
        result = self.trajectory.transition_counts_by_site()
        self.assertEqual(result.to_dict(), {
            0: {0: 0, 1: 3, 2: 1},
            1: {0: 2, 1: 0, 2: 0},
            2: {0: 1, 1: 4, 2: 0},
        })

    def test_site_with_no_outgoing_transitions(self):
        """Test that a site with no transitions appears as a row of zeros."""
        self.site0.transitions = Counter({1: 5})
        # site1 and site2 have no transitions
        result = self.trajectory.transition_counts_by_site()
        np.testing.assert_array_equal(result.matrix[1], [0, 0, 0])
        np.testing.assert_array_equal(result.matrix[2], [0, 0, 0])

    def test_asymmetric_transitions(self):
        """Test that A->B and B->A are independent entries."""
        self.site0.transitions = Counter({1: 3})
        # site1 has no transition back to site0
        result = self.trajectory.transition_counts_by_site()
        self.assertEqual(result.get(0, 1), 3)
        self.assertEqual(result.get(1, 0), 0)

    def test_no_transitions_at_all(self):
        """Test all zeros when no transitions recorded."""
        result = self.trajectory.transition_counts_by_site()
        np.testing.assert_array_equal(result.matrix, np.zeros((3, 3)))

    def test_self_transitions_are_preserved_if_present(self):
        """Test that pre-populated self-transition keys are preserved.

        Note: Trajectory.append_timestep() does not record self-transitions,
        so these should not appear in normal trajectory data. This test
        verifies that transition_counts_by_site() faithfully reports whatever is
        present in site.transitions without filtering.
        """
        self.site0.transitions = Counter({0: 2, 1: 3})
        result = self.trajectory.transition_counts_by_site()
        self.assertEqual(result.get(0, 0), 2)
        self.assertEqual(result.get(0, 1), 3)

    def test_unknown_destination_index_raises(self):
        """Test that a transition to a non-existent site index raises ValueError."""
        self.site0.transitions = Counter({1: 3, 99: 5})
        with self.assertRaises(ValueError):
            self.trajectory.transition_counts_by_site()

    def test_single_site(self):
        """Test single-site trajectory produces 1x1 zero matrix."""
        Site._newid = 0
        site = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=0.3)
        trajectory = Trajectory(sites=[site], atoms=[Atom(index=0)])
        result = trajectory.transition_counts_by_site()
        self.assertEqual(result.to_dict(), {0: {0: 0}})


class TransitionCountsByLabelTestCase(unittest.TestCase):
    """Tests for Trajectory.transition_counts_by_label()."""

    def setUp(self):
        Site._newid = 0
        # Two sites labelled "A", one labelled "B", one unlabelled
        self.site0 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=0.3, label="A")
        self.site1 = SphericalSite(frac_coords=np.array([0.25, 0.25, 0.25]), rcut=0.3, label="A")
        self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=0.3, label="B")
        self.site3 = SphericalSite(frac_coords=np.array([0.75, 0.75, 0.75]), rcut=0.3)
        self.sites = [self.site0, self.site1, self.site2, self.site3]
        self.atoms = [Atom(index=0)]
        self.trajectory = Trajectory(sites=self.sites, atoms=self.atoms)

    def test_basic_label_aggregation(self):
        """Test that transitions from sites sharing a label are summed."""
        # site0 (A) -> site2 (B): 3 hops
        self.site0.transitions = Counter({2: 3})
        # site1 (A) -> site2 (B): 2 hops
        self.site1.transitions = Counter({2: 2})
        # site2 (B) -> site0 (A): 1 hop
        self.site2.transitions = Counter({0: 1})
        result = self.trajectory.transition_counts_by_label()
        self.assertEqual(result.to_dict(), {
            "A": {"A": 0, "B": 5},
            "B": {"A": 1, "B": 0},
        })

    def test_unlabelled_sites_are_skipped_with_warning(self):
        """Test that transitions to unlabelled sites are excluded with a warning."""
        # unlabelled site3 has transitions
        self.site3.transitions = Counter({0: 10})
        # site0 (A) transitions to unlabelled site3
        self.site0.transitions = Counter({3: 7})
        with self.assertWarns(UserWarning):
            result = self.trajectory.transition_counts_by_label()
        # site3 should not appear; site0's transition to site3 is excluded
        self.assertNotIn(None, result.keys)
        self.assertEqual(result.to_dict(), {
            "A": {"A": 0, "B": 0},
            "B": {"A": 0, "B": 0},
        })

    def test_all_sites_unlabelled(self):
        """Test that all-unlabelled sites produce an empty table."""
        Site._newid = 0
        site = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=0.3)
        trajectory = Trajectory(sites=[site], atoms=[Atom(index=0)])
        result = trajectory.transition_counts_by_label()
        self.assertEqual(result.keys, ())
        self.assertEqual(result.matrix.shape, (0, 0))

    def test_square_output(self):
        """Test that labels appearing only as destinations still appear as keys."""
        # Only site0 (A) has transitions, to site2 (B)
        self.site0.transitions = Counter({2: 1})
        result = self.trajectory.transition_counts_by_label()
        # "B" should appear as a key even though it has no outgoing transitions
        self.assertIn("B", result.keys)
        np.testing.assert_array_equal(result.matrix[1], [0, 0])

    def test_intra_label_transitions(self):
        """Test transitions between sites that share the same label."""
        # site0 (A) -> site1 (A): this is an A->A transition
        self.site0.transitions = Counter({1: 4})
        result = self.trajectory.transition_counts_by_label()
        self.assertEqual(result.get("A", "A"), 4)

    def test_unlabelled_site_with_invalid_destination_raises(self):
        """Test that an unlabelled site transitioning to an invalid index raises."""
        self.site3.transitions = Counter({99: 5})
        with self.assertRaises(ValueError):
            self.trajectory.transition_counts_by_label()


class TransitionProbabilitiesTestCase(unittest.TestCase):
    """Tests for Trajectory.transition_probabilities_by_site() and transition_probabilities_by_label()."""

    def setUp(self):
        Site._newid = 0
        self.site0 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=0.3, label="A")
        self.site1 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=0.3, label="B")
        self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.0, 0.5]), rcut=0.3, label="C")
        self.sites = [self.site0, self.site1, self.site2]
        self.atoms = [Atom(index=0)]
        self.trajectory = Trajectory(sites=self.sites, atoms=self.atoms)

    def test_basic_normalisation(self):
        """Test row-normalised probabilities sum to 1.0."""
        self.site0.transitions = Counter({1: 3, 2: 1})  # total 4
        self.site1.transitions = Counter({0: 2, 2: 2})  # total 4
        result = self.trajectory.transition_probabilities_by_site()
        self.assertAlmostEqual(result.get(0, 1), 0.75)
        self.assertAlmostEqual(result.get(0, 2), 0.25)
        self.assertAlmostEqual(result.get(0, 0), 0.0)
        self.assertAlmostEqual(result.get(1, 0), 0.5)
        self.assertAlmostEqual(result.get(1, 2), 0.5)

    def test_row_with_zero_transitions(self):
        """Test that a row with no transitions remains all zeros."""
        self.site0.transitions = Counter({1: 1})
        # site1 and site2 have no transitions
        result = self.trajectory.transition_probabilities_by_site()
        np.testing.assert_array_equal(result.matrix[1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result.matrix[2], [0.0, 0.0, 0.0])

    def test_single_outgoing_transition(self):
        """Test that a single outgoing transition becomes 1.0."""
        self.site0.transitions = Counter({1: 5})
        result = self.trajectory.transition_probabilities_by_site()
        self.assertAlmostEqual(result.get(0, 1), 1.0)
        self.assertAlmostEqual(result.get(0, 0), 0.0)
        self.assertAlmostEqual(result.get(0, 2), 0.0)

    def test_by_label(self):
        """Test label-level probabilities are correctly normalised."""
        self.site0.transitions = Counter({1: 6, 2: 4})  # A: total 10
        self.site1.transitions = Counter({0: 3})         # B: total 3
        result = self.trajectory.transition_probabilities_by_label()
        self.assertAlmostEqual(result.get("A", "B"), 0.6)
        self.assertAlmostEqual(result.get("A", "C"), 0.4)
        self.assertAlmostEqual(result.get("B", "A"), 1.0)

    def test_zero_transition_label_by_label(self):
        """Test that a label with no outgoing transitions gives all-zero probabilities."""
        self.site0.transitions = Counter({1: 4})  # A -> B
        # site1 (B) and site2 (C) have no transitions
        result = self.trajectory.transition_probabilities_by_label()
        np.testing.assert_array_equal(result.matrix[1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result.matrix[2], [0.0, 0.0, 0.0])


class TransitionCustomKeysTestCase(unittest.TestCase):
    """Tests for custom key ordering on transition methods."""

    def setUp(self):
        Site._newid = 0
        self.site0 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=0.3, label="A")
        self.site1 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=0.3, label="B")
        self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.0, 0.5]), rcut=0.3, label="C")
        self.sites = [self.site0, self.site1, self.site2]
        self.atoms = [Atom(index=0)]
        self.trajectory = Trajectory(sites=self.sites, atoms=self.atoms)

    def test_custom_key_order_by_site(self):
        """Test that custom keys reorder rows and columns for counts."""
        self.site0.transitions = Counter({1: 3, 2: 1})
        self.site1.transitions = Counter({0: 2})
        result = self.trajectory.transition_counts_by_site(keys=[2, 0, 1])
        self.assertEqual(result.keys, (2, 0, 1))
        np.testing.assert_array_equal(result.matrix, np.array([
            [0, 0, 0],
            [1, 0, 3],
            [0, 2, 0],
        ]))

    def test_custom_key_order_by_label(self):
        """Test that custom keys reorder rows and columns for probabilities."""
        self.site0.transitions = Counter({1: 3, 2: 1})
        self.site1.transitions = Counter({0: 2})
        result = self.trajectory.transition_probabilities_by_label(keys=["C", "B", "A"])
        self.assertEqual(result.keys, ("C", "B", "A"))
        np.testing.assert_array_almost_equal(result.matrix, np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.75, 0.0],
        ]))

    def test_custom_key_order_probabilities_by_site(self):
        """Test that custom keys reorder probabilities by site."""
        self.site0.transitions = Counter({1: 3, 2: 1})
        self.site1.transitions = Counter({0: 2})
        result = self.trajectory.transition_probabilities_by_site(keys=[2, 0, 1])
        self.assertEqual(result.keys, (2, 0, 1))
        np.testing.assert_array_almost_equal(result.matrix, np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.75],
            [0.0, 1.0, 0.0],
        ]))

    def test_unknown_key_raises_value_error(self):
        """Test that an unknown key raises ValueError."""
        with self.assertRaises(ValueError):
            self.trajectory.transition_counts_by_label(keys=["A", "Z"])

    def test_subset_keys_raises_value_error(self):
        """Test that passing a subset of keys raises ValueError."""
        with self.assertRaises(ValueError):
            self.trajectory.transition_counts_by_label(keys=["A", "B"])

    def test_default_keys_are_sorted(self):
        """Test that keys are sorted by default."""
        self.site0.transitions = Counter({1: 3})
        result = self.trajectory.transition_counts_by_site()
        self.assertEqual(result.keys, (0, 1, 2))

    def test_default_label_keys_are_sorted(self):
        """Test that label keys are sorted by default."""
        self.site0.transitions = Counter({1: 3})
        result = self.trajectory.transition_counts_by_label()
        self.assertEqual(result.keys, ("A", "B", "C"))


class TrajectoryCommitmentRadiusTestCase(unittest.TestCase):
    """Tests for the commitment_radius argument to Trajectory."""

    def setUp(self):
        Site._newid = 0
        self.sites = [
            SphericalSite(frac_coords=np.array([0.3, 0.5, 0.5]), rcut=1.9, label="a"),
            SphericalSite(frac_coords=np.array([0.7, 0.5, 0.5]), rcut=1.9, label="b"),
        ]
        self.atoms = [Atom(index=0)]

    def test_commitment_off_by_default(self):
        """commitment_radius is None unless given."""
        trajectory = Trajectory(sites=self.sites, atoms=self.atoms)
        self.assertIsNone(trajectory.commitment_radius)

    def test_commitment_radius_is_stored(self):
        """The commitment_radius argument is stored on the trajectory."""
        trajectory = Trajectory(sites=self.sites, atoms=self.atoms,
                                commitment_radius={"a": 0.5, "b": 2.0})
        self.assertEqual(trajectory.commitment_radius, {"a": 0.5, "b": 2.0})

    def test_commitment_radius_is_read_only(self):
        """commitment_radius cannot be reassigned after construction."""
        trajectory = Trajectory(sites=self.sites, atoms=self.atoms,
                                commitment_radius=0.5)
        with self.assertRaises(AttributeError):
            trajectory.commitment_radius = 1.0

    def test_commitment_radius_copies_dict(self):
        """Changing the dict passed in does not change the stored radii."""
        radii = {"a": 0.5, "b": 2.0}
        trajectory = Trajectory(sites=self.sites, atoms=self.atoms,
                                commitment_radius=radii)
        radii["a"] = 9.0
        self.assertEqual(trajectory.commitment_radius, {"a": 0.5, "b": 2.0})

    def test_changing_returned_radii_has_no_effect(self):
        """Changing the dict returned by commitment_radius does not change it."""
        trajectory = Trajectory(sites=self.sites, atoms=self.atoms,
                                commitment_radius={"a": 0.5, "b": 2.0})
        trajectory.commitment_radius["a"] = 9.0
        self.assertEqual(trajectory.commitment_radius, {"a": 0.5, "b": 2.0})

    def test_non_positive_radius_raises(self):
        """Every commitment radius must be positive."""
        for radius in (0.0, -1.0, float("nan"), {"a": 0.5, "b": 0.0}):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    Trajectory(sites=self.sites, atoms=self.atoms,
                               commitment_radius=radius)

    def test_non_numeric_radius_raises(self):
        """A commitment radius must be a number or a dict of numbers."""
        for radius in ([1.0, 2.0], "1.0", np.array([1.0]), True,
                       {"a": "0.5", "b": 1.0}):
            with self.subTest(radius=radius):
                with self.assertRaises(TypeError):
                    Trajectory(sites=self.sites, atoms=self.atoms,
                               commitment_radius=radius)

    def test_infinite_radius_is_allowed(self):
        """An infinite commitment radius is accepted."""
        trajectory = Trajectory(sites=self.sites, atoms=self.atoms,
                                commitment_radius=float("inf"))
        self.assertEqual(trajectory.commitment_radius, float("inf"))

    def test_missing_label_raises(self):
        """A dict of radii must cover every site label."""
        with self.assertRaises(ValueError):
            Trajectory(sites=self.sites, atoms=self.atoms,
                       commitment_radius={"a": 0.5})

    def test_dict_with_unlabelled_site_raises(self):
        """A dict of radii needs every site to have a label."""
        sites = [
            SphericalSite(frac_coords=np.array([0.3, 0.5, 0.5]), rcut=1.9, label="a"),
            SphericalSite(frac_coords=np.array([0.7, 0.5, 0.5]), rcut=1.9),
        ]
        with self.assertRaisesRegex(ValueError, "every site has a label"):
            Trajectory(sites=sites, atoms=self.atoms,
                       commitment_radius={"a": 0.5})


class TrajectoryTransitionCountingTestCase(unittest.TestCase):
    """Tests for transition counting during trajectory analysis."""

    def setUp(self):
        Site._newid = 0
        self.site_a = SphericalSite(frac_coords=np.array([0.25, 0.25, 0.25]), rcut=0.5)
        self.site_b = SphericalSite(frac_coords=np.array([0.75, 0.25, 0.25]), rcut=0.5)
        self.atom = Atom(index=0)
        self.trajectory = Trajectory(sites=[self.site_a, self.site_b], atoms=[self.atom])
        lattice = Lattice.cubic(10.0)
        self.in_a = Structure(lattice, ["Li"], [[0.25, 0.25, 0.25]])
        self.between = Structure(lattice, ["Li"], [[0.5, 0.25, 0.25]])
        self.in_b = Structure(lattice, ["Li"], [[0.75, 0.25, 0.25]])

    def test_transition_recorded_between_appended_timesteps(self):
        """An atom moving A -> B records one A -> B transition (A has index 0)."""
        self.trajectory.trajectory_from_structures([self.in_a, self.in_b])
        self.assertEqual(self.site_a.index, 0)
        self.assertEqual(self.site_a.transitions, {self.site_b.index: 1})

    def test_transition_recorded_across_unassigned_timestep(self):
        """An atom moving A -> (between sites) -> B records one A -> B transition."""
        self.trajectory.trajectory_from_structures([self.in_a, self.between, self.in_b])
        self.assertEqual(self.atom.trajectory,
                         [self.site_a.index, None, self.site_b.index])
        self.assertEqual(self.site_a.transitions, {self.site_b.index: 1})

    def test_no_transition_when_atom_returns_to_same_site(self):
        """A -> A -> (between sites) -> A records no transitions."""
        self.trajectory.trajectory_from_structures(
            [self.in_a, self.in_a, self.between, self.in_a])
        self.assertEqual(self.site_a.transitions, Counter())
        self.assertEqual(self.site_b.transitions, Counter())

    def test_analyse_structure_records_no_transitions(self):
        """Direct analyse_structure calls do not record transitions."""
        self.trajectory.append_timestep(self.in_a)
        self.trajectory.analyse_structure(self.in_b)
        self.trajectory.analyse_structure(self.in_b)
        self.assertEqual(self.site_a.transitions, Counter())
        self.assertEqual(self.site_b.transitions, Counter())

    def test_analyse_structure_does_not_affect_later_transitions(self):
        """A one-off analysis between appended timesteps is ignored for transitions."""
        self.trajectory.analyse_structure(self.in_b)
        self.trajectory.append_timestep(self.in_a)
        self.trajectory.analyse_structure(self.in_b)
        self.trajectory.append_timestep(self.in_a)
        self.assertEqual(self.site_a.transitions, Counter())
        self.assertEqual(self.site_b.transitions, Counter())


class TrajectoryCommitmentTestCase(unittest.TestCase):
    """Tests for spatial commitment during trajectory analysis."""

    def setUp(self):
        Site._newid = 0
        self.site_a = SphericalSite(frac_coords=np.array([0.3, 0.5, 0.5]), rcut=1.9, label="a")
        self.site_b = SphericalSite(frac_coords=np.array([0.7, 0.5, 0.5]), rcut=1.9, label="b")
        self.lattice = Lattice.cubic(10.0)
        # Fractional x positions (y = z = 0.5). *_core is at a site centre.
        # *_edge is inside a site, 1.5 Angstrom from its centre, so outside
        # the default 0.5 Angstrom core. gap is in neither site.
        self.x = {"a_core": 0.30, "a_edge": 0.45, "gap": 0.50,
                  "b_edge": 0.55, "b_core": 0.70}

    def make_trajectory(self, n_atoms=1, commitment_radius=0.5):
        return Trajectory(sites=[self.site_a, self.site_b],
                          atoms=[Atom(index=i) for i in range(n_atoms)],
                          commitment_radius=commitment_radius)

    def frame(self, *positions):
        coords = [[self.x[p], 0.5, 0.5] for p in positions]
        return Structure(self.lattice, ["Li"] * len(positions), coords)

    def test_excursion_outside_core_records_no_transition(self):
        """A -> B outside its core -> A stays in A and records no transition."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("b_edge"), self.frame("a_core")])
        self.assertEqual(trajectory.atoms[0].trajectory, [0, 0, 0])
        self.assertEqual(self.site_a.transitions, Counter())
        self.assertEqual(self.site_b.trajectory, [[], [], []])

    def test_transition_recorded_on_entering_new_core(self):
        """A -> B outside its core -> B core records one A -> B transition."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("b_edge"), self.frame("b_core")])
        self.assertEqual(trajectory.atoms[0].trajectory, [0, 0, 1])
        self.assertEqual(self.site_a.transitions, {1: 1})

    def test_atom_stays_committed_through_gap(self):
        """A -> gap -> B core gives [A, A, B] and one A -> B transition."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("gap"), self.frame("b_core")])
        self.assertEqual(trajectory.atoms[0].trajectory, [0, 0, 1])
        self.assertEqual(self.site_a.transitions, {1: 1})

    def test_first_assignment_commits_outside_core(self):
        """In its first assigned frame an atom commits to its site, core or not."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("b_edge"), self.frame("b_edge")])
        self.assertEqual(trajectory.atoms[0].trajectory, [1, 1])
        self.assertEqual(self.site_b.transitions, Counter())

    def test_points_follow_geometric_site(self):
        """Positions are recorded with the site the atom is in, not its committed site."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("b_edge")])
        self.assertEqual(len(self.site_a.points), 1)
        self.assertEqual(len(self.site_b.points), 1)
        self.assertEqual(self.site_b.trajectory, [[], []])

    def test_site_can_hold_two_atoms(self):
        """An atom in transit and a newly committed atom can share a site."""
        trajectory = self.make_trajectory(n_atoms=2)
        trajectory.trajectory_from_structures(
            [self.frame("a_core", "b_core"), self.frame("b_edge", "a_core")])
        self.assertEqual(self.site_a.trajectory[1], [0, 1])
        self.assertEqual(self.site_b.trajectory[1], [])

    def test_excursion_stays_in_one_residence_run(self):
        """An excursion that does not commit stays inside one residence run."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame(p) for p in
             ("b_core", "a_core", "a_core", "b_edge", "a_core", "b_core")])
        self.assertEqual(trajectory.atoms[0].trajectory, [1, 0, 0, 0, 0, 1])
        self.assertEqual(self.site_a.residence_times(), (4,))

    def test_radius_per_label(self):
        """Each site uses the commitment radius for its label."""
        trajectory = self.make_trajectory(commitment_radius={"a": 0.5, "b": 2.0})
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("b_edge"), self.frame("a_edge")])
        self.assertEqual(trajectory.atoms[0].trajectory, [0, 1, 1])
        self.assertEqual(self.site_a.transitions, {1: 1})

    def test_recent_site_follows_geometric_site(self):
        """An atom in transit has its geometric site as its most recent site."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("b_edge")])
        atom = trajectory.atoms[0]
        self.assertEqual(atom.committed_site, 0)
        self.assertEqual(atom.most_recent_site, 1)

    def test_analyse_structure_is_geometric(self):
        """analyse_structure assigns the geometric site and keeps the committed site."""
        trajectory = self.make_trajectory()
        trajectory.append_timestep(self.frame("a_core"))
        trajectory.analyse_structure(self.frame("b_core"))
        atom = trajectory.atoms[0]
        self.assertEqual(atom.in_site, 1)
        self.assertEqual(atom.committed_site, 0)
        self.assertEqual(self.site_a.transitions, Counter())

    def test_reset_clears_committed_site(self):
        """Trajectory.reset() clears each atom's committed site."""
        trajectory = self.make_trajectory()
        trajectory.append_timestep(self.frame("a_core"))
        trajectory.reset()
        self.assertIsNone(trajectory.atoms[0].committed_site)

    def test_new_trajectory_clears_committed_sites(self):
        """Atoms reused in a new Trajectory start with no committed site."""
        atom = Atom(index=0)
        first = Trajectory(sites=[self.site_a, self.site_b], atoms=[atom],
                           commitment_radius=0.5)
        first.append_timestep(self.frame("a_core"))
        Trajectory(sites=[self.site_a, self.site_b], atoms=[atom],
                   commitment_radius=0.5)
        self.assertIsNone(atom.committed_site)

    def test_atom_unassigned_at_first_commits_on_first_assignment(self):
        """An atom in no site at first commits to the first site it is assigned to."""
        trajectory = self.make_trajectory()
        trajectory.trajectory_from_structures(
            [self.frame("gap"), self.frame("b_edge"), self.frame("b_edge")])
        self.assertEqual(trajectory.atoms[0].trajectory, [None, 1, 1])
        self.assertEqual(self.site_b.transitions, Counter())

    def test_move_through_one_site_into_another_core(self):
        """Committed to A, through B outside its core, into C's core records A -> C."""
        lattice = Lattice.cubic(20.0)
        sites = [SphericalSite(frac_coords=np.array([x, 0.5, 0.5]), rcut=1.9)
                 for x in (0.2, 0.4, 0.6)]
        site_a, site_b, site_c = sites
        trajectory = Trajectory(sites=sites, atoms=[Atom(index=0)],
                                commitment_radius=0.5)
        trajectory.trajectory_from_structures(
            [Structure(lattice, ["Li"], [[x, 0.5, 0.5]]) for x in (0.2, 0.475, 0.6)])
        self.assertEqual(trajectory.atoms[0].trajectory,
                         [site_a.index, site_a.index, site_c.index])
        self.assertEqual(site_a.transitions, {site_c.index: 1})
        self.assertEqual(site_b.transitions, Counter())

    def test_core_reached_across_periodic_boundary(self):
        """The core check uses the minimum-image distance across the cell boundary."""
        site_a = SphericalSite(frac_coords=np.array([0.97, 0.5, 0.5]), rcut=1.9)
        site_b = SphericalSite(frac_coords=np.array([0.40, 0.5, 0.5]), rcut=1.9)
        trajectory = Trajectory(sites=[site_a, site_b], atoms=[Atom(index=0)],
                                commitment_radius=0.6)
        trajectory.trajectory_from_structures([
            Structure(self.lattice, ["Li"], [[0.40, 0.5, 0.5]]),
            Structure(self.lattice, ["Li"], [[0.02, 0.5, 0.5]]),
        ])
        self.assertEqual(trajectory.atoms[0].trajectory, [site_b.index, site_a.index])

    def test_without_commitment_atom_follows_geometric_site(self):
        """Without commitment, A -> B outside its core -> A records two transitions."""
        trajectory = self.make_trajectory(commitment_radius=None)
        trajectory.trajectory_from_structures(
            [self.frame("a_core"), self.frame("b_edge"), self.frame("a_core")])
        self.assertEqual(trajectory.atoms[0].trajectory, [0, 1, 0])
        self.assertEqual(self.site_a.transitions, {1: 1})
        self.assertEqual(self.site_b.transitions, {0: 1})


class PolyhedralCommitmentTestCase(unittest.TestCase):
    """Tests for commitment with polyhedral sites."""

    def setUp(self):
        Site._newid = 0
        # Two tetrahedra sharing the face (v0, v1, v2) in the plane x = 5 Angstrom.
        # Tetrahedron A has apex v3 and tetrahedron B has apex v4; their
        # centres are at x = 4.6 and x = 5.4 Angstrom.
        self.vertices = [(5.0, 5.0, 6.0), (5.0, 4.134, 4.5), (5.0, 5.866, 4.5),
                         (3.4, 5.0, 5.0), (6.6, 5.0, 5.0)]
        self.lattice = Lattice.cubic(10.0)
        self.tet_a = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        self.tet_b = PolyhedralSite(vertex_indices=[0, 1, 2, 4])
        self.trajectory = Trajectory(sites=[self.tet_a, self.tet_b],
                                     atoms=[Atom(index=5)],
                                     commitment_radius=0.2)

    def frame(self, li, shift=(0.0, 0.0, 0.0)):
        coords = [np.add(v, shift) for v in self.vertices] + [np.add(li, shift)]
        return Structure(self.lattice, ["S"] * 5 + ["Li"], coords,
                         coords_are_cartesian=True)

    def test_commitment_uses_current_polyhedron_centre(self):
        """The atom commits to B at B's centre after the whole framework has moved."""
        self.trajectory.trajectory_from_structures([
            self.frame((4.6, 5.0, 5.0)),
            self.frame((5.1, 5.0, 5.0)),
            self.frame((5.4, 5.0, 5.0), shift=(0.0, 1.0, 0.0)),
        ])
        self.assertEqual(self.trajectory.atoms[0].trajectory, [0, 0, 1])
        self.assertEqual(self.tet_a.transitions, {1: 1})


class DynamicVoronoiCommitmentTestCase(unittest.TestCase):
    """Tests for commitment with dynamic Voronoi sites."""

    def setUp(self):
        Site._newid = 0
        # Site A is defined by four reference atoms around (3, 5, 5) Angstrom
        # and site B by four around (7, 5, 5). Their centres are the means of
        # the current reference positions.
        offsets = [(0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
        self.reference = ([np.add((3.0, 5.0, 5.0), d) for d in offsets]
                          + [np.add((7.0, 5.0, 5.0), d) for d in offsets])
        self.lattice = Lattice.cubic(10.0)
        self.site_a = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
        self.site_b = DynamicVoronoiSite(reference_indices=[4, 5, 6, 7])
        self.trajectory = Trajectory(sites=[self.site_a, self.site_b],
                                     atoms=[Atom(index=8)],
                                     commitment_radius=0.2)

    def frame(self, li, shift=(0.0, 0.0, 0.0)):
        coords = [np.add(r, shift) for r in self.reference] + [np.add(li, shift)]
        return Structure(self.lattice, ["S"] * 8 + ["Li"], coords,
                         coords_are_cartesian=True)

    def test_commitment_uses_current_dynamic_centre(self):
        """The atom commits to B at B's centre after the reference atoms have moved."""
        self.trajectory.trajectory_from_structures([
            self.frame((3.0, 5.0, 5.0)),
            self.frame((7.3, 5.0, 5.0)),
            self.frame((7.0, 5.0, 5.0), shift=(0.0, 1.0, 0.0)),
        ])
        self.assertEqual(self.trajectory.atoms[0].trajectory, [0, 0, 1])
        self.assertEqual(self.site_a.transitions, {1: 1})


if __name__ == '__main__':
    unittest.main()
