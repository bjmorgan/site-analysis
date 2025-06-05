import unittest
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.trajectory import Trajectory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.spherical_site import SphericalSite
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import Mock, patch, PropertyMock


class TrajectoryInitializationTestCase(unittest.TestCase):
    """Tests for Trajectory initialization with different site types."""

    def test_initialisation_with_polyhedral_sites(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.trajectory.PolyhedralSiteCollection') as mock_PolyhedralSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(trajectory.atoms, atoms)
        self.assertEqual(trajectory.sites, sites)
        mock_PolyhedralSiteCollection.assert_called_with(sites)
    
    def test_initialisation_with_voronoi_sites(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=VoronoiSite, index=1)]
        with patch('site_analysis.trajectory.VoronoiSiteCollection') as mock_VoronoiSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(trajectory.atoms, atoms)
        self.assertEqual(trajectory.sites, sites)
        mock_VoronoiSiteCollection.assert_called_with(sites)
    
    def test_initialisation_with_spherical_sites(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=SphericalSite, index=1)]
        with patch('site_analysis.trajectory.SphericalSiteCollection') as mock_SphericalSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(trajectory.atoms, atoms)
        self.assertEqual(trajectory.sites, sites)
        mock_SphericalSiteCollection.assert_called_with(sites)
    
    def test_initialisation_with_dynamic_voronoi_sites(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=DynamicVoronoiSite, index=1)]
        with patch('site_analysis.trajectory.DynamicVoronoiSiteCollection') as mock_DynamicVoronoiSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(trajectory.atoms, atoms)
        self.assertEqual(trajectory.sites, sites)
        mock_DynamicVoronoiSiteCollection.assert_called_with(sites)

    def test___len___returns_zero_for_empty_trajectory(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.trajectory.PolyhedralSiteCollection') as mock_PolyhedralSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(len(trajectory), 0)

    def test___len___returns_number_of_timesteps(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.trajectory.PolyhedralSiteCollection') as mock_PolyhedralSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        trajectory.timesteps = ['foo', 'bar']
        self.assertEqual(len(trajectory), 2)

    def test_init_raises_type_error_if_passed_mixed_site_types(self):
        sites = [Mock(spec=PolyhedralSite, index=1),
                 Mock(spec=VoronoiSite, index=2)]
        atoms = [Mock(spec=Atom, index=3)]
        with self.assertRaises(TypeError):
            trajectory = Trajectory(atoms=atoms, sites=sites)

    def test_init_raises_type_error_if_passed_invalid_site_type(self):
        sites = ["foo"]
        atoms = [Mock(spec=Atom, index=3)]
        with self.assertRaises(TypeError):
            trajectory = Trajectory(atoms=atoms, sites=sites)


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
        """Test that assign_site_occupations delegates to site_collection."""
        # Setup
        structure = Mock()
        
        # Mock the site_collection
        self.trajectory.site_collection = Mock()
        
        # Call the method
        self.trajectory.assign_site_occupations(structure)
        
        # Check delegation
        self.trajectory.site_collection.assign_site_occupations.assert_called_once_with(
            self.atoms, structure
        )
    
    def test_trajectory_from_structures_with_progress(self):
        """Test trajectory_from_structures with progress bar."""
        # Create structures
        structures = [self.structure, self.structure2]
        
        # Create a patch for append_timestep to avoid actual processing
        with patch.object(self.trajectory, 'append_timestep') as mock_append, \
             patch('site_analysis.trajectory.tqdm') as mock_tqdm:
            
            # Make tqdm return the original enumerate object for iteration
            mock_tqdm.return_value = enumerate(structures, 1)
            
            # Call with progress=True
            self.trajectory.trajectory_from_structures(structures, progress=True)
            
            # Check tqdm was called with the right arguments
            mock_tqdm.assert_called_once()
            args, kwargs = mock_tqdm.call_args
            self.assertEqual(kwargs['total'], 2)  # Should use the length of structures
            
            # Check append_timestep was called for each structure
            self.assertEqual(mock_append.call_count, 2)
    
    def test_trajectory_from_structures_with_notebook_progress(self):
        """Test trajectory_from_structures with notebook progress bar."""
        # Create structures
        structures = [self.structure, self.structure2]
        
        # Create a patch for append_timestep to avoid actual processing
        with patch.object(self.trajectory, 'append_timestep') as mock_append, \
             patch('site_analysis.trajectory.tqdm_notebook') as mock_tqdm_notebook:
            
            # Make tqdm_notebook return the original enumerate object for iteration
            mock_tqdm_notebook.return_value = enumerate(structures, 1)
            
            # Call with progress='notebook'
            self.trajectory.trajectory_from_structures(structures, progress='notebook')
            
            # Check tqdm_notebook was called with the right arguments
            mock_tqdm_notebook.assert_called_once()
            args, kwargs = mock_tqdm_notebook.call_args
            self.assertEqual(kwargs['total'], 2)  # Should use the length of structures
            
            # Check append_timestep was called for each structure
            self.assertEqual(mock_append.call_count, 2)
            
    def test_init_with_empty_sites(self):
        """Test that Trajectory raises ValueError with empty sites list."""
        atoms = [Mock(spec=Atom)]
        
        with self.assertRaises(ValueError) as context:
            Trajectory(sites=[], atoms=atoms)
        
        self.assertIn("empty sites list", str(context.exception))
    
    def test_init_with_empty_atoms(self):
        """Test that Trajectory raises ValueError with empty atoms list."""
        sites = [Mock(spec=PolyhedralSite)]
        
        with self.assertRaises(ValueError) as context:
            Trajectory(sites=sites, atoms=[])
        
        self.assertIn("empty atoms list", str(context.exception))


if __name__ == '__main__':
    unittest.main()
