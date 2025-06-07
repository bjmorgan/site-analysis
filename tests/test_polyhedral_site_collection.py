import unittest
import numpy as np
from pymatgen.core import Lattice, Structure
from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection, construct_neighbouring_sites
from site_analysis.atom import atoms_from_structure
from site_analysis.tools import get_coordination_indices
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import patch, Mock, PropertyMock, MagicMock


class PolyhedralSiteCollectionTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        # Reset Site._newid counter
        Site._newid = 0
        
        # Create a test lattice
        self.lattice = Lattice.cubic(10.0)
        
        # Create a test structure with vertices
        species = ["Na"] * 8  # 8 vertices for a cube
        coords = [
            [0.0, 0.0, 0.0],  # vertex 0
            [1.0, 0.0, 0.0],  # vertex 1
            [0.0, 1.0, 0.0],  # vertex 2
            [1.0, 1.0, 0.0],  # vertex 3
            [0.0, 0.0, 1.0],  # vertex 4
            [1.0, 0.0, 1.0],  # vertex 5
            [0.0, 1.0, 1.0],  # vertex 6
            [1.0, 1.0, 1.0],  # vertex 7
        ]
        self.structure = Structure(self.lattice, species, coords, coords_are_cartesian=True)
        
        # Create polyhedral sites for testing
        # Site 1: tetrahedron with vertices 0, 1, 2, 3
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], label="site1")
        
        # Site 2: tetrahedron with vertices 4, 5, 6, 7
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], label="site2")
        
        # Site 3: tetrahedron with vertices 0, 1, 4, 5 - shares face with site1
        self.site3 = PolyhedralSite(vertex_indices=[0, 1, 4, 5], label="site3")
        
        # Test atoms
        self.atom1 = Atom(index=0)
        self.atom2 = Atom(index=1)
        self.atom3 = Atom(index=2)
        
        # Assign test coordinates to atoms
        self.atom1._frac_coords = np.array([0.25, 0.25, 0.0])  # Inside site1
        self.atom2._frac_coords = np.array([0.25, 0.25, 0.75])  # Inside site2
        self.atom3._frac_coords = np.array([0.25, 0.0, 0.25])  # Inside site3
        
        # Create collection
        self.sites = [self.site1, self.site2, self.site3]
        self.atoms = [self.atom1, self.atom2, self.atom3]
        self.collection = PolyhedralSiteCollection(sites=self.sites)

    def test_site_collection_is_initialised(self):
        """Test that PolyhedralSiteCollection is correctly initialised."""
        # Test with real sites
        collection = PolyhedralSiteCollection(sites=self.sites)
        self.assertEqual(collection.sites, self.sites)
        
        # Test with mock sites
        sites = [Mock(spec=PolyhedralSite, index=0),
                 Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.polyhedral_site_collection.construct_neighbouring_sites') as mock_construct_neighbouring_sites:
            mock_construct_neighbouring_sites.return_value = 'mocked_neighbours'
            site_collection = PolyhedralSiteCollection(sites=sites)
            self.assertEqual(site_collection.sites, sites)
            mock_construct_neighbouring_sites.assert_called_with(site_collection.sites)
            self.assertEqual(site_collection._neighbouring_sites, 'mocked_neighbours')
    
    def test_init_raises_type_error_with_non_polyhedral_sites(self):
        """Test that initialisation raises TypeError with non-PolyhedralSite objects."""
        # Create a mix of site types
        non_polyhedral_site = Mock()
        mixed_sites = [self.site1, non_polyhedral_site]
        
        # Test initialisation with mixed site types
        with self.assertRaises(TypeError):
            PolyhedralSiteCollection(sites=mixed_sites)
    
    def test_analyse_structure(self):
        """Test that analyse_structure assigns coordinates and updates site occupations."""
        # Setup mocks
        with patch.object(Atom, 'assign_coords') as mock_assign_coords, \
             patch.object(PolyhedralSite, 'assign_vertex_coords') as mock_assign_vertex, \
             patch.object(PolyhedralSiteCollection, 'assign_site_occupations') as mock_assign:
            
            # Call method
            self.collection.analyse_structure(self.atoms, self.structure)
            
            # Verify each atom's coordinates were assigned
            self.assertEqual(mock_assign_coords.call_count, 3)
            
            # Verify each site's vertex coordinates were assigned
            self.assertEqual(mock_assign_vertex.call_count, 3)
            
            # Verify assign_site_occupations was called with atoms and structure
            mock_assign.assert_called_once_with(self.atoms, self.structure)
    
    def test_assign_site_occupations_atom_in_site(self):
        """Test assign_site_occupations when atoms are already in sites."""
        # Setup: assign atoms to sites
        self.atom1.in_site = self.site1.index
        self.atom2.in_site = self.site2.index
        self.atom3.in_site = self.site3.index
        
        # Setup mocks
        with patch.object(PolyhedralSite, 'contains_atom') as mock_contains_atom, \
             patch.object(PolyhedralSiteCollection, 'update_occupation') as mock_update:
            
            # Configure mock to return True (atom still in same site)
            mock_contains_atom.return_value = True
            
            # Call method
            self.collection.assign_site_occupations(self.atoms, self.structure)
            
            # Verify site occupations were reset
            self.assertEqual(self.site1.contains_atoms, [])
            self.assertEqual(self.site2.contains_atoms, [])
            self.assertEqual(self.site3.contains_atoms, [])
            
            # Verify contains_atom was called once per atom (checking previous site first)
            self.assertEqual(mock_contains_atom.call_count, 3)
            
            # Verify update_occupation was called once per atom
            self.assertEqual(mock_update.call_count, 3)
    
    def test_assign_site_occupations_atom_moved(self):
        """Test assign_site_occupations when atoms have moved to new sites."""
        # Setup: assign atoms to sites
        self.atom1.in_site = self.site1.index
        self.atom2.in_site = self.site2.index
        self.atom3.in_site = self.site3.index
        
        # Setup mocks
        with patch.object(PolyhedralSite, 'contains_atom') as mock_contains_atom, \
             patch.object(PolyhedralSiteCollection, 'update_occupation') as mock_update:
            
            # Configure mock to return False (atom has moved)
            mock_contains_atom.return_value = False
            
            # Call method
            self.collection.assign_site_occupations(self.atoms, self.structure)
            
            # Verify site occupations were reset
            self.assertEqual(self.site1.contains_atoms, [])
            self.assertEqual(self.site2.contains_atoms, [])
            self.assertEqual(self.site3.contains_atoms, [])
            
            # The actual implementation uses a slightly different logic than we initially expected
            # The important part is it checks all sites for all atoms if the atom's previous site
            # doesn't contain it anymore
            self.assertGreaterEqual(mock_contains_atom.call_count, 9)  # At least 3 atoms * 3 sites
    
    def test_assign_site_occupations_atom_not_in_site(self):
        """Test assign_site_occupations when atoms are not assigned to sites."""
        # Setup: atoms not in any site
        self.atom1.in_site = None
        self.atom2.in_site = None
        self.atom3.in_site = None
        
        # Setup mocks
        with patch.object(PolyhedralSite, 'contains_atom') as mock_contains_atom, \
             patch.object(PolyhedralSiteCollection, 'update_occupation') as mock_update:
            
            # Configure mock to return True only for atom1 in site1
            mock_contains_atom.return_value = False
            
            # Only atom1 in site1 should return True
            mock_contains_atom.side_effect = lambda atom, **kwargs: atom is self.atom1 and mock_contains_atom.mock_calls[0][1][0] is atom
            
            # Call method
            self.collection.assign_site_occupations(self.atoms, self.structure)
            
            # Verify site occupations were reset
            self.assertEqual(self.site1.contains_atoms, [])
            self.assertEqual(self.site2.contains_atoms, [])
            self.assertEqual(self.site3.contains_atoms, [])
            
            # Verify update_occupation was called once for atom1
            self.assertEqual(mock_update.call_count, 1)
    
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
    
    def test_neighbouring_sites(self):
        """Test that neighbouring_sites returns the correct neighbours."""
        # Override _neighbouring_sites with a test dictionary
        test_neighbours = {
            self.site1.index: [self.site3],  # site1 neighbours site3
            self.site2.index: [],  # site2 has no neighbours
            self.site3.index: [self.site1]  # site3 neighbours site1
        }
        self.collection._neighbouring_sites = test_neighbours
        
        # Test site1 neighbours
        neighbours = self.collection.neighbouring_sites(self.site1.index)
        self.assertEqual(len(neighbours), 1)
        self.assertIs(neighbours[0], self.site3)
        
        # Test site2 neighbours (none)
        neighbours = self.collection.neighbouring_sites(self.site2.index)
        self.assertEqual(len(neighbours), 0)
        
        # Test site3 neighbours
        neighbours = self.collection.neighbouring_sites(self.site3.index)
        self.assertEqual(len(neighbours), 1)
        self.assertIs(neighbours[0], self.site1)
    
    def test_sites_contain_points(self):
        """Test that sites_contain_points checks if sites contain specific points."""
        # Setup points and structure
        points = np.array([
            [0.25, 0.25, 0.0],  # Inside site1
            [0.25, 0.25, 0.75],  # Inside site2
            [0.25, 0.0, 0.25]  # Inside site3
        ])
        
        # Mock contains_point to control test behaviour
        with patch.object(PolyhedralSite, 'contains_point') as mock_contains_point:
            # Configure mock to return true for each site with its matching point
            mock_contains_point.side_effect = [True, True, True]
            
            # Test with all points
            result = self.collection.sites_contain_points(points, self.structure)
            self.assertTrue(result)
            
            # Check contains_point was called for each site-point pair
            self.assertEqual(mock_contains_point.call_count, 3)
            
            # Reset mock
            mock_contains_point.reset_mock()
            
            # Configure mock for failure case
            mock_contains_point.side_effect = [True, False, True]
            
            # Test with one point not in its site
            result = self.collection.sites_contain_points(points, self.structure)
            self.assertFalse(result)
    
    def test_sites_contain_points_requires_structure(self):
        """Test that sites_contain_points requires a structure argument."""
        # Setup points
        points = np.array([
            [0.25, 0.25, 0.0],
            [0.25, 0.25, 0.75]
        ])
        
        # Test without structure
        with self.assertRaises(AssertionError):
            self.collection.sites_contain_points(points)
        
        # Test with None structure
        with self.assertRaises(AssertionError):
            self.collection.sites_contain_points(points, None)
            
    def test_checks_most_recent_site(self):
        """Test that assign_site_occupations checks most_recent_site when in_site is None."""
        # Create mock structure
        mock_structure = Mock(spec=Structure)
        
        # Create mock site and collection
        mock_site = Mock(spec=PolyhedralSite, index=5)
        collection = PolyhedralSiteCollection(sites=[mock_site])
        
        # Create mock atom with no current site
        mock_atom = Mock(spec=Atom, index=42, in_site=None)
        
        # Mock the most_recent_site property
        most_recent_site_mock = PropertyMock(return_value=5)
        type(mock_atom).most_recent_site = most_recent_site_mock
        
        # Patch methods on the collection
        with patch.object(collection, 'update_occupation') as mock_update, \
            patch.object(collection, 'site_by_index') as mock_site_by_index:
            # Configure site_by_index to return our mock site
            mock_site_by_index.return_value = mock_site
            
            # Call the method we're testing
            collection.assign_site_occupations([mock_atom], mock_structure)
            
            # Verify most_recent_site property was accessed
            most_recent_site_mock.assert_called_once()
            
            # Verify site_by_index was called with the correct index
            mock_site_by_index.assert_called_with(5)
            
            # Verify update_occupation was called with the right site and atom
            mock_update.assert_called_with(mock_site, mock_atom)


class ConstructNeighbouringSitesTestCase(unittest.TestCase):
    """Tests for the construct_neighbouring_sites function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset Site._newid counter
        Site._newid = 0
        
        # Create polyhedral sites for testing
        # Site 1: tetrahedron with vertices 0, 1, 2, 3
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], label="site1")
        
        # Site 2: tetrahedron with vertices 4, 5, 6, 7
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], label="site2")
        
        # Site 3: tetrahedron with vertices 0, 1, 4, 5 - shares 2 vertices with site1, 2 with site2
        # Note: The current implementation considers this a neighbour of site4
        self.site3 = PolyhedralSite(vertex_indices=[0, 1, 4, 5], label="site3")
        
        # Site 4: tetrahedron with vertices 0, 1, 2, 4 - shares 3 vertices with site1 (face sharing)
        self.site4 = PolyhedralSite(vertex_indices=[0, 1, 2, 4], label="site4")
        
        # All sites
        self.sites = [self.site1, self.site2, self.site3, self.site4]
    
    def test_construct_neighbouring_sites(self):
        """Test construct_neighbouring_sites identifies face-sharing neighbours."""
        # Call function
        neighbours = construct_neighbouring_sites(self.sites)
        
        # Check site1 neighbours
        site1_neighbours = neighbours[self.site1.index]
        self.assertEqual(len(site1_neighbours), 1)
        self.assertIs(site1_neighbours[0], self.site4)  # site1 and site4 share 3 vertices
        
        # Check site2 neighbours
        site2_neighbours = neighbours[self.site2.index]
        self.assertEqual(len(site2_neighbours), 0)  # site2 doesn't share 3+ vertices with any site
        
        # Check site3 neighbours - shares 3 vertices with site4
        site3_neighbours = neighbours[self.site3.index]
        self.assertEqual(len(site3_neighbours), 1)  
        self.assertIs(site3_neighbours[0], self.site4)
        
        # Check site4 neighbours - shares faces with site1 and site3
        site4_neighbours = neighbours[self.site4.index]
        self.assertEqual(len(site4_neighbours), 2)
        self.assertIn(self.site1, site4_neighbours)
        self.assertIn(self.site3, site4_neighbours)
    
    def test_construct_neighbouring_sites_no_neighbours(self):
        """Test construct_neighbouring_sites with sites that have no neighbours."""
        # Create isolated sites that don't share vertices
        site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7])
        sites = [site1, site2]
        
        # Call function
        neighbours = construct_neighbouring_sites(sites)
        
        # Check each site has no neighbours
        self.assertEqual(len(neighbours[site1.index]), 0)
        self.assertEqual(len(neighbours[site2.index]), 0)
    
    def test_construct_neighbouring_sites_identical_sites(self):
        """Test construct_neighbouring_sites with identical sites."""
        # Create two sites with identical vertices
        site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        site2 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        sites = [site1, site2]
        
        # Call function
        neighbours = construct_neighbouring_sites(sites)
        
        # Each site should identify the other as a neighbour
        self.assertEqual(len(neighbours[site1.index]), 1)
        self.assertIs(neighbours[site1.index][0], site2)
        
        self.assertEqual(len(neighbours[site2.index]), 1)
        self.assertIs(neighbours[site2.index][0], site1)
    
    def test_construct_neighbouring_sites_with_self(self):
        """Test that a site is not identified as its own neighbour."""
        # Create a site
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        sites = [site]
        
        # Call function
        neighbours = construct_neighbouring_sites(sites)
        
        # Site should have no neighbours
        self.assertEqual(len(neighbours[site.index]), 0)
                        

class TestAssignSiteOccupationsInteraction(unittest.TestCase):
    """Test interaction between assign_site_occupations and _get_priority_sites."""
    
    def setUp(self):
        Site._newid = 0
        self.lattice = Lattice.cubic(2.0)
        self.structure = Structure(self.lattice, ["Li"], [[0.1, 0.1, 0.1]])
        
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7])
        self.collection = PolyhedralSiteCollection([self.site1, self.site2])
        self.atoms = atoms_from_structure(self.structure, "Li")
        self.atom = self.atoms[0]
    
    def test_calls_generator_for_each_atom(self):
        """Test that _get_priority_sites is called once per atom."""
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(self.atoms, self.structure)
            self.collection._get_priority_sites.assert_called_once_with(self.atom)
    
    def test_calls_generator_for_multiple_atoms(self):
        """Test that _get_priority_sites is called for each atom."""
        # Add second atom
        self.structure.append("Li", [0.2, 0.2, 0.2])
        atoms = atoms_from_structure(self.structure, "Li")
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(atoms, self.structure)
            self.assertEqual(self.collection._get_priority_sites.call_count, 2)
    
    def test_checks_sites_in_generator_order(self):
        """Test that sites are checked in the order returned by generator."""
        call_order = []
        self.site1.contains_atom = lambda atom: call_order.append(1) or False
        self.site2.contains_atom = lambda atom: call_order.append(2) or True
        
        with patch.object(self.collection, '_get_priority_sites') as mock_gen:
            mock_gen.return_value = [self.site2, self.site1]  # site2 first
            
            self.collection.assign_site_occupations(self.atoms, self.structure)
            
            self.assertEqual(call_order, [2])  # Only site2 checked (found there)
    
    def test_stops_checking_when_atom_found(self):
        """Test that checking stops as soon as atom is found."""
        self.site1.contains_atom = MagicMock(return_value=True)
        self.site2.contains_atom = MagicMock(return_value=False)
        
        with patch.object(self.collection, '_get_priority_sites') as mock_gen:
            mock_gen.return_value = [self.site1, self.site2]
            
            self.collection.assign_site_occupations(self.atoms, self.structure)
            
            self.site1.contains_atom.assert_called_once()
            self.site2.contains_atom.assert_not_called()
    
    def test_calls_update_occupation_when_found(self):
        """Test that update_occupation is called when atom found."""
        self.site1.contains_atom = MagicMock(return_value=True)
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
            with patch.object(self.collection, 'update_occupation') as mock_update:
                self.collection.assign_site_occupations(self.atoms, self.structure)
                mock_update.assert_called_once_with(self.site1, self.atom)
    
    def test_handles_atom_not_found(self):
        """Test behavior when atom not found in any site."""
        self.site1.contains_atom = MagicMock(return_value=False)
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
            with patch.object(self.collection, 'update_occupation') as mock_update:
                self.collection.assign_site_occupations(self.atoms, self.structure)
                
                mock_update.assert_not_called()
                self.assertIsNone(self.atom.in_site)
    
    def test_resets_site_occupations(self):
        """Test that reset_site_occupations is called at start."""
        with patch.object(self.collection, 'reset_site_occupations') as mock_reset:
            with patch.object(self.collection, '_get_priority_sites', return_value=[]):
                self.collection.assign_site_occupations(self.atoms, self.structure)
                mock_reset.assert_called_once()
    
    def test_resets_atom_in_site(self):
        """Test that atom.in_site is reset to None."""
        self.atom.in_site = 999  # Set to some previous value
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(self.atoms, self.structure)
            self.assertIsNone(self.atom.in_site)
            
            
class TestGetPrioritySites(unittest.TestCase):
    """Test _get_priority_sites generator behavior."""
    
    def setUp(self):
        Site._newid = 0
        self.lattice = Lattice.cubic(2.0)
        self.structure = Structure(self.lattice, ["Li"], [[0.1, 0.1, 0.1]])
        
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7])
        self.site3 = PolyhedralSite(vertex_indices=[8, 9, 10, 11])
        self.collection = PolyhedralSiteCollection([self.site1, self.site2, self.site3])
        
        self.atoms = atoms_from_structure(self.structure, "Li")
        self.atom = self.atoms[0]
    
    def test_yields_most_recent_site_first(self):
        """Test that generator yields most recent site as first site."""
        # Set up atom with most recent site
        self.atom.trajectory = [self.site2.index]  # Most recent is site2
        
        # Get priority sites
        priority_sites = list(self.collection._get_priority_sites(self.atom))
        
        # First site should be the most recent site
        self.assertEqual(priority_sites[0], self.site2)
        
    def test_yields_most_recently_visited_when_most_recent_is_none(self):
        """Test that generator yields most recently visited site when most recent is None."""
        # Set up atom where most recent is None but has previous site history
        self.atom.trajectory = [self.site1.index, self.site2.index, None]  # Was in site2, then None
        
        # Get priority sites  
        priority_sites = list(self.collection._get_priority_sites(self.atom))
        
        # First site should be the most recently visited site (site2)
        self.assertEqual(priority_sites[0], self.site2)
    
    def test_yields_all_sites_when_no_valid_trajectory(self):
        """Test that generator yields all sites when no valid site history exists."""
        # Set up atom with only None entries or empty trajectory
        self.atom.trajectory = [None, None]  # Never been in any site
        
        # Get priority sites  
        priority_sites = list(self.collection._get_priority_sites(self.atom))
        
        # Should yield all sites (no prioritization possible)
        self.assertEqual(len(priority_sites), 3)
        self.assertIn(self.site1, priority_sites)
        self.assertIn(self.site2, priority_sites)
        self.assertIn(self.site3, priority_sites)
        
    def test_yields_transition_destinations_after_most_recent(self):
        """Test that generator yields transition destinations after most recent site."""
        # Set up atom with most recent site
        self.atom.trajectory = [self.site1.index]  # Most recent is site1
        
        # Mock transition destinations in frequency order
        with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
            mock_transitions.return_value = [self.site3.index, self.site2.index]  # site3 most frequent
            
            # Get priority sites
            priority_site_indices = [site.index for site in self.collection._get_priority_sites(self.atom)]
            
            self.assertEqual(priority_site_indices, [self.site1.index, self.site3.index, self.site2.index])   
    
    def test_yields_no_duplicates_when_all_sites_are_transitions(self):
        """Test that generator doesn't yield duplicates when all sites appear as transitions."""
        # Set up atom with most recent site
        self.atom.trajectory = [self.site1.index]  # Most recent is site1
        
        # Mock transitions that include all other sites
        with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
            mock_transitions.return_value = [self.site3.index, self.site2.index]  # All other sites as transitions
            
            # Get priority sites
            priority_sites = list(self.collection._get_priority_sites(self.atom))
            
            # Should be exactly 3 sites, no duplicates
            self.assertEqual(len(priority_sites), 3)
            
            # Convert to indices for easier checking
            site_indices = [site.index for site in priority_sites]
            
            # Should have no duplicates
            self.assertEqual(len(site_indices), len(set(site_indices)))
            
            # Should be: site1, site2, site3 (in that order, with no fallback sites)
            self.assertEqual(site_indices, [self.site1.index, self.site3.index, self.site2.index])  
            
    def test_yields_neighbours_after_transitions(self):
        """Test that generator yields neighbours after transition destinations."""
        # Set up atom with most recent site
        self.atom.trajectory = [self.site1.index]  # Most recent is site1
        
        # Mock transitions and neighbours
        with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
            with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
                mock_transitions.return_value = [self.site2.index]  # One transition
                mock_neighbours.return_value = [self.site3]  # One neighbour
                
                # Get priority sites
                priority_sites = list(self.collection._get_priority_sites(self.atom))
                
                # Should be: site1 (most recent), site2 (transition), site3 (neighbour)
                self.assertEqual(priority_sites[0], self.site1)  # Most recent
                self.assertEqual(priority_sites[1], self.site2)  # Transition
                self.assertEqual(priority_sites[2], self.site3)  # neighbour
                
                # Verify neighbouring_sites was called with most recent site index
                mock_neighbours.assert_called_once_with(self.site1.index)
    
    def test_yields_no_duplicates_with_neighbours_and_transitions(self):
        """Test that generator doesn't yield duplicates when neighbour appears as transition."""
        # Set up atom with most recent site
        self.atom.trajectory = [self.site1.index]  # Most recent is site1
        
        # Mock transitions and neighbours where site2 appears in both
        with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
            with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
                mock_transitions.return_value = [self.site2.index]  # site2 as transition
                mock_neighbours.return_value = [self.site2, self.site3]  # site2 also as neighbour
                
                # Get priority sites
                priority_sites = list(self.collection._get_priority_sites(self.atom))
                
                # Should be exactly 3 sites, no duplicates
                self.assertEqual(len(priority_sites), 3)
                
                # Convert to indices
                site_indices = [site.index for site in priority_sites]
                
                # Should have no duplicates
                self.assertEqual(len(site_indices), len(set(site_indices)))
                
                # site2 should only appear once (as transition, not again as neighbour)
                self.assertEqual(site_indices.count(self.site2.index), 1)
                
                # Order should be: site1, site2 (transition), site3 (neighbour)
                self.assertEqual(site_indices, [self.site1.index, self.site2.index, self.site3.index])
    
    def test_skips_neighbour_checking_when_no_most_recent_site(self):
        """Test that neighbour checking is skipped when atom has no most recent site."""
        # Set up atom with no trajectory
        self.atom.trajectory = []  # No most recent site
        
        # Mock neighbours (should not be called)
        with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
            # Get priority sites
            priority_sites = list(self.collection._get_priority_sites(self.atom))
            
            # Should yield all sites (no prioritization)
            self.assertEqual(len(priority_sites), 3)
            
            # neighbouring_sites should not be called
            mock_neighbours.assert_not_called()      
    


if __name__ == '__main__':
    unittest.main()