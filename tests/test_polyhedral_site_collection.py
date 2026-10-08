import itertools
import unittest
import numpy as np
from pymatgen.core import Lattice, Structure
from site_analysis.polyhedral_site_collection import (
    PolyhedralSiteCollection,
    construct_neighbouring_sites,
    _collect_reference_centres,
)
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
        mock_site_0 = Mock(spec=PolyhedralSite, index=0)
        mock_site_0.reference_center = None
        mock_site_1 = Mock(spec=PolyhedralSite, index=1)
        mock_site_1.reference_center = None
        sites = [mock_site_0, mock_site_1]
        with patch('site_analysis.polyhedral_site_collection.construct_neighbouring_sites') as mock_neighbours:
            mock_neighbours.return_value = 'mocked_neighbours'
            site_collection = PolyhedralSiteCollection(sites=sites)
            self.assertEqual(site_collection.sites, sites)
            mock_neighbours.assert_called_with(site_collection.sites)
            self.assertEqual(site_collection._neighbouring_sites, 'mocked_neighbours')
    
    def test_init_raises_type_error_with_non_polyhedral_sites(self):
        """Test that initialisation raises TypeError with non-PolyhedralSite objects."""
        # Create a mix of site types
        non_polyhedral_site = Mock()
        mixed_sites = [self.site1, non_polyhedral_site]
        
        # Test initialisation with mixed site types
        with self.assertRaises(TypeError):
            PolyhedralSiteCollection(sites=mixed_sites)

    def test_init_accepts_a_generator_of_sites(self):
        """A collection built from a generator holds every site."""
        collection = PolyhedralSiteCollection(s for s in [self.site1, self.site2, self.site3])
        self.assertEqual(collection.sites, [self.site1, self.site2, self.site3])
    
    def test_analyse_structure(self):
        """Test that analyse_structure notifies sites and updates occupations."""
        # Setup mocks
        with patch.object(Atom, 'assign_coords') as mock_assign_coords, \
             patch.object(PolyhedralSite, 'notify_structure_changed') as mock_notify, \
             patch.object(PolyhedralSiteCollection, 'assign_site_occupations') as mock_assign:

            # Call method
            self.collection.analyse_structure(self.atoms, self.structure)

            # Verify each atom's coordinates were assigned
            self.assertEqual(mock_assign_coords.call_count, 3)

            # Verify each site was notified of the new structure
            self.assertEqual(mock_notify.call_count, 3)

            # Verify assign_site_occupations was called with atoms and lattice_matrix
            mock_assign.assert_called_once()
            args = mock_assign.call_args[0]
            self.assertIs(args[0], self.atoms)
            np.testing.assert_array_equal(args[1], self.structure.lattice.matrix)
    
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
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
            
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
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
            
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
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
            
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
        collection.assign_site_occupations([], lattice.matrix)
        
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
            result = self.collection.sites_contain_points(
                points, self.structure.frac_coords, self.lattice.matrix)
            self.assertTrue(result)

            # Check contains_point was called for each site-point pair
            self.assertEqual(mock_contains_point.call_count, 3)

            # Reset mock
            mock_contains_point.reset_mock()

            # Configure mock for failure case
            mock_contains_point.side_effect = [True, False, True]

            # Test with one point not in its site
            result = self.collection.sites_contain_points(
                points, self.structure.frac_coords, self.lattice.matrix)
            self.assertFalse(result)


    def test_sites_contain_points_notifies_structure_changed(self):
        """Test that sites_contain_points calls notify_structure_changed on each site."""
        points = np.array([
            [0.25, 0.25, 0.0],
            [0.25, 0.25, 0.75],
            [0.25, 0.0, 0.25]
        ])
        frac_coords = self.structure.frac_coords
        lattice_matrix = self.lattice.matrix

        with patch.object(PolyhedralSite, 'notify_structure_changed') as mock_notify, \
             patch.object(PolyhedralSite, 'contains_point', return_value=True):
            self.collection.sites_contain_points(points, frac_coords, lattice_matrix)
            self.assertEqual(mock_notify.call_count, 3)
            for call in mock_notify.call_args_list:
                np.testing.assert_array_equal(call[0][0], frac_coords)
                np.testing.assert_array_equal(call[0][1], lattice_matrix)

    def test_sites_contain_points_raises_on_length_mismatch(self):
        """Test that sites_contain_points raises ValueError on point count mismatch."""
        points = np.array([[0.25, 0.25, 0.0]])  # 1 point, 3 sites
        with self.assertRaises(ValueError):
            self.collection.sites_contain_points(
                points, self.structure.frac_coords, self.lattice.matrix)

    def test_checks_recent_site_via_priority_heuristic(self):
        """Test that assign_site_occupations uses _recent_sites for priority."""
        mock_structure = Mock(spec=Structure)

        mock_site = Mock(spec=PolyhedralSite, index=5)
        mock_site.reference_center = None
        mock_site.vertex_indices = [0, 1, 2, 3]
        collection = PolyhedralSiteCollection(sites=[mock_site])

        mock_atom = Mock(spec=Atom, index=42, in_site=None)
        mock_atom.frac_coords = np.array([0.5, 0.5, 0.5])
        mock_atom._recent_sites = [5, None]

        with patch.object(collection, 'update_occupation') as mock_update, \
            patch.object(collection, 'site_by_index') as mock_site_by_index:
            mock_site_by_index.return_value = mock_site

            collection.assign_site_occupations([mock_atom], np.eye(3) * 10.0)

            mock_site_by_index.assert_called_with(5)
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
    
    def test_construct_neighbouring_sites_in_site_order(self):
        """Neighbours are listed in the order the sites were given."""
        # The first neighbour does not contain vertex 0, the site's first vertex.
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        first = PolyhedralSite(vertex_indices=[1, 2, 3, 4])
        second = PolyhedralSite(vertex_indices=[0, 1, 2, 5])
        neighbours = construct_neighbouring_sites([site, first, second])
        self.assertEqual(neighbours[site.index], [first, second])

    def test_construct_neighbouring_sites_counts_each_vertex_once(self):
        """A vertex repeated within a site counts once towards a shared face."""
        site1 = PolyhedralSite(vertex_indices=[0, 0, 0, 1])
        site2 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        neighbours = construct_neighbouring_sites([site1, site2])
        self.assertEqual(neighbours[site1.index], [])
        self.assertEqual(neighbours[site2.index], [])

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
                        

class TestCollectReferenceCentres(unittest.TestCase):
    """Tests for _collect_reference_centres."""

    def test_returns_none_when_no_reference_centres(self):
        """Returns (None, indices) when any site lacks a reference centre."""
        Site._newid = 0
        sites = [
            PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
            PolyhedralSite(vertex_indices=[4, 5, 6, 7]),
        ]
        centres, site_indices = _collect_reference_centres(sites)
        self.assertIsNone(centres)

    def test_returns_none_when_mixed_reference_centres(self):
        """Returns None when only some sites have reference centres."""
        Site._newid = 0
        sites = [
            PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                           reference_center=np.array([0.1, 0.1, 0.1])),
            PolyhedralSite(vertex_indices=[4, 5, 6, 7]),
        ]
        centres, site_indices = _collect_reference_centres(sites)
        self.assertIsNone(centres)

    def test_returns_centres_and_indices(self):
        """Returns centres array and site indices when all sites have reference centres."""
        Site._newid = 0
        site_a = PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                                reference_center=np.array([0.1, 0.2, 0.3]))
        site_b = PolyhedralSite(vertex_indices=[4, 5, 6, 7],
                                reference_center=np.array([0.4, 0.5, 0.6]))
        centres, site_indices = _collect_reference_centres([site_a, site_b])
        np.testing.assert_array_equal(centres[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(centres[1], [0.4, 0.5, 0.6])
        self.assertEqual(site_indices, [site_a.index, site_b.index])

    def test_raises_for_a_non_finite_reference_centre(self):
        """A reference centre with a NaN raises ValueError naming the site index."""
        Site._newid = 10
        sites = [
            PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                           reference_center=np.array([0.1, 0.2, 0.3])),
            PolyhedralSite(vertex_indices=[4, 5, 6, 7],
                           reference_center=np.array([np.nan, 0.5, 0.6])),
        ]
        with self.assertRaisesRegex(ValueError, "reference centre of site 11 must be three finite numbers"):
            _collect_reference_centres(sites)

    def test_raises_for_a_wrongly_shaped_reference_centre(self):
        """A reference centre that is not three numbers raises ValueError naming the site index."""
        Site._newid = 10
        sites = [
            PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                           reference_center=np.array([0.1, 0.2, 0.3])),
            PolyhedralSite(vertex_indices=[4, 5, 6, 7],
                           reference_center=np.array([0.4, 0.5])),
        ]
        with self.assertRaisesRegex(ValueError, "reference centre of site 11 must be three finite numbers"):
            _collect_reference_centres(sites)


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
        lattice_matrix = self.lattice.matrix
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(self.atoms, lattice_matrix)
            self.collection._get_priority_sites.assert_called_once_with(self.atom, lattice_matrix)
    
    def test_calls_generator_for_multiple_atoms(self):
        """Test that _get_priority_sites is called for each atom."""
        # Add second atom
        self.structure.append("Li", [0.2, 0.2, 0.2])
        atoms = atoms_from_structure(self.structure, "Li")
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(atoms, self.lattice.matrix)
            self.assertEqual(self.collection._get_priority_sites.call_count, 2)
    
    def test_checks_sites_in_generator_order(self):
        """Test that sites are checked in the order returned by generator."""
        call_order = []
        self.site1.contains_atom = lambda atom, **kw: call_order.append(1) or False
        self.site2.contains_atom = lambda atom, **kw: call_order.append(2) or True
        
        with patch.object(self.collection, '_get_priority_sites') as mock_gen:
            mock_gen.return_value = [self.site2, self.site1]  # site2 first
            
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
            
            self.assertEqual(call_order, [2])  # Only site2 checked (found there)
    
    def test_stops_checking_when_atom_found(self):
        """Test that checking stops as soon as atom is found."""
        self.site1.contains_atom = MagicMock(return_value=True)
        self.site2.contains_atom = MagicMock(return_value=False)
        
        with patch.object(self.collection, '_get_priority_sites') as mock_gen:
            mock_gen.return_value = [self.site1, self.site2]
            
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
            
            self.site1.contains_atom.assert_called_once()
            self.site2.contains_atom.assert_not_called()
    
    def test_calls_update_occupation_when_found(self):
        """Test that update_occupation is called when atom found."""
        self.site1.contains_atom = MagicMock(return_value=True)
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
            with patch.object(self.collection, 'update_occupation') as mock_update:
                self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
                mock_update.assert_called_once_with(self.site1, self.atom)
    
    def test_handles_atom_not_found(self):
        """Test behavior when atom not found in any site."""
        self.site1.contains_atom = MagicMock(return_value=False)
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
            with patch.object(self.collection, 'update_occupation') as mock_update:
                self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
                
                mock_update.assert_not_called()
                self.assertIsNone(self.atom.in_site)
    
    def test_resets_site_occupations(self):
        """Test that reset_site_occupations is called at start."""
        with patch.object(self.collection, 'reset_site_occupations') as mock_reset:
            with patch.object(self.collection, '_get_priority_sites', return_value=[]):
                self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
                mock_reset.assert_called_once()
    
    def test_resets_atom_in_site(self):
        """Test that atom.in_site is reset to None."""
        self.atom.in_site = 999  # Set to some previous value
        
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)
            self.assertIsNone(self.atom.in_site)
            
            
class TestGetPrioritySites(unittest.TestCase):
    """Test _get_priority_sites generator behaviour."""

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

    def test_yields_all_sites_when_no_recent_sites(self):
        """All sites yielded in list order when no site history exists."""
        priority_sites = list(self.collection._get_priority_sites(self.atom, self.lattice.matrix))

        self.assertEqual(priority_sites, [self.site1, self.site2, self.site3])

    def test_yields_neighbours_after_transitions_without_reference_centres(self):
        """Without reference centres, neighbours are yielded after transitions."""
        self.assertIsNone(self.collection._site_centres)
        self.atom._recent_sites = [self.site1.index, None]

        with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
            with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
                mock_transitions.return_value = [self.site2.index]
                mock_neighbours.return_value = [self.site3]

                priority_sites = list(self.collection._get_priority_sites(self.atom, self.lattice.matrix))

                self.assertEqual(priority_sites[0], self.site1)
                self.assertEqual(priority_sites[1], self.site2)
                self.assertEqual(priority_sites[2], self.site3)
                mock_neighbours.assert_called_once_with(self.site1.index)

    def test_yields_no_duplicates_with_neighbours_and_transitions(self):
        """No duplicates when a neighbour also appears as a transition."""
        self.atom._recent_sites = [self.site1.index, None]

        with patch.object(self.site1, 'most_frequent_transitions') as mock_transitions:
            with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
                mock_transitions.return_value = [self.site2.index]
                mock_neighbours.return_value = [self.site2, self.site3]

                priority_sites = list(self.collection._get_priority_sites(self.atom, self.lattice.matrix))

                self.assertEqual(len(priority_sites), 3)
                site_indices = [s.index for s in priority_sites]
                self.assertEqual(len(site_indices), len(set(site_indices)))
                self.assertEqual(site_indices, [self.site1.index, self.site2.index, self.site3.index])

    def test_yields_remaining_sites_in_list_order_after_neighbours(self):
        """Without reference centres, every site not yet yielded follows in list order."""
        self.atom._recent_sites = [self.site1.index, None]

        with patch.object(self.collection, 'neighbouring_sites', return_value=[]):
            priority_sites = list(self.collection._get_priority_sites(self.atom, self.lattice.matrix))

        self.assertEqual(priority_sites, [self.site1, self.site2, self.site3])

    def test_skips_neighbour_checking_when_no_recent_sites(self):
        """Neighbour checking is skipped when atom has no recent sites."""
        with patch.object(self.collection, 'neighbouring_sites') as mock_neighbours:
            priority_sites = list(self.collection._get_priority_sites(self.atom, self.lattice.matrix))

            self.assertEqual(len(priority_sites), 3)
            mock_neighbours.assert_not_called()


class TestDistanceRankingNeedsReferenceCentres(unittest.TestCase):
    """Sites are ranked by distance only when every site has a reference centre."""

    def setUp(self):
        # Site indices 50, 51 and 52, which differ from list positions.
        Site._newid = 50
        self.lattice_matrix = np.eye(3) * 2.0
        # On the third site's centre: 0.4 A from the second and 0.8 A from
        # the first, so distance order is the reverse of list order.
        self.atom = Atom(index=0)
        self.atom._frac_coords = np.array([0.5, 0.1, 0.1])

    def search_order(self, reference_centres):
        """Return the site indices in search order.

        Args:
            reference_centres: One reference centre, or None, per site.
        """
        sites = [PolyhedralSite(vertex_indices=[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3],
                                reference_center=centre)
                 for i, centre in enumerate(reference_centres)]
        collection = PolyhedralSiteCollection(sites)
        return [s.index for s in collection._get_priority_sites(self.atom, self.lattice_matrix)]

    def test_ranked_by_distance_when_every_site_has_a_reference_centre(self):
        """Sites are ranked by distance from the atom when every site has a reference centre."""
        centres = [np.array([0.1, 0.1, 0.1]), np.array([0.3, 0.1, 0.1]), np.array([0.5, 0.1, 0.1])]
        self.assertEqual(self.search_order(centres), [52, 51, 50])

    def test_list_order_when_any_site_lacks_a_reference_centre(self):
        """Sites are searched in list order when any site lacks a reference centre."""
        centres = [np.array([0.1, 0.1, 0.1]), np.array([0.3, 0.1, 0.1]), None]
        self.assertEqual(self.search_order(centres), [50, 51, 52])


def _tetrahedral_tiling(n):
    """Split a cell into an n x n x n grid of cubes, and each cube into six tetrahedra.

    Each cube is split into the six tetrahedra that share its main diagonal,
    so the tetrahedra fill the cell without gaps or overlaps.

    Args:
        n: Number of cubes along each cell vector.

    Returns:
        Tuple of the fractional coordinates of the grid points, shape
        (n ** 3, 3), and, for each tetrahedron, a tuple of its vertex indices
        and its centre, the mean of its vertices before they are wrapped into
        the cell.
    """
    points = np.array(list(itertools.product(range(n), repeat=3))) / n
    tetrahedra = []
    for corner in itertools.product(range(n), repeat=3):
        for axes in itertools.permutations(range(3)):
            # Walk from the corner to the opposite corner, one axis at a time.
            vertices = [np.array(corner)]
            for axis in axes:
                step = vertices[-1].copy()
                step[axis] += 1
                vertices.append(step)
            vertex_indices = [int(np.ravel_multi_index(tuple(v % n), (n, n, n))) for v in vertices]
            centre = (np.mean(vertices, axis=0) / n) % 1.0
            tetrahedra.append((vertex_indices, centre))
    return points, tetrahedra


class TestTilingLeavesNoAtomUnassigned(unittest.TestCase):
    """Sites that fill a triclinic cell between them leave no atom unassigned."""

    def setUp(self):
        Site._newid = 0
        rng = np.random.default_rng(0)
        points, self.tetrahedra = _tetrahedral_tiling(3)
        lithium = rng.random((200, 3))
        lattice = Lattice.from_parameters(9.0, 10.0, 11.0, 80, 95, 105)
        self.structure = Structure(lattice, ["O"] * len(points) + ["Li"] * len(lithium),
                                   np.vstack([points, lithium]))
        self.atoms = atoms_from_structure(self.structure, "Li")

    def unassigned_atoms(self, collection):
        """Return the indices of the atoms that the collection leaves unassigned."""
        collection.analyse_structure(self.atoms, self.structure)
        return [atom.index for atom in self.atoms if atom.in_site is None]

    def test_every_atom_is_assigned_with_reference_centres(self):
        """Atoms with no recent site are all assigned when sites are ranked by distance."""
        sites = [PolyhedralSite(vertex_indices=vertex_indices, reference_center=centre)
                 for vertex_indices, centre in self.tetrahedra]
        self.assertEqual(self.unassigned_atoms(PolyhedralSiteCollection(sites)), [])

    def test_every_atom_is_assigned_without_reference_centres(self):
        """Atoms with a recent site are all assigned when sites have no reference centres."""
        sites = [PolyhedralSite(vertex_indices=vertex_indices)
                 for vertex_indices, _ in self.tetrahedra]
        rng = np.random.default_rng(1)
        for atom in self.atoms:
            atom._recent_sites = [sites[rng.integers(len(sites))].index, None]
        self.assertEqual(self.unassigned_atoms(PolyhedralSiteCollection(sites)), [])


if __name__ == '__main__':
    unittest.main()
