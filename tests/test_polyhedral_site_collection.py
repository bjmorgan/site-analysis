import itertools
import unittest
import warnings
from collections import Counter
import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.spatial import Delaunay
from site_analysis import TrajectoryBuilder
from site_analysis._compat import HAS_NUMBA
from site_analysis.faces import Face
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
from site_analysis.trajectory import Trajectory
from tests.test_faces import build_cubes, displaced, simple_cubic_cubes
from tests.test_helpers import interstitial_structure
from unittest.mock import patch, Mock, PropertyMock, MagicMock

# A reference centre for sites whose geometry these tests do not use.
CENTRE = np.array([0.5, 0.5, 0.5])

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
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], label="site1",
                                    reference_center=[0.05, 0.05, 0.0])
        
        # Site 2: tetrahedron with vertices 4, 5, 6, 7
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], label="site2",
                                    reference_center=[0.05, 0.05, 0.1])
        
        # Site 3: tetrahedron with vertices 0, 1, 4, 5 - shares face with site1
        self.site3 = PolyhedralSite(vertex_indices=[0, 1, 4, 5], label="site3",
                                    reference_center=[0.05, 0.0, 0.05])
        
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
        mock_site_0 = Mock(spec=PolyhedralSite, index=0, vertex_indices=[0, 1, 2, 3])
        mock_site_0.reference_center = None
        mock_site_1 = Mock(spec=PolyhedralSite, index=1, vertex_indices=[4, 5, 6, 7])
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
             patch.object(PolyhedralSiteCollection, 'find_shared_faces'), \
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
    
    def test_analyse_structure_with_too_few_atoms_raises(self):
        """A structure without an atom for every vertex index is rejected."""
        structure = Structure(self.lattice, self.structure.species[:7],
                              self.structure.frac_coords[:7])
        with self.assertRaisesRegex(
                ValueError, "^Site 1 has vertex index 7, but the structure has 7 atoms$"):
            self.collection.analyse_structure(self.atoms, structure)

    def test_find_shared_faces_with_too_few_atoms_raises(self):
        """Coordinates without a row for every vertex index are rejected."""
        with self.assertRaisesRegex(
                ValueError, "^Site 1 has vertex index 7, but the structure has 7 atoms$"):
            self.collection.find_shared_faces(
                self.structure.frac_coords[:7], self.lattice.matrix)

    def test_validate_atom_count_accepts_an_atom_for_every_vertex_index(self):
        """No error is raised when every vertex index is an atom."""
        self.collection.validate_atom_count(8)

    def test_validate_atom_count_names_the_first_site_in_index_order(self):
        """The error names the lowest-index site with a vertex index beyond the atoms."""
        Site._newid = 0
        first = PolyhedralSite(vertex_indices=[0, 1, 2, 8], reference_center=CENTRE)
        second = PolyhedralSite(vertex_indices=[0, 1, 2, 9], reference_center=CENTRE)
        collection = PolyhedralSiteCollection([second, first])
        with self.assertRaisesRegex(
                ValueError, "^Site 0 has vertex index 8, but the structure has 8 atoms$"):
            collection.validate_atom_count(8)

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
        site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=[0.15, 0.15, 0.15])
        site1.contains_atoms = [1, 2]
        
        site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], reference_center=[0.65, 0.65, 0.65])
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
        with patch.object(PolyhedralSiteCollection, 'find_shared_faces'), \
                patch.object(PolyhedralSite, 'contains_point') as mock_contains_point:
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

        with patch.object(PolyhedralSiteCollection, 'find_shared_faces'), \
             patch.object(PolyhedralSite, 'notify_structure_changed') as mock_notify, \
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

    def test_sites_contain_points_with_too_few_atoms_raises(self):
        """Coordinates without a row for every vertex index are rejected."""
        with self.assertRaisesRegex(ValueError, "^Site 1 has vertex index 7, but the structure has 7 atoms$"):
            self.collection.sites_contain_points(
                np.zeros((3, 3)), self.structure.frac_coords[:7], self.lattice.matrix)

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
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], label="site1",
                                    reference_center=CENTRE)
        
        # Site 2: tetrahedron with vertices 4, 5, 6, 7
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], label="site2",
                                    reference_center=CENTRE)
        
        # Site 3: tetrahedron with vertices 0, 1, 4, 5 - shares 2 vertices with site1, 2 with site2
        # Note: The current implementation considers this a neighbour of site4
        self.site3 = PolyhedralSite(vertex_indices=[0, 1, 4, 5], label="site3",
                                    reference_center=CENTRE)
        
        # Site 4: tetrahedron with vertices 0, 1, 2, 4 - shares 3 vertices with site1 (face sharing)
        self.site4 = PolyhedralSite(vertex_indices=[0, 1, 2, 4], label="site4",
                                    reference_center=CENTRE)
        
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
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        first = PolyhedralSite(vertex_indices=[1, 2, 3, 4], reference_center=CENTRE)
        second = PolyhedralSite(vertex_indices=[0, 1, 2, 5], reference_center=CENTRE)
        neighbours = construct_neighbouring_sites([site, first, second])
        self.assertEqual(neighbours[site.index], [first, second])

    def test_construct_neighbouring_sites_no_neighbours(self):
        """Test construct_neighbouring_sites with sites that have no neighbours."""
        # Create isolated sites that don't share vertices
        site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], reference_center=CENTRE)
        sites = [site1, site2]
        
        # Call function
        neighbours = construct_neighbouring_sites(sites)
        
        # Check each site has no neighbours
        self.assertEqual(len(neighbours[site1.index]), 0)
        self.assertEqual(len(neighbours[site2.index]), 0)
    
    def test_construct_neighbouring_sites_identical_sites(self):
        """Test construct_neighbouring_sites with identical sites."""
        # Create two sites with identical vertices
        site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        site2 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
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
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
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
        with self.assertWarns(FutureWarning):
            sites = [
                PolyhedralSite(vertex_indices=[0, 1, 2, 3]),
                PolyhedralSite(vertex_indices=[4, 5, 6, 7]),
            ]
        centres, site_indices = _collect_reference_centres(sites)
        self.assertIsNone(centres)

    def test_returns_none_when_mixed_reference_centres(self):
        """Returns None when only some sites have reference centres."""
        Site._newid = 0
        with self.assertWarns(FutureWarning):
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


class TestAssignSiteOccupationsInteraction(unittest.TestCase):
    """Test interaction between assign_site_occupations and _get_priority_sites."""

    def setUp(self):
        Site._newid = 0
        self.lattice = Lattice.cubic(2.0)
        self.structure = Structure(self.lattice, ["Li"], [[0.1, 0.1, 0.1]])
        
        self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3], reference_center=CENTRE)
        self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7], reference_center=CENTRE)
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
    """Test _get_priority_sites generator behaviour for sites without reference centres (deprecated)."""

    def setUp(self):
        Site._newid = 0
        self.lattice = Lattice.cubic(2.0)
        self.structure = Structure(self.lattice, ["Li"], [[0.1, 0.1, 0.1]])

        with self.assertWarns(FutureWarning):
            self.site1 = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
        with self.assertWarns(FutureWarning):
            self.site2 = PolyhedralSite(vertex_indices=[4, 5, 6, 7])
        with self.assertWarns(FutureWarning):
            self.site3 = PolyhedralSite(vertex_indices=[8, 9, 10, 11])
        self.collection = PolyhedralSiteCollection([self.site1, self.site2, self.site3])

        self.atoms = atoms_from_structure(self.structure, "Li")
        self.atom = self.atoms[0]

    def test_yields_all_sites_when_no_recent_sites(self):
        """All sites yielded in list order when no site history exists."""
        priority_sites = list(self.collection._get_priority_sites(self.atom, self.lattice.matrix))

        self.assertEqual(priority_sites, [self.site1, self.site2, self.site3])

    def test_neighbours_follow_transitions_without_repeats(self):
        """Without reference centres, the most recent site's neighbours follow its
        transitions, then the remaining sites in list order, each site once."""
        with self.assertWarns(FutureWarning):
            site4 = PolyhedralSite(vertex_indices=[12, 13, 14, 15])
        collection = PolyhedralSiteCollection([self.site1, self.site2, self.site3, site4])
        self.assertIsNone(collection._site_centres)
        self.atom._recent_sites = [self.site1.index, None]

        with patch.object(self.site1, 'most_frequent_transitions', return_value=[site4.index]):
            with patch.object(collection, 'neighbouring_sites',
                              return_value=[site4, self.site3]) as mock_neighbours:
                priority_sites = list(collection._get_priority_sites(self.atom, self.lattice.matrix))

        # Without the neighbour step, site2 would come before site3.
        self.assertEqual(priority_sites, [self.site1, site4, self.site3, self.site2])
        mock_neighbours.assert_called_once_with(self.site1.index)

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
        with self.assertWarns(FutureWarning):
            search_order = self.search_order(centres)
        self.assertEqual(search_order, [50, 51, 52])


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
        with self.assertWarns(FutureWarning):
            sites = [PolyhedralSite(vertex_indices=vertex_indices)
                     for vertex_indices, _ in self.tetrahedra]
        rng = np.random.default_rng(1)
        for atom in self.atoms:
            atom._recent_sites = [sites[rng.integers(len(sites))].index, None]
        self.assertEqual(self.unassigned_atoms(PolyhedralSiteCollection(sites)), [])


class TestAssignmentToSitesSpanningMoreThanTwoCells(unittest.TestCase):
    """Atoms are assigned to sites whose unwrapped vertices span more than two cells."""

    def test_atom_near_the_far_vertex_is_assigned(self):
        """An atom near the far vertex of a site spanning x from 0.9 to 2.2 is assigned to it."""
        # Rows (1,0,0), (3,1,0), (0,0,1) scaled to Angstrom: an unreduced basis.
        lattice_matrix = 4.0 * np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # The vertices' images nearest the reference centre (0.6, 0, 0.5).
        vertices = np.array([[2.2, -0.4, 0.5], [1.0, 0.0, 0.7], [1.1, -0.2, 0.4], [0.9, 0.0, 0.4]])
        point = 0.8 * vertices[0] + 0.2 * vertices.mean(axis=0)
        frac_coords = np.vstack([vertices, point]) % 1.0
        site = PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                              reference_center=np.array([0.6, 0.0, 0.5]))
        collection = PolyhedralSiteCollection([site])
        atom = Atom(index=4)
        atom.assign_coords(frac_coords)
        site.notify_structure_changed(frac_coords, lattice_matrix)
        collection.assign_site_occupations([atom], lattice_matrix)
        self.assertEqual(atom.in_site, site.index)


# Containment with numba where it is installed, and with numpy.
BACKENDS = [True, False] if HAS_NUMBA else [False]


def containing_sites(sites, points):
    """The number of sites containing each point."""
    return np.array([sum(site.contains_point(point) for site in sites) for point in points])


def noisy_frames(structure, n_frames=3, sigma=0.1):
    """Frames of a structure with every atom displaced by seeded Gaussian noise (sigma in Angstrom)."""
    return [displaced(structure, sigma, seed=seed) for seed in range(n_frames)]


class TestSpaceFilling(unittest.TestCase):
    """Sites that share faces divide the space between them exactly."""

    N_POINTS = 500

    def assert_space_filling(self, analyse, sites, frames):
        rng = np.random.default_rng(0)
        for frame_number, frame in enumerate(frames):
            analyse(frame)
            points = rng.random((self.N_POINTS, 3))
            counts = containing_sites(sites, points)
            with self.subTest(frame=frame_number):
                self.assertEqual(Counter(counts.tolist()), {1: self.N_POINTS})

    def assert_cube_grid_fills_space(self, reference_centres):
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                structure, sites = simple_cubic_cubes(4)
                if not reference_centres:
                    for site in sites:
                        site.reference_center = None
                collection = PolyhedralSiteCollection(sites)
                self.assert_space_filling(
                    lambda frame: collection.analyse_structure([], frame),
                    sites, noisy_frames(structure))

    def test_noisy_cubes_with_reference_centres(self):
        self.assert_cube_grid_fills_space(reference_centres=True)

    def test_noisy_cubes_without_reference_centres(self):
        self.assert_cube_grid_fills_space(reference_centres=False)

    def test_noisy_cubes_built_from_a_reference(self):
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                target, trajectory = build_cubes((3, 3, 3))
                self.assert_space_filling(
                    trajectory.analyse_structure, trajectory.sites, noisy_frames(target))


class TestConvexSitesMatchTheirHulls(unittest.TestCase):
    """Sites whose faces are all triangles contain the points of their convex hulls."""

    def test_fcc_tetrahedra_and_octahedra(self):
        structure = interstitial_structure()
        frame = displaced(structure, 0.1)
        rng = np.random.default_rng(1)
        points = rng.random((500, 3))
        shifts = np.array(list(itertools.product((-1, 0, 1, 2), repeat=3)))
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                trajectory = (TrajectoryBuilder()
                              .with_structure(structure)
                              .with_reference_structure(structure)
                              .with_mobile_species("Li")
                              .with_polyhedral_sites(centre_species="Li", vertex_species="O",
                                                     cutoff=2.0, n_vertices=4)
                              .with_polyhedral_sites(centre_species="Na", vertex_species="O",
                                                     cutoff=2.2, n_vertices=6)
                              .build())
                trajectory.analyse_structure(frame)
                for site in trajectory.sites:
                    contained = [site.contains_point(point) for point in points]
                    hull = Delaunay(site.vertex_coords)
                    expected = [bool(np.any(hull.find_simplex(point + shifts) >= 0))
                                for point in points]
                    self.assertEqual(contained, expected)


class TestBackendsAgree(unittest.TestCase):

    def test_numba_and_numpy_assign_every_atom_to_the_same_site(self):
        if not HAS_NUMBA:
            self.skipTest("numba not available")
        structure, _ = simple_cubic_cubes(4)
        rng = np.random.default_rng(2)
        lithium = rng.random((40, 3))
        with_lithium = Structure(structure.lattice,
                                 structure.species + ["Li"] * len(lithium),
                                 np.vstack([structure.frac_coords, lithium]))
        frames = noisy_frames(with_lithium, n_frames=5, sigma=0.15)
        trajectories = []
        for has_numba in (True, False):
            with patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                _, sites = simple_cubic_cubes(4)
                trajectory = Trajectory(sites=sites,
                                        atoms=atoms_from_structure(with_lithium, "Li"))
                trajectory.trajectory_from_structures(frames, progress=False)
                trajectories.append([atom.trajectory for atom in trajectory.atoms])
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertNotIn(None, [site for atom in trajectories[0] for site in atom])


def two_cubes(push=0.0):
    """Two cubes of O atoms sharing the square at x = 6 Angstrom in a 12 Angstrom cell.

    Atom 0, a corner of the left cube's unshared square at x = 4 Angstrom,
    is moved by ``push`` Angstrom along x.

    Returns:
        ``(structure, sites)``.
    """
    corners = [(x, y, z) for x in (4.0, 6.0, 8.0) for y in (5.0, 7.0) for z in (5.0, 7.0)]
    corners[0] = (4.0 + push, 5.0, 5.0)
    structure = Structure(Lattice.cubic(12.0), ["O"] * 12, corners, coords_are_cartesian=True)
    Site.reset_index(0)
    sites = [PolyhedralSite(vertex_indices=list(range(0, 8)),
                            reference_center=np.array([5.0, 6.0, 6.0]) / 12.0),
             PolyhedralSite(vertex_indices=list(range(4, 12)),
                            reference_center=np.array([7.0, 6.0, 6.0]) / 12.0)]
    return structure, sites


def triangles_on(site, atoms):
    """The site's triangles whose atoms are all in a set of atoms, as sets of atom indices."""
    return {frozenset(site.vertex_indices[i] for i in triangle)
            for triangle in site._triangles.tolist()
            if {site.vertex_indices[i] for i in triangle} <= set(atoms)}


class TestFindSharedFacesSetsTriangles(unittest.TestCase):

    def test_shared_squares_are_triangulated_identically(self):
        structure, sites = two_cubes()
        collection = PolyhedralSiteCollection(sites)
        collection.find_shared_faces(structure.frac_coords, structure.lattice.matrix)
        shared = [4, 5, 6, 7]
        self.assertEqual(len(triangles_on(sites[0], shared)), 2)
        self.assertEqual(triangles_on(sites[0], shared), triangles_on(sites[1], shared))

    def test_first_finding_does_not_warn(self):
        structure, sites = two_cubes()
        collection = PolyhedralSiteCollection(sites)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            collection.find_shared_faces(structure.frac_coords, structure.lattice.matrix)

    def test_finding_the_same_faces_again_does_not_warn(self):
        structure, sites = two_cubes()
        collection = PolyhedralSiteCollection(sites)
        collection.find_shared_faces(structure.frac_coords, structure.lattice.matrix)
        noisy = displaced(structure, 0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            collection.find_shared_faces(noisy.frac_coords, noisy.lattice.matrix)
        self.assertEqual(collection.shared_faces(), (Face((0, 1), (4, 5, 6, 7)),))

    def test_finding_different_faces_warns_and_replaces_them(self):
        # Two cubes without reference centres sharing atoms 4-7. In the second
        # structure the left cube spans the cell boundary and unwraps two of the
        # shared atoms by a cell and two not, so the cubes no longer share a face.
        def cubes(x_left, x_shared, x_right):
            coords = np.zeros((12, 3))
            for i in range(12):
                coords[i] = [(x_left, x_shared, x_right)[i // 4], 0.4 + 0.2 * ((i >> 1) & 1),
                             0.4 + 0.2 * (i & 1)]
            return coords
        joined = cubes(0.75, 0.95, 0.15)
        torn = cubes(0.85, 0.05, 0.25)
        torn[[6, 7], 0] = 0.55
        lattice_matrix = Lattice.cubic(10.0).matrix
        Site.reset_index(0)
        with self.assertWarns(FutureWarning):
            sites = [PolyhedralSite(vertex_indices=list(range(0, 8))),
                     PolyhedralSite(vertex_indices=list(range(4, 12)))]
        collection = PolyhedralSiteCollection(sites)
        collection.find_shared_faces(joined, lattice_matrix)
        self.assertEqual(collection.shared_faces(), (Face((0, 1), (4, 5, 6, 7)),))
        with self.assertWarnsRegex(
                UserWarning,
                r"^The shared faces differ from those found before \(0 added, 1 removed\), so the "
                r"connectivity of the sites has changed$"):
            collection.find_shared_faces(torn, lattice_matrix)
        self.assertEqual(collection.shared_faces(), ())

    def test_a_second_structure_replaces_the_triangles(self):
        outward, sites = two_cubes(push=-0.1)
        inward, _ = two_cubes(push=0.1)
        collection = PolyhedralSiteCollection(sites)
        collection.find_shared_faces(outward.frac_coords, outward.lattice.matrix)
        unshared, shared = [0, 1, 2, 3], [4, 5, 6, 7]
        before = triangles_on(sites[0], unshared), triangles_on(sites[0], shared)
        collection.find_shared_faces(inward.frac_coords, inward.lattice.matrix)
        after = triangles_on(sites[0], unshared), triangles_on(sites[0], shared)
        # Moving atom 0 from outside to inside the square's plane moves the hull's diagonal.
        self.assertNotEqual(after[0], before[0])
        self.assertEqual(after[1], before[1])

    def test_shared_face_folding_through_a_site_centre_is_too_distorted(self):
        # Two sites on either side of a bent square; the fan of the square
        # from its lowest atom folds through the centre of the thinner site 1.
        coords = [(4.0, 4.0, 5.3), (6.0, 4.0, 5.0), (6.0, 6.0, 5.3), (4.0, 6.0, 5.0),
                  (5.0, 5.0, 3.0), (5.0, 5.0, 5.6)]
        structure = Structure(Lattice.cubic(10.0), ["O"] * 6, coords, coords_are_cartesian=True)
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=[0, 1, 2, 3, 4],
                                reference_center=np.array([0.5, 0.5, 0.45])),
                 PolyhedralSite(vertex_indices=[0, 1, 2, 3, 5],
                                reference_center=np.array([0.5, 0.5, 0.53]))]
        collection = PolyhedralSiteCollection(sites)
        with self.assertRaisesRegex(
                RuntimeError,
                r"^Site 1 \(vertices \[0, 1, 2, 3, 5\]\): polyhedron is too distorted for its "
                r"triangles to enclose its centre$"):
            collection.find_shared_faces(structure.frac_coords, structure.lattice.matrix)

    def test_restored_sites_in_a_collection_regain_their_containment(self):
        structure, _ = two_cubes()
        frame = displaced(structure, 0.1)
        points = np.random.default_rng(3).random((500, 3))
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                _, sites = two_cubes()
                collection = PolyhedralSiteCollection(sites)
                collection.find_shared_faces(structure.frac_coords, structure.lattice.matrix)
                collection.analyse_structure([], frame)
                expected = [[site.contains_point(point) for point in points] for site in sites]
                restored = PolyhedralSiteCollection(
                    [PolyhedralSite.from_dict(site.as_dict()) for site in sites])
                restored.analyse_structure([], frame)
                for copy, contained in zip(restored.sites, expected):
                    self.assertEqual([copy.contains_point(point) for point in points], contained)
                self.assertTrue(any(any(contained) for contained in expected))

    def test_shared_atoms_that_cannot_be_triangulated_identically_name_both_sites(self):
        # A cube, and a site on four of its corners, no three on one square, and an outside atom.
        corners = [(x, y, z) for x in (4.0, 6.0) for y in (4.0, 6.0) for z in (4.0, 6.0)]
        structure = Structure(Lattice.cubic(10.0), ["O"] * 9, corners + [(2.0, 2.0, 7.0)],
                              coords_are_cartesian=True)
        Site.reset_index(0)
        sites = [PolyhedralSite(vertex_indices=list(range(8)),
                                reference_center=np.array([0.5, 0.5, 0.5])),
                 PolyhedralSite(vertex_indices=[0, 3, 5, 6, 8],
                                reference_center=np.array([0.4, 0.4, 0.5]))]
        collection = PolyhedralSiteCollection(sites)
        with self.assertRaisesRegex(
                ValueError,
                r"^Sites 0 and 1: Shared atoms \[0, 3, 5, 6\] cannot be triangulated "
                r"identically .*; the sites may overlap$"):
            collection.find_shared_faces(structure.frac_coords, structure.lattice.matrix)
        with self.assertRaisesRegex(
                RuntimeError,
                "^Shared faces have not been found: analyse a structure, or call "
                "find_shared_faces with one$"):
            collection.shared_faces()

class TestAnalyseStructureFindsSharedFaces(unittest.TestCase):

    def test_first_analysis_finds_the_shared_faces_of_its_structure(self):
        structure, sites = two_cubes()
        collection = PolyhedralSiteCollection(sites)
        collection.analyse_structure([], structure)
        self.assertEqual(collection.shared_faces(), (Face((0, 1), (4, 5, 6, 7)),))

    def test_later_analysis_keeps_the_triangles(self):
        outward, sites = two_cubes(push=-0.1)
        inward, _ = two_cubes(push=0.1)
        collection = PolyhedralSiteCollection(sites)
        collection.analyse_structure([], outward)
        triangles = [site._triangles.tolist() for site in sites]
        # Finding the faces from this structure would move the left cube's hull diagonal.
        collection.analyse_structure([], inward)
        self.assertEqual([site._triangles.tolist() for site in sites], triangles)

    def test_sites_contain_points_accepts_lists(self):
        structure, sites = two_cubes()
        collection = PolyhedralSiteCollection(sites)
        for has_numba in BACKENDS:
            with self.subTest(numba=has_numba), \
                    patch('site_analysis.polyhedron_surface.HAS_NUMBA', has_numba):
                self.assertTrue(collection.sites_contain_points(
                    [[5.0 / 12.0, 0.5, 0.5], [7.0 / 12.0, 0.5, 0.5]],
                    structure.frac_coords, structure.lattice.matrix))

    def test_sites_contain_points_finds_the_shared_faces_of_its_structure(self):
        structure, sites = two_cubes()
        collection = PolyhedralSiteCollection(sites)
        points = np.array([[5.0, 6.0, 6.0], [7.0, 6.0, 6.0]]) / 12.0
        self.assertTrue(collection.sites_contain_points(
            points, structure.frac_coords, structure.lattice.matrix))
        self.assertEqual(collection.shared_faces(), (Face((0, 1), (4, 5, 6, 7)),))

    def test_too_distorted_site_names_itself(self):
        # One tetrahedron; in the second frame its apex has moved through its base.
        base = [(5.0, 5.0, 6.0), (5.0, 4.134, 4.5), (5.0, 5.866, 4.5)]
        def frame(apex_x):
            return Structure(Lattice.cubic(10.0), ["O"] * 4 + ["Li"],
                             [*base, (apex_x, 5.0, 5.0), (4.9, 5.0, 5.0)],
                             coords_are_cartesian=True)
        Site.reset_index(0)
        trajectory = Trajectory(sites=[PolyhedralSite(vertex_indices=[0, 1, 2, 3],
                                                      reference_center=[0.46, 0.5, 0.5])],
                                atoms=[Atom(index=4)])
        trajectory.analyse_structure(frame(3.4))
        with self.assertRaisesRegex(
                RuntimeError,
                r"^Site 0 \(vertices \[0, 1, 2, 3\]\): polyhedron is too distorted"):
            trajectory.analyse_structure(frame(6.6))


if __name__ == '__main__':
    unittest.main()
