import unittest
from collections import Counter
from unittest.mock import patch, Mock, PropertyMock
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.spherical_site_collection import SphericalSiteCollection
from site_analysis.spherical_site import SphericalSite
from site_analysis.atom import Atom
from site_analysis.distances import mic_distance
from site_analysis.site import Site


class SphericalSiteCollectionTestCase(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures with real objects rather than complex mocks."""
        # Reset Site ID counter
        Site._newid = 0

        # Create a real lattice
        self.lattice = Lattice.cubic(10.0)

        # Create real SphericalSite objects
        self.site1 = SphericalSite(
            frac_coords=np.array([0.1, 0.1, 0.1]),
            rcut=1.5,
            label="site1"
        )
        self.site2 = SphericalSite(
            frac_coords=np.array([0.5, 0.5, 0.5]),
            rcut=1.5,
            label="site2"
        )

        # Create collection with real sites
        self.sites = [self.site1, self.site2]
        self.collection = SphericalSiteCollection(sites=self.sites)

        # Create real atoms
        self.atom1 = Atom(index=0)
        self.atom2 = Atom(index=1)
        self.atoms = [self.atom1, self.atom2]

        # Create real structure
        self.structure = Structure(
            lattice=self.lattice,
            species=["Na", "Na"],
            coords=[[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]
        )

    def test_initialization(self):
        """Test that SphericalSiteCollection initializes correctly."""
        self.assertEqual(self.collection.sites, self.sites)

    def test_init_with_empty_sites_list(self):
        """Test that __init__ works with empty sites list."""
        collection = SphericalSiteCollection([])

        self.assertEqual(collection.sites, [])
        self.assertEqual(collection._site_lookup, {})

    def test_init_raises_type_error_with_non_spherical_sites(self):
        """Test that initialisation raises TypeError with non-SphericalSite objects."""
        # Create a mix of site types
        non_spherical_site = Mock()
        mixed_sites = [self.site1, non_spherical_site]

        # Test initialisation with mixed site types
        with self.assertRaises(TypeError):
            SphericalSiteCollection(sites=mixed_sites)

    def test_init_accepts_a_generator_of_sites(self):
        """A collection built from a generator holds every site."""
        collection = SphericalSiteCollection(s for s in [self.site1, self.site2])
        self.assertEqual(collection.sites, [self.site1, self.site2])

    def test_analyse_structure(self):
        """Test that analyse_structure calls assign_coords and assign_site_occupations."""
        # Patch the methods we want to verify
        with patch.object(Atom, 'assign_coords') as mock_assign_coords, \
             patch.object(SphericalSiteCollection, 'assign_site_occupations') as mock_assign_occupations:

            # Call analyse_structure
            self.collection.analyse_structure(self.atoms, self.structure)

            # Verify that assign_coords was called for each atom
            self.assertEqual(mock_assign_coords.call_count, 2)

            # Verify that assign_site_occupations was called with lattice_matrix
            mock_assign_occupations.assert_called_once()
            args = mock_assign_occupations.call_args[0]
            self.assertIs(args[0], self.atoms)
            np.testing.assert_array_equal(args[1], self.structure.lattice.matrix)

    def test_site_occupation_reset(self):
        """Test that site occupations are reset at the beginning of assign_site_occupations."""
        # Add atoms to sites
        self.site1.contains_atoms = [0]
        self.site2.contains_atoms = [1]

        # Patch update_occupation to prevent it from running
        with patch.object(self.collection, 'update_occupation'):
            # Call the method
            self.collection.assign_site_occupations([], self.lattice.matrix)

            # Verify sites were reset
            self.assertEqual(self.site1.contains_atoms, [])
            self.assertEqual(self.site2.contains_atoms, [])

    def test_high_level_site_allocation(self):
        """Test the high-level site allocation logic without mocking fine details."""
        # Set up atoms with coordinates that match the sites
        # This avoids mocking complex interactions with SphericalSite.contains_atom
        self.atom1._frac_coords = np.array([0.1, 0.1, 0.1])  # Near site1
        self.atom2._frac_coords = np.array([0.5, 0.5, 0.5])  # Near site2

        # Patch reset_site_occupations and update_occupation
        with patch.object(SphericalSiteCollection, 'reset_site_occupations'), \
             patch.object(SphericalSiteCollection, 'update_occupation') as mock_update:

            # Call the method
            self.collection.assign_site_occupations(self.atoms, self.lattice.matrix)

            # Verify update_occupation was called twice (once per atom)
            self.assertEqual(mock_update.call_count, 2)

    def test_integration(self):
        """Integration test with actual behavior for a simple case."""
        # Set up atoms with coordinates
        self.atom1._frac_coords = np.array([0.1, 0.1, 0.1])  # Inside site1's radius
        self.atom2._frac_coords = np.array([0.5, 0.5, 0.5])  # Inside site2's radius

        # Reset sites
        self.site1.contains_atoms = []
        self.site2.contains_atoms = []

        # Call the method directly
        self.collection.analyse_structure(self.atoms, self.structure)

        # Verify atoms were assigned to the correct sites
        self.assertEqual(self.atom1.in_site, self.site1.index)
        self.assertEqual(self.atom2.in_site, self.site2.index)

        # Verify sites contain the correct atoms
        self.assertEqual(self.site1.contains_atoms, [self.atom1.index])
        self.assertEqual(self.site2.contains_atoms, [self.atom2.index])

    def test_update_occupation(self):
        """Test the update_occupation method."""
        # Initialize an atom not in any site
        atom = Atom(index=5)
        atom.in_site = None
        atom._frac_coords = np.array([0.3, 0.3, 0.3])

        # Call update_occupation to assign the atom to a site
        self.collection.update_occupation(self.site1, atom)

        # Verify atom has been assigned to the site
        self.assertEqual(atom.in_site, self.site1.index)

        # Verify site contains the atom
        self.assertIn(atom.index, self.site1.contains_atoms)

        # Verify atom coords are added to site points
        np.testing.assert_array_equal(self.site1.points[-1], atom.frac_coords)

    def test_update_occupation_moves_atom_without_recording_transition(self):
        """update_occupation assigns the atom but leaves transitions to append_timestep."""
        # Initialize an atom in site2
        atom = Atom(index=5)
        atom.in_site = self.site2.index
        atom.update_recent_site(atom.in_site)
        atom._frac_coords = np.array([0.3, 0.3, 0.3])

        # Call update_occupation to move the atom to site1
        self.collection.update_occupation(self.site1, atom)

        # Verify atom has been assigned to the new site
        self.assertEqual(atom.in_site, self.site1.index)

        # Verify no transition was recorded
        self.assertEqual(self.site2.transitions, Counter())

    def test_full_optimised_assignment_integration(self):
        """Integration test for the complete optimised site assignment algorithm."""
        # Simple setup: 3 sites, 2 atoms
        site1 = SphericalSite(frac_coords=np.array([0.0, 0.0, 0.0]), rcut=1.5)
        site2 = SphericalSite(frac_coords=np.array([0.1, 0.0, 0.0]), rcut=1.5)
        site3 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.5)

        collection = SphericalSiteCollection([site1, site2, site3])
        lattice = Lattice.cubic(10.0)
        structure = Structure(lattice, ["Li", "Li"], [[0.06, 0.05, 0.05], [0.15, 0.05, 0.05]])

        # Create atoms with some trajectory history
        atom1 = Atom(index=0)
        atom1.trajectory = [site1.index]
        atom1._recent_sites = [site1.index, None]
        # Inside both site1 and site2, and nearer site2's centre (0.81 A
        # against 0.93 A), so it goes to site1 only because that is its
        # recent site.
        atom1._frac_coords = np.array([0.06, 0.05, 0.05])

        atom2 = Atom(index=1)
        atom2.trajectory = [site1.index]
        atom2._recent_sites = [site1.index, None]
        atom2._frac_coords = np.array([0.15, 0.05, 0.05])  # Should move to site2

        atoms = [atom1, atom2]

        # Run the optimised assignment
        collection.assign_site_occupations(atoms, structure.lattice.matrix)

        # Verify key integration points
        self.assertEqual(atom1.in_site, site1.index)           # Correct assignment
        self.assertEqual(atom2.in_site, site2.index)           # Correct assignment
        self.assertEqual(site1.contains_atoms, [atom1.index])  # Site updated
        self.assertEqual(site2.contains_atoms, [atom2.index])  # Site updated


class TestAssignSiteOccupationsInteraction(unittest.TestCase):
    """Test interaction between assign_site_occupations and _get_priority_sites."""

    def setUp(self):
        Site._newid = 0
        self.lattice = Lattice.cubic(2.0)
        self.structure = Structure(self.lattice, ["Li"], [[0.1, 0.1, 0.1]])

        self.site1 = SphericalSite(frac_coords=np.array([0.1, 0.1, 0.1]), rcut=1.5)
        self.site2 = SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.5)
        self.collection = SphericalSiteCollection([self.site1, self.site2])

        self.atom = Atom(index=0)
        self.atom._frac_coords = np.array([0.2, 0.2, 0.2])
        self.atoms = [self.atom]

    def test_calls_generator_for_each_atom(self):
        """Test that _get_priority_sites is called once per atom."""
        lattice_matrix = self.structure.lattice.matrix
        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(self.atoms, lattice_matrix)
            self.collection._get_priority_sites.assert_called_once_with(self.atom, lattice_matrix)

    def test_calls_generator_for_multiple_atoms(self):
        """Test that _get_priority_sites is called for each atom."""
        # Add second atom
        atom2 = Atom(index=1)
        atom2._frac_coords = np.array([0.3, 0.3, 0.3])
        atoms = [self.atom, atom2]

        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(atoms, self.structure.lattice.matrix)
            self.assertEqual(self.collection._get_priority_sites.call_count, 2)

    def test_checks_sites_in_generator_order(self):
        """Test that sites are checked in the order returned by generator."""
        call_order = []
        self.site1.contains_atom = lambda atom, **kwargs: call_order.append(1) or False
        self.site2.contains_atom = lambda atom, **kwargs: call_order.append(2) or True

        with patch.object(self.collection, '_get_priority_sites') as mock_gen:
            mock_gen.return_value = [self.site2, self.site1]  # site2 first

            self.collection.assign_site_occupations(self.atoms, self.structure.lattice.matrix)

            self.assertEqual(call_order, [2])  # Only site2 checked (found there)

    def test_stops_checking_when_atom_found(self):
        """Test that checking stops as soon as atom is found."""
        with patch.object(self.site1, 'contains_atom', return_value=True) as mock_contains1:
            with patch.object(self.site2, 'contains_atom', return_value=False) as mock_contains2:
                with patch.object(self.collection, '_get_priority_sites') as mock_gen:
                    mock_gen.return_value = [self.site1, self.site2]

                    self.collection.assign_site_occupations(self.atoms, self.structure.lattice.matrix)

                    mock_contains1.assert_called_once()
                    mock_contains2.assert_not_called()

    def test_calls_update_occupation_when_found(self):
        """Test that update_occupation is called when atom found."""
        with patch.object(self.site1, 'contains_atom', return_value=True):
            with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
                with patch.object(self.collection, 'update_occupation') as mock_update:
                    self.collection.assign_site_occupations(self.atoms, self.structure.lattice.matrix)
                    mock_update.assert_called_once_with(self.site1, self.atom)

    def test_handles_atom_not_found(self):
        """Test behavior when atom not found in any site."""
        with patch.object(self.site1, 'contains_atom', return_value=False):
            with patch.object(self.collection, '_get_priority_sites', return_value=[self.site1]):
                with patch.object(self.collection, 'update_occupation') as mock_update:
                    self.collection.assign_site_occupations(self.atoms, self.structure.lattice.matrix)

                    mock_update.assert_not_called()
                    self.assertIsNone(self.atom.in_site)

    def test_resets_site_occupations(self):
        """Test that reset_site_occupations is called at start."""
        with patch.object(self.collection, 'reset_site_occupations') as mock_reset:
            with patch.object(self.collection, '_get_priority_sites', return_value=[]):
                self.collection.assign_site_occupations(self.atoms, self.structure.lattice.matrix)
                mock_reset.assert_called_once()

    def test_resets_atom_in_site(self):
        """Test that atom.in_site is reset to None."""
        self.atom.in_site = 999  # Set to some previous value

        with patch.object(self.collection, '_get_priority_sites', return_value=[]):
            self.collection.assign_site_occupations(self.atoms, self.structure.lattice.matrix)
            self.assertIsNone(self.atom.in_site)


class TestOverlappingSites(unittest.TestCase):
    """Tests for assigning atoms where spherical sites overlap."""

    def test_atom_leaving_its_site_goes_to_nearest_containing_site(self):
        """An atom that leaves its recent site for an overlap goes to the nearer centre."""
        Site._newid = 0
        previous = SphericalSite(frac_coords=np.array([0.1, 0.5, 0.5]), rcut=1.0)
        far = SphericalSite(frac_coords=np.array([0.45, 0.5, 0.5]), rcut=2.5)
        near = SphericalSite(frac_coords=np.array([0.55, 0.5, 0.5]), rcut=2.5)
        collection = SphericalSiteCollection([previous, far, near])
        atom = Atom(index=0)
        atom._recent_sites = [previous.index, None]
        # Inside both overlapping sites: 0.8 A from far's centre and 0.2 A
        # from near's. Ranked from the previous site's centre instead, far
        # would be checked first.
        atom._frac_coords = np.array([0.53, 0.5, 0.5])

        collection.assign_site_occupations([atom], np.eye(3) * 10.0)

        self.assertEqual(atom.in_site, near.index)


class TestReach(unittest.TestCase):
    """Tests for limiting the site search to the largest site radius."""

    def setUp(self):
        Site._newid = 0
        self.lattice_matrix = np.eye(3) * 10.0
        self.small = SphericalSite(frac_coords=np.array([0.3, 0.5, 0.5]), rcut=0.5)
        self.large = SphericalSite(frac_coords=np.array([0.6, 0.5, 0.5]), rcut=3.0)
        self.far_small = SphericalSite(frac_coords=np.array([0.9, 0.5, 0.5]), rcut=0.5)
        # The large site is neither first nor last, so its radius is not
        # picked out by position.
        self.collection = SphericalSiteCollection([self.small, self.large, self.far_small])
        self.atom = Atom(index=0)

    def test_sites_beyond_the_largest_radius_are_not_offered(self):
        """An atom further than the largest radius from every centre is offered no sites."""
        # 7.1 A from small's centre, 7.7 A from large's and 8.1 A from far_small's.
        self.atom._frac_coords = np.array([0.3, 0.0, 0.0])
        self.assertEqual(
            list(self.collection._get_priority_sites(self.atom, self.lattice_matrix)), [])

    def test_atom_found_in_large_site_beyond_smaller_radii(self):
        """The reach is the largest radius, so a large site beyond small ones is found."""
        # 1 A from small's centre, outside it, 2 A from large's, inside it,
        # and 5 A from far_small's.
        self.atom._frac_coords = np.array([0.4, 0.5, 0.5])
        self.collection.assign_site_occupations([self.atom], self.lattice_matrix)
        self.assertEqual(self.atom.in_site, self.large.index)

    def test_site_with_infinite_radius_is_offered_from_anywhere(self):
        """A site with an infinite radius is found for an atom far from every centre."""
        unbounded = SphericalSite(frac_coords=np.array([0.6, 0.5, 0.5]), rcut=np.inf)
        collection = SphericalSiteCollection([self.small, unbounded])
        # 7.1 A from small's centre and 7.7 A from unbounded's.
        self.atom._frac_coords = np.array([0.3, 0.0, 0.0])
        collection.assign_site_occupations([self.atom], self.lattice_matrix)
        self.assertEqual(self.atom.in_site, unbounded.index)


class TestNonFiniteAtomCoordinates(unittest.TestCase):
    """Tests for assigning atoms whose coordinates are not finite."""

    def test_non_finite_atom_coordinates_raise(self):
        """An atom with non-finite coordinates raises ValueError once it reaches the ranking."""
        Site._newid = 0
        collection = SphericalSiteCollection(
            [SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=1.0)])
        atom = Atom(index=0)
        atom._frac_coords = np.array([np.nan, 0.5, 0.5])
        with self.assertRaises(ValueError):
            collection.assign_site_occupations([atom], np.eye(3) * 10.0)


def _nearest_containing_site(sites, point, lattice_matrix):
    """Return the index of the containing site with the nearest centre, or None.

    Checks every site. Of containing sites at equal distances, the first in
    the list is returned.
    """
    nearest_index, nearest_distance = None, np.inf
    for site in sites:
        distance = mic_distance(site.frac_coords, point, lattice_matrix)
        if distance <= site.rcut and distance < nearest_distance:
            nearest_index, nearest_distance = site.index, distance
    return nearest_index


class TestMatchesBruteForce(unittest.TestCase):
    """Assignments match checking every site."""

    def test_atoms_without_history_go_to_the_nearest_containing_site(self):
        """Atoms with no recent site go to the containing site with the nearest centre."""
        rng = np.random.default_rng(3)
        lattice_matrix = Lattice.from_parameters(8.0, 9.0, 10.0, 75, 100, 110).matrix
        # Site indices differ from list positions.
        Site._newid = 100
        sites = [SphericalSite(frac_coords=centre, rcut=rcut)
                 for centre, rcut in zip(rng.random((40, 3)), rng.uniform(0.5, 2.5, 40))]
        collection = SphericalSiteCollection(sites)
        atoms = [Atom(index=i) for i in range(60)]
        # Some atoms are outside the cell, to test wrapping.
        for atom, point in zip(atoms, rng.uniform(-0.2, 1.2, (60, 3))):
            atom._frac_coords = point

        collection.assign_site_occupations(atoms, lattice_matrix)

        self.assertEqual(
            [atom.in_site for atom in atoms],
            [_nearest_containing_site(sites, atom.frac_coords, lattice_matrix) for atom in atoms])


class TestReachTolerance(unittest.TestCase):
    """Tests for widening the reach so that rounding cannot leave out a site."""

    def test_atom_at_the_radius_is_assigned_without_numba(self):
        """An atom exactly at the site radius is assigned when numba is not used."""
        # Without numba, the containment test and the reach query compute
        # the same distance by different numpy operations, which usually
        # round alike. These specific coordinates are a case where they do
        # not, on the machine where they were found: the reach query's
        # distance is one ulp larger than the containment test's, which is
        # the radius, so the atom is found only because the reach is
        # widened. The rounding may differ on other platforms, so the
        # tolerance is also tested directly in test_site_collection.py.
        centre = np.array([0.12428327649956394, 0.6706244146936303, 0.6471895115742501])
        point = np.array([1.4898346963218356, 0.1318870907354004, -0.1345752421468751])
        lattice_matrix = np.array([[10.76328146769828, 0.0, -3.6633263423816893],
                                   [-2.1357799113482145, 6.384391996928692, 3.732980239510001],
                                   [0.0, 0.0, 5.409735239361947]])
        with patch("site_analysis.distances.HAS_NUMBA", False):
            rcut = mic_distance(centre, point, lattice_matrix)
            site = SphericalSite(frac_coords=centre, rcut=rcut)
            collection = SphericalSiteCollection([site])
            atom = Atom(index=0)
            atom._frac_coords = point
            collection.assign_site_occupations([atom], lattice_matrix)
        self.assertEqual(atom.in_site, site.index)


if __name__ == '__main__':
    unittest.main()
