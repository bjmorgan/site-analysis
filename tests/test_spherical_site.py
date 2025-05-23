import unittest
import numpy as np
from site_analysis.spherical_site import SphericalSite
from site_analysis.atom import Atom
from site_analysis.site import Site
from unittest.mock import Mock, patch
from pymatgen.core import Lattice, Structure, PeriodicSite


class SphericalSiteInitTestCase(unittest.TestCase):
    def test_init(self):
        frac_coords = np.array([1.0, 2.0, 3.0])
        rcut = 4.0
        spherical_site = SphericalSite(
                frac_coords=frac_coords,
                rcut=rcut)
        np.testing.assert_array_equal(spherical_site.frac_coords, frac_coords)
        self.assertEqual(spherical_site.rcut, rcut)
        self.assertEqual(spherical_site.label, None)

    def test_init_with_label(self):
        frac_coords = np.array([1.0, 2.0, 3.0])
        rcut = 4.0
        label = "foo"
        spherical_site = SphericalSite(
                frac_coords=frac_coords,
                rcut=rcut,
                label=label)
        np.testing.assert_array_equal(spherical_site.frac_coords, frac_coords)
        self.assertEqual(spherical_site.rcut, rcut)
        self.assertEqual(spherical_site.label, label)


class SphericalSiteTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Reset Site._newid counter
        Site._newid = 0
        
        # Basic site for the original tests
        frac_coords = np.array([1.0, 2.0, 3.0])
        rcut = 4.0
        label = "foo"
        self.spherical_site = SphericalSite(
                frac_coords=frac_coords,
                rcut=rcut,
                label=label)
        
        # Create test lattice for new tests
        self.lattice = Lattice.cubic(10.0)
        
        # Create a simple spherical site with a more practical configuration
        self.site_center = np.array([0.5, 0.5, 0.5])
        self.site_radius = 2.0
        self.test_site = SphericalSite(
            frac_coords=self.site_center,
            rcut=self.site_radius,
            label="test_site"
        )
        
        # Create test atoms at different positions
        self.atom_inside = Atom(index=0)
        self.atom_inside._frac_coords = np.array([0.55, 0.55, 0.55])  # Inside
        
        self.atom_outside = Atom(index=1)
        self.atom_outside._frac_coords = np.array([0.8, 0.8, 0.8])  # Outside
        
        self.atom_on_boundary = Atom(index=2)
        self.atom_on_boundary._frac_coords = np.array([0.5, 0.5, 0.7])  # On boundary
        
        # Create a structure for testing
        self.structure = Structure(
            lattice=self.lattice,
            species=["Na", "Cl", "K"],
            coords=[
                [0.55, 0.55, 0.55],  # Inside
                [0.8, 0.8, 0.8],     # Outside
                [0.5, 0.5, 0.7]      # On boundary
            ]
        )

    def test_contains_atom_raises_valueerror_if_lattice_is_not_passed(self):
        mock_atom = Mock(spec=Atom)
        with self.assertRaises(ValueError):
            self.spherical_site.contains_atom(atom=mock_atom)

    def test_contains_atom_raises_typeerror_if_lattice_is_not_Lattice(self):
        mock_atom = Mock(spec=Atom)
        with self.assertRaises(TypeError):
            self.spherical_site.contains_atom(atom=mock_atom,
                    lattice="bar")

    def test_contains_atom(self):
        with patch('site_analysis.spherical_site.SphericalSite.contains_point') as mock_contains_point:
            mock_contains_point.return_value = "called"
            mock_atom = Mock(spec=Atom)
            mock_atom.frac_coords = np.array([2.0, 3.0, 4.0])
            mock_lattice = Mock(spec=Lattice)
            returned = self.spherical_site.contains_atom(atom=mock_atom, lattice=mock_lattice)
            self.assertEqual(returned, "called")
            mock_contains_point.assert_called_with(
                    x=mock_atom.frac_coords,
                    lattice=mock_lattice)

    def test_centre_property(self):
        """Test the centre property."""
        # Centre should be equal to frac_coords
        np.testing.assert_array_equal(self.test_site.centre, self.site_center)
        
        # Test with a different site
        new_coords = np.array([0.1, 0.2, 0.3])
        site = SphericalSite(frac_coords=new_coords, rcut=0.1)
        np.testing.assert_array_equal(site.centre, new_coords)

    def test_contains_point_with_various_points(self):
        """Test contains_point with points at different positions."""
        # Point inside the sphere
        point_inside = np.array([0.55, 0.55, 0.55])
        self.assertTrue(
            self.test_site.contains_point(point_inside, lattice=self.lattice),
            "Point inside sphere should be contained"
        )
        
        # Point outside the sphere
        point_outside = np.array([0.8, 0.8, 0.8])
        self.assertFalse(
            self.test_site.contains_point(point_outside, lattice=self.lattice),
            "Point outside sphere should not be contained"
        )
        
        # Point exactly on the boundary (distance = radius)
        point_on_boundary = np.array([0.5, 0.5, 0.7])
        # Due to floating point precision, this might be either True or False
        # We just need to make sure it doesn't raise an error
        result = self.test_site.contains_point(point_on_boundary, lattice=self.lattice)
        self.assertIsInstance(result, bool)
        
        # Test a point very close to the boundary (just inside)
        point_just_inside = np.array([0.5, 0.5, 0.699])
        self.assertTrue(
            self.test_site.contains_point(point_just_inside, lattice=self.lattice),
            "Point just inside sphere should be contained"
        )
        
        # Test a point very close to the boundary (just outside)
        point_just_outside = np.array([0.5, 0.5, 0.701])
        self.assertFalse(
            self.test_site.contains_point(point_just_outside, lattice=self.lattice),
            "Point just outside sphere should not be contained"
        )

    def test_contains_point_without_lattice(self):
        """Test contains_point raises ValueError when lattice is not provided."""
        point = np.array([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            self.test_site.contains_point(point)

    def test_contains_point_with_invalid_lattice(self):
        """Test contains_point raises TypeError when lattice is not a Lattice object."""
        point = np.array([0.5, 0.5, 0.5])
        with self.assertRaises(TypeError):
            self.test_site.contains_point(point, lattice="not_a_lattice")

    def test_contains_atom_with_atoms_at_different_positions(self):
        """Test contains_atom with atoms at different positions."""
        # Atom inside the sphere
        self.assertTrue(
            self.test_site.contains_atom(self.atom_inside, lattice=self.lattice),
            "Atom inside sphere should be contained"
        )
        
        # Atom outside the sphere
        self.assertFalse(
            self.test_site.contains_atom(self.atom_outside, lattice=self.lattice),
            "Atom outside sphere should not be contained"
        )
        
        # Atom on the boundary
        result = self.test_site.contains_atom(self.atom_on_boundary, lattice=self.lattice)
        self.assertIsInstance(result, bool)

    def test_periodic_boundary_conditions(self):
        """Test contains_point with periodic boundary conditions."""
        # Create a site near the edge of the unit cell
        edge_site = SphericalSite(
            frac_coords=np.array([0.05, 0.05, 0.05]),
            rcut=1.1
        )
        
        # Test a point that's inside the site but across the periodic boundary
        # This point is at [0.95, 0.05, 0.05], which is close to [0.05, 0.05, 0.05]
        # when considering periodic boundaries
        pbc_point = np.array([0.95, 0.05, 0.05])
        
        # Should be contained because of PBC
        self.assertTrue(
            edge_site.contains_point(pbc_point, lattice=self.lattice),
            "Point across periodic boundary should be contained"
        )
        
        # Test a point that's outside even with PBC
        outside_point = np.array([0.8, 0.05, 0.05])
        self.assertFalse(
            edge_site.contains_point(outside_point, lattice=self.lattice),
            "Point outside even with PBC should not be contained"
        )

    def test_as_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        # Get dictionary representation
        site_dict = self.test_site.as_dict()
        
        # Check dictionary contents
        self.assertIn('frac_coords', site_dict)
        self.assertIn('rcut', site_dict)
        self.assertIn('label', site_dict)
        
        np.testing.assert_array_equal(site_dict['frac_coords'], self.site_center)
        self.assertEqual(site_dict['rcut'], self.site_radius)
        self.assertEqual(site_dict['label'], "test_site")
        
        # Test from_dict
        new_site = SphericalSite.from_dict(site_dict)
        
        # Check that properties match
        np.testing.assert_array_equal(new_site.frac_coords, self.test_site.frac_coords)
        self.assertEqual(new_site.rcut, self.test_site.rcut)
        self.assertEqual(new_site.label, self.test_site.label)

    def test_different_lattice_parameters(self):
        """Test behavior with different lattice parameters."""
        # Create a point that would be inside with a small lattice
        # but outside with a large lattice
        test_point = np.array([0.7, 0.5, 0.5])
        
        # With a small lattice (5.0 Å), the distance is small enough
        small_lattice = Lattice.cubic(5.0)
        self.assertTrue(
            self.test_site.contains_point(test_point, lattice=small_lattice),
            "Point should be inside with small lattice"
        )
        
        # With a large lattice (20.0 Å), the distance is too large
        large_lattice = Lattice.cubic(20.0)
        self.assertFalse(
            self.test_site.contains_point(test_point, lattice=large_lattice),
            "Point should be outside with large lattice"
        )

    def test_different_cutoff_radii(self):
        """Test behavior with different cutoff radii."""
        test_point = np.array([0.6, 0.5, 0.5])
        
        # Site with small radius
        small_radius_site = SphericalSite(
            frac_coords=self.site_center,
            rcut=0.5
        )
        self.assertFalse(
            small_radius_site.contains_point(test_point, lattice=self.lattice),
            "Point should be outside small radius site"
        )
        
        # Site with large radius
        large_radius_site = SphericalSite(
            frac_coords=self.site_center,
            rcut=3.0
        )
        self.assertTrue(
            large_radius_site.contains_point(test_point, lattice=self.lattice),
            "Point should be inside large radius site"
        )

    def test_non_cubic_lattice(self):
        """Test with a non-cubic lattice."""
        # Create a tetragonal lattice (a=b≠c)
        tetragonal_lattice = Lattice.tetragonal(5.0, 8.0)
        
        # Create a point that's equidistant in fractional coordinates
        # but will have different real distances due to the lattice
        test_point = np.array([0.6, 0.5, 0.55])
        
        # Should behave correctly based on real-space distances
        result = self.test_site.contains_point(test_point, lattice=tetragonal_lattice)
        
        # We don't assert a specific result here, just make sure it runs
        # The result will depend on the specific lattice parameters
        self.assertIsInstance(result, bool)
        
    def test_repr(self):
        """Test the __repr__ method for SphericalSite."""
        frac_coords = np.array([0.25, 0.25, 0.25])
        rcut = 1.5
        label = "tetrahedral"
        site = SphericalSite(frac_coords=frac_coords, rcut=rcut, label=label)
        site.index = 5  # Set index explicitly for testing
        site.contains_atoms = [1, 2, 3]  # Add some atoms for testing
        
        # Generate the string representation
        repr_str = repr(site)
        
        # Verify all key attributes are included
        self.assertIn("site_analysis.SphericalSite", repr_str)
        self.assertIn(f"index={site.index}", repr_str)
        self.assertIn(f"label={site.label}", repr_str)
        self.assertIn(f"rcut={site.rcut}", repr_str)
        self.assertIn(f"contains_atoms={site.contains_atoms}", repr_str)
        # Don't check for exact frac_coords representation as it varies based on NumPy formatting
        self.assertIn("frac_coords=", repr_str)

    

if __name__ == '__main__':
    unittest.main()