import unittest
import numpy as np
from site_analysis.spherical_site import SphericalSite
from site_analysis.atom import Atom
from unittest.mock import Mock, patch
from pymatgen.core import Lattice

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
        frac_coords = np.array([1.0, 2.0, 3.0])
        rcut = 4.0
        label = "foo"
        spherical_site = SphericalSite(
                frac_coords=frac_coords,
                rcut=rcut,
                label=label)
        self.spherical_site = spherical_site

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

if __name__ == "__main__":
    unittest.main()
