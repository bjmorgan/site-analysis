import unittest
import numpy as np
from pymatgen.core import Lattice


class TestFracToCart(unittest.TestCase):
    """Tests for fractional to Cartesian coordinate conversion."""

    def test_single_point_cubic(self):
        """Conversion matches pymatgen for a cubic lattice."""
        from site_analysis.distances import frac_to_cart
        lattice = Lattice.cubic(10.0)
        frac = np.array([0.1, 0.2, 0.3])
        expected = lattice.get_cartesian_coords(frac)
        result = frac_to_cart(frac, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_single_point_triclinic(self):
        """Conversion matches pymatgen for a triclinic lattice."""
        from site_analysis.distances import frac_to_cart
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac = np.array([0.3, 0.5, 0.7])
        expected = lattice.get_cartesian_coords(frac)
        result = frac_to_cart(frac, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_batch_points(self):
        """Conversion works for an (N, 3) array of points."""
        from site_analysis.distances import frac_to_cart
        lattice = Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60)
        frac = np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6],
                         [0.7, 0.8, 0.9]])
        expected = lattice.get_cartesian_coords(frac)
        result = frac_to_cart(frac, lattice.matrix)
        np.testing.assert_allclose(result, expected, atol=1e-12)
