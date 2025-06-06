"""Tests for coordinate wrapping in site analysis with unwrapped coordinates.

These tests verify that all site types work correctly when input structures
contain unwrapped coordinates from sources like VASP AIMD simulations.
"""

import unittest
import numpy as np
from pymatgen.core import Structure, Lattice
from site_analysis.atom import atoms_from_structure
from site_analysis.spherical_site import SphericalSite
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite


class UnwrappedCoordinatesSiteTestCase(unittest.TestCase):
	"""Tests for site occupation with unwrapped coordinates."""

	def test_spherical_site_with_unwrapped_coordinates(self):
		"""Test that spherical sites work correctly with unwrapped coordinates."""
		# Create structure with unwrapped coordinates
		lattice = Lattice.cubic(5.0)
		structure = Structure(
			lattice=lattice,
			species=["Li"],
			coords=[[1.1, 1.2, 1.3]]  # Unwrapped coordinates
		)
		
		# Create atom through normal pipeline
		atoms = atoms_from_structure(structure, "Li")
		atom = atoms[0]
		
		# Create spherical site at wrapped position
		site = SphericalSite(
			frac_coords=np.array([0.1, 0.2, 0.3]),
			rcut=0.1
		)
		
		# Should contain the atom after coordinate wrapping
		self.assertTrue(site.contains_atom(atom, lattice))

	def test_voronoi_site_with_unwrapped_coordinates(self):
		"""Test that Voronoi sites work correctly with unwrapped coordinates."""
		# Create structure with unwrapped coordinates
		lattice = Lattice.cubic(5.0)
		structure = Structure(
			lattice=lattice,
			species=["Li"],
			coords=[[1.1, 1.2, 1.3]]  # Unwrapped coordinates
		)
		
		# Create atom through normal pipeline
		atoms = atoms_from_structure(structure, "Li")
		atom = atoms[0]
		
		# Create multiple Voronoi sites - atom should be closest to the one at wrapped position
		site_target = VoronoiSite(frac_coords=np.array([0.1, 0.2, 0.3]))  # Should contain atom
		site_other1 = VoronoiSite(frac_coords=np.array([0.5, 0.5, 0.5]))  # Further away
		site_other2 = VoronoiSite(frac_coords=np.array([0.8, 0.8, 0.8]))  # Further away
		
		# Test using VoronoiSiteCollection for proper Voronoi assignment
		from site_analysis.voronoi_site_collection import VoronoiSiteCollection
		
		sites = [site_target, site_other1, site_other2]
		collection = VoronoiSiteCollection(sites)
		collection.analyse_structure(atoms, structure)
		
		# The Li atom should be assigned to the target site (closest to wrapped position)
		self.assertEqual(len(site_target.contains_atoms), 1)
		self.assertEqual(len(site_other1.contains_atoms), 0)
		self.assertEqual(len(site_other2.contains_atoms), 0)
		self.assertIn(atoms[0].index, site_target.contains_atoms)

	def test_polyhedral_site_with_unwrapped_coordinates(self):
		"""Test that polyhedral sites work correctly with unwrapped coordinates."""
		# Create structure with mixed wrapped/unwrapped coordinates
		lattice = Lattice.cubic(5.0)
		structure = Structure(
			lattice=lattice,
			species=["Li", "O", "O", "O", "O"],
			coords=[
				[1.1, 1.2, 1.3],  # Li at unwrapped position (should wrap to [0.1, 0.2, 0.3])
				[0.0, 0.0, 0.0],  # O vertices in wrapped coordinates forming tetrahedron
				[0.2, 0.2, 0.2],  # around the wrapped Li position [0.1, 0.2, 0.3]
				[0.2, 0.0, 0.4],
				[0.0, 0.2, 0.4]
			]
		)
		
		# Create atoms through normal pipeline
		atoms = atoms_from_structure(structure, "Li")
		
		# Create polyhedral site
		site = PolyhedralSite(vertex_indices=[1, 2, 3, 4])
		
		# Use PolyhedralSiteCollection to test the proper analysis pipeline
		from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection
		
		sites = [site]
		collection = PolyhedralSiteCollection(sites)
		collection.analyse_structure(atoms, structure)
		
		# The Li atom should be assigned to the polyhedral site after coordinate wrapping
		self.assertEqual(len(site.contains_atoms), 1)
		self.assertIn(atoms[0].index, site.contains_atoms)

	def test_dynamic_voronoi_site_with_unwrapped_coordinates(self):
		"""Test that dynamic Voronoi sites work correctly with unwrapped coordinates."""
		# Create structure with unwrapped coordinates
		lattice = Lattice.cubic(5.0)
		structure = Structure(
			lattice=lattice,
			species=["Li", "O", "O", "O", "O"],
			coords=[
				[1.1, 1.2, 1.3],  # Li at unwrapped position
				[1.0, 1.2, 1.3],  # O reference atoms near target position
				[1.2, 1.2, 1.3],
				[1.5, 1.5, 1.5],  # O reference atoms for other site
				[1.7, 1.5, 1.5]
			]
		)
		
		# Create atoms through normal pipeline
		atoms = atoms_from_structure(structure, "Li")
		atom = atoms[0]
		
		# Create multiple dynamic Voronoi sites
		site_target = DynamicVoronoiSite(reference_indices=[1, 2])  # Should contain atom
		site_other = DynamicVoronoiSite(reference_indices=[3, 4])   # Further away
		
		# Calculate centres for both sites
		site_target.calculate_centre(structure)
		site_other.calculate_centre(structure)
		
		# Test using DynamicVoronoiSiteCollection for proper assignment
		from site_analysis.dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection
		
		sites = [site_target, site_other]
		collection = DynamicVoronoiSiteCollection(sites)
		collection.analyse_structure(atoms, structure)
		
		# The Li atom should be assigned to the target site (closest to wrapped position)
		self.assertEqual(len(site_target.contains_atoms), 1)
		self.assertEqual(len(site_other.contains_atoms), 0)
		self.assertIn(atoms[0].index, site_target.contains_atoms)


if __name__ == '__main__':
	unittest.main()