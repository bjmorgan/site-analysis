import unittest
from unittest.mock import Mock, patch
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.voronoi_site_collection import VoronoiSiteCollection
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.atom import Atom
from site_analysis.site import Site


class VoronoiSiteCollectionTestCase(unittest.TestCase):
	
	def setUp(self):
		"""Set up test fixtures."""
		# Reset Site._newid counter
		Site._newid = 0
		
		# Create a real lattice (for proper method behavior)
		self.lattice = Lattice.cubic(10.0)
		
		# Create VoronoiSites with real objects for their core attributes
		self.site1 = VoronoiSite(
			frac_coords=np.array([0.1, 0.1, 0.1]),
			label="site1"
		)
		self.site2 = VoronoiSite(
			frac_coords=np.array([0.5, 0.5, 0.5]),
			label="site2"
		)
		
		# Create test atoms
		self.atom1 = Atom(index=0)
		self.atom1._frac_coords = np.array([0.15, 0.15, 0.15])
		
		self.atom2 = Atom(index=1)
		self.atom2._frac_coords = np.array([0.45, 0.45, 0.45])
		
		self.atoms = [self.atom1, self.atom2]
		
		# Create a test structure with a real lattice
		self.structure = Structure(
			lattice=self.lattice,
			species=["Na", "Na"],
			coords=[[0.15, 0.15, 0.15], [0.45, 0.45, 0.45]]
		)
		
		# Create the collection
		self.collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
	
	def test_initialization_type_checking(self):
		"""Test that initialization enforces VoronoiSite types."""
		# Valid initialization with VoronoiSites
		collection = VoronoiSiteCollection(sites=[self.site1, self.site2])
		self.assertEqual(collection.sites, [self.site1, self.site2])
		
		# Invalid initialization with non-VoronoiSite
		non_voronoi_site = Mock()
		non_voronoi_site.index = 2
		
		with self.assertRaises(TypeError):
			VoronoiSiteCollection(sites=[self.site1, non_voronoi_site])

	def test_init_accepts_a_generator_of_sites(self):
		"""A collection built from a generator holds every site."""
		collection = VoronoiSiteCollection(s for s in [self.site1, self.site2])
		self.assertEqual(collection.sites, [self.site1, self.site2])
	
	def test_analyse_structure(self):
		"""Test that analyse_structure assigns coordinates and calls assign_site_occupations."""
		with patch.object(Atom, 'assign_coords') as mock_assign_coords, \
			 patch.object(self.collection, 'assign_site_occupations') as mock_assign:

			self.collection.analyse_structure(self.atoms, self.structure)

			self.assertEqual(mock_assign_coords.call_count, 2)
			mock_assign.assert_called_once()
			args = mock_assign.call_args[0]
			self.assertIs(args[0], self.atoms)
			np.testing.assert_array_equal(args[1], self.structure.lattice.matrix)
	
	def test_assigns_atoms_to_nearest_site(self):
		"""Each atom is assigned to the site with the nearest centre, across boundaries."""
		atom3 = Atom(index=2)
		atom3._frac_coords = np.array([0.95, 0.95, 0.95])  # nearest site1 through the boundary
		self.collection.assign_site_occupations(
			[self.atom1, self.atom2, atom3], self.lattice.matrix)
		self.assertEqual(self.site1.contains_atoms, [0, 2])
		self.assertEqual(self.site2.contains_atoms, [1])

	def test_equidistant_atom_assigned_to_first_site(self):
		"""An atom equidistant from two sites is assigned to the first."""
		site_a = VoronoiSite(frac_coords=np.array([0.25, 0.5, 0.5]))
		site_b = VoronoiSite(frac_coords=np.array([0.75, 0.5, 0.5]))
		collection = VoronoiSiteCollection(sites=[site_a, site_b])
		atom = Atom(index=0)
		atom._frac_coords = np.array([0.5, 0.5, 0.5])
		collection.assign_site_occupations([atom], self.lattice.matrix)
		self.assertEqual(site_a.contains_atoms, [0])
		self.assertEqual(site_b.contains_atoms, [])
	
	def test_nearest_site_uses_the_lattice(self):
		"""Distances to sites are Cartesian, so the shape of the lattice matters."""
		# Nearer site_a in fractional coordinates, but nearer site_b in
		# Cartesian coordinates (1.2 A against 4.0 A).
		site_a = VoronoiSite(frac_coords=np.array([0.5, 0.5, 0.3]))
		site_b = VoronoiSite(frac_coords=np.array([0.2, 0.5, 0.5]))
		collection = VoronoiSiteCollection(sites=[site_a, site_b])
		atom = Atom(index=0)
		atom._frac_coords = np.array([0.5, 0.5, 0.5])
		collection.assign_site_occupations([atom], Lattice.orthorhombic(4.0, 4.0, 20.0).matrix)
		self.assertEqual(site_a.contains_atoms, [])
		self.assertEqual(site_b.contains_atoms, [0])

	def test_nearest_site_in_a_hexagonal_cell(self):
		"""Distances use the rows of a non-symmetric lattice matrix as the lattice vectors."""
		# The atom is 1.20 A from near_site and 1.28 A from far_site. With
		# the lattice transposed, far_site would be nearer.
		far_site = VoronoiSite(frac_coords=np.array([0.5, 0.82, 0.5]))
		near_site = VoronoiSite(frac_coords=np.array([0.8, 0.8, 0.5]))
		collection = VoronoiSiteCollection(sites=[far_site, near_site])
		atom = Atom(index=0)
		atom._frac_coords = np.array([0.5, 0.5, 0.5])
		collection.assign_site_occupations([atom], Lattice.hexagonal(4.0, 6.0).matrix)
		self.assertEqual(near_site.contains_atoms, [0])
		self.assertEqual(far_site.contains_atoms, [])

	def test_empty_atoms_list(self):
		"""Test behaviour with empty atoms list."""
		self.site1.contains_atoms = [1, 2]
		self.site2.contains_atoms = [3, 4]
		self.collection.assign_site_occupations([], self.lattice.matrix)
		self.assertEqual(self.site1.contains_atoms, [])
		self.assertEqual(self.site2.contains_atoms, [])
	
	def test_reset_site_occupations(self):
		"""Test that reset_site_occupations clears the contains_atoms lists."""
		# Add some atoms to the sites
		self.site1.contains_atoms = [0, 1]
		self.site2.contains_atoms = [2, 3]
		
		# Reset the sites
		self.collection.reset_site_occupations()
		
		# Verify contains_atoms lists are empty
		self.assertEqual(self.site1.contains_atoms, [])
		self.assertEqual(self.site2.contains_atoms, [])
	
	def test_update_occupation(self):
		"""Test the update_occupation method."""
		# Create a test atom
		atom = Atom(index=3)
		atom._frac_coords = np.array([0.3, 0.3, 0.3])
		
		# Call update_occupation to assign the atom to a site
		self.collection.update_occupation(self.site1, atom)
		
		# Verify atom has been assigned to the site
		self.assertEqual(atom.in_site, self.site1.index)
		
		# Verify site contains the atom
		self.assertIn(atom.index, self.site1.contains_atoms)
	
	def test_integration(self):
		"""Integration test with actual behavior."""
		# Reset site occupations
		self.site1.contains_atoms = []
		self.site2.contains_atoms = []
		
		# Reset atom site assignments
		self.atom1.in_site = None
		self.atom2.in_site = None
		
		# Set up atoms with coordinates that make them clearly closer to specific sites
		self.atom1._frac_coords = np.array([0.1, 0.1, 0.11])  # Very close to site1
		self.atom2._frac_coords = np.array([0.5, 0.5, 0.51])  # Very close to site2
		
		# Call the method with real objects
		self.collection.analyse_structure(self.atoms, self.structure)
		
		# Atom1 should be assigned to site1 (closest) and atom2 to site2
		self.assertEqual(self.atom1.in_site, self.site1.index)
		self.assertEqual(self.atom2.in_site, self.site2.index)
		
		# Verify sites contain the correct atoms
		self.assertIn(self.atom1.index, self.site1.contains_atoms)
		self.assertIn(self.atom2.index, self.site2.contains_atoms)
		

if __name__ == '__main__':
	unittest.main()