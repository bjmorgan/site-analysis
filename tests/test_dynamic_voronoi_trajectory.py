import unittest
import numpy as np
from site_analysis.trajectory import Trajectory
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.atom import Atom
from pymatgen.core import Structure, Lattice

class DynamicVoronoiTrajectoryIntegrationTestCase(unittest.TestCase):
	"""Integration tests for DynamicVoronoiSite with Trajectory class."""
	
	def test_dynamic_voronoi_sites_simple(self):
		"""Test a simple structure with two metal atoms and one oxygen atom.
		
		Creates a structure with two rhenium atoms along the x-axis and an
		oxygen atom in the middle. Verifies that the oxygen atom is correctly
		assigned to the site defined by the two rhenium atoms.
		"""
		# Create a very simple structure with 2 Re atoms and 1 O atom in the middle
		lattice = Lattice.cubic(10.0)  # Large cell to avoid PBC issues
		re_coords = [[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]]  # Two Re atoms along x axis
		o_coords = [[0.5, 0.5, 0.5]]  # O atom in the middle
		coords = re_coords + o_coords
		species = ['Re', 'Re', 'O']
		structure = Structure(lattice, species, coords)
		
		# Create an Atom object for the O atom (index 2)
		o_atom = Atom(index=2)
		
		# Create a DynamicVoronoiSite using the two Re atoms
		site = DynamicVoronoiSite(reference_indices=[0, 1])  # Re atoms at indices 0 and 1
		
		# Create a Trajectory object
		trajectory = Trajectory(sites=[site], atoms=[o_atom])
		
		# Analyze the structure to assign atoms to sites
		trajectory.analyse_structure(structure)
		
		# Verify that the O atom is assigned to the site
		self.assertEqual(o_atom.in_site, site.index)
		self.assertIn(o_atom.index, site.contains_atoms)
		
		# Verify the site center is calculated correctly
		# The center should be at (0.5, 0.5, 0.5)
		np.testing.assert_array_almost_equal(site.centre, np.array([0.5, 0.5, 0.5]))
		
	def test_dynamic_voronoi_sites_simple2(self):
		"""Test a simple structure with two metal atoms and one oxygen atom.
		
		Creates a structure with two rhenium atoms along the x-axis and an
		oxygen atom in the middle. Verifies that the oxygen atom is correctly
		assigned to the site defined by the two rhenium atoms.
		"""
		# Create a very simple structure with 2 Re atoms and 1 O atom in the middle
		lattice = Lattice.cubic(10.0)  # Large cell to avoid PBC issues
		re_coords = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]]  # Two Re atoms along x axis
		o_coords = [[0.0, 0.25, 0.0]]  # O atom in the middle
		coords = re_coords + o_coords
		species = ['Re', 'Re', 'O']
		structure = Structure(lattice, species, coords)
		
		# Create an Atom object for the O atom (index 2)
		o_atom = Atom(index=2)
		
		# Create a DynamicVoronoiSite using the two Re atoms
		site = DynamicVoronoiSite(reference_indices=[0, 1])  # Re atoms at indices 0 and 1
		
		# Create a Trajectory object
		trajectory = Trajectory(sites=[site], atoms=[o_atom])
		
		# Analyze the structure to assign atoms to sites
		trajectory.analyse_structure(structure)
		
		# Verify that the O atom is assigned to the site
		self.assertEqual(o_atom.in_site, site.index)
		self.assertIn(o_atom.index, site.contains_atoms)
		
		# Verify the site center is calculated correctly
		# The center should be at (0.5, 0.5, 0.5)
		np.testing.assert_array_almost_equal(site.centre, np.array([0.0, 0.25, 0.0]))
	
	def test_dynamic_voronoi_sites_multiple(self):
		"""Test a structure with multiple sites and oxygen atoms.
		
		Creates a ReO3-like fragment with 3 metal atoms in a V-shape and
		2 oxygen atoms positioned between pairs of metal atoms. Verifies
		that each oxygen atom is assigned to the correct site.
		"""
		# Create a structure with 3 Re atoms and 2 O atoms
		lattice = Lattice.cubic(10.0)  # Large cell to avoid PBC issues
		
		# Re atoms forming a V shape
		re_coords = [
			[0.3, 0.3, 0.5],  # Re0
			[0.5, 0.5, 0.5],  # Re1
			[0.7, 0.3, 0.5]   # Re2
		]
		
		# O atoms at the middle of each Re-Re pair
		o_coords = [
			[0.4, 0.4, 0.5],  # O0 - between Re0 and Re1
			[0.6, 0.4, 0.5]   # O1 - between Re1 and Re2
		]
		
		coords = re_coords + o_coords
		species = ['Re', 'Re', 'Re', 'O', 'O']
		structure = Structure(lattice, species, coords)
		
		# Create Atom objects for the O atoms
		o_atoms = [Atom(index=3), Atom(index=4)]  # Indices 3 and 4
		
		# Create DynamicVoronoiSite objects
		site1 = DynamicVoronoiSite(reference_indices=[0, 1])  # Re0-Re1
		site2 = DynamicVoronoiSite(reference_indices=[1, 2])  # Re1-Re2
		
		# Create a Trajectory object
		trajectory = Trajectory(sites=[site1, site2], atoms=o_atoms)
		
		# Analyze the structure to assign atoms to sites
		trajectory.analyse_structure(structure)
		
		# Verify that O0 (index 3) is assigned to site1 (Re0-Re1)
		self.assertEqual(trajectory.atom_by_index(3).in_site, site1.index)
		self.assertIn(3, site1.contains_atoms)
		
		# Verify that O1 (index 4) is assigned to site2 (Re1-Re2)
		self.assertEqual(trajectory.atom_by_index(4).in_site, site2.index)
		self.assertIn(4, site2.contains_atoms)
		
		# Verify the site centers are calculated correctly
		np.testing.assert_array_almost_equal(site1.centre, np.array([0.4, 0.4, 0.5]))
		np.testing.assert_array_almost_equal(site2.centre, np.array([0.6, 0.4, 0.5]))
		
	def test_dynamic_voronoi_sites_reo3(self):
		"""Test with a proper ReO3-type structure.
		
		Creates a ReO3 structure using space group symmetry. Uses the get_nearest_neighbour_indices
		function to identify the pairs of Re atoms nearest to each O atom and creates 
		DynamicVoronoiSites accordingly. Verifies that each O atom is assigned to the correct site.
		"""
		from site_analysis.tools import get_nearest_neighbour_indices
		
		# Create a ReO3 structure using space group
		a = 3.75  # Approximate ReO3 lattice parameter in Angstroms
		lattice = Lattice.cubic(a)
		species = ["Re", "O"]
		coords = [[0.0, 0.0, 0.0],  # Re
				  [0.5, 0.0, 0.0]]  # O - will be expanded to all equivalent positions
		
		# Create the structure with the correct space group symmetry
		structure = Structure.from_spacegroup("Pm-3m", lattice, species, coords) * [3,3,3]
		
		# Find all O atoms
		o_indices = [i for i, site in enumerate(structure) if site.species_string == "O"]
		
		# Create a reference structure with only O atoms
		o_coords = [structure[i].frac_coords for i in o_indices]
		o_species = ["O"] * len(o_indices)
		o_structure = Structure(structure.lattice, o_species, o_coords)
		
		# Find the 2 nearest Re atoms for each O atom
		nearest_re_pairs = get_nearest_neighbour_indices(
			structure=structure,
			ref_structure=o_structure,
			vertex_species=["Re"],
			n_coord=2
		)
		
		# Create Atom objects for all O atoms
		o_atoms = [Atom(index=i) for i in o_indices]
		
		# Create DynamicVoronoiSite objects based on the nearest Re pairs
		sites = [DynamicVoronoiSite(reference_indices=re_pair)
			for i, re_pair in enumerate(nearest_re_pairs)
		]
		
		# Create a Trajectory object
		trajectory = Trajectory(sites=sites, atoms=o_atoms)
		
		# Analyze the structure to assign atoms to sites
		trajectory.analyse_structure(structure)
		
		# Verify each O atom is assigned to a site
		for i, o_atom in enumerate(o_atoms):
			# Each O atom should be assigned to a site
			self.assertIsNotNone(o_atom.in_site)
			
			# The corresponding site should contain this atom
			site = trajectory.site_by_index(o_atom.in_site)
			self.assertIn(o_atom.index, site.contains_atoms)
			
			# The site center should be very close to the O atom position
			o_coords = structure[o_atom.index].frac_coords
			np.testing.assert_array_almost_equal(site.centre, o_coords, decimal=5)
			
			# Verify that the site is defined by the two nearest Re atoms
			re_pair = nearest_re_pairs[i]
			actual_re_indices = sorted(site.reference_indices)
			expected_re_indices = sorted(re_pair)
			
			self.assertEqual(actual_re_indices, expected_re_indices)
		
	

if __name__ == '__main__':
	unittest.main()