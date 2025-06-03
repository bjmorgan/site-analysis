import unittest
import numpy as np
from pymatgen.core import Structure, Lattice

from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.site import Site
from site_analysis.pbc_utils import apply_legacy_pbc_correction, unwrap_vertices_to_reference_centre, unwrap_vertices_vectorised


class TestLegacyPBCCorrection(unittest.TestCase):
	"""Tests for the isolated legacy PBC correction algorithm."""
	
	def test_apply_legacy_pbc_correction_no_correction_needed(self):
		"""Test that coordinates with spread < 0.5 remain unchanged."""
		# All coordinates well within [0, 1), no correction needed
		frac_coords = np.array([
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.2], 
			[0.3, 0.3, 0.3],
			[0.4, 0.4, 0.4]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		# Should remain unchanged
		np.testing.assert_array_equal(result, frac_coords)
		
	def test_apply_legacy_pbc_correction_with_correction_needed(self):
		"""Test coordinates with spread > 0.5 get corrected properly."""
		frac_coords = np.array([
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.9],
			[0.3, 0.3, 0.1],
			[0.4, 0.4, 0.1]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		expected = np.array([
			[0.1, 0.1, 1.1],
			[0.2, 0.2, 0.9],
			[0.3, 0.3, 1.1],
			[0.4, 0.4, 1.1]
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_apply_legacy_pbc_correction_mixed_dimensions(self):
		"""Test correction applied independently to each dimension."""
		frac_coords = np.array([
			[0.1, 0.1, 0.1],
			[0.4, 0.9, 0.2],
			[0.2, 0.1, 0.4]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		expected = np.array([
			[0.1, 1.1, 0.1],
			[0.4, 0.9, 0.2],
			[0.2, 1.1, 0.4]
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_apply_legacy_pbc_correction_negative_coordinates(self):
		"""Test that negative coordinates are handled correctly."""
		frac_coords = np.array([
			[0.1, -0.1, 0.2],
			[0.2, -0.05, 0.3],
			[0.3, 0.0, 0.4],
			[0.4, 0.1, 0.5]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		np.testing.assert_array_equal(result, frac_coords)
	
	def test_apply_legacy_pbc_correction_edge_case_coordinates(self):
		"""Test edge cases with coordinates at exactly 0.0, 0.5, 1.0."""
		frac_coords = np.array([
			[0.0, 0.0, 0.0],
			[0.5, 0.5, 0.5],
			[1.0, 1.0, 1.0]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		expected = np.array([
			[1.0, 1.0, 1.0],
			[0.5, 0.5, 0.5],
			[1.0, 1.0, 1.0]
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_apply_legacy_pbc_correction_all_coordinates_above_half(self):
		"""Test that no shifting occurs when all coordinates > 0.5 even with large spread."""
		frac_coords = np.array([
			[0.6, 0.6, 0.6],
			[0.7, 0.7, 0.7],
			[0.8, 0.8, 0.8],
			[0.9, 0.9, 0.9]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		np.testing.assert_array_equal(result, frac_coords)
	
	def test_apply_legacy_pbc_correction_large_spread_all_above_half(self):
		"""Test large spread with all coordinates > 0.5 - no shifting should occur."""
		frac_coords = np.array([
			[0.6, 0.6, 0.6],
			[0.7, 0.7, 0.7],
			[0.8, 0.8, 0.8],
			[1.2, 1.2, 1.2]
		])
		
		result = apply_legacy_pbc_correction(frac_coords)
		
		np.testing.assert_array_equal(result, frac_coords)
	
	
class TestCurrentPBCBehaviorRegression(unittest.TestCase):
	"""Regression tests to preserve correct current PBC behavior during refactoring."""

	def setUp(self):
		"""Set up test fixtures."""
		Site._newid = 0
		self.lattice = Lattice.cubic(10.0)

	def test_polyhedral_site_no_pbc_correction_needed(self):
		"""Test case where no PBC correction is needed - should remain unchanged."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.2],
			[0.3, 0.3, 0.3],
			[0.4, 0.4, 0.4]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)
		
		# Should remain unchanged (this works correctly)
		expected_coords = np.array(coords)
		np.testing.assert_array_almost_equal(site.vertex_coords, expected_coords, decimal=7)

	def test_dynamic_voronoi_site_no_pbc_correction_needed(self):
		"""Test case where no PBC correction is needed - should calculate centre correctly."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.2],
			[0.3, 0.3, 0.3],
			[0.4, 0.4, 0.4]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure)
		
		# Should calculate centre correctly (this works correctly)
		expected_centre = np.mean(coords, axis=0)
		np.testing.assert_array_almost_equal(site.centre, expected_centre, decimal=7)

	def test_polyhedral_site_working_pbc_case(self):
		"""Test a case where current PBC correction works correctly."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.9],   # Large z-coordinate, should trigger correction
			[0.3, 0.3, 0.1],
			[0.4, 0.4, 0.1]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)
		
		# Current algorithm should correctly shift small z-coordinates
		expected_coords = np.array([
			[0.1, 0.1, 1.1],   # z < 0.5, gets +1.0
			[0.2, 0.2, 0.9],   # z >= 0.5, unchanged  
			[0.3, 0.3, 1.1],   # z < 0.5, gets +1.0
			[0.4, 0.4, 1.1]    # z < 0.5, gets +1.0
		])
		np.testing.assert_array_almost_equal(site.vertex_coords, expected_coords, decimal=7)

	def test_dynamic_voronoi_site_working_pbc_case(self):
		"""Test a case where current PBC correction works correctly."""
		species = ["Re"] * 4
		coords = [
			[0.1, 0.1, 0.1],
			[0.2, 0.2, 0.9],   # Large z-coordinate, should trigger correction
			[0.3, 0.3, 0.1],
			[0.4, 0.4, 0.1]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure)
		
		# DynamicVoronoiSite applies correction AND wraps centre back to [0,1)
		corrected_coords = np.array([
			[0.1, 0.1, 1.1],
			[0.2, 0.2, 0.9],   
			[0.3, 0.3, 1.1],
			[0.4, 0.4, 1.1]
		])
		centre_before_wrap = np.mean(corrected_coords, axis=0)
		expected_centre = centre_before_wrap % 1.0  # DynamicVoronoiSite wraps back
		np.testing.assert_array_almost_equal(site.centre, expected_centre, decimal=7)

	def test_polyhedral_site_negative_coordinates_no_correction(self):
		"""Test negative coordinates that don't require PBC correction."""
		species = ["Re"] * 4
		coords = [
			[0.1, -0.1, 0.2],   # Negative y-coordinate
			[0.2, -0.05, 0.3],  # y-spread: 0.1 - (-0.1) = 0.2 < 0.5
			[0.3, 0.0, 0.4],
			[0.4, 0.1, 0.5]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = PolyhedralSite(vertex_indices=[0, 1, 2, 3])
		site.assign_vertex_coords(structure)
		
		# Should remain unchanged (spread < 0.5, no correction needed)
		expected_coords = np.array(coords)
		np.testing.assert_array_almost_equal(site.vertex_coords, expected_coords, decimal=7)

	def test_dynamic_voronoi_site_negative_coordinates_no_correction(self):
		"""Test negative coordinates that don't require PBC correction."""
		species = ["Re"] * 4
		coords = [
			[0.1, -0.1, 0.2],   # Negative y-coordinate
			[0.2, -0.05, 0.3],  # y-spread: 0.1 - (-0.1) = 0.2 < 0.5
			[0.3, 0.0, 0.4],
			[0.4, 0.1, 0.5]
		]
		structure = Structure(self.lattice, species, coords)
		
		site = DynamicVoronoiSite(reference_indices=[0, 1, 2, 3])
		site.calculate_centre(structure)
		
		# Should calculate centre correctly (no correction needed)
		expected_centre = np.mean(coords, axis=0) % 1.0
		np.testing.assert_array_almost_equal(site.centre, expected_centre, decimal=7)
		
class TestReferenceBasedUnwrapping(unittest.TestCase):
	"""Tests for reference centre-based vertex unwrapping."""
	
	def setUp(self):
		self.lattice = Lattice.cubic(10.0)
	
	def test_unwrap_vertices_no_unwrapping_needed(self):
		"""Test case where vertices form a tetrahedron around the reference centre."""
		# Simple tetrahedron centered around [0.5, 0.5, 0.5]
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6]
		])
		reference_centre = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_centre(vertex_coords, reference_centre, self.lattice)
		
		# Should remain unchanged since all vertices are already close
		np.testing.assert_array_almost_equal(result, vertex_coords, decimal=7)
		
	def test_unwrap_vertices_simple_case(self):
		"""Test unwrapping for an octahedron spanning the boundary at x=0."""
		# Octahedron centered at [0.05, 0.5, 0.5] spanning the x=0 boundary
		vertex_coords = np.array([
			[0.1, 0.5, 0.5],   # +x vertex
			[-0.05, 0.5, 0.5], # -x vertex
			[0.05, 0.6, 0.5],  # +y vertex  
			[0.05, 0.4, 0.5],  # -y vertex
			[0.05, 0.5, 0.6],  # +z vertex
			[0.05, 0.5, 0.4]   # -z vertex
		])
		reference_centre = np.array([0.05, 0.5, 0.5])
		
		result = unwrap_vertices_to_reference_centre(vertex_coords, reference_centre, self.lattice)
		
		# After unwrapping and shifting to ensure non-negative coordinates
		expected = np.array([
			[1.1, 0.5, 0.5],   # 0.1 + 1.0 (shifted)
			[0.95, 0.5, 0.5],  # -0.05 + 1.0 (unwrapped and shifted)
			[1.05, 0.6, 0.5],  # 0.05 + 1.0 (shifted)
			[1.05, 0.4, 0.5],  # 0.05 + 1.0 (shifted)
			[1.05, 0.5, 0.6],  # 0.05 + 1.0 (shifted)
			[1.05, 0.5, 0.4]   # 0.05 + 1.0 (shifted)
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_unwrap_vertices_pathological_case(self):
		"""Test octahedron that would break the legacy spread-based algorithm."""
		vertex_coords = np.array([
			[0.55, 0.25, 0.25],  
			[0.95, 0.25, 0.25],  
			[0.25, 0.95, 0.25],  
			[0.25, 0.55, 0.25],  
			[0.25, 0.25, 0.55],   
			[0.25, 0.25, 0.95]    
		])
		reference_centre = np.array([0.25, 0.25, 0.25])
		expected_result_centre = np.array([1.25, 1.25, 1.25])
		
		result = unwrap_vertices_to_reference_centre(vertex_coords, reference_centre, self.lattice)
		
		# The centre of the unwrapped vertices should be close to the reference centre
		result_centre = np.mean(result, axis=0)
		np.testing.assert_array_almost_equal(result_centre, expected_result_centre, decimal=2)
		
class TestVectorisedUnwrapping(unittest.TestCase):
	"""Tests for vectorised reference centre-based vertex unwrapping."""
	
	def setUp(self):
		self.lattice = Lattice.cubic(10.0)
	
	def test_unwrap_vertices_vectorised_no_unwrapping_needed(self):
		"""Test vectorised case where vertices form a tetrahedron around the reference centre."""
		# Simple tetrahedron centered around [0.5, 0.5, 0.5]
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6]
		])
		reference_centre = np.array([0.5, 0.5, 0.5])
		
		result = unwrap_vertices_vectorised(vertex_coords, reference_centre, self.lattice)
		
		# Should remain unchanged since all vertices are already close
		np.testing.assert_array_almost_equal(result, vertex_coords, decimal=7)
		
	def test_unwrap_vertices_vectorised_simple_case(self):
		"""Test vectorised unwrapping for an octahedron spanning the boundary at x=0."""
		# Octahedron centered at [0.05, 0.5, 0.5] spanning the x=0 boundary
		vertex_coords = np.array([
			[0.1, 0.5, 0.5],   # +x vertex
			[-0.05, 0.5, 0.5], # -x vertex
			[0.05, 0.6, 0.5],  # +y vertex  
			[0.05, 0.4, 0.5],  # -y vertex
			[0.05, 0.5, 0.6],  # +z vertex
			[0.05, 0.5, 0.4]   # -z vertex
		])
		reference_centre = np.array([0.05, 0.5, 0.5])
		
		result = unwrap_vertices_vectorised(vertex_coords, reference_centre, self.lattice)
		
		# After unwrapping and shifting to ensure non-negative coordinates
		expected = np.array([
			[1.1, 0.5, 0.5],   # 0.1 + 1.0 (shifted)
			[0.95, 0.5, 0.5],  # -0.05 + 1.0 (unwrapped and shifted)
			[1.05, 0.6, 0.5],  # 0.05 + 1.0 (shifted)
			[1.05, 0.4, 0.5],  # 0.05 + 1.0 (shifted)
			[1.05, 0.5, 0.6],  # 0.05 + 1.0 (shifted)
			[1.05, 0.5, 0.4]   # 0.05 + 1.0 (shifted)
		])
		np.testing.assert_array_almost_equal(result, expected, decimal=7)
	
	def test_unwrap_vertices_vectorised_pathological_case(self):
		"""Test vectorised octahedron that would break the legacy spread-based algorithm."""
		vertex_coords = np.array([
			[0.55, 0.25, 0.25],  
			[0.95, 0.25, 0.25],  
			[0.25, 0.95, 0.25],  
			[0.25, 0.55, 0.25],  
			[0.25, 0.25, 0.55],   
			[0.25, 0.25, 0.95]    
		])
		reference_centre = np.array([0.25, 0.25, 0.25])
		expected_result_centre = np.array([1.25, 1.25, 1.25])
		
		result = unwrap_vertices_vectorised(vertex_coords, reference_centre, self.lattice)
		
		# The centre of the unwrapped vertices should be close to the reference centre
		result_centre = np.mean(result, axis=0)
		np.testing.assert_array_almost_equal(result_centre, expected_result_centre, decimal=2)


class TestUnwrappingConsistency(unittest.TestCase):
	"""Tests that both unwrapping methods produce identical results."""
	
	def setUp(self):
		self.lattice = Lattice.cubic(10.0)
	
	def test_both_methods_produce_same_results_simple_case(self):
		"""Test that loop-based and vectorised methods give identical results."""
		vertex_coords = np.array([
			[0.1, 0.5, 0.5],   
			[-0.05, 0.5, 0.5], 
			[0.05, 0.6, 0.5],  
			[0.05, 0.4, 0.5],  
			[0.05, 0.5, 0.6],  
			[0.05, 0.5, 0.4]   
		])
		reference_centre = np.array([0.05, 0.5, 0.5])
		
		result_loop = unwrap_vertices_to_reference_centre(vertex_coords, reference_centre, self.lattice)
		result_vectorised = unwrap_vertices_vectorised(vertex_coords, reference_centre, self.lattice)
		
		np.testing.assert_array_almost_equal(result_loop, result_vectorised, decimal=7)
	
	def test_both_methods_produce_same_results_pathological_case(self):
		"""Test consistency for the pathological case."""
		vertex_coords = np.array([
			[0.55, 0.25, 0.25],  
			[0.95, 0.25, 0.25],  
			[0.25, 0.95, 0.25],  
			[0.25, 0.55, 0.25],  
			[0.25, 0.25, 0.55],   
			[0.25, 0.25, 0.95]    
		])
		reference_centre = np.array([0.25, 0.25, 0.25])
		
		result_loop = unwrap_vertices_to_reference_centre(vertex_coords, reference_centre, self.lattice)
		result_vectorised = unwrap_vertices_vectorised(vertex_coords, reference_centre, self.lattice)
		
		np.testing.assert_array_almost_equal(result_loop, result_vectorised, decimal=7)
	
	def test_both_methods_tetrahedral_coordination(self):
		"""Test consistency for 4-coordinate tetrahedral case."""
		vertex_coords = np.array([
			[0.4, 0.4, 0.4],
			[0.6, 0.6, 0.4],
			[0.6, 0.4, 0.6],
			[0.4, 0.6, 0.6]
		])
		reference_centre = np.array([0.5, 0.5, 0.5])
		
		result_loop = unwrap_vertices_to_reference_centre(vertex_coords, reference_centre, self.lattice)
		result_vectorised = unwrap_vertices_vectorised(vertex_coords, reference_centre, self.lattice)
		
		np.testing.assert_array_almost_equal(result_loop, result_vectorised, decimal=7)
		
import time

class TestPerformanceComparison(unittest.TestCase):
	"""Performance comparison between legacy, loop-based, and vectorised unwrapping methods."""
	
	def setUp(self):
		self.lattice = Lattice.cubic(10.0)
		self.reference_centre = np.array([0.25, 0.25, 0.25])
	
	def _generate_boundary_spanning_coords(self, n_vertices, reference_centre):
		"""Generate coordinates that span periodic boundaries around a reference centre."""
		# Create vertices that will require unwrapping
		np.random.seed(42)  # For reproducible results
		
		# Generate vertices in a pattern that spans boundaries
		coords = []
		for i in range(n_vertices):
			# Create coordinates that are sometimes on opposite sides of boundaries
			coord = reference_centre + np.random.uniform(-0.4, 0.4, 3)
			# Force some coordinates to wrap around boundaries
			if i % 3 == 0:
				coord[0] += 0.6  # This will put some vertices > 1.0
			if i % 3 == 1:
				coord[1] -= 0.6  # This will put some vertices < 0.0
			coords.append(coord)
		
		return np.array(coords)
	
	def test_performance_comparison_all_methods(self):
		"""Compare performance of legacy, loop-based, and vectorised unwrapping methods."""
		coord_numbers = [4, 6, 8]
		n_trials = 1000
		
		print(f"\nPerformance comparison of all methods ({n_trials} trials each):")
		print("-" * 70)
		
		for cn in coord_numbers:
			# Generate test coordinates
			vertex_coords = self._generate_boundary_spanning_coords(cn, self.reference_centre)
			
			# Time the legacy method
			start_time = time.perf_counter()
			for _ in range(n_trials):
				result_legacy = apply_legacy_pbc_correction(vertex_coords)
			legacy_time = time.perf_counter() - start_time
			
			# Time the loop-based method
			start_time = time.perf_counter()
			for _ in range(n_trials):
				result_loop = unwrap_vertices_to_reference_centre(
					vertex_coords, self.reference_centre, self.lattice
				)
			loop_time = time.perf_counter() - start_time
			
			# Time the vectorised method
			start_time = time.perf_counter()
			for _ in range(n_trials):
				result_vectorised = unwrap_vertices_vectorised(
					vertex_coords, self.reference_centre, self.lattice
				)
			vectorised_time = time.perf_counter() - start_time
			
			# Verify reference-based methods give the same results
			np.testing.assert_array_almost_equal(result_loop, result_vectorised, decimal=7)
			
			# Calculate speedups relative to loop method
			legacy_speedup = loop_time / legacy_time if legacy_time > 0 else float('inf')
			vectorised_speedup = loop_time / vectorised_time if vectorised_time > 0 else float('inf')
			
			# Report results
			print(f"Coordination number {cn}:")
			print(f"  Legacy:       {legacy_time*1000/n_trials:.4f} ms per call (speedup: {legacy_speedup:.2f}x)")
			print(f"  Loop-based:   {loop_time*1000/n_trials:.4f} ms per call (baseline)")
			print(f"  Vectorised:   {vectorised_time*1000/n_trials:.4f} ms per call (speedup: {vectorised_speedup:.2f}x)")
			print()
	
	def test_performance_scaling_with_coordination_number(self):
		"""Test how performance scales with coordination number."""
		coord_numbers = [4, 6, 8, 12, 16, 20, 24]
		n_trials = 200
		
		print(f"\nPerformance scaling with coordination number ({n_trials} trials each):")
		print("-" * 80)
		print(f"{'CN':<4} {'Legacy (ms)':<12} {'Loop (ms)':<12} {'Vector (ms)':<12} {'Vec/Loop':<10} {'Vec/Legacy':<10}")
		print("-" * 80)
		
		for cn in coord_numbers:
			vertex_coords = self._generate_boundary_spanning_coords(cn, self.reference_centre)
			
			# Time legacy method
			start_time = time.perf_counter()
			for _ in range(n_trials):
				apply_legacy_pbc_correction(vertex_coords)
			legacy_time = time.perf_counter() - start_time
			
			# Time loop-based method
			start_time = time.perf_counter()
			for _ in range(n_trials):
				unwrap_vertices_to_reference_centre(
					vertex_coords, self.reference_centre, self.lattice
				)
			loop_time = time.perf_counter() - start_time
			
			# Time vectorised method
			start_time = time.perf_counter()
			for _ in range(n_trials):
				unwrap_vertices_vectorised(
					vertex_coords, self.reference_centre, self.lattice
				)
			vectorised_time = time.perf_counter() - start_time
			
			# Calculate metrics
			legacy_ms = legacy_time * 1000 / n_trials
			loop_ms = loop_time * 1000 / n_trials
			vectorised_ms = vectorised_time * 1000 / n_trials
			
			vec_vs_loop = loop_time / vectorised_time if vectorised_time > 0 else float('inf')
			vec_vs_legacy = legacy_time / vectorised_time if vectorised_time > 0 else float('inf')
			
			print(f"{cn:<4} {legacy_ms:<12.4f} {loop_ms:<12.4f} {vectorised_ms:<12.4f} {vec_vs_loop:<10.2f} {vec_vs_legacy:<10.2f}")


if __name__ == '__main__':
	unittest.main()