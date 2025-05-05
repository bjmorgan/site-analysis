"""Integration tests for the ReferenceBasedSites class.

These tests demonstrate the use of ReferenceBasedSites with realistic crystal structures
to generate both PolyhedralSites and DynamicVoronoiSites.
"""

import unittest
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from site_analysis.reference_workflow.reference_based_sites import ReferenceBasedSites
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from ..test_helpers import apply_random_displacement


class TestCsPbI3Integration(unittest.TestCase):
    """Integration tests using CsPbI3 perovskite structure."""
    
    def setUp(self):
        """Create reference and target CsPbI3 structures."""
        # Create an ideal cubic perovskite CsPbI3 reference structure
        a = 6.0  # Approximate lattice parameter in Angstroms
        lattice = Lattice.cubic(a)
        
        # Positions for cubic perovskite
        species = ["Cs", "Pb", "I", "I", "I"]
        coords = [
            [0.0, 0.0, 0.0],    # Cs at corner
            [0.5, 0.5, 0.5],    # Pb at center
            [0.5, 0.5, 0.0],    # I at face center
            [0.5, 0.0, 0.5],    # I at face center
            [0.0, 0.5, 0.5]     # I at face center
        ]
        unit_cell = Structure(lattice, species, coords)
        
        # Create a 2x2x2 supercell for more realistic testing
        self.reference = unit_cell * [2, 2, 2]
        
        # Create a slightly distorted target structure
        self.target = apply_random_displacement(self.reference, magnitude=0.1, seed=42)
    
    def test_dynamic_voronoi_sites_for_iodine(self):
        """Test creating dynamic voronoi sites for I atoms in CsPbI3.
        
        I atoms sit between two Pb atoms, so we use Pb as reference atoms.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create dynamic voronoi sites for I atoms
        # I atoms are coordinated by 2 Pb atoms
        sites = rbs.create_dynamic_voronoi_sites(
            center_species="I",        # I is at the center of the site
            reference_species="Pb",    # Pb atoms define the site
            cutoff=4.5,                # Typical Pb-I distance is ~3.2 Å
            n_reference=2,             # Each I is coordinated by 2 Pb
            label="I_site"             # Label for the sites
        )
        
        # Verify that we got the correct number of sites (24 I atoms in 2x2x2 supercell)
        self.assertEqual(len(sites), 24)
        
        # Verify that these are DynamicVoronoiSite objects
        for site in sites:
            self.assertIsInstance(site, DynamicVoronoiSite)
            self.assertEqual(site.label, "I_site")
            
            # Each site should have 2 reference atoms
            self.assertEqual(len(site.reference_indices), 2)
    
    def test_polyhedral_sites_for_cesium(self):
        """Test creating polyhedral sites for Cs atoms in CsPbI3.
        
        Cs atoms sit in a cavity formed by 8 Pb atoms in the cubic phase.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create polyhedral sites for Cs atoms
        # In cubic perovskite, Cs is coordinated by 8 I atoms
        sites = rbs.create_polyhedral_sites(
            center_species="Cs",       # Cs is at the center of the site
            vertex_species="Pb",       # Pb atoms define the vertices
            cutoff=6.0,                # Typical Cs-Pb distance is ~6 Å
            n_vertices=8,              # Each Cs is coordinated by 8 Pb atoms
            label="Cs_site"            # Label for the sites
        )
        
        # Verify that we got the correct number of sites (8 Cs atoms in 2x2x2 supercell)
        self.assertEqual(len(sites), 8)
        
        # Verify that these are PolyhedralSite objects
        for site in sites:
            self.assertIsInstance(site, PolyhedralSite)
            self.assertEqual(site.label, "Cs_site")
            
            # Each site should have 8 vertex atoms
            self.assertEqual(len(site.vertex_indices), 8)


class TestNaClIntegration(unittest.TestCase):
    """Integration tests using NaCl (rock salt) structure."""
    
    def setUp(self):
        """Create reference and target NaCl structures."""
        # Create an ideal rock salt NaCl reference structure
        a = 5.64  # Approximate lattice parameter in Angstroms
        lattice = Lattice.cubic(a)
        
        # Create the structure with the correct space group symmetry
        structure = Structure.from_spacegroup(
            sg="Fm-3m",
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0, 0]]
        )
        
        # Create a 3x3x3 supercell for more realistic testing
        self.reference = structure * [3, 3, 3]
        
        # Create a distorted target structure
        self.target = apply_random_displacement(self.reference, magnitude=0.1, seed=42)
    
    def test_polyhedral_sites_for_sodium(self):
        """Test creating polyhedral sites for Na atoms in NaCl.
        
        Na atoms are octahedrally coordinated by 6 Cl atoms.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create polyhedral sites for Na atoms
        sites = rbs.create_polyhedral_sites(
            center_species="Na",       # Na is at the center of the site
            vertex_species="Cl",       # Cl atoms define the vertices
            cutoff=3.5,                # Na-Cl distance is ~2.8 Å
            n_vertices=6,              # Each Na is coordinated by 6 Cl
            label="Na_site"            # Label for the sites
        )
        
        # Count the number of Na atoms in the structure
        na_count = len([site for site in self.reference if site.species_string == "Na"])
        
        # Verify we got one site per Na atom
        self.assertEqual(len(sites), na_count)
        
        # Verify that these are PolyhedralSite objects
        for site in sites:
            self.assertIsInstance(site, PolyhedralSite)
            self.assertEqual(site.label, "Na_site")
            
            # Each site should have 6 vertex atoms (octahedral)
            self.assertEqual(len(site.vertex_indices), 6)
    
    def test_dynamic_voronoi_sites_for_chlorine(self):
        """Test creating dynamic voronoi sites for Cl atoms in NaCl.
        
        Cl atoms are octahedrally coordinated by 6 Na atoms.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create dynamic voronoi sites for Cl atoms
        sites = rbs.create_dynamic_voronoi_sites(
            center_species="Cl",       # Cl is at the center of the site
            reference_species="Na",    # Na atoms define the site
            cutoff=3.5,                # Cl-Na distance is ~2.8 Å
            n_reference=6,             # Each Cl is coordinated by 6 Na
            label="Cl_site"            # Label for the sites
        )
        
        # Count the number of Cl atoms in the structure
        cl_count = len([site for site in self.reference if site.species_string == "Cl"])
        
        # Verify we got one site per Cl atom
        self.assertEqual(len(sites), cl_count)
        
        # Verify that these are DynamicVoronoiSite objects
        for site in sites:
            self.assertIsInstance(site, DynamicVoronoiSite)
            self.assertEqual(site.label, "Cl_site")
            
            # Each site should have 6 reference atoms (octahedral)
            self.assertEqual(len(site.reference_indices), 6)


class TestZnOIntegration(unittest.TestCase):
    """Integration tests using ZnO (wurtzite) structure."""
    
    def setUp(self):
        """Create reference and target ZnO structures."""
        # Create a wurtzite ZnO reference structure
        a = 3.25  # Approximate a parameter in Angstroms
        c = 5.20  # Approximate c parameter in Angstroms
        lattice = Lattice.hexagonal(a, c)
        
        # Wurtzite structure positions
        species = ["Zn", "Zn", "O", "O"]
        coords = [
            [1/3, 2/3, 0],         # Zn
            [2/3, 1/3, 0.5],       # Zn
            [1/3, 2/3, 3/8],       # O
            [2/3, 1/3, 7/8]        # O
        ]
        unit_cell = Structure(lattice, species, coords)
        
        # Create a 3x3x2 supercell for more realistic testing
        self.reference = unit_cell * [3, 3, 2]
        
        # Create a distorted target structure
        self.target = apply_random_displacement(self.reference, magnitude=0.1, seed=42)
    
    def test_polyhedral_sites_for_zinc(self):
        """Test creating polyhedral sites for Zn atoms in ZnO.
        
        Zn atoms are tetrahedrally coordinated by 4 O atoms.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create polyhedral sites for Zn atoms
        sites = rbs.create_polyhedral_sites(
            center_species="Zn",       # Zn is at the center of the site
            vertex_species="O",        # O atoms define the vertices
            cutoff=2.5,                # Zn-O distance is ~1.9 Å
            n_vertices=4,              # Each Zn is coordinated by 4 O
            label="Zn_tetrahedral"     # Label for the sites
        )
        
        # Count the number of Zn atoms in the structure
        zn_count = len([site for site in self.reference if site.species_string == "Zn"])
        
        # Verify we got one site per Zn atom
        self.assertEqual(len(sites), zn_count)
        
        # Verify that these are PolyhedralSite objects with tetrahedral coordination
        for site in sites:
            self.assertIsInstance(site, PolyhedralSite)
            self.assertEqual(site.label, "Zn_tetrahedral")
            
            # Each site should have 4 vertex atoms (tetrahedral)
            self.assertEqual(len(site.vertex_indices), 4)
    
    def test_dynamic_voronoi_sites_for_oxygen(self):
        """Test creating dynamic voronoi sites for O atoms in ZnO.
        
        O atoms are tetrahedrally coordinated by 4 Zn atoms.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create dynamic voronoi sites for O atoms
        sites = rbs.create_dynamic_voronoi_sites(
            center_species="O",        # O is at the center of the site
            reference_species="Zn",    # Zn atoms define the site
            cutoff=2.5,                # O-Zn distance is ~1.9 Å
            n_reference=4,             # Each O is coordinated by 4 Zn
            label="O_tetrahedral"      # Label for the sites
        )
        
        # Count the number of O atoms in the structure
        o_count = len([site for site in self.reference if site.species_string == "O"])
        
        # Verify we got one site per O atom
        self.assertEqual(len(sites), o_count)
        
        # Verify that these are DynamicVoronoiSite objects with tetrahedral coordination
        for site in sites:
            self.assertIsInstance(site, DynamicVoronoiSite)
            self.assertEqual(site.label, "O_tetrahedral")
            
            # Each site should have 4 reference atoms (tetrahedral)
            self.assertEqual(len(site.reference_indices), 4)


class TestTiO2Integration(unittest.TestCase):
    """Integration tests using TiO2 (rutile) structure."""
    
    def setUp(self):
        """Create reference and target TiO2 structures."""
        # Create a rutile TiO2 reference structure
        a = 4.59  # Approximate a parameter in Angstroms
        c = 2.96  # Approximate c parameter in Angstroms
        lattice = Lattice.tetragonal(a, c)
        
        # Rutile structure positions
        species = ["Ti", "Ti", "O", "O", "O", "O"]
        coords = [
            [0.0, 0.0, 0.0],       # Ti
            [0.5, 0.5, 0.5],       # Ti
            [0.3, 0.3, 0.0],       # O
            [0.7, 0.7, 0.0],       # O
            [0.8, 0.2, 0.5],       # O
            [0.2, 0.8, 0.5]        # O
        ]
        unit_cell = Structure(lattice, species, coords)
        
        # Create a 2x2x3 supercell for more realistic testing
        self.reference = unit_cell * [2, 2, 3]
        
        # Create a distorted target structure
        self.target = apply_random_displacement(self.reference, magnitude=0.1, seed=42)
    
    def test_polyhedral_sites_for_titanium(self):
        """Test creating polyhedral sites for Ti atoms in TiO2.
        
        Ti atoms are octahedrally coordinated by 6 O atoms.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create polyhedral sites for Ti atoms
        sites = rbs.create_polyhedral_sites(
            center_species="Ti",       # Ti is at the center of the site
            vertex_species="O",        # O atoms define the vertices
            cutoff=2.5,                # Ti-O distance is ~2.0 Å
            n_vertices=6,              # Each Ti is coordinated by 6 O
            label="Ti_octahedral"      # Label for the sites
        )
        
        # Count the number of Ti atoms in the structure
        ti_count = len([site for site in self.reference if site.species_string == "Ti"])
        
        # Verify we got one site per Ti atom
        self.assertEqual(len(sites), ti_count)
        
        # Verify that these are PolyhedralSite objects with octahedral coordination
        for site in sites:
            self.assertIsInstance(site, PolyhedralSite)
            self.assertEqual(site.label, "Ti_octahedral")
            
            # Each site should have 6 vertex atoms (octahedral)
            self.assertEqual(len(site.vertex_indices), 6)
    
    def test_dynamic_voronoi_sites_for_oxygen(self):
        """Test creating dynamic voronoi sites for O atoms in TiO2.
        
        O atoms are coordinated by 3 Ti atoms in rutile.
        """
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(self.reference, self.target, align=True)
        
        # Create dynamic voronoi sites for O atoms
        sites = rbs.create_dynamic_voronoi_sites(
            center_species="O",        # O is at the center of the site
            reference_species="Ti",    # Ti atoms define the site
            cutoff=2.5,                # O-Ti distance is ~2.0 Å
            n_reference=3,             # Each O is coordinated by 3 Ti
            label="O_trigonal"         # Label for the sites
        )
        
        # Count the number of O atoms in the structure
        o_count = len([site for site in self.reference if site.species_string == "O"])
        
        # Verify we got one site per O atom
        self.assertEqual(len(sites), o_count)
        
        # Verify that these are DynamicVoronoiSite objects
        for site in sites:
            self.assertIsInstance(site, DynamicVoronoiSite)
            self.assertEqual(site.label, "O_trigonal")
            
            # Each site should have 3 reference atoms (trigonal coordination)
            self.assertEqual(len(site.reference_indices), 3)


class TestMixedStructureIntegration(unittest.TestCase):
    """Integration tests using mixed structures where reference and target differ."""

    
    def test_with_different_atom_ordering(self):
        """Test ReferenceBasedSites with different atom ordering.
        
        This tests the case where atoms are ordered differently in reference and target.
        """
        # Create a simple structure
        lattice = Lattice.cubic(5.0)
        species1 = ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"]
        coords1 = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5]
        ]
        reference = Structure(lattice, species1, coords1)
        
        # Create a target with same atoms but completely different ordering
        # Shuffle both species and coordinates
        np.random.seed(42)
        indices = np.random.permutation(len(reference))
        species2 = [reference[i].species_string for i in indices]
        coords2 = [reference[i].frac_coords for i in indices]
        
        target_reordered = Structure(lattice, species2, coords2)
        
        # Add distortion to the target
        target = apply_random_displacement(target_reordered, magnitude=0.1, seed=42)
        
        # Initialize ReferenceBasedSites
        rbs = ReferenceBasedSites(reference, target, align=True)
        
        # Create dynamic voronoi sites for Na atoms
        sites = rbs.create_dynamic_voronoi_sites(
            center_species="Na",
            reference_species="Cl",
            cutoff=3.5,
            n_reference=2,
            label="Na_site"
        )
        
        # We should get sites for Na atoms
        na_count = len([site for site in reference if site.species_string == "Na"])
        self.assertEqual(len(sites), na_count)
        
        # Sites should be DynamicVoronoiSite objects
        for site in sites:
            self.assertIsInstance(site, DynamicVoronoiSite)
            self.assertEqual(len(site.reference_indices), 2)


if __name__ == "__main__":
    unittest.main()
