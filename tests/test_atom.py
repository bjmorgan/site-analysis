import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock, Mock
from pymatgen.core import Structure, Lattice
import numpy as np

from site_analysis.atom import (
    Atom, 
    atoms_from_species_string, 
    atoms_from_structure, 
    atoms_from_indices
)

class AtomTestCase(unittest.TestCase):

    def test_atom_is_initialised(self):
        atom = Atom(index=22)
        self.assertEqual(atom.index, 22)
        self.assertEqual(atom.in_site, None)
        self.assertEqual(atom._frac_coords, None)
        self.assertEqual(atom.trajectory, [])
        
    def test_init_with_species_string(self):
        """Test that Atom correctly stores species_string when provided."""
        species = "Na"
        atom = Atom(index=1, species_string=species)
        self.assertEqual(atom.species_string, species)

    def test_reset(self):
        atom = Atom(index=12)
        atom.in_site = 3
        atom._frac_coords = np.array( [0,0, 0.0, 0.0] )
        atom.trajectory = [1,2,3]
        atom.reset()
        self.assertEqual( atom.in_site, None )
        self.assertEqual( atom._frac_coords, None )
        self.assertEqual( atom.trajectory, [] )

    def test___str__(self):
        atom = Atom(index=12)
        self.assertEqual(str(atom), 'Atom: 12')

    def test___repr__(self):
        atom = Atom(index=12)
        self.assertEqual(atom.__repr__(), 'site_analysis.Atom(index=12, in_site=None, frac_coords=None, species_string=None)')

    def test_assign_coords(self):
        atom = Atom(index=1)
        structure = example_structure()
        atom.assign_coords(structure=structure)
        np.testing.assert_array_equal(atom._frac_coords, structure[1].frac_coords )
   
    def test_frac_coords_getter_raises_atttribute_error_if_frac_coords_is_none(self):
        atom = Atom(index=1)
        atom._frac_coords = None
        with self.assertRaises(AttributeError):
            atom.frac_coords

    def test_frac_coords_getter(self):
        atom = Atom(index=12)
        c = np.array([0.3, 0.4, 0.5])
        atom._frac_coords = c
        np.testing.assert_array_equal(atom.frac_coords, c)

    def test_as_dict(self):
        index = 11
        in_site = 4
        c = np.array([0.1, 0.2, 0.3])
        atom = Atom(index=index)
        atom.in_site = in_site
        atom._frac_coords = c
        d = atom.as_dict()
        self.assertEqual(d['index'], index)
        self.assertEqual(d['in_site'], in_site)
        np.testing.assert_array_equal(d['frac_coords'], c)
 
def example_structure(species=None):
    if not species:
        species = ['S']*5
    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    cartesian_coords = np.array([[1.0, 1.0, 1.0],
                                 [9.0, 1.0, 1.0],
                                 [5.0, 5.0, 5.0],
                                 [1.0, 9.0, 9.0],
                                 [9.0, 9.0, 9.0]])
    structure = Structure(coords=cartesian_coords,
                          lattice=lattice,
                          species=species,
                          coords_are_cartesian=True)
    return structure 
    
class AtomUtilityFunctionsTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test structures and data for utility function tests."""
        # Create a simple cubic lattice
        self.lattice = Lattice.cubic(5.0)
        
        # Create a mixed-species structure
        self.mixed_structure = Structure(
            lattice=self.lattice,
            species=["Na", "Cl", "Na", "K", "O", "Cl"],
            coords=[
                [0.1, 0.1, 0.1],  # Na (0)
                [0.2, 0.2, 0.2],  # Cl (1)
                [0.3, 0.3, 0.3],  # Na (2)
                [0.4, 0.4, 0.4],  # K  (3)
                [0.5, 0.5, 0.5],  # O  (4)
                [0.6, 0.6, 0.6],  # Cl (5)
            ]
        )
        
        # Create an empty structure
        self.empty_structure = Structure(
            lattice=self.lattice,
            species=[],
            coords=[]
        )
        
        # Sample indices for testing
        self.test_indices = [0, 3, 5]
        
    def test_atoms_from_species_string_basic(self):
        """Test atoms_from_species_string with a species that exists."""
        # Test with Na atoms
        atoms = atoms_from_species_string(self.mixed_structure, "Na")
        
        # Should find 2 Na atoms with indices 0 and 2
        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms[0].index, 0)
        self.assertEqual(atoms[1].index, 2)
        
        # Verify these are Atom objects without species_string set
        for atom in atoms:
            self.assertIsInstance(atom, Atom)
            self.assertIsNone(atom.species_string)
        
        # Test with Cl atoms
        atoms = atoms_from_species_string(self.mixed_structure, "Cl")
        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms[0].index, 1)
        self.assertEqual(atoms[1].index, 5)
    
    def test_atoms_from_species_string_nonexistent(self):
        """Test atoms_from_species_string with a species that doesn't exist."""
        atoms = atoms_from_species_string(self.mixed_structure, "Ca")
        
        # Should return an empty list
        self.assertEqual(len(atoms), 0)
        self.assertEqual(atoms, [])
    
    def test_atoms_from_species_string_empty_structure(self):
        """Test atoms_from_species_string with an empty structure."""
        atoms = atoms_from_species_string(self.empty_structure, "Na")
        
        # Should return an empty list
        self.assertEqual(len(atoms), 0)
        self.assertEqual(atoms, [])
        
    def test_atoms_from_structure_string_species(self):
        """Test atoms_from_structure with a string species."""
        atoms = atoms_from_structure(self.mixed_structure, "Na")
        
        # Should find 2 Na atoms with indices 0 and 2
        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms[0].index, 0)
        self.assertEqual(atoms[1].index, 2)
        
        # Verify species_string and frac_coords are set
        for atom in atoms:
            self.assertEqual(atom.species_string, "Na")
            self.assertIsNotNone(atom._frac_coords)
    
    def test_atoms_from_structure_list_species(self):
        """Test atoms_from_structure with a list of species."""
        atoms = atoms_from_structure(self.mixed_structure, ["Na", "K"])
        
        # Should find 3 atoms: 2 Na (indices 0, 2) and 1 K (index 3)
        self.assertEqual(len(atoms), 3)
        
        # Get the indices and verify they match expected
        indices = [atom.index for atom in atoms]
        self.assertIn(0, indices)  # Na
        self.assertIn(2, indices)  # Na
        self.assertIn(3, indices)  # K
        
        # Verify each atom has the correct species_string
        for atom in atoms:
            if atom.index in [0, 2]:
                self.assertEqual(atom.species_string, "Na")
            elif atom.index == 3:
                self.assertEqual(atom.species_string, "K")
    
    def test_atoms_from_structure_nonexistent_species(self):
        """Test atoms_from_structure with species that don't exist."""
        atoms = atoms_from_structure(self.mixed_structure, ["Ca", "Mg"])
        
        # Should return an empty list
        self.assertEqual(len(atoms), 0)
        self.assertEqual(atoms, [])
    
    def test_atoms_from_structure_empty_structure(self):
        """Test atoms_from_structure with an empty structure."""
        atoms = atoms_from_structure(self.empty_structure, "Na")
        
        # Should return an empty list
        self.assertEqual(len(atoms), 0)
        self.assertEqual(atoms, [])
    
    def test_atoms_from_structure_frac_coords(self):
        """Test that atoms_from_structure correctly sets fractional coordinates."""
        atoms = atoms_from_structure(self.mixed_structure, "Na")
        
        # Verify both atoms have correctly assigned fractional coordinates
        na_coords = [self.mixed_structure[0].frac_coords, self.mixed_structure[2].frac_coords]
        
        for i, atom in enumerate(atoms):
            np.testing.assert_array_equal(atom._frac_coords, na_coords[i])
    
    def test_atoms_from_indices_basic(self):
        """Test atoms_from_indices with a list of indices."""
        atoms = atoms_from_indices(self.test_indices)
        
        # Should create 3 atoms with the specified indices
        self.assertEqual(len(atoms), 3)
        self.assertEqual(atoms[0].index, 0)
        self.assertEqual(atoms[1].index, 3)
        self.assertEqual(atoms[2].index, 5)
        
        # Verify these are Atom objects
        for atom in atoms:
            self.assertIsInstance(atom, Atom)
    
    def test_atoms_from_indices_empty(self):
        """Test atoms_from_indices with an empty list."""
        atoms = atoms_from_indices([])
        
        # Should return an empty list
        self.assertEqual(len(atoms), 0)
        self.assertEqual(atoms, [])
    
    def test_atoms_from_indices_duplicates(self):
        """Test atoms_from_indices with duplicate indices."""
        # List with duplicate indices
        duplicate_indices = [0, 3, 0, 5, 3]
        atoms = atoms_from_indices(duplicate_indices)
        
        # Should create 5 atoms with the specified indices, including duplicates
        self.assertEqual(len(atoms), 5)
        self.assertEqual(atoms[0].index, 0)
        self.assertEqual(atoms[1].index, 3)
        self.assertEqual(atoms[2].index, 0)  # Duplicate
        self.assertEqual(atoms[3].index, 5)
        self.assertEqual(atoms[4].index, 3)  # Duplicate
        
        # Check that duplicated indices result in distinct Atom objects
        self.assertIsNot(atoms[0], atoms[2])  # Different objects despite same index
        self.assertIsNot(atoms[1], atoms[4])  # Different objects despite same index
    
    def test_atoms_from_indices_negative(self):
        """Test atoms_from_indices with negative indices."""
        negative_indices = [-1, -5, 0]
        atoms = atoms_from_indices(negative_indices)
        
        # Should create atoms with the specified indices, even if negative
        self.assertEqual(len(atoms), 3)
        self.assertEqual(atoms[0].index, -1)
        self.assertEqual(atoms[1].index, -5)
        self.assertEqual(atoms[2].index, 0)   
        
    def test_most_recent_site_empty_trajectory(self):
        """Test most_recent_site with an empty trajectory."""
        atom = Atom(index=1)
        # Empty trajectory should return None
        self.assertIsNone(atom.most_recent_site)
    
    def test_most_recent_site_none_values(self):
        """Test most_recent_site when trajectory only contains None values."""
        atom = Atom(index=1)
        atom.trajectory = [None, None, None]
        
        # Only None values in trajectory should return None
        self.assertIsNone(atom.most_recent_site)
    
    def test_most_recent_site_with_values(self):
        """Test most_recent_site returns the last non-None value."""
        atom = Atom(index=1)
        
        # Simple case with only site indices
        atom.trajectory = [2, 4, 6, 8]
        self.assertEqual(atom.most_recent_site, 8)
        
        # With None at the end
        atom.trajectory = [2, 4, 6, None]
        self.assertEqual(atom.most_recent_site, 6)
        
        # With None values interspersed
        atom.trajectory = [2, None, 4, None, 6, None]
        self.assertEqual(atom.most_recent_site, 6)
        
        # With None values at the beginning
        atom.trajectory = [None, None, 6, 8]
        self.assertEqual(atom.most_recent_site, 8)
        
        # Mixed case with None values
        atom.trajectory = [None, 3, None, None, 7, None]
        self.assertEqual(atom.most_recent_site, 7)

class AtomSerialisationTestCase(unittest.TestCase):
    """Simple unit tests for Atom serialisation methods."""

    def test_from_dict_creates_atom_with_correct_attributes(self):
        """Test from_dict sets all attributes correctly."""
        atom_dict = {
            "index": 5,
            "in_site": 2,
            "frac_coords": [0.1, 0.2, 0.3],
            "species_string": "Li"
        }

        atom = Atom.from_dict(atom_dict)

        self.assertEqual(atom.index, 5)
        self.assertEqual(atom.in_site, 2)
        self.assertEqual(atom.species_string, "Li")
        np.testing.assert_array_equal(atom._frac_coords, [0.1, 0.2, 0.3])

    def test_from_dict_handles_none_values(self):
        """Test from_dict handles None values correctly."""
        atom_dict = {
            "index": 1,
            "in_site": None,
            "frac_coords": [0.0, 0.0, 0.0]
        }

        atom = Atom.from_dict(atom_dict)

        self.assertEqual(atom.index, 1)
        self.assertIsNone(atom.in_site)
        self.assertIsNone(atom.species_string)

    def test_to_returns_json_string(self):
        """Test to() method returns valid JSON string."""
        atom = Atom(index=10, species_string="Na")
        atom._frac_coords = np.array([0.5, 0.5, 0.5])
        atom.in_site = 3

        json_string = atom.to()

        # Should be valid JSON
        parsed = json.loads(json_string)
        self.assertEqual(parsed["index"], 10)
        self.assertEqual(parsed["species_string"], "Na")

    def test_to_writes_file_when_filename_provided(self):
        """Test to() writes to file when filename given."""
        atom = Atom(index=1)
        atom._frac_coords = np.array([0.1, 0.1, 0.1])

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            filename = tmp_file.name

        try:
            atom.to(filename=filename)

            # File should exist and contain JSON
            self.assertTrue(os.path.exists(filename))
            with open(filename, 'r') as f:
                data = json.load(f)
            self.assertEqual(data["index"], 1)
        finally:
            os.unlink(filename)

    def test_from_str_creates_atom_from_json(self):
        """Test from_str creates atom from JSON string."""
        json_string = '{"index": 7, "in_site": null, "frac_coords": [0.2, 0.3, 0.4]}'

        atom = Atom.from_str(json_string)

        self.assertEqual(atom.index, 7)
        self.assertIsNone(atom.in_site)

    def test_from_str_raises_error_for_invalid_json(self):
        """Test from_str raises JSONDecodeError for invalid JSON."""
        invalid_json = "not valid json"

        with self.assertRaises(json.JSONDecodeError):
            Atom.from_str(invalid_json)

    def test_from_file_reads_atom_from_file(self):
        """Test from_file reads atom from JSON file."""
        atom_data = {"index": 15, "in_site": 5, "frac_coords": [0.7, 0.8, 0.9]}

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            json.dump(atom_data, tmp_file)
            filename = tmp_file.name

        try:
            atom = Atom.from_file(filename)

            self.assertEqual(atom.index, 15)
            self.assertEqual(atom.in_site, 5)
        finally:
            os.unlink(filename)

    def test_from_file_raises_error_for_missing_file(self):
        """Test from_file raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            Atom.from_file("nonexistent_file.json")
    
if __name__ == '__main__':
    unittest.main()
    
