import unittest
from site_analysis.atom import Atom
from unittest.mock import patch, MagicMock, Mock
from pymatgen import Structure, Lattice
import numpy as np

class AtomTestCase(unittest.TestCase):

    def test_atom_is_initialised(self):
        atom = Atom(index=22)
        self.assertEqual(atom.index, 22)
        self.assertEqual(atom.in_site, None)
        self.assertEqual(atom._frac_coords, None)
        self.assertEqual(atom.trajectory, [])

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
        self.assertEqual(atom.__repr__(), 'site_analysis.Atom(index=12, in_site=None, frac_coords=None)')

    def test_assign_coords(self):
        atom = Atom(index=1)
        structure = example_structure()
        atom.assign_coords(structure=structure)
        np.testing.assert_array_equal(atom._frac_coords, structure[1].frac_coords )
   
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
        
if __name__ == '__main__':
    unittest.main()
    
