import unittest
from site_analysis.atom import Atom
from unittest.mock import patch, MagicMock, Mock
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

if __name__ == '__main__':
    unittest.main()
    
