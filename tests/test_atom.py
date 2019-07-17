import unittest
from site_analysis import Atom
from unittest.mock import patch, MagicMock, Mock
import numpy as np

class AtomTestCase(unittest.TestCase):

    def test_atom_is_initialised(self):
        with patch('site_analysis.Atom.newid') as mock_new_id:
            mock_new_id.__next__ = Mock(side_effect = [1])
            species_string = 'foo'
            atom = Atom(species_string=species_string)
            self.assertEqual( atom.species_string, species_string )
            self.assertEqual( atom.index, int(Atom.newid) )
            self.assertEqual( atom.in_site, None )
            self.assertEqual( atom._frac_coords, None )
            self.assertEqual( atom.trajectory, [] )

    def test_reset(self):
        atom = Atom(species_string='Li')
        atom.in_site = 3
        atom._frac_coords = np.array( [0,0, 0.0, 0.0] )
        atom.trajectory = [1,2,3]
        atom.reset()
        self.assertEqual( atom.in_site, None )
        self.assertEqual( atom._frac_coords, None )
        self.assertEqual( atom.trajectory, [] )

if __name__ == '__main__':
    unittest.main()
    
