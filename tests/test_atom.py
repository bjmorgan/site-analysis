import unittest
from site_analysis import Atom
from unittest.mock import patch, MagicMock, Mock

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
       
if __name__ == '__main__':
    unittest.main()
    
