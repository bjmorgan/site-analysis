import unittest
from site_analysis.tools import get_vertex_indices, x_pbc
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from pymatgen import Lattice, Structure
from collections import Counter

class ToolsTestCase(unittest.TestCase):

    def test_get_vertex_indices(self):
        # Create a 2x2x2 NaCl supercell
        lattice = Lattice.from_parameters(a=5.0, b=5.0, c=5.0, 
                      alpha=90, beta=90, gamma=90)
        structure = Structure.from_spacegroup(sg='Fm-3m', lattice=lattice, 
                                              species=['Na','Cl'],
                                              coords=[[0.0, 0.0, 0.0],
                                                      [0.5, 0.0, 0.0]])*[2,2,2]
        vertex_indices = get_vertex_indices(structure=structure, centre_species='Na',
                                            vertex_species='Cl', cutoff=3.0,
                                            n_vertices=6)
        c = Counter()
        for vi in vertex_indices:
            self.assertEqual( len(vi), 6 )
            c += Counter( vi )
            for i in vi:
                self.assertEqual( structure[i].species_string, 'Cl' )
        for i in range(33,64):
            self.assertEqual( c[i], 6 )

    def test_x_pbc(self):
        pbc_coords = x_pbc(np.array([0.1, 0.2, 0.3]))
        expected_coords = np.array( [[ 0.1, 0.2, 0.3 ],
                                     [ 1.1, 0.2, 0.3 ],
                                     [ 0.1, 1.2, 0.3 ],
                                     [ 0.1, 0.2, 1.3 ],
                                     [ 1.1, 1.2, 0.3 ],
                                     [ 1.1, 0.2, 1.3 ], 
                                     [ 0.1, 1.2, 1.3 ],
                                     [ 1.1, 1.2, 1.3 ]] )
        for c in expected_coords:
            self.assertEqual( c in pbc_coords, True )
 
if __name__ == '__main__':
    unittest.main()
    
