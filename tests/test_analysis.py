import unittest
from site_analysis import Analysis

class AnalysisTestCase(unittest.TestCase):

    def test_analysis_is_initialised(self):
        atoms = 'foo'
        polyhedra = 'bar'
        analysis = Analysis(atoms=atoms, polyhedra=polyhedra)
        self.assertEqual( analysis.atoms, atoms )
        self.assertEqual( analysis.polyhedra, polyhedra )

if __name__ == '__main__':
    unittest.main()
    
