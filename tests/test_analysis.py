import unittest
from site_analysis import Analysis

class AnalysisTestCase(unittest.TestCase):

    def test_analysis_is_initialised(self):
        atoms = 'foo'
        sites = 'bar'
        analysis = Analysis(atoms=atoms, sites=sites)
        self.assertEqual( analysis.atoms, atoms )
        self.assertEqual( analysis.sites, sites )

if __name__ == '__main__':
    unittest.main()
    
