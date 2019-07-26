import unittest
from site_analysis import Analysis
from site_analysis import PolyhedralSite, Atom
from unittest.mock import Mock

class AnalysisTestCase(unittest.TestCase):

    def test_analysis_is_initialised(self):
        atoms = [ Mock(spec=Atom, index=2) ]
        sites = [ Mock(spec=PolyhedralSite, index=1) ]
        analysis = Analysis(atoms=atoms, sites=sites)
        self.assertEqual( analysis.atoms, atoms )
        self.assertEqual( analysis.sites, sites )

if __name__ == '__main__':
    unittest.main()
    
