import unittest
from site_analysis.trajectory import Trajectory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.atom import Atom
from unittest.mock import Mock

class TrajectoryTestCase(unittest.TestCase):

    def test_trajectory_is_initialised(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(trajectory.atoms, atoms)
        self.assertEqual(trajectory.sites, sites)

if __name__ == '__main__':
    unittest.main()
    
