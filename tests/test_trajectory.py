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

    def test___len___returns_zero_for_empty_trajectory(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(len(trajectory), 0)

    def test___len___returns_number_of_timesteps(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        trajectory = Trajectory(atoms=atoms, sites=sites)
        trajectory.timesteps = [ 'foo', 'bar' ]
        self.assertEqual(len(trajectory), 2)

if __name__ == '__main__':
    unittest.main()
    
