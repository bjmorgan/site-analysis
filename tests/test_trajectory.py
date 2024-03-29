import unittest
from site_analysis.trajectory import Trajectory
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.atom import Atom
from unittest.mock import Mock, patch

class TrajectoryTestCase(unittest.TestCase):

    def test_trajectory_is_initialised(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.trajectory.PolyhedralSiteCollection') as mock_PolyhedralSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(trajectory.atoms, atoms)
        self.assertEqual(trajectory.sites, sites)
        mock_PolyhedralSiteCollection.assert_called_with(sites)

    def test___len___returns_zero_for_empty_trajectory(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.trajectory.PolyhedralSiteCollection') as mock_PolyhedralSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        self.assertEqual(len(trajectory), 0)

    def test___len___returns_number_of_timesteps(self):
        atoms = [Mock(spec=Atom, index=2)]
        sites = [Mock(spec=PolyhedralSite, index=1)]
        with patch('site_analysis.trajectory.PolyhedralSiteCollection') as mock_PolyhedralSiteCollection:
            trajectory = Trajectory(atoms=atoms, sites=sites)
        trajectory.timesteps = [ 'foo', 'bar' ]
        self.assertEqual(len(trajectory), 2)

    def test_init_raises_valueerror_if_passed_mixed_site_types(self):
        sites = [Mock(spec=PolyhedralSite, index=1),
                 Mock(spec=VoronoiSite, index=2)]
        atoms = [Mock(spec=Atom, index=3)]
        with self.assertRaises(TypeError):
            trajectory = Trajectory(atoms=atoms, sites=sites)

    def test_init_raises_valueerror_if_passed_invalid_site_type(self):
        sites = ["foo"]
        atoms = [Mock(spec=Atom, index=3)]
        with self.assertRaises(TypeError):
            trajectory = Trajectory(atoms=atoms, sites=sites)
        
if __name__ == '__main__':
    unittest.main()
    
