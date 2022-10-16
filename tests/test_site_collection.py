import unittest
from site_analysis.site_collection import SiteCollection
from site_analysis.site import Site
from site_analysis.atom import Atom
from pymatgen.core import Structure
from unittest.mock import patch, Mock
import numpy as np
from collections import Counter

class ConcreteSiteCollection(SiteCollection):

    def assign_site_occupations(self,
                                atoms,
                                structure):
        raise NotImplementedError

    def analyse_structure(self,
                          atoms,
                          structure):
        raise NotImplementedError


class SiteCollectionTestCase(unittest.TestCase):

    def test_site_collection_is_initialised(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        site_collection = ConcreteSiteCollection(sites=sites)
        self.assertEqual(site_collection.sites, sites)

    def test_assign_site_occupations_raises_not_implemented_error(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        atoms = [Mock(spec=Atom), Mock(spec=Atom)]
        structure = Mock(spec=Structure)
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.assign_site_occupations(atoms, structure)

    def test_analyse_structure_raises_not_implemented_error(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        atoms = [Mock(spec=Atom), Mock(spec=Atom)]
        structure = Mock(spec=Structure)
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.analyse_structure(atoms, structure)

    def test_neighbouring_sites_raises_not_implemented_error(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        site_collection = ConcreteSiteCollection(sites=sites)
        with self.assertRaises(NotImplementedError):
            site_collection.neighbouring_sites(site_index=27)
   
    def test_site_by_index(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        site_collection = ConcreteSiteCollection(sites=sites)
        self.assertEqual(site_collection.site_by_index(12), sites[0])
        self.assertEqual(site_collection.site_by_index(42), sites[1])
        with self.assertRaises(ValueError):
            site_collection.site_by_index(93)
   
    def test_update_occupation_if_atom_was_unassigned(self):
        sites = [Mock(spec=Site)]
        sites[0].index = 12
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.in_site = None
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].contains_atoms, [atom.index])
        np.testing.assert_array_equal(sites[0].points, [atom.frac_coords])
        self.assertEqual(atom.in_site, sites[0].index)

    def test_update_occupation_if_atom_has_not_moved(self):
        sites = [Mock(spec=Site)]
        sites[0].index = 12
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.in_site = 12
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].contains_atoms, [atom.index])
        np.testing.assert_array_equal(sites[0].points, [atom.frac_coords])
        self.assertEqual(atom.in_site, sites[0].index)

    def test_update_occupation_if_atom_has_moved(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].index = 12
        sites[1].index = 42
        sites[1].transitions = Counter()
        sites[0].contains_atoms = []
        sites[0].points = []
        site_collection = ConcreteSiteCollection(sites=sites)
        site_collection.site_by_index = Mock(return_value=sites[1])
        atom = Mock(spec=Atom)
        atom.index = 4
        atom.in_site = 42
        atom.frac_coords = np.array([0.5, 0.5, 0.5])
        site_collection.update_occupation(site=sites[0], atom=atom)
        self.assertEqual(sites[0].contains_atoms, [atom.index])
        np.testing.assert_array_equal(sites[0].points, [atom.frac_coords])
        self.assertEqual(atom.in_site, sites[0].index)
        self.assertEqual(sites[1].transitions, {12: 1})
  
    def test_reset_site_occupations(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        sites[0].contains_atoms = [12]
        sites[1].contains_atoms = [42]
        site_collection = ConcreteSiteCollection(sites=sites)
        site_collection.reset_site_occupations()
        self.assertEqual(sites[0].contains_atoms, [])
        self.assertEqual(sites[1].contains_atoms, [])
   
    def test_sites_contain_points_raises_not_implemented_error(self):
        sites = [Mock(spec=Site), Mock(spec=Site)]
        site_collection = ConcreteSiteCollection(sites=sites)
        points = np.array([[0.0, 0.0, 0.0],
                           [0.5, 0.5, 0.5]])
        with self.assertRaises(NotImplementedError):
            site_collection.sites_contain_points(points=points)  

if __name__ == '__main__':
    unittest.main()
    
