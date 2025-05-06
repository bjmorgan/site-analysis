from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, Any
from pymatgen.core import Structure
from .site_collection import SiteCollection
from .site import Site
from .dynamic_voronoi_site import DynamicVoronoiSite
from .atom import Atom

class DynamicVoronoiSiteCollection(SiteCollection):
	"""A collection of DynamicVoronoiSite objects.
	
	This collection manages a set of dynamic Voronoi sites and handles
	the assignment of atoms to sites based on their dynamically calculated centres.
	
	Attributes:
		sites (List[DynamicVoronoiSite]): List of DynamicVoronoiSite objects.
	"""
	
	def __init__(self,
				 sites: List[Site]) -> None:
		"""Create a DynamicVoronoiSiteCollection instance.
		
		Args:
			sites (List[DynamicVoronoiSite]): List of DynamicVoronoiSite objects.
			
		Returns:
			None
			
		Raises:
			TypeError: If any of the sites is not a DynamicVoronoiSite.
		"""
		for s in sites:
			if not isinstance(s, DynamicVoronoiSite):
				raise TypeError("All sites must be DynamicVoronoiSite instances")
		super(DynamicVoronoiSiteCollection, self).__init__(sites)
		self.sites = self.sites  # type: List[DynamicVoronoiSite]
		
	def analyse_structure(self,
						  atoms: List[Atom],
					      structure: Structure) -> None:
		"""Analyze a structure to assign atoms to sites.
		
		This method:
		1. Assigns coordinates to atoms
		2. Calculates the centres of all dynamic Voronoi sites
		3. Assigns atoms to sites based on these centres
		
		Args:
			atoms (List[Atom]): List of atoms to be assigned to sites.
			structure (Structure): Pymatgen Structure containing atom positions.
			
		Returns:
			None
		"""
		for atom in atoms:
			atom.assign_coords(structure)
		for site in self.sites:
			site.calculate_centre(structure)
		self.assign_site_occupations(atoms, structure)
		
	def assign_site_occupations(self,
								atoms: List[Atom],
								structure: Structure) -> None:
		"""Assign atoms to sites based on Voronoi tessellation.
		
		This method assigns each atom to the nearest site center,
		taking into account periodic boundary conditions.
		
		Args:
			atoms (List[Atom]): List of atoms to be assigned to sites.
			structure (Structure): Pymatgen Structure containing atom positions.
			
		Returns:
			None
		"""
		self.reset_site_occupations()
		if not atoms:
			return
		lattice = structure.lattice
		site_coords = np.array([site.centre for site in self.sites])
		atom_coords = np.array([atom.frac_coords for atom in atoms])
		dist_matrix = lattice.get_all_distances(site_coords, atom_coords)
		site_list_indices = np.argmin(dist_matrix, axis=0)
		for atom, site_list_index in zip( atoms, site_list_indices):
			site = self.sites[site_list_index]
			self.update_occupation(site, atom)