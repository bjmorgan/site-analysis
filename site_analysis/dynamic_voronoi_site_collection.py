"""Collection manager for dynamic Voronoi sites in crystal structures.

This module provides the DynamicVoronoiSiteCollection class, which manages a
collection of DynamicVoronoiSite objects and implements methods for assigning
atoms to these sites based on their positions in a crystal structure.

The DynamicVoronoiSiteCollection extends the base SiteCollection class with
specific functionality for dynamic Voronoi sites, including:
1. Calculating the dynamic centers of sites based on reference atom positions
2. Assigning atoms to sites using Voronoi tessellation principles

For atom assignment, the collection:
1. First updates each site's center by calculating the mean position of its
   reference atoms, with special handling for periodic boundary conditions
2. Calculates distances from each (dynamically determined) site center to each atom
3. Assigns each atom to the site with the nearest center
4. Uses the structure's lattice to correctly handle distances across
   periodic boundaries

This collection is particularly useful for tracking sites in frameworks
that deform during simulation, as the site centers adapt to the changing
positions of the reference atoms.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from site_analysis.site_collection import SiteCollection
from site_analysis.site import Site
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.atom import Atom
from typing import Optional, Any

class DynamicVoronoiSiteCollection(SiteCollection):
	"""A collection of DynamicVoronoiSite objects.
	
	This collection manages a set of dynamic Voronoi sites and handles
	the assignment of atoms to sites based on their dynamically calculated centres.
	
	Attributes:
		sites (list[DynamicVoronoiSite]): list of DynamicVoronoiSite objects.
	"""
	
	def __init__(self,
				 sites: list[Site]) -> None:
		"""Create a DynamicVoronoiSiteCollection instance.
		
		Args:
			sites (list[DynamicVoronoiSite]): list of DynamicVoronoiSite objects.
			
		Returns:
			None
			
		Raises:
			TypeError: If any of the sites is not a DynamicVoronoiSite.
		"""
		for s in sites:
			if not isinstance(s, DynamicVoronoiSite):
				raise TypeError("All sites must be DynamicVoronoiSite instances")
		super(DynamicVoronoiSiteCollection, self).__init__(sites)
		self.sites = self.sites  # type: list[DynamicVoronoiSite]
		
	def analyse_structure(self,
						  atoms: list[Atom],
					      structure: Structure) -> None:
		"""Analyze a structure to assign atoms to sites.
		
		This method:
		1. Assigns coordinates to atoms
		2. Calculates the centres of all dynamic Voronoi sites
		3. Assigns atoms to sites based on these centres
		
		Args:
			atoms (list[Atom]): list of atoms to be assigned to sites.
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
								atoms: list[Atom],
								structure: Structure) -> None:
		"""Assign atoms to sites based on Voronoi tessellation.
		
		This method assigns each atom to the nearest site center,
		taking into account periodic boundary conditions.
		
		Args:
			atoms (list[Atom]): list of atoms to be assigned to sites.
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