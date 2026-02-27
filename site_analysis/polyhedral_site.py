"""Polyhedral site representation for crystal structure analysis.

This module provides the PolyhedralSite class, which represents a site defined
by a polyhedron formed by a set of vertex atoms. These sites are commonly used
to represent coordination environments in crystal structures, such as tetrahedral
or octahedral sites.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy.spatial import Delaunay # type: ignore
from pymatgen.core import Structure
from site_analysis.site import Site
from site_analysis.tools import x_pbc, species_string_from_site
from site_analysis.atom import Atom
from site_analysis.containment import HAS_NUMBA, FaceTopologyCache
from site_analysis.pbc_utils import apply_legacy_pbc_correction, unwrap_vertices_to_reference_center
from typing import Any


class PolyhedralSite(Site):
    """Describes a site defined by the polyhedral volume enclosed by a set
    of vertex atoms.

    A PolyhedralSite determines whether atoms are inside the site volume by
    constructing a convex polyhedron from the vertex atoms and checking whether
    points lie within this polyhedron. The containment algorithm is selected
    automatically: when numba is available, a JIT-compiled surface normal method
    with cached face topology is used; otherwise, falls back to Delaunay
    tessellation via scipy.

    The polyhedron vertices are defined using atom indices in a structure, and
    their coordinates are assigned from the structure when needed. This allows the
    polyhedron shape to adapt to changes in the crystal structure.

    Attributes:
        vertex_indices (list[int]): List of integer indices for the vertex atoms
            (counting from 0).
        vertex_coords (np.ndarray or None): Fractional coordinates of the vertices.
            Set using assign_vertex_coords() from a Structure.
        reference_center (np.ndarray or None): Optional reference centre for PBC handling.

    See Also:
        :class:`~site_analysis.site.Site`: Parent class documenting inherited attributes
            (index, label, contains_atoms, trajectory, points, transitions).
    """

    def __init__(self,
        vertex_indices: list[int],
        label: str | None=None,
        reference_center: np.ndarray | None=None):
        """Create a PolyhedralSite instance.
        
        Args:
            vertex_indices: List of integer indices for the vertex atoms (counting from 0).
            label: Optional label for this site.
            reference_center: Optional reference centre for PBC handling.
        
        Returns:
            None
        
        Raises:
            ValueError: If vertex_indices is empty.
            TypeError: If any element in vertex_indices is not an integer.
        """
        if isinstance(vertex_indices, np.ndarray):
            vertex_indices = vertex_indices.tolist()
        
        if not vertex_indices:
            raise ValueError("vertex_indices cannot be empty")

        if not all(isinstance(idx, int) for idx in vertex_indices):
            raise TypeError("All vertex indices must be integers")
        
        super(PolyhedralSite, self).__init__(label=label)
        self.vertex_indices = vertex_indices
        self.vertex_coords: np.ndarray | None = None
        self._delaunay: Delaunay | None = None
        self._face_topology_cache: FaceTopologyCache | None = None
        self.reference_center = reference_center

    def __repr__(self) -> str:
        string = ('site_analysis.PolyhedralSite('
                  f'index={self.index}, '
                  f'label={self.label}, '
                  f'vertex_indices={self.vertex_indices}, '
                  f'contains_atoms={self.contains_atoms})')
        return string
                  
    def reset(self) -> None:
        """Reset the trajectory for this site.

        Resets the contains_atoms and trajectory attributes to empty lists.
        Vertex coordinates and Delaunay tessellation are unset. The face
        topology cache is preserved as it depends only on vertex indices,
        which are immutable.
        """
        super(PolyhedralSite, self).reset()
        self.vertex_coords = None
        self._delaunay = None
 
    @property
    def delaunay(self) -> Delaunay:
        """Delaunay tessellation of the vertex coordinates for this site.

        This is calculated the first time the attribute is requested,
        and then stored for reuse, unless the site is reset.

        Returns:
            scipy.spatial.Delaunay

        """
        if not self._delaunay:
            if self.vertex_coords is None:
                raise RuntimeError("Vertex coordinates have not been assigned.")
            self._delaunay = Delaunay(self.vertex_coords)
        return self._delaunay

    @property
    def coordination_number(self) -> int:
        """Coordination number for this site, defined as the number of 
        vertices

        Returns:
            int

        """
        return len(self.vertex_indices)
    
    @property
    def cn(self) -> int:
        """Coordination number for this site, defined as the number of
        vertices

        Convenience property for coordination_number()

        Returns:
            int

        """
        return self.coordination_number
        
    def assign_vertex_coords(self,
            structure: Structure) -> None:
        """Assign fractional coordinates to the polyhedra vertices
        from the corresponding atom positions in a pymatgen Structure.
        
        Args:
            structure (Structure): The pymatgen Structure used to assign
                the fractional coordinates of the vertices.
        
        Returns:
            None
        
        Notes:
            This method assumes the coordinates of the vertices may 
            have changed, so unsets the Delaunay tesselation for this site.
        
        """
        frac_coords = np.array([s.frac_coords for s in 
            [structure[i] for i in self.vertex_indices]])
        if self.reference_center is not None:
            frac_coords = unwrap_vertices_to_reference_center(frac_coords, self.reference_center, structure.lattice)
        else:
            frac_coords = apply_legacy_pbc_correction(frac_coords)
        self.vertex_coords = frac_coords
        self._delaunay = None
        if self._face_topology_cache is not None:
            self._face_topology_cache.update(self.vertex_coords)

    def get_vertex_species(self,
            structure: Structure) -> list[str]:
        """Returns a list of species strings for the vertex atoms of this
        polyhedral site.

        Args:
            structure (Structure): Pymatgen Structure used to assign species
                to each vertex atom.

        Returns:
            (list(str)): list of species strings of the vertex atoms.

        """
        return [structure[i].species_string for i in self.vertex_indices]

    def contains_point(self,
            x: np.ndarray,
            structure: Structure | None = None,
            algo: str | None = None,
            *args,
            **kwargs) -> bool:
        """Test whether a specific point is enclosed by this polyhedral site.

        The containment algorithm is selected automatically based on available
        dependencies. When numba is installed, uses a JIT-compiled surface
        normal method. Otherwise, falls back to Delaunay tessellation.

        Args:
            x: Fractional coordinates of the point to test (length 3 array).
            structure: Optional pymatgen Structure. If provided, the vertex
                coordinates for this polyhedral site will be assigned using
                this structure.
            algo: Deprecated. Previously selected the algorithm. Now ignored;
                the best available method is used automatically.

        Returns:
            True if the point is inside the polyhedron.
        """
        if algo is not None:
            warnings.warn(
                "The 'algo' parameter is deprecated and will be removed in a "
                "future version. The best available containment algorithm is "
                "now selected automatically.",
                DeprecationWarning,
                stacklevel=2,
            )
        if structure:
            self.assign_vertex_coords(structure)
        if self.vertex_coords is None:
            raise RuntimeError(
                f'no vertex coordinates set for polyhedral_site {self.index}'
            )
        x_images = x_pbc(x)
        if HAS_NUMBA:
            if self._face_topology_cache is None:
                self._face_topology_cache = FaceTopologyCache(self.vertex_coords)
            return self._face_topology_cache.contains_point(x_images)
        return self._contains_point_delaunay(x_images)
   
    def _contains_point_delaunay(self, x: np.ndarray) -> bool:
        """Test containment using Delaunay tessellation.

        Args:
            x: Fractional coordinates as (3,) or (N, 3) array.

        Returns:
            True if any point is inside a simplex of the tessellation.
        """
        return bool(np.any(self.delaunay.find_simplex(x) >= 0))

    def contains_atom(self,
            atom: Atom,
            algo: str | None = None,
            *args: Any,
            **kwargs: Any) -> bool:
        """Test whether an atom is inside this polyhedron.

        Args:
            atom: The atom to test.
            algo: Deprecated. Previously selected the algorithm. Now ignored;
                the best available method is used automatically.

        Returns:
            True if the atom is inside the polyhedron.
        """
        if algo is not None:
            warnings.warn(
                "The 'algo' parameter is deprecated and will be removed in a "
                "future version. The best available containment algorithm is "
                "now selected automatically.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.contains_point(atom.frac_coords)

    def as_dict(self) -> dict:
        d = super(PolyhedralSite, self).as_dict()
        d['vertex_indices'] = self.vertex_indices
        d['vertex_coords'] = self.vertex_coords
        return d

    @classmethod
    def from_dict(cls, d):
        polyhedral_site = cls(vertex_indices=d['vertex_indices'])
        polyhedral_site.vertex_coords = d['vertex_coords']
        polyhedral_site.contains_atoms = d['contains_atoms']
        polyhedral_site.label = d.get('label')
        return polyhedral_site 

    @property
    def centre(self) -> np.ndarray:
        """Returns the fractional coordinates of the centre point of
        this polyhedral site.

        Args:
            None

        Returns:
            (np.array): (3,) numpy array.
 
        """
        if self.vertex_coords is None:
            raise RuntimeError("Vertex coordinates have not been assigned.")
        centre_coords = np.mean(self.vertex_coords, axis=0)
        return np.array(centre_coords) 

    @classmethod
    def sites_from_vertex_indices(cls,
        vertex_indices: list[list[int]],
        label: str | None=None) -> list[PolyhedralSite]:
        sites = [cls(vertex_indices=vi, label=label) for vi in vertex_indices]
        return sites
