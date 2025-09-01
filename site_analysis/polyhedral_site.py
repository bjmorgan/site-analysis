"""Polyhedral site representation for crystal structure analysis.

This module provides the PolyhedralSite class, which represents a site defined
by a polyhedron formed by a set of vertex atoms. These sites are commonly used
to represent coordination environments in crystal structures, such as tetrahedral
or octahedral sites.
"""

from __future__ import annotations

import itertools
import numpy as np 
from scipy.spatial import Delaunay, ConvexHull # type: ignore
from pymatgen.core import Structure
from site_analysis.site import Site
from site_analysis.tools import x_pbc, species_string_from_site
from site_analysis.atom import Atom
from site_analysis.pbc_utils import apply_legacy_pbc_correction, unwrap_vertices_to_reference_center
from typing import Optional, Any


class PolyhedralSite(Site):
    """Describes a site defined by the polyhedral volume enclosed by a set
    of vertex atoms.
    
    A PolyhedralSite determines whether atoms are inside the site volume by constructing
    a convex polyhedron from the vertex atoms and checking whether points lie within
    this polyhedron. It supports multiple algorithms for this containment check:
    
    - 'simplex': Uses Delaunay tessellation to check if a point is in any simplex
    - 'sn': Uses surface normal directions to check if a point is inside all faces
    
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
        label: Optional[str]=None,
        reference_center: Optional[np.ndarray]=None):
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
        print(vertex_indices)
        
        if not all(isinstance(idx, int) for idx in vertex_indices):
            raise TypeError("All vertex indices must be integers")
        
        super(PolyhedralSite, self).__init__(label=label)
        self.vertex_indices = vertex_indices
        self.vertex_coords: Optional[np.ndarray] = None
        self._delaunay: Optional[Delaunay] = None
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

        Resets the contains_atoms and trajectory attributes
        to empty lists.

        Vertex coordinates and Delaunay tesselation are unset.

        Args:
            None

        Returns:
            None

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
            structure: Optional[Structure]=None,
            algo: str='simplex',
            *args,
            **kwargs) -> bool:
        """Test whether a specific point is enclosed by this polyhedral site.

        Args:
            x (np.array): Fractional coordinates of the point to test (length 3 numpy array).
            structure (:obj:`Structure`, optional): Optional pymatgen Structure. If provided,
                the vertex coordinates for this polyhedral site will be assigned using
                this structure. Default is None.
            algo (str): Select the algorithm for testing whether a point is contained
                by the site:
    
                simplex: Use scipy.spatial.Delaunay.find_simplex to test if any of
                         the simplices that make up this polyhedron contain the point.
       
                sn:      Compute the sign of the surface normal for each polyhedron 
                         face with respect to the point, to test if the point lies
                         "inside" every face.
                
        Returns:
            bool

        """
        contains_point_algos = {'simplex': self.contains_point_simplex,
                                'sn': self.contains_point_sn}
        if algo not in contains_point_algos.keys():
            raise ValueError(f'{algo} is not a valid algorithm keyword for contains_point()')
        if structure:
            self.assign_vertex_coords(structure)
        if self.vertex_coords is None:
            raise RuntimeError('no vertex coordinates set for polyhedral_site {}'.format(self.index))
        return contains_point_algos[algo](x_pbc(x))
   
    def contains_point_simplex(self,
            x: np.ndarray) -> bool:
        """Test whether one or more points are inside this site, by checking 
        whether these points are contained inside the simplices of the Delaunay 
        tesselation defined by the vertex coordinates.

        Args:
            x (np.array): Fractional coordinates for one or more points, as a
                (3x1) or (3xN) numpy array.

        Returns:
            bool

        """
        return bool(np.any(self.delaunay.find_simplex(x) >= 0))
 
    def contains_point_sn(self,
            x_list: np.ndarray) -> bool:
        """Test whether one or more points are inside this site, by calculating 
        the sign of the surface normal for each face with respect to each point.

        Args:
            x (np.array): Fractional coordinates for one or more points, as a
                (3x1) or (3xN) numpy array.

        Returns:
            bool

        Note:
            This method could be made more efficient by caching the 
            surface_normal vectors and in-face vectors.

            This is also a possible target for optimisation with f2py etc.

        """
        hull = ConvexHull(self.vertex_coords)
        faces = hull.points[hull.simplices]
        centre = self.centre
        inside = []
        for x in x_list:
            dotsum = 0
            for f in faces:
                surface_normal = np.cross(f[0]-f[2], f[1]-f[2])
                c_sign = np.sign(np.dot( surface_normal, centre-f[0]))
                p_sign = np.sign(np.dot( surface_normal, x-f[0]))
                dotsum += c_sign * p_sign
                inside.append(dotsum == len(faces))
        return any(inside)

    def contains_atom(self,
            atom: Atom,
            algo: Optional[str]='simplex',
            *args: Any,
            **kwargs: Any) -> bool:
        """Test whether an atom is inside this polyhedron.

        Args:
            atom (Atom): The atom to test.
            algo (:obj:`str`, optional): Select the algorithm to us. Options are
                'simplex' and 'sn'. See the documentation for the contains_point()
                method for more details. Default is 'simplex'.

        Returns:
            bool
        """
        contains_point_algos = ['simplex', 'sn']
        if algo not in contains_point_algos:
            raise ValueError(f'{algo} is not a valid algorithm keyword for contains_atom()')
        return self.contains_point(atom.frac_coords, algo=algo)

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
        assert isinstance(self.vertex_coords, np.ndarray)
        centre_coords = np.mean(self.vertex_coords, axis=0)
        return np.array(centre_coords) 

    @classmethod
    def sites_from_vertex_indices(cls,
        vertex_indices: list[list[int]],
        label: Optional[str]=None) -> list[PolyhedralSite]:
        sites = [cls(vertex_indices=vi, label=label) for vi in vertex_indices]
        return sites
