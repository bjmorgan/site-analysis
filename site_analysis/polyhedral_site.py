import numpy as np 
from scipy.spatial import Delaunay, ConvexHull # type: ignore
from .site import Site
from .tools import x_pbc
from typing import List, Optional, Any, Dict
from pymatgen.core import Structure
from .atom import Atom

class PolyhedralSite(Site):
    """Describes a site defined by the polyhedral volume enclosed by a set
    of vertex atoms.

    Attributes:
        index (int): Numerical ID, intended to be unique to each site.
        label (`str`: optional): Optional string given as a label for this site.
            Default is `None`.
        contains_atoms (list): List of the atoms contained by this site in the
            structure last processed.
        trajectory (list): List of sites this atom has visited at each timestep?
        points (list): List of fractional coordinates for atoms assigned as
            occupying this site.
        transitions (collections.Counter): Stores observed transitions from this
            site to other sites. Format is {index: count} with ``index`` giving
            the index of each destination site, and ``count`` giving the number 
            of observed transitions to this site.
        vertex_indices (list(int)): List of integer indices for the vertex atoms
            (counting from 0). 
        label (:obj:`str`, optional): Optional label for the site.
   
    """ 

    def __init__(self,
            vertex_indices: List[int],
            label: Optional[str]=None):
        """Create a PolyhedralSite instance.

        Args:
            vertex_indices (list(int)): List of integer indices for the vertex atoms (counting from 0).
            label (:obj:`str`, optional): Optional label for this site.

        Returns:
            None

        """
        super(PolyhedralSite, self).__init__(label=label)
        self.vertex_indices = vertex_indices
        self.vertex_coords: Optional[np.ndarray] = None
        
        self._delaunay: Optional[Delaunay] = None
        self._faces = None
        self._hull: ConvexHull = None
        self._surface_normals: Optional[np.ndarray] = None
        self._c_signs = None


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
        self._hull = None
        self._faces = None
        self._surface_normals = None
        self._c_signs = None
        if hasattr(self, '_bounding_box'):
            del self._bounding_box

    def assign_vertex_coords(self,
            structure: Structure) -> None:
        """Assign fractional coordinates to the polyhedra vertices
        from the corresponding atom positions in a pymatgen Structure.

        Args:
            structure (Structure): The pymatgen Structure used to assign
                the vertices fractional coordinates.

        Returns:
            None

        Notes:
            This method assumes the coordinates of the vertices may 
            have changed, so unsets the Delaunay tesselation for this site.

        """
        frac_coords = np.array([ s.frac_coords for s in 
            [ structure[i] for i in self.vertex_indices ] ] )
        # Handle periodic boundary conditions:
        # If the range of fractional coordinates along x, y, or z 
        # exceeds 0.5, assume that this polyhedron wraps around the 
        # periodic boundary in that dimension. 
        # Fractional coordinates for that dimension that are less 
        # than 0.5 will be incremented by 1.0 
        for i in range(3):
            spread = max(frac_coords[:,i]) - min(frac_coords[:,i])
            if spread > 0.5:
                for j, fc in enumerate(frac_coords):
                    if fc[i] < 0.5:
                        frac_coords[j,i] += 1.0
        self.vertex_coords = frac_coords
        self._delaunay = None
        self._hull = None
        self._faces = None
        self._surface_normals = None
        self._c_signs = None

    def _calculate_surface_properties(self) -> None:
        """Calculate all surface properties at once.
        
        Mathematical Description:
        -----------------------
        For each triangular face of the convex hull, we:
        
        1. Get face vertices (v0, v1, v2) from the hull simplices
        
        2. Calculate two edge vectors of the triangle:
        edge1 = v1 - v2
        edge2 = v0 - v2
        
        3. Compute face normal vectors using cross product:
        normal = edge1 × edge2
        This gives normal vectors pointing outward from the hull
        
        4. Determine orientation of each face relative to center:
        - Calculate vector from face to polyhedron center: 
            center_vec = center - v0
        - Compute sign of dot product: sign(normal · center_vec)
        - If positive: face normal points toward center
        - If negative: face normal points away from center
        """
        current_hull = self.hull 
        self._faces = current_hull.points[current_hull.simplices]
        centre = self.centre()

        face_edges_1 = self._faces[:, 1] - self._faces[:, 2]  # v1 - v2
        face_edges_2 = self._faces[:, 0] - self._faces[:, 2]  # v0 - v2
        
        self._surface_normals = np.cross(face_edges_1, face_edges_2)
        
        self._c_signs = np.sign(np.einsum('ij,ij->i', 
                                        self._surface_normals, 
                                        centre - self._faces[:, 0]))


    def get_vertex_species(self,
            structure: Structure) -> List[str]:
        """Returns a list of species strings for the vertex atoms of this
        polyhedral site.

        Args:
            structure (Structure): Pymatgen Structure used to assign species
                to each vertex atom.

        Returns:
            (list(str)): List of species strings of the vertex atoms.

        """
        return [structure[i].species_string for i in self.vertex_indices]

    def contains_point(self,
            x: np.ndarray,
            structure: Optional[Structure]=None,
            algo: str='sn',
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

                         BEMCHMARK: Ratio (Simplex/SN): 0.51x with N=9 over 1000 runs
                         ie. 2x slower for N=9

                sn:      FASTER. Compute the sign of the surface normal for each polyhedron 
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
        if algo == 'sn':
            self._calculate_surface_properties()
        return contains_point_algos[algo](x_pbc(x))
   
    def contains_point_simplex(self,
            x: np.ndarray) -> bool:
        """Test whether one or more points are inside this site, by checking 
        whether these points are contained inside the simplices of the Delaunay 
        tesselation defined by the vertex coordinates.

        Args:
            x (np.array): Fractional coordinates for one or more points, as a
                (3x1) or (3xN) numpy array.

        Returns:a
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
            From 7.51x for N=1 to 500x faster for N=1000 compared to the previous implemntation.
  
            This is also a possible target for optimisation with f2py etc.
        """        
        if x_list.ndim == 1:
            x_list = x_list.reshape((1, 3))  # Ensure it is (N,3) for a single point
        elif x_list.shape[1] != 3:
            x_list = x_list.T  # Transpose to make it (N,3)

        # Vectorized computation of p_signs
        diffs = x_list[:, np.newaxis, :] - self.faces[np.newaxis, :, 0]
        p_signs = np.sign(np.einsum('ijk,jk->ij', diffs, self.surface_normals))
        dotsums = np.sum(self.c_signs * p_signs, axis=1)

        # Check if any point is inside by comparing dotsum with the number of faces
        inside = dotsums >= len(self.faces)

        return np.any(inside)

    def contains_atom(self,
            atom: Atom,
            algo: Optional[str]='sn',
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

    def as_dict(self) -> Dict:
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

    def centre(self) -> np.ndarray:
        """Returns the fractional coordinates of the centre point of
        this polyhedral site.

        Args:
            None

        Returns:
            (np.array): (3,) numpy array.
 
        """
        assert(isinstance(self.vertex_coords, np.ndarray))
        return np.mean(self.vertex_coords, axis=0)

    @classmethod
    def sites_from_vertex_indices(cls, vertex_indices, label=None):
        sites = [cls(vertex_indices=vi, label=label) for vi in vertex_indices]
        return sites

    @property
    def hull(self) -> ConvexHull:
        """Convex hull of the vertex coordinates for this site.
        Calculated once and cached until reset.
        """
        if self._hull is None:
            if self.vertex_coords is None:
                raise RuntimeError('No vertex coordinates set')
            self._hull = ConvexHull(self.vertex_coords)
        return self._hull

    @property
    def faces(self) -> np.ndarray:
        """Get the faces of the polyhedron."""
        if self._faces is None:
            self._calculate_surface_properties()
        return self._faces

    @property
    def surface_normals(self) -> np.ndarray:
        """Get the surface normals of the polyhedron faces."""
        if self._surface_normals is None:
            self._calculate_surface_properties()
        return self._surface_normals

    @property
    def c_signs(self) -> np.ndarray:
        """Get the c_signs for the polyhedron faces."""
        if self._c_signs is None:
            self._calculate_surface_properties()
        return self._c_signs

    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the min and max coordinates of the bounding box
        for this polyhedron in fractional coordinates.
        """
        if not hasattr(self, '_bounding_box'):
            mins = np.min(self.vertex_coords, axis=0)
            maxs = np.max(self.vertex_coords, axis=0)
            self._bounding_box = (mins, maxs)
        return self._bounding_box

 
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
