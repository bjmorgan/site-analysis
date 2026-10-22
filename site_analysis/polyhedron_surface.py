"""Fixed surface triangles of a polyhedral site, and containment within them.

The surface of a polyhedral site is a fixed set of triangles over its
vertices: the convex hull of its vertices, with each face it shares with a
neighbouring site triangulated by a rule that depends only on the shared
atoms, so that both sites give the face the same triangles. A point is
inside the site when it is inside one of the tetrahedra formed by the
site's centre (the mean of its vertex coordinates) and a surface triangle,
which bounds the site exactly while it is strictly star-shaped about its
centre: the centre is strictly on the inner side of every triangle's plane.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Collection, Sequence

import numpy as np
from scipy.spatial import ConvexHull

from site_analysis._compat import HAS_NUMBA
from site_analysis.tools import _SHIFTS

if HAS_NUMBA:
    import numba

    @numba.njit(cache=True)
    def _numba_star_planes(
        vertex_coords: np.ndarray,
        triangles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        """JIT-compiled inward planes of the tetrahedra of a star-shaped polyhedron.

        Performs the arithmetic of ``_numpy_star_planes`` in the same
        order, without temporary arrays.

        Args:
            vertex_coords: (N_vertices, 3) array of vertex positions.
            triangles: (N_triangles, 3) array of vertex indices per surface
                triangle, each wound so that ``(b - a) x (c - a)`` points
                away from the centre.

        Returns:
            Tuple of (normals, ref_points, star_shaped, within_two_cells),
            as returned by ``_numpy_star_planes``.
        """
        n_vertices = vertex_coords.shape[0]
        n_triangles = triangles.shape[0]
        c0 = vertex_coords[0, 0]
        c1 = vertex_coords[0, 1]
        c2 = vertex_coords[0, 2]
        for i in range(1, n_vertices):
            c0 += vertex_coords[i, 0]
            c1 += vertex_coords[i, 1]
            c2 += vertex_coords[i, 2]
        c0 /= n_vertices
        c1 /= n_vertices
        c2 /= n_vertices
        within_two_cells = True
        for i in range(n_vertices):
            for k in range(3):
                if not (0.0 <= vertex_coords[i, k] < 2.0):
                    within_two_cells = False

        normals = np.empty((n_triangles, 4, 3))
        ref_points = np.empty((n_triangles, 4, 3))
        star_shaped = True
        for t in range(n_triangles):
            ia = triangles[t, 0]
            ib = triangles[t, 1]
            ic = triangles[t, 2]
            ax = vertex_coords[ia, 0]
            ay = vertex_coords[ia, 1]
            az = vertex_coords[ia, 2]
            ux = vertex_coords[ib, 0] - ax
            uy = vertex_coords[ib, 1] - ay
            uz = vertex_coords[ib, 2] - az
            vx = vertex_coords[ic, 0] - ax
            vy = vertex_coords[ic, 1] - ay
            vz = vertex_coords[ic, 2] - az
            nx = uz * vy - uy * vz
            ny = ux * vz - uz * vx
            nz = uy * vx - ux * vy
            if not nx * (c0 - ax) + ny * (c1 - ay) + nz * (c2 - az) > 0.0:
                star_shaped = False
            normals[t, 3, 0] = nx
            normals[t, 3, 1] = ny
            normals[t, 3, 2] = nz
            ref_points[t, 3, 0] = ax
            ref_points[t, 3, 1] = ay
            ref_points[t, 3, 2] = az
            for q in range(3):
                i0 = triangles[t, q]
                i1 = triangles[t, (q + 1) % 3]
                px = vertex_coords[i0, 0] - c0
                py = vertex_coords[i0, 1] - c1
                pz = vertex_coords[i0, 2] - c2
                rx = vertex_coords[i1, 0] - c0
                ry = vertex_coords[i1, 1] - c1
                rz = vertex_coords[i1, 2] - c2
                normals[t, q, 0] = py * rz - pz * ry
                normals[t, q, 1] = pz * rx - px * rz
                normals[t, q, 2] = px * ry - py * rx
                ref_points[t, q, 0] = c0
                ref_points[t, q, 1] = c1
                ref_points[t, q, 2] = c2
        return normals, ref_points, star_shaped, within_two_cells

    @numba.njit(cache=True)
    def _numba_star_query(
        x: np.ndarray,
        image_shifts: np.ndarray,
        normals: np.ndarray,
        ref_points: np.ndarray,
    ) -> bool:
        """JIT-compiled test of whether an image of a point is in a tetrahedron.

        Gives the result of ``_numpy_star_query`` with the same
        arithmetic, exiting on the first plane an image is outside of and
        on the first tetrahedron containing an image.

        Args:
            x: (3,) fractional coordinates of the point, in any periodic
                image.
            image_shifts: (M, 3) integer array of whole-cell shifts from
                the wrapped point to the images to test.
            normals: (N_triangles, 4, 3) inward plane normals.
            ref_points: (N_triangles, 4, 3) a point on each plane.

        Returns:
            True if an image is on the non-negative side of all four
            planes of a tetrahedron.

        Raises:
            ValueError: If ``x`` is not three finite numbers.
        """
        if x.shape[0] != 3 or not (np.isfinite(x[0]) and np.isfinite(x[1])
                                   and np.isfinite(x[2])):
            raise ValueError("point must be three finite numbers")
        w0 = x[0] - np.floor(x[0])
        w1 = x[1] - np.floor(x[1])
        w2 = x[2] - np.floor(x[2])
        for i in range(image_shifts.shape[0]):
            p0 = w0 + image_shifts[i, 0]
            p1 = w1 + image_shifts[i, 1]
            p2 = w2 + image_shifts[i, 2]
            for t in range(normals.shape[0]):
                inside = True
                for q in range(4):
                    dot = ((p0 - ref_points[t, q, 0]) * normals[t, q, 0]
                           + (p1 - ref_points[t, q, 1]) * normals[t, q, 1]
                           + (p2 - ref_points[t, q, 2]) * normals[t, q, 2])
                    if not dot >= 0.0:
                        inside = False
                        break
                if inside:
                    return True
        return False

_MAX_IMAGE_SHIFTS = 1000
_FLOAT64 = np.dtype(np.float64)
_NDARRAY = np.ndarray


def _image_shifts(vertex_coords: np.ndarray) -> np.ndarray:
    """Whole-cell shifts from a point in [0, 1) to its images that can lie inside a polyhedron.

    An image ``x + n`` of a point ``x`` in [0, 1) lies inside the
    vertices' bounding box only if ``floor(min_k) <= n_k <= floor(max_k)``
    along each axis ``k``, where ``min_k`` and ``max_k`` are the lowest and
    highest vertex coordinates.

    Args:
        vertex_coords: (N_vertices, 3) array of finite vertex positions.

    Returns:
        (M, 3) integer array of the shifts, one for each cell the bounding
        box meets.

    Raises:
        ValueError: If the bounding box meets more than
            ``_MAX_IMAGE_SHIFTS`` cells, as it can for a site in a
            strongly skewed or unreduced cell.
    """
    lower = np.floor(vertex_coords.min(axis=0)).astype(np.int64)
    upper = np.floor(vertex_coords.max(axis=0)).astype(np.int64)
    cells = upper - lower + 1
    if np.prod(cells, dtype=np.float64) > _MAX_IMAGE_SHIFTS:
        raise ValueError(
            f"vertices span {cells.tolist()} cells along the lattice axes, more than "
            f"the {_MAX_IMAGE_SHIFTS} cells in which a site's images can be tested; "
            "use a reduced cell")
    axes = [np.arange(lo, hi + 1) for lo, hi in zip(lower, upper)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _numpy_star_planes(
    vertex_coords: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Inward planes of the tetrahedra formed by a polyhedron's centre and its surface triangles.

    For each triangle ``(a, b, c)`` the four planes are, in order, the
    planes through the centre ``o`` and each edge, with normals
    ``(a - o) x (b - o)``, ``(b - o) x (c - o)`` and ``(c - o) x (a - o)``
    through ``o``, and the triangle's own plane, with normal
    ``-(b - a) x (c - a)`` through ``a``. The numba query tests them in
    this order, so the edge planes, which reject most points outside a
    tetrahedron, come first. When the centre is strictly on the inner side
    of every triangle's plane, every normal points into its tetrahedron.

    Args:
        vertex_coords: (N_vertices, 3) array of vertex positions.
        triangles: (N_triangles, 3) array of vertex indices per surface
            triangle, each wound so that ``(b - a) x (c - a)`` points away
            from the centre.

    Returns:
        Tuple of (normals, ref_points, star_shaped, within_two_cells):
        (N_triangles, 4, 3) arrays of the plane normals and of a point on
        each plane; whether the centre is strictly on the inner side of
        every triangle's plane; and whether every vertex coordinate is in
        [0, 2).
    """
    centre = np.add.accumulate(vertex_coords, axis=0)[-1] / len(vertex_coords)
    corners = vertex_coords[triangles]
    a = corners[:, 0]
    u = corners[:, 1] - a
    v = corners[:, 2] - a
    normals = np.empty((len(triangles), 4, 3))
    ref_points = np.empty((len(triangles), 4, 3))
    normals[:, 3, 0] = u[:, 2] * v[:, 1] - u[:, 1] * v[:, 2]
    normals[:, 3, 1] = u[:, 0] * v[:, 2] - u[:, 2] * v[:, 0]
    normals[:, 3, 2] = u[:, 1] * v[:, 0] - u[:, 0] * v[:, 1]
    ref_points[:, :3] = centre
    ref_points[:, 3] = a
    to_centre = centre - a
    centre_side = (normals[:, 3, 0] * to_centre[:, 0] + normals[:, 3, 1] * to_centre[:, 1]
                   + normals[:, 3, 2] * to_centre[:, 2])
    p = corners - centre
    r = p[:, [1, 2, 0]]
    normals[:, :3, 0] = p[..., 1] * r[..., 2] - p[..., 2] * r[..., 1]
    normals[:, :3, 1] = p[..., 2] * r[..., 0] - p[..., 0] * r[..., 2]
    normals[:, :3, 2] = p[..., 0] * r[..., 1] - p[..., 1] * r[..., 0]
    star_shaped = bool(np.all(centre_side > 0.0))
    within_two_cells = bool(np.all((0.0 <= vertex_coords) & (vertex_coords < 2.0)))
    return normals, ref_points, star_shaped, within_two_cells


def _numpy_star_query(
    x: np.ndarray,
    image_shifts: np.ndarray,
    normals: np.ndarray,
    ref_points: np.ndarray,
) -> bool:
    """Test whether an image of a point is in one of a polyhedron's tetrahedra.

    The point is wrapped into [0, 1) and each of its images given by
    ``image_shifts`` is tested against the four planes of each
    tetrahedron.

    Args:
        x: (3,) fractional coordinates of the point, in any periodic image.
        image_shifts: (M, 3) integer array of whole-cell shifts from the
            wrapped point to the images to test.
        normals: (N_triangles, 4, 3) inward plane normals.
        ref_points: (N_triangles, 4, 3) a point on each plane.

    Returns:
        True if an image is on the non-negative side of all four planes of
        a tetrahedron.

    Raises:
        ValueError: If ``x`` is not three finite numbers.
    """
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("point must be three finite numbers")
    images = (x - np.floor(x)) + image_shifts
    d = images[:, np.newaxis, np.newaxis, :] - ref_points
    dots = d[..., 0] * normals[..., 0] + d[..., 1] * normals[..., 1] + d[..., 2] * normals[..., 2]
    return bool(np.any(np.all(dots >= 0.0, axis=2)))


class _SharedFaceError(ValueError):
    """A shared atom set that cannot be triangulated identically on a site.

    Attributes:
        atoms: Atom indices of the shared set, in ascending order.
    """

    def __init__(self, atoms: Sequence[int], vertex_indices: Sequence[int]) -> None:
        """Describe the shared atom set and the site's vertices.

        Args:
            atoms: Atom indices of the shared set, in ascending order.
            vertex_indices: Atom indices of the site's vertices.
        """
        super().__init__(
            f"Shared atoms {list(atoms)} cannot be triangulated identically on the site "
            f"with vertices {list(vertex_indices)}")
        self.atoms: tuple[int, ...] = tuple(atoms)


def _surface_triangles(
    vertex_indices: Sequence[int],
    coords: np.ndarray,
    lattice_matrix: np.ndarray,
    shared_atom_sets: Sequence[Collection[int]],
) -> np.ndarray:
    """Surface triangles of a site, triangulating each shared face by its atoms alone.

    Starts from the convex hull of the vertices' Cartesian coordinates.
    Each shared atom set, taken in ascending order of its sorted atoms,
    then fixes the triangles over its atoms:

    - three atoms that are a hull triangle are kept;
    - four or more atoms whose hull triangles cover them (the triangles'
      atoms are the set, and there are two fewer triangles than atoms) are
      re-triangulated as a fan: the atoms are ordered by angle around their
      best-fit plane, starting from the lowest atom index, and each
      consecutive pair forms a triangle with that first atom. The pairs of
      atoms adjacent in that order, the last and first included, must be
      the edges that belong to only one of those triangles.

    Every triangle is then wound so that its normal ``(b - a) x (c - a)``
    in ``coords`` points away from the centre (the mean of ``coords``),
    whatever the handedness of the lattice. The wound triangles must then
    form a consistently oriented surface, each edge traversed once in each
    direction, which they do exactly when the centre is strictly inside
    every triangle's plane; a fan that folds through the centre does not.

    Each shared atom set is taken to be a face of the site. A set that is
    not raises ValueError, unless the hull happens to have it as a face
    anyway, as when two sites interpenetrate; overlapping sites are not
    supported.

    Args:
        vertex_indices: Atom indices of the site's vertices.
        coords: (N_vertices, 3) unwrapped fractional coordinates of the
            vertices, in the order of ``vertex_indices``.
        lattice_matrix: (3, 3) lattice vectors as rows, in Angstrom.
        shared_atom_sets: Atom indices of each face shared with another
            site, every atom being one of ``vertex_indices``.

    Returns:
        (N_triangles, 3) int64 array of local vertex indices (into
        ``vertex_indices``) per triangle.

    Raises:
        _SharedFaceError: If a shared atom set is not a face of the site
            that can be triangulated identically by the rules above.
        RuntimeError: If the wound triangles are not consistently
            oriented, so that they do not enclose the centre.
    """
    local = {atom: i for i, atom in enumerate(vertex_indices)}
    cartesian = coords @ lattice_matrix
    triangles: list[frozenset[int]] = [frozenset(t) for t in ConvexHull(cartesian).simplices.tolist()]
    for atoms in sorted((sorted(s) for s in shared_atom_sets)):
        face = frozenset(local[atom] for atom in atoms)
        replaced, added = _shared_face_triangles(face, triangles, cartesian, vertex_indices)
        if replaced is None:
            raise _SharedFaceError(atoms, vertex_indices)
        triangles = [t for t in triangles if t not in replaced] + added
    wound = _wound_away_from_centre(triangles, coords)
    directed = np.concatenate([wound[:, [0, 1]], wound[:, [1, 2]], wound[:, [2, 0]]])
    if len(np.unique(directed, axis=0)) != len(directed):
        raise RuntimeError("polyhedron is too distorted for its triangles to enclose its centre")
    return wound


def _hull_triangles(coords: np.ndarray) -> np.ndarray:
    """Triangles of the convex hull of a polyhedron's vertices, wound away from its centre.

    The hull is computed from ``coords`` as given. Its faces are the same
    in any coordinates related to them by an invertible linear map, such
    as fractional and Cartesian coordinates, but a flat face of four or
    more vertices may be split into triangles differently.

    Args:
        coords: (N_vertices, 3) coordinates of the vertices.

    Returns:
        (N_triangles, 3) int64 array of vertex indices per triangle, each
        wound so that ``(b - a) x (c - a)`` points away from the centre
        (the mean of ``coords``).
    """
    hull = [frozenset(t) for t in ConvexHull(coords).simplices.tolist()]
    return _wound_away_from_centre(hull, coords)


def _wound_away_from_centre(triangles: Sequence[frozenset[int]], coords: np.ndarray) -> np.ndarray:
    """Triangles as an array, each wound so that its normal points away from the centre.

    Args:
        triangles: Vertex indices of each triangle.
        coords: (N_vertices, 3) coordinates of the vertices.

    Returns:
        (N_triangles, 3) int64 array of vertex indices per triangle, each
        wound so that ``(b - a) x (c - a)`` in ``coords`` points away from
        the centre (the mean of ``coords``).
    """
    ordered = np.array([sorted(t) for t in triangles], dtype=np.int64)
    a, b, c = (coords[ordered[:, k]] for k in range(3))
    normal = np.cross(b - a, c - a)
    inward = np.einsum("ij,ij->i", normal, a - coords.mean(axis=0)) < 0.0
    ordered[inward] = ordered[inward][:, [0, 2, 1]]
    return ordered


def _shared_face_triangles(
    face: frozenset[int],
    triangles: list[frozenset[int]],
    cartesian: np.ndarray,
    vertex_indices: Sequence[int],
) -> tuple[set[frozenset[int]] | None, list[frozenset[int]]]:
    """The triangles a shared face replaces, and those it adds in their place.

    Args:
        face: Local vertex indices of the shared face.
        triangles: The current surface triangles, as sets of local vertex
            indices.
        cartesian: (N_vertices, 3) Cartesian coordinates of the vertices.
        vertex_indices: Atom indices of the site's vertices.

    Returns:
        Tuple of (replaced, added): the triangles to remove, or None if the
        face cannot be triangulated identically, and the triangles to add.
        A face that is already a triangle replaces and adds itself.
    """
    if len(face) == 3:
        return ({face}, [face]) if face in triangles else (None, [])
    inside = [t for t in triangles if t <= face]
    if len(face) < 4 or len(inside) != len(face) - 2 or frozenset().union(*inside) != face:
        return None, []
    ids = np.array(sorted(face))
    centred = cartesian[ids] - cartesian[ids].mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    order = ids[np.argsort(np.arctan2(centred @ vt[1], centred @ vt[0]))].tolist()
    edges = Counter(frozenset(edge) for t in inside for edge in itertools.combinations(t, 2))
    if ({edge for edge, n in edges.items() if n == 1}
            != {frozenset(pair) for pair in zip(order, order[1:] + order[:1])}):
        return None, []
    start = min(range(len(order)), key=lambda k: vertex_indices[order[k]])
    order = order[start:] + order[:start]
    fan = [frozenset((order[0], order[k], order[k + 1])) for k in range(1, len(order) - 1)]
    return set(inside), fan


def _containment_planes(
    triangles: np.ndarray, vertex_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Containment planes of a polyhedron bounded by fixed triangles, in one frame.

    The four inward planes of the tetrahedron formed by the centre (the
    mean of the vertex coordinates) and each surface triangle, in
    fractional coordinates, as containment is invariant under the linear
    map to Cartesian coordinates. Uses the numba kernel when numba is
    installed and numpy otherwise, with identical results.

    Args:
        triangles: (N_triangles, 3) int64 array of vertex indices per
            surface triangle, each wound so that ``(b - a) x (c - a)``
            points away from the centre, as from ``_surface_triangles``.
        vertex_coords: (N_vertices, 3) array of the current vertex
            positions.

    Returns:
        Tuple of ``(image_shifts, normals, ref_points)``, the arguments of
        the query kernels after the point, for ``_contains_point``: the
        (M, 3) integer whole-cell shifts from a point in [0, 1) to its
        images that can lie inside the polyhedron (the eight shifts with
        components in {0, 1} when every vertex coordinate is in [0, 2),
        otherwise those from ``_image_shifts``), the (N_triangles, 4, 3)
        inward plane normals, and (N_triangles, 4, 3) a point on each
        plane.

    Raises:
        RuntimeError: If the centre is not strictly on the inner side of
            every triangle's plane, so that the triangles do not enclose
            it.
        ValueError: If the vertices span more than ``_MAX_IMAGE_SHIFTS``
            cells, counted over the three axes.
    """
    planes = _numba_star_planes if HAS_NUMBA else _numpy_star_planes
    normals, ref_points, star_shaped, within_two_cells = planes(vertex_coords, triangles)
    if not star_shaped:
        raise RuntimeError("polyhedron is too distorted for its triangles to enclose its centre")
    image_shifts = _SHIFTS if within_two_cells else _image_shifts(vertex_coords)
    return image_shifts, normals, ref_points


def _contains_point(x: np.ndarray, planes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """Test whether an image of a point is inside a polyhedron.

    The point is wrapped into [0, 1) and each of its images that can lie
    inside the polyhedron is tested. Uses the numba kernel when numba is
    installed and numpy otherwise, with identical results.

    Args:
        x: Fractional coordinates of the point, three numbers, in any
            periodic image.
        planes: The polyhedron's containment planes for the current
            frame, from ``_containment_planes``.

    Returns:
        True if an image of the point is in the tetrahedron formed by the
        centre and one of the triangles.

    Raises:
        ValueError: If ``x`` is not three finite numbers.
    """
    if x.__class__ is not _NDARRAY or x.dtype is not _FLOAT64:
        x = np.asarray(x, dtype=np.float64)
    query = _numba_star_query if HAS_NUMBA else _numpy_star_query
    return query(x, *planes)
