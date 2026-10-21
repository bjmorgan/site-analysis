"""Fixed surface triangles of a polyhedral site, and containment within them.

The surface of a polyhedral site is a fixed set of triangles over its
vertices: the convex hull of its vertices, with each face it shares with a
neighbouring site triangulated by a rule that depends only on the shared
atoms, so that both sites give the face the same triangles. A point is
inside the site when it is inside one of the tetrahedra formed by the
site's centre (the mean of its vertex coordinates) and a surface triangle,
which bounds the site exactly while it is star-shaped about its centre.
"""

from __future__ import annotations

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
        """
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


# The most cells a polyhedral site's vertices may span, counted over all
# three axes, for the site to test the images of a point in each of them.
_MAX_IMAGE_SHIFTS = 1000


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
    ``-(b - a) x (c - a)`` through ``a``. A query tests them in this
    order, so the edge planes, which reject most points outside a
    tetrahedron, come first. When the centre is strictly on the inner side of every
    triangle's plane, every normal points into its tetrahedron.

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
    a = vertex_coords[triangles[:, 0]]
    u = vertex_coords[triangles[:, 1]] - a
    v = vertex_coords[triangles[:, 2]] - a
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
    for q in range(3):
        p = vertex_coords[triangles[:, q]] - centre
        r = vertex_coords[triangles[:, (q + 1) % 3]] - centre
        normals[:, q, 0] = p[:, 1] * r[:, 2] - p[:, 2] * r[:, 1]
        normals[:, q, 1] = p[:, 2] * r[:, 0] - p[:, 0] * r[:, 2]
        normals[:, q, 2] = p[:, 0] * r[:, 1] - p[:, 1] * r[:, 0]
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
    """
    images = (x - np.floor(x)) + image_shifts
    d = images[:, np.newaxis, np.newaxis, :] - ref_points
    dots = d[..., 0] * normals[..., 0] + d[..., 1] * normals[..., 1] + d[..., 2] * normals[..., 2]
    return bool(np.any(np.all(dots >= 0.0, axis=2)))


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
    - three atoms ``{a, b, c}`` that are half of a four-atom face
      ``{a, b, c, d}`` the hull split along the other diagonal replace the
      hull's two triangles there with ``{a, b, c}`` and the leftover
      triangle ``{a, c, d}``, where ``d`` is the vertex nearest the plane
      of ``{a, b, c}`` and exactly two hull triangles have all their atoms
      in ``{a, b, c, d}``;
    - four or more atoms whose hull triangles cover them (the triangles'
      atoms are the set, and there are two fewer triangles than atoms) are
      re-triangulated as a fan: the atoms are ordered by angle around their
      best-fit plane, starting from the lowest atom index, and each
      consecutive pair forms a triangle with that first atom.

    Every triangle is then wound so that its normal ``(b - a) x (c - a)``
    in ``coords`` points away from the centre (the mean of ``coords``),
    whatever the handedness of the lattice.

    Each shared atom set is taken to be a face of the site: a set that is
    not either raises ValueError here or makes the sites sharing it
    overlap, which ``_first_overlap`` detects.

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
        ValueError: If a shared atom set is not a face of the site that
            can be triangulated identically by the rules above, or if
            fixing its triangles would remove a triangle of an earlier
            shared atom set.
    """
    local = {atom: i for i, atom in enumerate(vertex_indices)}
    cartesian = coords @ lattice_matrix
    triangles: list[frozenset[int]] = [frozenset(t) for t in ConvexHull(cartesian).simplices.tolist()]
    fixed: set[frozenset[int]] = set()
    for atoms in sorted((sorted(s) for s in shared_atom_sets)):
        face = frozenset(local[atom] for atom in atoms)
        replaced, added = _shared_face_triangles(face, triangles, cartesian, vertex_indices)
        if replaced is None or (replaced - set(added)) & fixed:
            raise ValueError(
                f"Shared atoms {atoms} cannot be triangulated identically on the site "
                f"with vertices {list(vertex_indices)}")
        triangles = [t for t in triangles if t not in replaced] + added
        fixed.update(t for t in added if t <= face)
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
        if face in triangles:
            return {face}, [face]
        a, b, c = (cartesian[i] for i in sorted(face))
        others = np.array([i for i in range(len(cartesian)) if i not in face])
        distances = np.abs((cartesian[others] - a) @ np.cross(b - a, c - a))
        d = int(others[np.argmin(distances)])
        quad = [t for t in triangles if t <= face | {d}]
        if len(quad) != 2:
            return None, []
        first, second = quad
        leftover = (face - (first & second)) | {d}
        return {first, second}, [face, leftover]
    inside = [t for t in triangles if t <= face]
    if len(face) < 4 or len(inside) != len(face) - 2 or frozenset().union(*inside) != face:
        return None, []
    ids = np.array(sorted(face))
    centred = cartesian[ids] - cartesian[ids].mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    order = ids[np.argsort(np.arctan2(centred @ vt[1], centred @ vt[0]))].tolist()
    start = min(range(len(order)), key=lambda k: vertex_indices[order[k]])
    order = order[start:] + order[:start]
    fan = [frozenset((order[0], order[k], order[k + 1])) for k in range(1, len(order) - 1)]
    return set(inside), fan


# Overlap of two projections, as a fraction of the larger projected
# extent, at or below which an axis separates two tetrahedra.
_SEPARATION_MARGIN = 1e-9
# Length of the cross product of two edges, as a fraction of the product
# of their lengths, at or below which the edges are parallel and give no
# separating axis.
_PARALLEL_EDGES = 1e-12
# The most tetrahedron pairs, before bounding-box filtering, tested at once.
_TETRAHEDRON_PAIR_CHUNK = 16384
# The most pairs of sites whose bounding boxes are compared at once.
_BOX_PAIR_BLOCK = 1 << 20
# Vertex indices of the faces and of the edges of a tetrahedron.
_TETRAHEDRON_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
_TETRAHEDRON_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def _first_overlap(
    vertex_coords: Sequence[np.ndarray],
    triangles: Sequence[np.ndarray],
) -> tuple[int, int] | None:
    """The first pair of polyhedral sites that overlap with positive volume.

    Each site is the union of the tetrahedra formed by its centre (the
    mean of its vertex coordinates) and each of its surface triangles.
    Every pair of sites ``(i, j)`` with ``i <= j``, and every integer
    lattice translation ``t`` for which the bounding boxes of site ``i``
    and of site ``j`` translated by ``t`` meet, is tested, except a site
    with itself untranslated. Two tetrahedra overlap when their
    projections onto every candidate separating axis (the face normals of
    both, and the cross products of an edge of each, skipping those of
    parallel edges) overlap by more than ``_SEPARATION_MARGIN`` times the
    larger of the two projected extents, so tetrahedra that only touch at
    a face, edge or vertex do not overlap. Works in fractional
    coordinates, as positive-volume intersection is invariant under the
    linear map to Cartesian coordinates. The bounding boxes are compared
    in blocks of rows ``i`` against every ``j``, so that the memory needed
    grows with the number of sites and with the number of pairs whose
    boxes meet.

    Args:
        vertex_coords: Unwrapped fractional coordinates of each site's
            vertices, each of shape ``(N_vertices, 3)``.
        triangles: Each site's ``(N_triangles, 3)`` array of local vertex
            indices per surface triangle, as from ``_surface_triangles``.

    Returns:
        The first overlapping pair ``(i, j)`` in ascending order of ``i``,
        then ``j``, where ``i == j`` for a site that overlaps a periodic
        image of itself, or None if no sites overlap.
    """
    if not vertex_coords:
        return None
    tetrahedra = np.concatenate([_star_tetrahedra(c, t)
                                 for c, t in zip(vertex_coords, triangles, strict=True)])
    counts = np.array([len(t) for t in triangles], dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    first, second, shifts = _bounding_box_translations(
        np.array([c.min(axis=0) for c in vertex_coords]),
        np.array([c.max(axis=0) for c in vertex_coords]))
    tetrahedron_lower = tetrahedra.min(axis=1)
    tetrahedron_upper = tetrahedra.max(axis=1)
    pair_counts = counts[first] * counts[second]
    cumulative = np.concatenate([[0], np.cumsum(pair_counts)])
    start = 0
    while start < len(first):
        stop = max(start + 1, int(np.searchsorted(
            cumulative, cumulative[start] + _TETRAHEDRON_PAIR_CHUNK, side="right")) - 1)
        rows = np.repeat(np.arange(start, stop), pair_counts[start:stop])
        k = np.arange(len(rows)) - (cumulative[rows] - cumulative[start])
        a = offsets[first[rows]] + k // counts[second[rows]]
        b = offsets[second[rows]] + k % counts[second[rows]]
        shift = shifts[rows]
        meet = np.all((tetrahedron_lower[a] <= tetrahedron_upper[b] + shift)
                      & (tetrahedron_lower[b] + shift <= tetrahedron_upper[a]), axis=1)
        overlapping = _tetrahedra_overlap(
            tetrahedra[a[meet]], tetrahedra[b[meet]] + shift[meet][:, np.newaxis, :])
        if overlapping.any():
            row = rows[meet][np.argmax(overlapping)]
            return int(first[row]), int(second[row])
        start = stop
    return None


def _star_tetrahedra(vertex_coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """The tetrahedra formed by a polyhedron's centre and each of its surface triangles.

    Args:
        vertex_coords: (N_vertices, 3) array of vertex positions.
        triangles: (N_triangles, 3) array of vertex indices per surface
            triangle.

    Returns:
        (N_triangles, 4, 3) array of the vertices of each tetrahedron: the
        centre (the mean of the vertex coordinates), then the triangle's
        three vertices.
    """
    centre = np.broadcast_to(vertex_coords.mean(axis=0), (len(triangles), 1, 3))
    return np.concatenate([centre, vertex_coords[triangles]], axis=1)


def _bounding_box_translations(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs of boxes, and the lattice translations of the second, for which the boxes meet.

    The box of ``i`` meets that of ``j`` translated by the integer vector
    ``t`` when ``ceil(lower_i - upper_j) <= t <= floor(upper_i - lower_j)``
    along each axis.

    Compares blocks of ``max(1, _BOX_PAIR_BLOCK // N)`` rows ``i`` at a
    time against every ``j >= i``.

    Args:
        lower: (N, 3) array of the lowest coordinates of each box, N >= 1.
        upper: (N, 3) array of the highest coordinates of each box.

    Returns:
        Tuple of (first, second, shifts): the indices ``i <= j`` of each
        pair, in ascending order of ``i`` then ``j``, and the (M, 3)
        integer translations of the second box, excluding the zero
        translation of a box with itself.
    """
    n = len(lower)
    block = max(1, _BOX_PAIR_BLOCK // n)
    first_blocks, second_blocks = [], []
    for start in range(0, n, block):
        rows = np.arange(start, min(start + block, n))
        meet = rows[:, np.newaxis] <= np.arange(start, n)
        for axis in range(3):
            meet &= (np.floor(upper[rows, np.newaxis, axis] - lower[np.newaxis, start:, axis])
                     >= np.ceil(lower[rows, np.newaxis, axis] - upper[np.newaxis, start:, axis]))
        block_first, block_second = np.nonzero(meet)
        first_blocks.append(block_first + start)
        second_blocks.append(block_second + start)
    first = np.concatenate(first_blocks)
    second = np.concatenate(second_blocks)
    low = np.ceil(lower[first] - upper[second]).astype(np.int64)
    sizes = np.floor(upper[first] - lower[second]).astype(np.int64) - low + 1
    per_pair = np.prod(sizes, axis=1)
    pairs = np.repeat(np.arange(len(first)), per_pair)
    k = np.arange(len(pairs)) - np.repeat(np.cumsum(per_pair) - per_pair, per_pair)
    ny, nz = sizes[pairs, 1], sizes[pairs, 2]
    shifts = low[pairs] + np.stack([k // (ny * nz), (k // nz) % ny, k % nz], axis=1)
    keep = (first[pairs] != second[pairs]) | np.any(shifts != 0, axis=1)
    return first[pairs][keep], second[pairs][keep], shifts[keep]


def _tetrahedra_overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Whether each pair of tetrahedra intersects with positive volume.

    By the separating axis theorem, two tetrahedra are disjoint exactly
    when their projections are disjoint along a face normal of either or
    along the cross product of an edge of each. An axis separates them
    here when their projections overlap by no more than
    ``_SEPARATION_MARGIN`` times the larger projected extent; the cross
    products of edges whose cross product is no longer than
    ``_PARALLEL_EDGES`` times the product of their lengths are skipped.

    Args:
        first: (P, 4, 3) array of the vertices of the first tetrahedron of
            each pair.
        second: (P, 4, 3) array of the vertices of the second tetrahedron
            of each pair.

    Returns:
        (P,) boolean array, True where no axis separates the pair.
    """
    first_vertices, second_vertices = (np.ascontiguousarray(np.moveaxis(t, 2, 0))
                                       for t in (first, second))
    face_normals = [
        _cross(t[:, :, _TETRAHEDRON_FACES[:, 1]] - t[:, :, _TETRAHEDRON_FACES[:, 0]],
               t[:, :, _TETRAHEDRON_FACES[:, 2]] - t[:, :, _TETRAHEDRON_FACES[:, 0]])
        for t in (first_vertices, second_vertices)]
    first_edges, second_edges = (t[:, :, _TETRAHEDRON_EDGES[:, 1]] - t[:, :, _TETRAHEDRON_EDGES[:, 0]]
                                 for t in (first_vertices, second_vertices))
    edge_normals = _cross(first_edges[:, :, :, np.newaxis], second_edges[:, :, np.newaxis, :])
    parallel = (np.sum(edge_normals ** 2, axis=0)
                <= _PARALLEL_EDGES ** 2 * np.sum(first_edges ** 2, axis=0)[:, :, np.newaxis]
                * np.sum(second_edges ** 2, axis=0)[:, np.newaxis, :])
    axes = np.concatenate([*face_normals, edge_normals.reshape(3, -1, 36)], axis=2)
    first_lower, first_upper = _projection_bounds(axes, first_vertices)
    second_lower, second_upper = _projection_bounds(axes, second_vertices)
    overlap = np.minimum(first_upper, second_upper) - np.maximum(first_lower, second_lower)
    extent = np.maximum(first_upper - first_lower, second_upper - second_lower)
    separated = overlap <= _SEPARATION_MARGIN * extent
    separated[:, 8:] &= ~parallel.reshape(-1, 36)
    return ~np.any(separated, axis=1)


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cross products of vectors stored with their components along the first axis.

    Args:
        u: (3, ...) array of vectors.
        v: (3, ...) array of vectors, broadcastable against ``u``.

    Returns:
        (3, ...) array of the cross products ``u x v``.
    """
    return np.stack([u[1] * v[2] - u[2] * v[1],
                     u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]])


def _projection_bounds(axes: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest projections of each tetrahedron's vertices onto its axes.

    Args:
        axes: (3, P, A) array of the axes of each of P tetrahedra, with
            their components along the first axis.
        vertices: (3, P, 4) array of the vertices of each tetrahedron,
            with their components along the first axis.

    Returns:
        Tuple of (lower, upper), (P, A) arrays of the lowest and highest
        projections.
    """
    projections = np.stack([
        axes[0] * vertices[0, :, v, np.newaxis] + axes[1] * vertices[1, :, v, np.newaxis]
        + axes[2] * vertices[2, :, v, np.newaxis] for v in range(4)])
    return projections.min(axis=0), projections.max(axis=0)


class _SurfaceCache:
    """Fixed surface triangles of a polyhedron and its per-frame containment planes.

    The triangles are fixed when the cache is created. Each update
    recomputes, from the current vertex coordinates, the four inward
    planes of the tetrahedron formed by the centre (the mean of the
    vertex coordinates) and each triangle, and the image shifts. Works in
    fractional coordinates, as containment is invariant under the linear
    map to Cartesian coordinates. Uses numba kernels when numba is
    installed and numpy otherwise, with identical results.

    Attributes:
        triangles: (N_triangles, 3) array of vertex indices per surface
            triangle, each wound so that ``(b - a) x (c - a)`` points away
            from the centre.
        image_shifts: (M, 3) integer array of whole-cell shifts from a
            point in [0, 1) to its images that can lie inside the
            polyhedron: the eight shifts with components in {0, 1} when
            every vertex coordinate is in [0, 2), otherwise those from
            ``_image_shifts``.
    """

    def __init__(self, triangles: np.ndarray, vertex_coords: np.ndarray) -> None:
        """Fix the triangles and compute the planes for the initial coordinates.

        Args:
            triangles: (N_triangles, 3) int64 array of vertex indices per
                surface triangle, wound as from ``_surface_triangles``.
            vertex_coords: (N_vertices, 3) array of vertex positions.

        Raises:
            RuntimeError: If the centre is not strictly on the inner side
                of every triangle's plane.
            ValueError: If the vertices span more than ``_MAX_IMAGE_SHIFTS``
                cells, from ``_image_shifts``.
        """
        self.triangles: np.ndarray = triangles
        self._normals: np.ndarray
        self._ref_points: np.ndarray
        self.image_shifts: np.ndarray
        self.update(vertex_coords)

    def update(self, vertex_coords: np.ndarray) -> None:
        """Recompute the containment planes and image shifts from new coordinates.

        Args:
            vertex_coords: (N_vertices, 3) array of current vertex positions.

        Raises:
            RuntimeError: If the centre is not strictly on the inner side
                of every triangle's plane, with the triangles wound as
                fixed, so that the triangles do not enclose it.
            ValueError: If the vertices span more than ``_MAX_IMAGE_SHIFTS``
                cells, from ``_image_shifts``.
        """
        planes = _numba_star_planes if HAS_NUMBA else _numpy_star_planes
        normals, ref_points, star_shaped, within_two_cells = planes(vertex_coords, self.triangles)
        if not star_shaped:
            raise RuntimeError(
                "polyhedron is too distorted for its triangles to enclose its centre")
        self.image_shifts = _SHIFTS if within_two_cells else _image_shifts(vertex_coords)
        self._normals = normals
        self._ref_points = ref_points

    def contains_point(self, x: np.ndarray) -> bool:
        """Test whether an image of a point is inside the polyhedron.

        The point is wrapped into [0, 1) and each of its images given by
        ``image_shifts`` is tested.

        Args:
            x: (3,) fractional coordinates of the point, in any periodic
                image.

        Returns:
            True if an image of the point is in the tetrahedron formed by
            the centre and one of the triangles.
        """
        query = _numba_star_query if HAS_NUMBA else _numpy_star_query
        return bool(query(x, self.image_shifts, self._normals, self._ref_points))
