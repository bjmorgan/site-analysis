# Working with Polyhedral Sites

Polyhedral sites are defined by coordination polyhedra formed by sets of vertex atoms. Unlike spherical sites with fixed geometry or Voronoi sites with mathematically determined boundaries, polyhedral sites adapt their shape based on the positions of coordinating atoms, making them ideal for representing crystallographic coordination environments.

For a conceptual overview of polyhedral sites, see the [polyhedral sites concepts page](../concepts/sites.md#polyhedral-sites).

## When to Use Polyhedral Sites

Polyhedral sites are best suited for:
- Analyzing specific coordination environments (tetrahedral, octahedral, etc.)
- Systems where site geometry should match chemical coordination
- Materials with well-defined coordination polyhedra
- Studies requiring sites that adapt to structural distortions
- Close-packed structures where coordination polyhedra fill space

**Consider alternative site types when**:
- You need guaranteed complete spatial coverage (use Voronoi sites)
- The structure lacks well-defined coordination environments
- Computational efficiency is critical (polyhedral containment tests are expensive)
- Site centres should be fixed rather than geometry-dependent

## Understanding Polyhedral Site Properties

Polyhedral sites have unique characteristics:

- **Vertex-based**: Defined by sets of atoms forming polyhedron vertices
- **Adaptive shape**: Site geometry follows vertex atom movements
- **Coordination-specific**: Directly represents chemical environments
- **Structure-dependent**: Shape changes with structural distortions

The key concept is that sites are defined by their vertices (coordinating atoms) rather than by a centre and radius. This creates sites that match crystallographic coordination environments and adapt to local structure deformations. Each site is bounded by triangles over its vertices, and neighbouring sites triangulate the faces they share identically (see [Site Boundaries](#site-boundaries)).

## Creating Polyhedral Sites

Polyhedral sites use the [reference-based sites workflow](reference_workflow.md) to identify coordination environments.

### Basic Setup

```python
from site_analysis import TrajectoryBuilder

trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_polyhedral_sites(
        centre_species="Li",      # Species at polyhedron centres
        vertex_species="O",       # Species forming vertices
        cutoff=2.5,              # Distance cutoff for coordination
        n_vertices=4,            # Number of vertices (4 = tetrahedral)
        label="tetrahedral"
    )
    .build())
```

### With Structure Alignment

When reference and target structures have different origins or orientations:

```python
trajectory = (TrajectoryBuilder()
    .with_structure(target)
    .with_reference_structure(ideal_structure)
    .with_mobile_species("Li")
    .with_structure_alignment(
        align=True,
        align_species=["O"]      # Align using framework atoms
    )
    .with_site_mapping(mapping_species=["O"])  # Map vertex atoms
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=6,            # Octahedral coordination
        label="octahedral"
    )
    .build())
```

## Example: Mixed Coordination Environments

Many materials contain multiple types of coordination environments:

```python
# Example: Spinel structure with tetrahedral and octahedral sites
trajectory = (TrajectoryBuilder()
    .with_structure(spinel_structure)
    .with_reference_structure(ideal_spinel)
    .with_mobile_species("Li")
    .with_structure_alignment(align=True, align_species=["O"])
    # Tetrahedral sites
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.0,
        n_vertices=4,
        label="tetrahedral"
    )
    # Octahedral sites
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=6,
        label="octahedral"
    )
    .build())
```

## Advanced Usage: Direct Site Creation

For more control, you can create polyhedral sites directly.

### Using ReferenceBasedSites

```python
from site_analysis.reference_workflow import ReferenceBasedSites

# Create reference-based sites instance
rbs = ReferenceBasedSites(
    reference_structure=ideal_structure,
    target_structure=md_structure,
    align=True,
    align_species=["O"]
)

# Create polyhedral sites
sites = rbs.create_polyhedral_sites(
    center_species="Li",
    vertex_species="O",
    cutoff=2.5,
    n_vertices=4,
    label="tetrahedral",
    target_species=["O"]
)

# Create trajectory manually
from site_analysis import atoms_from_structure, Trajectory
atoms = atoms_from_structure(md_structure, "Li")
trajectory = Trajectory(sites=sites, atoms=atoms)
```

### Creating Sites from Vertex Indices

For complete control, specify vertex atoms directly:

```python
from site_analysis.polyhedral_site import PolyhedralSite

# Define vertices for each site (atom indices)
site1_vertices = [10, 15, 22, 28]  # Indices of O atoms forming tetrahedron
site2_vertices = [11, 16, 23, 29, 35, 41]  # Indices for octahedron

# Create individual polyhedral sites
site1 = PolyhedralSite(vertex_indices=site1_vertices, label="tet_1")
site2 = PolyhedralSite(vertex_indices=site2_vertices, label="oct_1")

# Create from lists of vertex indices
vertex_lists = [[10, 15, 22, 28], [11, 16, 23, 29, 35, 41]]
sites = PolyhedralSite.sites_from_vertex_indices(
    vertex_indices=vertex_lists,
    label="polyhedral"
)

# Use in trajectory
trajectory = Trajectory(sites=sites, atoms=atoms)
```

This approach is useful when you have pre-identified coordination environments or need to work with non-standard polyhedra.

Vertex indices are Python or numpy integers; `PolyhedralSite` raises `TypeError` for any other value, including a `bool`. Every structure the trajectory analyses must have an atom for every vertex index: `analyse_structure()` (and so `append_timestep()`) and `TrajectoryBuilder.build()` raise `ValueError` ("Site {index} has vertex index {k}, but the structure has {n} atoms") naming the first such site in index order. `PolyhedralSiteCollection.validate_atom_count(n_atoms)` makes the same check.

For a comparison of all site types and guidance on choosing between them, see the [site selection guide](../concepts/sites.md#selecting-the-right-site-type).

## Site Boundaries

A polyhedral site is the polyhedron bounded by a fixed set of triangles over its vertex atoms. Each face a site shares with a neighbouring site (see [Shared Faces and Adjacency](#shared-faces-and-adjacency)) is triangulated by a rule that depends only on the face's atoms, so both sites give the face the same triangles, and the faces it shares with no other site keep the triangles of the convex hull of its vertices. Sites that share faces therefore fill the space between them, with no gaps or overlaps at their shared faces, including faces of four or more atoms that are not flat.

The triangles are fixed, as triples of vertex atoms, when the shared faces are found, from the same vertex coordinates: by `TrajectoryBuilder.build()` from the reference geometry, by `trajectory.find_shared_faces(structure)`, or from the first structure analysed if neither has found them. They are kept for every later frame, while the vertex atoms move. A `PolyhedralSite` used on its own, outside a collection that has found its shared faces, takes the triangles of the convex hull of its vertices in the first frame it is tested in, unless they are set with `PolyhedralSite.set_triangles()`.

A point is inside a site when it is inside one of the tetrahedra joining the site's centre (the mean of its vertex coordinates) to each of its triangles. These tetrahedra bound the site exactly while the centre is strictly on the inner side of the plane of every triangle. When a site is tested in a frame in which it is not, containment raises `RuntimeError`, naming the site and its vertex atoms and saying that it is too distorted for its triangles to enclose its centre. A site is checked on its first containment test after its vertex coordinates change, so a site that no atom's search reaches in a frame does not raise in that frame.

Polyhedral sites must not overlap. When the shared faces are found, every pair of sites, and every site with its own periodic images, is tested for overlap, whatever the number of vertex atoms they share: two sites overlap when a tetrahedron of one and a tetrahedron of the other intersect with positive volume (beyond a margin for floating-point rounding), so sites that touch at a face, an edge or a vertex do not overlap. Overlapping, coincident and nested sites raise `ValueError` (see below).

## Shared Faces and Adjacency

Two polyhedral sites share a face when they have three or more vertex atoms in common under one lattice translation: the same atoms, translated by one lattice vector, are vertices of both. These atoms are triangulated identically on both sites, by the rules described below; finding the shared faces raises `ValueError` when they cannot be, or when the two sites overlap. A shared face is a `Face` with `site_indices`, the indices of the two sites, and `vertex_indices`, the indices of the atoms, both in ascending order. Creating a `Face` stores its indices as `int` and raises `TypeError` for an index that is not a Python or numpy integer, including a `bool`.

No tolerance is involved. The faces depend on which periodic image of each vertex atom is nearest its site's reference centre (or, for a site without a reference centre, on the spread-based unwrapping of its vertices), so a structure distorted enough to change that image changes the faces.

In a cell two sites wide, two neighbouring sites share two faces, one on each side.

When every site is polyhedral and every vertex atom of every site was mapped from the reference structure by a `with_polyhedral_sites` definition, `build()` finds the shared faces from the reference geometry (aligned when alignment is on), mapped to the target structure's atom indices. Otherwise (for example if the sites are created directly with `Trajectory(sites=..., atoms=...)`, or the builder also adds sites with `with_existing_sites` with vertex atoms that no `with_polyhedral_sites` definition mapped), `build()` does not find the faces. They are then found from the first structure analysed (by `analyse_structure()`, and so `append_timestep()` and `trajectory_from_structures()`, or by `PolyhedralSiteCollection.sites_contain_points()`), or by calling `trajectory.find_shared_faces(structure)` first. `find_shared_faces` can also be called to recompute the faces, and the triangles, from another structure, which replaces those found before. The structure must have its atoms in the same order as the structures the trajectory analyses, because the sites' vertex indices refer to that order. The faces and triangles are kept by `reset()`.

Finding the shared faces, wherever it happens, raises `ValueError`, keeping the faces and triangles found before, for two sites that coincide (they have the same vertex atoms at the same positions), for a site that lies inside another (all its vertex atoms are vertices of the other site under one lattice translation), for two sites that overlap, or a site that overlaps a periodic image of itself (see [Site Boundaries](#site-boundaries)), for shared atoms that cannot be triangulated identically on both sites, for a vertex index not less than the number of atoms in the structure, and for non-finite vertex coordinates. Shared atoms are triangulated identically when they are a whole face of each site, or three atoms that are half of a four-atom face; shared atoms that are part of a larger flat face may raise ("cannot be triangulated identically"). It raises `RuntimeError` for a site whose vertices are degenerate (for example, all in one plane). `shared_faces()` and `site_adjacency()` raise `RuntimeError` before the shared faces are found. `find_shared_faces`, `shared_faces()` and `site_adjacency()` raise `TypeError` for sites that are not polyhedral.

`shared_faces()` returns every shared face as a tuple of `Face` objects sorted by `site_indices` and then `vertex_indices`:

```python
from collections import Counter

faces = trajectory.shared_faces()
faces_of_site_7 = [face for face in faces if 7 in face.site_indices]

species = [atom.species_string for atom in structure]
faces_by_species = Counter(
    tuple(sorted(species[i] for i in face.vertex_indices)) for face in faces)
```

`site_adjacency()` gives which sites share a face, as a symmetric boolean `scipy.sparse.csr_array` indexed by `site.index`, with shape `(n, n)` where `n` is one more than the largest site index. It can be joined with the transition events to find whether each transition was between sites that share a face:

```python
adjacency = trajectory.site_adjacency()

adjacency[3, 7]                       # True if sites 3 and 7 share a face
neighbours = adjacency[[7]].indices   # indices of the sites sharing a face with site 7

# Whether each transition was between sites that share a face
events = trajectory.transition_events()
between_neighbours = adjacency[events.from_site[:, None],
                               events.to_site[:, None]].toarray().ravel()
not_between_neighbours = events[~between_neighbours]
```

Indexing with column vectors (`[:, None]`) gives a sparse array whether or not there are any transitions, so `.toarray().ravel()` always gives a boolean array with one element per transition.

Transitions record the sites an atom left and entered, not faces, so in a cell two sites wide a transition between neighbouring sites cannot be matched to one of their two faces.

## Containment

A site's containment test uses the four planes of each of its tetrahedra (see [Site Boundaries](#site-boundaries)), recomputed from the current vertex coordinates on the site's first test after they change. A point may be given in any periodic image: each of its images in the cells that the site's vertices span is tested. A point exactly on a face shared by two sites may be counted in both, since each site's copy of a shared atom can differ in the last bit, and goes to the first of them that the [site search](../concepts/site_collections.md#priority-based-assignment-spherical-and-polyhedral) tests.

The test is the same with and without [numba](https://numba.pydata.org/), with identical results. When numba is installed, it runs as JIT-compiled code, which makes per-frame analysis of the argyrodite tutorial trajectory about three times faster:

```bash
pip install site-analysis[fast]  # installs numba
```
