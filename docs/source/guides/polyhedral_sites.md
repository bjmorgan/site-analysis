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

The key concept is that sites are defined by their vertices (coordinating atoms) rather than by a centre and radius. This creates sites that match crystallographic coordination environments and adapt to local structure deformations.

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

## Shared Faces and Adjacency

Two polyhedral sites share a face when they have three or more vertex atoms in common under one lattice translation: the same atoms, translated by one lattice vector, are vertices of both. A shared face is a `Face` with `site_indices`, the indices of the two sites, and `vertex_indices`, the indices of the atoms, both in ascending order. Creating a `Face` stores its indices as `int` and raises `TypeError` for an index that is not a Python or numpy integer, including a `bool`.

No tolerance is involved. The faces depend on which periodic image of each vertex atom is nearest its site's reference centre (or, for a site without a reference centre, on the spread-based unwrapping of its vertices), so a structure distorted enough to change that image changes the faces.

In a cell two sites wide, two neighbouring sites share two faces, one on each side.

When every site is polyhedral and every vertex atom of every site is a vertex of a site created by `with_polyhedral_sites`, `build()` finds the shared faces from the reference geometry (aligned when alignment is on), mapped to the target structure's atom indices. Otherwise (for example if the sites are created directly with `Trajectory(sites=..., atoms=...)`, or the builder also adds sites with `with_existing_sites` that have other vertex atoms), `build()` does not find the faces, and `trajectory.find_shared_faces(structure)` must be called. It can also be called to recompute the faces from another structure, which replaces the faces found before. The structure must have its atoms in the same order as the structures the trajectory analyses, because the sites' vertex indices refer to that order.

Polyhedral sites may overlap partially.

`find_shared_faces` raises `ValueError`, keeping the faces found before, for two sites that coincide (they have the same vertex atoms at the same positions), for a site that lies inside another (all its vertex atoms are vertices of the other site under one lattice translation), for a vertex index not less than the number of atoms in the structure, and for non-finite vertex coordinates. `shared_faces()` and `site_adjacency()` raise `RuntimeError` before the shared faces are found. `find_shared_faces`, `shared_faces()` and `site_adjacency()` raise `TypeError` for sites that are not polyhedral.

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

## Containment Algorithms

The best available algorithm is selected automatically. When [numba](https://numba.pydata.org/) is installed, a JIT-compiled surface normal method with cached hull triangles is used. Otherwise, the package falls back to Delaunay tessellation via scipy. Installing numba typically gives a ~7x speedup for polyhedral site analysis:

```bash
pip install site-analysis[fast]  # installs numba
```
