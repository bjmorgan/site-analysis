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

For a comparison of all site types and guidance on choosing between them, see the [site selection guide](../concepts/sites.md#selecting-the-right-site-type).

## Site Adjacency

`trajectory.site_adjacency()` gives which polyhedral sites share a face (three or more vertex atoms), as a symmetric boolean `scipy.sparse.csr_array` indexed by `site.index`:

```python
adjacency = trajectory.site_adjacency()

adjacency[3, 7]                       # True if sites 3 and 7 share a face
neighbours = adjacency[[7]].indices   # indices of the sites sharing a face with site 7

# Whether each transition was between face-sharing sites
events = trajectory.transition_events()
between_neighbours = adjacency[events.from_site[:, None],
                               events.to_site[:, None]].toarray().ravel()
not_between_neighbours = events[~between_neighbours]
```

Indexing with column vectors (`[:, None]`) gives a sparse matrix whether or not there are any transitions, so `.toarray().ravel()` always gives a boolean array with one element per transition. (Indexing with the flat arrays, `adjacency[events.from_site, events.to_site]`, gives a NumPy array when there are transitions but a sparse array when there are none.)

Sites that share only an edge (two vertex atoms) are not adjacent: an atom moving between them must cross another region, so a transition recorded directly between them means the frames in between did not capture that region. To find the vertex atoms two sites share, for example to label a link by the species on its shared face, use their vertex indices:

```python
shared = (set(trajectory.site_by_index(3).vertex_indices)
          & set(trajectory.site_by_index(7).vertex_indices))
```

The matrix can be used directly with `scipy.sparse.csgraph`, for example to find connected networks of sites.

## Containment Algorithms

The best available algorithm is selected automatically. When [numba](https://numba.pydata.org/) is installed, a JIT-compiled surface normal method with cached face topology is used. Otherwise, the package falls back to Delaunay tessellation via scipy. Installing numba typically gives a ~7x speedup for polyhedral site analysis:

```bash
pip install site-analysis[fast]  # installs numba
```
