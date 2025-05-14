# Working with Dynamic Voronoi Sites

Dynamic Voronoi sites extend the concept of [regular Voronoi sites](voronoi_sites.md) by calculating site centres dynamically based on the positions of reference atoms. While regular Voronoi sites have fixed centres, dynamic Voronoi sites adapt their centres to structural changes, making them ideal for systems with flexible frameworks or thermal distortions.

For a conceptual overview of dynamic Voronoi sites, see the [dynamic Voronoi sites concepts page](../concepts/sites.md#dynamic-voronoi-sites).

## When to Use Dynamic Voronoi Sites

Dynamic Voronoi sites are best suited for:
- Systems with flexible or deformable frameworks
- Structures that undergo significant thermal motion
- Materials where site positions should follow framework distortions
- Analysis where sites need to adapt to local structural changes
- When you want the benefits of Voronoi partitioning with adaptive site positions

**Consider alternative site types when**:
- The framework is rigid (use regular Voronoi sites)
- You need sites tied to specific coordination polyhedra (use polyhedral sites)
- Site centres should remain at fixed crystallographic positions
- Computational efficiency is paramount (regular Voronoi sites are faster)

## Understanding Dynamic Centres

The key difference from regular Voronoi sites is how centres are determined:

1. **Reference atoms**: Each site is defined by a set of reference atoms (typically framework atoms)
2. **Dynamic calculation**: At each timestep, the site centre is calculated as the mean position of its reference atoms
3. **Adaptive geometry**: As the framework distorts, site centres move accordingly

This means the same spatial partitioning logic applies as for regular Voronoi sites, but the partition adapts to structural changes.

## Creating Dynamic Voronoi Sites

Dynamic Voronoi sites require both a structure and a reference structure, using the [reference-based sites workflow](reference_workflow.md).

### Basic Setup

```python
from site_analysis import TrajectoryBuilder

trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_dynamic_voronoi_sites(
        centre_species="Li",          # Species defining site centres
        reference_species="O",        # Species used as reference atoms
        cutoff=3.0,                  # Distance cutoff
        n_reference=6,               # Number of reference atoms per site
        label="dynamic_octahedral"
    )
    .build())
```

### Parameters Explained

- **centre_species**: The species at the centre of coordination environments in the reference structure
- **reference_species**: The species that will be used as reference atoms to calculate dynamic centres
- **cutoff**: Distance within which to search for reference atoms
- **n_reference**: Number of reference atoms required for each site
- **label**: Optional label for all sites

### Reference Structure Requirements

Dynamic Voronoi sites use the reference-based workflow, which means:

1. **Reference structure**: Should contain well-defined coordination environments
2. **Target structure**: Where the sites will be created and used for analysis
3. **Structure alignment**: The builder can automatically align structures if needed
4. **Species mapping**: The builder identifies corresponding atoms between structures

```python
trajectory = (TrajectoryBuilder()
    .with_structure(target)
    .with_reference_structure(reference)
    .with_mobile_species("Li")
    .with_structure_alignment(
        align=True,
        align_species=["O", "P"],  # Align on framework
        align_metric='rmsd'
    )
    .with_dynamic_voronoi_sites(
        centre_species="Li",
        reference_species="O",
        cutoff=2.5,
        n_reference=6
    )
    .build())
```

## Example: Li Sites in Li₃OCl Antiperovskite

Li₃OCl is a lithium superionic conductor with an antiperovskite structure. The O/Cl host framework undergoes thermal vibrations and distortions during MD simulations. Dynamic Voronoi sites use these framework atom positions to define Li site centres that adapt to the instantaneous structure:

```python
# Dynamic sites for Li diffusion in Li3OCl
trajectory = (TrajectoryBuilder()
    .with_structure(md_structure)
    .with_reference_structure(ideal_li3ocl)
    .with_mobile_species("Li")
    .with_structure_alignment(align=True, align_species=["O", "Cl"])
    .with_dynamic_voronoi_sites(
        centre_species="Li",         # Li-centred sites
        reference_species=["O", "Cl"],  # Framework atoms as references
        cutoff=3.0,
        n_reference=4,              # Coordination to O/Cl atoms
        label="Li_site"
    )
    .build())
```

In this example:
- Sites are initially defined at Li positions from the reference structure
- The site centres are dynamically calculated based on surrounding O/Cl framework atoms
- As the framework vibrates and distorts, the Li site centres adjust accordingly
- This captures how the local environment for Li migration changes with framework dynamics

This approach is particularly useful for superionic conductors where mobile ion sites are coupled to framework dynamics.

## Advanced Usage: Direct Site Creation

For more control over site creation, you can bypass the builder and create dynamic Voronoi sites directly.

### Using the ReferenceBasedSites Workflow

```python
from site_analysis.reference_workflow import ReferenceBasedSites

# Create the reference-based sites instance
rbs = ReferenceBasedSites(
    reference_structure=ideal_structure,
    target_structure=md_structure,
    align=True,
    align_species=["O", "Cl"],
    align_metric='rmsd'
)

# Create dynamic Voronoi sites
sites = rbs.create_dynamic_voronoi_sites(
    center_species="Li",
    reference_species=["O", "Cl"],
    cutoff=3.0,
    n_reference=4,
    label="Li_site",
    target_species=["O", "Cl"]  # Species to map in target
)

# Create the collection and atoms
from site_analysis import atoms_from_structure
atoms = atoms_from_structure(md_structure, "Li")

# Create trajectory with custom sites
from site_analysis import Trajectory
trajectory = Trajectory(sites=sites, atoms=atoms)
```

### Creating Sites from Reference Indices

For complete control, create sites by specifying reference atom indices directly:

```python
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection

# Define reference atoms for each site (by index)
site1_refs = [0, 5, 12, 18]  # O/Cl atoms around first Li site
site2_refs = [1, 6, 13, 19]  # O/Cl atoms around second Li site

# Create individual sites
site1 = DynamicVoronoiSite(reference_indices=site1_refs, label="Li_site_1")
site2 = DynamicVoronoiSite(reference_indices=site2_refs, label="Li_site_2")

# Or create multiple sites at once
reference_indices_list = [site1_refs, site2_refs]
sites = DynamicVoronoiSite.sites_from_reference_indices(
    reference_indices_list=reference_indices_list,
    label="Li_site"
)

# Create collection
site_collection = DynamicVoronoiSiteCollection(sites)

# Use in analysis
site_collection.analyse_structure(atoms, structure)
```

This approach gives you full control over:
- Which specific atoms define each site
- How sites are created and configured
- The workflow for structure analysis

## Comparison with Other Site Types

### Advantages over Regular Voronoi Sites
- Centres adapt to structural changes
- Better for flexible frameworks
- Sites follow local distortions
- More physically meaningful for deformable systems

### Advantages over Polyhedral Sites
- Maintains space-filling property of Voronoi sites
- Computationally more efficient than polyhedral sites
- Simpler geometric definition
- No complex containment calculations

### Limitations
- More complex setup than regular Voronoi sites
- Requires careful choice of reference atoms
- Computationally more expensive than fixed-centre sites
- Site shapes still determined by relative positions only

## Troubleshooting

### Problem: No sites found
**Solutions**:
- Verify the reference structure contains the specified centre species
- Adjust cutoff distance to capture reference atoms
- Check that n_reference matches actual coordination
- Ensure proper structure alignment

### Problem: Sites not adapting as expected
**Solutions**:
- Verify reference atoms are correctly identified
- Check that reference species are framework atoms (not mobile)
- Ensure mapping is working correctly between structures

### Problem: Unexpected site positions
**Solutions**:
- Visualise the reference atoms for each site
- Check for periodic boundary condition issues
- Verify the mean position calculation is appropriate for your system
- Consider if some reference atoms span periodic boundaries
