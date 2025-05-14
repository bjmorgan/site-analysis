# Working with Spherical Sites

Spherical sites are the simplest site type in `site_analysis`, defined by a centre position and radius. They represent spherical volumes within a crystal structure where mobile ions can reside. An atom is considered to be within a spherical site if the distance between the atom and the site centre is less than or equal to the site's radius.

For a conceptual overview of spherical sites, see the [spherical sites concepts page](../concepts/sites.md#spherical-sites).

This guide covers practical aspects of using spherical sites in your analysis, including when to use them, how to configure them, and how to handle their limitations.

## When to Use Spherical Sites

Spherical sites are best suited for:
- Initial exploratory analysis
- Simple systems with well-separated sites  
- Comparing with published work that used spherical sites
- Quick visualizations of site locations

**Consider alternative site types when**:
- You need complete spatial coverage (use Voronoi or polyhedral sites)
- Sites have specific coordination geometries (use polyhedral sites)
- The framework undergoes topology-preserving distortions (use dynamic Voronoi or polyhedral sites)
- You're analyzing close-packed structures where polyhedral sites provide both complete coverage and correspond to crystallographic coordination environments

## Understanding Spherical Site Properties

Spherical sites have unique characteristics that affect their use:

- **Fixed geometry**: Sites remain spherical regardless of structure
- **Potential gaps**: Space between non-overlapping spheres remains unassigned
- **Potential overlaps**: Overlapping spheres create ambiguous regions
- **Radius dependence**: Site definition requires choosing appropriate radii

The radius parameter presents a fundamental trade-off:
- Small radii → gaps between sites → unassigned atoms
- Large radii → overlapping sites → ambiguous assignments

The package handles overlaps through a persistence algorithm: atoms remain in their current site when in overlapping regions, reducing spurious transitions.

## Creating Spherical Sites

### Basic Setup

```python
from site_analysis import TrajectoryBuilder

# Define site centres and radii
site_centres = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
site_radii = 2.0  # Same radius for all sites

trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(centres=site_centres, radii=site_radii)
    .build())
```

### Variable Radii

Different sites often require different radii:

```python
# Larger radius for octahedral sites, smaller for tetrahedral
octahedral_centres = [[0.5, 0.5, 0.5]]
tetrahedral_centres = [[0.25, 0.25, 0.25]]

trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(
        centres=octahedral_centres, 
        radii=2.5,
        labels="octahedral"
    )
    .with_spherical_sites(
        centres=tetrahedral_centres, 
        radii=1.8,
        labels="tetrahedral"
    )
    .build())
```

### Choosing Appropriate Radii

Site radii should be chosen based on:
- Crystal structure geometry
- Typical bond distances
- Desired site overlap/separation

A common approach is to start with radii based on typical bond distances and adjust empirically:

```python
# Test different radii and check site occupations
test_radii = [1.5, 2.0, 2.5]

for radius in test_radii:
    trajectory = (TrajectoryBuilder()
        .with_structure(structure)
        .with_mobile_species("Li")
        .with_spherical_sites(centres=centres, radii=radius)
        .build())
    
    # Analyze first frame
    trajectory.analyse_structure(structure)
    
    # Check how many atoms are assigned
    assigned_atoms = sum(1 for atom in trajectory.atoms if atom.in_site is not None)
    print(f"Radius {radius}: {assigned_atoms}/{len(trajectory.atoms)} atoms assigned")
```

## Example: Interstitial Sites in Close-Packed Structures

```python
# FCC structure with octahedral and tetrahedral interstitials
octahedral = [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
tetrahedral = [
    [0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
    [0.75, 0.25, 0.75], [0.25, 0.75, 0.75],
    [0.75, 0.75, 0.75], [0.25, 0.25, 0.75],
    [0.25, 0.75, 0.25], [0.75, 0.25, 0.25]
]

# Different radii for different site types
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(centres=octahedral, radii=2.0, labels="oct")
    .with_spherical_sites(centres=tetrahedral, radii=1.5, labels="tet")
    .build())
```

## Advanced Usage: Direct Site Creation

For more control over site creation, you can bypass the builder and create spherical sites directly.

### Creating Individual Sites

```python
from site_analysis.spherical_site import SphericalSite
from site_analysis.spherical_site_collection import SphericalSiteCollection
import numpy as np

# Create individual spherical sites
site1 = SphericalSite(
    frac_coords=np.array([0.5, 0.5, 0.5]), 
    rcut=2.0, 
    label="octahedral"
)
site2 = SphericalSite(
    frac_coords=np.array([0.0, 0.0, 0.0]), 
    rcut=1.5, 
    label="tetrahedral"
)

# Create a collection
sites = [site1, site2]
site_collection = SphericalSiteCollection(sites)

# Create atoms and trajectory
from site_analysis import atoms_from_structure, Trajectory
atoms = atoms_from_structure(structure, "Li")
trajectory = Trajectory(sites=sites, atoms=atoms)
```

### Using Existing Site Objects

```python
# Create sites from a previous analysis
from site_analysis import SphericalSite

# Example: Load sites from a previous analysis or create programmatically
sites = []
site_data = [
    {"pos": [0.5, 0.5, 0.5], "radius": 2.0, "label": "oct_1"},
    {"pos": [0.0, 0.0, 0.0], "radius": 1.8, "label": "tet_1"},
]

for data in site_data:
    site = SphericalSite(
        frac_coords=np.array(data["pos"]),
        rcut=data["radius"],
        label=data["label"]
    )
    sites.append(site)

# Use sites with the builder
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_existing_sites(sites)
    .build())
```

This approach gives you full control over:
- Manually creating sites with specific properties
- Reusing sites from previous analyses
- Integrating with other workflows that generate site objects

## Comparison with Other Site Types

### Advantages
- Conceptually simple and intuitive
- Easy to define and visualise
- Computationally efficient

### Limitations
- Do not completely fill space (gaps between sites)
- May overlap, causing ambiguous assignment
- Size needs to be carefully chosen
- Less physically meaningful than geometry-based approaches
- Generally inferior to other site types for most analyses

## Handling Overlapping Sites

When sites overlap, the assignment algorithm prioritizes stability:

1. **Check previous assignment**: If the atom was in a site during the previous timestep and that site still contains the atom's current position, keep the atom in that site
2. **Find new assignment**: If the atom wasn't previously assigned OR has moved outside its previous site, check all sites in order and assign the atom to the first site that contains it

This persistence-based approach means atoms tend to remain in their current sites even when they're in overlapping regions, reducing spurious transitions between overlapping sites.

**Note**: Overlapping sites can be deliberately used to minimize spurious "transitions" caused by large amplitude thermal vibrations that don't represent true diffusive motion.

## Handling Gaps Between Sites

Gaps are inherent to spherical sites. To monitor unassigned atoms:

```python
# Analyze structure
trajectory.analyse_structure(structure)

# Find unassigned atoms
unassigned = [atom for atom in trajectory.atoms if atom.in_site is None]

if unassigned:
    print(f"{len(unassigned)} atoms not assigned to any site")
    
    # Check positions of unassigned atoms
    for atom in unassigned:
        pos = structure[atom.index].frac_coords
        print(f"Atom {atom.index} at {pos} is unassigned")
```

### Gap Mitigation Strategies

1. **Increase radii** (but may cause overlaps)
2. **Add intermediate sites** at key positions
3. **Switch to space-filling sites** (Voronoi or polyhedral)

## Troubleshooting

### Problem: Too many unassigned atoms
**Solutions**:
- Increase site radii
- Add more sites
- Consider using Voronoi or polyhedral sites

### Problem: Atoms jumping between overlapping sites
**Solutions**:
- Reduce site radii to minimize overlap
- The default persistence-based assignment helps minimize spurious jumps
- Consider polyhedral sites for better-defined boundaries

### Problem: Sites too small/large for structure
```python
# Check site occupancies
trajectory.analyse_structure(structure)

for site in trajectory.sites:
    n_atoms = len(site.contains_atoms)
    print(f"Site {site.index}: {n_atoms} atoms")
```
**Solutions**:
- Adjust radii based on occupancy patterns
- Use different radii for different site types
