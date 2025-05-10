# Working with Spherical Sites

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

## Choosing Appropriate Radii

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

## Handling Overlapping Sites

When sites overlap, the assignment algorithm prioritizes stability:

For each atom:
1. **Check previous assignment**: If the atom was in a site during the previous timestep and that site still contains the atom's current position, keep the atom in that site
2. **Find new assignment**: If the atom wasn't previously assigned OR has moved outside its previous site, check all sites in order and assign the atom to the first site that contains it

This persistence-based approach means atoms tend to remain in their current sites even when they're in overlapping regions, reducing spurious transitions between overlapping sites.

**Note**: Overlapping sites can be deliberately used to minimize spurious "transitions" caused by large amplitude thermal vibrations that don't represent true diffusive motion. By creating overlap regions between sites, atoms undergoing vibrational motion near site boundaries are less likely to register as having transitioned between sites.

To detect potential overlaps:

```python
import numpy as np

def check_site_overlaps(centres, radii, lattice):
    """Check which sites overlap"""
    overlaps = []
    for i, (c1, r1) in enumerate(zip(centres, radii)):
        for j, (c2, r2) in enumerate(zip(centres, radii)):
            if i >= j:
                continue
            
            # Calculate distance between centres
            distance = lattice.get_distance_and_image(c1, c2)[0]
            
            # Check if sites overlap
            if distance < (r1 + r2):
                overlaps.append((i, j, distance, r1 + r2))
    
    return overlaps
```

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

## Common Patterns

### Interstitial Sites in Close-Packed Structures

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

