# Working with Voronoi Sites

Voronoi sites divide space into regions where each point in a region is closer to its site centre than to any other site centre. This creates a complete partitioning of space with no gaps or overlaps, ensuring every atom is assigned to exactly one site.

For a conceptual overview of Voronoi sites, see the [Voronoi sites concepts page](../concepts/sites.md#voronoi-sites).

This guide covers practical aspects of using Voronoi sites in your analysis, including when to use them, how to configure them, and how to leverage their unique properties.

## When to Use Voronoi Sites

Voronoi sites are best suited for:
- Ensuring complete spatial coverage (no unassigned atoms)
- Systems where proximity-based assignment is appropriate
- Avoiding ambiguity in site assignment
- Simple systems where distance to site centres determines occupation
- When computational efficiency is important

**Consider alternative site types when**:
- You need sites tied to specific coordination environments (use polyhedral sites)
- Site centres need to adapt to structural changes (use dynamic Voronoi sites)
- You're analysing complex coordination geometries
- Chemical bonding patterns don't align with simple distance metrics
- You need to match published work using different site definitions

## Understanding Space Partitioning

Voronoi sites completely fill space—every point belongs to exactly one site. This has important implications:

- **No gaps**: All atoms will always be assigned to a site
- **No overlaps**: Each atom belongs to exactly one site
- **Distance-based**: Assignment is determined solely by proximity to site centres

This makes Voronoi sites ideal when you need to ensure every atom is tracked, but means the site boundaries may not correspond to physical or chemical boundaries in your system.

## Creating Voronoi Sites

### Basic Setup

```python
from site_analysis import TrajectoryBuilder

# Define site centres
site_centres = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]

trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_voronoi_sites(centres=site_centres)
    .build())
```

### With Labels

Labels help identify different types of sites:

```python
# Label different crystallographic positions
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_voronoi_sites(
        centres=site_centres,
        labels=["octahedral", "tetrahedral_1", "tetrahedral_2"]
    )
    .build())
```

### Choosing Site Centres

Unlike spherical sites where you must choose both centres and radii, Voronoi sites only require centre positions. The site boundaries are automatically determined by the relative positions of all centres.

Common approaches for choosing centres:

- **From crystallographic positions**: Use known interstitial sites
- **From known ion positions**: Use equilibrium positions from a relaxed structure
- **From symmetry analysis**: Extract Wyckoff positions for the mobile species

## Example: Interstitial Sites in FCC Structure

```python
# Common interstitial positions in FCC structure
octahedral = [[0.5, 0.5, 0.5], [0.0, 0.5, 0.5], 
              [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
tetrahedral = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.25],
               [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]

all_centres = octahedral + tetrahedral

# Create Voronoi sites with meaningful labels
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_voronoi_sites(
        centres=octahedral,
        labels=["oct_1", "oct_2", "oct_3", "oct_4"]
    )
    .with_voronoi_sites(
        centres=tetrahedral,
        labels=["tet_1", "tet_2", "tet_3", "tet_4"]
    )
    .build())
```

## Advanced Usage: Direct Site Creation

For more control over site creation, you can bypass the builder and create Voronoi sites directly.

### Creating Individual Sites

```python
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.voronoi_site_collection import VoronoiSiteCollection
import numpy as np

# Create individual Voronoi sites
site1 = VoronoiSite(frac_coords=np.array([0.5, 0.5, 0.5]), label="octahedral")
site2 = VoronoiSite(frac_coords=np.array([0.0, 0.0, 0.0]), label="tetrahedral")

# Create a collection
sites = [site1, site2]
site_collection = VoronoiSiteCollection(sites)

# Create atoms and trajectory
from site_analysis import atoms_from_structure, Trajectory
atoms = atoms_from_structure(structure, "Li")
trajectory = Trajectory(sites=sites, atoms=atoms)
```

### Using with Custom Workflows

```python
# Example: Create sites based on analysis of equilibrium structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Analyze structure symmetry
analyzer = SpacegroupAnalyzer(equilibrium_structure)
sym_structure = analyzer.get_symmetrized_structure()

# Extract unique Li positions
li_sites = []
for equiv_sites in sym_structure.equivalent_sites:
    if equiv_sites[0].species_string == "Li":
        # Take one representative from each set of equivalent sites
        pos = equiv_sites[0].frac_coords
        li_sites.append(VoronoiSite(frac_coords=pos))

# Create collection and trajectory
site_collection = VoronoiSiteCollection(li_sites)
atoms = atoms_from_structure(md_structure, "Li")
trajectory = Trajectory(sites=li_sites, atoms=atoms)
```

This approach gives you full control over:
- How sites are identified and created
- Site labelling and metadata
- Integration with other analysis tools

For a comparison of all site types and guidance on choosing between them, see the [site selection guide](../concepts/sites.md#selecting-the-right-site-type).

## Troubleshooting

### Problem: Sites have unexpected shapes
**Solutions**:
- Ensure centres are appropriately spaced
- Add more centres in sparse regions
- Consider if Voronoi is appropriate for your system

### Problem: Sites don't match expected chemistry
**Solutions**:
- Voronoi sites are purely geometric—consider polyhedral sites for chemistry-based definitions
- Adjust centre positions to better match chemical environments
- Use dynamic Voronoi sites if centres should adapt to structure

### Problem: Too many/too few sites
**Solutions**:
- Review your crystallographic analysis
- Ensure you're not duplicating symmetry-equivalent positions
- Consider the appropriate level of spatial discretisation for your analysis
