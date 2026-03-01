# Using the Builder Pattern

The `site_analysis` package uses the Builder pattern as its primary interface for creating `Trajectory` objects. This guide explains how to use the `TrajectoryBuilder` class to set up your analysis.

## Overview of the Builder Pattern

The builder pattern provides a step-by-step approach to creating complex objects. Instead of passing many arguments to a constructor, you chain method calls to configure your object before building it:

```python
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(centres=centres, radii=2.0)  # Single radius for all sites
    .build())
```

This approach offers several advantages:
- Clear, readable code that follows a logical setup sequence
- Optional parameters can be easily added or omitted
- Complex configurations remain manageable
- Validation can happen at build time

## Basic Usage

### Essential Components

Every trajectory needs at minimum:
1. A structure to define atom indices
2. Mobile species to track
3. Site definitions

Here's the simplest possible example:

```python
from site_analysis import TrajectoryBuilder
from pymatgen.core import Structure

# Load a structure that defines atom indices
structure = Structure.from_file("structure.cif")

# Define site centres (fractional coordinates)
site_centres = [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]

# Build the trajectory with a single radius for all sites
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(centres=site_centres, radii=1.5)  # Same radius for all sites
    .build())
```

### Method Chaining

The builder methods return `self`, allowing you to chain calls:

```python
builder = TrajectoryBuilder()
builder.with_structure(structure)
builder.with_mobile_species("Li")
builder.with_spherical_sites(centres=centres, radii=2.0)
trajectory = builder.build()

# Equivalent to:
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(centres=centres, radii=2.0)
    .build())
```

## Builder Methods Reference

### Core Configuration Methods

#### `with_structure(structure)`
Sets the structure that defines atom indices for the analysis.

This structure serves several purposes:
- Identifies which atom indices correspond to your mobile species
- Provides atom positions for site definition (when using coordination-based sites)
- Establishes the indexing scheme that will be used throughout the analysis

**Important**: This structure doesn't need to be one that you'll analyse later. Common approaches include:
- Using the first frame from your MD simulation
- Using an idealised structure with atoms at crystallographic positions (often better for site definition)

**Parameters:**
- `structure`: A pymatgen `Structure` object

**Example:**
```python
# Using an idealised structure for cleaner site definition
ideal_structure = Structure.from_file("ideal_structure.cif")
builder.with_structure(ideal_structure)

# Or using the first frame from an MD simulation
first_frame = structures[0]
builder.with_structure(first_frame)
```

#### `with_mobile_species(species)`
Defines which species to track during the analysis.

The builder uses the structure to identify which atom indices correspond to these species.

**Parameters:**
- `species`: Either a single species string (e.g., "Li") or a list of species strings (e.g., ["Li", "Na"])

**Examples:**
```python
# Track a single species
builder.with_mobile_species("Li")

# Track multiple species
builder.with_mobile_species(["Li", "Na"])
```

### Site Definition Methods

You must define sites using one or more of these methods, but all sites must be of the same type. You can call the same method multiple times to create multiple groups of sites:

#### `with_spherical_sites(centres, radii, labels=None)`
Defines spherical sites with specified centres and radii.

For spherical sites, structural disorder in the builder's structure is less important since sites are defined by fixed centre positions rather than atom coordinates.

**Parameters:**
- `centres`: List of fractional coordinate centres
- `radii`: Either a single radius (float) to use for all sites, or a list of radii (one per centre) in Ångstroms
- `labels`: Optional single label (str) to use for all sites, list of labels (one per centre), or None

**Examples:**
```python
# Single radius for all sites
builder.with_spherical_sites(
    centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    radii=1.5,  # Same 1.5 Å radius for both sites
    labels="interstitial"  # Same label for all sites
)

# Different radii for each site
builder.with_spherical_sites(
    centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    radii=[2.0, 1.5],  # Different radii
    labels=["octahedral", "tetrahedral"]  # Different labels
)
```

#### `with_voronoi_sites(centres, labels=None)`
Defines Voronoi sites with specified centres.

Like spherical sites, Voronoi sites use fixed centre positions, so structural disorder in the builder's structure has minimal impact.

**Parameters:**
- `centres`: List of fractional coordinate centres
- `labels`: Optional list of labels for the sites

**Example:**
```python
builder.with_voronoi_sites(
    centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    labels=["site_A", "site_B"]
)
```

#### `with_polyhedral_sites(centre_species, vertex_species, cutoff, n_vertices, label=None, use_reference_centers=True)`
Defines polyhedral sites using coordination environments.

This method benefits from using an idealised structure in the builder, as regular coordination environments are easier to identify and define. However, the ReferenceBasedSites workflow (which this uses internally) can handle some degree of structural distortion.

**Parameters:**
- `centre_species`: Species at the center of coordination environments
- `vertex_species`: Species at vertices of coordination environments
- `cutoff`: Distance cutoff for coordination environment
- `n_vertices`: Number of vertices per environment
- `label`: Optional label for all sites
- `use_reference_centers`: Whether to use reference centre unwrapping for periodic boundary conditions (default: `True`)
  - `True`: Uses each site's central atom position as an anchor point for unwrapping vertex coordinates to their closest periodic images
  - `False`: Uses spread-based detection, which identifies wrapped sites based on the spatial distribution of reference atoms

**Example:**
```python
builder.with_reference_structure(reference_structure)
       .with_polyhedral_sites(
           centre_species="Li",
           vertex_species="O",
           cutoff=2.5,
           n_vertices=4,
           label="tetrahedral"
       )
```

For detailed explanations of periodic boundary condition handling, see {doc}`../concepts/pbc_handling`.

#### `with_dynamic_voronoi_sites(centre_species, reference_species, cutoff, n_reference, label=None, use_reference_centers=True)`
Defines dynamic Voronoi sites using reference atom positions.

**Parameters:**
- `centre_species`: Species at the center of coordination environments
- `reference_species`: Species of reference atoms for dynamic centres
- `cutoff`: Distance cutoff for finding reference atoms
- `n_reference`: Number of reference atoms per site
- `label`: Optional label for all sites
- `use_reference_centers`: Whether to use reference centre unwrapping for periodic boundary conditions (default: `True`)
  - `True`: Uses each site's central atom position as an anchor point for unwrapping vertex coordinates to their closest periodic images
  - `False`: Uses spread-based detection, which identifies wrapped sites based on the spatial distribution of reference atoms

**Example:**
```python
builder.with_reference_structure(reference_structure)
       .with_dynamic_voronoi_sites(
           centre_species="Li",
           reference_species="O",
           cutoff=3.0,
           n_reference=6,
           label="dynamic_octahedral"
       )
```

For detailed explanations of periodic boundary condition handling, see {doc}`../concepts/pbc_handling`.

### Advanced Configuration Methods

#### `with_reference_structure(reference_structure)`
Sets a reference structure for polyhedral and dynamic Voronoi sites.

**Parameters:**
- `reference_structure`: A pymatgen `Structure` object representing the ideal reference structure

**Example:**
```python
builder.with_reference_structure(reference_structure)
```

#### `with_structure_alignment(align=True, align_species=None, align_metric='rmsd', align_algorithm='Nelder-Mead', align_minimizer_options=None, align_tolerance=1e-4)`
Controls structure alignment between reference and target structures.

**Note**: Structure alignment is **enabled by default** when using polyhedral or dynamic Voronoi sites, even if this method is not explicitly called. To disable alignment, you must call this method with `align=False`.

All parameters are optional and have sensible defaults:

**Parameters:**
- `align`: Whether to perform alignment (default: `True`)
- `align_species`: Species to use for alignment (default: `None`)
  - If `None`, mapping species will be used if specified, otherwise all common species
  - Can be a single species string (e.g., `"O"`) or a list of species (e.g., `["O", "Ti"]`)
  - Typically framework atoms are used to avoid issues with different mobile ion counts
- `align_metric`: Metric for alignment (default: `'rmsd'`)
  - `'rmsd'`: Root-mean-square deviation - minimises the average distance between corresponding atoms
  - `'max_dist'`: Maximum distance - minimises the largest distance between any corresponding atom pair
- `align_algorithm`: Algorithm for optimization (default: `'Nelder-Mead'`)
  - `'Nelder-Mead'`: Local optimization, faster but may find local minima
  - `'differential_evolution'`: Global optimization, more robust but slower
- `align_minimizer_options`: Additional options for the minimizer as a dictionary (default: `None`)
- `align_tolerance`: Convergence tolerance for the alignment optimization (default: `1e-4`)
  - Lower values (e.g., `1e-5`) give more precise alignment but may take longer

**Examples:**
```python
# Use default alignment (enabled, all species)
builder.with_reference_structure(reference)
       .with_polyhedral_sites(...)

# Specify alignment species explicitly
builder.with_structure_alignment(align_species=["O", "Ti"])
       .with_polyhedral_sites(...)

# Disable alignment
builder.with_structure_alignment(align=False)
       .with_polyhedral_sites(...)

# Use global optimization for challenging alignments
builder.with_structure_alignment(
    align_algorithm='differential_evolution',
    align_minimizer_options={'popsize': 20}
)
```

#### `with_site_mapping(mapping_species)`
Sets species to use for mapping sites between reference and target structures.

**Parameters:**
- `mapping_species`: Species to use for mapping (string or list of strings)

**Example:**
```python
builder.with_site_mapping(mapping_species=["O", "Ti"])
```

#### `with_existing_sites(sites)`
Uses pre-existing site objects instead of creating new ones.

**Parameters:**
- `sites`: List of site objects

#### `with_existing_atoms(atoms)`
Uses pre-existing atom objects instead of creating new ones.

**Parameters:**
- `atoms`: List of atom objects

## Common Patterns

### Multiple Site Groups of the Same Type

You can define multiple groups of sites by calling the same site definition method multiple times:

```python
# Create a trajectory with multiple groups of spherical sites
builder = TrajectoryBuilder()
builder.with_structure(structure)
       .with_mobile_species("Li")
       .with_spherical_sites(
           centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
           radii=2.0,
           labels="octahedral"
       )
       .with_spherical_sites(
           centres=[[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
           radii=1.5,
           labels="tetrahedral"
       )

trajectory = builder.build()  # Creates 4 spherical sites in total
```

### Simple Analysis with Spherical Sites

```python
# Using an idealised structure to define sites
ideal_structure = Structure.from_file("ideal_structure.cif")

trajectory = (TrajectoryBuilder()
    .with_structure(ideal_structure)
    .with_mobile_species("Li")
    .with_spherical_sites(
        centres=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        radii=2.0,  # Same radius for all sites
        labels="interstitial"  # Same label for all sites
    )
    .build())
```

### Coordination-Based Sites with Idealised Structure

```python
# Using an idealised structure for cleaner coordination environments
ideal_structure = Structure.from_file("ideal_LiCoO2.cif")

trajectory = (TrajectoryBuilder()
    .with_structure(ideal_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_structure_alignment(align=True, align_species=["O"])
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=6,
        label="octahedral"
    )
    .build())
```

### Using MD Frame as Structure

```python
# Using first MD frame when site positions are well-defined
first_frame = xdatcar.structures[0]

trajectory = (TrajectoryBuilder()
    .with_structure(first_frame)
    .with_mobile_species(["Li", "Na"])
    .with_voronoi_sites(
        centres=site_centres,
        labels=site_labels
    )
    .build())
```

## Understanding the Structure Parameter

The structure you provide to the builder serves as a reference for:

1. **Atom indexing**: Determines which indices correspond to your mobile species
2. **Site definition**: Provides atom positions when sites are defined from coordination environments

### Common Approaches

#### Using an Idealised Structure
Often the best approach for defining sites, especially coordination-based ones:
```python
# Atoms at ideal crystallographic positions
ideal_structure = Structure.from_file("ideal_structure.cif")
builder.with_structure(ideal_structure)
```

**Advantages:**
- Regular coordination environments are easier to identify
- Site definitions are more consistent
- Works well with polyhedral sites

#### Using an MD Frame
Useful when the structure is already well-ordered:
```python
# First frame from MD simulation
md_structure = xdatcar.structures[0]
builder.with_structure(md_structure)
```

**Advantages:**
- Direct correspondence with simulation data
- No need for separate ideal structure
- Works well for simple site definitions

### Impact on Different Site Types

- **Spherical/Voronoi sites**: Structural disorder has minimal impact since sites use fixed centres
- **Polyhedral sites**: Benefit from idealised structures with regular coordination
- **Dynamic Voronoi sites**: Can adapt to some structural variation

## Factory Functions

For convenience, the package provides factory functions that wrap common builder patterns:

### `create_trajectory_with_spherical_sites()`
```python
from site_analysis import create_trajectory_with_spherical_sites

trajectory = create_trajectory_with_spherical_sites(
    structure=structure,
    mobile_species="Li",
    centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    radii=2.0,  # Single radius for all sites
    labels="octahedral"  # Single label for all sites
)
```

### `create_trajectory_with_polyhedral_sites()`
```python
from site_analysis import create_trajectory_with_polyhedral_sites

trajectory = create_trajectory_with_polyhedral_sites(
    structure=ideal_structure,
    reference_structure=reference_structure,
    mobile_species="Li",
    centre_species="Li",
    vertex_species="O",
    cutoff=2.5,
    n_vertices=4,
    label="tetrahedral",
    align=True,
    align_species=["O"]
)
```

### `create_trajectory_with_voronoi_sites()`
```python
from site_analysis import create_trajectory_with_voronoi_sites

trajectory = create_trajectory_with_voronoi_sites(
    structure=structure,
    mobile_species="Li",
    centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    labels=["site_A", "site_B"]
)
```

### `create_trajectory_with_dynamic_voronoi_sites()`
```python
from site_analysis import create_trajectory_with_dynamic_voronoi_sites

trajectory = create_trajectory_with_dynamic_voronoi_sites(
    structure=target_structure,
    reference_structure=reference_structure,
    mobile_species="Li",
    centre_species="Li",
    reference_species="O",
    cutoff=3.0,
    n_reference=6,
    label="dynamic_octahedral",
    align=True,
    align_species=["O"]
)
```

## Best Practices

1. **Choose appropriate structure**: Use idealised structures for coordination-based sites
2. **Build in logical order**: Structure → mobile species → (reference structure) → sites
3. **Match structure to site type**: Consider how structural disorder affects your chosen site type
4. **Validate before building**: The builder validates at build time, so check error messages carefully

## Error Handling

The builder validates your configuration when you call `build()`. Common errors include:

```python
# Missing required components
builder = TrajectoryBuilder()
builder.with_structure(structure)
# Forgot mobile_species and sites
trajectory = builder.build()  # Raises ValueError

# Mixing different site types
builder.with_spherical_sites(centres=centres1, radii=1.5)
       .with_voronoi_sites(centres=centres2)  # Error: can't mix site types
trajectory = builder.build()  # Raises TypeError

# Missing reference structure
builder.with_polyhedral_sites(...)  # Without with_reference_structure()
trajectory = builder.build()  # Raises ValueError
```
