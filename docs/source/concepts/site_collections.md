# Site Collections

## What is a Site Collection?

A **site collection** is a fundamental abstraction in `site_analysis` that manages a group of related sites and handles the assignment of atoms to these sites based on their positions in a crystal structure. While individual sites define bounded volumes, site collections orchestrate how atoms are assigned to sites and maintain the overall spatial organization.

Site collections are essential for several reasons:
- They ensure consistent handling of atom assignments across multiple sites
- They implement different assignment strategies appropriate for each site type
- They provide methods for analyzing structures and tracking occupations
- They maintain relationships between sites (e.g., neighboring sites)

## Core Site Collection Properties

All site collection types in `site_analysis` share common responsibilities:

- **Site management**: Organizing and maintaining a group of related sites
- **Atom assignment**: Implementing the logic to determine which atoms belong to which sites
- **Structure analysis**: Processing structures to update site assignments
- **Occupation tracking**: Maintaining records of which atoms occupy which sites
- **Transition tracking**: Recording movements of atoms between sites

Each specific site collection type provides specialised implementations of these responsibilities appropriate for its corresponding site type.

## Site Collections and Spatial Discretisation

Site collections determine how space is discretised and how ambiguities are resolved. This is particularly important for:

1. **Overlapping sites**: How to decide which site an atom belongs to when it could belong to multiple sites
2. **Gaps between sites**: How to handle atoms that don't clearly belong to any site
3. **Prioritization**: What rules to follow when assignment is ambiguous

Different site collection types implement different strategies for these scenarios, resulting in distinct behaviors that influence analysis results.

## Available Site Collection Types

The `site_analysis` package provides four different site collection types, each paired with its corresponding site type:

### SphericalSiteCollection

Manages a collection of spherical sites and implements specialised assignment logic for spherical volumes.

**Assignment strategy**:
- Implements a persistence-based approach for handling overlapping sites
- First checks if an atom is still in its previously assigned site
- If not, sequentially checks all sites until a match is found
- The first site found to contain the atom claims it (no further checks)

**Advantages**:
- Efficient for small oscillatory movements
- Reduces spurious transitions when sites overlap
- Provides stable assignments despite ambiguities

**Limitations**:
- Assignment depends on atom history and site order
- No physical basis for the persistence heuristic
- Cannot resolve the fundamental gaps between spherical sites

### PolyhedralSiteCollection

Manages a collection of polyhedral sites and implements assignment based on polyhedron containment tests.

**Assignment strategy**:
- Similar to SphericalSiteCollection, uses a persistence-based approach
- First checks the previously assigned site
- If the atom has moved, sequentially checks all sites
- Maintains information about face-sharing neighboring polyhedra

**Advantages**:
- Supports coordination environment analysis
- Can identify neighboring sites that share faces
- Adapts to structural distortions

**Limitations**:
- Computationally intensive containment tests requiring Delaunay tessellation
- Cannot easily vectorise operations for performance
- May not completely fill space in some structures
- Requires complex geometric calculations for each site at each timestep

### VoronoiSiteCollection

Manages a collection of Voronoi sites and implements assignment based on proximity to site centers.

**Assignment strategy**:
- For each atom, calculates distances to all site centers
- Assigns the atom to the site with the nearest center
- Uses the lattice to correctly handle periodic boundary conditions

**Advantages**:
- Unambiguous assignment (every atom belongs to exactly one site)
- Complete spatial coverage (no gaps)
- Computationally efficient global approach

**Limitations**:
- Purely distance-based assignment
- No consideration of chemical bonding or coordination

### DynamicVoronoiSiteCollection

Manages a collection of dynamic Voronoi sites where the centers are calculated from reference atom positions.

**Assignment strategy**:
- First dynamically updates each site's center based on reference atom positions
- Then follows the same assignment logic as VoronoiSiteCollection
- Calculates distances to the dynamic centers and assigns to nearest

**Advantages**:
- Adapts to structural changes and distortions
- Maintains complete spatial coverage
- Accounts for framework flexibility
- More computationally efficient than polyhedral sites for large systems

**Limitations**:
- Site centers and shapes may change during analysis
- Not directly tied to coordination environments

## How Site Collections Assign Atoms to Sites

The site assignment process is a core function of site collections. While the specific algorithms differ between collection types, the general workflow is:

1. **Reset**: Clear previous site occupations
2. **Process structure**: Update atomic positions from the current structure
3. **Apply site-specific logic**: Determine which atoms belong to which sites
4. **Update occupation records**: Record the new assignments
5. **Track transitions**: Identify and count transitions between sites

## Special Case: Overlapping Sites vs. Gaps

The handling of overlapping regions and gaps between sites is a critical aspect of site collections:

### Overlapping Sites

When sites overlap, an atom could potentially belong to multiple sites. Different site collections handle this differently:

- **Spherical and Polyhedral**: Use persistence to maintain assignment to the previous site if possible, otherwise assign to the first containing site found
- **Voronoi and Dynamic Voronoi**: Avoid overlaps entirely through mathematical partitioning of space

### Gaps Between Sites

When there are gaps between sites, an atom might not belong to any site:

- **Spherical and Polyhedral**: Atoms in gaps are not assigned to any site
- **Voronoi and Dynamic Voronoi**: Eliminate gaps entirely, ensuring every atom is assigned to some site

## Site Collections and Trajectory Analysis

In the context of analyzing an entire trajectory:

1. Site collections maintain the state of site occupations at each timestep
2. The Trajectory class uses site collections to process sequences of structures
3. Site collections accumulate statistics about occupations and transitions
4. This data forms the basis for higher-level analysis of diffusion mechanisms

By managing the complexities of atom assignment, site collections provide the foundation for quantitative analysis of diffusion pathways and mechanisms in molecular dynamics simulations.

## Relationship to Builder Pattern

Users rarely interact directly with site collection classes. Instead, the TrajectoryBuilder provides a convenient interface for creating appropriate site collections based on the chosen site type.

For example, when using `.with_spherical_sites()`, the builder automatically creates a SphericalSiteCollection to manage those sites. This abstraction simplifies the API while maintaining the specialised behavior needed for each site type.
