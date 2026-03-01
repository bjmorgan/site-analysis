# Sites

## What is a Site?

A **site** is a fundamental abstraction in `site_analysis` that represents a bounded volume within a crystal structure that can contain zero, one, or multiple mobile ions. Sites provide a way to discretise continuous atomic trajectories into a more interpretable representation based on site occupation and transitions.

When analyzing ion migration in solids, it's often useful to think about ions moving between specific locations ("sites") in the crystal structure. These sites typically correspond to local energy minima, where ions tend to reside for extended periods before jumping to adjacent sites. The `site_analysis` package lets you define these sites in various ways and then analyze how mobile ions move between them.

## Core Site Properties

All site types in `site_analysis` share common attributes:

- **Spatial definition**: Each site occupies a specific region in the crystal structure
- **Containment logic**: Logic to determine whether an atom is contained within the site
- **Occupation tracking**: Record of which atoms occupy the site over time
- **Transition tracking**: Record of transitions to and from other sites

Sites are organized into **Site Collections**, which manage groups of related sites and handle the assignment of atoms to these sites based on their positions.

## Available Site Types

The `site_analysis` package provides four different site types, each with specific characteristics suited to different analysis scenarios.

### Spherical Sites

Spherical sites are the simplest site type, defined by a center position and radius. They represent spherical volumes within the crystal structure.

**Advantages**:
- Conceptually simple and intuitive
- Easy to define and visualise
- Computationally efficient

**Limitations**:
- Do not completely fill space (gaps between sites)
- May overlap, causing ambiguous assignment
- Size needs to be carefully chosen
- Less physically meaningful than geometry-based approaches
- Generally inferior to other site types for most analyses

The radius parameter for spherical sites presents a fundamental trade-off. When using non-overlapping spheres, the sites naturally cannot fill all the available space in a crystal structure. This leads to "null" regions where mobile ions are not assigned to any site. These regions often include the transition paths between sites, which can be problematic for analyzing diffusion mechanisms, as ions may temporarily exist in an unassigned state during transitions.

To mitigate this issue, one might be tempted to increase the site radii to ensure fuller coverage of space. However, this introduces a new problem: overlapping sites. When spheres overlap, a mobile ion might simultaneously satisfy the containment criteria for multiple sites, creating ambiguity in site assignment.

The `site_analysis` package addresses this ambiguity through a priority-based assignment algorithm: each atom's recently occupied sites are checked first, followed by sites ordered by learned transition frequency and distance. The first containing site found claims the atom. This means an atom in an overlapping region will tend to remain in its current site, reducing spurious transitions from small oscillations.

For details on how the priority ordering works, see the [site collections](site_collections.md) page. While effective for maintaining assignment consistency, this approach does not resolve the fundamental spatial coverage issues with spherical sites, making other site definitions generally preferable for detailed mechanistic analysis.

**Best for**:
- Initial exploratory analysis
- Simple visualizations
- Comparison with published results using spherical site definitions
- Systems where precise site geometry is not critical

Note that spherical sites are primarily included in `site_analysis` for compatibility with published literature that uses this approach, rather than as a recommended option for most analyses. For most applications, polyhedral or Voronoi sites provide more accurate and physically meaningful results.

### Polyhedral Sites

Polyhedral sites are defined by a collection of vertex atoms that form a polyhedron. The site encompasses the volume within this polyhedron. This approach is especially useful for sites with coordination environments that correspond to common polyhedra (tetrahedra, octahedra, etc.).

In close-packed lattices where the mobile ions occupy interstitial sites with well-defined coordination environments, polyhedral sites can completely fill space if the set of all coordination polyhedra is used. This makes them particularly valuable for analyzing systems like lithium-ion battery materials or ionic conductors.

**Advantages**:
- Accurately represents coordination environments
- Shape adapts to the local structure
- Can represent complex geometries
- Captures structural distortions
- Can provide more physically meaningful discretisation than Voronoi sites for irregular polyhedra
- Space-filling in close-packed lattices when using complete set of coordination polyhedra

**Limitations**:
- More complex to define
- May not completely fill space in non-close-packed structures
- Requires careful selection of vertex atoms

**Best for**:
- When specific coordination environments are important
- Systems with well-defined polyhedral sites (e.g., tetrahedral, octahedral)
- When structural distortions need to be captured
- Mixed site geometries (e.g., combinations of tetrahedral and octahedral sites)

### Voronoi Sites

Voronoi sites divide space into regions where each point in a region is closer to its site center than to any other site center. This creates a complete partitioning of space with no gaps or overlaps.

**Advantages**:
- Completely fills space (no gaps or overlaps)
- Unambiguous assignment of atoms to sites
- Simple mathematical definition based on proximity
- Computationally efficient spatial partitioning

**Limitations**:
- Site shapes are determined by neighbor positions, not customisable
- Not directly tied to coordination environments
- Fixed center positions may not adapt to structural changes
- May not accurately reflect physical site shapes in complex environments
- Purely mathematical partitioning that doesn't consider chemical bonding
- Poor representation of elongated or asymmetric sites (will tend toward regular polyhedra)
- Site boundaries determined solely by proximity to site centers, regardless of actual physical site shapes

**Best for**:
- Complete spatial discretisation
- When gaps between sites are problematic
- Simple systems where proximity-based assignment is sufficient
- When every point must be assigned to exactly one site

### Dynamic Voronoi Sites

Dynamic Voronoi sites extend the Voronoi approach by calculating site centers dynamically based on the positions of reference atoms. This allows the sites to adapt to structural changes and distortions.

**Advantages**:
- Adapts to structural changes and distortions
- Completely fills space (no gaps or overlaps)
- Combines benefits of polyhedra and Voronoi approaches
- Works well for flexible or disordered structures

**Limitations**:
- More computationally intensive
- More complex to set up and understand
- Sites may change shape and size during analysis

**Best for**:
- Structures that deform during simulation
- Frameworks with significant thermal motion
- Materials with flexible coordination environments

## Site Occupation and Transitions

For all site types, the `site_analysis` package tracks:

1. **Site occupation**: Which atoms occupy each site at each timestep
2. **Site transitions**: Each site records transitions *to* other sites in a counter format

Specifically, each site maintains a `transitions` counter dictionary where:
- Keys are the destination site indices
- Values are the count of transitions to that destination site

For example, if site A has recorded 3 transitions to site B (with index 5) and 2 transitions to site C (with index 8), its transitions counter would contain `{5: 3, 8: 2}`.

Note that the package only explicitly tracks transitions *to* other sites. If you need to analyze transitions *from* specific sites, you would need to construct this data from the complete set of transitions *to* data.

This transition data forms the basis for analyzing diffusion mechanisms, including:
- Preferred migration pathways
- Frequency of specific site-to-site jumps
- Construction of diffusion networks
- Statistical analysis of migration processes

## Selecting the Right Site Type

The choice of site type depends on your specific research questions and the nature of your system:

| If you want to... | Consider using... |
|-------------------|-------------------|
| Analyze specific coordination environments | Polyhedral sites |
| Ensure complete spatial coverage with no gaps | Voronoi or Polyhedral sites |
| Account for framework flexibility or distortion | Dynamic Voronoi sites |
| Analyze systems with mixed site geometries | Polyhedral sites |
| Balance accuracy and computational efficiency | Voronoi sites |
| Perform detailed mechanistic analysis | Polyhedral or Dynamic Voronoi sites |
| Analyze close-packed structures with well-defined interstitial sites | Polyhedral sites |
| Analyze systems with irregularly shaped physical sites | Polyhedral sites |
| Compare with published results using spherical site definitions | Spherical sites |

While spherical sites are included for compatibility with published literature that uses this approach, they are generally not recommended for most analyses. Polyhedral or Voronoi sites typically provide more accurate and physically meaningful results.
