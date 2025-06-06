# Handling Periodic Boundary Conditions

For site types defined using reference atoms ({class}`~site_analysis.polyhedral_site.PolyhedralSite` and {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite`), it is necessary to correctly identify and unwrap sets of reference atoms that span the periodic boundaries of the simulation cell.

```{figure} ../_static/figures/pbcs_unwrapping.png
:width: 90%

**Caption:** (a) A polyhedral site defined by four vertex atoms. (b) A 4-coordinate polyhedral site that wraps around the simulation cell periodic boundary. (c) Neglecting periodic boundary conditions leads to incorrect site geometry. (d) Properly unwrapping the wrapped atoms gives the correct site geometry.

`site_analysis` implements two approaches for handling periodic boundary conditions in reference-based site types: **spread-based detection**, which identifies wrapped sites based on the spatial distribution of reference atoms, and **reference centre unwrapping**, which uses the position of each site's central atom as a reference point for unwrapping vertex coordinates to their closest periodic images.

## Technical Overview

### Spread-Based Detection

This approach detects when reference atoms have wrapped around periodic boundaries by examining their spatial distribution. When the coordinate range of reference atoms exceeds 0.5 (half the unit cell) in any dimension, the algorithm assumes the site spans a periodic boundary and shifts coordinates accordingly to unwrap the site.

This method works well provided all sites have spans less than 0.5 times the simulation cell dimension, even under thermal distortions. However, it produces incorrect results when sites legitimately span more than 50% of the unit cell dimension. This situation commonly arises in small supercells—for example, in 2×2×2 expansions of FCC structures, octahedral sites extend exactly across half the simulation cell in the ideal structure and can extend beyond this under simulation dynamics.

```{figure} ../_static/figures/pbcs_spread_method.png
:width: 90%

**Caption:** Spread-based detection in different scenarios: (a) Site with spans < &frac12; in all dimensions - method works correctly. (b) Site wrapping around periodic boundary with span > &frac12; in one dimension - method fails and produces incorrect geometry. (c) Small supercell where sites naturally span ≈ &frac12; of unit cell dimensions - the method becomes unreliable even for legitimate site geometries.

### Reference Centre Unwrapping (**default**)

This approach uses the position of each site's central atom as an anchor point for unwrapping. For each reference atom, the algorithm tests all 27 possible periodic images and selects the one closest to the central atom's position. This method works correctly even in small supercells where sites may legitimately span more than 50% of unit cell dimensions.

**Reference Centre Determination:**
The reference centre for each site is the fractional coordinates of the central atom used to generate the coordination environment when using the {class}`~site_analysis.builders.TrajectoryBuilder` or {class}`~site_analysis.reference_workflow.ReferenceBasedSites` workflows (e.g., the Li atom in a Li-O<sub>4</sub> tetrahedral environment). When creating sites manually using constructors like {class}`~site_analysis.polyhedral_site.PolyhedralSite` or {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite`, you must explicitly pass `reference_center` positions to the constructor. If no reference center is provided, the analysis will automatically fall back to the spread-based detection method.

```{figure} ../_static/figures/pbcs_reference_center_method.png
:width: 80%

**Caption:** Reference center unwrapping process: (a) Site with central atom (×) and coordinating atoms that wrap around periodic boundaries. (b) The reference center method considers all periodic images (3×3 supercell view) and selects the closest image of each coordinating atom relative to the central atom. (c) Result: correctly unwrapped site geometry.

**Important limitation:** This method assumes that the central atoms remain reasonably close to their initial positions throughout the analysis. If sites drift significantly from their reference positions during a simulation, the unwrapping may become unreliable.

### Performance Considerations

The two methods give similar performance, but for some molecular dynamics analyses there may be a moderate speed improvement from using one method over the other. The specific performance characteristics depend on your system—the number of sites, system size, and coordination environments all influence computational cost. The reference-center method is the default and recommended approach for all cases due to its reliability. Some systems may show improved analysis speed with the spread-based method, but this should only be used when you are confident that all sites have spreads well below 0.5 times the simulation cell dimensions in all directions.

## Controlling PBC Handling in Your Code

### Using TrajectoryBuilder or ReferenceBasedSites

You can control the PBC method using the `use_reference_centers` parameter:

```python
# Reference center approach (recommended, default)
builder.with_polyhedral_sites(..., use_reference_centers=True)

# Spread-based approach (advanced usage, do not use for small simulation cells)
builder.with_polyhedral_sites(..., use_reference_centers=False)
```

For more details on using these workflows, see {doc}`trajectory_builder` and {doc}`reference_workflow`.

### Manual Site Creation

When creating sites directly using constructors:

```python
# Reference center approach
site = PolyhedralSite(vertex_indices=[...], reference_center=center_coords)

# Spread-based approach (omit reference_center)
site = PolyhedralSite(vertex_indices=[...])
```

For more information on manual site creation, see the {class}`~site_analysis.polyhedral_site.PolyhedralSite` and {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite` documentation.
