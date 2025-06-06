# Handling Periodic Boundary Conditions

For site types defined using reference atoms ({class}`~site_analysis.polyhedral_site.PolyhedralSite` and {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite`), it is necessary to correctly identify and unwrap sets of reference atoms that span the periodic boundaries of the simulation cell.

```{figure} ../_static/figures/pbcs_unwrapping.png
:width: 90%

**Caption:** (a) A polyhedral site defined by four vertex atoms. (b) A 4-coordinate polyhedral site that wraps around the simulation cell periodic boundary. (c) Neglecting periodic boundary conditions leads to incorrect site geometry. (d) Properly unwrapping the wrapped atoms gives the correct site geometry.

`site_analysis` implements two approaches for handling periodic boundary conditions in reference-based site types: **spread-based detection**, which identifies wrapped sites based on the spatial distribution of reference atoms, and **reference centre unwrapping**, which uses the position of each site's central atom as a reference point for unwrapping vertex coordinates to their closest periodic images.

## Technical Overview

### Spread-Based Detection

This approach detects when reference atoms have wrapped around periodic boundaries by examining their spatial distribution. When the coordinate range of reference atoms exceeds 0.5 (half the unit cell) in any dimension, the algorithm assumes the site spans a periodic boundary and shifts coordinates accordingly to unwrap the site.

This method works well provided all sites have spans less than 0.5 times the simulation cell dimension, even under thermal distortions. However, it produces incorrect results when sites legitimately span more than 50% of the unit cell dimension. This situation commonly arises in small supercells—for example, in 2&times;2&times;2 expansions of FCC structures, octahedral sites extend exactly across half the simulation cell in the perfect crystal structure and can extend beyond this under simulation dynamics.

```{figure} ../_static/figures/pbcs_spread_method.png
:width: 90%

**Caption:** (a) If a site spans < &frac12 of the simulation cell in all dimensions it is considered to not wrap around the periodic boundaries and no unwrapping is needed. (b) Sites with spans > &frac12 of the simulation cell along any one dimension are assumed to wrap around the periodic boundaries (as in this figure). Unwrapping is performed to restore the correct site geometry. (c) In small supercells where sites naturally span &asymp; &frac12; of unit cell dimensions the method becomes unreliable, as legitimate site geometries can erroneously be assigned as wrapping around the periodic boundaries. These sites are then &ldquo;unwrapped&rdquo; to produce incorrect site geometries and false site assignments.

### Reference Centre Unwrapping (**default**)

This approach uses the position of each site's central atom as an anchor point for unwrapping. For each reference atom, the algorithm tests all 27 possible periodic images and selects the one closest to the central atom's position. This method works correctly even in small supercells where sites may legitimately span more than 50% of unit cell dimensions.

**Reference Centre Determination:**
The reference centre for each site is the fractional coordinates of the central atom used to generate the coordination environment when using the {class}`~site_analysis.builders.TrajectoryBuilder` or {class}`~site_analysis.reference_workflow.ReferenceBasedSites` workflows (e.g., the Li atom in a Li-O<sub>4</sub> tetrahedral environment). When creating sites manually using constructors like {class}`~site_analysis.polyhedral_site.PolyhedralSite` or {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite`, you must explicitly pass `reference_center` positions to the constructor. If no reference center is provided, the analysis will automatically fall back to the spread-based detection method.

```{figure} ../_static/figures/pbcs_reference_center_method.png
:width: 80%

**Caption:** Reference center unwrapping process: (a) Site with central atom (×) and coordinating atoms that wrap around periodic boundaries. (b) The reference center method considers all periodic images (3×3 supercell view) and selects the closest image of each coordinating atom relative to the central atom. (c) Result: correctly unwrapped site geometry.

**Important limitation:** This method assumes that the central atoms remain reasonably close to their initial positions throughout the analysis. If sites drift significantly from their reference positions during a simulation, the unwrapping may become unreliable.

### Performance Considerations

For some systems the spread-based method can be up to 20% faster than the reference centre method, which may be advantageous when analysing large simulation trajectories. The specific performance characteristics depend on your system&mdash;the number of sites, system size, and coordination environments all influence computational cost. The reference-center method is the default approach when using the {class}`~site_analysis.builders.TrajectoryBuilder` or {class}`~site_analysis.reference_workflow.ReferenceBasedSites` workflows, due to its correctness in small simulation cells. If you are certain that all sites have spreads well below &frac12; the simulation cell dimensions, even under dynamic distortions, you may wish to try the spread-based method (e.g., by setting `use_reference_centers=True`) to compare the relative performance of the two methods.

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
