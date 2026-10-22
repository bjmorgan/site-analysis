# Handling Periodic Boundary Conditions

For site types defined using reference atoms ({class}`~site_analysis.polyhedral_site.PolyhedralSite` and {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite`), it is necessary to correctly identify and unwrap sets of reference atoms that span the periodic boundaries of the simulation cell.

```{figure} ../_static/figures/pbcs_unwrapping.png
:width: 90%

**Caption:** (a) A polyhedral site defined by four vertex atoms. (b) A 4-coordinate polyhedral site that wraps around the simulation cell periodic boundary. (c) Neglecting periodic boundary conditions leads to incorrect site geometry. (d) Properly unwrapping the wrapped atoms gives the correct site geometry.
```

`site_analysis` unwraps the reference atoms of these site types by **reference centre unwrapping**, which uses the position of each site's central atom as a reference point for unwrapping vertex coordinates to their closest periodic images. Sites without a reference centre use **spread-based detection**, which identifies wrapped sites based on the spatial distribution of reference atoms. Sites without a reference centre are deprecated: creating one issues a `FutureWarning`, and version 2.0 will require a reference centre for every polyhedral and dynamic Voronoi site and remove spread-based detection.

## Technical Overview

### Spread-Based Detection (deprecated)

This approach, used only by sites without a reference centre, detects when reference atoms have wrapped around periodic boundaries by examining their spatial distribution. When the coordinate range of reference atoms exceeds 0.5 (half the unit cell) in any dimension, the algorithm assumes the site spans a periodic boundary and shifts coordinates accordingly to unwrap the site.

This method works well provided all sites have spans less than 0.5 times the simulation cell dimension, even under thermal distortions. However, it produces incorrect results when sites legitimately span more than 50% of the unit cell dimension. This situation commonly arises in small supercells—for example, in 2&times;2&times;2 expansions of FCC structures, octahedral sites extend exactly across half the simulation cell in the perfect crystal structure and can extend beyond this under simulation dynamics.

```{figure} ../_static/figures/pbcs_spread_method.png
:width: 90%

**Caption:** (a) If a site spans < &frac12; of the simulation cell in all dimensions it is considered to not wrap around the periodic boundaries and no unwrapping is needed. (b) Sites with spans > &frac12; of the simulation cell along any one dimension are assumed to wrap around the periodic boundaries (as in this figure). Unwrapping is performed to restore the correct site geometry. (c) In small supercells where sites naturally span &asymp; &frac12; of unit cell dimensions the method becomes unreliable, as legitimate site geometries can erroneously be assigned as wrapping around the periodic boundaries. These sites are then &ldquo;unwrapped&rdquo; to produce incorrect site geometries and false site assignments.
```

### Reference Centre Unwrapping (**default**)

This approach uses the position of each site's central atom as an anchor point for unwrapping. Each vertex is moved to its periodic image nearest the central atom, exactly in any cell, by the same search used for minimum-image distances. The vertices and the central atom may therefore be in any periodic images. The unwrapped vertices are then translated by whole cells so that the lowest coordinate along each axis is in [0, 1). This method works correctly even in small supercells where sites may legitimately span more than 50% of unit cell dimensions.

**Reference Centre Determination:**
The reference centre for each site is the fractional coordinates of the central atom used to generate the coordination environment when using the {class}`~site_analysis.builders.TrajectoryBuilder` or {class}`~site_analysis.reference_workflow.ReferenceBasedSites` workflows (e.g., the Li atom in a Li-O<sub>4</sub> tetrahedral environment). When creating sites manually using constructors like {class}`~site_analysis.polyhedral_site.PolyhedralSite` or {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite`, pass `reference_center`, usually the position of the site's central atom. Omitting it is deprecated: the site then falls back to spread-based detection and a `FutureWarning` is issued.

```{figure} ../_static/figures/pbcs_reference_center_method.png
:width: 80%

**Caption:** Reference centre unwrapping process: (a) Site with central atom (×) and coordinating atoms that wrap around periodic boundaries. (b) The reference centre method considers all periodic images (3×3 supercell view) and selects the closest image of each coordinating atom relative to the central atom. (c) Result: correctly unwrapped site geometry.
```

**Important limitation:** This method assumes that the central atoms remain reasonably close to their initial positions throughout the analysis. If sites drift significantly from their reference positions during a simulation, the unwrapping may become unreliable.

### Performance Considerations

Both methods unwrap vertices at similar cost. PBC image shifts are cached after the first frame and updated incrementally on subsequent frames, so the initial unwrapping cost is amortised over the trajectory.

The reference centres also drive the site search for polyhedral sites: they are used to rank candidate sites by distance from each atom. If any site lacks one (deprecated), the search for the whole collection falls back to the neighbours of the atom's most recent site, then list order, which can make assignment slower.

## Controlling PBC Handling in Your Code

### Using TrajectoryBuilder or ReferenceBasedSites

The {class}`~site_analysis.builders.TrajectoryBuilder` and {class}`~site_analysis.reference_workflow.ReferenceBasedSites` workflows give every site a reference centre:

```python
builder.with_polyhedral_sites(...)
```

Their `use_reference_centers` parameter defaults to `True`. Passing `use_reference_centers=False`, which creates sites without reference centres, is deprecated and issues a `FutureWarning`; it will be removed in version 2.0.

For more details on using these workflows, see the {doc}`../guides/builders` and {doc}`../guides/reference_workflow` guides.

## Manual Site Creation

When creating sites directly using constructors, pass the reference centre, usually the fractional coordinates of the site's central atom:

```python
site = PolyhedralSite(vertex_indices=[...], reference_center=centre_coords)
site = DynamicVoronoiSite(reference_indices=[...], reference_center=centre_coords)
```

Omitting `reference_center` is deprecated and issues a `FutureWarning`, as do `PolyhedralSite.sites_from_vertex_indices()`, `DynamicVoronoiSite.sites_from_reference_indices()`, and `from_dict()` for a dict without a reference centre.

For more information on manual site creation, see the {class}`~site_analysis.polyhedral_site.PolyhedralSite` and {class}`~site_analysis.dynamic_voronoi_site.DynamicVoronoiSite` documentation.
