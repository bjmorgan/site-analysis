# PBC correction unification refactoring

Notes captured during the dynamic Voronoi optimisation work (PR #38).

## Problem

PolyhedralSite and DynamicVoronoiSite share near-identical PBC correction
logic that has been independently implemented in each class.

## Duplicated code

### 1. Full PBC unwrapping decision (identical in both)

Both `PolyhedralSite._store_vertex_coords` and
`DynamicVoronoiSite._compute_corrected_coords` contain:

```python
if self.reference_center is not None:
    corrected, image_shifts = unwrap_vertices_to_reference_center(
        frac_coords, self.reference_center, lattice,
        return_image_shifts=True)
else:
    corrected = apply_legacy_pbc_correction(frac_coords)
    image_shifts = np.round(corrected - frac_coords).astype(int)
```

This is character-for-character the same. It could be extracted into a
single utility function in `pbc_utils.py`.

### 2. Per-site PBC shift caching (polyhedral site only, post-simplification)

After the dynamic Voronoi simplification, only `PolyhedralSite` retains
per-site PBC shift caching via `_pbc_image_shifts` / `_pbc_cached_raw_frac`
and calls to `update_pbc_shifts`. This includes:

- Cache check and incremental update path
- Full computation fallback
- Cache attributes initialised in `__init__`, cleared in `reset()`

### 3. Batch PBC shift update reimplements `update_pbc_shifts`

`DynamicVoronoiSiteCollection._batch_calculate_centres` reimplements the
same wrapping-detection algorithm as `containment._numpy_update_pbc_shifts`,
but operating on `(n_sites, n_ref, 3)` tensors rather than `(n_ref, 3)`.
The core logic is the same:

```python
diff = coords - cached
wrapping = np.round(diff).astype(np.int64)
physical_diff = diff - wrapping
if np.all(np.abs(physical_diff) < 0.3):
    new_shifts = shifts - wrapping
    corrected = coords + new_shifts
    # uniform non-negative shift ...
```

### 4. Shared attributes

Both site types initialise and manage:
- `reference_center`
- `_pbc_image_shifts` (polyhedral only, post-simplification)
- `_pbc_cached_raw_frac` (polyhedral only, post-simplification)

## Design smell: `_compute_corrected_coords` dual behaviour

`DynamicVoronoiSite._compute_corrected_coords` sets `_centre_coords` as
its primary effect but returns `image_shifts` for the collection's batch
fallback. The main effect is the "side" effect; the return value serves
the caller. This is a consequence of the PBC correction logic living on
the site rather than being a standalone utility.

Extracting the PBC correction into a shared function would resolve this
naturally: `calculate_centre` would call the utility and compute the mean;
the batch fallback would call the utility and store shifts in the group.

## Possible refactoring approach

### Extract PBC correction utility

A function in `pbc_utils.py` that encapsulates the unwrapping decision:

```python
def correct_pbc(
    frac_coords: np.ndarray,
    reference_center: np.ndarray | None,
    lattice: Lattice,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PBC correction to fractional coordinates.

    Returns:
        (corrected_coords, image_shifts)
    """
    if reference_center is not None:
        return unwrap_vertices_to_reference_center(
            frac_coords, reference_center, lattice,
            return_image_shifts=True)
    corrected = apply_legacy_pbc_correction(frac_coords)
    image_shifts = np.round(corrected - frac_coords).astype(int)
    return corrected, image_shifts
```

Both site types would call this instead of duplicating the logic.

### Consider batch-aware `update_pbc_shifts`

Add a batch variant to `containment.py` that operates on
`(n_sites, n_ref, 3)` tensors, so the collection doesn't need to
reimplement the wrapping detection. Or reshape to `(n_sites * n_ref, 3)`,
call the existing function, and reshape back -- though this loses the
per-group all-or-nothing invalidation semantics.

### Consider a PBC cache mixin

If per-site caching is retained for polyhedral sites, a mixin or base
class could manage `_pbc_image_shifts`, `_pbc_cached_raw_frac`, and the
two-phase update pattern. This would avoid the current situation where
PolyhedralSite has the caching pattern but DynamicVoronoiSite does not.

## Caching strategy should remain per-site-type

The refactoring should extract the shared PBC *correction* logic but
leave the *caching strategy* where it makes sense for each site type's
access pattern:

- **Dynamic Voronoi**: collection-level batch caching. All centres must
  be computed every frame (for the distance matrix), so batch computation
  is natural. The collection groups sites by vertex count and does
  vectorised shift updates across all sites simultaneously.

- **Polyhedral**: per-site caching. `contains_point` is called per-site
  per-atom with lazy evaluation -- a site only computes its vertex coords
  when actually queried. The collection doesn't know which sites will be
  queried in what order, so batch precomputation would remove the benefit
  of lazy evaluation.

Additionally, polyhedral sites have variable vertex counts and need
full `(n_vertices, 3)` corrected coords stored back (for Delaunay / face
topology containment), not just a scalar centre. The batch path would be
more complex for less clear benefit.

If profiling ever shows polyhedral PBC correction is a bottleneck, the
dynamic Voronoi batch pattern is there as a template, but it shouldn't
be assumed to be needed without evidence.

## Scope

This is a pure refactoring -- no behaviour change. It should be done as
a separate piece of work after PR #38 is merged.
