# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.8.0] - 2026-03-27

### Added

- `TransitionTable.__str__()` for formatted text display with auto-detected integer/float formatting.
- `TransitionTable._repr_html_()` for automatic HTML table rendering in Jupyter notebooks.
- Displaying Tables section in the transition tables guide.

### Changed

- Minimum Python version bumped from 3.10 to 3.11 (required by pymatgen-core).
- Runtime dependency changed from `pymatgen` to `pymatgen-core`; full `pymatgen` moved to dev dependencies.

### Fixed

- Typo in comparing_site_definitions tutorial ("unambigious" -> "unambiguous").

## [1.7.0] - 2026-03-22

### Added

- `Site.residence_times()` method for computing per-atom consecutive-occupation run lengths from site trajectory data.
- `filter_length` parameter for smoothing brief boundary-crossing artefacts before computing run lengths.
- `Trajectory.transition_counts_by_site()`, `Trajectory.transition_counts_by_label()`, `Trajectory.transition_probabilities_by_site()`, and `Trajectory.transition_probabilities_by_label()` methods for aggregating per-site transition data.
- `TransitionTable` class: a labelled square matrix with `.matrix`, `.get()`, `.to_dict()`, and `.reorder()` access patterns.
- `TransitionTable` exported from top-level `site_analysis` package.
- `include_edge_runs` parameter to control whether runs truncated by trajectory boundaries are included (excluded by default to avoid biasing towards shorter times).
- `TransitionTable.filter()` method for filtering transitions by site label.
- Residence time analysis guide in documentation.
- `site_analysis.distances` module providing minimum-image convention distance functions (`mic_distance`, `all_mic_distances`, `frac_to_cart`) with optional numba acceleration, independent of pymatgen.
- `indices_for_species()` helper in `tools.py` replacing `structure.indices_from_symbol()`.
- Builder validation: `TrajectoryBuilder.build()` now checks for same-species atom pairs in the reference structure closer than `min_atom_distance` (default 0.5), catching incorrect Wyckoff positions that produce duplicate sites. Configurable via `.with_min_atom_distance()`.
- Builder validation: post-build duplicate site detection for polyhedral and dynamic Voronoi sites (rejects sites sharing the same vertex/reference indices).

### Changed

- **Pymatgen decoupling (#59)**: Structure and Lattice objects are now only used at public API boundaries (`TrajectoryBuilder`, `analyse_structure`, `ReferenceBasedSites`, `StructureAligner.align`). All internal computation operates on numpy arrays and lattice matrices. This is a purely internal refactoring — the public TrajectoryBuilder API is unchanged.
  - `Atom.assign_coords()` accepts a coordinate array instead of a Structure.
  - `correct_pbc()` and `unwrap_vertices_to_reference_center()` accept a lattice matrix instead of a Lattice.
  - `assign_site_occupations()` accepts a lattice matrix instead of a Structure across all site collection types.
  - `SphericalSite.contains_point()` and `contains_atom()` accept a lattice matrix instead of a Lattice.
  - `PolyhedralSite.assign_vertex_coords()` accepts arrays instead of a Structure.
  - `DynamicVoronoiSite.calculate_centre()` accepts arrays instead of a Structure.
  - `calculate_species_distances()`, `site_index_mapping()`, and `get_coordination_indices()` accept arrays instead of Structure objects.
  - `StructureAligner` optimisation loop no longer creates temporary Structure objects per iteration.
  - `IndexMapper` and `CoordinationEnvironmentFinder` extract arrays at their boundaries.
  - `ReferenceBasedSites` eagerly extracts arrays from both structures in its constructor.
- `Site.contains_point()` and `Site.contains_atom()` base class signatures no longer use `*args/**kwargs`. Subclasses declare concrete keyword-only parameters, so invalid arguments raise `TypeError` immediately.
- `PriorityAssignmentMixin` is now generic over site type (`PriorityAssignmentMixin[SiteT]`), so `_get_priority_sites` yields the concrete site type.
- `PolyhedralSite.contains_point()` no longer accepts a `structure` parameter. Use `notify_structure_changed()` or `assign_vertex_coords()` to set vertex coordinates before calling.
- `PolyhedralSiteCollection.sites_contain_points()` accepts arrays instead of a Structure, with length validation.

### Fixed

- Fixed type 2 (Mg) site Wyckoff position in argyrodite reference structure. The coordinate `[0.23, 0.92, 0.09]` was on the 96i general position, producing 384 duplicate sites. Replaced with `[0.77, 0.585, 0.585]` on the 48h special position.
- Removed stale `# type: ignore` on scipy import (scipy-stubs now provides types).
- Removed unused pymatgen imports from `polyhedral_site.py`, `dynamic_voronoi_site.py`, `site_collection.py`, `site.py`, and `index_mapper.py`.
- Fixed `get_vertex_indices` deprecation warning referencing non-existent `get_coordinating_indices` (should be `get_coordination_indices`).

### Performance

Benchmarked on Li6PS5Cl argyrodite (1056 sites, 192 mobile Li atoms, 140 MD frames):

| Site type | Before (ms/frame) | After (ms/frame) | Speedup |
|-----------|-------------------|-------------------|---------|
| Polyhedral | 9.95 | 7.26 | 1.37x |
| Voronoi | 7.35 | 3.28 | 2.24x |
| Dynamic Voronoi | 7.86 | 3.59 | 2.19x |
| Spherical | 4.79 | 1.70 | 2.82x |

Speedups come from replacing pymatgen's Cython `lattice.get_all_distances` with numba-accelerated `all_mic_distances`, and eliminating per-atom pymatgen PeriodicSite object creation in `assign_coords`.

## [1.6.0] - 2026-03-01

### Changed

- Replaced deprecated `tqdm_notebook` with `tqdm.auto`, which auto-detects the environment (terminal, Jupyter notebook, Jupyter lab). The `progress` parameter on `trajectory_from_structures` is now a simple boolean.
- Added PEP 561 `py.typed` marker for downstream type checking support.
- Added `types-tqdm` dev dependency; removed inline `# type: ignore` on tqdm import.
- Removed redundant mypy ignores for `tqdm`, `pymatgen`, and `scipy` (now covered by type stubs).
- Added Python 3.14 to CI test matrix; separated mypy into its own CI job.

### Removed

- Removed `progress='notebook'` option from `trajectory_from_structures` (`tqdm.auto` handles environment detection automatically).

## [1.5.0] - 2026-02-28

### Changed

- Extracted shared priority-based site assignment logic into `PriorityAssignmentMixin`, used by both `PolyhedralSiteCollection` and `SphericalSiteCollection`.
- Improved priority heuristic for both site collections: tracks two recent distinct sites (catches bounce-back) and uses precomputed distance-ranked site ordering instead of neighbour-cutoff-based ordering. ~68% fewer containment checks and ~1.4x wall-clock speedup on argyrodite Li6PS5Cl trajectories.
- Unified duplicated PBC correction logic into `correct_pbc()` in `pbc_utils`, used by both `PolyhedralSite` and `DynamicVoronoiSite`.
- Dissolved `containment.py`: moved `HAS_NUMBA` to `_compat.py`, PBC shift updates to `pbc_utils.py`, and face topology cache into `polyhedral_site.py`.

### Removed

- Removed `neighbour_cutoff` parameter from `SphericalSiteCollection` (distance-ranked ordering replaces neighbour-based ordering).
- Removed `containment` module (internals relocated; no public API change).

## [1.4.0] - 2026-02-28

### Changed

- Batch vectorised centre calculation for `DynamicVoronoiSite` collections, grouped by reference count. After the first frame, incremental PBC shift updates avoid per-site O(27-image) unwrapping (~4x speedup on MD trajectories).
- `SiteCollection` now owns site reset via `reset()`. `Trajectory.reset()` delegates to `site_collection.reset()`, ensuring collection-level caches are properly cleared.

## [1.3.0] - 2026-02-27

### Added

- Optional Numba acceleration for polyhedral site containment testing (~7x speedup). Install with `pip install site-analysis[fast]`.

### Changed

- Cached face topology with JIT-compiled surface normal queries, replacing per-timestep Delaunay construction.
- Lazy vertex coordinate assignment defers PBC correction until a site is actually queried.
- PBC image shift caching avoids the expensive 27-image distance search.
- Per-atom PBC image computation (`x_pbc` computed once per atom per frame rather than once per atom-site pair).
- `algo` parameter on `contains_point`/`contains_atom` is deprecated; algorithm is now selected automatically.

### Fixed

- Fixed `use_reference_centers` typo in PBC handling documentation.

## [1.2.7] - 2026-02-21

### Changed

- Replaced deprecated `typing` imports (`Optional`, `Union`, `List`, etc.) with built-in equivalents.
- Replaced defensive `assert` statements with proper exceptions (`ValueError`, `TypeError`, `RuntimeError`).
- Standardised indentation to 4 spaces across all source files.
- Removed redundant `object` inheritance from `Atom` and `Trajectory`.
- Replaced type-narrowing hacks with annotations.
- Consolidated lazy initialisation helpers in `ReferenceBasedSites`.
- Use public `assign_coords()` API instead of accessing private `Atom._frac_coords`.

### Fixed

- Fixed `StructureAligner._run_differential_evolution` mutating the caller's `minimizer_options` dict.
- Fixed incorrect docstrings (typos, wrong default values, mismatched parameter names).

## [1.2.6] - 2026-02-21

### Fixed

- Fixed float distance truncation in `get_coordination_indices` (neighbour distances were cast to `int`).
- Fixed `append_timestep` dropping timestep `t=0` (truthiness check treated 0 as falsy).
- Fixed transitions from site index 0 not being recorded (truthiness check treated 0 as falsy).
- Fixed `contains_point_sn` loop structure (`inside.append()` was inside the inner face loop).
- Fixed `Atom.from_dict` crashing when `frac_coords` is absent.
- Removed debug `print` statement from `PolyhedralSite.__init__`.
- Removed dead code and variable shadowing in `Trajectory.__init__`.
- Fixed `x_pbc` docstring (stated `(9, 3)` return shape; actual shape is `(8, 3)`).

## [1.2.5] - 2026-01-17

### Fixed

- Fixed quickstart example to work out-of-the-box (#31).
- Added `example_data/XDATCAR` with a minimal Li trajectory for the quickstart example.
- Added helpful error message when mobile species is not found in structure.

## [1.2.4] - 2025-09-01

### Added

- `Trajectory.site_summaries()` method returning structured summary statistics for all sites.
- `Trajectory.write_site_summaries(filename)` method for exporting summaries to JSON.
- Optional `metrics` parameter for selecting specific statistics (occupancy, transitions, etc.).
- Numpy array inputs for `PolyhedralSite`.
- Description label for `tqdm` progress output.

## [1.2.1] - 2025-06-07

### Changed

- Optimised `SphericalSiteCollection` site assignment using the same priority-based algorithm introduced in 1.2.0, reducing complexity from O(N^2) to O(kN).
- Configurable `neighbour_cutoff` parameter (default 10.0 A) for distance-based neighbour determination.

## [1.2.0] - 2025-06-07

### Added

- `most_frequent_transitions()` method on `Site` class.

### Changed

- Optimised `PolyhedralSiteCollection` site assignment with a priority-based algorithm leveraging spatial relationships and learned transition patterns. 10x speedup for large systems (1440 sites).

## [1.1.3] - 2025-06-06

### Fixed

- Fixed errors in PBC handling documentation.

## [1.1.0] - 2025-06-06

### Added

- Reference-centre PBC unwrapping method that correctly handles coordination environments spanning more than half the simulation cell. This is now the default.
- Previous spread-based method remains available via `use_reference_centers=False`.
- PBC concepts documentation.

### Fixed

- Fixed bug in mapping central atom indices to coordination environments.
- Fixed handling of unwrapped mobile atom coordinates.

## [1.0.3] - 2025-06-03

### Fixed

- Fixed `ImportError` when importing `ReferenceBasedSites`.

## [1.0.1] - 2025-06-03

### Fixed

- Fixed missing `reference_workflow` subpackage in PyPI distribution.

## [1.0.0] - 2025-05-23

First stable release, accompanying the JOSS paper submission.

## [0.5.0] - 2025-05-14

### Changed

- Refactored structure alignment for modularity.
- Added global minimiser option (`differential_evolution`) for structure alignment.
- Exposed alignment tolerance parameter in Trajectory Builder and Reference-Based Sites workflows.
- Added comprehensive documentation.

## [0.3.1] - 2025-05-09

### Added

- `most_recent_site` property on `Atom` for tracking last known site.
- O(1) site lookup dictionary in `SiteCollection`.

### Changed

- Priority-based site checking in `PolyhedralSiteCollection` and `SphericalSiteCollection` (check most recent site first).

## [0.3.0] - 2025-05-09

### Changed

- `TrajectoryBuilder.build()` now numbers generated sites from zero.
- Multiple site definitions (of the same type) supported in a single trajectory.
- Separated structure alignment from site mapping in the builder API.
- Intelligent defaults for alignment and mapping species.

## [0.2.0] - 2025-05-08

### Added

- Dynamic Voronoi site types.
- `ReferenceBasedSites` workflow for generating sites from reference structures.
- Trajectory builders.

### Changed

- Minimum Python version increased to 3.10.
- Migrated to `pyproject.toml`.

## [0.0.1] - 2020-04-21

Initial release.
