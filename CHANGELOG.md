# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
