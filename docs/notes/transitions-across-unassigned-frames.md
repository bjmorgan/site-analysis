# Counting transitions across unassigned frames

## Problem

Site-to-site transitions are recorded only in
`SiteCollection.update_occupation`. Before this change, it took the previous
site from `atom.trajectory[-1]`. If the atom was unassigned (`None`) in the
previous recorded frame, no transition was recorded, so a hop A -> (one or
more unassigned frames) -> B was never counted.

This affected any site type that can leave atoms unassigned: spherical sites,
and polyhedral sites that do not fill space. With small spherical sites most
real hops pass through at least one unassigned frame, so transitions could be
substantially undercounted. Hops with longer transit paths were lost more
often, which also biased transition probabilities. Everything built on
`site.transitions` inherited this: `Trajectory.transition_counts_by_site()`,
`transition_counts_by_label()`, the probability methods,
`Site.most_frequent_transitions()`, and summaries.

## Decision

A hop A -> (unassigned frames) -> B counts as one A -> B transition. Gaps in
site coverage are presumed deliberate: the sites are the defined states, and
the only site-level information is "last site A, next site B".

## Change

`SiteCollection.update_occupation` now takes the previous site from
`atom.most_recent_site` (the last site the atom was assigned to in any
analysed structure) instead of `atom.trajectory[-1]`. If that site is not
`None` and differs from the new site, it increments
`previous.transitions[new]`. The call to `atom.update_recent_site(site.index)`
remains at the end of the method, so `most_recent_site` still holds the
previous assignment when the transition is checked.

`most_recent_site` and the last non-`None` entry of `atom.trajectory` hold
the same information in the normal `Trajectory` workflow. They differ only
when `analyse_structure` is called without `append_timestep`; there,
`most_recent_site` is consistent with the other state that
`update_occupation` already updates on every analysed structure.

Unchanged: `None` entries in atom trajectories. Occupations and residence
times are also unchanged, except where sites overlap (see Consequences).

## Consequences

- A -> (any number of unassigned frames) -> B records one A -> B transition.
- A -> (unassigned frames) -> A records nothing.
- A first assignment records nothing.
- Voronoi and dynamic Voronoi sites are unaffected by counting across
  unassigned frames (they never produce `None`); only the
  `analyse_structure` change below applies to them.
- Transitions are recorded relative to the last site the atom was assigned
  to in any analysed structure (`most_recent_site`), including structures
  passed directly to `analyse_structure` rather than through
  `append_timestep`. (Before this change they were compared with the last
  appended timestep, so repeated calls could count the same hop more than
  once.)
- Where sites overlap, site assignments can change.
  `PriorityAssignmentMixin._get_priority_sites` checks the atom's two most
  recent sites first, then the recorded transition destinations of the
  anchor site (the most recent site, or the nearest site centre if the atom
  has no history), and only then the remaining sites by distance (or
  neighbour) ranking. Recording different transitions can therefore change
  which of two overlapping sites is checked first.

## Tests

In `tests/test_site_collection.py`, following the existing style
(`Mock(spec=Site)` sites, `Mock(spec=Atom)` atoms):

New tests:

| Case | `atom.most_recent_site` | Assigned to | Expected transitions |
|---|---|---|---|
| Across unassigned frames | 12 | 42 | site 12: `{42: 1}` |
| Return to the same site | 12 | 12 | none |
| No prior site | `None` | 42 | none |

These set only `most_recent_site`, which is all `update_occupation` now
reads.

Updated tests: `test_update_occupation_if_atom_has_moved`,
`test_update_occupation_if_atom_has_not_moved`,
`test_update_occupation_records_transition_from_site_index_zero` and
`test_update_occupation_calls_update_recent_site` (in
`tests/test_site_collection.py`), and `test_update_occupation_with_transition`
(in `tests/test_spherical_site_collection.py`). Before this change these set
the previous site through `atom.trajectory` (or left it unset); they now set
it through `most_recent_site` (or `update_recent_site` for a real `Atom`).

Tests with real objects, in `TrajectoryTransitionCountingTestCase` in
`tests/test_trajectory.py`:

- `test_transition_recorded_across_unassigned_timestep` runs the
  A -> (unassigned) -> B case through `Trajectory.trajectory_from_structures`.
- `test_repeated_analyse_structure_does_not_double_count` checks that two
  direct `analyse_structure` calls after a hop record one transition. This
  fails before this change, which counted the hop twice.

`test_reset` in `tests/test_trajectory.py` also checks that `reset()` clears
`most_recent_site`.

## Documentation

- `CHANGELOG.md`, "Unreleased": a "Changed" entry for recording transitions
  relative to `most_recent_site` (including the `Trajectory.reset()` advice
  after direct `analyse_structure()` calls) and a "Fixed" entry for counting
  transitions through unassigned timesteps.
- `docs/source/guides/trajectories.md`: a paragraph in "Handling Unassigned
  Timesteps" on how transitions are counted across `None` entries, and a note
  in "Single Structure Analysis" on calling `reset()` after diagnostic
  `analyse_structure()` calls.
