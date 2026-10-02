# Counting transitions across unassigned frames

## Problem

Site-to-site transitions are recorded only in
`SiteCollection.update_occupation`, which takes the previous site from
`atom.trajectory[-1]`. If the atom was unassigned (`None`) in the previous
recorded frame, no transition is recorded. A hop A -> (one or more
unassigned frames) -> B is therefore never counted.

This affects any site type that can leave atoms unassigned: spherical sites,
and polyhedral sites that do not fill space. With small spherical sites most
real hops pass through at least one unassigned frame, so transition counts
can be substantially undercounted. Hops with longer transit paths are lost
more often, which also biases transition probabilities. Everything built on
`site.transitions` inherits this: `Trajectory.transition_counts_by_site()`,
`transition_counts_by_label()`, the probability methods,
`Site.most_frequent_transitions()`, and summaries.

## Decision

A hop A -> (unassigned frames) -> B counts as one A -> B transition. Gaps in
site coverage are presumed deliberate: the sites are the defined states, and
the only site-level information is "last site A, next site B".

## Change

In `SiteCollection.update_occupation`, take the previous site from
`atom.most_recent_site` (the last site the atom was assigned to in any
analysed structure) instead of `atom.trajectory[-1]`. If it is not `None`
and differs from the new site, increment `previous.transitions[new]`. The
call to `atom.update_recent_site(site.index)` stays at the end of the
method, so `most_recent_site` still holds the previous assignment when the
transition is checked.

`most_recent_site` and the last non-`None` entry of `atom.trajectory` hold
the same information in the normal `Trajectory` workflow. They differ only
when `analyse_structure` is called without `append_timestep`; there,
`most_recent_site` is consistent with the other state that
`update_occupation` already updates on every analysed structure.

Unchanged: occupation updates, `None` entries in atom trajectories, and
residence times.

## Consequences

- A -> (any number of unassigned frames) -> B records one A -> B transition.
- A -> (unassigned frames) -> A records nothing.
- A first assignment records nothing.
- Voronoi and dynamic Voronoi sites behave identically (they never produce
  `None`).
- Structures analysed with `analyse_structure` but not appended now also
  record transitions.

## Tests

In `tests/test_site_collection.py`, following the existing style
(`Mock(spec=Site)` sites, `Mock(spec=Atom)` atoms):

New tests:

| Case | `atom.trajectory` | `atom.most_recent_site` | Assigned to | Expected transitions |
|---|---|---|---|---|
| Across one unassigned frame | `[12, None]` | 12 | 42 | site 12: `{42: 1}` |
| Across several unassigned frames | `[12, None, None]` | 12 | 42 | site 12: `{42: 1}` |
| Return to the same site | `[12, None]` | 12 | 12 | none |
| No prior site | `[None]` | `None` | 42 | none |

The first two fail on `main`; the last two are guards that pass already.

Updated tests: `test_update_occupation_if_atom_has_moved`,
`test_update_occupation_if_atom_has_not_moved`,
`test_update_occupation_records_transition_from_site_index_zero` and
`test_update_occupation_calls_update_recent_site` (in
`tests/test_site_collection.py`), and `test_update_occupation_with_transition`
(in `tests/test_spherical_site_collection.py`), set the previous site through
`atom.trajectory` (or left it unset). They now set it through
`most_recent_site` (or `update_recent_site` for a real `Atom`).

A further test with real objects,
`TrajectoryTransitionCountingTestCase.test_transition_recorded_across_unassigned_timestep`
in `tests/test_trajectory.py`, runs the A -> (unassigned) -> B case through
`Trajectory.trajectory_from_structures`.

Then run the full test suite.

## Documentation

- `CHANGELOG.md`: a "Fixed" entry under an unreleased section. Transitions
  through unassigned frames are now counted, so transition counts from
  spherical or non-space-filling polyhedral analyses will increase.
- `docs/source/guides/trajectories.md`, "Handling Unassigned Timesteps": one
  sentence stating that transitions are recorded between consecutive
  assigned sites, so A -> None -> B counts as one A -> B transition.
