# Spatial Commitment

By default, each atom is assigned to the site it is inside at each timestep. Atoms vibrating near a site boundary can cross back and forth between two sites, recording spurious transitions.

Commitment (spatial milestoning) counts a hop only when an atom reaches the core of its new site. Each site has a core: the part of the site within a commitment radius of its centre. Each atom has a committed site. When an atom moves into a different site, it commits to that site once it is inside the site's core, and stays committed to its previous site until then.

## Turning Commitment On

```python
trajectory = (TrajectoryBuilder()
    .with_structure(structure)
    .with_mobile_species("Li")
    .with_spherical_sites(centres=centres, radii=1.5)
    .with_commitment()
    .build())
```

`with_commitment()` uses a commitment radius of 1.0 Å for every site. Pass `radius` to set one radius for every site, or a dict mapping site labels to radii:

```python
# One radius for every site
builder.with_commitment(radius=0.8)

# A radius per site label
builder.with_commitment(radius={"tet": 0.6, "oct": 1.0})
```

When constructing a `Trajectory` directly, pass `commitment_radius`:

```python
trajectory = Trajectory(sites=sites, atoms=atoms, commitment_radius=1.0)
```

Every radius must be positive. A dict must give a radius for every site label, and can be used only if every site has a label.

## Effect on the Analysis

With commitment on, `append_timestep()` and `trajectory_from_structures()` use each atom's committed site:

- `atom.in_site` and `atom.trajectory` give each atom's committed site.
- `site.contains_atoms` and `site.trajectory` list the atoms committed to each site.
- A transition is recorded each time an atom's committed site changes from one site to another, so transitions match the atom trajectories exactly.
- Residence times and occupations are computed from the site trajectories, so they use the committed sites.

In the first timestep in which an atom is assigned to a site, it commits to that site. From then on, its trajectory has no `None` entries: an atom between sites, or inside another site but outside that site's core, keeps its committed site.

More than one atom can be committed to a site at once. For example, an atom can still be committed to a site it has left when a second atom commits to the same site.

`site.points` holds the positions of the atoms inside each site, whatever their committed sites. `analyse_structure()` assigns atoms to the sites they are in, with or without commitment.

Where sites overlap, the [site search](../concepts/site_collections.md#priority-based-assignment-spherical-and-polyhedral) checks an atom's recent sites first, then the sites that atoms have previously moved to from its most recent site. With commitment on, those previous moves are the committed transitions, so an atom in an overlapping region can be assigned to a different site than in an analysis without commitment.

## Choosing a Radius

If the commitment radius covers the whole site, the core is the whole site, and an atom commits to that site as soon as it enters it.

Small sites need small radii. For example, a tetrahedron of sulfide ions with edges of about 4 Å has an inradius (the distance from its centre to its faces) of about 0.8 Å, so a 1.0 Å radius extends beyond its faces and gives a core of about half its volume. Use a dict of radii per site label when sites differ in size.

An atom must be inside a core at an analysed timestep to commit to that site, so the core needs to be large compared with how far an atom moves between timesteps.

## Commitment and Residence-Time Filtering

`Site.residence_times()` can also smooth out short excursions with [`filter_length`](residence_times.md#filtering-short-gaps), which fills short gaps in a site's occupation. That filter acts on one site at a time, after the analysis, and is set as a number of timesteps. Commitment is applied during the analysis, is set as a distance, and is used by the trajectories, transitions, occupations and residence times. The two can be used together.
