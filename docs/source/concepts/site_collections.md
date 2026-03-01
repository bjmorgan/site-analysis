# Site Collections

## What is a Site Collection?

A **site collection** manages a group of related [sites](sites.md) and handles the assignment of atoms to these sites based on their positions in a crystal structure. While individual sites define bounded volumes, site collections orchestrate how atoms are assigned to sites at each timestep.

Users rarely interact directly with site collection classes — the `TrajectoryBuilder` creates the appropriate collection automatically when you call methods like `.with_spherical_sites()` or `.with_polyhedral_sites()`. This page explains the assignment logic that happens behind the scenes, which is useful for understanding how your results are produced.

## Assignment Strategies

Each site collection type implements a different strategy for assigning atoms to sites. The choice of strategy is determined by the spatial properties of the corresponding site type.

### Priority-Based Assignment (Spherical and Polyhedral)

Spherical and polyhedral sites can overlap or leave gaps between them. These collections check sites one at a time and assign the atom to the first containing site found. To avoid unnecessary containment checks, a priority heuristic determines the order in which sites are tested:

1. **Recent history**: the atom's most recently occupied site(s) are checked first
2. **Learned transitions**: sites that atoms have previously transitioned to from the anchor site, ordered by frequency
3. **Distance ranking**: remaining sites ordered by distance from the anchor site centre

If no trajectory history exists (e.g. the first timestep), the nearest site centre is used as the starting anchor instead.

This ordering means atoms that remain in or near their current site are resolved in a single check, and common transitions are tested early. In practice, this eliminates the majority of containment checks compared to a naive sequential scan.

If no site contains the atom, it is left unassigned (`None`).

For polyhedral sites, the collection also maintains information about face-sharing neighbouring polyhedra, which can be used for identifying connected diffusion pathways.

### Nearest-Centre Assignment (Voronoi and Dynamic Voronoi)

Voronoi-based sites partition space completely, so there are no gaps or overlaps. Assignment is straightforward: each atom is assigned to whichever site centre is nearest, using the lattice to correctly handle periodic boundary conditions.

For dynamic Voronoi sites, the collection first updates each site's centre from the current positions of its reference atoms, then applies the same nearest-centre logic.

## Overlaps and Gaps

The handling of spatial ambiguity is the key difference between collection types:

| Scenario | Spherical / Polyhedral | Voronoi / Dynamic Voronoi |
|---|---|---|
| Overlapping sites | Priority-based: atom stays in previous site if possible | Cannot occur (space is partitioned) |
| Gaps between sites | Atom is unassigned (`None`) | Cannot occur (space is partitioned) |

See the [sites concepts page](sites.md) for guidance on choosing a site type based on these trade-offs.
