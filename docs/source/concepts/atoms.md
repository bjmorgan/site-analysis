# Atoms

## What is an Atom in site_analysis?

In `site_analysis`, an **Atom** represents a single mobile ion that is tracked throughout a simulation. Unlike conventional molecular dynamics representations that simply store coordinates, the `Atom` class maintains information about site occupation and movement history, enabling analysis of diffusion pathways.

Each `Atom` has a unique index that corresponds to its position in the original structure and maintains its identity across timesteps as its position changes. This persistent identity allows the package to track how ions move between sites over time.

## Core Atom Properties

The `Atom` class maintains several key attributes:

- **index**: Unique numerical identifier that matches the atom's index in the source structure
- **species_string**: Chemical symbol of the atom (e.g., "Li", "Na")
- **in_site**: Current site index that this atom occupies (or None if not in any site)
- **frac_coords**: Current fractional coordinates in the crystal structure
- **trajectory**: List of site indices (or None values) that this atom has occupied over time

These properties provide the foundation for analyzing how atoms move through the crystal structure during a simulation.

## Atoms and Site Occupation

The relationship between atoms and sites is bidirectional:

1. Each atom records which site it currently occupies via its `in_site` attribute
2. Each site records which atoms it currently contains via its `contains_atoms` list

When analyzing a structure, the appropriate site collection determines which atoms belong to which sites.

## Tracking Atom Trajectories

As a simulation progresses, an atom's trajectory is built by recording the sequence of sites it occupies at each timestep. This trajectory is stored as a list of site indices or `None` values, where each entry corresponds to a timestep in the simulation:

```
atom.trajectory = [5, 5, 5, 12, 12, 12, 12, 15, 15, ...]
```

In this example, the atom started in site 5, moved to site 12 after three timesteps, and then to site 15 after four more timesteps.

If an atom is not in any site at a particular timestep (e.g., it's in a gap between sites), the trajectory will contain `None` for that timestep. The `in_site` attribute is also set to `None` when an atom is not in any site, ensuring that both the current state and the trajectory consistently represent periods when an atom is not assigned to any site.

## Creating and Managing Atoms

In most workflows, atoms are created automatically by the `TrajectoryBuilder` when you specify mobile species:

```python
trajectory = (TrajectoryBuilder()
             .with_structure(structure)
             .with_mobile_species("Li")  # Creates Atom objects for all Li ions
             .with_polyhedral_sites(...)
             .build())
```

However, the package provides several utility functions for creating atoms in different ways:

- **atoms_from_structure**: Create atoms for specified species in a structure
- **atoms_from_species_string**: Create atoms for a specific species in a structure
- **atoms_from_indices**: Create atoms with specific atom indices

## Bulk Analysis of Atom Movements

The `Trajectory` class provides methods to analyze the collective behavior of atoms:

- **atom_sites**: Returns the current site for each atom
- **atoms_trajectory**: Returns the site occupation history for all atoms over time

These methods facilitate analysis of diffusion patterns, correlations between atom movements, and statistical properties of ion migration.

## Atoms vs. Sites Perspective

`site_analysis` allows you to analyze diffusion from two complementary perspectives:

1. **Atom-centric**: Following individual atoms as they move between sites
2. **Site-centric**: Analyzing which atoms occupy each site over time

The atom-centric view is useful for tracking specific ions and their pathways, while the site-centric view helps identify preferred sites, occupation probabilities, and transition frequencies.
