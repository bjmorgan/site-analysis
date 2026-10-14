# Working with Trajectories

The `Trajectory` class is the central component in site-analysis for tracking ion migration through crystallographic sites. This guide explains how to use Trajectory objects to analyse structures and interpret the results.

## What is a Trajectory?

A Trajectory serves as the bridge between atoms and sites, with two main roles:
1. **Integration**: It combines sites, atoms, and structures into a unified framework
2. **Analysis**: It processes structures to track which atoms occupy which sites over time, recording both site occupations and transitions

## Creating Trajectory Objects

The primary way to create a Trajectory is through the [`TrajectoryBuilder`](builders.md):

```python
from site_analysis import TrajectoryBuilder
from pymatgen.core import Structure

structure = Structure.from_file("my_structure.cif")

trajectory = (TrajectoryBuilder()
             .with_structure(structure)
             .with_mobile_species("Li")
             .with_spherical_sites(
                 centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                 radii=2.0
             )
             .build())
```

For specific site types, see the relevant guides:
- [Working with Spherical Sites](spherical_sites.md)
- [Working with Voronoi Sites](voronoi_sites.md)
- [Working with Polyhedral Sites](polyhedral_sites.md)
- [Working with Dynamic Voronoi Sites](dynamic_voronoi_sites.md)

For advanced cases, you can create a Trajectory directly by providing sites and atoms:

```python
from site_analysis import Trajectory, atoms_from_structure
from site_analysis.spherical_site import SphericalSite
import numpy as np

# Create sites and atoms manually
sites = [
    SphericalSite(frac_coords=np.array([0.5, 0.5, 0.5]), rcut=2.0, label="octahedral")
]
atoms = atoms_from_structure(structure, "Li")

# Create trajectory directly
trajectory = Trajectory(sites=sites, atoms=atoms)
```

## Analysing Structures

### Single Structure Analysis

To analyse a single structure:

```python
# Analyse the structure
trajectory.analyse_structure(structure)

# Check site assignments
for atom in trajectory.atoms:
    if atom.in_site is not None:
        site = trajectory.site_by_index(atom.in_site)
        print(f"Atom {atom.index} is in site {site.label or site.index}")
```

### Molecular Dynamics Trajectory Analysis

For analysing a sequence of structures:

```python
# Method 1: Using a loop
for i, structure in enumerate(structures):
    trajectory.append_timestep(structure, t=i)
    
# Method 2: Built-in method with progress bar
trajectory.trajectory_from_structures(structures, progress=True)
```

## Accessing Analysis Results

After analysis, you can access results in several ways:

### Basic Site Occupation

```python
# Current state
geometric_sites = trajectory.geometric_sites  # Site each atom is inside
committed_sites = trajectory.committed_sites  # Committed site of each atom (with commitment)
site_occupations = trajectory.site_occupations  # Atoms counted in each site (committed atoms with commitment)

# Site occupation counts
for i, site in enumerate(trajectory.sites):
    label = site.label or f"Site {i}"
    print(f"{label} contains {len(site.contains_atoms)} atoms")
```

### Trajectory History

```python
# Full trajectory data
atoms_trajectory = trajectory.atoms_trajectory  # Sites per atom over time
sites_trajectory = trajectory.sites_trajectory  # Atoms per site over time

# Example: Track one atom through the simulation
atom = trajectory.atoms[0]
print(f"Site history for atom {atom.index}: {atom.trajectory}")
```

### Handling Unassigned Timesteps

When using spherical or polyhedral sites (which may not fill space completely), an atom's trajectory can contain `None` entries for timesteps where it was not inside any site:

```python
# Example trajectory with gaps
# [5, 5, None, None, 12, 12, 12]
#         ^^^^  ^^^^  atom was between sites

atom = trajectory.atoms[0]

# Count how often the atom was unassigned
unassigned = sum(1 for s in atom.trajectory if s is None)
total = len(atom.trajectory)
print(f"Atom {atom.index}: unassigned for {unassigned}/{total} timesteps")

# Filter to only assigned timesteps
assigned_sites = [s for s in atom.trajectory if s is not None]
```

A high proportion of `None` entries may indicate that site radii are too small or that a space-filling site type (Voronoi or dynamic Voronoi) would be more appropriate. Voronoi-based sites never produce `None` entries because they partition space completely.

Transitions are recorded between successive assigned sites in each atom's trajectory, skipping `None` entries. For example, an atom whose trajectory is `[5, None, None, 12]` contributes one transition, from site 5 to site 12.

To count a hop only once an atom is within a set distance of its new site's centre, see the [commitment guide](commitment.md).

### Analysis Data

The analysis builds up data about site occupations and transitions:

```python
# Site occupation positions and transitions
for site in trajectory.sites:
    # points: recorded positions of atoms assigned to this site
    if site.points:
        print(f"Site {site.index}: {len(site.points)} position records")
    
    # transitions: observed migrations from this site to others
    if site.transitions:
        print(f"Transitions from site {site.index}:")
        for dest_site, count in site.transitions.items():
            print(f"  → Site {dest_site}: {count} transitions")
```

### Transition Events

`site.transitions` counts how many times atoms moved from one site to another. To know when each transition happened, and which atom made it, use `transition_events()`:

```python
events = trajectory.transition_events()

len(events)          # number of transitions
events[0]            # TransitionEvent(frame=12, timestep=13, entry_frame=12, entry_timestep=13, atom=3, from_site=7, to_site=15)
events[0].frame      # 12

# Each field is an integer array with one element per transition (timesteps may be None)
events.to_site                     # destination of every transition
into_site = events[events.to_site == 15]   # transitions into site 15
atom_3 = events[events.atom == 3]          # every transition of atom 3

for event in atom_3:
    print(event.frame, event.from_site, event.to_site)
```

Transitions follow the same rule as `site.transitions`: a transition is a change of site between consecutive frames in which the atom has a site (those where its trajectory is not `None`), skipping the others, and its frame is the frame in which the atom is first counted in its new site. An atom whose trajectory is `[5, None, None, 12]` makes one transition, from site 5 to site 12, at frame 3. With [commitment](commitment.md), a transition is a change of committed site.

`frame` counts frames from 0 and indexes `atoms_trajectory` and `sites_trajectory` directly. `timestep` gives the label passed for each frame (`trajectory_from_structures()` numbers frames from 1); it is `None` if some frames were appended without a label. `atom` is the atom's `index`, as in `site.contains_atoms` and `sites_trajectory`; its column in `atoms_trajectory` is `trajectory.atom_lookup[event.atom]`. `from_site` and `to_site` are site indices, the values stored in `atoms_trajectory`. Events are sorted by frame, then by atom index.

`entry_frame` is the earliest frame from which the atom stayed inside its new site, without leaving, up to `frame`, and `entry_timestep` is its label. Without commitment they equal `frame` and `timestep`. With [commitment](commitment.md#entry-frames), an atom is counted in a new site only once it reaches the site's core, so `entry_frame` can be earlier than `frame`.

The events are derived from the stored trajectory each time `transition_events()` is called, so they agree with it, including after more frames are appended.

### Summary Statistics

Trajectory objects provide methods to extract and export summary statistics for all sites in a structured format:

```python
# Get summary statistics for all sites
trajectory.site_summaries()

# Example output
[{'index': 0,
  'site_type': 'SphericalSite',
  'label': 'octahedral',
  'average_occupation': 0.85,
  'transitions': {1: 12, 2: 3}},
 {'index': 1,
  'site_type': 'SphericalSite',
  'label': 'tetrahedral_1',
  'average_occupation': 0.15,
  'transitions': {0: 12, 2: 5}},
 ...
]
```

The summary includes key metrics for each site:
- `index`: The site's unique identifier
- `site_type`: The class of site (e.g., SphericalSite, PolyhedralSite)
- `label`: The site label if one was assigned
- `average_occupation`: Fraction of timesteps where the site was occupied (0.0-1.0)
- `transitions`: Dictionary mapping destination site indices to transition counts

#### Selecting Specific Metrics

The `site_summaries` method takes an optional `metrics` keyword that controls which metrics are included in the summary:

```python
# Get only specific metrics
trajectory.site_summaries(metrics=['index', 'average_occupation'])

# Result contains only requested fields
[{'index': 0, 'average_occupation': 0.85},
 {'index': 1, 'average_occupation': 0.15},
 ...]
```

Available metrics:
- `'index'`: Site's unique identifier
- `'label'`: Site label (if set)
- `'site_type'`: Class name of the site
- `'average_occupation'`: Fraction of timesteps occupied
- `'transitions'`: Transition counts to other sites

#### Exporting to JSON

Trajectory objects also offer a `write_site_summaries()` method that exports the summary statistics directly to a JSON file, making it easy to save results for subsequent analysis:

```python
# Write all summary statistics to a JSON file
trajectory.write_site_summaries('site_analysis.json')

# Write only selected metrics to file
trajectory.write_site_summaries('occupations.json', 
                                metrics=['index', 'label', 'average_occupation'])
```

The resulting JSON file contains the same structured data as returned by `site_summaries()`, formatted for easy reading and processing by other tools.

## Complete Workflow Example

```python
from site_analysis import TrajectoryBuilder
from pymatgen.io.vasp import Xdatcar
import matplotlib.pyplot as plt
import numpy as np

# Load MD trajectory
xdatcar = Xdatcar("XDATCAR")
structures = xdatcar.structures

# Create and configure the trajectory
trajectory = (TrajectoryBuilder()
              .with_structure(structures[0])
              .with_mobile_species("Li")
              .with_spherical_sites(
                  centres=[[0.5, 0.5, 0.5]], 
                  radii=2.0, 
                  labels=["octahedral"]
              )
              .with_spherical_sites(
                  centres=[[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], 
                  radii=1.5, 
                  labels=["tetrahedral1", "tetrahedral2"]
              )
              .build())

# Analyse the trajectory
trajectory.trajectory_from_structures(structures, progress=True)

# Plot site occupancy over time
timesteps = len(structures)
site_occupancy = np.zeros((len(trajectory.sites), timesteps))

for t in range(timesteps):
    for site_idx, site_atoms in enumerate(trajectory.sites_trajectory[t]):
        site_occupancy[site_idx, t] = len(site_atoms)

plt.figure(figsize=(10, 6))
for site_idx, site in enumerate(trajectory.sites):
    label = site.label or f"Site {site.index}"
    plt.plot(site_occupancy[site_idx], label=label)
    
plt.xlabel("Timestep")
plt.ylabel("Number of atoms")
plt.title("Site occupation over time")
plt.legend()
plt.show()
```

## Advanced Features

### Managing Trajectory State

Reset all trajectory data to start fresh:

```python
trajectory.reset()  # Clears all trajectory history
```

### Analysing Transition Statistics

```python
# Calculate transition probabilities
for site in trajectory.sites:
    total = sum(site.transitions.values())
    if total > 0:
        print(f"Transition probabilities from site {site.index}:")
        for dest, count in site.transitions.items():
            prob = count / total
            print(f"  → Site {dest}: {prob:.3f}")
```

The analysis results depend on your choice of site type. See the [Sites Concepts page](../concepts/sites.md) and the specific [Site Collections page](../concepts/site_collections.md) for details on how different site types handle atom assignments.
