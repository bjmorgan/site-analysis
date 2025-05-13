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
atom_sites = trajectory.atom_sites  # Which site each atom is in
site_occupations = trajectory.site_occupations  # Which atoms each site contains

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

### Analysis Data

The analysis builds up valuable data about site occupations and transitions:

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

For manually assigning atoms to sites:

```python
trajectory.assign_site_occupations(atoms, structure)
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
