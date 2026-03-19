# Residence Times and Transition Probabilities in Argyrodite

This tutorial demonstrates how to compute residence times and transition probabilities from a site-analysis trajectory. We reuse the argyrodite Li6PS5Cl dataset from {doc}`the previous tutorial </tutorials/argyrodite_site_analysis>`, so the setup code here is largely the same — see that tutorial for a detailed explanation of the reference structure and trajectory builder configuration.

## Prerequisites

This tutorial requires the following packages:

- `site-analysis`
- `numpy`
- `matplotlib`

This tutorial uses the same trajectory data as the argyrodite tutorial.
The data files are included in the `site-analysis` GitHub repository at
`tutorials/data/`:

- `Li6PS5Cl_0p_XDATCAR.gz` — 0% anion site exchange (fully ordered)
- `Li6PS5Cl_50p_XDATCAR.gz` — 50% anion site exchange
- `Li6PS5Cl_100p_XDATCAR.gz` — 100% anion site exchange

These files are available in the [tutorials/data](https://github.com/bjmorgan/site-analysis/tree/main/tutorials/data) directory of the GitHub repository. The code examples below assume you are running from the repository root directory.

A {download}`Jupyter notebook version <../../../tutorials/residence_times_and_transitions.ipynb>` of this tutorial is available for download.

## Setting up the trajectory

We set up the reference structure and trajectory builder exactly as in the argyrodite tutorial. See {doc}`/tutorials/argyrodite_site_analysis` for a full explanation of these steps.

```python
from pymatgen.io.vasp import Xdatcar
from pymatgen.core import Structure, Lattice
import numpy as np
from site_analysis import TrajectoryBuilder

lattice = Lattice.cubic(a=10.155)

coords = np.array(
    [[0.5,     0.5,     0.5],     # P (type 0) - PS4 tetrahedra
     [0.9,     0.9,     0.6],     # type 1 (Li in reference)
     [0.23,    0.92,    0.09],    # type 2 (Mg in reference)
     [0.25,    0.25,    0.25],    # type 3 (Na in reference)
     [0.15,    0.15,    0.15],    # type 4 (Be in reference)
     [0.0,     0.183,   0.183],   # type 5 (K in reference)
     [0.0,     0.0,     0.0],     # S - anion position (4a site)
     [0.75,    0.25,    0.25],    # S - anion position (4c site)
     [0.11824, 0.11824, 0.38176]] # S - anion position (16e site)
)

reference_structure = Structure.from_spacegroup(
    sg="F-43m",
    lattice=lattice,
    species=['P', 'Li', 'Mg', 'Na', 'Be', 'K', 'S', 'S', 'S'],
    coords=coords) * [2, 2, 2]
```

```python
def build_trajectory(structure):
    builder = TrajectoryBuilder()
    builder.with_reference_structure(reference_structure)
    builder.with_structure(structure)
    builder.with_mobile_species('Li')

    builder.with_polyhedral_sites(
        centre_species='Li', vertex_species='S',
        cutoff=3.0, n_vertices=4, label='type 1')
    builder.with_polyhedral_sites(
        centre_species='Mg', vertex_species='S',
        cutoff=3.0, n_vertices=4, label='type 2')
    builder.with_polyhedral_sites(
        centre_species='Na', vertex_species='S',
        cutoff=3.0, n_vertices=4, label='type 3')
    builder.with_polyhedral_sites(
        centre_species='Be', vertex_species='S',
        cutoff=3.0, n_vertices=4, label='type 4')
    builder.with_polyhedral_sites(
        centre_species='K', vertex_species='S',
        cutoff=3.0, n_vertices=4, label='type 5')

    builder.with_structure_alignment(align_species='P')
    builder.with_site_mapping(mapping_species=['S', 'Cl'])

    trajectory = builder.build()
    return trajectory
```

We analyse all three disorder levels:

```python
datasets = {
    '0%': 'tutorials/data/Li6PS5Cl_0p_XDATCAR.gz',
    '50%': 'tutorials/data/Li6PS5Cl_50p_XDATCAR.gz',
    '100%': 'tutorials/data/Li6PS5Cl_100p_XDATCAR.gz',
}

trajectories = {}
for label, path in datasets.items():
    md_structures = Xdatcar(path).structures
    traj = build_trajectory(md_structures[0])
    traj.trajectory_from_structures(md_structures, progress=True)
    trajectories[label] = traj
    print(f"{label} disorder: {len(md_structures)} frames")
```

## Computing residence times

The `residence_times()` method on each site returns the lengths of consecutive occupation runs for all atoms that visited that site. For more details, see the {doc}`residence times guide </guides/residence_times>`.

We collect residence times for each site type by grouping sites by label:

```python
from collections import defaultdict

site_labels = ['type 2', 'type 4', 'type 5']

for disorder, traj in trajectories.items():
    residence_by_label = defaultdict(list)
    for site in traj.sites:
        if site.label is not None:
            times = site.residence_times()
            residence_by_label[site.label].extend(times)

    print(f"\n{disorder} disorder:")
    for label in site_labels:
        times = residence_by_label[label]
        if times:
            arr = np.array(times)
            print(f"  {label}: {len(arr)} visits, "
                  f"mean = {arr.mean():.1f}, "
                  f"median = {np.median(arr):.1f}, "
                  f"max = {arr.max()}")
        else:
            print(f"  {label}: no visits")
```

### Effect of filtering

In molecular dynamics trajectories, atoms can briefly leave a site due to thermal fluctuations before returning. The `filter_length` parameter fills short gaps in the occupation sequence before computing run lengths. We demonstrate using the 50% disordered trajectory:

```python
traj_50 = trajectories['50%']

for filter_length in [0, 1, 2]:
    residence_by_label = defaultdict(list)
    for site in traj_50.sites:
        if site.label is not None:
            times = site.residence_times(filter_length=filter_length)
            residence_by_label[site.label].extend(times)

    print(f"\nfilter_length={filter_length}:")
    for label in site_labels:
        times = residence_by_label[label]
        if times:
            arr = np.array(times)
            print(f"  {label}: {len(arr)} visits, mean = {arr.mean():.1f}")
```

### Residence time distributions

```python
import matplotlib.pyplot as plt

disorder_levels = ['0%', '50%', '100%']

fig, axes = plt.subplots(len(site_labels), len(disorder_levels),
                         figsize=(12, 9), sharey='row')

for j, disorder in enumerate(disorder_levels):
    traj = trajectories[disorder]
    residence_by_label = defaultdict(list)
    for site in traj.sites:
        if site.label is not None:
            times = site.residence_times()
            residence_by_label[site.label].extend(times)

    for i, label in enumerate(site_labels):
        ax = axes[i, j]
        times = residence_by_label[label]
        if times:
            ax.hist(times, bins='auto', edgecolor='black', linewidth=0.5)
        if i == 0:
            ax.set_title(f'{disorder} disorder')
        if j == 0:
            ax.set_ylabel(f'{label}\nCount')
        if i == len(site_labels) - 1:
            ax.set_xlabel('Residence time (frames)')

fig.tight_layout()
plt.show()
```

## Transition probabilities

The `transition_probabilities_by_label()` method returns a row-normalised matrix showing the probability of transitioning from one site type to another. Each row sums to 1.0.

Since types 1 and 3 are essentially unoccupied, we focus on the three active site types. We define a helper to print transition tables for a subset of labels:

```python
site_type_keys = ['type 2', 'type 4', 'type 5']

def print_transition_table(table, fmt='.3f'):
    print(f"{'':>8}", end='')
    for key in table.keys:
        print(f"{key:>8}", end='')
    print()
    for from_key in table.keys:
        print(f"{from_key:>8}", end='')
        for to_key in table.keys:
            print(f"{table.get(from_key, to_key):{fmt}}", end='')
        print()
```

```python
for disorder, traj in trajectories.items():
    probs = traj.transition_probabilities_by_label().filter(site_type_keys)

    print(f"\n{disorder} disorder — transition probabilities:")
    print_transition_table(probs)
```

The underlying counts are available via `transition_counts_by_label()`:

```python
for disorder, traj in trajectories.items():
    counts = traj.transition_counts_by_label().filter(site_type_keys)

    print(f"\n{disorder} disorder — transition counts:")
    print_transition_table(counts, fmt='>8')
```

### Visualising the transition matrices

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (disorder, traj) in zip(axes, trajectories.items()):
    probs = traj.transition_probabilities_by_label().filter(site_type_keys)

    im = ax.imshow(probs.matrix, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(range(len(probs.keys)))
    ax.set_xticklabels(probs.keys, rotation=45, ha='right')
    ax.set_yticks(range(len(probs.keys)))
    ax.set_yticklabels(probs.keys)

    ax.set_xlabel('To')
    ax.set_ylabel('From')
    ax.set_title(f'{disorder} disorder')

    for i in range(len(probs.keys)):
        for j in range(len(probs.keys)):
            val = probs.matrix[i, j]
            if val > 0.005:
                colour = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center', color=colour)

fig.colorbar(im, ax=axes, label='Transition probability', shrink=0.8)
fig.tight_layout()
plt.show()
```
