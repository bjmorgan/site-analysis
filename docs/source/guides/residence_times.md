# Residence Time Analysis

Residence times quantify how long individual atoms remain at a particular site before leaving. The `Site.residence_times()` method computes these as run lengths: the number of consecutive timesteps an atom continuously occupies a site.

## Basic Usage

After running a trajectory analysis, call `residence_times()` on any site:

```python
for site in trajectory.sites:
    times = site.residence_times()
    if times:
        label = site.label or f"Site {site.index}"
        print(f"{label}: mean residence = {sum(times)/len(times):.1f} timesteps")
```

The method returns a flat tuple of integers, where each integer is the length of one continuous occupation by one atom. For example, if atom 1 occupies a site for 5 timesteps, leaves, then atom 2 occupies it for 3 timesteps, the result is `(5, 3)`.

## How It Works

For each atom that visits the site at any point during the trajectory:

1. A binary occupied/not-occupied sequence is built across all timesteps
2. Consecutive runs of occupied frames are identified
3. The length of each run is recorded

Run lengths from all atoms are collected into a single flat tuple.

### Example

Consider a site with the following trajectory (atom indices per timestep):

```
timestep:   0     1     2     3     4     5     6     7     8     9
atoms:      []   [1]   [1]   [1]   []    []    [2]   [2]   [2]   []
```

- Atom 1 is present for timesteps 1-3: run length = 3
- Atom 2 is present for timesteps 6-8: run length = 3
- Result: `(3, 3)`

## Edge Run Exclusion

By default, runs that touch the first or last timestep of the trajectory are excluded. These runs are truncated by the trajectory boundary and underestimate the true residence time, which would bias the distribution towards shorter values.

```
timestep:   0     1     2     3     4     5     6     7     8
atoms:     [1]   [1]   []    [1]   [1]   [1]   []    [1]   [1]
                       edge --|-- interior --|-- edge
```

In this example:
- The run at timesteps 0-1 touches the start: **excluded** (we don't know when atom 1 actually arrived)
- The run at timesteps 3-5 is fully interior: **included** (run length = 3)
- The run at timesteps 7-8 touches the end: **excluded** (we don't know when atom 1 would have left)

Result: `(3,)`

To include all runs regardless of whether they touch the trajectory boundary:

```python
site.residence_times(include_edge_runs=True)  # (2, 3, 2)
```

## Filtering Short Gaps

In molecular dynamics simulations, atoms may briefly leave a site due to thermal fluctuations before returning. The `filter_length` parameter smooths out these short excursions before computing run lengths.

```python
# Without filtering: brief excursion creates two short runs
site.residence_times()  # e.g. (4, 6)

# With filtering: single-frame gaps are filled
site.residence_times(filter_length=1)  # e.g. (11,)
```

A gap is filled when:
- It is `filter_length` or fewer consecutive unoccupied frames
- It is flanked by occupied frames from the **same atom** on both sides (interior gap)

Gaps at the trajectory edges are never filled, as this would always bias towards longer occupation times. Gaps between runs of **different** atoms are also never filled.

### Filtering examples

Given atom 1's occupation sequence at a site (`O` = occupied, `-` = unoccupied):

```
O O O - O O O     filter_length=1  ->  O O O O O O O   (interior gap filled)
O O - - O O O     filter_length=1  ->  O O - - O O O   (gap = 2, exceeds filter)
O O - - O O O     filter_length=2  ->  O O O O O O O   (interior gap filled)
- O O O O O O     filter_length=1  ->  - O O O O O O   (edge gap, not filled)
```

## Statistical Analysis

The returned tuple of run lengths can be used for further statistical analysis:

```python
import numpy as np

times = site.residence_times(filter_length=1)

if times:
    times_array = np.array(times)
    print(f"Number of visits: {len(times_array)}")
    print(f"Mean residence time: {times_array.mean():.1f} timesteps")
    print(f"Median residence time: {np.median(times_array):.1f} timesteps")
    print(f"Max residence time: {times_array.max()} timesteps")
```

To convert from timesteps to physical time units, multiply by the time interval between frames in your simulation.
