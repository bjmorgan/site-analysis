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
timestep:   0     1     2     3     4     5     6     7
atoms:     [1]   [1]   [1]   []    []    [2]   [2]   [2]
```

- Atom 1 is present for timesteps 0-2: run length = 3
- Atom 2 is present for timesteps 5-7: run length = 3
- Result: `(3, 3)`

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
- It is flanked by occupied frames from the **same atom** on both sides (interior gap), or
- It is at the start or end of the trajectory with one occupied neighbour from the same atom (edge gap)

Gaps between runs of **different** atoms are never filled, regardless of their length.

### Filtering examples

Given atom 1's occupation sequence at a site (`O` = occupied, `-` = unoccupied):

```
O O O - O O O     filter_length=1  ->  O O O O O O O   (gap filled, run = 7)
O O - - O O O     filter_length=1  ->  O O - - O O O   (gap = 2, exceeds filter)
O O - - O O O     filter_length=2  ->  O O O O O O O   (gap filled, run = 7)
- O O O O O O     filter_length=1  ->  O O O O O O O   (edge gap filled, run = 7)
- - O O O O O     filter_length=1  ->  - - O O O O O   (edge gap = 2, exceeds filter)
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
