# Transition Tables

Transition tables summarise how atoms move between sites during a trajectory. The `Trajectory` class provides methods that return `TransitionTable` objects containing either raw hop counts or row-normalised probabilities.

## Basic Usage

After running a trajectory analysis, use the `_by_site` or `_by_label` methods:

```python
# Counts and probabilities keyed by site index
counts = trajectory.transition_counts_by_site()
probs = trajectory.transition_probabilities_by_site()

# Counts and probabilities keyed by site label
counts = trajectory.transition_counts_by_label()
probs = trajectory.transition_probabilities_by_label()
```

Both return a `TransitionTable` — a labelled square matrix with convenient access patterns.

## Accessing the Data

`TransitionTable` provides several ways to access the data:

```python
probs = trajectory.transition_probabilities_by_label()

# Raw numpy array (read-only)
probs.matrix

# Row and column labels
probs.keys

# Single value lookup
probs.get("A", "B")

# Convert to a dict-of-dicts
probs.to_dict()
```

## Filtering to a Subset of Keys

Use `.filter()` to extract a sub-table for a subset of keys. This is useful when some sites or labels are uninteresting (e.g. rarely occupied site types):

```python
# Full 5x5 table
probs = trajectory.transition_probabilities_by_label()

# 3x3 sub-table for the active site types only
active = probs.filter(["type 2", "type 4", "type 5"])
```

The filtered table preserves the order of the keys you provide. Note that filtering does not re-normalise probabilities — the values are extracted as-is from the original table. If you need probabilities that sum to 1.0 over the filtered subset, filter the counts table first and then compute probabilities:

```python
counts = trajectory.transition_counts_by_label().filter(["type 2", "type 4", "type 5"])
probs = trajectory._normalise_counts(counts)
```

## Reordering Keys

Use `.reorder()` to change the row and column order without changing the data:

```python
probs = trajectory.transition_probabilities_by_label()
reordered = probs.reorder(["C", "B", "A"])
```

This requires all original keys to be present — it is a permutation, not a filter.

## Custom Key Ordering

The trajectory methods accept an optional `keys` parameter to control the ordering of the returned table:

```python
# Default: keys are sorted
probs = trajectory.transition_probabilities_by_label()  # keys sorted alphabetically

# Custom ordering
probs = trajectory.transition_probabilities_by_label(keys=["C", "A", "B"])
```

## Counts vs Probabilities

The `transition_counts_*` methods return raw hop counts: how many times an atom transitioned from one site to another during the trajectory. The `transition_probabilities_*` methods row-normalise these counts so that each row sums to 1.0, giving the empirical fraction of hops from each source site that land on each destination.

These are **not** Markov chain transition matrices. They represent "of all hops out of site A, what fraction land on site B?", not "at any given timestep, what is the probability of moving from A to B?". The latter would require accounting for timesteps where the atom stays put (the diagonal), which is a different calculation.

Rows with no outgoing transitions remain as all zeros in both the counts and probabilities tables.

## Unlabelled Sites

When using `_by_label` methods, sites without labels are excluded from the table. If any transitions involving unlabelled sites are dropped, a warning is emitted with the number of excluded transitions.
