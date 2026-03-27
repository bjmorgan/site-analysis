# Tutorials

Jupyter notebook tutorials for `site-analysis`.

## Available tutorials

- **[Comparing Site Definitions](comparing_site_definitions.ipynb)** — Track ion migration through an FCC lattice using spherical, Voronoi, and polyhedral site definitions, and compare the tradeoffs between each approach. No external data required.

- **[Analysing Complex Crystal Structures](argyrodite_site_analysis.ipynb)** — Analyse lithium site occupations in argyrodite Li6PS5Cl with multiple crystallographically distinct site types and chemical disorder. Requires trajectory data from `tutorials/data/`.

- **[Residence Times and Transition Probabilities](residence_times_and_transitions.ipynb)** — Compute residence times and transition probabilities from the argyrodite trajectories, comparing how anion disorder affects ion dynamics. Requires the same trajectory data as the argyrodite tutorial.

## Running the tutorials

Install the required packages:

```bash
pip install site-analysis matplotlib
```

Then launch Jupyter from the `tutorials/` directory:

```bash
cd tutorials
jupyter notebook
```

## Tutorial data

The second and third tutorials require MD trajectory data files located in the `data/` subdirectory. These are included in the repository.
