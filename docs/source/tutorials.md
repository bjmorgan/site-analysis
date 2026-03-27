# Tutorials

Executable Jupyter notebook tutorials are available in the [`tutorials/`](https://github.com/bjmorgan/site-analysis/tree/main/tutorials) directory of the GitHub repository. The tutorials include example data and are designed to be run locally.

To get the tutorials, clone the repository:

```bash
git clone https://github.com/bjmorgan/site-analysis.git
cd site-analysis/tutorials
jupyter notebook
```

The tutorials are designed to be worked through in order:

**Comparing Site Definitions** — [`comparing_site_definitions.ipynb`](https://github.com/bjmorgan/site-analysis/blob/main/tutorials/comparing_site_definitions.ipynb)

An introductory tutorial that demonstrates the three site definition methods — spherical, Voronoi, and polyhedral — using a synthetic lithium-ion migration trajectory in an FCC lattice. This tutorial uses generated data, so no external files are needed.

You will learn how to:

- Set up a `TrajectoryBuilder` and define sites
- Compare how different site types assign ions during migration events
- Understand the tradeoffs between site definitions

**Argyrodite Site Analysis** — [`argyrodite_site_analysis.ipynb`](https://github.com/bjmorgan/site-analysis/blob/main/tutorials/argyrodite_site_analysis.ipynb)

A realistic example analysing lithium-ion site occupations in Li6PS5Cl argyrodite solid electrolytes with varying degrees of anion disorder. Uses MD trajectory data included in the repository at `tutorials/data/`.

You will learn how to:

- Define multiple crystallographically distinct site types using a reference structure with dummy atoms
- Use structure alignment and species mapping for structures with different compositions
- Analyse how site occupations change with anion disorder

**Residence Times and Transition Probabilities** — [`residence_times_and_transitions.ipynb`](https://github.com/bjmorgan/site-analysis/blob/main/tutorials/residence_times_and_transitions.ipynb)

Builds on the argyrodite tutorial to compute residence times and transition probabilities. Uses the same MD trajectory data.

You will learn how to:

- Compute and filter residence times for different site types
- Visualise residence time distributions
- Calculate and visualise transition probability matrices
