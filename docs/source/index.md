# site-analysis

`site-analysis` is a Python module for analysing molecular dynamics simulations of solid-state ion transport, by assigning positions of mobile ions to specific "sites" within the host structure.

The code is built on top of [pymatgen](https://pymatgen.org) and operates on molecular dynamics trajectories represented as lists of pymatgen `Structure` objects. Any trajectory source that can produce pymatgen structures can be used as input.

[![Build Status](https://github.com/bjmorgan/site-analysis/actions/workflows/build.yml/badge.svg)](https://github.com/bjmorgan/site-analysis/actions/workflows/build.yml)
[![PyPI version](https://badge.fury.io/py/site-analysis.svg)](https://badge.fury.io/py/site-analysis)
[![status](https://joss.theoj.org/papers/0a447aeb167964e77c8d381f7d1db89a/status.svg)](https://joss.theoj.org/papers/0a447aeb167964e77c8d381f7d1db89a)

## Site Types

The package offers several approaches for defining sites:

1. **Spherical sites**: Simple spherical volumes defined by a center position and radius
2. **Polyhedral sites**: Sites defined by polyhedra with vertices at specified atomic positions
3. **Voronoi sites**: Sites defined by Voronoi decomposition of the lattice 
4. **Dynamic Voronoi sites**: Sites using Voronoi decomposition but with centers calculated dynamically based on framework atom positions

## Documentation Structure

```{toctree}
:maxdepth: 1
:caption: Getting Started

introduction
installation
quickstart
```

```{toctree}
:maxdepth: 1
:caption: Concepts

concepts/what_is_site_analysis
concepts/sites
concepts/site_collections
concepts/atoms
concepts/trajectories
concepts/pbc_handling

```

```{toctree}
:maxdepth: 1
:caption: Guides

guides/builders
guides/reference_workflow
guides/spherical_sites
guides/voronoi_sites
guides/dynamic_voronoi_sites
guides/polyhedral_sites
guides/trajectories

```
```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/simple_fcc_example
tutorials/argyrodite_site_definitions

```

```{toctree}
:maxdepth: 1
:caption: Resources

resources/publications

```

```{toctree}
:maxdepth: 2
:caption: API Reference

modules
```

## Code Requirements

- Python 3.10 or later
- Dependencies: pymatgen, numpy, scipy

## Development

For development or to access the latest features:

```bash
# Clone the repository
git clone https://github.com/bjmorgan/site-analysis.git
cd site-analysis

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Reference
* {ref}`genindex`
* {ref}`modindex`
