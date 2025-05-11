# site-analysis

`site-analysis` is a Python module for analysing molecular dynamics simulations of solid-state ion transport, by assigning positions of mobile ions to specific "sites" within the host structure.

The code is built on top of [pymatgen](https://pymatgen.org) and processes molecular dynamics trajectory data, with particular support for VASP XDATCAR format files.

[![Build Status](https://github.com/bjmorgan/site-analysis/actions/workflows/build.yml/badge.svg)](https://github.com/bjmorgan/site-analysis/actions/workflows/build.yml)
[![Coverage Status](https://coveralls.io/repos/github/bjmorgan/site-analysis/badge.svg?branch=main)](https://coveralls.io/github/bjmorgan/site-analysis?branch=main)
[![PyPI version](https://badge.fury.io/py/site-analysis.svg)](https://badge.fury.io/py/site-analysis)

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

context
concepts/sites
concepts/site_collections
concepts/atoms
concepts/trajectories

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

```

```{toctree}
:maxdepth: 1
:caption: Tutorials & Examples

examples/shortest_distance_sites
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
