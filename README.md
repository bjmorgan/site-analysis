# site-analysis

<img src='https://github.com/bjmorgan/site-analysis/blob/main/logo/site-analysis-logo.png' width='180'>

![Build Status](https://github.com/bjmorgan/site-analysis/actions/workflows/build.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/bjmorgan/site-analysis/badge.svg?branch=main)](https://coveralls.io/github/bjmorgan/site-analysis?branch=main)
[![Documentation Status](https://readthedocs.org/projects/site-analysis/badge/?version=latest)](https://site-analysis.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/site-analysis.svg)](https://badge.fury.io/py/site-analysis)
[![status](https://joss.theoj.org/papers/0a447aeb167964e77c8d381f7d1db89a/status.svg)](https://joss.theoj.org/papers/0a447aeb167964e77c8d381f7d1db89a)

`site-analysis` is a Python module for analysing molecular dynamics simulations of solid-state ion transport, by assigning positions of mobile ions to specific &ldquo;sites&rdquo; within the host structure.

The code is built on top of [`pymatgen`](https://pymatgen.org) and takes VASP XDATCAR files as molecular dynamics trajectory inputs.

The code can use the following definitions for assigning mobile ions to sites:
1. **Spherical cutoff**: Atoms occupy a site if they lie within a spherical cutoff from a fixed position.
2. **Voronoi decomposition**: Atoms are assigned to sites based on a Voronoi decomposition of the lattice into discrete volumes.
3. **Polyhedral decomposition**: Atoms are assigned to sites based on occupation of polyhedra defined by the instantaneous positions of lattice atoms.
4. **Dynamic Voronoi sites**: Sites using Voronoi decomposition but with centres calculated dynamically based on framework atom positions.

## Installation

### Standard Installation

```bash
pip install site-analysis
```

### Development Installation

For development or to access the latest features:

```bash
# Clone the repository
git clone https://github.com/bjmorgan/site-analysis.git
cd site-analysis

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Documentation

Complete documentation, including tutorials, examples, and API reference, is available at [Read the Docs](https://site-analysis.readthedocs.io/en/latest/).

## Testing

Automated testing of the latest build happens on [GitHub Actions](https://github.com/bjmorgan/site-analysis/actions).

To run tests locally:

```bash
# Using pytest (recommended)
pytest

# Using unittest
python -m unittest discover
```

The code requires Python 3.10 or above.
