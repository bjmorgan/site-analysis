# site-analysis

[![Build Status](https://travis-ci.org/bjmorgan/site-analysis.svg?branch=master)](https://travis-ci.org/bjmorgan/site-analysis)
[![Test Coverage](https://api.codeclimate.com/v1/badges/cb871e86f11b715efad6/test_coverage)](https://codeclimate.com/github/bjmorgan/site-analysis/test_coverage)

`site-analysis` is a Python module for analysing molecular dynamics simulations of solid-state ion transport, by assigning positions of mobile ions to specific &ldquo;sites&rdquo; within the host structure.

The code is built on top of [`pymatgen`](https://pymatgen.org) and takes VASP XDATCAR files as molecular dynamics trajectory inputs.

The code can use the following definitions for assigning mobile ions to sites:
1. Spherical cutoff: Atoms occupy a site if they lie within a spherical cutoff from a fixed position.
2. Voronoi decomposition: Atoms are assigned to sites based on a Voronoi decomposition of the lattice into discrete volumes.
3. Polyhedral decomposition: Atoms are assigned to sites based on occupation of polyhedra defined by the instantaneous positions of lattice atoms.

## Installation

From PyPI:
```
pip install site-analysis
```

Or clone the latest development version from [GitHub](https://github.com/bjmorgan/site-analysis) and install
```
 git clone git@github.com:bjmorgan/site-analysis.git
cd site-analysis
python setup.py install
```

Alternatively you can install the latest version using `pip`, direct from GitHub, e.g.
```
pip3 install git+https://github.com/bjmorgan/site-analysis.git
```

## Documentation
Documentation and tutorials live at [Read the Docs](https://site-analysis.readthedocs.io/en/latest/)
