# site-analysis

<img src='https://github.com/bjmorgan/site-analysis/blob/master/logo/site-analysis-logo.png' width='180'>

![Build Status](https://github.com/bjmorgan/site-analysis/actions/workflows/build.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/bjmorgan/site-analysis/badge.svg?branch=master)](https://coveralls.io/github/bjmorgan/site-analysis?branch=master)
[![Documentation Status](https://readthedocs.org/projects/site-analysis/badge/?version=latest)](https://site-analysis.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/site-analysis.svg)](https://badge.fury.io/py/site-analysis)

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
Documentation and tutorials live at [Read the Docs](https://site-analysis.readthedocs.io/en/latest/).

Example notebooks are also available of [GitHub](https://github.com/bjmorgan/site-analysis/examples).

## Tests
Automated testing of the latest build happens [here](https://travis-ci.org/github/bjmorgan/site-analysis).

Manual tests can be run using 
```
python3 -m unittest discover
```
The code has been tested with Python versions 3.6 and above.
