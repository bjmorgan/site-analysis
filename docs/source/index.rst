site-analysis
=============

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   installation
   context
   tutorials
   modules

`site_analysis` is a Python module for anakysing molecular dynamics simulations of solid-state ionic transport, by assigning mobile ions to discrete "sites" within host structures.

The code is built on top of `pymatgen`_  and takes VASP `XDATCAR` files as molecular dynamics trajectory inputs.

The code currently can use on of three schemes for assigning mobile ions to discrete sites:

1. Spherical cutoff: Atoms occupy a site if they lie within a cutoff radis from a fixed position.
2. Voronoi decomposition: Atoms are assigned to sites based on a Voronoi decomposition of the lattice into discrete volumes.
3. Polyhedral decomposition: Atoms are assigned to sites based on occupation of polyhedra defined by the instantaneous positions of lattice atoms.

.. _pymatgen: https://pymatgen.org

Reference
=========

* :ref:`genindex`
* :ref:`modindex`
