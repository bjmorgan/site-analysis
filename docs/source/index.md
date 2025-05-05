# site-analysis

`site-analysis` is a Python module for analysing molecular dynamics simulations of solid-state ionic transport by assigning mobile ions to discrete "sites" within host structures.

The code is built on top of [pymatgen](https://pymatgen.org/) and takes VASP `XDATCAR` files as molecular dynamics trajectory inputs.

The code provides four schemes for assigning mobile ions to discrete sites:

1. **Spherical cutoff**: Atoms occupy a site if they lie within a cutoff radius from a fixed position.
2. **Polyhedral decomposition**: Atoms are assigned to sites based on occupation of polyhedra defined by the instantaneous positions of sets of host framework atoms.
3. **Voronoi decomposition**: Atoms are assigned to sites based on a Voronoi decomposition of the structure into discrete volumes, using fixed Voronoi centres.
4. **Dynamic Voronoi**: Atoms are assigned to sites based on a Vornoi decomposition, with the Voronoi centres defined by the instantaneous positions of sets of host framework atoms.

```{toctree}
:hidden:
:maxdepth: 2

introduction
installation
examples/2D_lattice_examples
tutorials
modules
```

## Reference

* {ref}`genindex`
* {ref}`modindex`
