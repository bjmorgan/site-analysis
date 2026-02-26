# Introduction to site-analysis

## Overview

`site-analysis` is a Python module for analyzing molecular dynamics simulations of solid-state ionic transport by assigning mobile ions to discrete "sites" within host structures.

The package is built on top of [pymatgen](https://pymatgen.org/) and operates on molecular dynamics trajectories represented as lists of pymatgen `Structure` objects. Any trajectory source that can produce pymatgen structures can be used as input.

## Context

Ionic transport in solid materials underlies the function of many devices including batteries, fuel cells, and sensors. While molecular dynamics simulations can model this transport, extracting meaningful mechanistic information from raw trajectory data presents challenges.

`site-analysis` addresses this by discretizing atomic trajectories through projection onto defined sites. This approach converts continuous atomic motion into a representation based on site occupation and transitions. By mapping mobile ions to specific structural sites, the analysis becomes simpler and more quantitative.

For systems where ions move between local energy minima, this discretization captures the essential physics while reducing complexity. The package defines these sites as bounded volumes within the simulation cell, then assigns mobile ions to them at each timestep.

## Key Concepts

### Sites and Site Collections

A **site** represents a bounded volume within a crystal structure that can contain zero, one, or multiple mobile ions. 

`site-analysis` offers several types of sites:

1. **Spherical sites**: Simple spherical volumes defined by a center position and radius
2. **Polyhedral sites**: Sites defined by polyhedra with vertices at specified atomic positions
3. **Voronoi sites**: Sites defined by Voronoi decomposition of the lattice
4. **Dynamic Voronoi sites**: Sites using Voronoi decomposition but with centers calculated dynamically based on framework atom positions as reference

A **site collection** manages a group of related sites and handles the assignment of atoms to these sites.

Each site can be occupied by zero, one, or more mobile ions. Similarly, each atom may be assigned to zero, one, or multiple sites depending on how the sites are defined.

### Atoms and Trajectories

The package tracks mobile ions throughout a simulation, recording:
- Site occupation at each timestep
- Sequence of visited sites (trajectory)
- Statistical information about occupations and transitions

### Applications

This discretized representation enables various quantitative analyses:
- Time-averaged site-occupation probabilities
- Sequential site visitation patterns
- Temporal and spatial correlations between ion movements
