# Example Data

This directory contains example data files for use with the `site-analysis` package.

## Files

### XDATCAR

A minimal VASP XDATCAR trajectory file for demonstrating the quickstart example.

- **Structure**: Simple cubic O lattice (8 atoms) with one interstitial Li ion
- **Cell**: 8 Angstrom cubic cell
- **Composition**: O8 Li1
- **Frames**: 5 frames showing Li movement from [0.25, 0.25, 0.25] to [0.65, 0.25, 0.25]

This is a pedagogical example structure (not a real material) designed to demonstrate the basic workflow of site-analysis. The Li ion moves between interstitial sites, which can be tracked using spherical sites centred at positions like [0.25, 0.25, 0.25] and [0.75, 0.25, 0.25].

For real-world usage, you would use your own VASP trajectory data and define sites appropriate for your material system.
