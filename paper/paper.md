---
title: 'site-analysis: A Python package for crystallographic site-projection analysis of molecular dynamics trajectories'
tags:
  - Python
  - molecular dynamics
  - ionic conductors
  - diffusion
  - crystallography
  - solid-state ionics
authors:
  - name: Benjamin J. Morgan
    affiliation: "1, 2"
    orcid: 0000-0002-3056-8233
affiliations:
  - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, United Kingdom
    index: 1
  - name: The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
    index: 2
date: 11 May 2025
bibliography: paper.bib
---

# Summary

Ionic transport in crystalline solids can often be considered to proceed via discrete "jumps", where mobile ions move between potential energy minima corresponding to specific crystallographic sites. `site-analysis` is a Python package that transforms continuous molecular dynamics trajectories into discrete site-occupation sequences, enabling quantitative analysis of ionic diffusion mechanisms. It provides four site-definition methods—spherical, polyhedral, Voronoi, and dynamic Voronoi—each offering different trade-offs between simplicity and crystallographic fidelity. The package provides a reference-based workflow that automates coordination-based site generation by identifying coordination environments in ideal structures and mapping these to thermally distorted configurations. Integration with pymatgen provides compatibility with existing computational materials science workflows.

# Statement of need

Molecular dynamics simulations are widely used to study ionic transport in battery materials, solid electrolytes, and ceramic ion conductors. While these simulations provide continuous atomic trajectories, ionic diffusion in crystalline solids is more accurately described as a sequence of discrete jumps between potential energy minima. These minima typically correspond to specific arrangements of mobile ions within crystallographic sites. Analysing these simulations requires methods that can spatially discretise the continuous trajectories onto sets of discrete sites.

General-purpose trajectory analysis packages lack specialised functionality for crystallographic site analysis in periodic systems. The most common approach uses spherical site definitions, which have fundamental limitations: incomplete spatial coverage leading to unassigned atoms during transitions, arbitrary radius parameters requiring manual optimisation, and poor representation of anisotropic coordination environments.

More sophisticated coordination-based approaches—such as polyhedral sites defined by framework atoms or dynamic Voronoi tessellations—provide better crystallographic representation but typically require custom code for each material system. This creates a significant barrier to adoption, limiting most analyses to simpler but less accurate spherical projections.

`site-analysis` addresses these challenges through a unified framework implementing multiple site-definition paradigms. By projecting continuous coordinates onto discrete sites, it provides two complementary perspectives: tracking which sites each atom visits over time, and monitoring which atoms occupy each site. This enables calculation of time-averaged site-occupation probabilities, analysis of sequential site visitation patterns, and quantification of temporal and spatial correlations between ion movements.

The package includes a reference-based workflow that automates the generation of coordination-based sites, eliminating the need for manual site specification. Users provide an ideal reference structure and specify coordination criteria; the software identifies all matching environments and automatically maps them to target structures, with sites adapting to thermal distortions while preserving chemical identity.

The package employs a builder pattern interface that provides method chaining for intuitive configuration while ensuring parameter validation. 
This enables researchers to apply sophisticated site definitions without developing custom analysis code for each system. The software supports standard molecular dynamics formats including VASP XDATCAR files and integrates with the pymatgen ecosystem, facilitating incorporation into existing computational workflows. Full documentation and tutorials are available at https://site-analysis.readthedocs.io.

The software has previously been used to analyse ion-transport mechanisms in lithium-ion and fluoride-ion solid electrolytes [@Morgan2021; @Mercadier2023; @Krenzer2023; @Hu2025]. The package was initially written to reproduce the functionality of an earlier Fortran code that used dynamically defined coordination polyhedra, which was used to study ion-transport and defect distributions in a range of solid electrolyte materials [@Burbano2016; @Morgan2014; @Morgan2014b; @Morgan2012; @Morgan2011]. This code, in turn, was motivated by earlier studies that used projections onto tetrahedral sites [@Castiglione1999; @Castiglione2001; @Marrocchelli2009].

# Acknowledgements

B.J.M. acknowledges support from the Royal Society (UF130329 and URF/R/191006).

# References
