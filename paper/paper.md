---
title: 'site-analysis: A Python package for site-projection analysis of molecular dynamics trajectories' 

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
  - index: 1
    name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, United Kingdom
  - index: 2
    name: The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
date: 23 May 2025 
bibliography: paper.bib
---

# Summary

Understanding ionic transport in crystalline solids is important for developing materials for batteries, fuel cells, and other electrochemical devices. Molecular dynamics (MD) simulations provide detailed information about the microscopic dynamics of mobile ions, but extracting mechanistic insights from these complex data can be challenging. In crystalline solids, ionic diffusion can often be considered to proceed by discrete jumps between "sites"—persistent local minima on the potential energy surface. `site-analysis` is a Python package that processes MD trajectories to produce time-resolved site-occupation data that provide a coarse-grained description of ion transport. The package implements four site-definition methods—spherical, polyhedral, Voronoi, and dynamic Voronoi—allowing users to choose the approach best suited to their requirements. For polyhedral and dynamic Voronoi sites, `site-analysis` implements a reference-based workflow that automates site generation from coordination environment specifications, allowing users to analyse trajectories without manually defining site geometries. `site-analysis` integrates with existing computational materials science workflows through the pymatgen ecosystem and provides both atom-centric and site-centric representations of ion dynamics as simple Python data structures.

# Statement of need

Molecular dynamics (MD) simulations are widely used to study ionic transport in battery materials, solid electrolytes, and ceramic ion conductors [@LandgrafEtAl_ChemRxiv2024; @Mercadier2023; @Morgan2021; @PoletayevEtAl_NatureMater2022]. 
While these simulations produce detailed atomic trajectories, extracting mechanistic understanding from raw trajectory data can be challenging. A productive approach is to seek a coarse-grained description that retains only the essential features of ion transport. In many crystalline solids, ionic transport can be understood as a series of discrete "jumps" between crystallographic sites—persistent local minima on the potential energy surface where mobile ions reside between jumps. This site-hopping perspective motivates a coarse-graining strategy: transform continuous trajectories into sequences of site occupations, capturing only motion between sites while discarding vibrational motion within sites.

Transforming MD trajectories into site-occupation sequences requires both defining what constitutes a "site" and implementing algorithms to track site occupancy throughout the simulation. While the concept is straightforward, the practical implementation presents challenges: sites must be defined and occupancy criteria must handle edge cases. Without accessible software tools, each research group must independently implement these methods, leading to duplicated effort and potential inconsistencies in analysis approaches.

`site-analysis` provides a Python framework that implements multiple site-definition methods within a consistent interface. For each analysis timestep in an MD trajectory, the package determines which site (if any) each mobile ion occupies. This information can be accessed in two ways: atom trajectories, which record the sequence of sites visited by each ion (e.g., ion 1 visits sites `[0, 0, 1, 1, 2, ...]`), and site trajectories, which record which ions occupy each site over time. These outputs are represented using simple Python lists, which allows easy integration into downstream analysis for computing, for example, site-occupation probabilities, site--site transition frequencies, or details of correlated ion movements.

The package provides four site-definition methods—spherical, polyhedral, Voronoi, and dynamic Voronoi:

- **Spherical sites** are defined as spheres of fixed radius centred on crystallographic positions, providing a simple and widely-used approach.
- **Voronoi sites** divide all space into regions where each point belongs to its nearest site centre. This ensures complete spatial coverage with no gaps or overlaps.
- **Dynamic Voronoi sites** use Voronoi centres recalculated each frame based on instantaneous positions of coordinating host-framework atoms, accounting for thermal distortions while maintaining complete spatial coverage.
- **Polyhedral sites** are defined by coordination polyhedra formed by host-framework atoms. The vertex positions update according to the instantaneous positions of the coordinating host-framework atoms, allowing these sites to track changing local environments.

To assist with analyses using dynamic Voronoi and polyhedral sites, the package provides a reference-based workflow that automates site generation: users provide an ideal reference structure and specify coordination criteria (e.g., "Li ions coordinated by 4 O atoms"), and the software identifies all matching environments and maps them to target structures. This automated approach allows analyses using these dynamic sites without requiring the user to manually identify all relevant coordination environments.

The package uses a builder pattern interface for configuration and parameter validation. Analyses can be set up and executed with a few lines of code, with results returned as Python lists that integrate simply into downstream analysis workflows. The software natively supports VASP XDATCAR files and integrates with the pymatgen ecosystem.

The materials modelling community has developed various tools for site-based trajectory analysis, including `pymatgen-analysis-diffusion` [@pymatgen-analysis-diffusion; @DengEtAl_ChemMater2016], `SITATOR` [@sitator; @KahleEtAl_PhysRevMater2019], `IonDiff` [@LopezEtAl_JOpenSourceSoftw2024], and `gemdat` [@GEMDAT]. These packages each implement specific schemes for defining sites and assigning occupations, and are often tightly integrated with downstream workflows for particular analysis tasks. `site-analysis` complements these tools by focusing on generality: it provides multiple site-definition methods within a consistent interface and produces output using simple data structures that can feed into any downstream analysis. To our knowledge, `site-analysis` is unique in implementing geometric sites that dynamically update during the simulation (polyhedral sites and dynamic Voronoi sites) in a publicly available package. We note that the use of dynamic polyhedral sites is particularly valuable for materials with close-packed host-framework structures, where mobile ion coordination environments provide a natural and intuitive basis for describing transport mechanisms [@Burbano2016; @Morgan2021; @Mercadier2023].

`site-analysis` enables researchers to apply site-projection analysis to characterise and quantify mechanisms of ion transport in solid electrolytes. The software provides both atom-centric views (tracking which sites each ion visits) and site-centric views (recording which ions occupy each site over time), supporting different analytical perspectives on the same transport processes. This dual representation, combined with multiple site-definition methods including dynamically-updating geometric sites, provides a flexible toolkit for analysing ionic transport mechanisms across diverse materials.

The software has been used to analyse ion-transport mechanisms in lithium-ion and fluoride-ion solid electrolytes [@Morgan2021; @Mercadier2023; @Krenzer2023; @Hu2025]. The package reproduces functionality from an earlier Fortran code that used dynamically defined coordination polyhedra to study ion-transport and defect distributions [@Burbano2016; @Morgan2014; @Morgan2014b; @Morgan2012; @Morgan2011], which was itself motivated by earlier studies using projections onto tetrahedral sites [@Castiglione1999; @Castiglione2001; @Marrocchelli2009]. Documentation and tutorials are available at [https://site-analysis.readthedocs.io](https://site-analysis.readthedocs.io).

# Acknowledgements

B.J.M. acknowledges support from the Royal Society (UF130329 and URF/R/191006) and the Faraday Institution (FIRG016).

# Bibliography
