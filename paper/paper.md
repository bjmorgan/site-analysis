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

Understanding ionic transport in crystalline solids is important for developing materials for batteries, fuel cells, and other electrochemical devices. Molecular dynamics (MD) simulations provide complete information about the microscopic dynamics of mobile ions, but extracting mechanistic insights from these complex data can be challenging. In crystalline solids, ionic diffusion can often be considered to be effected by discrete jumps between "sites"—persistent local minima on the potential energy surface. `site-analysis` is a Python package that processes MD trajectories to produce discrete time-resolved site-occupation data that provide a coarse-grained description of ion transport behaviour. The package implements four site-definition methods—spherical, polyhedral, Voronoi, and dynamic Voronoi—allowing users to choose the approach best suited to their requirements. For polyhedral and dynamic Voronoi sites `site-analysis` implements a reference-based workflow that automates site generation from coordination environment specifications, allowing users to analyse trajectories using these site types without manually defining site geometries. `site-analysis` integrates with existing computational materials science workflows through the pymatgen ecosystem and provides both atom-centric and site-centric representations of ion dynamics as simple Python data structures.

# Statement of need

Molecular dynamics simulations are widely used to study ionic transport in battery materials, solid electrolytes, and ceramic ion conductors [@LandgrafEtAl_ChemRxiv2024; @Mercadier2023; @Morgan2021; @PoletayevEtAl_NatureMater2022]. These simulations, in principle, provide complete descriptions of atomic-scale dynamics, but extracting mechanistic insights from these data often presents challenges. In many crystalline solids, ionic transport is well described as occurring through a series of discrete "jumps" between crystallographic sites—persistent local minima on the potential energy surface where mobile ions reside between diffusive events. This site-hopping model provides a framework for understanding and quantifying transport mechanisms by separating ionic motion between sites from non-diffusive motion within sites. Extracting a site-projected description of ion dynamics from MD data requires methods that can transform continuous ion trajectories into discrete time-resolved sequences of site occupations. This coarse-graining process filters out vibrational motion and reveals the events that contribute to macroscopic transport, enabling analysis and characterisation of specific details of the transport mechanism, including identification of rate-limiting steps, preferred pathways, and correlated ion movements.

Implementing site-projection analysis requires defining spatial regions (sites) and assigning mobile ions to these sites at each timestep. Once site positions are known—whether from crystallographic knowledge or data-driven methods [@ChenEtAl_SciRep2017; @KahleEtAl_PhysRevMater2019; @LopezEtAl_JAmChemSoc2024]—various geometric definitions can be applied; for example, spherical regions with distance cutoffs [@deKlerkEtAl_ChemMater2016; @deKlerkAndWagemaker_ChemMater2016], space-filling Voronoi tessellations [@SicoloEtAl_SolidStateIonics2018; @ClarkeEtAl_ACSApplEnergyMater2021; @Krenzer2023; @RichardsEtAl_NatCommun2016], or coordination polyhedra [@Morgan2021; @Burbano2016; @KozinskyEtAl_PhysRevLett2016; @ZouEtAl_AdvFunctMater2021; @ZhangEtAl_AdvEnergyMater2019]. Each approach involves implementation challenges and parameter choices that can affect the resulting analysis. Without accessible tools, researchers must develop custom code for their specific systems, particularly when using complex site definitions that account for dynamical distortions of the host framework structure or non-spherical site geometries.

`site-analysis` provides a Python framework that implements multiple site-definition methods within a consistent interface. For each analysis timestep in an MD trajectory, the package determines which site (if any) each mobile ion occupies. This information can be accessed in two ways: atom trajectories, which record the sequence of sites visited by each ion (e.g., ion 1 visits sites `[0, 0, 1, 1, 2, ...]`), and site trajectories, which record which ions occupy each site over time. These outputs are represented using simple Python lists, which allows easy downstream analysis to compute, for example, site-occupation probabilities, site--site transition frequencies, and quantification of correlated ion movements.

The package provides four site-definition methods—spherical, polyhedral, Voronoi, and dynamic Voronoi:

- **Spherical sites** are conceptually simple and widely used in the literature.
- **Voronoi sites** partition space based on proximity to site centres, ensuring complete spatial coverage—every point in space belongs to exactly one site.
- **Dynamic Voronoi sites** use Voronoi partitioning to assign atoms to sites, but use Voronoi centres that are dynamically calculated each frame according to the instantaneous positions of coordinating host-framework atoms; this accounts for thermal distortions and host framework dynamics whilst maintaining spatial partitioning.
- **Polyhedral sites** define sites as coordination polyhedra formed by host-framework atoms, directly reflecting local coordination environments. Like dynamic Voronoi sites, polyhedral sites update their geometry based on the instantaneous positions of the vertex host-framework atoms.

To assist with analyses using dynamic Voronoi and polyhedral sites, the package provides a reference-based workflow that automates site generation: users provide an ideal reference structure and specify coordination criteria (e.g., "Li ions coordinated by 4 O atoms"), and the software identifies all matching environments and maps them to target structures. This automated approach allows analyses using these dynamic sites without requiring the user to pre-identify all relevant coordination environments.

The package uses a builder pattern interface that provides method chaining for configuration and ensures parameter validation. Analyses can be set up and executed with a few lines of code, with results returned as Python lists that integrate simply into downstream analysis workflows. The software natively supports VASP XDATCAR files and integrates with the pymatgen ecosystem.

The materials modelling community has developed various tools for site-based trajectory analysis, including `pymatgen-analysis-diffusion` [@pymatgen-analysis-diffusion; @DengEtAl_ChemMater2016], `SITATOR` [@sitator; @KahleEtAl_PhysRevMater2019], `IonDiff` [@LopezEtAl_JOpenSourceSoftw2024], and `gemdat` [@GEMDAT]. These packages each implement specific schemes for defining sites and assigning occupations, and are often tightly integrated with downstream workflows for particular analysis tasks. `site-analysis` complements these tools by focusing on generality: it provides multiple site-definition methods within a consistent interface and produces output using simple data structures that can feed into any downstream analysis. To our knowledge, `site-analysis` is unique in implementing geometric sites that dynamically update during the simulation (polyhedral sites and dynamic Voronoi sites) in a publicly available package. We note that the use of dynamic polyhedral sites is particularly valuable for materials with close-packed host-framework structures, where mobile ion coordination environments provide a natural and intuitive basis for describing transport mechanisms [@Burbano2016; @Morgan2021; @Mercadier2023].

By transforming continuous MD trajectories into discrete site-occupation sequences, `site-analysis` enables researchers to apply site-projection coarse-graining to characterise and quantify mechanisms of ion transport in solid electrolytes. The software provides both atom-centric views (tracking which sites each ion visits) and site-centric views (recording which ions occupy each site over time), supporting different analytical perspectives on the same transport processes. This dual representation, combined with multiple site-definition methods including dynamically-updating geometric sites, provides a flexible toolkit for analysing ionic transport mechanisms across diverse materials.

The software has been used to analyse ion-transport mechanisms in lithium-ion and fluoride-ion solid electrolytes [@Morgan2021; @Mercadier2023; @Krenzer2023; @Hu2025]. The package reproduces functionality from an earlier Fortran code that used dynamically defined coordination polyhedra to study ion-transport and defect distributions [@Burbano2016; @Morgan2014; @Morgan2014b; @Morgan2012; @Morgan2011], which was itself motivated by earlier studies using projections onto tetrahedral sites [@Castiglione1999; @Castiglione2001; @Marrocchelli2009]. Documentation and tutorials are available at [https://site-analysis.readthedocs.io](https://site-analysis.readthedocs.io)..

# Acknowledgements

B.J.M. acknowledges support from the Royal Society (UF130329 and URF/R/191006) and the Faraday Institution (FIRG016).

# Bibliography
