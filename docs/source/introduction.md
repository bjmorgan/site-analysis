# Introduction

## Background

Ionic transport in solid materials underlies the function of many devices including batteries, fuel cells, and sensors. While molecular dynamics simulations generate detailed atomic trajectories, it can be challenging to extract quantiative information about the microsopic transport mechanisms.

## The Site Analysis Approach

Site analysis addresses this challenge by discretising continuous atomic motions:

1. **Define sites**: Identify bounded volumes within the crystal structure
2. **Map ions to sites**: At each simulation timestep, Determine which sites contain which mobile ions

This approach converts complex continuous three-dimensional trajectories into sequences of discrete site occupations.

## Key Concepts

### Sites

In site_analysis, a **site** is a bounded region in space that may contain mobile ions. Sites can be defined in various ways:

- As spheres centred on set coordinates
- As polyhedra defined by framework atoms
- Using Voronoi decomposition, either with fixed Voronoi centers or with Voronoi centers dynamically calculated based on the instantaneous positions of the framework atoms

### Mobile Ions

Mobile ions move through the structure during a simulation. Site analysis tracks:

- Which site(s) each ion occupies at each timestep
- The sequence of sites visited by each ion
- Records of site occupations that can be further analyzed

### Site Occupation Data

Once ions are mapped to sites, the package records:

- Which sites are occupied at each timestep
- Which atoms occupy each site
- When atoms transition between sites

These data can then be accessed for subsequent analysis, such as:

- Calculating site occupation probabilities
- Identifying common migration pathways
- Analyzing residence times
- Quantifying correlations in ion movements
