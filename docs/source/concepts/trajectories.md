# Trajectories

## What is a Trajectory in site_analysis?

A **Trajectory** in `site_analysis` serves two purposes:

1. **Collection**: It brings together the sites and atoms used to analyze a molecular dynamics simulation
2. **Analysis**: After processing, it provides access to how atoms move between sites during the simulation

Unlike raw MD trajectories that only store atom positions, a `site_analysis` Trajectory translates these coordinates into a record of which sites atoms occupy at each timestep. This simpler representation makes it easier to identify diffusion pathways and patterns.

A Trajectory combines:
1. **Sites**: Defined regions in the crystal structure
2. **Atoms**: Mobile ions being tracked
3. **Time**: The sequence of structures from the simulation

## Dual Perspectives on Trajectory Data

A key concept in understanding trajectories is that they provide two complementary views of the same data:

### Atom-centric View

The atom-centric view focuses on the journey of individual atoms through different sites. For each atom, the trajectory records which site it occupies at each timestep. This perspective helps answer questions like:

- Which sequence of sites does an atom visit?
- How long does an atom stay in each site?
- Do multiple atoms follow the same pathways?

### Site-centric View

The site-centric view focuses on the changing occupation of individual sites. For each site, the trajectory records which atoms occupy it at each timestep. This perspective helps answer questions like:

- Which sites are most frequently occupied?
- How does site occupation change over time?
- Are there patterns in how atoms enter and leave sites?

These dual perspectives allow for comprehensive analysis of diffusion mechanisms.

## Trajectory Analysis Concepts

Analyzing trajectory data reveals several important aspects of ion transport:

### Site Occupations

Site occupations represent which atoms are in which sites at a particular moment. Over time, the pattern of site occupations can reveal:

- Preferred sites (energetically favorable positions)
- Avoided sites (energetically unfavorable positions)
- Site occupation probabilities
- Correlations between different site occupations

### Transitions

Transitions occur when atoms move from one site to another. Tracking transitions reveals:

- Preferred diffusion pathways
- Barriers to diffusion (infrequent transitions)
- Diffusion mechanisms (pattern of transitions)
- Correlation between transitions of different atoms
