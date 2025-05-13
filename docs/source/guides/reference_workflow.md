# The Reference-Based Sites Workflow

## Introduction

The Reference-Based Sites workflow provides a method for defining crystallographic sites based on coordination environments rather than fixed spatial coordinates. This approach defines sites according to their chemical context within the structure, which persists even as the structure undergoes distortions during molecular dynamics simulations.

Rather than defining sites at absolute coordinates, this method identifies sites by their coordination environment (e.g., a site where a lithium ion is coordinated by four oxygen atoms). This definition remains meaningful even when atomic positions fluctuate due to thermal motion or structural relaxation.

The primary advantage of this workflow is its efficiency: it automates the creation of polyhedral and dynamic Voronoi sites, eliminating the need to define these sites manually. This simplifies what would otherwise be a laborious process of identifying and specifying site geometries.

## Basic Implementation

The most common way to use the Reference-Based Sites workflow is through the `TrajectoryBuilder` class, which provides a convenient interface to the underlying functionality:

```python
from site_analysis import TrajectoryBuilder
from pymatgen.core import Structure

# Load structures
ideal_structure = Structure.from_file("ideal_structure.cif")
target_structure = Structure.from_file("md_frame.vasp")

# Create a trajectory with tetrahedral Li sites
# This uses the RBS workflow internally through the TrajectoryBuilder
trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=4,
        label="tetrahedral"
    )
    .build())
```

This implementation:
1. Uses an ideal reference structure to define tetrahedral coordination environments
2. Creates corresponding sites in the target structure 
3. Automatically aligns the reference structure to the target structure, ensuring the coordination environments can be correctly mapped between them

The TrajectoryBuilder's `.with_polyhedral_sites()` and `.with_dynamic_voronoi_sites()` methods both utilize the RBS workflow internally, providing a streamlined interface for this functionality.

## Workflow Mechanism

The Reference-Based Sites workflow operates through these steps:

1. Structure alignment between reference and target
2. Identification of specified coordination environments in the reference structure
3. Mapping of these environments from reference to target structure
4. Creation of appropriate site objects in the target structure

## Structure Requirements

For the Reference-Based Sites workflow to function correctly, the reference and target structures must satisfy these essential requirements:

1. **Matching atom counts for alignment species**: 
   
   The atomic species used for alignment must have identical counts in both the reference and target structures. This requirement can be satisfied through either full structure alignment (when reference and target have identical stoichiometry) or subset alignment (when they differ).
   
   Subset alignment allows sites defined in a stoichiometric reference structure to be mapped onto an off-stoichiometric target structure (containing vacancies or interstitials) by aligning solely on the framework atoms that are present in equal numbers in both structures.

2. **Same supercell dimensions**: 
   
   Both structures must use consistent supercell conventions to ensure proper mapping between them. The absolute size of both structures must match to maintain the spatial relationship between coordination environments.

3. **Same orientation**: 
   
   The structures must have identical orientations as the alignment process only handles translation, not rotation. Lattice vectors must be oriented consistently between the reference and target structures to ensure proper superposition.

## Configuration Options

### Structure Alignment

When analyzing structures with different mobile ion content, alignment must be performed using framework atoms:

```python
trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_structure_alignment(align_species=["O", "Ti"])  # Framework atoms only
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=4
    )
    .build())
```

This example shows how to specify framework atoms for alignment by using the `align_species` parameter. By selecting only framework species that have consistent counts in both structures, proper alignment can be achieved even when the mobile ion content differs.

### Site Mapping Configuration

The site mapping process identifies which atoms define site geometry:

```python
trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_structure_alignment(align_species=["O", "Ti"])
    .with_site_mapping(mapping_species=["O"])  # Use oxygen atoms for site definition
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=4
    )
    .build())
```

The `mapping_species` parameter controls which atoms are used to define site geometries. For polyhedral sites, these atoms form the vertices of the polyhedra, while for dynamic Voronoi sites, they serve as reference atoms for calculating site centers.

## Best Practices

For optimal results:

1. **Use crystallographically defined reference structures**
   
   Use structures with well-defined, regular coordination environments as references. Geometry-optimized structures often provide clearer coordination patterns and more consistent site definitions. Ensure the reference contains examples of all relevant site types you wish to analyze in your target structures.

2. **Select appropriate alignment species**
   
   Use immobile framework atoms for alignment to ensure reliable mapping between structures. Framework atoms typically maintain more consistent positions compared to mobile species being studied.

3. **Adjust cutoff parameters as needed**
   
   Optimize cutoff values based on typical bond distances in your system. Different materials may require different cutoff values to properly identify coordination environments, so some experimentation with these parameters may be necessary.

## Troubleshooting

Common issues and their solutions:

### No Sites Found

```
ValueError: No polyhedral sites found for centre_species='Li', vertex_species='O'...
```

This error occurs when the specified coordination environments cannot be found in the reference structure. Adjust the `cutoff` value incrementally to capture the appropriate coordination shells. Verify that the specified species exist in the reference structure, and confirm that `n_vertices` corresponds to the actual coordination environment.

### Alignment Errors

```
ValueError: Different number of O atoms: 24 in reference vs 20 in target
```

This error indicates a mismatch in atom counts between structures. Verify that both structures use consistent supercell dimensions and that alignment species have identical counts in both structures. For structures with different compositions, use only framework atoms with consistent counts for alignment.

### Mapping Conflicts

```
ValueError: 1:1 mapping violation: Multiple reference atoms map to the same target atom(s)
```

This error occurs when the mapping cannot establish a one-to-one correspondence between reference and target atoms. Verify that structures have consistent orientations, as alignment only handles translation. Try alternative mapping species if the current selection causes ambiguities. Check for unusually distorted coordination environments that might prevent clean mapping.

## Advanced Implementation

For specialized applications requiring more direct control, the RBS workflow can be accessed directly:

```python
from site_analysis.reference_workflow import ReferenceBasedSites

# Initialize the reference-based workflow
rbs = ReferenceBasedSites(
    reference_structure=reference_structure,
    target_structure=target_structure,
    align=True,
    align_species=["O"]
)

# Create polyhedral sites
polyhedral_sites = rbs.create_polyhedral_sites(
    center_species="Li",
    vertex_species="O",
    cutoff=2.5,
    n_vertices=4,
    label="tetrahedral"
)

# These sites can then be used with custom analysis workflows
from site_analysis import Trajectory, atoms_from_structure
atoms = atoms_from_structure(target_structure, "Li")
trajectory = Trajectory(sites=polyhedral_sites, atoms=atoms)
```

This direct approach provides additional control for specialized analyses and can be integrated with custom analysis workflows.
