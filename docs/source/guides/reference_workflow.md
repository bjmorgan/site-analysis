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
reference_structure = Structure.from_file("ideal_structure.cif")
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

The TrajectoryBuilder's `.with_polyhedral_sites()` and `.with_dynamic_voronoi_sites()` methods both utilise the RBS workflow internally, providing a streamlined interface for this functionality.

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

When working with reference and target structures that have different numbers of mobile atoms (such as when analysing vacancies, interstitials, or defective structures), alignment can be performed using a subset of atomic species, leaving out those with defects.

**Important**: Structure alignment is **enabled by default** for the reference-based workflow, even if not explicitly configured. To disable alignment, you must explicitly set `align=False`.

```python
trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_structure_alignment(
        align=True,                  # Enable alignment (default)
        align_species=["O", "Ti"]    # Framework atoms only
    )
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=4
    )
    .build())
```

This example shows how to specify framework atoms for alignment by using the `align_species` parameter. By selecting only framework species that have consistent counts in both structures, proper alignment can be achieved even when the mobile ion content differs between reference and target structures.

Alignment finds origin offsets of any size, for example between the two origin choices of a space group, and it is deterministic. When the alignment species have translational symmetry, several translations superimpose them equally well. They differ only in which index each site receives, and any one of them may be returned, even when the target already shares the reference's origin. To give the sites the same indices in several trajectories whose structures share an origin with one another, align the reference to a structure from one of them, shift the reference by the translation found, and build every trajectory from the shifted reference with alignment switched off:

```python
from site_analysis.reference_workflow.reference_based_sites import ReferenceBasedSites

# frame: a structure from one of the trajectories
rbs = ReferenceBasedSites(reference_structure, frame, align_species=["O", "Ti"])
shifted_reference = reference_structure.copy()
shifted_reference.translate_sites(
    range(len(shifted_reference)), rbs.translation_vector, to_unit_cell=True)
# then .with_reference_structure(shifted_reference) and .with_structure_alignment(align=False)
```

See the [`with_structure_alignment` documentation](builders.md#with_structure_alignmentaligntrue-align_speciesnone) in the builder guide.

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

The `mapping_species` parameter controls which atoms are used to define site geometries. For polyhedral sites, these atoms form the vertices of the polyhedra, while for dynamic Voronoi sites, they serve as reference atoms for calculating site centres.

### Periodic Boundary Condition Handling

The Reference-Based Sites workflow gives each polyhedral and dynamic Voronoi site a reference centre: the position of its central atom in the reference structure (aligned when alignment is on). Each site's vertices, or reference atoms, are unwrapped to their periodic images nearest its reference centre, which works correctly even in small supercells where sites span more than half the cell. For polyhedral sites, the reference centres also rank candidate sites by distance from each atom in the site search.

```python
trajectory = (TrajectoryBuilder()
    .with_structure(target_structure)
    .with_reference_structure(reference_structure)
    .with_mobile_species("Li")
    .with_polyhedral_sites(
        centre_species="Li",
        vertex_species="O",
        cutoff=2.5,
        n_vertices=4
    )
    .build())
```

The `use_reference_centers` parameter of `with_polyhedral_sites`, `with_dynamic_voronoi_sites`, the `create_trajectory_with_*` functions and `ReferenceBasedSites.create_*_sites` defaults to `True`. Passing `use_reference_centers=False`, which creates sites without reference centres that fall back to spread-based detection, is deprecated: it issues a `FutureWarning` and will be removed in version 2.0.

For details of both methods, see {doc}`../concepts/pbc_handling`.

## Best Practices

For optimal results:

1. **Use crystallographically defined reference structures**
   
   Use structures with well-defined, regular coordination environments as references. Geometry-optimised structures often provide clearer coordination patterns and more consistent site definitions. Ensure the reference contains examples of all relevant site types you wish to analyse in your target structures. Where possible, build the reference from its space group and Wyckoff positions (for example with `Structure.from_spacegroup`), which makes equivalent sites exactly equivalent; a structure read from a file or relaxed is symmetric only approximately. A symmetric reference is also much faster to align in a supercell. Translations related by a symmetry of the reference give equivalent alignments, so alignment needs to try only those within one primitive cell. A reference whose atoms are displaced by more than about 0.01 Å from symmetric positions has no such symmetry, so every translation must be tried.

2. **Select appropriate alignment species**
   
   Use immobile framework atoms for alignment to ensure reliable mapping between structures. Framework atoms typically maintain more consistent positions compared to mobile species being studied. Leave out any species with a defect in the reference, such as a vacancy, a substitution or a displaced atom: alignment can fail if the alignment atoms are not the same in every repeat of a supercell.

3. **Adjust cutoff parameters as needed**
   
   Optimise cutoff values based on typical bond distances in your system. Different materials may require different cutoff values to properly identify coordination environments, so some experimentation with these parameters may be necessary.

## Troubleshooting

Common issues and their solutions:

### No Sites Found

```
ValueError: No polyhedral sites found for centre_species='Li', vertex_species='O'...
```

This error occurs when the specified coordination environments cannot be found in the reference structure. Adjust the `cutoff` value incrementally to capture the appropriate coordination shells. Verify that the specified species exist in the reference structure, and confirm that `n_vertices` corresponds to the actual coordination environment.

### Alignment Errors

```
ValueError: Structure alignment failed: Different number of O atoms: 24 in reference vs 20 in target. Align on species present in equal numbers in both structures
```

This error indicates a mismatch in atom counts between structures. Verify that both structures use consistent supercell dimensions and that alignment species have identical counts in both structures. For structures with different compositions, use only framework atoms with consistent counts for alignment.

```
ValueError: Structure alignment failed: Could not superimpose the reference on the target using O. Try aligning only on immobile framework species (align_species), and check that both structures use the same cell and orientation. See "Alignment Errors" in the reference workflow guide.
```

Alignment looks for a translation that puts every reference atom of the alignment species close to a target atom of the same species: within half the shortest distance between two reference atoms of that species. That limit guarantees that each reference atom is matched to a different target atom. The error means that no translation achieved it. Common causes:

- **A mobile species among the alignment species.** If neither `align_species` nor mapping species are set, every species is used, including mobile ions, which rarely sit on their reference sites. Align on framework species instead, for example `align_species=["O"]`.
- **A different cell setting.** Alignment only translates. If the supercell differs (for example 2 x 1 x 1 against 1 x 2 x 1) or the axes are permuted or mirrored, usually no translation fits; build the reference in the same setting as the target. If the alignment atoms happen to look the same in both settings, alignment can succeed even though the other species do not line up, so make sure the two structures use the same setting.
- **A strongly disordered target.** An alignment atom that has moved far from its site, for instance a framework atom in a hot MD frame, can exceed the limit. Try a different frame, or align on a stiffer species.
- **A defect in an alignment species.** A vacancy, substitution or displaced atom of an alignment species in the reference can make alignment fail. Align on species without defects.
- **A distorted reference.** The limit is set by the reference's single shortest same-species distance, so one short pair in a distorted reference tightens it for every atom. If the reference is not close to the target's average structure, for example a reference relaxed around a defect, an idealised or symmetrised reference can help.

A species with a single atom has no such limit: aligning only on it always succeeds and rests on that one atom, so include a species with several atoms.

Candidates are screened on a sample of atoms before they are refined, so an alignment whose worst atom is very close to the limit can occasionally be rejected even though it would pass. A different frame or alignment species usually resolves this.

The limit and the reported alignment metrics are measured in the reference structure's lattice. Alignment works in fractional coordinates, so a target cell that is strained relative to the reference, such as an NPT frame or a relaxed cell, still aligns correctly.

### Mapping Conflicts

```
ValueError: Failed to map coordination environments for Li centres and O coordinating atoms: 1:1 mapping violation: target atom 12 is mapped from reference atoms 3 and 7
```

This error occurs when the mapping cannot establish a one-to-one correspondence between reference and target atoms. The message names the centre species and the coordinating species of the site definition that failed (a list of coordinating species is joined with ", "). It has two causes:

- Within one site definition, two reference atoms are nearest to the same target atom.
- Two site definitions conflict: a definition maps a reference atom to a target atom that an earlier definition in the same build mapped from a different reference atom.

With `ReferenceBasedSites` used directly, the part of the message after the prefix can also read `1:1 mapping violation: reference atom 3 is mapped to target atoms 12 and 15`, when a call maps a reference atom to a different target atom from an earlier call because the calls use different `target_species`. Verify that structures have consistent orientations, as alignment only handles translation. Try alternative mapping species if the current selection causes ambiguities. Check for unusually distorted coordination environments that might prevent clean mapping.

## Advanced Implementation

For specialised applications requiring more direct control, the RBS workflow can be accessed directly:

```python
from site_analysis.reference_workflow import ReferenceBasedSites

# Initialise the reference-based workflow
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

The sites created by successive `create_polyhedral_sites` and `create_dynamic_voronoi_sites` calls share one reference-to-target mapping; a call that conflicts with it raises `ValueError` ("1:1 mapping violation", see [Mapping Conflicts](#mapping-conflicts)). `mapped_reference_frac_coords()` returns the fractional coordinates of the reference atom mapped to each target atom (aligned to the target when alignment is on), as an array of shape `(n_target_atoms, 3)` with NaN for target atoms that no site uses:

```python
reference_coords = rbs.mapped_reference_frac_coords()
```

This direct approach provides additional control for specialised analyses and can be integrated with custom analysis workflows.
