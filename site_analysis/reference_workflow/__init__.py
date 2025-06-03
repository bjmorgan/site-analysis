"""Reference-based workflow for defining sites in crystal structures.

This package provides tools for defining crystallographic sites in a target structure
based on coordination environments identified in a reference structure.

The main entry point is the ReferenceBasedSites class, which orchestrates the entire
workflow of structure alignment, coordination environment finding, index mapping,
and site creation.

Example:
    ```python
    from site_analysis.reference_workflow import ReferenceBasedSites
    
    # Create reference-based sites workflow
    rbs = ReferenceBasedSites(
        reference_structure=reference_structure,
        target_structure=target_structure
    )
    
    # Create polyhedral sites
    sites = rbs.create_polyhedral_sites(
        center_species="Li",
        vertex_species="O", 
        cutoff=3.0,
        n_vertices=4
    )
    ```
"""

from .reference_based_sites import ReferenceBasedSites
