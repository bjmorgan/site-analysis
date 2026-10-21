"""Builder pattern implementation for site_analysis.

This module provides a fluent builder interface for creating site analysis
objects, making it easier to set up and run site analysis workflows.

Examples::

    # Using the builder directly
    trajectory = (TrajectoryBuilder()
                 .with_structure(structure)
                 .with_mobile_species("Li")
                 .with_spherical_sites(
                     centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
                     radii=2.0,
                     labels=["octahedral", "tetrahedral"]
                 )
                 .build())

    # Using a factory function
    trajectory = create_trajectory_with_spherical_sites(
        structure=structure,
        mobile_species="Li",
        centres=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
        radii=2.0,
        labels=["octahedral", "tetrahedral"]
    )

    # Creating a trajectory with polyhedral sites
    trajectory = create_trajectory_with_polyhedral_sites(
        structure=target_structure,
        reference_structure=reference_structure,
        mobile_species="Li",
        centre_species="O",
        vertex_species=["Li", "Na"],
        cutoff=3.0,
        n_vertices=4,
        label="tetrahedral"
    )
"""

from __future__ import annotations

from pymatgen.core import Structure
from site_analysis.atom import Atom, atoms_from_structure
from site_analysis.site import Site
from site_analysis.spherical_site import SphericalSite
from site_analysis.voronoi_site import VoronoiSite
from site_analysis.polyhedral_site import PolyhedralSite
from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection
from site_analysis.site_collection import SiteCollection
from site_analysis.dynamic_voronoi_site import DynamicVoronoiSite
from site_analysis.trajectory import Trajectory
from site_analysis.reference_workflow.reference_based_sites import ReferenceBasedSites
from site_analysis.reference_workflow.structure_aligner import _check_deprecated_alignment_options
import numpy as np
from collections.abc import Callable, Sequence
from typing import Any

# Returns the ReferenceBasedSites of a build, created on the first call
_ReferenceBasedSitesGetter = Callable[[], ReferenceBasedSites]

# Creates the sites of one with_*_sites definition during build()
_SiteGenerator = Callable[[_ReferenceBasedSitesGetter], Sequence[Site]]


class TrajectoryBuilder:
    """Builder for creating Trajectory objects for site analysis.

    This class provides a step-by-step approach to creating a Trajectory
    object for analysing site occupations in crystal structures.

    **Structure Alignment and Site Mapping**

    The builder supports separate control over structure alignment (finding optimal
    translations) and site mapping (identifying corresponding sites):

    - Structure alignment: Use ``with_structure_alignment()`` to control whether
      and how structures are aligned before site creation.
    - Site mapping: Use ``with_site_mapping()`` to specify which species are used
      to identify corresponding sites between structures.

    **Default Behaviours**

    - Alignment is enabled by default.
    - If mapping species are specified but alignment species are not, mapping
      species will be used for alignment.
    - If alignment species are specified but mapping species are not, alignment
      species will be used for mapping.

    Example::

        builder = TrajectoryBuilder()

        builder.with_structure(structure)
            .with_reference_structure(reference_structure)
            .with_mobile_species("Li")
            .with_structure_alignment(align=True, align_species=["O"])
            .with_site_mapping(mapping_species=["Na"])
            .with_polyhedral_sites(
                centre_species="Li",
                vertex_species="O",
                cutoff=2.0,
                n_vertices=4
            )

        trajectory = builder.build()
    """
    
    def __init__(self) -> None:
        """Initialize a TrajectoryBuilder.
        
        Creates a new builder with all attributes set to their default values.
        """
        # Call reset() to set all attributes to their default values
        self.reset()
    
    def reset(self) -> 'TrajectoryBuilder':
        """Reset the builder state to default values.
        
        This method clears all configuration and returns the builder to its
        initial state. It is called automatically during initialisation and
        after build(), but can also be called explicitly if needed.
        
        Returns:
            self: For method chaining
        """
        self._structure: Structure | None = None
        self._reference_structure: Structure | None= None
        self._mobile_species: str | list[str] | None = None
        self._atoms: list[Atom] | None = None
        
        # Alignment options
        self._align = True
        self._align_species: list[str] | None = None
        
        # Mapping options
        self._mapping_species: list[str] | None = None

        # Validation options
        self._min_atom_distance: float = 0.5

        # Commitment options
        self._commitment_radius: float | dict[str, float] | None = None

        # Functions to be called during build() to create sites
        self._site_generators: list[_SiteGenerator] = []

        return self
        
    def with_structure(self, structure) -> TrajectoryBuilder:
        """Set the structure to analyse.
        
        Args:
            structure: A pymatgen Structure object
            
        Returns:
            self: For method chaining
        """
        self._structure = structure
        return self
        
    def with_reference_structure(self, reference_structure) -> TrajectoryBuilder:
        """Set the reference structure for complex site types.
        
        Args:
            reference_structure: A pymatgen Structure object representing
                the ideal reference structure
                
        Returns:
            self: For method chaining
        """
        self._reference_structure = reference_structure
        return self
        
    def with_mobile_species(self, species: str | list[str]) -> TrajectoryBuilder:
        """Set the mobile species to track.
        
        Args:
            species: Species string or list of species strings for mobile atoms
                
        Returns:
            self: For method chaining
        """
        self._mobile_species = species
        return self
    
    def with_structure_alignment(self,
                align: bool = True,
                align_species: str | list[str] | None = None,
                align_metric: str | None = None,
                align_algorithm: str | None = None,
                align_minimizer_options: dict[str, Any] | None = None,
                align_tolerance: float | None = None) -> 'TrajectoryBuilder':
        """Set options for aligning reference and target structures.

        Structure alignment finds the translation that superimposes the
        reference structure on the target structure, whatever the offset
        between their origins, so that sites defined in the reference can
        be placed in the target. Translations related by a translational
        symmetry of the alignment atoms fit equally well, and any one of
        them may be chosen; they differ only in which index each site gets.

        Note:
            Structure alignment is ENABLED by default when using polyhedral or dynamic
            Voronoi sites, even if this method is not explicitly called. To disable
            alignment, call this method with align=False.

        Args:
            align: Whether to perform structure alignment. Default is True.
            align_species: Species to use for alignment. Can be a string or list of strings.
                Default is None, which means:
                - If mapping species have been set with with_site_mapping(), those species will be used
                - Otherwise, all species will be used, and both structures must have the same composition
                Use species whose atoms stay on their sites, such as framework atoms.
            align_metric: Deprecated; has no effect, except that 'max_dist'
                raises ValueError. Will be removed in version 2.0.
            align_algorithm: Deprecated; has no effect. Will be removed in
                version 2.0.
            align_minimizer_options: Deprecated; has no effect. Will be
                removed in version 2.0.
            align_tolerance: Deprecated; has no effect. Will be removed in
                version 2.0.

        Returns:
            self: For method chaining

        Raises:
            ValueError: If align_metric is 'max_dist', which is no longer
                supported.

        Example::

            # Use default alignment (enabled, all species)
            builder.with_reference_structure(reference)
            builder.with_polyhedral_sites(...)

            # Specify alignment species explicitly
            builder.with_structure_alignment(align_species=["O", "Ti"])

            # Disable alignment
            builder.with_structure_alignment(align=False)
        """
        _check_deprecated_alignment_options(
            align_metric=align_metric, align_algorithm=align_algorithm,
            align_minimizer_options=align_minimizer_options, align_tolerance=align_tolerance)

        self._align = align

        # Convert single species string to a list
        if isinstance(align_species, str):
            self._align_species = [align_species]
        else:
            self._align_species = align_species
        return self

    def with_site_mapping(self, mapping_species: str | list[str] | None) -> TrajectoryBuilder:
        """Set the species to use for mapping sites between reference and target structures.
        
        Site mapping identifies corresponding sites between structures even when
        atom counts differ, for example when structures have different numbers
        of mobile ions but the same framework atoms.
        
        If mapping species are specified but alignment species are not:
        - The mapping species will also be used for alignment (unless alignment is disabled)
        
        If mapping species are not specified:
        - The alignment species will be used for mapping
        
        Args:
            mapping_species: Species to use for mapping. Can be a string or list of strings.
                If None, alignment species will be used for mapping.
                
        Returns:
            self: For method chaining
        """
        # Convert single species string to a list
        if isinstance(mapping_species, str):
            self._mapping_species = [mapping_species]
        else:
            self._mapping_species = mapping_species
        return self
        
    def with_min_atom_distance(self, distance: float) -> 'TrajectoryBuilder':
        """Set the minimum allowed distance between same-species atoms
        in the reference structure.

        If any pair of atoms of the same species in the reference
        structure is closer than this threshold, ``build()`` raises
        ``ValueError``. This catches reference structures where atoms
        sit on a general Wyckoff position instead of the correct
        special position, producing duplicate coordination environments.

        This check runs whenever a reference structure is set,
        regardless of site type. It is most relevant for polyhedral
        and dynamic Voronoi site workflows, but also applies when a
        reference structure is provided for mapping or alignment.

        Set to 0 to disable the check.

        Args:
            distance: Minimum distance in the same units as the
                lattice parameters. Must be non-negative.
                The builder default is 0.5.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If distance is negative or NaN.
        """
        if not distance >= 0:
            raise ValueError(
                f"min_atom_distance must be non-negative, got {distance}"
            )
        self._min_atom_distance = distance
        return self

    def with_commitment(self,
            radius: float | dict[str, float] = 1.0) -> TrajectoryBuilder:
        """Turn on spatial commitment for the trajectory.

        With commitment on, an atom moves to a new site only once it is in
        that site and within the commitment radius of its centre, and stays
        committed to its previous site until then. See the commitment guide.

        Args:
            radius: Commitment radius in Å, either one value for every site
                or a dict mapping site labels to radii. Default is 1.0.
                Radii are validated when ``build()`` is called.

        Returns:
            self: For method chaining.
        """
        self._commitment_radius = radius
        return self

    def with_spherical_sites(self,
                    centres: list[list[float]], 
                    radii: float | list[float], 
                    labels: str | list[str] | None = None) -> TrajectoryBuilder:
        """Define spherical sites.
        
        Note: Sites will be generated when build() is called, which raises
        ``ValueError`` if a centre is not three finite numbers or a radius
        is negative or NaN.
        
        Args:
            centres: list of fractional coordinate centres for spherical sites
            radii: either a single radius (float) to use for all sites, or a list 
                of radii (one per centre)
            labels: either a single label (str) to use for all sites, a list of 
                labels (one per centre), or None
            
        Returns:
            self: For method chaining
        """
        # Convert single radius to list if needed
        if isinstance(radii, (float, int)):
            radii = [radii] * len(centres)
            
        # Convert single label to list if needed
        if isinstance(labels, str):
            labels = [labels] * len(centres)
            
        # Validate lengths
        if len(centres) != len(radii):
            raise ValueError("Number of centres must match number of radii")
            
        if labels is not None and len(centres) != len(labels):
            raise ValueError("Number of centres must match number of labels")
            
        # Define the site generation function but don't execute it yet
        def create_spherical_sites(_: _ReferenceBasedSitesGetter) -> Sequence[SphericalSite]:
            # Create spherical sites
            sites = []
            for i, (centre, radius) in enumerate(zip(centres, radii)):
                label = labels[i] if labels and i < len(labels) else None
                sites.append(SphericalSite(
                    frac_coords=np.array(centre), 
                    rcut=radius, 
                    label=label
                ))
            return sites
            
        # Store the function for later execution
        self._site_generators.append(create_spherical_sites)
        return self
        
    def with_voronoi_sites(self, 
                        centres: list[list[float]], 
                        labels: str | list[str] | None = None) -> TrajectoryBuilder:
        """Define Voronoi sites.

        Note: Sites will be generated when build() is called.

        Args:
            centres: list of fractional coordinate centres for Voronoi sites
            labels: either a single label (str) to use for all sites, a list of
                labels (one per centre), or None

        Returns:
            self: For method chaining

        Raises:
            ValueError: If a list of labels does not have one label per centre.
        """
        # Convert single label to list if needed
        if isinstance(labels, str):
            labels = [labels] * len(centres)

        if labels is not None and len(centres) != len(labels):
            raise ValueError("Number of centres must match number of labels")

        # Define the site generation function but don't execute it yet
        def create_voronoi_sites(_: _ReferenceBasedSitesGetter) -> Sequence[VoronoiSite]:
            # Create Voronoi sites
            sites = []
            for i, centre in enumerate(centres):
                label = labels[i] if labels and i < len(labels) else None
                sites.append(VoronoiSite(
                    frac_coords=np.array(centre), 
                    label=label
                ))
            return sites
            
        # Store the function for later execution
        self._site_generators.append(create_voronoi_sites)
        return self
        
    def with_polyhedral_sites(self, 
                        centre_species: str, 
                        vertex_species: str | list[str], 
                        cutoff: float, 
                        n_vertices: int, 
                        label: str | None = None,
                        use_reference_centers: bool = True) -> TrajectoryBuilder:
        """Define polyhedral sites using the ReferenceBasedSites workflow.
        
        Creates polyhedral sites by identifying coordination environments in the reference 
        structure and mapping them to corresponding sites in the target structure. Each site 
        is defined by a polyhedron formed by vertex atoms around a central atom.
        
        The workflow involves:
        1. Finding coordination environments in the reference structure
        2. Mapping these environments to the target structure  
        3. Creating PolyhedralSite objects with proper periodic boundary handling
        
        Note:
            Sites will be generated when build() is called, not immediately.
            Requires both structure and reference_structure to be set.
        
        Args:
            centre_species: Atomic species at the centre of coordination environments.
            vertex_species: Atomic species at vertices of coordination polyhedra. 
                Can be a single species string or list of species strings.
            cutoff: Maximum distance (in Ångströms) for vertex atoms to be considered 
                part of the coordination environment.
            n_vertices: Number of vertex atoms required for each polyhedral site.
            label: Optional label to assign to all created sites. Default is None.
            use_reference_centers: Controls periodic boundary condition handling for
                reference-based sites (polyhedral and dynamic Voronoi sites).
                Default is True.

                * ``True`` (recommended) -- Reference-based PBC correction. Defines
                  a reference center for each site and unwraps vertex coordinates
                  relative to this center. Correctly handles sites that naturally
                  span >50% of unit cell dimensions, even in small simulation cells.
                * ``False`` (advanced usage) -- Spread-based PBC correction. If
                  vertex coordinates span >50% of the unit cell in any dimension,
                  assumes this indicates PBC wrapping and shifts coordinates
                  accordingly. WARNING: Gives incorrect results when sites
                  legitimately span >50% of the unit cell (e.g., octahedral sites
                  in a 2x2x2 FCC supercell). It is no faster: after the first
                  frame, both settings update the unwrapping in the same way.
                  The polyhedral site search also falls back to the neighbours
                  of an atom's most recent site, then list order, instead of
                  ranking sites by distance, which can make assignment slower.
                  Only use after verifying it works correctly for your
                  structures.

        Returns:
            self: For method chaining

        Raises:
            ValueError: At build() time if no coordination environments are found,
                if required structures are not set, or if the mapping to the
                target structure is not one-to-one (see "Mapping Conflicts" in
                the reference workflow guide).

        Example::

            # Tetrahedral sites around Li atoms
            builder.with_polyhedral_sites(
                centre_species="Li",
                vertex_species="O",
                cutoff=2.5,
                n_vertices=4,
                label="tetrahedral"
            )

            # Octahedral sites with mixed vertex species
            builder.with_polyhedral_sites(
                centre_species="Ti",
                vertex_species=["O", "F"],
                cutoff=2.8,
                n_vertices=6,
                label="octahedral"
            )
        """
        # Define the site generation function but don't execute it yet
        def create_polyhedral_sites(
                get_reference_based_sites: _ReferenceBasedSitesGetter) -> Sequence[PolyhedralSite]:
            rbs = get_reference_based_sites()

            # Create sites
            sites = rbs.create_polyhedral_sites(
                        center_species=centre_species,
                        vertex_species=vertex_species,
                        cutoff=cutoff,
                        n_vertices=n_vertices,
                        label=label,
                        target_species=self._effective_mapping_species,
                        use_reference_centers=use_reference_centers
                    )
            if not sites:
                raise ValueError(
                    f"No polyhedral sites found for centre_species='{centre_species}', "
                    f"vertex_species='{vertex_species}', cutoff={cutoff}, "
                    f"n_vertices={n_vertices}. Try adjusting these parameters or verify "
                    f"that the specified species exist in the structure.")
            return sites
            
        # Store the function for later execution
        self._site_generators.append(create_polyhedral_sites)
        return self
        
    def with_dynamic_voronoi_sites(self,
    centre_species: str,
    reference_species: str | list[str],
    cutoff: float,
    n_reference: int,
    label: str | None = None,
    use_reference_centers: bool = True) -> TrajectoryBuilder:
        """Define dynamic Voronoi sites using the ReferenceBasedSites workflow.
        
        Creates dynamic Voronoi sites where the site centres are dynamically calculated
        from the positions of reference atoms. Unlike fixed Voronoi sites, these adapt
        to structural changes as the reference atoms move.
        
        The workflow involves:
        1. Finding coordination environments in the reference structure
        2. Mapping these environments to the target structure
        3. Creating DynamicVoronoiSite objects that calculate centres from reference atoms
        
        Note:
            Sites will be generated when build() is called, not immediately.
            Requires both structure and reference_structure to be set.
        
        Args:
            centre_species: Atomic species at centres where sites will be located.
            reference_species: Atomic species used as reference atoms to define 
                dynamic site centres. Can be a single species string or list of species strings.
            cutoff: Maximum distance (in Ångströms) for reference atoms to be considered 
                part of the coordination environment.
            n_reference: Number of reference atoms required for each dynamic site.
            label: Optional label to assign to all created sites. Default is None.
            use_reference_centers: Controls periodic boundary condition handling for
                reference-based sites (polyhedral and dynamic Voronoi sites).
                Default is True.

                * ``True`` (recommended) -- Reference-based PBC correction. Defines
                  a reference center for each site and unwraps vertex coordinates
                  relative to this center. Correctly handles sites that naturally
                  span >50% of unit cell dimensions, even in small simulation cells.
                * ``False`` (advanced usage) -- Spread-based PBC correction. If
                  vertex coordinates span >50% of the unit cell in any dimension,
                  assumes this indicates PBC wrapping and shifts coordinates
                  accordingly. WARNING: Gives incorrect results when sites
                  legitimately span >50% of the unit cell (e.g., octahedral sites
                  in a 2x2x2 FCC supercell). It is no faster: after the first
                  frame, both settings update the unwrapping in the same way.
                  Only use after verifying it works correctly for your
                  structures.

        Returns:
            self: For method chaining

        Raises:
            ValueError: At build() time if no coordination environments are found,
                if required structures are not set, or if the mapping to the
                target structure is not one-to-one (see "Mapping Conflicts" in
                the reference workflow guide).

        Example::

            # Dynamic sites at Li positions defined by neighbouring O atoms
            builder.with_dynamic_voronoi_sites(
                centre_species="Li",
                reference_species="O",
                cutoff=3.0,
                n_reference=4,
                label="tetrahedral_dynamic"
            )

            # Sites with mixed reference species
            builder.with_dynamic_voronoi_sites(
                centre_species="Na",
                reference_species=["O", "F"],
                cutoff=2.8,
                n_reference=6,
                label="mixed_coordination"
            )
        """
        # Define the site generation function but don't execute it yet
        def create_dynamic_voronoi_sites(
                get_reference_based_sites: _ReferenceBasedSitesGetter) -> Sequence[DynamicVoronoiSite]:
            rbs = get_reference_based_sites()

            # Create sites
            sites = rbs.create_dynamic_voronoi_sites(
                        center_species=centre_species,
                        reference_species=reference_species,
                        cutoff=cutoff,
                        n_reference=n_reference,
                        label=label,
                        target_species=self._effective_mapping_species,
                        use_reference_centers=use_reference_centers
                    )
            if not sites:
                raise ValueError(
                    f"No dynamic Voronoi sites found for centre_species='{centre_species}', "
                    f"reference_species='{reference_species}', cutoff={cutoff}, "
                    f"n_reference={n_reference}. Try adjusting these parameters or verify "
                    f"that the specified species exist in the structure.")
            return sites
            
        # Store the function for later execution
        self._site_generators.append(create_dynamic_voronoi_sites)
        return self
        
    def with_existing_sites(self, sites: Sequence[Site]) -> TrajectoryBuilder:
        """Use existing site objects.

        The sites are used as given. They are checked when ``build()`` is
        called, not here.

        Args:
            sites: The site objects to use, for example ``PolyhedralSite``
                or ``SphericalSite`` objects. All the sites of a build must
                be of one type.

        Returns:
            self: For method chaining

        Raises:
            ValueError: At ``build()``, if a polyhedral site has a vertex
                index that is not an atom of the structure, or if two sites
                of the build are duplicates (coincident or nested when the
                shared faces are found from the reference geometry,
                otherwise with the same vertex or reference indices).
            TypeError: At ``build()``, if the sites of the build (these and
                those from other site definitions) are of mixed site types.
        """
        # Define a simple function that returns the provided sites
        def return_existing_sites(_: _ReferenceBasedSitesGetter) -> Sequence[Site]:
            return sites
            
        # Store the function for later execution
        self._site_generators.append(return_existing_sites)
        return self
        
    def with_existing_atoms(self, atoms: list) -> TrajectoryBuilder:
        """Use existing atom objects.
        
        Args:
            atoms: list of atom objects
                
        Returns:
            self: For method chaining
        """
        self._atoms = atoms
        return self
        
    def _validate_reference_atom_distances(self) -> None:
        """Check that no same-species atom pairs in the reference
        structure are closer than ``_min_atom_distance``."""
        from site_analysis.neighbour_search import PeriodicNeighbourIndex
        from site_analysis.tools import indices_for_species

        ref = self._reference_structure
        lattice_matrix = np.array(ref.lattice.matrix)  # type: ignore[union-attr]
        species_list = [s.species_string for s in ref]  # type: ignore[union-attr]
        frac_coords = np.array(ref.frac_coords)  # type: ignore[union-attr]

        for sp in sorted(set(species_list)):
            indices = indices_for_species(species_list, sp)
            if len(indices) < 2:
                continue

            coords = frac_coords[indices]
            index = PeriodicNeighbourIndex(coords, lattice_matrix)
            i, j, dists = index.query_within(coords, self._min_atom_distance)
            close = (i != j) & (dists < self._min_atom_distance)
            if np.any(close):
                # The closest pair, with ties broken by the lower indices.
                i, j, dists = i[close], j[close], dists[close]
                closest = np.lexsort((j, i, dists))[0]
                i_local, j_local = i[closest], j[closest]
                min_dist = float(dists[closest])
                raise ValueError(
                    f"Reference structure has {sp} atoms at indices "
                    f"{indices[i_local]} and {indices[j_local]} that are "
                    f"only {min_dist:.3f} apart (threshold: "
                    f"{self._min_atom_distance}). This typically means "
                    f"the atom coordinate is on a general Wyckoff position "
                    f"instead of the correct special position, which "
                    f"produces duplicate coordination environments. "
                    f"To disable this check, call "
                    f".with_min_atom_distance(0) on the builder."
                )

    def _new_reference_based_sites(self) -> ReferenceBasedSites:
        """Create a ReferenceBasedSites from the builder's current settings.

        It maps the reference structure to the structure with the builder's
        alignment settings; if alignment is on and no alignment species are
        set, the mapping species are used for alignment.

        Returns:
            A new ReferenceBasedSites.

        Raises:
            ValueError: If the structure or reference structure is not set,
                or if alignment fails.
        """
        if not self._structure or not self._reference_structure:
            raise ValueError(
                "Both structure and reference_structure must be set for "
                "polyhedral and dynamic Voronoi sites")
        align_species = self._align_species
        if self._align and align_species is None and self._mapping_species is not None:
            align_species = self._mapping_species
        return ReferenceBasedSites(
            reference_structure=self._reference_structure,
            target_structure=self._structure,
            align=self._align,
            align_species=align_species
        )

    @property
    def _effective_mapping_species(self) -> list[str] | None:
        """Species that reference-based sites map to.

        The mapping species if set, otherwise the alignment species.
        """
        if self._mapping_species is not None:
            return self._mapping_species
        return self._align_species

    @staticmethod
    def _find_reference_shared_faces(
            rbs: ReferenceBasedSites | None,
            site_collection: SiteCollection) -> bool:
        """Find the shared faces of polyhedral sites from the reference geometry.

        Args:
            rbs: The ReferenceBasedSites that created the build's
                reference-based sites, or None if there are none.
            site_collection: The site collection of the built trajectory,
                whose sites' vertex indices are atoms of the structure
                (``build()`` checks this before calling this method).

        Returns:
            True if the shared faces were found: the sites are polyhedral,
            ``rbs`` is not None, and the row of
            ``rbs.mapped_reference_frac_coords()`` for every vertex atom is
            finite. False otherwise.

        Raises:
            ValueError: If two sites coincide, one lies inside another, two
                sites overlap, or the atoms two sites share cannot be
                triangulated identically on both. The message says that this
                typically means the reference structure has multiple atoms
                inside the same coordination environment.
            RuntimeError: If a site's reference vertices are degenerate.
        """
        if rbs is None or not isinstance(site_collection, PolyhedralSiteCollection):
            return False
        coords = rbs.mapped_reference_frac_coords()
        vertex_indices = np.concatenate(
            [site.vertex_indices for site in site_collection.sites])
        if not np.isfinite(coords[vertex_indices]).all():
            return False
        try:
            site_collection.find_shared_faces(coords, rbs.reference_structure.lattice.matrix)
        except ValueError as error:
            raise ValueError(
                f"{error}. This typically means the reference structure has "
                f"multiple atoms inside the same coordination environment.") from error
        return True

    def _validate_unique_sites(self, sites: list[Site]) -> None:
        """Check that no two sites share the same defining indices.

        Applies to PolyhedralSite (vertex_indices) and DynamicVoronoiSite
        (reference_indices). Skipped for VoronoiSite and SphericalSite where
        duplicate detection based on coordinates is unreliable. ``build()``
        calls this only when the shared faces were not found from the
        reference geometry, which rejects coincident and nested polyhedral
        sites by position instead.

        Args:
            sites: The sites to check.
        """
        if not sites:
            return

        if isinstance(sites[0], PolyhedralSite):
            index_attr = "vertex_indices"
        elif isinstance(sites[0], DynamicVoronoiSite):
            index_attr = "reference_indices"
        else:
            return  # VoronoiSite, SphericalSite or an unknown site type

        seen: dict[frozenset[int], int] = {}
        for site in sites:
            key = frozenset(getattr(site, index_attr))
            if key in seen:
                raise ValueError(
                    f"Duplicate sites: site {seen[key]} and site "
                    f"{site.index} share the same {index_attr} "
                    f"{sorted(key)}. This typically means the reference "
                    f"structure has multiple atoms inside the same "
                    f"coordination environment."
                )
            seen[key] = site.index

    def build(self) -> Trajectory:
        """Build and return the Trajectory object.
        
        This method validates all required parameters and generates sites
        using the previously configured site generators. When every site is
        polyhedral and every vertex atom of every site was mapped from the
        reference structure by a ``with_polyhedral_sites`` definition, it
        also finds the faces the sites share, and fixes each site's
        triangles, from the reference geometry; otherwise they are found
        from the first structure analysed, or when
        ``Trajectory.find_shared_faces`` is called.
        
        Sites are created first, then the atoms, then the trajectory, so a
        missing mobile species is reported before mixed site types, an
        invalid commitment radius or a vertex index that is not an atom of
        the structure.

        Returns:
            The constructed Trajectory object.

        Raises:
            ValueError: Missing or invalid configuration: no structure, no
                site definition, no reference structure for reference-based
                sites, no mobile species or no atoms of it, reference atoms of
                one species closer than ``min_atom_distance``, a failed
                structure alignment, or commitment radii that are not
                positive, do not cover every site label, or are given as a
                dict while some sites have no label.
            ValueError: Invalid sites: no sites found for a definition, a
                spherical site whose centre is not three finite numbers or
                whose radius is negative or NaN, a polyhedral site with a
                vertex index that is not an atom of the structure, or
                duplicate, coincident, nested or overlapping sites.
            ValueError: Mapping failures: coordination environments that
                cannot be found or mapped one-to-one to the target structure
                (see "Mapping Conflicts" in the reference workflow guide).
            TypeError: Mixed site types (the message names the types), or a
                commitment radius that is not a number.
            RuntimeError: A polyhedral site whose reference vertices are
                degenerate, when the shared faces are found from the
                reference geometry.
        """
        # Validate basic requirements
        if not self._structure:
            raise ValueError("Structure must be set")
        if not self._site_generators:
            raise ValueError("Site type must be defined using one of the with_*_sites methods")
            
        # Pre-build: validate reference structure
        if self._reference_structure is not None and self._min_atom_distance > 0:
            self._validate_reference_atom_distances()

        # Reset the site index counter
        Site.reset_index()

        # The reference-based site generators share one ReferenceBasedSites,
        # created on first use
        reference_based_sites: ReferenceBasedSites | None = None

        def get_reference_based_sites() -> ReferenceBasedSites:
            nonlocal reference_based_sites
            if reference_based_sites is None:
                reference_based_sites = self._new_reference_based_sites()
            return reference_based_sites

        # Generate all sites
        sites: list[Site] = []
        for generator in self._site_generators:
            sites.extend(generator(get_reference_based_sites))

        # Use the existing atoms if set, otherwise create them
        atoms = self._atoms
        if not atoms:
            if not self._mobile_species:
                raise ValueError("Mobile species must be set")
            atoms = atoms_from_structure(self._structure, self._mobile_species)
            if not atoms:
                available_species = sorted(set(str(site.specie) for site in self._structure))
                raise ValueError(
                    f"No atoms found matching mobile species '{self._mobile_species}'. "
                    f"Available species in structure: {available_species}"
                )

        # Create trajectory
        trajectory = Trajectory(sites=sites, atoms=atoms,
                                commitment_radius=self._commitment_radius)
        if isinstance(trajectory.site_collection, PolyhedralSiteCollection):
            trajectory.site_collection.validate_atom_count(len(self._structure))

        # Find shared faces from the reference geometry, which also rejects
        # coincident and nested sites; otherwise check for duplicate sites by index
        if not self._find_reference_shared_faces(
                reference_based_sites, trajectory.site_collection):
            self._validate_unique_sites(sites)

        # Reset the builder state for future use
        self.reset()
        
        return trajectory


def create_trajectory_with_spherical_sites(
    structure: Structure, 
    mobile_species: str | list[str], 
    centres: list[list[float]], 
    radii: float | list[float], 
    labels: str | list[str] | None = None
) -> Trajectory:
    """Create a Trajectory with spherical sites for site analysis.
    
    Creates a trajectory object configured with spherical sites defined by centres
    and radii. Each spherical site represents a spherical volume in the crystal
    structure where atoms can be assigned based on distance from the centre.
    
    Args:
        structure: Pymatgen Structure object containing the crystal structure to analyse.
        mobile_species: Species string (e.g., "Li") or list of species strings 
            (e.g., ["Li", "Na"]) identifying the mobile atoms to track.
        centres: List of fractional coordinate triplets defining the centres of 
            spherical sites. Each centre should be a list of three floats [x, y, z].
        radii: Cutoff radius in Ångströms for spherical sites. If a single float 
            is provided, all sites use the same radius. If a list is provided, 
            it must have the same length as centres.
        labels: Optional labels for the sites. Can be a single string applied to 
            all sites, a list of strings (one per site), or None for no labels.
            
    Returns:
        Trajectory: Configured trajectory object ready for site analysis.
        
    Raises:
        ValueError: If the number of centres doesn't match the number of radii 
            or labels when lists are provided.
    """
    builder = TrajectoryBuilder()
    return (builder
            .with_structure(structure)
            .with_mobile_species(mobile_species)
            .with_spherical_sites(centres, radii, labels)
            .build())


def create_trajectory_with_voronoi_sites(
    structure: Structure, 
    mobile_species: str | list[str], 
    centres: list[list[float]], 
    labels: str | list[str] | None = None
) -> Trajectory:
    """Create a Trajectory with Voronoi sites for site analysis.
    
    Creates a trajectory object configured with Voronoi sites that partition space
    based on proximity to site centres. Each point in space is assigned to the
    Voronoi site with the nearest centre, ensuring complete space-filling coverage.
    
    Args:
        structure: Pymatgen Structure object containing the crystal structure to analyse.
        mobile_species: Species string (e.g., "Li") or list of species strings 
            (e.g., ["Li", "Na"]) identifying the mobile atoms to track.
        centres: List of fractional coordinate triplets defining the centres of 
            Voronoi sites. Each centre should be a list of three floats [x, y, z].
        labels: Optional labels for the sites. Can be a single string applied to
            all sites, a list of strings (one per site), or None for no labels.
            
    Returns:
        Trajectory: Configured trajectory object ready for site analysis.
        
    Raises:
        ValueError: If the number of labels doesn't match the number of centres.
    """
    builder = TrajectoryBuilder()
    return (builder
            .with_structure(structure)
            .with_mobile_species(mobile_species)
            .with_voronoi_sites(centres, labels)
            .build())


def create_trajectory_with_polyhedral_sites(
    structure: Structure, 
    reference_structure: Structure, 
    mobile_species: str | list[str], 
    centre_species: str, 
    vertex_species: str | list[str], 
    cutoff: float, 
    n_vertices: int, 
    label: str | None = None,
    align: bool = True,
    align_species: str | list[str] | None = None,
    align_metric: str | None = None,
    align_algorithm: str | None = None,
    align_minimizer_options: dict[str, Any] | None = None,
    mapping_species: str | list[str] | None = None,
    align_tolerance: float | None = None,
    use_reference_centers: bool = True
) -> Trajectory:
    """Create a Trajectory with polyhedral sites using reference-based workflow.
    
    Creates a trajectory object with polyhedral sites by identifying coordination
    environments in a reference structure and mapping them to corresponding sites
    in the target structure. Each site is defined by a polyhedron formed by vertex
    atoms around a central atom.
    
    Args:
        structure: Pymatgen Structure object containing the target structure to analyse.
        reference_structure: Pymatgen Structure object defining ideal coordination 
            environments to identify and map.
        mobile_species: Species string (e.g., "Li") or list of species strings 
            identifying the mobile atoms to track.
        centre_species: Atomic species at the centre of coordination environments
            in the reference structure.
        vertex_species: Atomic species at vertices of coordination polyhedra. 
            Can be a single species string or list of species strings.
        cutoff: Maximum distance in Ångströms for vertex atoms to be considered 
            part of the coordination environment.
        n_vertices: Number of vertex atoms required for each polyhedral site.
        label: Optional string label to assign to all created sites.
        align: Whether to perform structure alignment before site mapping.
        align_species: Species to use for structure alignment. If None and 
            mapping_species is specified, mapping_species will be used.
            If neither is given, all species are used and both structures
            must have the same composition.
        align_metric: Deprecated; has no effect, except that 'max_dist'
            raises ValueError. Will be removed in version 2.0.
        align_algorithm: Deprecated; has no effect. Will be removed in
            version 2.0.
        align_minimizer_options: Deprecated; has no effect. Will be
            removed in version 2.0.
        align_tolerance: Deprecated; has no effect. Will be removed in
            version 2.0.
        mapping_species: Species to use for mapping sites between structures. 
            If None, align_species will be used.
        use_reference_centers: Whether to use reference centers for 
            PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
            Default is True.
            
    Returns:
        Trajectory: Configured trajectory object ready for site analysis.
        
    Raises:
        ValueError: If no coordination environments are found, if required 
            structures are not set, if structure alignment fails, or if align_metric is 'max_dist'.
    """
    _check_deprecated_alignment_options(
        align_metric=align_metric, align_algorithm=align_algorithm,
        align_minimizer_options=align_minimizer_options, align_tolerance=align_tolerance)
    builder = TrajectoryBuilder()
    return (builder
            .with_structure(structure)
            .with_reference_structure(reference_structure)
            .with_mobile_species(mobile_species)
            .with_structure_alignment(align=align, align_species=align_species)
            .with_site_mapping(mapping_species)
            .with_polyhedral_sites(
                centre_species,
                vertex_species,
                cutoff,
                n_vertices,
                label,
                use_reference_centers)
            .build())


def create_trajectory_with_dynamic_voronoi_sites(
    structure: Structure, 
    reference_structure: Structure, 
    mobile_species: str | list[str], 
    centre_species: str, 
    reference_species: str | list[str], 
    cutoff: float, 
    n_reference: int, 
    label: str | None = None,
    align: bool = True,
    align_species: str | list[str] | None = None,
    align_metric: str | None = None,
    align_algorithm: str | None = None,
    align_minimizer_options: dict[str, Any] | None = None,
    mapping_species: str | list[str] | None = None,
    align_tolerance: float | None = None,
    use_reference_centers: bool = True
) -> Trajectory:
    """Create a Trajectory with dynamic Voronoi sites using reference-based workflow.
    
    Creates a trajectory object with dynamic Voronoi sites where site centres are
    dynamically calculated from the positions of reference atoms. Unlike fixed 
    Voronoi sites, these adapt to structural changes as the reference atoms move,
    making them suitable for analysing deformable frameworks.
    
    Args:
        structure: Pymatgen Structure object containing the target structure to analyse.
        reference_structure: Pymatgen Structure object defining coordination 
            environments to identify and map.
        mobile_species: Species string (e.g., "Li") or list of species strings 
            identifying the mobile atoms to track.
        centre_species: Atomic species at centres where dynamic sites will be located
            in the reference structure.
        reference_species: Atomic species used as reference atoms to define dynamic 
            site centres. Can be a single species string or list of species strings.
        cutoff: Maximum distance in Ångströms for reference atoms to be considered 
            part of the coordination environment.
        n_reference: Number of reference atoms required for each dynamic site.
        label: Optional string label to assign to all created sites.
        align: Whether to perform structure alignment before site mapping.
        align_species: Species to use for structure alignment. If None and 
            mapping_species is specified, mapping_species will be used.
            If neither is given, all species are used and both structures
            must have the same composition.
        align_metric: Deprecated; has no effect, except that 'max_dist'
            raises ValueError. Will be removed in version 2.0.
        align_algorithm: Deprecated; has no effect. Will be removed in
            version 2.0.
        align_minimizer_options: Deprecated; has no effect. Will be
            removed in version 2.0.
        align_tolerance: Deprecated; has no effect. Will be removed in
            version 2.0.
        mapping_species: Species to use for mapping sites between structures. 
            If None, align_species will be used.
        use_reference_centers: Whether to use reference centers for 
            PBC handling. See TrajectoryBuilder.with_polyhedral_sites() for details.
            Default is True.
            
    Returns:
        Trajectory: Configured trajectory object ready for site analysis.
        
    Raises:
        ValueError: If no coordination environments are found, if required 
            structures are not set, if structure alignment fails, or if align_metric is 'max_dist'.
    """
    _check_deprecated_alignment_options(
        align_metric=align_metric, align_algorithm=align_algorithm,
        align_minimizer_options=align_minimizer_options, align_tolerance=align_tolerance)
    builder = TrajectoryBuilder()
    return (builder
            .with_structure(structure)
            .with_reference_structure(reference_structure)
            .with_mobile_species(mobile_species)
            .with_structure_alignment(align=align, align_species=align_species)
            .with_site_mapping(mapping_species)
            .with_dynamic_voronoi_sites(
                centre_species,
                reference_species,
                cutoff,
                n_reference,
                label,
                use_reference_centers)
            .build())
