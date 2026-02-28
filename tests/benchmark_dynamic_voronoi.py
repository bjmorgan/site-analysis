"""Benchmark dynamic Voronoi site analysis on real MD data.

Profiles the full trajectory analysis workflow to identify where time
is spent: centre calculation, Voronoi assignment, atom coord
assignment, and bookkeeping (update_occupation, trajectory tracking).

Usage:
    python tests/benchmark_dynamic_voronoi.py [--repeats N] [--profile]
"""

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Xdatcar

from site_analysis import TrajectoryBuilder
from site_analysis.dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection
from site_analysis.site import Site

DATA_DIR = Path(__file__).resolve().parent.parent / "docs" / "source"
ARGYRODITE_XDATCAR = DATA_DIR / "tutorials" / "data" / "Li6PS5Cl_0p_XDATCAR.gz"


def build_dynamic_voronoi_trajectory(structures):
    """Build a trajectory with dynamic Voronoi sites for argyrodite."""
    Site._newid = 0

    lattice = Lattice.cubic(a=10.155)
    coords = np.array([
        [0.5, 0.5, 0.5],
        [0.9, 0.9, 0.6],
        [0.23, 0.92, 0.09],
        [0.25, 0.25, 0.25],
        [0.15, 0.15, 0.15],
        [0.0, 0.183, 0.183],
        [0.0, 0.0, 0.0],
        [0.75, 0.25, 0.25],
        [0.11824, 0.11824, 0.38176],
    ])
    reference_structure = Structure.from_spacegroup(
        sg="F-43m",
        lattice=lattice,
        species=["P", "Li", "Mg", "Na", "Be", "K", "S", "S", "S"],
        coords=coords,
    ) * [2, 2, 2]

    builder = (
        TrajectoryBuilder()
        .with_structure(structures[0])
        .with_reference_structure(reference_structure)
        .with_mobile_species("Li")
        .with_structure_alignment(align_species="P")
        .with_site_mapping(mapping_species=["S", "Cl"])
    )

    for centre_species, label in [
        ("Li", "type 1"), ("Mg", "type 2"), ("Na", "type 3"),
        ("Be", "type 4"), ("K", "type 5"),
    ]:
        builder.with_dynamic_voronoi_sites(
            centre_species=centre_species,
            reference_species="S",
            cutoff=3.0,
            n_reference=4,
            label=label,
        )

    return builder.build()


def _original_analyse_structure(self, atoms, structure):
    """Original analysis without PBC shift caching.

    Simulates the pre-optimisation path by clearing each site's
    PBC caches before computing the centre, forcing the full
    27-image unwrap every timestep.
    """
    for atom in atoms:
        atom.assign_coords(structure)
    for site in self.sites:
        site._pbc_image_shifts = None
        site._pbc_cached_raw_frac = None
        site.calculate_centre(structure)
    self.assign_site_occupations(atoms, structure)


def benchmark_full_workflow(trajectory, structures, repeats):
    """Benchmark the full trajectory_from_structures workflow."""
    all_structures = structures * repeats
    trajectory.reset()

    start = time.perf_counter()
    trajectory.trajectory_from_structures(all_structures)
    elapsed = time.perf_counter() - start
    return elapsed, len(all_structures)


def profile_full_workflow(trajectory, structures, repeats):
    """Profile the full workflow with cProfile."""
    all_structures = structures * repeats
    trajectory.reset()

    profiler = cProfile.Profile()
    profiler.enable()
    trajectory.trajectory_from_structures(all_structures)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    print("\n--- Top 30 by cumulative time ---")
    stats.print_stats(30)

    print("\n--- Top 20 by total (self) time ---")
    stats.sort_stats("tottime")
    stats.print_stats(20)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dynamic Voronoi site analysis")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of times to repeat the trajectory (default: 3)")
    parser.add_argument("--profile", action="store_true",
                        help="Run cProfile and print breakdown")
    args = parser.parse_args()

    if not ARGYRODITE_XDATCAR.exists():
        print(f"Data file not found: {ARGYRODITE_XDATCAR}")
        sys.exit(1)

    print("Loading argyrodite trajectory...")
    structures = Xdatcar(str(ARGYRODITE_XDATCAR)).structures
    n_frames = len(structures)
    total_frames = n_frames * args.repeats
    print(f"  {n_frames} frames x {args.repeats} repeats = {total_frames} frames")

    trajectory = build_dynamic_voronoi_trajectory(structures)
    n_sites = len(trajectory.sites)
    n_atoms = len(trajectory.atoms)
    print(f"  {n_sites} dynamic Voronoi sites, {n_atoms} mobile atoms")

    if args.profile:
        print("\nProfiling full workflow...")
        profile_full_workflow(trajectory, structures, args.repeats)
    else:
        # Original (no caching) path
        print("\nOriginal (no PBC shift caching):")
        trajectory_orig = build_dynamic_voronoi_trajectory(structures)
        with patch.object(DynamicVoronoiSiteCollection, "analyse_structure",
                          _original_analyse_structure):
            elapsed_orig, n = benchmark_full_workflow(
                trajectory_orig, structures, args.repeats)
        print(f"  {elapsed_orig:.2f}s total, {elapsed_orig/n*1000:.1f}ms/frame")

        # Optimised path
        print("\nOptimised (bulk extraction + PBC shift caching):")
        trajectory_opt = build_dynamic_voronoi_trajectory(structures)
        elapsed_opt, n = benchmark_full_workflow(
            trajectory_opt, structures, args.repeats)
        print(f"  {elapsed_opt:.2f}s total, {elapsed_opt/n*1000:.1f}ms/frame")

        # Summary
        print(f"\n--- Summary ---")
        print(f"  Original:  {elapsed_orig:.2f}s  ({elapsed_orig/n*1000:.1f}ms/frame)")
        print(f"  Optimised: {elapsed_opt:.2f}s  ({elapsed_opt/n*1000:.1f}ms/frame)")
        print(f"  Speedup:   {elapsed_orig/elapsed_opt:.2f}x")


if __name__ == "__main__":
    main()
