"""Benchmark comparing containment strategies on real MD data.

Isolates the site assignment cost (assign_vertex_coords + contains_point)
by running the argyrodite trajectory multiple times through the same
trajectory object, emulating a longer trajectory.

Usage:
    python tests/benchmark_containment.py [--repeats N]
"""

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Xdatcar

from site_analysis import TrajectoryBuilder
from site_analysis.containment import HAS_NUMBA
from site_analysis.polyhedral_site_collection import PolyhedralSiteCollection
from site_analysis.site import Site

DATA_DIR = Path(__file__).resolve().parent.parent / "docs" / "source"
ARGYRODITE_XDATCAR = DATA_DIR / "tutorials" / "data" / "Li6PS5Cl_0p_XDATCAR.gz"


def build_argyrodite_trajectory(structures):
    """Build a trajectory object for the argyrodite analysis."""
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
        builder.with_polyhedral_sites(
            centre_species=centre_species,
            vertex_species="S",
            cutoff=3.0,
            n_vertices=4,
            label=label,
        )

    return builder.build()


def _eager_analyse_structure(self, atoms, structure):
    """Original eager analysis: assign all vertex coords upfront."""
    for a in atoms:
        a.assign_coords(structure)
    for s in self.sites:
        s.assign_vertex_coords(structure)
    self.assign_site_occupations(atoms, structure)


def benchmark_assignment(trajectory, structures, repeats):
    """Benchmark just the per-frame analysis loop."""
    # Warm up: run one frame to trigger lazy cache creation
    trajectory.append_timestep(structures[0], t=0)

    # Now time repeated passes over the trajectory
    all_structures = structures * repeats
    start = time.perf_counter()
    for i, s in enumerate(all_structures):
        trajectory.analyse_structure(s)
    elapsed = time.perf_counter() - start
    return elapsed, len(all_structures)


def main():
    parser = argparse.ArgumentParser(description="Benchmark containment methods")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of times to repeat the trajectory (default: 3)")
    args = parser.parse_args()

    if not ARGYRODITE_XDATCAR.exists():
        print(f"Data file not found: {ARGYRODITE_XDATCAR}")
        sys.exit(1)

    if not HAS_NUMBA:
        print("numba is not available -- only Delaunay will be benchmarked")

    print(f"Loading argyrodite trajectory...")
    structures = Xdatcar(str(ARGYRODITE_XDATCAR)).structures
    n_frames = len(structures)
    total_frames = n_frames * args.repeats
    print(f"  {n_frames} frames x {args.repeats} repeats = {total_frames} frames")

    # Count sites and atoms
    trajectory = build_argyrodite_trajectory(structures)
    n_sites = len(trajectory.sites)
    n_atoms = len(trajectory.atoms)
    print(f"  {n_sites} polyhedral sites, {n_atoms} mobile atoms")

    # --- Delaunay comparisons ---
    # Eager Delaunay (original baseline)
    print(f"\nDelaunay (eager coords):")
    trajectory_de = build_argyrodite_trajectory(structures)
    with patch("site_analysis.polyhedral_site.HAS_NUMBA", False), \
         patch.object(PolyhedralSiteCollection, "analyse_structure",
                      _eager_analyse_structure):
        elapsed_de, n = benchmark_assignment(trajectory_de, structures, args.repeats)
    print(f"  {elapsed_de:.2f}s total, {elapsed_de/n*1000:.1f}ms/frame")

    print(f"\nDelaunay (lazy coords):")
    trajectory_dl = build_argyrodite_trajectory(structures)
    with patch("site_analysis.polyhedral_site.HAS_NUMBA", False):
        elapsed_dl, n = benchmark_assignment(trajectory_dl, structures, args.repeats)
    print(f"  {elapsed_dl:.2f}s total, {elapsed_dl/n*1000:.1f}ms/frame")

    if HAS_NUMBA:
        # --- Numba comparisons ---
        print(f"\nNumba (eager coords):")
        trajectory_ne = build_argyrodite_trajectory(structures)
        with patch.object(PolyhedralSiteCollection, "analyse_structure",
                          _eager_analyse_structure):
            elapsed_ne, n = benchmark_assignment(trajectory_ne, structures, args.repeats)
        print(f"  {elapsed_ne:.2f}s total, {elapsed_ne/n*1000:.1f}ms/frame")

        print(f"\nNumba (lazy coords):")
        trajectory_nl = build_argyrodite_trajectory(structures)
        elapsed_nl, n = benchmark_assignment(trajectory_nl, structures, args.repeats)
        print(f"  {elapsed_nl:.2f}s total, {elapsed_nl/n*1000:.1f}ms/frame")

        # Summary
        print(f"\n--- Summary ---")
        print(f"  Delaunay eager:  {elapsed_de:.2f}s  ({elapsed_de/n*1000:.1f}ms/frame)")
        print(f"  Delaunay lazy:   {elapsed_dl:.2f}s  ({elapsed_dl/n*1000:.1f}ms/frame)")
        print(f"  Numba eager:     {elapsed_ne:.2f}s  ({elapsed_ne/n*1000:.1f}ms/frame)")
        print(f"  Numba lazy:      {elapsed_nl:.2f}s  ({elapsed_nl/n*1000:.1f}ms/frame)")
        print(f"\n  Lazy coord speedup (Delaunay): {elapsed_de/elapsed_dl:.2f}x")
        print(f"  Lazy coord speedup (Numba):    {elapsed_ne/elapsed_nl:.2f}x")
        print(f"  Overall speedup (eager Delaunay vs lazy Numba): {elapsed_de/elapsed_nl:.2f}x")


if __name__ == "__main__":
    main()
