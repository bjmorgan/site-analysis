"""Benchmark per-frame site assignment for all four site types.

Isolates the assign_site_occupations hot path by pre-building the
trajectory and then timing only the per-frame analyse_structure calls
(which include coord assignment + site assignment).

Uses the Li6PS5Cl argyrodite MD trajectory.

Usage:
    python tests/benchmark_all_site_types.py [--repeats N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Xdatcar

from site_analysis import TrajectoryBuilder
from site_analysis.site import Site

DATA_DIR = Path(__file__).resolve().parent.parent / "docs" / "source"
ARGYRODITE_XDATCAR = DATA_DIR / "tutorials" / "data" / "Li6PS5Cl_0p_XDATCAR.gz"


def build_reference_structure():
    lattice = Lattice.cubic(a=10.155)
    coords = np.array([
        [0.5, 0.5, 0.5],
        [0.9, 0.9, 0.6],
        [0.77, 0.585, 0.585],
        [0.25, 0.25, 0.25],
        [0.15, 0.15, 0.15],
        [0.0, 0.183, 0.183],
        [0.0, 0.0, 0.0],
        [0.75, 0.25, 0.25],
        [0.11824, 0.11824, 0.38176],
    ])
    return Structure.from_spacegroup(
        sg="F-43m", lattice=lattice,
        species=["P", "Li", "Mg", "Na", "Be", "K", "S", "S", "S"],
        coords=coords,
    ) * [2, 2, 2]


def build_trajectory(site_type, structures, reference_structure):
    Site._newid = 0
    builder = (
        TrajectoryBuilder()
        .with_structure(structures[0])
        .with_reference_structure(reference_structure)
        .with_mobile_species("Li")
        .with_structure_alignment(align_species="P")
        .with_site_mapping(mapping_species=["S", "Cl"])
    )

    if site_type == "polyhedral":
        for sp, label in [("Li", "1"), ("Mg", "2"), ("Na", "3"),
                          ("Be", "4"), ("K", "5")]:
            builder.with_polyhedral_sites(
                centre_species=sp, vertex_species="S",
                cutoff=3.0, n_vertices=4, label=label)

    elif site_type == "voronoi":
        li_idx = [i for i, s in enumerate(reference_structure)
                  if s.species_string == "Li"]
        centres = [reference_structure[i].frac_coords.tolist() for i in li_idx]
        builder.with_voronoi_sites(centres=centres)

    elif site_type == "dynamic_voronoi":
        for sp, label in [("Li", "1"), ("Mg", "2"), ("Na", "3"),
                          ("Be", "4"), ("K", "5")]:
            builder.with_dynamic_voronoi_sites(
                centre_species=sp, reference_species="S",
                cutoff=3.0, n_reference=4, label=label)

    elif site_type == "spherical":
        centres = []
        for sp in ["Li", "Mg", "Na", "Be", "K"]:
            idx = [i for i, s in enumerate(reference_structure)
                   if s.species_string == sp]
            centres.extend([reference_structure[i].frac_coords.tolist()
                           for i in idx])
        builder.with_spherical_sites(centres=centres, radii=1.5)

    return builder.build()


def benchmark_site_type(site_type, structures, reference_structure, repeats):
    trajectory = build_trajectory(site_type, structures, reference_structure)

    # Warm up: first frame populates caches
    trajectory.append_timestep(structures[0], t=0)

    # Benchmark: repeated passes through the trajectory
    all_structures = structures * repeats
    start = time.perf_counter()
    for s in all_structures:
        trajectory.analyse_structure(s)
    elapsed = time.perf_counter() - start

    n_frames = len(all_structures)
    n_sites = len(trajectory.sites)
    n_atoms = len(trajectory.atoms)

    return {
        "site_type": site_type,
        "n_sites": n_sites,
        "n_atoms": n_atoms,
        "n_frames": n_frames,
        "elapsed_s": elapsed,
        "ms_per_frame": elapsed / n_frames * 1000,
        "frames_per_s": n_frames / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not ARGYRODITE_XDATCAR.exists():
        print(f"Data file not found: {ARGYRODITE_XDATCAR}")
        sys.exit(1)

    print("Loading trajectory data...")
    xdatcar = Xdatcar(str(ARGYRODITE_XDATCAR))
    structures = xdatcar.structures
    n_frames = len(structures)
    reference_structure = build_reference_structure()
    print(f"  {n_frames} frames, {len(reference_structure)} atoms in reference")
    print(f"  {args.repeats} repeats = {n_frames * args.repeats} total frames\n")

    print(f"{'Site type':<20} {'Sites':>6} {'Atoms':>6} "
          f"{'ms/frame':>10} {'frames/s':>10}")
    print("-" * 60)

    for site_type in ["polyhedral", "voronoi", "dynamic_voronoi", "spherical"]:
        result = benchmark_site_type(
            site_type, structures, reference_structure, args.repeats)
        print(f"{result['site_type']:<20} {result['n_sites']:>6} "
              f"{result['n_atoms']:>6} {result['ms_per_frame']:>10.2f} "
              f"{result['frames_per_s']:>10.1f}")


if __name__ == "__main__":
    main()
