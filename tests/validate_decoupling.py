"""Validation and benchmarking for pymatgen decoupling (#59).

Runs the full Li6PS5Cl argyrodite analysis with all four site types
(polyhedral, Voronoi, dynamic Voronoi, spherical) using the high-level
TrajectoryBuilder API. Captures per-frame site assignments and timing.

This script uses only the public TrajectoryBuilder API, which is
unchanged between the pre-decoupling baseline and the current code.
Run on both commits and compare the JSON output files for correctness.

Usage:
    # On current code:
    python tests/validate_decoupling.py --output results_after.json

    # On baseline (git checkout 7379d58):
    python tests/validate_decoupling.py --output results_before.json

    # Compare:
    python tests/validate_decoupling.py --compare results_before.json results_after.json
"""

import argparse
import json
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

# Reference structure for Li6PS5Cl argyrodite (2x2x2 supercell)
LATTICE = Lattice.cubic(a=10.155)
REF_COORDS = np.array([
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
REF_SPECIES = ["P", "Li", "Mg", "Na", "Be", "K", "S", "S", "S"]


def build_reference_structure():
    return Structure.from_spacegroup(
        sg="F-43m", lattice=LATTICE, species=REF_SPECIES, coords=REF_COORDS,
    ) * [2, 2, 2]


def build_trajectory(site_type, structures, reference_structure):
    """Build a trajectory for one site type."""
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

    elif site_type == "voronoi":
        # Use all Li-related site centres (matching other site types)
        centres = []
        for sp in ["Li", "Mg", "Na", "Be", "K"]:
            indices = [i for i, s in enumerate(reference_structure)
                       if s.species_string == sp]
            centres.extend([reference_structure[i].frac_coords.tolist()
                           for i in indices])
        builder.with_voronoi_sites(centres=centres)

    elif site_type == "dynamic_voronoi":
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

    elif site_type == "spherical":
        # Use all Li-related site centres with a 1.5 A radius
        centres = []
        for sp in ["Li", "Mg", "Na", "Be", "K"]:
            indices = [i for i, s in enumerate(reference_structure)
                       if s.species_string == sp]
            centres.extend([reference_structure[i].frac_coords.tolist()
                           for i in indices])
        builder.with_spherical_sites(centres=centres, radii=1.5)

    else:
        raise ValueError(f"Unknown site type: {site_type}")

    return builder.build()


def run_analysis(site_type, structures, reference_structure, n_repeats=1):
    """Run analysis and capture results."""
    trajectory = build_trajectory(site_type, structures, reference_structure)

    n_sites = len(trajectory.sites)
    n_atoms = len(trajectory.atoms)

    # Run the trajectory analysis with timing
    all_structures = structures * n_repeats
    start = time.perf_counter()
    for i, s in enumerate(all_structures):
        trajectory.append_timestep(s, t=i)
    elapsed = time.perf_counter() - start

    # Capture per-atom trajectories (site assignments over time)
    atom_trajectories = {}
    for atom in trajectory.atoms:
        atom_trajectories[str(atom.index)] = [
            int(s) if s is not None else None
            for s in atom.trajectory
        ]

    # Capture site occupancy counts
    site_data = {}
    for site in trajectory.sites:
        site_data[str(site.index)] = {
            "label": site.label,
            "average_occupation": float(site.average_occupation)
                if hasattr(site, 'average_occupation') and site.average_occupation is not None
                else None,
        }

    return {
        "site_type": site_type,
        "n_sites": n_sites,
        "n_atoms": n_atoms,
        "n_frames": len(all_structures),
        "elapsed_s": elapsed,
        "ms_per_frame": elapsed / len(all_structures) * 1000,
        "atom_trajectories": atom_trajectories,
        "site_data": site_data,
    }


def run_all(structures, reference_structure, n_repeats=1):
    """Run analysis for all four site types."""
    results = {}
    for site_type in ["polyhedral", "voronoi", "dynamic_voronoi", "spherical"]:
        print(f"  Running {site_type}...", end=" ", flush=True)
        try:
            result = run_analysis(
                site_type, structures, reference_structure, n_repeats)
            print(f"{result['n_sites']} sites, {result['ms_per_frame']:.2f} ms/frame")
            results[site_type] = result
        except Exception as e:
            print(f"FAILED: {e}")
            results[site_type] = {"error": str(e)}
    return results


def compare_results(before_path, after_path):
    """Compare two result files for correctness and performance."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print("\n=== Correctness Comparison ===\n")

    all_match = True
    for site_type in ["polyhedral", "voronoi", "dynamic_voronoi", "spherical"]:
        b = before.get(site_type, {})
        a = after.get(site_type, {})

        if "error" in b or "error" in a:
            print(f"{site_type}: SKIPPED (error in one or both)")
            if "error" in b:
                print(f"  Before: {b['error']}")
            if "error" in a:
                print(f"  After: {a['error']}")
            all_match = False
            continue

        # Compare site counts
        if b["n_sites"] != a["n_sites"]:
            print(f"{site_type}: MISMATCH — {b['n_sites']} sites before, "
                  f"{a['n_sites']} after")
            all_match = False
            continue

        # Compare atom trajectories frame by frame
        b_traj = b["atom_trajectories"]
        a_traj = a["atom_trajectories"]

        mismatches = 0
        total_assignments = 0
        for atom_idx in b_traj:
            if atom_idx not in a_traj:
                mismatches += len(b_traj[atom_idx])
                total_assignments += len(b_traj[atom_idx])
                continue
            for frame_i, (bv, av) in enumerate(
                    zip(b_traj[atom_idx], a_traj[atom_idx])):
                total_assignments += 1
                if bv != av:
                    mismatches += 1

        if mismatches == 0:
            print(f"{site_type}: MATCH ({total_assignments} assignments identical)")
        else:
            print(f"{site_type}: MISMATCH — {mismatches}/{total_assignments} "
                  f"assignments differ ({mismatches/total_assignments*100:.1f}%)")
            all_match = False

    print("\n=== Performance Comparison ===\n")
    print(f"{'Site type':<20} {'Before (ms/f)':>15} {'After (ms/f)':>15} {'Speedup':>10}")
    print("-" * 65)
    for site_type in ["polyhedral", "voronoi", "dynamic_voronoi", "spherical"]:
        b = before.get(site_type, {})
        a = after.get(site_type, {})
        if "error" in b or "error" in a:
            print(f"{site_type:<20} {'error':>15} {'error':>15} {'—':>10}")
            continue
        b_ms = b["ms_per_frame"]
        a_ms = a["ms_per_frame"]
        speedup = b_ms / a_ms if a_ms > 0 else float("inf")
        print(f"{site_type:<20} {b_ms:>15.2f} {a_ms:>15.2f} {speedup:>9.2f}x")

    if all_match:
        print("\nAll site assignments match. Decoupling is correctness-preserving.")
    else:
        print("\nWARNING: Some site assignments differ. Investigate before closing #59.")

    return all_match


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two result files")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of times to repeat the trajectory")
    args = parser.parse_args()

    if args.compare:
        success = compare_results(args.compare[0], args.compare[1])
        sys.exit(0 if success else 1)

    if not ARGYRODITE_XDATCAR.exists():
        print(f"Data file not found: {ARGYRODITE_XDATCAR}")
        sys.exit(1)

    print("Loading trajectory data...")
    xdatcar = Xdatcar(str(ARGYRODITE_XDATCAR))
    structures = xdatcar.structures
    print(f"  {len(structures)} frames")

    reference_structure = build_reference_structure()
    print(f"  Reference: {len(reference_structure)} atoms")

    print(f"\nRunning analysis (repeats={args.repeats})...")
    results = run_all(structures, reference_structure, n_repeats=args.repeats)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Print summary
        print("\n=== Summary ===\n")
        print(f"{'Site type':<20} {'Sites':>8} {'Atoms':>8} {'ms/frame':>12}")
        print("-" * 50)
        for site_type, result in results.items():
            if "error" in result:
                print(f"{site_type:<20} {'ERROR':>8}")
            else:
                print(f"{site_type:<20} {result['n_sites']:>8} "
                      f"{result['n_atoms']:>8} {result['ms_per_frame']:>12.2f}")


if __name__ == "__main__":
    main()
