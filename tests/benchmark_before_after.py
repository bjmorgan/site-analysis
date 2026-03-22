"""Simple before/after benchmark for pymatgen decoupling.

Build trajectory, then time trajectory_from_structures across real MD data.

Usage:
    python tests/benchmark_before_after.py
"""

import sys
import time
from pathlib import Path

# Ensure we import from the repo containing this script, not an installed package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Xdatcar

from site_analysis import TrajectoryBuilder
from site_analysis.site import Site

DATA_DIR = Path(__file__).resolve().parent.parent / "docs" / "source"
ARGYRODITE_XDATCAR = DATA_DIR / "tutorials" / "data" / "Li6PS5Cl_0p_XDATCAR.gz"


def build_reference():
    lattice = Lattice.cubic(a=10.155)
    coords = np.array([
        [0.5, 0.5, 0.5], [0.9, 0.9, 0.6], [0.77, 0.585, 0.585],
        [0.25, 0.25, 0.25], [0.15, 0.15, 0.15], [0.0, 0.183, 0.183],
        [0.0, 0.0, 0.0], [0.75, 0.25, 0.25], [0.11824, 0.11824, 0.38176],
    ])
    return Structure.from_spacegroup(
        sg="F-43m", lattice=lattice,
        species=["P", "Li", "Mg", "Na", "Be", "K", "S", "S", "S"],
        coords=coords,
    ) * [2, 2, 2]


def run_trajectory(site_type, structures, ref):
    """Build and run a full trajectory. Returns (n_sites, elapsed_s, n_frames)."""
    Site._newid = 0
    builder = (
        TrajectoryBuilder()
        .with_structure(structures[0])
        .with_reference_structure(ref)
        .with_mobile_species("Li")
        .with_structure_alignment(align_species="P")
        .with_site_mapping(mapping_species=["S", "Cl"])
    )

    if site_type == "polyhedral":
        for sp, l in [("Li", "1"), ("Mg", "2"), ("Na", "3"), ("Be", "4"), ("K", "5")]:
            builder.with_polyhedral_sites(
                centre_species=sp, vertex_species="S", cutoff=3.0, n_vertices=4, label=l)
    elif site_type == "voronoi":
        centres = []
        for sp in ["Li", "Mg", "Na", "Be", "K"]:
            idx = [i for i, s in enumerate(ref) if s.species_string == sp]
            centres.extend([ref[i].frac_coords.tolist() for i in idx])
        builder.with_voronoi_sites(centres=centres)
    elif site_type == "dynamic_voronoi":
        for sp, l in [("Li", "1"), ("Mg", "2"), ("Na", "3"), ("Be", "4"), ("K", "5")]:
            builder.with_dynamic_voronoi_sites(
                centre_species=sp, reference_species="S", cutoff=3.0, n_reference=4, label=l)
    elif site_type == "spherical":
        centres = []
        for sp in ["Li", "Mg", "Na", "Be", "K"]:
            idx = [i for i, s in enumerate(ref) if s.species_string == sp]
            centres.extend([ref[i].frac_coords.tolist() for i in idx])
        builder.with_spherical_sites(centres=centres, radii=1.5)

    trajectory = builder.build()
    n_sites = len(trajectory.sites)

    # Time the actual MD trajectory analysis
    start = time.perf_counter()
    for i, s in enumerate(structures):
        trajectory.append_timestep(s, t=i)
    elapsed = time.perf_counter() - start

    return n_sites, elapsed, len(structures)


def main():
    # Verify which code we're running
    import site_analysis
    import inspect
    from site_analysis.dynamic_voronoi_site_collection import DynamicVoronoiSiteCollection as D
    src = inspect.getsource(D.assign_site_occupations)
    dist_fn = "get_all_distances" if "get_all_distances" in src else "all_mic_distances"
    print(f"Code: {site_analysis.__file__}")
    print(f"Distance function: {dist_fn}")

    print("\nLoading Li6PS5Cl trajectory...")
    xdatcar = Xdatcar(str(ARGYRODITE_XDATCAR))
    structures = xdatcar.structures
    ref = build_reference()
    print(f"  {len(structures)} frames, {len(ref)} atoms in reference\n")

    print(f"{'Site type':<20} {'Sites':>6} {'Total (s)':>10} {'ms/frame':>10}")
    print("-" * 50)

    for site_type in ["polyhedral", "voronoi", "dynamic_voronoi", "spherical"]:
        n_sites, elapsed, n_frames = run_trajectory(site_type, structures, ref)
        ms_per_frame = elapsed / n_frames * 1000
        print(f"{site_type:<20} {n_sites:>6} {elapsed:>10.3f} {ms_per_frame:>10.2f}")


if __name__ == "__main__":
    main()
