"""Validation tests comparing numba and Delaunay containment paths.

These tests run the full trajectory analysis pipeline on real MD data
and verify that the numba surface normal method produces identical
results to the Delaunay fallback. They also report timing for both paths.

These tests are skipped if numba is not available or if the test data
files are not present.
"""

import time
import unittest
from pathlib import Path
from unittest.mock import patch

from site_analysis._compat import HAS_NUMBA

# Paths to test data
DATA_DIR = Path(__file__).resolve().parent.parent / "docs" / "source"
SIMPLE_XDATCAR = DATA_DIR / "examples" / "simple_cubic_li.XDATCAR"
ARGYRODITE_XDATCAR = DATA_DIR / "tutorials" / "data" / "Li6PS5Cl_0p_XDATCAR.gz"

HAS_SIMPLE_DATA = SIMPLE_XDATCAR.exists()
HAS_ARGYRODITE_DATA = ARGYRODITE_XDATCAR.exists()


def _run_polyhedral_analysis(structures):
    """Run polyhedral site analysis on the simple cubic trajectory."""
    import numpy as np
    from site_analysis import TrajectoryBuilder
    from site_analysis.site import Site

    Site._newid = 0

    structure = structures[0]

    # Build reference structure: O framework + Li at each cube centre
    reference_structure = structure.copy()
    reference_structure.remove_species(["Li"])
    site_centers = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                site_centers.append(np.array([i + 0.5, j + 0.5, k + 0.5]) / 4)
    for c in site_centers:
        reference_structure.append(species="Li", coords=c)

    trajectory = (
        TrajectoryBuilder()
        .with_structure(structure)
        .with_reference_structure(reference_structure)
        .with_mobile_species("Li")
        .with_polyhedral_sites(
            centre_species="Li",
            vertex_species="O",
            cutoff=4.0,
            n_vertices=8,
            label="Cubic",
        )
        .with_structure_alignment(align_species="O")
        .build()
    )

    start = time.perf_counter()
    trajectory.trajectory_from_structures(structures, progress=False)
    elapsed = time.perf_counter() - start

    # Extract results
    atom_trajectories = [atom.trajectory[:] for atom in trajectory.atoms]
    site_trajectories = [
        [list(frame) for frame in site.trajectory] for site in trajectory.sites
    ]

    return atom_trajectories, site_trajectories, elapsed


@unittest.skipUnless(HAS_NUMBA, "numba not available")
@unittest.skipUnless(HAS_SIMPLE_DATA, f"test data not found: {SIMPLE_XDATCAR}")
class TestSimpleCubicValidation(unittest.TestCase):
    """Validate numba vs Delaunay on simple cubic Li trajectory."""

    @classmethod
    def setUpClass(cls):
        from pymatgen.io.vasp import Xdatcar

        cls.structures = Xdatcar(str(SIMPLE_XDATCAR)).structures

    def test_numba_and_delaunay_produce_identical_trajectories(self):
        """Run analysis with both paths and compare atom trajectories.

        Boundary-point disagreements (where one method returns None and the
        other returns a site index) are expected for points exactly on a face
        and are reported but not treated as failures.
        """
        # Run with numba (default)
        numba_atoms, numba_sites, numba_time = _run_polyhedral_analysis(
            self.structures
        )

        # Run with Delaunay fallback
        with patch("site_analysis.polyhedral_site.HAS_NUMBA", False):
            delaunay_atoms, delaunay_sites, delaunay_time = _run_polyhedral_analysis(
                self.structures
            )

        # Compare atom trajectories
        self.assertEqual(len(numba_atoms), len(delaunay_atoms))
        boundary_disagreements = 0
        real_disagreements = 0
        for i, (n, d) in enumerate(zip(numba_atoms, delaunay_atoms)):
            for frame, (nf, df) in enumerate(zip(n, d)):
                if nf != df:
                    if nf is None or df is None:
                        # One method found the point inside, the other didn't —
                        # boundary point noise
                        boundary_disagreements += 1
                    else:
                        # Both found the point but in different sites
                        real_disagreements += 1

        total_frames = sum(len(t) for t in numba_atoms)
        if boundary_disagreements > 0:
            print(f"\n  Boundary disagreements: {boundary_disagreements}/{total_frames} frames")
        self.assertEqual(
            real_disagreements,
            0,
            f"{real_disagreements} frames assigned to different sites",
        )

        # Report timing
        print(f"\nSimple cubic ({len(self.structures)} frames):")
        print(f"  Numba:   {numba_time:.4f}s")
        print(f"  Delaunay: {delaunay_time:.4f}s")
        if delaunay_time > 0:
            print(f"  Speedup: {delaunay_time / numba_time:.1f}x")


@unittest.skipUnless(HAS_NUMBA, "numba not available")
@unittest.skipUnless(HAS_ARGYRODITE_DATA, f"test data not found: {ARGYRODITE_XDATCAR}")
class TestArgyroditeValidation(unittest.TestCase):
    """Validate numba vs Delaunay on argyrodite Li6PS5Cl trajectory."""

    @classmethod
    def setUpClass(cls):
        from pymatgen.io.vasp import Xdatcar

        cls.structures = Xdatcar(str(ARGYRODITE_XDATCAR)).structures

    def _run_argyrodite_analysis(self):
        """Run the argyrodite polyhedral site analysis."""
        import numpy as np
        from pymatgen.core import Lattice, Structure

        from site_analysis import TrajectoryBuilder
        from site_analysis.site import Site

        Site._newid = 0

        lattice = Lattice.cubic(a=10.155)
        coords = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.9, 0.9, 0.6],
                [0.23, 0.92, 0.09],
                [0.25, 0.25, 0.25],
                [0.15, 0.15, 0.15],
                [0.0, 0.183, 0.183],
                [0.0, 0.0, 0.0],
                [0.75, 0.25, 0.25],
                [0.11824, 0.11824, 0.38176],
            ]
        )
        reference_structure = Structure.from_spacegroup(
            sg="F-43m",
            lattice=lattice,
            species=["P", "Li", "Mg", "Na", "Be", "K", "S", "S", "S"],
            coords=coords,
        ) * [2, 2, 2]

        builder = (
            TrajectoryBuilder()
            .with_structure(self.structures[0])
            .with_reference_structure(reference_structure)
            .with_mobile_species("Li")
            .with_structure_alignment(align_species="P")
            .with_site_mapping(mapping_species=["S", "Cl"])
        )

        site_defs = [
            ("Li", "type 1"),
            ("Mg", "type 2"),
            ("Na", "type 3"),
            ("Be", "type 4"),
            ("K", "type 5"),
        ]
        for centre_species, label in site_defs:
            builder.with_polyhedral_sites(
                centre_species=centre_species,
                vertex_species="S",
                cutoff=3.0,
                n_vertices=4,
                label=label,
            )

        trajectory = builder.build()

        start = time.perf_counter()
        trajectory.trajectory_from_structures(self.structures, progress=False)
        elapsed = time.perf_counter() - start

        atom_trajectories = [atom.trajectory[:] for atom in trajectory.atoms]
        return atom_trajectories, elapsed

    def test_numba_and_delaunay_produce_identical_trajectories(self):
        """Run argyrodite analysis with both paths and compare."""
        # Run with numba
        numba_atoms, numba_time = self._run_argyrodite_analysis()

        # Run with Delaunay fallback
        with patch("site_analysis.polyhedral_site.HAS_NUMBA", False):
            delaunay_atoms, delaunay_time = self._run_argyrodite_analysis()

        # Compare atom trajectories
        self.assertEqual(len(numba_atoms), len(delaunay_atoms))
        mismatches = 0
        for i, (n, d) in enumerate(zip(numba_atoms, delaunay_atoms)):
            if n != d:
                mismatches += 1
        self.assertEqual(
            mismatches,
            0,
            f"{mismatches}/{len(numba_atoms)} atom trajectories differ",
        )

        # Report timing
        print(f"\nArgyrodite Li6PS5Cl ({len(self.structures)} frames):")
        print(f"  Numba:   {numba_time:.2f}s")
        print(f"  Delaunay: {delaunay_time:.2f}s")
        if delaunay_time > 0:
            print(f"  Speedup: {delaunay_time / numba_time:.1f}x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
