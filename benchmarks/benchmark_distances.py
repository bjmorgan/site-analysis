"""Benchmark: minimum-image distance functions vs pymatgen.

Compares:
1. pymatgen Lattice.get_distance_and_image (single pair, Cython)
2. mic_distance numpy fallback (single pair)
3. mic_distance numba (single pair, if available)
4. pymatgen Lattice.get_all_distances (batch, Cython)
5. all_mic_distances numpy (batch)

Tests across:
- Cubic and triclinic lattices
- Varying batch sizes (10x10, 50x50, 200x200)
"""

import timeit
import numpy as np
from pymatgen.core import Lattice

from site_analysis.distances import (
    mic_distance,
    all_mic_distances,
    frac_to_cart,
)
from site_analysis._compat import HAS_NUMBA

if HAS_NUMBA:
    from site_analysis.distances import _mic_distance_numba


def benchmark_single_pair(lattice, n_pairs=1000, n_repeats=5):
    """Benchmark single-pair distance calculations."""
    rng = np.random.default_rng(42)
    pairs = [(rng.random(3), rng.random(3)) for _ in range(n_pairs)]
    matrix = lattice.matrix

    # Pymatgen
    def pymatgen_single():
        for f1, f2 in pairs:
            lattice.get_distance_and_image(f1, f2)

    # Numpy fallback (bypass numba dispatch via mock)
    def numpy_single():
        from site_analysis.distances import _SHIFTS_27
        for f1, f2 in pairs:
            d_frac = f1 - f2
            d_frac_all = d_frac + _SHIFTS_27
            d_cart_all = d_frac_all @ matrix
            float(np.min(np.linalg.norm(d_cart_all, axis=1)))

    # mic_distance (auto-dispatches to numba if available)
    def mic_single():
        for f1, f2 in pairs:
            mic_distance(f1, f2, matrix)

    results = {}
    results['pymatgen'] = min(timeit.repeat(pymatgen_single, number=n_repeats)) / n_repeats
    results['numpy'] = min(timeit.repeat(numpy_single, number=n_repeats)) / n_repeats
    results['mic_distance'] = min(timeit.repeat(mic_single, number=n_repeats)) / n_repeats

    if HAS_NUMBA:
        # Warm up JIT
        _mic_distance_numba(pairs[0][0], pairs[0][1], matrix)

        def numba_single():
            for f1, f2 in pairs:
                _mic_distance_numba(f1, f2, matrix)

        results['numba'] = min(timeit.repeat(numba_single, number=n_repeats)) / n_repeats

    return results


def benchmark_batch(lattice, sizes=None, n_repeats=5):
    """Benchmark batch distance matrix calculations."""
    if sizes is None:
        sizes = [(10, 10), (50, 50), (200, 200)]

    rng = np.random.default_rng(42)
    matrix = lattice.matrix
    results = {}

    for n, m in sizes:
        frac1 = rng.random((n, 3))
        frac2 = rng.random((m, 3))

        def pymatgen_batch(f1=frac1, f2=frac2):
            lattice.get_all_distances(f1, f2)

        def numpy_batch(f1=frac1, f2=frac2, mat=matrix):
            all_mic_distances(f1, f2, mat)

        key = f"{n}x{m}"
        results[key] = {
            'pymatgen': min(timeit.repeat(pymatgen_batch, number=n_repeats)) / n_repeats,
            'numpy': min(timeit.repeat(numpy_batch, number=n_repeats)) / n_repeats,
        }

    return results


def benchmark_frac_to_cart(lattice, n_points=1000, n_repeats=5):
    """Benchmark fractional to Cartesian conversion."""
    rng = np.random.default_rng(42)
    frac = rng.random((n_points, 3))
    matrix = lattice.matrix

    def pymatgen_cart():
        lattice.get_cartesian_coords(frac)

    def numpy_cart():
        frac_to_cart(frac, matrix)

    return {
        'pymatgen': min(timeit.repeat(pymatgen_cart, number=n_repeats)) / n_repeats,
        'numpy': min(timeit.repeat(numpy_cart, number=n_repeats)) / n_repeats,
    }


if __name__ == '__main__':
    lattices = {
        'cubic': Lattice.cubic(10.0),
        'triclinic': Lattice.from_parameters(5.0, 6.0, 7.0, 80, 70, 60),
    }

    for name, lattice in lattices.items():
        print(f"\n{'='*60}")
        print(f"Lattice: {name}")
        print(f"{'='*60}")

        print(f"\n--- Single pair ({1000} pairs) ---")
        single = benchmark_single_pair(lattice)
        for method, time in single.items():
            per_call = time / 1000 * 1e6
            print(f"  {method:20s}: {per_call:8.2f} us/pair")

        print(f"\n--- Batch distance matrix ---")
        batch = benchmark_batch(lattice)
        for size, times in batch.items():
            print(f"  {size}:")
            for method, time in times.items():
                print(f"    {method:20s}: {time*1000:8.3f} ms")

        print(f"\n--- frac_to_cart ({1000} points) ---")
        cart = benchmark_frac_to_cart(lattice)
        for method, time in cart.items():
            print(f"  {method:20s}: {time*1000:8.3f} ms")
