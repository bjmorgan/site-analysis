"""Benchmark: minimum-image distance functions and periodic neighbour search.

Compares:
1. pymatgen Lattice.get_distance_and_image (single pair, Cython)
2. mic_distance numpy fallback (single pair)
3. mic_distance numba (single pair, if available)
4. pymatgen Lattice.get_all_distances (batch, Cython)
5. all_mic_distances numpy (batch)
6. PeriodicNeighbourIndex nearest and within-cutoff queries, with the
   nearest query also compared against computing every pairwise distance

Tests across:
- Cubic and triclinic lattices
- Varying batch sizes (10x10, 50x50, 200x200)
- Neighbour search for 10^2 to 10^5 points at a fixed density
"""

import timeit
import numpy as np
from pymatgen.core import Lattice

from site_analysis.distances import (
    mic_distance,
    all_mic_distances,
    frac_to_cart,
    paired_mic_distances,
)
from site_analysis.neighbour_search import PeriodicNeighbourIndex
from site_analysis._compat import HAS_NUMBA

if HAS_NUMBA:
    from site_analysis.distances import _mic_distance_numba, _all_mic_distances_numba


def benchmark_single_pair(lattice, n_pairs=1000, n_repeats=5):
    """Benchmark single-pair distance calculations."""
    rng = np.random.default_rng(42)
    pairs = [(rng.random(3), rng.random(3)) for _ in range(n_pairs)]
    matrix = lattice.matrix

    # Pymatgen
    def pymatgen_single():
        for f1, f2 in pairs:
            lattice.get_distance_and_image(f1, f2)

    # Force the numpy fallback path by patching HAS_NUMBA
    import site_analysis.distances as _dist_mod
    from unittest.mock import patch as _patch
    _numpy_patch = _patch.object(_dist_mod, 'HAS_NUMBA', False)

    def numpy_single():
        with _numpy_patch:
            for f1, f2 in pairs:
                mic_distance(f1, f2, matrix)

    # mic_distance (auto-dispatches to numba if available)
    def mic_single():
        for f1, f2 in pairs:
            mic_distance(f1, f2, matrix)

    if HAS_NUMBA:
        # Warm up JIT before any timing that may dispatch to numba
        _mic_distance_numba(pairs[0][0], pairs[0][1], matrix)

    results = {}
    results['pymatgen'] = min(timeit.repeat(pymatgen_single, number=n_repeats)) / n_repeats
    results['numpy'] = min(timeit.repeat(numpy_single, number=n_repeats)) / n_repeats
    results['mic_distance'] = min(timeit.repeat(mic_single, number=n_repeats)) / n_repeats

    if HAS_NUMBA:

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

    if HAS_NUMBA:
        # Warm up JIT before timing
        warm = rng.random((2, 3))
        _all_mic_distances_numba(warm, warm, matrix)

    for n, m in sizes:
        frac1 = rng.random((n, 3))
        frac2 = rng.random((m, 3))

        def pymatgen_batch(f1=frac1, f2=frac2):
            lattice.get_all_distances(f1, f2)

        def numpy_batch(f1=frac1, f2=frac2, mat=matrix):
            all_mic_distances(f1, f2, mat)

        key = f"{n}x{m}"
        times = {
            'pymatgen': min(timeit.repeat(pymatgen_batch, number=n_repeats)) / n_repeats,
            'all_mic_distances': min(timeit.repeat(numpy_batch, number=n_repeats)) / n_repeats,
        }

        if HAS_NUMBA:
            def numba_batch(f1=frac1, f2=frac2, mat=matrix):
                _all_mic_distances_numba(f1, f2, mat)

            times['numba'] = min(timeit.repeat(numba_batch, number=n_repeats)) / n_repeats

        results[key] = times

    return results


def brute_force_nearest(query, points, matrix, chunk=256):
    """Find each query point's nearest distance from every pairwise distance.

    Distances are computed in blocks with ``paired_mic_distances``, on
    coordinates repeated for each pair with ``np.repeat`` and ``np.tile``,
    so this takes up to about twice as long as a kernel that computes a
    full distance matrix directly.

    Args:
        query: Fractional coordinates of the query points, shape (M, 3).
        points: Fractional coordinates of the points, shape (N, 3).
        matrix: (3, 3) lattice matrix.
        chunk: Number of query points per block of distances.

    Returns:
        (M,) array of distances from each query point to its nearest point.
    """
    nearest = np.empty(len(query))
    for start in range(0, len(query), chunk):
        block = query[start:start + chunk]
        distances = paired_mic_distances(np.repeat(block, len(points), axis=0),
                                         np.tile(points, (len(block), 1)), matrix)
        nearest[start:start + chunk] = distances.reshape(len(block), len(points)).min(axis=1)
    return nearest


def benchmark_neighbour_search(lattice, sizes=(100, 1000, 3000, 10000, 100000),
                               max_all_pairs=3000, density=0.08, cutoff=3.0):
    """Benchmark PeriodicNeighbourIndex against computing every pairwise distance.

    The cell is scaled so that the points have a fixed number density, so
    the number of neighbours within the cutoff stays the same as the number
    of points grows. Each time is the best of three runs.

    Args:
        lattice: pymatgen Lattice giving the cell shape.
        sizes: Numbers of points, and of query points, to time.
        max_all_pairs: Largest size for which every pairwise distance is
            also computed, for comparison.
        density: Number of points per cubic Angstrom.
        cutoff: Cutoff for ``query_within``, in Angstrom.

    Returns:
        Dictionary mapping each size to a dictionary of times in seconds,
        keyed by method.
    """
    rng = np.random.default_rng(42)
    # Warm up the kernels (numba compilation, thread pool) before timing.
    warm = rng.random((10, 3))
    PeriodicNeighbourIndex(warm, lattice.matrix).query_nearest(warm)
    brute_force_nearest(warm, warm, lattice.matrix)
    results = {}
    for n in sizes:
        matrix = lattice.matrix * (n / density / lattice.volume) ** (1 / 3)
        points = rng.random((n, 3))
        query = rng.random((n, 3))
        index = PeriodicNeighbourIndex(points, matrix)
        timed = {
            'build index': lambda: PeriodicNeighbourIndex(points, matrix),
            'query_nearest': lambda: index.query_nearest(query),
            'query_within': lambda: index.query_within(query, cutoff),
        }
        if n <= max_all_pairs:
            # The index must agree with computing every pairwise distance.
            assert np.array_equal(index.query_nearest(query)[1],
                                  brute_force_nearest(query, points, matrix))
            timed['all pairs, nearest'] = lambda: brute_force_nearest(query, points, matrix)
        results[n] = {method: min(timeit.repeat(function, number=1, repeat=3))
                      for method, function in timed.items()}
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

        density, cutoff, max_all_pairs = 0.08, 3.0, 3000
        print(f"\n--- Neighbour search ({density} points per A^3, cutoff {cutoff} A;"
              f" all pairs only up to N = {max_all_pairs}) ---")
        search = benchmark_neighbour_search(lattice, max_all_pairs=max_all_pairs,
                                            density=density, cutoff=cutoff)
        for n, times in search.items():
            print(f"  N = {n}:")
            for method, seconds in times.items():
                print(f"    {method:20s}: {seconds*1000:10.2f} ms")
