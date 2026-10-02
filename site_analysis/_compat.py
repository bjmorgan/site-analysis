"""Optional dependency detection."""

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
