"""Site analysis package for molecular dynamics trajectories.

This package provides functionality for analysing molecular dynamics trajectories,
with a focus on tracking ion migration through crystallographic sites.
"""

try:
    # Get version from installed package metadata
    from importlib.metadata import version
    __version__ = version("site_analysis")
except ImportError:
    # For Python < 3.8
    try:
        from importlib_metadata import version
        __version__ = version("site_analysis")
    except ImportError:
        __version__ = "unknown"

# Import and expose the builder tools
from site_analysis.builders import (
    TrajectoryBuilder,
    create_trajectory_with_spherical_sites,
    create_trajectory_with_voronoi_sites,
    create_trajectory_with_polyhedral_sites,
    create_trajectory_with_dynamic_voronoi_sites
)