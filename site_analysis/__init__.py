"""Site analysis package for molecular dynamics trajectories.

This package provides functionality for analysing molecular dynamics trajectories,
with a focus on tracking ion migration pathways through crystallographic sites.
"""

# Get version from installed package metadata
try:
    from importlib.metadata import version as importlib_version  # type: ignore
    __version__ = importlib_version("site_analysis")
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
