"""Site analysis package for molecular dynamics trajectories.

This package provides functionality for analysing molecular dynamics trajectories,
with a focus on tracking ion migration pathways through crystallographic sites.
"""

from site_analysis.version import __version__

# Import and expose the builder tools
from site_analysis.builders import (
    TrajectoryBuilder,
    create_trajectory_with_spherical_sites,
    create_trajectory_with_voronoi_sites,
    create_trajectory_with_polyhedral_sites,
    create_trajectory_with_dynamic_voronoi_sites
)
