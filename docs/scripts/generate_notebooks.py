"""Generate downloadable Jupyter notebooks for the tutorials.

This script creates .ipynb files that mirror the code in the
markdown tutorials, so users can download and run them directly.
"""

import json
import uuid
from pathlib import Path

TUTORIALS_DIR = Path(__file__).resolve().parent.parent / "source" / "tutorials"
NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent.parent / "tutorials"


def new_notebook():
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": [],
    }


def md_cell(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": source.splitlines(True)}


def code_cell(source):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.splitlines(True),
        "outputs": [],
        "execution_count": None,
    }


def generate_comparing_site_definitions():
    nb = new_notebook()
    cells = nb["cells"]

    cells.append(md_cell(
        "# Comparing Site Definitions: Ion Migration in an FCC Lattice\n"
        "\n"
        "This tutorial introduces the core concepts of `site-analysis` by working through a simple "
        "example: tracking a lithium ion as it migrates between interstitial sites in an FCC lattice. "
        "You will define sites using three different approaches \u2014 spherical, Voronoi, and polyhedral "
        "\u2014 and compare how each one assigns the mobile ion during a migration event, highlighting "
        "the tradeoffs between methods.\n"
        "\n"
        "## Prerequisites\n"
        "\n"
        "This tutorial requires the following packages:\n"
        "\n"
        "- `site-analysis`\n"
        "- `matplotlib`\n"
        "\n"
        "All data in this tutorial is generated synthetically, so no external data files are needed.\n"
        "\n"
        "## Overview\n"
        "\n"
        "We will create a 3x3x3 FCC oxygen lattice and a synthetic trajectory of a single lithium ion "
        "migrating from an octahedral site, through a tetrahedral site, to another octahedral site. "
        "We then analyse this trajectory using three different site definitions and compare the results."
    ))

    cells.append(code_cell(
        "%config InlineBackend.figure_format = 'retina'"
    ))

    cells.append(md_cell(
        "## Creating the FCC lattice\n"
        "\n"
        "First, create the host framework \u2014 a 3x3x3 supercell of an FCC oxygen lattice:"
    ))

    cells.append(code_cell(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from scipy.interpolate import make_interp_spline\n"
        "from pymatgen.core import Lattice, Structure, DummySpecies\n"
        "from site_analysis import TrajectoryBuilder\n"
        "\n"
        "a = 4.0  # lattice parameter\n"
        "lattice = Lattice.cubic(a)\n"
        "supercell_expansion = [3, 3, 3]\n"
        "fcc_structure = Structure.from_spacegroup(\n"
        "    sg='Fm-3m',\n"
        "    lattice=lattice,\n"
        "    species=['O'],\n"
        "    coords=[[0, 0, 0]]\n"
        ") * supercell_expansion"
    ))

    cells.append(md_cell(
        "The FCC structure has two types of interstitial sites:\n"
        "\n"
        "- **Octahedral sites**: at edge midpoints (0.5, 0.0, 0.0) and body centres (0.5, 0.5, 0.5)\n"
        "- **Tetrahedral sites**: at positions like (0.25, 0.25, 0.25)\n"
        "\n"
        "We generate the fractional coordinates for both site types using `pymatgen`'s space group "
        "machinery with a `DummySpecies`:"
    ))

    cells.append(code_cell(
        "fcc_octahedral_coords = (Structure.from_spacegroup(\n"
        "    sg='Fm-3m',\n"
        "    lattice=lattice,\n"
        "    species=[DummySpecies('Q')],\n"
        "    coords=[[0.5, 0, 0]]\n"
        ") * supercell_expansion).frac_coords\n"
        "\n"
        "fcc_tetrahedral_coords = (Structure.from_spacegroup(\n"
        "    sg='Fm-3m',\n"
        "    lattice=lattice,\n"
        "    species=[DummySpecies('Q')],\n"
        "    coords=[[0.25, 0.25, 0.25]]\n"
        ") * supercell_expansion).frac_coords"
    ))

    cells.append(md_cell(
        "## Generating a test trajectory\n"
        "\n"
        "Next, create a synthetic trajectory of a lithium ion migrating from one "
        "octahedral site, through a tetrahedral site, to another octahedral site. This mimics "
        "a common diffusion mechanism in close-packed structures.\n"
        "\n"
        "We define three waypoints \u2014 one at each site \u2014 and use a quadratic spline to "
        "interpolate a smooth curved path between them, sampled at 21 evenly spaced frames:"
    ))

    cells.append(code_cell(
        "waypoints = np.array([\n"
        "    [1/3, 2/3, 1/2],      # octahedral site\n"
        "    [5/12, 7/12, 7/12],   # tetrahedral site\n"
        "    [1/2, 2/3, 2/3],      # octahedral site\n"
        "])\n"
        "\n"
        "t_waypoints = np.array([0.0, 0.5, 1.0])\n"
        "t = np.linspace(0, 1, 21)\n"
        "\n"
        "spline = make_interp_spline(t_waypoints, waypoints, k=2)\n"
        "mobile_ion_coords = spline(t)\n"
        "\n"
        "md_trajectory = []\n"
        "for c in mobile_ion_coords:\n"
        "    structure = fcc_structure.copy().append(species='Li', coords=c)\n"
        "    md_trajectory.append(structure)\n"
        "\n"
        "print(f'{len(md_trajectory)} frames')"
    ))

    cells.append(md_cell(
        "## Analysing with spherical sites\n"
        "\n"
        "We use the `TrajectoryBuilder` to set up each analysis. The builder configures the structure, "
        "mobile species, and site definitions, then builds a `Trajectory` object. Calling "
        "`trajectory_from_structures` processes each frame, assigning the mobile ion to a site at each "
        "timestep. For more details on the builder pattern, see the "
        "[builders guide](https://site-analysis.readthedocs.io/en/latest/guides/builders.html).\n"
        "\n"
        "Spherical sites are the simplest site type: each site is a sphere defined by a centre "
        "position and radius. We choose radii equal to the largest spheres that fit inside each "
        "polyhedron without overlapping:\n"
        "\n"
        "- Octahedral radius: r = l / sqrt(6) = 1.155 A\n"
        "- Tetrahedral radius: r = l * sqrt(6) / 12 = 0.577 A\n"
        "\n"
        "For more details, see the [spherical sites guide]"
        "(https://site-analysis.readthedocs.io/en/latest/guides/spherical_sites.html)."
    ))

    cells.append(code_cell(
        "builder = TrajectoryBuilder()\n"
        "builder.with_structure(md_trajectory[0])\n"
        'builder.with_mobile_species("Li")\n'
        "\n"
        "builder.with_spherical_sites(\n"
        "    centres=fcc_octahedral_coords,\n"
        "    radii=1.155,\n"
        '    labels="octahedral"\n'
        ")\n"
        "builder.with_spherical_sites(\n"
        "    centres=fcc_tetrahedral_coords,\n"
        "    radii=0.577,\n"
        '    labels="tetrahedral"\n'
        ")\n"
        "\n"
        "trajectory = builder.build()\n"
        "trajectory.trajectory_from_structures(md_trajectory, progress=True)\n"
        "\n"
        "spherical_trajectory = trajectory.atoms[0].trajectory\n"
        "print(spherical_trajectory)"
    ))

    cells.append(md_cell(
        "Notice the `None` values at frames 6 and 14 \u2014 the ion is between sites, "
        "in regions not covered by any sphere.\n"
        "\n"
        "## Analysing with Voronoi sites\n"
        "\n"
        "Voronoi sites partition the entire space based on proximity to site centres \u2014 "
        "every point is assigned to its nearest site centre, so there are no gaps or overlaps. "
        "For more details, see the [Voronoi sites guide]"
        "(https://site-analysis.readthedocs.io/en/latest/guides/voronoi_sites.html)."
    ))

    cells.append(code_cell(
        "builder = TrajectoryBuilder()\n"
        "builder.with_structure(md_trajectory[0])\n"
        'builder.with_mobile_species("Li")\n'
        "\n"
        "builder.with_voronoi_sites(\n"
        "    centres=fcc_octahedral_coords,\n"
        '    labels="octahedral"\n'
        ")\n"
        "builder.with_voronoi_sites(\n"
        "    centres=fcc_tetrahedral_coords,\n"
        '    labels="tetrahedral"\n'
        ")\n"
        "\n"
        "trajectory = builder.build()\n"
        "trajectory.trajectory_from_structures(md_trajectory, progress=True)\n"
        "\n"
        "voronoi_trajectory = trajectory.atoms[0].trajectory\n"
        "print(voronoi_trajectory)"
    ))

    cells.append(md_cell(
        "With Voronoi sites the ion is always assigned to a site \u2014 no `None` values.\n"
        "\n"
        "## Analysing with polyhedral sites\n"
        "\n"
        "Polyhedral sites are defined by coordination polyhedra formed by the host lattice atoms. "
        "This requires a reference structure that marks where each site type is located, using dummy atoms. "
        "For more details, see the [polyhedral sites guide]"
        "(https://site-analysis.readthedocs.io/en/latest/guides/polyhedral_sites.html)."
    ))

    cells.append(code_cell(
        "reference_structure = fcc_structure.copy()\n"
        "\n"
        "for frac_coord in fcc_octahedral_coords:\n"
        '    reference_structure.append("Mg", frac_coord)\n'
        "\n"
        "for site in fcc_tetrahedral_coords:\n"
        '    reference_structure.append("Na", site)\n'
        "\n"
        "builder = TrajectoryBuilder()\n"
        "builder.with_structure(md_trajectory[0])\n"
        "builder.with_reference_structure(reference_structure)\n"
        'builder.with_mobile_species("Li")\n'
        "\n"
        "builder.with_polyhedral_sites(\n"
        '    centre_species="Mg",\n'
        '    vertex_species="O",\n'
        "    cutoff=3.0,\n"
        "    n_vertices=6,\n"
        '    label="octahedral"\n'
        ")\n"
        "builder.with_polyhedral_sites(\n"
        '    centre_species="Na",\n'
        '    vertex_species="O",\n'
        "    cutoff=2.5,\n"
        "    n_vertices=4,\n"
        '    label="tetrahedral"\n'
        ")\n"
        "builder.with_site_mapping(mapping_species='O')\n"
        "\n"
        "trajectory = builder.build()\n"
        "trajectory.trajectory_from_structures(md_trajectory, progress=True)\n"
        "\n"
        "polyhedral_trajectory = trajectory.atoms[0].trajectory\n"
        "print(polyhedral_trajectory)"
    ))

    cells.append(md_cell(
        "Like Voronoi sites, polyhedral sites fill space completely through their face-sharing "
        "structure, so the ion is always assigned.\n"
        "\n"
        "## Comparing the results\n"
        "\n"
        "We can visualise the site assignment for each method to see how they differ:"
    ))

    cells.append(code_cell(
        "site_index = {17: 1, 283: 2, 70: 3, None: 0}\n"
        "\n"
        "trajectories = {\n"
        "    'Spherical Sites': spherical_trajectory,\n"
        "    'Voronoi Sites': voronoi_trajectory,\n"
        "    'Polyhedral Sites': polyhedral_trajectory,\n"
        "}\n"
        "\n"
        "fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)\n"
        "\n"
        "for ax, (title, traj) in zip(axes, trajectories.items()):\n"
        "    ax.plot([site_index[i] for i in traj], 'o--', markersize=3, color='tab:red')\n"
        "    ax.set_yticks([0, 1, 2, 3])\n"
        "    ax.set_yticklabels(['unassigned', 'oct', 'tet', 'oct'])\n"
        "    ax.set_ylim(-0.2, 3.2)\n"
        "    ax.set_xlim(-0.8, 20.8)\n"
        "    ax.set_title(title, loc='center')\n"
        "\n"
        "axes[-1].set_xticks(list(range(0, 21, 5)))\n"
        "axes[-1].set_xlabel('Frame')\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(md_cell(
        "## Analysis\n"
        "\n"
        "All three methods capture the essential migration path: octahedral -> tetrahedral -> octahedral. "
        "The differences lie in how they handle the transitions. For a more detailed discussion of site "
        "types and their tradeoffs, see the [sites concept page]"
        "(https://site-analysis.readthedocs.io/en/latest/concepts/sites.html).\n"
        "\n"
        "**Spherical sites** leave gaps between sites. The inradii we chose do not cover the full space, "
        "so the ion appears \"unassigned\" during transitions (frames 6 and 14). This makes transitions "
        "appear abrupt but can lose information about intermediate states.\n"
        "\n"
        "**Voronoi sites** partition all space by proximity to site centres. Transitions occur when the "
        "ion crosses a boundary equidistant from neighbouring centres. This ensures continuous tracking "
        "but defines boundaries purely geometrically, without considering the local coordination environment.\n"
        "\n"
        "**Polyhedral sites** define boundaries based on the shared faces between coordination polyhedra. "
        "Transitions occur when the ion passes through these shared faces, giving a physically meaningful "
        "boundary based on the actual crystal structure. The transition timing differs slightly from the "
        "Voronoi approach because the polyhedral face is not necessarily at the midpoint between site centres.\n"
        "\n"
        "## Summary\n"
        "\n"
        "- **Spherical sites** require choosing both centre positions and radii. If the radii are too "
        "small, there will be gaps between sites and unassigned periods during transitions. If the radii "
        "are large enough to fill space, sites will overlap and assignment depends on the precedence "
        "rules in the code.\n"
        "- **Voronoi sites** require only centre positions and fill space completely based on proximity, "
        "ensuring continuous tracking.\n"
        "- **Polyhedral sites** also fill space completely but use physically meaningful coordination "
        "polyhedra to define boundaries.\n"
        "\n"
        "The choice of site definition affects both the assignment of ions at each timestep and the "
        "apparent timing of transitions. Voronoi sites are the simplest to set up (only centre positions "
        "needed), while polyhedral sites often provide the most physically meaningful results for "
        "quantitative analysis of migration mechanisms."
    ))

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = NOTEBOOKS_DIR / "comparing_site_definitions.ipynb"
    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {output_path} with {len(cells)} cells")


def generate_argyrodite_site_analysis():
    nb = new_notebook()
    cells = nb["cells"]

    cells.append(md_cell(
        "# Analysing Complex Crystal Structures: Argyrodite Solid Electrolytes\n"
        "\n"
        "This tutorial demonstrates how to apply `site-analysis` to a realistic materials science "
        "problem: analysing lithium-ion site occupations in Li6PS5Cl argyrodite solid electrolytes. "
        "You will learn how to define multiple crystallographically distinct site types using a "
        "reference structure with dummy atoms, handle structure alignment and site mapping between "
        "structures with different chemical compositions, and examine how site occupations change "
        "with varying degrees of S/Cl anion disorder.\n"
        "\n"
        "## Prerequisites\n"
        "\n"
        "This tutorial requires the following packages:\n"
        "\n"
        "- `site-analysis`\n"
        "\n"
        "This tutorial uses trajectory data from molecular dynamics simulations.\n"
        "The data files are included in the `site-analysis` GitHub repository at\n"
        "`tutorials/data/`:\n"
        "\n"
        "- `Li6PS5Cl_0p_XDATCAR.gz` \u2014 0% anion site exchange (fully ordered)\n"
        "- `Li6PS5Cl_50p_XDATCAR.gz` \u2014 50% anion site exchange\n"
        "- `Li6PS5Cl_100p_XDATCAR.gz` \u2014 100% anion site exchange\n"
        "\n"
        "These files are available in the [tutorials/data]"
        "(https://github.com/bjmorgan/site-analysis/tree/main/tutorials/data) "
        "directory of the GitHub repository. "
        "The code examples below assume you are running from the repository root directory.\n"
        "\n"
        "## Background: the argyrodite structure\n"
        "\n"
        "The argyrodite Li6PS5Cl is a tetrahedrally close-packed structure where the anions "
        "(S2-, Cl-) form the vertices of tetrahedra. The centres of these tetrahedra are interstitial "
        "sites that can be classified into six crystallographically distinct types, numbered 0 to 5:\n"
        "\n"
        "- **Type 0** sites are occupied by phosphorus (forming PS4 tetrahedra)\n"
        "- **Types 1\u20135** are, in principle, available for lithium occupation\n"
        "\n"
        "In Li6PS5Cl, the S and Cl anions can exchange between the 4a and 4c Wyckoff positions. "
        "This anion disorder modifies the local coordination environments around lithium sites, "
        "affecting which sites lithium ions prefer to occupy. Increased disorder activates previously "
        "unfavourable site types (particularly type 4), leading to enhanced ionic conductivity.\n"
        "\n"
        "For more details, see [Morgan (2021)](https://doi.org/10.1021/acs.chemmater.0c03738).\n"
        "\n"
        "In this tutorial we analyse three MD simulations with different degrees of S/Cl site exchange "
        "(0%, 50%, and 100%) and calculate the time-averaged distribution of lithium ions over the "
        "five interstitial site types."
    ))

    cells.append(md_cell(
        "## Creating a reference structure\n"
        "\n"
        "To define multiple site types with polyhedral sites, we need a reference structure where "
        "each site type is marked by a different dummy atom species. This lets us call "
        "`with_polyhedral_sites` separately for each type. For more details on the reference "
        "structure workflow, see the [reference workflow guide]"
        "(https://site-analysis.readthedocs.io/en/latest/guides/reference_workflow.html)."
    ))

    cells.append(code_cell(
        "%config InlineBackend.figure_format = 'retina'"
    ))

    cells.append(code_cell(
        "from pymatgen.io.vasp import Xdatcar\n"
        "from pymatgen.core import Structure, Lattice\n"
        "import numpy as np\n"
        "from collections import Counter\n"
        "from site_analysis import TrajectoryBuilder\n"
        "\n"
        "lattice = Lattice.cubic(a=10.155)\n"
        "\n"
        "coords = np.array(\n"
        "    [[0.5,     0.5,     0.5],     # P (type 0) - PS4 tetrahedra\n"
        "     [0.9,     0.9,     0.6],     # type 1 (Li in reference)\n"
        "     [0.77,    0.585,   0.585],   # type 2 (Mg in reference, 48h)\n"
        "     [0.25,    0.25,    0.25],    # type 3 (Na in reference)\n"
        "     [0.15,    0.15,    0.15],    # type 4 (Be in reference)\n"
        "     [0.0,     0.183,   0.183],   # type 5 (K in reference)\n"
        "     [0.0,     0.0,     0.0],     # S - anion position (4a site)\n"
        "     [0.75,    0.25,    0.25],    # S - anion position (4c site)\n"
        "     [0.11824, 0.11824, 0.38176]] # S - anion position (16e site)\n"
        ")\n"
        "\n"
        "reference_structure = Structure.from_spacegroup(\n"
        '    sg="F-43m",\n'
        "    lattice=lattice,\n"
        "    species=['P', 'Li', 'Mg', 'Na', 'Be', 'K', 'S', 'S', 'S'],\n"
        "    coords=coords) * [2, 2, 2]\n"
        "\n"
        'print(f"Reference structure contains {len(reference_structure)} atoms")\n'
        'print(f"Composition: {reference_structure.composition.formula}")'
    ))

    cells.append(md_cell(
        "Each dummy species (Li, Mg, Na, Be, K) marks a different tetrahedral site type. "
        "The reference uses all-S anions \u2014 we will handle the S/Cl disorder through species mapping later.\n"
        "\n"
        "## Building the trajectory analyser\n"
        "\n"
        "We now write a function that configures a `TrajectoryBuilder` for the argyrodite analysis "
        "(see the [builders guide](https://site-analysis.readthedocs.io/en/latest/guides/builders.html) "
        "for a full overview). This demonstrates several advanced features:"
    ))

    cells.append(code_cell(
        "def build_trajectory(structure):\n"
        "    builder = TrajectoryBuilder()\n"
        "\n"
        "    # Set the reference and target structures\n"
        "    builder.with_reference_structure(reference_structure)\n"
        "    builder.with_structure(structure)\n"
        "\n"
        "    # Specify lithium as the mobile species\n"
        "    builder.with_mobile_species('Li')\n"
        "\n"
        "    # Define five tetrahedral site types, one per dummy species\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Li', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 1')\n"
        "\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Mg', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 2')\n"
        "\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Na', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 3')\n"
        "\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Be', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 4')\n"
        "\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='K', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 5')\n"
        "\n"
        "    # Align the reference and target structures using phosphorus positions\n"
        "    builder.with_structure_alignment(align_species='P')\n"
        "\n"
        "    # Map between S and Cl -- the reference has all-S anions, but the\n"
        "    # real structures have a mix of S and Cl at these positions\n"
        "    builder.with_site_mapping(mapping_species=['S', 'Cl'])\n"
        "\n"
        "    trajectory = builder.build()\n"
        "    return trajectory"
    ))

    cells.append(md_cell(
        "Key points:\n"
        "\n"
        "- We call `with_polyhedral_sites` five times, once per site type, using a different "
        "`centre_species` each time. For more details on polyhedral sites, see the "
        "[polyhedral sites guide]"
        "(https://site-analysis.readthedocs.io/en/latest/guides/polyhedral_sites.html).\n"
        "- `with_structure_alignment(align_species='P')` aligns the reference to the target "
        "structure using the phosphorus atom positions.\n"
        "- `with_site_mapping(mapping_species=['S', 'Cl'])` tells the builder that S and Cl "
        "are interchangeable when mapping between the all-S reference and the real (disordered) structure.\n"
        "\n"
        "## Analysing site occupations\n"
        "\n"
        "We define a helper function to calculate the percentage of time lithium ions spend in each site type:"
    ))

    cells.append(code_cell(
        "def print_site_occupations(trajectory, title=None):\n"
        "    site_types = ['type 5', 'type 4', 'type 3', 'type 2', 'type 1']\n"
        "\n"
        "    site_labels = []\n"
        "    for atom in trajectory.atoms:\n"
        "        for site_idx in atom.trajectory:\n"
        "            if site_idx is not None:\n"
        "                site_labels.append(trajectory.sites[site_idx].label)\n"
        "\n"
        "    c = Counter(site_labels)\n"
        "    total_sites = sum(c.values())\n"
        "\n"
        "    if title:\n"
        '        print(f"\\nSite occupation analysis - {title}:")\n'
        '        print("-" * 40)\n'
        "\n"
        "    for t in site_types:\n"
        "        percentage = (c.get(t, 0) / total_sites * 100) if total_sites > 0 else 0\n"
        "        print(f'{t}: {percentage:.2f}%')"
    ))

    cells.append(md_cell(
        "## Ordered structure (0% anion site exchange)\n"
        "\n"
        "We start with the fully ordered Li6PS5Cl, where S and Cl occupy their equilibrium "
        "Wyckoff positions with no site exchange:"
    ))

    cells.append(code_cell(
        "md_structures = Xdatcar('tutorials/data/Li6PS5Cl_0p_XDATCAR.gz').structures\n"
        'print(f"Loaded trajectory with {len(md_structures)} frames")\n'
        "\n"
        "trajectory_0p = build_trajectory(md_structures[0])\n"
        "trajectory_0p.trajectory_from_structures(md_structures, progress=True)\n"
        "\n"
        'print_site_occupations(trajectory_0p, "0% anion disorder")'
    ))

    cells.append(md_cell(
        "In the ordered structure, lithium ions predominantly occupy type 5 sites (~80%) "
        "with type 2 as the secondary preference (~20%). Other site types are essentially unoccupied.\n"
        "\n"
        "## Partially disordered structure (50% anion site exchange)\n"
        "\n"
        "With 50% S/Cl site exchange \u2014 maximal anion disorder:"
    ))

    cells.append(code_cell(
        "md_structures = Xdatcar('tutorials/data/Li6PS5Cl_50p_XDATCAR.gz').structures\n"
        'print(f"Loaded trajectory with {len(md_structures)} frames")\n'
        "\n"
        "trajectory_50p = build_trajectory(md_structures[0])\n"
        "trajectory_50p.trajectory_from_structures(md_structures, progress=True)\n"
        "\n"
        'print_site_occupations(trajectory_50p, "50% anion disorder")'
    ))

    cells.append(md_cell(
        "With 50% disorder, type 5 occupation decreases while type 2 and type 4 occupation increases.\n"
        "\n"
        "## Fully inverted structure (100% anion site exchange)\n"
        "\n"
        "With complete S/Cl site exchange:"
    ))

    cells.append(code_cell(
        "md_structures = Xdatcar('tutorials/data/Li6PS5Cl_100p_XDATCAR.gz').structures\n"
        'print(f"Loaded trajectory with {len(md_structures)} frames")\n'
        "\n"
        "trajectory_100p = build_trajectory(md_structures[0])\n"
        "trajectory_100p.trajectory_from_structures(md_structures, progress=True)\n"
        "\n"
        'print_site_occupations(trajectory_100p, "100% anion disorder")'
    ))

    cells.append(md_cell(
        "## Comparing results across disorder levels\n"
        "\n"
        "The trend across the three simulations is clear:\n"
        "\n"
        "| Site type | 0% disorder | 50% disorder | 100% disorder |\n"
        "|-----------|-------------|--------------|---------------|\n"
        "| Type 5    | 80.20%      | 65.92%       | 53.39%        |\n"
        "| Type 4    | 0.02%       | 2.63%        | 7.33%         |\n"
        "| Type 3    | 0.00%       | 0.00%        | 0.00%         |\n"
        "| Type 2    | 19.78%      | 31.43%       | 39.28%        |\n"
        "| Type 1    | 0.01%       | 0.02%        | 0.00%         |\n"
        "\n"
        "With increasing anion disorder:\n"
        "\n"
        "- Type 5 occupation **decreases** from ~80% to ~53%\n"
        "- Type 2 occupation **increases** from ~20% to ~39%\n"
        "- Type 4 occupation **emerges**, growing from negligible to ~7%\n"
        "- Type 3 remains unoccupied regardless of disorder level\n"
        "\n"
        "The activation of type 4 sites with increasing disorder is significant \u2014 these sites "
        "provide additional pathways for lithium diffusion, contributing to the enhanced ionic "
        "conductivity observed experimentally in disordered argyrodites.\n"
        "\n"
        "## Summary\n"
        "\n"
        "This tutorial demonstrated several advanced `site_analysis` features for complex crystal structures:\n"
        "\n"
        "- **Multiple site types**: calling `with_polyhedral_sites` multiple times with different "
        "`centre_species` to define crystallographically distinct sites\n"
        "- **Structure alignment**: using `with_structure_alignment` to align reference and target "
        "structures via a shared atomic species\n"
        "- **Species mapping**: using `with_site_mapping` with multiple species to handle structures "
        "where two species (S and Cl) are interchangeable between the reference and real structures\n"
        "\n"
        "These techniques are applicable to any material with multiple distinct site types or chemical "
        "disorder between structurally equivalent positions."
    ))

    output_path = NOTEBOOKS_DIR / "argyrodite_site_analysis.ipynb"
    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {output_path} with {len(cells)} cells")


def generate_residence_times_and_transitions():
    nb = new_notebook()
    cells = nb["cells"]

    cells.append(md_cell(
        "# Residence Times and Transition Probabilities in Argyrodite\n"
        "\n"
        "This tutorial demonstrates how to compute residence times and transition probabilities "
        "from a site-analysis trajectory. We reuse the argyrodite Li6PS5Cl dataset from "
        "[the previous tutorial](argyrodite_site_analysis.ipynb), so the setup code here is "
        "largely the same — see that tutorial for a detailed explanation of the reference "
        "structure and trajectory builder configuration.\n"
        "\n"
        "## Prerequisites\n"
        "\n"
        "This tutorial requires the following packages:\n"
        "\n"
        "- `site-analysis`\n"
        "- `numpy`\n"
        "- `matplotlib`\n"
        "\n"
        "This tutorial uses the same trajectory data as the argyrodite tutorial.\n"
        "The data files are included in the `site-analysis` GitHub repository at\n"
        "`tutorials/data/`:\n"
        "\n"
        "- `Li6PS5Cl_0p_XDATCAR.gz` — 0% anion site exchange (fully ordered)\n"
        "- `Li6PS5Cl_50p_XDATCAR.gz` — 50% anion site exchange\n"
        "- `Li6PS5Cl_100p_XDATCAR.gz` — 100% anion site exchange\n"
        "\n"
        "These files are available in the [tutorials/data]"
        "(https://github.com/bjmorgan/site-analysis/tree/main/tutorials/data) "
        "directory of the GitHub repository. "
        "The code examples below assume you are running from the repository root directory."
    ))

    cells.append(code_cell(
        "%config InlineBackend.figure_format = 'retina'"
    ))

    cells.append(md_cell(
        "## Setting up the trajectory\n"
        "\n"
        "We set up the reference structure and trajectory builder exactly as in the argyrodite "
        "tutorial. See [that tutorial](argyrodite_site_analysis.ipynb) for a full explanation "
        "of these steps."
    ))

    cells.append(code_cell(
        "from pymatgen.io.vasp import Xdatcar\n"
        "from pymatgen.core import Structure, Lattice\n"
        "import numpy as np\n"
        "from site_analysis import TrajectoryBuilder\n"
        "\n"
        "lattice = Lattice.cubic(a=10.155)\n"
        "\n"
        "coords = np.array(\n"
        "    [[0.5,     0.5,     0.5],     # P (type 0) - PS4 tetrahedra\n"
        "     [0.9,     0.9,     0.6],     # type 1 (Li in reference)\n"
        "     [0.77,    0.585,   0.585],   # type 2 (Mg in reference, 48h)\n"
        "     [0.25,    0.25,    0.25],    # type 3 (Na in reference)\n"
        "     [0.15,    0.15,    0.15],    # type 4 (Be in reference)\n"
        "     [0.0,     0.183,   0.183],   # type 5 (K in reference)\n"
        "     [0.0,     0.0,     0.0],     # S - anion position (4a site)\n"
        "     [0.75,    0.25,    0.25],    # S - anion position (4c site)\n"
        "     [0.11824, 0.11824, 0.38176]] # S - anion position (16e site)\n"
        ")\n"
        "\n"
        "reference_structure = Structure.from_spacegroup(\n"
        '    sg="F-43m",\n'
        "    lattice=lattice,\n"
        "    species=['P', 'Li', 'Mg', 'Na', 'Be', 'K', 'S', 'S', 'S'],\n"
        "    coords=coords) * [2, 2, 2]"
    ))

    cells.append(code_cell(
        "def build_trajectory(structure):\n"
        "    builder = TrajectoryBuilder()\n"
        "    builder.with_reference_structure(reference_structure)\n"
        "    builder.with_structure(structure)\n"
        "    builder.with_mobile_species('Li')\n"
        "\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Li', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 1')\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Mg', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 2')\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Na', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 3')\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='Be', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 4')\n"
        "    builder.with_polyhedral_sites(\n"
        "        centre_species='K', vertex_species='S',\n"
        "        cutoff=3.0, n_vertices=4, label='type 5')\n"
        "\n"
        "    builder.with_structure_alignment(align_species='P')\n"
        "    builder.with_site_mapping(mapping_species=['S', 'Cl'])\n"
        "\n"
        "    return builder.build()"
    ))

    cells.append(md_cell(
        "We analyse all three disorder levels:"
    ))

    cells.append(code_cell(
        "datasets = {\n"
        "    '0%': 'tutorials/data/Li6PS5Cl_0p_XDATCAR.gz',\n"
        "    '50%': 'tutorials/data/Li6PS5Cl_50p_XDATCAR.gz',\n"
        "    '100%': 'tutorials/data/Li6PS5Cl_100p_XDATCAR.gz',\n"
        "}\n"
        "\n"
        "trajectories = {}\n"
        "for label, path in datasets.items():\n"
        "    md_structures = Xdatcar(path).structures\n"
        "    traj = build_trajectory(md_structures[0])\n"
        "    traj.trajectory_from_structures(md_structures, progress=True)\n"
        "    trajectories[label] = traj\n"
        '    print(f"{label} disorder: {len(md_structures)} frames")'
    ))

    cells.append(md_cell(
        "## Computing residence times\n"
        "\n"
        "The `residence_times()` method on each site returns the lengths of consecutive "
        "occupation runs for all atoms that visited that site. For more details, see the "
        "[residence times guide]"
        "(https://site-analysis.readthedocs.io/en/latest/guides/residence_times.html).\n"
        "\n"
        "We collect residence times for each site type by grouping sites by label:"
    ))

    cells.append(code_cell(
        "from collections import defaultdict\n"
        "\n"
        "site_labels = ['type 2', 'type 4', 'type 5']\n"
        "\n"
        "for disorder, traj in trajectories.items():\n"
        "    residence_by_label = defaultdict(list)\n"
        "    for site in traj.sites:\n"
        "        if site.label is not None:\n"
        "            times = site.residence_times()\n"
        "            residence_by_label[site.label].extend(times)\n"
        "\n"
        '    print(f"\\n{disorder} disorder:")\n'
        "    for label in site_labels:\n"
        "        times = residence_by_label[label]\n"
        "        if times:\n"
        "            arr = np.array(times)\n"
        '            print(f"  {label}: {len(arr)} visits, "\n'
        '                  f"mean = {arr.mean():.1f}, "\n'
        '                  f"median = {np.median(arr):.1f}, "\n'
        '                  f"max = {arr.max()}")\n'
        "        else:\n"
        '            print(f"  {label}: no visits")'
    ))

    cells.append(md_cell(
        "### Effect of filtering\n"
        "\n"
        "In molecular dynamics trajectories, atoms can briefly leave a site due to thermal "
        "fluctuations before returning. The `filter_length` parameter fills short gaps in the "
        "occupation sequence before computing run lengths. We demonstrate using the 50% "
        "disordered trajectory:"
    ))

    cells.append(code_cell(
        "traj_50 = trajectories['50%']\n"
        "\n"
        "for filter_length in [0, 1, 2]:\n"
        "    residence_by_label = defaultdict(list)\n"
        "    for site in traj_50.sites:\n"
        "        if site.label is not None:\n"
        "            times = site.residence_times(filter_length=filter_length)\n"
        "            residence_by_label[site.label].extend(times)\n"
        "\n"
        '    print(f"\\nfilter_length={filter_length}:")\n'
        "    for label in site_labels:\n"
        "        times = residence_by_label[label]\n"
        "        if times:\n"
        "            arr = np.array(times)\n"
        '            print(f"  {label}: {len(arr)} visits, mean = {arr.mean():.1f}")'
    ))

    cells.append(md_cell(
        "### Residence time distributions"
    ))

    cells.append(code_cell(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "disorder_levels = ['0%', '50%', '100%']\n"
        "\n"
        "fig, axes = plt.subplots(len(site_labels), len(disorder_levels),\n"
        "                         figsize=(12, 9), sharey='row')\n"
        "\n"
        "for j, disorder in enumerate(disorder_levels):\n"
        "    traj = trajectories[disorder]\n"
        "    residence_by_label = defaultdict(list)\n"
        "    for site in traj.sites:\n"
        "        if site.label is not None:\n"
        "            times = site.residence_times()\n"
        "            residence_by_label[site.label].extend(times)\n"
        "\n"
        "    for i, label in enumerate(site_labels):\n"
        "        ax = axes[i, j]\n"
        "        times = residence_by_label[label]\n"
        "        if times:\n"
        "            ax.hist(times, bins='auto', edgecolor='black', linewidth=0.5)\n"
        "        if i == 0:\n"
        "            ax.set_title(f'{disorder} disorder')\n"
        "        if j == 0:\n"
        "            ax.set_ylabel(f'{label}\\nCount')\n"
        "        if i == len(site_labels) - 1:\n"
        "            ax.set_xlabel('Residence time (frames)')\n"
        "\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(md_cell(
        "## Transition probabilities\n"
        "\n"
        "The `transition_probabilities_by_label()` method returns a row-normalised matrix "
        "showing the probability of transitioning from one site type to another. Each row "
        "sums to 1.0.\n"
        "\n"
        "Since types 1 and 3 are essentially unoccupied, we focus on the three active site "
        "types. We define a helper to print transition tables for a subset of labels:"
    ))

    cells.append(code_cell(
        "site_type_keys = ['type 2', 'type 4', 'type 5']\n"
        "\n"
        "def print_transition_table(table, fmt='.3f'):\n"
        "    print(f\"{'':>8}\", end='')\n"
        "    for key in table.keys:\n"
        "        print(f'{key:>8}', end='')\n"
        "    print()\n"
        "    for from_key in table.keys:\n"
        "        print(f'{from_key:>8}', end='')\n"
        "        for to_key in table.keys:\n"
        "            print(f'{table.get(from_key, to_key):{fmt}}', end='')\n"
        "        print()"
    ))

    cells.append(code_cell(
        "for disorder, traj in trajectories.items():\n"
        "    probs = traj.transition_probabilities_by_label().filter(site_type_keys)\n"
        "\n"
        '    print(f"\\n{disorder} disorder — transition probabilities:")\n'
        "    print_transition_table(probs)"
    ))

    cells.append(md_cell(
        "The underlying counts are available via `transition_counts_by_label()`:"
    ))

    cells.append(code_cell(
        "for disorder, traj in trajectories.items():\n"
        "    counts = traj.transition_counts_by_label().filter(site_type_keys)\n"
        "\n"
        '    print(f"\\n{disorder} disorder — transition counts:")\n'
        "    print_transition_table(counts, fmt='>8')"
    ))

    cells.append(md_cell(
        "### Visualising the transition matrices"
    ))

    cells.append(code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "\n"
        "for ax, (disorder, traj) in zip(axes, trajectories.items()):\n"
        "    probs = traj.transition_probabilities_by_label().filter(site_type_keys)\n"
        "\n"
        "    im = ax.imshow(probs.matrix, cmap='Blues', vmin=0, vmax=1)\n"
        "\n"
        "    ax.set_xticks(range(len(probs.keys)))\n"
        "    ax.set_xticklabels(probs.keys, rotation=45, ha='right')\n"
        "    ax.set_yticks(range(len(probs.keys)))\n"
        "    ax.set_yticklabels(probs.keys)\n"
        "\n"
        "    ax.set_xlabel('To')\n"
        "    ax.set_ylabel('From')\n"
        "    ax.set_title(f'{disorder} disorder')\n"
        "\n"
        "    for i in range(len(probs.keys)):\n"
        "        for j in range(len(probs.keys)):\n"
        "            val = probs.matrix[i, j]\n"
        "            if val > 0.005:\n"
        "                colour = 'white' if val > 0.5 else 'black'\n"
        "                ax.text(j, i, f'{val:.2f}',\n"
        "                        ha='center', va='center', color=colour)\n"
        "\n"
        "fig.colorbar(im, ax=axes, label='Transition probability', shrink=0.8)\n"
        "fig.tight_layout()\n"
        "plt.show()"
    ))

    output_path = NOTEBOOKS_DIR / "residence_times_and_transitions.ipynb"
    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {output_path} with {len(cells)} cells")


if __name__ == "__main__":
    generate_comparing_site_definitions()
    generate_argyrodite_site_analysis()
    generate_residence_times_and_transitions()
