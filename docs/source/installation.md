# Installation

This guide covers how to install the `site_analysis` package.

## Requirements

`site_analysis` requires Python 3.10 or later.

## Installing from PyPI

The recommended way to install `site_analysis` is via pip:

```bash
pip install site-analysis
```

This will automatically install all required dependencies.

## Optional Dependencies

For faster polyhedral site analysis, you can install with [numba](https://numba.pydata.org/) acceleration:

```bash
pip install site-analysis[fast]
```

This enables JIT-compiled containment testing, which typically gives a ~7x speedup when analysing polyhedral sites.

## Installing from Source

For development or to access the latest features before release, you can install directly from the source code:

1. Clone the repository:
   ```bash
   git clone https://github.com/bjmorgan/site_analysis.git
   cd site_analysis
   ```

2. Install in development mode:
   ```bash
   pip install -e .
   ```

The `-e` flag installs the package in editable mode, meaning changes to the source code will be reflected in your environment without reinstalling.

## Verifying Installation

To verify that `site_analysis` installed correctly, you can import it in Python:

```python
import site_analysis
```

If no errors occur, the installation was successful.

## Development Installation

If you plan to contribute to `site_analysis`, you may want to install additional development dependencies:

```bash
pip install -e ".[dev]"
```

This includes packages needed for testing and building documentation.
