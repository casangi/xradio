
# Development Setup for `xradio` with `casatools` I/O Backend

This guide describes how to set up a local development environment for `xradio`, specifically configured to use the `casatools` I/O backend for interacting with CASA MeasurementSets (MS) and Images.
This setup provides a unified development environment that can combine RADPS-targeted new libraries (e.g. `xradio`, `astroviper`) with legacy CASA/Pipeline codebase, facilitating experimentation, prototyping, and benchmarking.

**Prerequisites:**

* Git
* Conda, e.g. [`Miniforge`](https://github.com/conda-forge/miniforge)

## 1. Clone Repositories

Clone the necessary repositories. If you only need to develop `xradio` itself, cloning it might be sufficient. However, for full local development, especially if modifying dependencies, clone the `viper` libraries as well.

```console
# Clone xradio (required)
git clone https://github.com/casangi/xradio.git
cd xradio

# Clone viper libraries (optional, but recommended for full local dev)
# Ensure these are cloned in a location accessible for editable installs,
# e.g., alongside the xradio directory.
# git clone https://github.com/casangi/astroviper.git ../astroviper
# git clone https://github.com/casangi/toolviper.git ../toolviper
# git clone https://github.com/casangi/graphviper.git ../graphviper
```

(Adjust paths like ../astroviper based on your preferred directory structure)

## 2. Create and Activate Conda Environment

Create a dedicated Conda environment using the provided YAML configuration. This ensures all dependencies, including casatools, are installed correctly.

* Create a file named xradio_casatools_dev.yaml (or similar) with the following content:

```yaml
name: xradio_casatools_dev # Changed name slightly for clarity

channels:
  - conda-forge
  - defaults

dependencies:
  # Core environment tools
  - conda-ecosystem-user-package-isolation
  - python=3.12
  - ipython
  - pip

  # CASA dependencies via pip using NRAO's repository
  - pip:
    # Specify NRAO's pip index for casatools/casatasks
    - --extra-index-url https://casa-pip.nrao.edu/repository/pypi-group/simple
    - casatools == 6.7.0.* # Specify desired casatools version
    - casatasks == 6.7.0.* # Optional: if casatasks needed

    # Install xradio in editable mode from the current directory clone
    # Include optional dependencies for interactive use and documentation
    - -e .[interactive,docs]

    # --- Add these lines if you cloned the viper libraries for local dev ---
    # Install viper libraries in editable mode from their respective clones
    # Adjust paths relative to where you run the pip install command (e.g., from within the xradio dir)
    # - -e ../astroviper/.
    # - -e ../toolviper/.
    # - -e ../graphviper/.
    # --------------------------------------------------------------------
```

* Create and activate the environment:

```console
# Navigate to the directory containing the YAML file if needed
conda env create --file xradio_casatools_dev.yaml
conda activate xradio_casatools_dev
```

(Use conda env update --file xradio_casatools_dev.yaml --prune if the environment already exists and you want to update it)

## 3. Handle Potential `python-casacore` Conflicts

The casatools package includes its own casacore build and Python binding. If python-casacore (a separate package often installed from conda-forge or PyPI) is also present, it can lead to namespace conflicts. The current workaround for activiting the casatools-backend is to uninstall `python-casacore` in the Python development environment when using `casatools`.

```console
# Run this after activating the environment
pip uninstall python-casacore
```

### 4. Running Unit Tests Locally

You can run unit tests using pytest. Execute these commands from the root directory of your xradio clone.

* Run a specific test function:

    ```console
    python -m pytest -vvs tests/unit/test_image.py::TestCasacoreToXdsToCasacore::test_pixels_and_mask
    ```

    (Note: Class name might differ slightly depending on test structure, adjust if needed)

* Run all tests in a specific file:

    ```console
    python -m pytest -vvs tests/unit/test_image.py
    ```

* Run all xradio tests (not fully working yet):

    ```console
    python -m pytest -vvs tests/
    ```

## 5. Optional: Setup Parallelization (Dask Cluster)

For performance testing or parallel processing workflows, you can set up a local Dask cluster using toolviper.

Important Note: casatools (and potentially python-casacore) are generally not thread-safe. When using Dask with xradio tasks involving these libraries, configure your Dask cluster to use processes instead of threads to avoid race conditions and crashes.

* Start a local Dask cluster using processes:

```python
# Example Python script or interactive session
from toolviper.dask.client import local_client

# Use processes=True or set n_workers > 0 and threads_per_worker=1
# serial_execution=False enables parallelism
viper_client = local_client(
    cores=8,             # Total number of cores (processes) for the cluster
    memory_limit="4GB",  # Memory limit per worker process
    serial_execution=False,
)

print(viper_client) # Display cluster information

```

* Adjust logging verbosity (optional):

```python
import toolviper.utils.logger as logger

viperlog = logger.get_logger()
viperlog.setLevel('DEBUG') # Set desired level (e.g., INFO, DEBUG, WARNING)
```
