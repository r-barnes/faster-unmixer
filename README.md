# Unmixing nested observed concentrations in river networks for source regions

This repository implements an efficient solution to the unmixing of nested concentrations in a (river) network using convex optimisation. The method is described in our [EGU abstract](https://meetingorganizer.copernicus.org/EGU23/EGU23-5368.html) 

## Data input assumptions

The algorithm requires:

1) A GDAL readable raster of D8 flow directions. We use the ESRI/Arc D8 convention of representing directions with increasing powers of 2 (i.e., 1, 2, 4, 8 etc.) with sink pixels indicated by 0. We assume that every cell in the domain eventually flows into a sink node within the domain (or is itself a sink node). This assumption requires that **every boundary pixel is set to be a sink**.

2) A **space**-delimited file which contains the names, locations and geochemical observations at the sample sites. Sample names are given in column `Sample.Code`, and the x and y-coordinates of the sample sites in columns `x_coordinate` and `y_coordinate`. The x and y-coordinates of the sample sites need to be in the same reference system as the D8 raster. It is assumed that the sample sites have already been manually aligned onto the drainage network. Subsequent columns contain the name of a given tracer (e.g., `Mg`) and their concentrations (arbitrary units).

`funmixer` does include some basic data preprocessing functions that can be used to align the sample sites to the drainage network and fix the boundary conditions of the D8 raster. An example of use is given in the `examples/` directory. Example, valid, datasets are contained in `data/d8.asc` and `sample_data.dat`. 

Some common data input problems can be solved by: 
- Checking that there is not trailing white-space at the end of the sample site data table. 
- Ensuring that the D8 flow-direction raster is in the same reference system as the sample site coordinates. 
- Ensuring that the D8 raster is surrounded by a boundary of sink pixels.


## Installation

The following assumes a UNIX operating systems. If running Windows OS you will need to install a [Linux subsystem](https://learn.microsoft.com/en-us/windows/wsl/about). 

First, *clone* the repository into a local directory:

```
git clone --recurse-submodules https://github.com/r-barnes/faster-unmixer/ [LOCAL_DIRECTORY]
```
If you've cloned without getting submodules you can acquire them by navigating to the local directory and running:
```
git submodule update --init --recursive
```

If using conda environments, a conda environment file (`requirements.yaml`) is provided containing the python dependencies. A conda environment entitled `funmixer` can be generated from it using `conda env create -f requirements.yaml`. The environment can then be activated using `conda activate funmixer`.

Next, install the python package using:

```
pip install -e .
```

This command installs the `funmixer` python package that can then be imported as normal (e.g., `import funmixer`).

### Problem solving

If you encounter any problems with installation you can contact us or raise an issue on this repository. Based on user feedback, some common problems and solutions are given below:

- If you're having problems related to permissions, try using `sudo` before the `pip` command (e.g., `sudo pip install -e .`).

- Some users have reported installation problems due to missing pybind11 headers. If this is the case, try installing pybind11 directly. Instructions are available [here](https://pybind11.readthedocs.io/en/stable/installing.html).

## Testing

To check if installation has happened correctly you can run the synthetic test script:

```
python3 tests/synthetic_test.py
```

This script generates a synthetic dataset and recovers the original input. The results are then plotted.

### Unit-tests

Formal unit-tests can be run using:

```
pytest tests/random_networks_test.py
```
These tests randomly generate sample networks (of three types: random trees, full R-ary trees, and balanced trees) up to 100 nodes in size, with random source concentrations and sub-basin areas drawn from distributions spanning two orders of magnitude. The tests pass if all the inputted upstream source chemistry is recovered to a relative accuracy of 1%.

### Runtime Benchmark

A timing benchmark can be run using:

```
python tests/runtime_benchmark.py run
```
This script benchmarks the runtime of the algorithm for the `GUROBI`, `ECOS` and `SCS` solvers for branching networks up to 500 nodes. This takes ~ 30 minutes to run on standard laptop hardware. The results are cached to file and be visualised using: 

```
python tests/runtime_benchmark.py plot
```

## Usage

Some documented example scripts are given in the directory `examples/`, and are run from the root directory of the repository, e.g.,

```
python examples/unmix_montecarlo_mwe.py
```

## Cite 

If you use this please cite the preprint, which is under review at *Water Resources Research*.

> Barnes, R. and Lipp, A. _Using convex optimization to efficiently apportion tracer and pollutant sources from point concentration observations_, Preprint DOI [10.31223/X5708M](https://doi.org/10.31223/X5708M), 2023. 

A `.cff` citation file is also provided in the repository.
