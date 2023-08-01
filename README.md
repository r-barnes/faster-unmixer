# Unmixing nested observed concentrations in river networks for source regions

This repository implements an efficient solution to the unmixing of nested concentrations in a (river) network using convex optimisation. The method is described in our [EGU abstract](https://meetingorganizer.copernicus.org/EGU23/EGU23-5368.html) 

## Data input assumptions

The algorithm requires:

1) A GDAL readable raster of D8 flow directions. We use the ESRI/Arc D8 convention of representing directions with increasing powers of 2 (i.e., 1, 2, 4, 8 etc.) with sink pixels indicated by 0. We assume that every cell in the domain eventually flows into a sink node within the domain (or is itself a sink node). This assumption requires that **every boundary pixel is set to be a sink**.

2) A tab-delimited file which contains the names, locations and geochemical observations at the sample sites. Sample names are given in column `Sample.Code`, and the x and y-coordinates of the sample sites in columns `x_coordinate` and `y_coordinate`. The x and y-coordinates of the sample sites need to be in the same reference system as the D8 raster. It is assumed that the sample sites have already been manually aligned onto the drainage network.  Subsequent columns contain the name of a given tracer (e.g., `Mg`) and their concentrations (arbitrary units).

Example datasets are given in `data/d8.asc` and `sample_data.dat`.


## Compiling 

The following assumes a UNIX operating systems. If running Windows OS you will need to install a [Linux subsystem](https://learn.microsoft.com/en-us/windows/wsl/about). 

Check out submodules:
```
git submodule update --init --recursive
```

Install the python package.

```
pip install -e .
```

This command installs the `funmixer` python package that can be imported as normal (e.g., `import funmixer`).

A conda environment file (`requirements.yaml`) is provided containing the python dependencies. A conda environment entitled `funmixer` can be generated from it using `conda env create -f requirements.yaml`.    

### Testing compilation

To test if installation has happened correctly run the unit test:
```
python3 synthetic_test.py
```


## Usage

Some documented example scripts are given in the directory `examples/`.
```

## Cite 

If you use this please cite: 

Lipp, A. and Barnes, R.: Identifying tracer and pollutant sources in drainage networks from point observations using an efficient convex unmixing scheme, EGU General Assembly 2023, Vienna, Austria, 24–28 Apr 2023, EGU23-5368, [DOI: 10.5194/egusphere-egu23-5368](https://doi.org/10.5194/egusphere-egu23-5368), 2023.