# Unmixing river sediments for the elemental geochemistry of their source regions

This notebook provides a minimum working example (`fast_inversion_mwe.ipynb`) demonstrating how to invert river sediment geochemistry for the composition of their source regions.

Unlike the previously published method [(Lipp et al. 2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GC009838) this is significantlyfaster. This speed-up is achieved through using Powell's algorithm instead of the Nelder-Mead to optimise the objective function and by parallelising the flow accumulation. We also initialise the solution closer to the solution than if just a constant initial composition is used but this has just a marginal impact on the runtime.

# Data input assumptions 

The algorithm requires:

1) A GDAL readable raster of D8 flow directions. We use the ESRI/Arc D8 convention of representing directions with increasing powers of 2 (i.e., 1, 2, 4, 8 etc.) with sink pixels indicated by 0. 

2) A tab-delimited file which contains the names, locations and geochemical observations at the sample sites. Sample names are given in column `Sample.Code`, and the x and y-coordinates of the sample sites in columns `x_coordinate` and `y_coordinate`. The x and y-coordinates of the sample sites should correspond to the *upper left* corner of pixels, in the same reference system as the D8 raster. It is assumed that the sample sites have already been manually aligned onto the drainage grid.  Subsequent columns contain the name of a given tracer (e.g., `Mg`) and their concentrations (arbitrary units). 

Example datasets are given in `data/d8.asc` and `sample_data.dat`.


# Compiling

Check out submodules:
```
git submodule update --init --recursive
```

Install prereqs:
```
sudo apt install pybind11-dev cmake
```

Compile the C++ code:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_GDAL=ON ..
make
cd ..
```

Run the optimization problem:
```
python3 run.py
```