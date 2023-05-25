# Unmixing river sediments for the elemental geochemistry of their source regions

This notebook provides a minimum working example (`fast_inversion_mwe.ipynb`) demonstrating how to invert river sediment geochemistry for the composition of their source regions.

Unlike the previously published method [(Lipp et al. 2021)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GC009838) this is significantlyfaster. This speed-up is achieved through using Powell's algorithm instead of the Nelder-Mead to optimise the objective function and by parallelising the flow accumulation. We also initialise the solution closer to the solution than if just a constant initial composition is used but this has just a marginal impact on the runtime.



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