#!/usr/bin/env python3
"""
This script demonstrates the preprocessing capabilities of funmixer. Specifically: 
1. Checking that a D8 flow direction raster is correctly formatted for use in funmixer
2. Fixing a D8 raster that has incorrect boundary conditions 
3. Snapping sample sites to the nearest drainage network
"""

from funmixer import (
    check_d8,
    get_sample_graphs,
    plot_network,
    set_d8_boundaries_to_zero,
    snap_to_drainage,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Checking D8 flow directions and fixing boundary conditions ###
# The check_d8 function checks two things.
# 1. if the flow-direction values are the expected values e.g., 0, 1, 2, 4, 8, 16, 32, 64, 128
# 2. if the boundary conditions are correct, i.e., all boundary cells are sinks (0's)
# The example file "d8_bad_bounds.tif" has correct values but incorrect boundary conditions.
# We can test this using the check_d8 function.

check_d8("data/d8_bad_bounds.tif")

# We can fix the boundary conditions using the set_d8_boundaries_to_zero function which sets all boundary cells to 0,
# writing the corrected raster to a new file.

set_d8_boundaries_to_zero("data/d8_bad_bounds.tif")

# Now we can check the corrected raster.
check_d8("data/d8_bad_bounds_fix_bounds.tif")

### Snapping misaligned sample sites to the nearest drainage network ###
# In general, sample sites are not perfectly aligned with the drainage network, due to
# uncertain locations or the inherent simplification in representing flow using D8. This
# means that when genreating a "sample_network" using the get_sample_graphs function,
# the generated network may be incorrect. e.g., the sample sites may be connected to the wrong
# tributary or simply be disconnected from the network entirely. This can be fixed by snapping
# the sample sites to the nearest drainage network using the snap_to_drainage function.

# First, let's generate some noisy sample sites.
# Load in real samples
samples = pd.read_csv("data/sample_data.dat", sep=" ")
sample_x, sample_y = samples["x_coordinate"], samples["y_coordinate"]
# Jitter the locations by up to 1000m
# Set the seed
np.random.seed(42)
max_noise = 1000
sample_x_noise = sample_x + np.random.uniform(-max_noise, max_noise, len(sample_x))
sample_y_noise = sample_y + np.random.uniform(-max_noise, max_noise, len(sample_x))

# Replaced the original coordinates with the noisy ones and save the file to "data/noisy_sample_data.dat"
samples["x_coordinate"] = sample_x_noise
samples["y_coordinate"] = sample_y_noise
samples.to_csv("data/noisy_sample_data.dat", sep=" ", index=False)
noisy_samples = pd.read_csv("data/noisy_sample_data.dat", sep=" ")

# When we build the sample network using the noisy samples, we can see that the network is not connected properly.

# Load sample network
sample_network, _ = get_sample_graphs(
    flowdirs_filename="data/d8.asc",
    sample_data_filename="data/noisy_sample_data.dat",
)

plt.figure(figsize=(15, 10))  # Visualise network
plot_network(sample_network)
plt.show()

# We can snap the noisy samples to the nearest drainage network using the snap_to_drainage function.
# This requires the flow direction raster and sample sites as input. The drainage_area_threshold parameter
# tells the function to only snap to drainage pixels with drainage area greater than the threshold. This is
# in the same units as the flow direction raster. The plot and save parameters control whether the snapped
# sample sites are plotted and saved to file respectively.
# The nudges parameter allows for manual nudges of sample sites to the nearest drainage pixel. For example, the
# below code nudges the sample site "CG001" by 1000m in the x-direction and -1000m in the y-direction. These
# can be visualised by setting plot=True. It may take some trial and error to get the nudges right and the samples
# snapped to the correct drainage pixel.

snap_to_drainage(
    flow_dirs_filename="data/d8.asc",
    sample_sites_filename="data/noisy_sample_data.dat",
    drainage_area_threshold=1e3,
    plot=True,
    save=True,
    nudges={"CG001": np.array([1000, -1000])},
)

# Once this is done, we can load in the snapped sample sites and build the sample network again.
# Load sample network
sample_network, _ = get_sample_graphs(
    flowdirs_filename="data/d8.asc",
    sample_data_filename="data/noisy_sample_data_snapped.dat",
)

plt.figure(figsize=(15, 10))  # Visualise network
plot_network(sample_network)
plt.show()

# The network should now be connected properly, but should be checked to ensure that the snapping was successful
# and the samples have been snapped to the correct part of the network.
