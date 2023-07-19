#!/usr/bin/env python3

"""
This script is a minimum working example for how to unmix the downstream observations for a specific element, using monte-carlo resampling
to estimate uncertainties on the predictions. For a given relative uncertainty on the provided input observations, it performs a number of
resamples of the observations, and for each resample, it solves the unmixing problem and obtains predicted downstream and upstream concentrations.
It treats each sub-basin defined by a sample as a discrete variable, and solves for the upstream concentrations of the element in each sub-basin.
For each sub-basin, it returns a list of predicted concentrations for each resampling, which can be used to estimate the uncertainty on the prediction, 
propagating the uncertainty on the observations into the predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import sample network unmixer module.
# This module contains the SampleNetworkUnmixer class, which builds the optimization problem and solves it.
import sample_network_unmix as snu

# Constants
element = "Mg"  # Set element

# Load sample network
sample_network, _ = snu.get_sample_graphs(
    flowdirs_filename="data/d8.asc",
    sample_data_filename="data/sample_data.dat",
)

# Load in observations
obs_data = pd.read_csv("data/sample_data.dat", delimiter=" ")
obs_data = obs_data.drop(columns=["Bi", "S"])

plt.figure(figsize=(15, 10))  # Visualise network
snu.plot_network(sample_network)
plt.show()
print("Building problem...")
problem = snu.SampleNetworkUnmixer(sample_network=sample_network)


element_data = snu.get_element_obs(
    element, obs_data
)  # Return dictionary of {sample_name:concentration}

snu.plot_sweep_of_regularizer_strength(problem, element_data, -5, -1, 11)
regularizer_strength = 10 ** (-3.0)
print(
    f"Chose regularization strength of {regularizer_strength} at 'elbow' of misfit-roughness curve."
)

relative_error = 10  # % relative error on observations

element_pred_down_mc, element_pred_up_mc = problem.solve_montecarlo(
    element_data,
    relative_error=relative_error,
    num_repeats=50,
    regularization_strength=regularizer_strength,
    solver="ecos",
)

downstream_means, downstream_uncertainties = {}, {}
for sample, values in element_pred_down_mc.items():
    downstream_uncertainties[sample] = np.std(values)
    downstream_means[sample] = np.mean(values)

upstream_means, upstream_uncertainties = {}, {}
for sample, values in element_pred_up_mc.items():
    upstream_uncertainties[sample] = np.std(values)
    upstream_means[sample] = np.mean(values)

area_dict = snu.get_unique_upstream_areas(sample_network)  # Extract areas for each basin
upstream_map = snu.get_upstream_concentration_map(
    area_dict, upstream_means
)  # Assign to upstream preds

upstream_uncertainty_map = snu.get_upstream_concentration_map(
    area_dict, upstream_uncertainties
)  # Assign to upstream preds

# Visualise outputs upstream
plt.imshow(upstream_map)
cb = plt.colorbar()
cb.set_label(element + "concentration mg/kg")
plt.title("Upstream Concentration Map")
plt.show()

# Visualise outputs upstream
plt.imshow(upstream_uncertainty_map)
cb = plt.colorbar()
cb.set_label(element + "uncertainty mg/kg")
plt.title("Upstream Uncertainties")
plt.show()
