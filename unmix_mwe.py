# Preamble
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

import geochem_inverse_optimize as gio

print(sys.version)
print(os.getcwd())

# Load in observations
obs_data = pd.read_csv("data/geochem_no_dupes.dat", delimiter=" ")
obs_data = obs_data.drop(columns=["Bi", "S"])

element = "Be"  # Set element
sample_network, sample_adjacency = gio.get_sample_graphs("data/")

plt.figure(figsize=(15, 10))  # Visualise network
gio.plot_network(sample_network)
plt.show()
problem = gio.SampleNetwork(sample_network=sample_network, sample_adjacency=sample_adjacency)


element_data = gio.get_element_obs(
    element, obs_data
)  # Return dictionary of {sample_name:concentration}
element_pred_down = problem.solve(
    element_data, solver="ecos", regularization_strength=1e-3
)  # Solve for downstream predictions


element_pred_upstream = gio.get_upstream_prediction_dictionary(
    sample_network
)  # Get upstream predictions
area_dict = gio.get_unique_upstream_areas(sample_network)  # Extract areas for each basin
upstream_map = gio.get_upstream_concentration_map(
    area_dict, element_pred_upstream
)  # Assign to upstream preds

# Visualise outputs downstream
gio.visualise_downstream(pred_dict=element_pred_down, obs_dict=element_data, element=element)
plt.show()
# Visualise outputs upstream
plt.imshow(upstream_map)
cb = plt.colorbar()
cb.set_label("Be concentration mg/kg")
plt.show()
