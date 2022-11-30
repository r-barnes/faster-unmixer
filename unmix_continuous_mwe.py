# Preamble
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import geochem_inverse_optimize as gio

sample_network, sample_adjacency = gio.get_sample_graphs("data/")

obs_data = pd.read_csv("data/geochem_no_dupes.dat", delimiter=" ")
obs_data = obs_data.drop(columns=["Bi", "S"])

element = "Mg"  # Set element
sample_network, sample_adjacency = gio.get_sample_graphs("data/")
regularizer_strength = 10 ** (-1)
area_map = plt.imread("labels.tif")[:, :, 0]

print("Building problem...")
problem = gio.SampleNetwork(
    sample_network=sample_network, ny=60, nx=60, area_labels=area_map, continuous=True
)
element_data = gio.get_element_obs(
    element, obs_data
)  # Return dictionary of {sample_name:concentration}

gio.plot_sweep_of_regularizer_strength(problem, element_data, -5, -1, 11)

print("Solving problem...")
down_dict, upst_map = problem.solve(
    element_data, regularization_strength=regularizer_strength, solver="ecos"
)
print("Visualising output...")
plt.imshow(upst_map)
plt.colorbar()
plt.show()

gio.visualise_downstream(pred_dict=down_dict, obs_dict=element_data, element=element)
plt.show()


relative_error = 10  #%
print("Calculating uncertainties with monte-carlo sampling")
element_pred_down_mc, element_pred_up_mc = problem.solve_montecarlo(
    element_data,
    relative_error=relative_error,
    num_repeats=50,
    regularization_strength=regularizer_strength,
    solver="ecos",
)
stacked_upst = np.dstack(element_pred_up_mc)
stds = np.std(stacked_upst, axis=2)
plt.imshow(stds)
plt.title("Upstream Uncertainties")
plt.colorbar()
plt.show()
