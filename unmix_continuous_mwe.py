# Preamble
import matplotlib.pyplot as plt
import pandas as pd

import geochem_inverse_optimize as gio

sample_network, sample_adjacency = gio.get_sample_graphs("data/")

obs_data = pd.read_csv("data/geochem_no_dupes.dat", delimiter=" ")
obs_data = obs_data.drop(columns=["Bi", "S"])

element = "Mg"  # Set element
sample_network, sample_adjacency = gio.get_sample_graphs("data/")

area_map = plt.imread("labels.tif")[:, :, 0]

print("Building problem...")
problem = gio.SampleNetworkContinuous(
    sample_network=sample_network, area_labels=area_map, nx=60, ny=60
)
element_data = gio.get_element_obs(
    element, obs_data
)  # Return dictionary of {sample_name:concentration}
print("Solving problem...")
down_dict, upst_map = problem.solve(
    element_data, regularization_strength=10 ** (-0.5), solver="ecos"
)

print("Visualising output...")
plt.imshow(upst_map)
plt.colorbar()
plt.show()

gio.visualise_downstream(pred_dict=down_dict, obs_dict=element_data, element=element)
plt.show()
