import matplotlib.pyplot as plt
import numpy as np

import geochem_inverse_optimize as gio

max_val = 1000  # maximum value for synthetic input
rel_err = 5  # 5% relative error

# Load in Drainage Network
sample_network, sample_adjacency = gio.get_sample_graphs(
    flowdirs_filename="data/synthetic_topo_d8.asc",
    sample_data_filename="data/synthetic_samples.dat",
)  # Get upstream basins

plt.figure(figsize=(15, 10))  # Visualise network
plt.title("Sample Network")
gio.plot_network(sample_network)

areas = gio.get_unique_upstream_areas(sample_network)

# Pick random values to set each basin
synth_upst_concs = {sample: max_val * np.random.rand() for sample in areas.keys()}

# Pick random export rates to set for each sub-basin
input_export_rates = {sample: np.random.rand() for sample in areas.keys()}

# Generate synthetic upstream concentration map
upst_conc_map = gio.get_upstream_concentration_map(areas, synth_upst_concs)
# Predict concentration at downstream observation points
mixed_synth_down, _ = gio.mix_downstream(
    sample_network, areas, upst_conc_map, export_rates=input_export_rates
)
## Add noise
# mixed_synth_down = {
#     sample: value * np.random.normal(loc=1, scale=rel_err / 100)
#     for sample, value in mixed_synth_down.items()
# }
# Set up problem
problem = gio.SampleNetwork(sample_network, sample_adjacency, use_regularization=False)
# Solve problem using synthetic downstream observations as input
recovered_down, recovered_up = problem.solve(
    mixed_synth_down, solver="ecos", export_rates=input_export_rates
)
# Generate recovered upstream concentration map
recovered_conc_map = gio.get_upstream_concentration_map(areas, recovered_up)

# Extract arrays of predicted and `observed' concentrations
down_preds = [recovered_down[sample] for sample in recovered_down.keys()]
down_obs = [mixed_synth_down[sample] for sample in recovered_down.keys()]
up_preds = [recovered_up[sample] for sample in recovered_down.keys()]
up_obs = [synth_upst_concs[sample] for sample in recovered_down.keys()]

# Visualise results of synthetic test
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.title("Inputted")
plt.imshow(upst_conc_map, vmin=0, vmax=1000)
cb = plt.colorbar()
cb.set_label("Concentration")
plt.subplot(2, 2, 2)
plt.title("Recovered")
plt.imshow(recovered_conc_map, vmin=0, vmax=1000)
plt.subplot(2, 2, 3)
plt.plot([0, 1e6], [0, 1e6], alpha=0.5, color="grey")
plt.scatter(down_obs, down_preds)
plt.axis("equal")
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel("Synthetic downstream concentration")
plt.ylabel("Recovered downstream concentration")
plt.subplot(2, 2, 4)
plt.plot([0, 1e6], [0, 1e6], alpha=0.5, color="grey")
plt.scatter(up_obs, up_preds)
plt.axis("equal")
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.xlabel("Synthetic upstream concentration")
plt.ylabel("Recovered upstream concentration")
plt.tight_layout()
plt.show()
