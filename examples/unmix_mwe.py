#!/usr/bin/env python3

"""
This script is a minimum working example for how to unmix the downstream observations for a specific element.
It treats each sub-basin defined by a sample as a discrete variable, and solves for the upstream concentrations of the element in each sub-basin.

It loads a sample network graph and observations from data files, visualizes the network, and builds the optimization problem.
Then, it performs a sweep of different regularization strengths for the problem and visualizes the results.
Next, it solves the problem using a specified solver and regularization strength, obtaining predicted downstream and upstream concentrations.
The script also calculates unique upstream areas for each basin in the network and generates an upstream concentration map based on the predictions.
Finally, it visualizes the predicted downstream concentrations and the upstream concentration map for the specified element (default: Mg).
"""

import logging

# pyre-fixme[21]: Could not find module `matplotlib.pyplot`.
import matplotlib.pyplot as plt
import pandas as pd

# Import sample network unmixer module.
# This module contains the SampleNetworkUnmixer class, which builds the optimization problem and solves it.
import funmixer

logging.getLogger().addHandler(logging.StreamHandler())


def main() -> None:
    # Constants
    element = "Mg"  # Set element

    # Load sample network
    sample_network, _ = funmixer.get_sample_graphs(
        flowdirs_filename="data/d8.asc",
        sample_data_filename="data/sample_data.dat",
    )

    # Load in observations
    obs_data = pd.read_csv("data/sample_data.dat", delimiter=" ")
    obs_data = obs_data.drop(columns=["Bi", "S"])

    plt.figure(figsize=(15, 10))  # Visualise network
    funmixer.plot_network(sample_network)
    plt.show()
    print("Building problem...")
    problem = funmixer.SampleNetworkUnmixer(sample_network=sample_network)

    element_data = funmixer.get_element_obs(
        element, obs_data
    )  # Return dictionary of {sample_name:concentration}

    funmixer.plot_sweep_of_regularizer_strength(problem, element_data, -5, -1, 11)
    regularizer_strength = 10 ** (-3.0)
    print(
        f"Chose regularization strength of {regularizer_strength} at 'elbow' of misfit-roughness curve."
    )

    print("Solving problem...")
    solution = problem.solve(element_data, solver="ecos", regularization_strength=10 ** (-3))

    area_dict = funmixer.get_unique_upstream_areas(sample_network)  # Extract areas for each basin
    upstream_map = funmixer.get_upstream_concentration_map(
        area_dict,
        solution.upstream_preds,
    )  # Assign to upstream preds

    # Visualise outputs downstream
    funmixer.visualise_downstream(
        pred_dict=solution.downstream_preds, obs_dict=element_data, element=element
    )
    plt.show()
    # Visualise outputs upstream
    plt.imshow(upstream_map)
    cb = plt.colorbar()
    cb.set_label(element + "concentration mg/kg")
    plt.title("Upstream Concentration Map")
    plt.show()


if __name__ == "__main__":
    main()
