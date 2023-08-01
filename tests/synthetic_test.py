# pyre-fixme[21]: Could not find module `matplotlib.pyplot`.
import matplotlib.pyplot as plt
import numpy as np

import funmixer


def main() -> None:
    max_val = 1000  # maximum value for synthetic input
    rel_err = 5  # 5% relative error

    # Load in Drainage Network
    sample_network, _ = funmixer.get_sample_graphs(
        flowdirs_filename="data/synthetic_topo_d8.asc",
        sample_data_filename="data/synthetic_samples.dat",
    )  # Get upstream basins

    # plt.figure(figsize=(15, 10))  # Visualise network
    # plt.title("Sample Network")
    # funmixer.plot_network(sample_network)

    areas = funmixer.get_unique_upstream_areas(sample_network)

    # Pick random values to set each basin
    np.random.seed(0)
    print("Generating synthetic data...")
    synth_upst_concs = {sample: max_val * np.random.rand() for sample in areas.keys()}

    # Pick random export rates to set for each sub-basin
    input_export_rates = {sample: np.random.rand() for sample in areas.keys()}

    # Generate synthetic upstream concentration map
    upst_conc_map = funmixer.get_upstream_concentration_map(areas, synth_upst_concs)
    # Predict concentration at downstream observation points
    print("Predicting downstream concentrations...")
    mixed_synth_down, _ = funmixer.mix_downstream(
        sample_network=sample_network,
        areas=areas,
        concentration_map=upst_conc_map,
        export_rates=input_export_rates,
    )
    ## Add noise
    # mixed_synth_down = {
    #     sample: value * np.random.normal(loc=1, scale=rel_err / 100)
    #     for sample, value in mixed_synth_down.items()
    # }
    # Set up problem
    print("Building problem...")
    problem = funmixer.SampleNetworkUnmixer(sample_network, use_regularization=False)
    # Solve problem using synthetic downstream observations as input
    solution = problem.solve(mixed_synth_down, solver="ecos", export_rates=input_export_rates)
    # Generate recovered upstream concentration map
    print("Generating recovered upstream concentration map...")
    # pyre-fixme[6]: For 2nd argument expected `Dict[str, ndarray[typing.Any,
    #  dtype[typing.Any]]]` but got `Union[ndarray[typing.Any, typing.Any], Dict[str,
    #  float]]`.
    recovered_conc_map = funmixer.get_upstream_concentration_map(areas, solution.upstream_preds)

    # Extract arrays of predicted and `observed' concentrations
    down_preds = [solution.downstream_preds[sample] for sample in solution.downstream_preds.keys()]
    down_obs = [mixed_synth_down[sample] for sample in solution.downstream_preds.keys()]
    # pyre-fixme[16]: Item `ndarray` of `Union[Dict[str, float], ndarray[typing.Any,
    #  np.dtype[typing.Any]]]` has no attribute `keys`.
    up_preds = [solution.upstream_preds[sample] for sample in solution.upstream_preds.keys()]
    up_obs = [synth_upst_concs[sample] for sample in solution.downstream_preds.keys()]

    print("Testing results...")
    tolerance = 1e-5  # Tolerance for asserting equality of values
    # Assert down_preds and down_obs contain equal values within a tolerance
    # pyre-fixme[6]: For 1st argument expected `Union[_SupportsArray[dtype[typing.Any]],
    #  _NestedSequence[_SupportsArray[dtype[typing.Any]]], _NestedSequence[Union[bool,
    #  bytes, complex, float, int, str]], bool, bytes, complex, float, int, str]` but got
    #  `List[float]`.
    # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[typing.Any]],
    #  _NestedSequence[_SupportsArray[dtype[typing.Any]]], _NestedSequence[Union[bool,
    #  bytes, complex, float, int, str]], bool, bytes, complex, float, int, str]` but got
    #  `List[float]`.
    if np.allclose(down_preds, down_obs, rtol=tolerance):
        print(
            "SUCCESS: Downstream concentrations match to within tolerance of {}%".format(tolerance)
        )
    else:
        raise AssertionError(
            "WARNING: Inputted and recovered downstream concentrations do not match with tolerance of {}%".format(
                tolerance
            )
        )
    # Assert up_preds and up_obs contain equal values within a tolerance
    # pyre-fixme[6]: For 1st argument expected `Union[_SupportsArray[dtype[typing.Any]],
    #  _NestedSequence[_SupportsArray[dtype[typing.Any]]], _NestedSequence[Union[bool,
    #  bytes, complex, float, int, str]], bool, bytes, complex, float, int, str]` but got
    #  `List[Union[ndarray[typing.Any, dtype[typing.Any]], float]]`.
    # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[typing.Any]],
    #  _NestedSequence[_SupportsArray[dtype[typing.Any]]], _NestedSequence[Union[bool,
    #  bytes, complex, float, int, str]], bool, bytes, complex, float, int, str]` but got
    #  `List[typing.Any]`.
    if np.allclose(up_preds, up_obs, rtol=tolerance):
        print("SUCCESS: Upstream concentrations match to within tolerance of {}%".format(tolerance))
    else:
        raise AssertionError(
            "WARNING: Inputted and recovered upstream concentrations do not match with tolerance of {}%".format(
                tolerance
            )
        )

    print("SUCCESS: All tests passed!")
    # Visualise results of synthetic test
    print("Visualising results...")
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


if __name__ == "__main__":
    main()
