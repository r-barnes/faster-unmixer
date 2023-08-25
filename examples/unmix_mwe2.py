import logging

import cvxpy as cp
import funmixer
import pandas as pd

logger: logging.Logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


def main() -> None:
    flowdirs_filename = "data/d8.asc"
    data_filename = "data/sample_data.dat"
    excluded_elements = ["Bi", "S"]

    sample_network, _ = funmixer.get_sample_graphs(flowdirs_filename, data_filename)

    funmixer.plot_network(sample_network)
    obs_data = pd.read_csv(data_filename, delimiter=" ")
    obs_data = obs_data.drop(columns=excluded_elements)

    problem = funmixer.SampleNetworkUnmixer(sample_network=sample_network)

    funmixer.get_unique_upstream_areas(problem.sample_network)

    if len(funmixer.ELEMENT_LIST) == 0:
        raise Exception("No elements to process!")

    results = None
    # TODO(r-barnes,alexlipp): Loop over all elements once we achieve acceptable results
    for element in funmixer.ELEMENT_LIST[0:20]:
        if element not in obs_data.columns:
            continue

        logger.info(f"\n\033[94mProcessing element '{element}'...\033[39m")

        element_data = funmixer.get_element_obs(element=element, obs_data=obs_data)
        try:
            solution = problem.solve(element_data, solver="ecos", regularization_strength=1e-3)
        except cp.error.SolverError as err:
            logger.error(f"\033[91mSolver Error - skipping this element!\n{err}")
            continue

        if results is None:
            results = pd.DataFrame(element_data.keys())
        results[element + "_obs"] = [element_data[sample] for sample in element_data]
        results[element + "_dwnst_prd"] = [
            solution.downstream_preds[sample] for sample in element_data
        ]

    assert results is not None
    print(results)


if __name__ == "__main__":
    main()
