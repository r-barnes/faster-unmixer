import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, List, Optional, Tuple, TypeVar

# pyre-fixme[21]: Could not find module `matplotlib.pyplot`.
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm

# pyre-fixme[21]: Could not find module `random_networks_test`.
from random_networks_test import (
    conc_list_to_dict,
    draw_random_log_uniform,
    generate_r_ary_sample_network,
)

import funmixer

T = TypeVar("T")

# 100 networks up to 500 nodes in size takes about 12 minutes to run on my machine

# Set the range to explore upstream concentration values over
MINIMUM_CONC = 1
MAXIMUM_CONC = 1e2

# Set the range to explore sub-basin area values over
MINIMUM_AREA = 1
MAXIMUM_AREA = 1e2

# Set branching factor
BRANCHING_FACTOR = 3

# Set maximum number of nodes to try
MAXIMUM_NETWORK_SIZE = 250

# Set number of networks to test
NUMBER_OF_NETWORKS = 100

# Number of times to run the test to average run-time variations
TEST_REPEATS = 1


@dataclass
class BenchmarkResults:
    first_solves: DefaultDict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    subsequent_solves: DefaultDict[int, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    solver_times: DefaultDict[int, List[float]] = field(default_factory=lambda: defaultdict(list))


def bavg(x: DefaultDict[int, List[float]], scale: float) -> List[float]:
    return [scale * float(np.mean(np.array(x[n]))) for n in sorted(x.keys())]


def bstd(x: DefaultDict[int, List[float]], scale: float) -> List[float]:
    return [scale * float(np.std(np.array(x[n]))) for n in sorted(x.keys())]


def plot_first_second(
    network_sizes: List[int],
    b: BenchmarkResults,
    name: str,
    c1: str,
    c2: str,
    scale: float = 1,
    symbol: str = "o-",
) -> None:
    plt.errorbar(
        network_sizes,
        bavg(b.first_solves, scale=scale),
        fmt=symbol,
        yerr=bstd(b.first_solves, scale=scale),
        label=f"{name} Total Time (1$^{{st}}$ solve)",
        c=c1,
        markersize=4,
    )
    plt.errorbar(
        network_sizes,
        bavg(b.subsequent_solves, scale=scale),
        yerr=bstd(b.subsequent_solves, scale=scale),
        c=c2,
        label=f"{name} Total Time (2$^{{nd}}$ solve)",
        fmt=symbol,
        markersize=4,
    )


def plot_solver_time(
    network_sizes: List[int],
    b: BenchmarkResults,
    name: str,
    c: str,
    scale: float = 1,
    symbol: str = "o-",
) -> None:
    plt.errorbar(
        network_sizes,
        bavg(b.solver_times, scale=scale),
        yerr=bstd(b.solver_times, scale=scale),
        c=c,
        markersize=4,
        fmt=symbol,
        label=f"{name}",
    )


def none_throws(x: Optional[T]) -> T:
    assert x is not None
    return x


def run_benchmark(solver: str, sizes: List[int], repeats: int = TEST_REPEATS) -> BenchmarkResults:
    ret = BenchmarkResults()

    for n in tqdm.tqdm(sizes):
        for _ in range(repeats):
            areas = lambda: draw_random_log_uniform(MINIMUM_AREA, MAXIMUM_AREA)
            concentrations = lambda: draw_random_log_uniform(MINIMUM_CONC, MAXIMUM_CONC)
            network = generate_r_ary_sample_network(
                N=n, branching_factor=BRANCHING_FACTOR, areas=areas
            )
            upstream = conc_list_to_dict(network, concentrations)
            downstream = funmixer.forward_model(
                sample_network=network, upstream_concentrations=upstream
            )
            problem = funmixer.SampleNetworkUnmixer(
                sample_network=network, use_regularization=False
            )
            solution = problem.solve(downstream, solver=solver)
            ret.first_solves[n].append(solution.total_time)
            ret.solver_times[n].append(none_throws(solution.solve_time))
            solution = problem.solve(downstream, solver=solver)
            ret.subsequent_solves[n].append(solution.total_time)

    return ret


def do_benchmark() -> None:
    # Generate list of network sizes
    network_sizes: List[int] = np.unique(
        np.rint(
            np.logspace(np.log10(2), np.log10(MAXIMUM_NETWORK_SIZE), NUMBER_OF_NETWORKS)
        ).astype(int)
    ).tolist()

    # NOTE: Select a slice here for doing dev work on the benchmarks
    network_sizes = network_sizes[:]

    print("#" * 80)
    print(
        f"Running benchmark for {len(network_sizes)} R-ary networks with branching factor {BRANCHING_FACTOR}, up to {MAXIMUM_NETWORK_SIZE} nodes."
    )
    print(
        f"Node concentrations and areas randomly varied between: {MINIMUM_CONC} and {MAXIMUM_CONC}, {MINIMUM_AREA} and {MAXIMUM_AREA}, respectively."
    )
    print("#" * 80)
    start = time.time()

    print("Testing ECOS solver...")
    ecos_bench = run_benchmark("ecos", network_sizes)

    print("Testing GUROBI solver...")
    gurobi_bench: Optional[BenchmarkResults] = None
    try:
        gurobi_bench = run_benchmark("gurobi", network_sizes)
    except Exception as err:
        print(f"Could not benchmark Gurobi. Error: {err}")

    print("Testing SCS solver...")
    scs_bench = run_benchmark("scs", network_sizes)

    with open("benchmark_results.pkl", "wb") as fout:
        pickle.dump([network_sizes, ecos_bench, gurobi_bench, scs_bench], fout)

    end = time.time()
    print(f"Benchmarking took {end - start} seconds.")
    print("Benchmark complete.")
    print("#" * 80)


def plot_benchmark() -> None:
    with open("benchmark_results.pkl", "rb") as fin:
        network_sizes, ecos_bench, gurobi_bench, scs_bench = pickle.load(fin)

    # Plot results of total runtime
    # Plot solve time against number of nodes
    plt.figure(figsize=(10, 5))

    # plt.subplot(2, 1, 1)
    plot_first_second(network_sizes, ecos_bench, "ECOS", "#1f78b4", "#a6cee3")
    plot_first_second(network_sizes, scs_bench, "SCS", "#e31a1c", "#fb9a99")
    if gurobi_bench:
        plot_first_second(network_sizes, gurobi_bench, "GUROBI", "#33a02c", "#b2df8a")

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Total runtime")
    plt.ylabel("Solve time (s)")
    plt.legend()
    plt.grid(True, which="both")
    # plt.subplot(2, 1, 2)

    # Plot results of just the CVXPY solve time
    plot_solver_time(network_sizes, ecos_bench, "ECOS Solver Time", "#1f78b4", symbol="o--")
    plot_solver_time(
        network_sizes, scs_bench, "SCS Solver Time", "#e31a1c", scale=1e-3, symbol="o--"
    )
    if gurobi_bench:
        plot_solver_time(network_sizes, gurobi_bench, "GUROBI Solver Time", "#33a02c", symbol="o--")

    plt.xscale("log")
    plt.title("Optimizer time")
    plt.yscale("log")
    plt.xlabel("Number of nodes")
    plt.ylabel("Solve time (s)")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig("runtime_benchmark.png", dpi=400, bbox_inches="tight")
    plt.show()


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Need to specify `run` or `plot`")
    if sys.argv[1] == "run":
        do_benchmark()
    elif sys.argv[1] == "plot":
        plot_benchmark()
    else:
        raise ValueError("Unknown command: need `run` or `plot`")


if __name__ == "__main__":
    main()
