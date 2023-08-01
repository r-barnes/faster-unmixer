from random_networks_test import (
    generate_r_ary_sample_network,
    draw_random_log_uniform,
    conc_list_to_dict,
)
import funmixer
import matplotlib.pyplot as plt
import numpy as np
import time

# Set the range to explore upstream concentration values over
minimum_conc, maximum_conc = 1, 1e2
# Set the range to explore sub-basin area values over
minimum_area, maximum_area = 1, 1e2
# Set branching factor
branching_factor = 3
# Set maximum number of nodes to try
maximum_network_size = 500
# Set number of networks to test
number_of_networks = 100

# 100 networks up to 500 nodes in size takes about 12 minutes to run on my machine

# Generate list of network sizes
network_sizes = np.unique(
    np.rint(np.logspace(np.log10(2), np.log10(maximum_network_size), number_of_networks)).astype(
        int
    )
)

print("#" * 80)
print(
    f"Running benchmark for {network_sizes.size} R-ary networks with branching factor {branching_factor}, up to {maximum_network_size} nodes."
)
print(
    f"Node concentrations and areas randomly varied between: {minimum_conc} and {maximum_conc}, {minimum_area} and {maximum_area}, respectively."
)
print("#" * 80)
start = time.time()
# Test ECOS solver
print("Testing ECOS solver...")
ecos_first_solves = []
ecos_subsequent_solves = []
ecos_cvxpy_solve = []

for N in network_sizes:
    areas = lambda: draw_random_log_uniform(minimum_area, maximum_area)
    concentrations = lambda: draw_random_log_uniform(minimum_conc, maximum_conc)
    network = generate_r_ary_sample_network(N=N, branching_factor=branching_factor, areas=areas)
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    solution = problem.solve(downstream, solver="ecos")
    ecos_first_solves.append(solution.total_time)
    ecos_cvxpy_solve.append(solution.solve_time)
    solution = problem.solve(downstream, solver="ecos")
    ecos_subsequent_solves.append(solution.total_time)


print("Testing GUROBI solver...")
gurobi_first_solves = []
gurobi_subsequent_solves = []
gurobi_cvxpy_solve = []


for N in network_sizes:
    areas = lambda: draw_random_log_uniform(minimum_area, maximum_area)
    concentrations = lambda: draw_random_log_uniform(minimum_conc, maximum_conc)
    network = generate_r_ary_sample_network(N=N, branching_factor=branching_factor, areas=areas)
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    solution = problem.solve(downstream, solver="gurobi")
    gurobi_first_solves.append(solution.total_time)
    gurobi_cvxpy_solve.append(solution.solve_time)
    solution = problem.solve(downstream, solver="gurobi")
    gurobi_subsequent_solves.append(solution.total_time)

print("Testing SCS solver...")
scs_first_solves = []
scs_subsequent_solves = []
scs_cvxpy_solve = []

for N in network_sizes:
    areas = lambda: draw_random_log_uniform(minimum_area, maximum_area)
    concentrations = lambda: draw_random_log_uniform(minimum_conc, maximum_conc)
    network = generate_r_ary_sample_network(N=N, branching_factor=branching_factor, areas=areas)
    upstream = conc_list_to_dict(network, concentrations)
    downstream = funmixer.forward_model(sample_network=network, upstream_concentrations=upstream)
    problem = funmixer.SampleNetworkUnmixer(sample_network=network, use_regularization=False)
    solution = problem.solve(downstream, solver="scs")
    scs_first_solves.append(solution.total_time)
    scs_cvxpy_solve.append(solution.solve_time * 1e-3)  # Convert ms to s
    solution = problem.solve(downstream, solver="scs")
    scs_subsequent_solves.append(solution.total_time)


end = time.time()
print(f"Benchmarking took {end - start} seconds.")
print("Benchmark complete.")
print("#" * 80)
# Plot results of total runtime
# Plot solve time against number of nodes
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(
    network_sizes,
    ecos_first_solves,
    "o-",
    c="#1f78b4",
    label="ECOS (1$^{st}$ solve)",
    markersize=4,
)
plt.plot(
    network_sizes,
    ecos_subsequent_solves,
    "o-",
    c="#a6cee3",
    # Make points smaller
    markersize=4,
    label="ECOS (2$^{nd}$ solve)",
)
plt.plot(
    network_sizes,
    gurobi_first_solves,
    "o-",
    c="#33a02c",
    label="GUROBI (1$^{st}$ solve)",
    markersize=4,
)
plt.plot(
    network_sizes,
    gurobi_subsequent_solves,
    "o-",
    c="#b2df8a",
    label="GUROBI (2$^{nd}$ solve)",
    markersize=4,
)
plt.plot(
    network_sizes,
    scs_first_solves,
    "o-",
    c="#e31a1c",
    label="SCS (1$^{st}$ solve)",
    markersize=4,
)
plt.plot(
    network_sizes,
    scs_subsequent_solves,
    "o-",
    c="#fb9a99",
    label="SCS (2$^{nd}$ solve)",
    markersize=4,
)
plt.xscale("log")
plt.yscale("log")
plt.title("Total runtime")
plt.ylabel("Solve time (s)")
plt.legend()
plt.grid(True, which="both")
plt.subplot(2, 1, 2)

# Plot results of just the CVXPY solve time
plt.plot(
    network_sizes,
    ecos_cvxpy_solve,
    "o-",
    c="#1f78b4",
    label="ECOS",
    markersize=4,
)
plt.plot(
    network_sizes,
    gurobi_cvxpy_solve,
    "o-",
    c="#33a02c",
    markersize=4,
    label="GUROBI",
)
plt.plot(
    network_sizes,
    scs_cvxpy_solve,
    "o-",
    c="#e31a1c",
    label="SCS",
    markersize=4,
)
plt.xscale("log")
plt.title("CVXPY solve time")
plt.yscale("log")
plt.xlabel("Number of nodes")
plt.ylabel("Solve time (s)")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("runtime_benchmark.png", dpi=400, bbox_inches="tight")
plt.show()
