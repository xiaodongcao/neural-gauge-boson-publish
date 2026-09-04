"""Example for loading the G1_initial results saved by bench_csr.json."""

import numpy as np


simulation_dir = "your_simulation_dir"

ed_data = np.load(f"{simulation_dir}/output/simulation/ed_benchmark.npz")
neural_data = np.load(
    f"{simulation_dir}/output/simulation/simulation_observables_neural_graph.npz"
)

ed_times = ed_data["times"]
ed_g1_initial = ed_data["G1_initial"][:, 0, 0]

neural_times = neural_data["times"]
neural_g1_initial = neural_data["G1_initial"][:, 0, 0]

ed_g1_at_neural_times = np.interp(neural_times, ed_times, ed_g1_initial)

print("time, ED G1_initial, neural-gauge G1_initial")
for time, ed_g1, neural_g1 in zip(
    neural_times, ed_g1_at_neural_times, neural_g1_initial
):
    print(time, ed_g1, neural_g1)
