# Neural Gauge-P Representation for Interacting Open Bosons

This repository contains the JAX/Flax implementation of the neural gauge-P
method introduced in:

> Xiaodong Cao and Zhicheng Zhong, **“Neural Gauge-P Representation for Open
> Quantum Dynamics of Interacting Bosons,”**
> [arXiv:2607.17534](https://arxiv.org/abs/2607.17534) (2026).

The code trains neural drift and diffusion gauges for positive-P/gauge-P
stochastic simulations of real-time interacting bosonic systems. The primary
training signal is a covariance-normalized residual of exact moment equations,
with optional Pareto-tail, gauge-output, parameter-norm, and weight-entropy
regularization.

This is a **code-only release**. The trained checkpoints and numerical data
used for the figures in the paper are not included.

## Implemented scope

The maintained implementation supports:

- driven-dissipative Bose-Hubbard/Kerr lattices;
- ungauged positive-P, an analytical gauge-P baseline, and learned neural
  gauge-P dynamics;
- neural graph, convolutional, and multilayer-perceptron gauge models;
- learned drift gauges, diffusion gauges, or both;
- covariance-normalized, window-integrated projected-residual training;
- staged and overlapping-segment training;
- single- and multi-device JAX execution;
- NPZ output, optional Zarr output, observable error estimates, and optional
  exact-diagonalization benchmarks for sufficiently small systems.

Fermionic representations, imaginary-time evolution, and complex-time
contours are not part of this release.

## Repository layout

```text
neural_gauge_boson/
├── configs/
│   └── bench_csr.json          # production calculation configuration
├── docs/
│   └── technical_reference.md  # equations and implementation details
├── scripts/
│   ├── train_gauge.py          # training entry point
│   └── simulation_run.py       # simulation entry point
├── src/nsgr/                   # installable Python package
├── pyproject.toml
└── README.md
```

Runtime outputs are created at the path selected by `io.save_dir` in the JSON
configuration.

## Requirements

- Python 3.10 or 3.11
- JAX/JAXlib 0.4.20
- Flax 0.7.5
- Optax 0.1.7
- NumPy 1.26.4
- SciPy 1.11.3
- Matplotlib 3.8.1

The exact Python dependencies are pinned in `pyproject.toml`. The supplied
production calculation requires CUDA-capable accelerators and substantial
device memory. CPU execution is intended only for installation checks and
very small tests.

## Installation

Create an isolated Python 3.10 or 3.11 environment from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The command above installs the pinned default JAX build. On an NVIDIA system,
install the JAX 0.4.20 build appropriate for the cluster's CUDA/cuDNN stack
before installing this package. See the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Verify the installation and inspect the devices visible to JAX:

```bash
python -c "import nsgr; print(nsgr.__version__)"
python -c "import jax; print(jax.devices())"
```

The executable scripts provide built-in configuration summaries:

```bash
python scripts/train_gauge.py --help
python scripts/simulation_run.py --help
```

## Supplied production configuration

[`configs/bench_csr.json`](configs/bench_csr.json) is the configuration of an
actual calculation. It is not a toy example. Its main workload settings are:

- a single-site Kerr problem with `U=1`, `gamma=0.3`, and initial occupation
  `n0=1`;
- neural graph drift and diffusion gauges with U(1)-invariant features;
- the `interaction_picture_exact_local` SDE solver;
- covariance-normalized residual channels `(0,1)`, `(1,0)`, `(1,1)`, and
  `(2,2)`;
- one overlapping-segment training stage with 100001 epochs and 10000
  walkers;
- a production simulation with 10000000 walkers, split into walker batches;
- simulation step size `dt=5e-4`, 20 steps per window, and 1500 windows.

Running this file unchanged can require a long GPU calculation. To test a new
installation, first copy it:

```bash
cp configs/bench_csr.json configs/my_run.json
```

Then reduce the copied configuration. For a smoke test:

- set both `training.multi_device.enabled` and
  `simulation.multi_device.enabled` to `false`;
- reduce `n_epoch`, `num_walker`, and `n_windows_per_segment` in
  `training.segmented_overlap.stage_schedule[0]`;
- reduce `simulation.num_walker` and `simulation.N_windows`;
- disable `simulation.walker_batches` and `simulation.ed`;
- keep each training-stage `N_steps` divisible by
  `training.residual_gmm_integrator_nodes - 1`;
- if `training.apply_neural_gauge_every_steps` is present, keep it smaller
  than every stage's `N_steps`.

Reduced settings only test whether the software runs; they are not
scientifically converged parameters.

## Output-path convention

Paths in a configuration are resolved relative to the JSON file itself. For
example, in `configs/my_run.json`,

```json
"save_dir": "../runs/my_run"
```

writes to `runs/my_run/` at the repository root. In the supplied configuration,
`"save_dir": "./output"` resolves to `configs/output/`.

## Train a neural gauge

Run from the repository root:

```bash
python scripts/train_gauge.py configs/bench_csr.json
```

To run a modified copy instead:

```bash
python scripts/train_gauge.py configs/my_run.json
```

If no JSON path is supplied, the script uses `configs/bench_csr.json`.
Training writes the canonical checkpoint to:

```text
<save_dir>/train/model_params.msgpack
```

It also records the resolved configuration, histories, metadata, diagnostic
plots, and a provenance snapshot of the imported `nsgr` source.

## Run a simulation

After training, use the same configuration:

```bash
python scripts/simulation_run.py configs/bench_csr.json
```

For a neural gauge, the simulator loads
`<save_dir>/train/model_params.msgpack` by default. To use a checkpoint from
another run, set `simulation.params_path`. Relative checkpoint paths are also
resolved relative to the JSON file.

To run an ungauged positive-P baseline without a neural checkpoint, set:

```json
"simulation": {
  "gauge_mode": "zero_gauge"
}
```

To use the implemented adaptive analytical gauge-P baseline, set
`simulation.gauge_mode` to `"wuster_adaptive"`.

## Main configuration sections

The JSON configuration has six top-level sections:

| Section | Purpose |
|---|---|
| `io` | Output directory and clean-start behavior |
| `lattice` | Geometry, hopping, interaction, drive, loss, detuning, and initial occupation |
| `training` | Gauge model, rollout, residual loss, tail penalties, staging, and checkpoints |
| `simulation` | Production rollout, batching, observables, storage, and ED benchmark |
| `optimizer` | Optimizer, learning-rate schedule, and gradient clipping |
| `model` | Neural architecture, time features, U(1) invariance, and output bounds |

Unknown configuration keys are rejected rather than silently ignored. See
[`docs/technical_reference.md`](docs/technical_reference.md) for the equations,
valid solver controls, projected-residual construction, Pareto-k objective,
segmented training, and saved-data conventions.

## SDE solver selection

Set `training.sde_solver` and `simulation.sde_solver` to one of:

- `"semi_implicit_midpoint"` — fixed-iteration Picard midpoint solver;
- `"interaction_picture_implicit_midpoint"` — interaction-picture midpoint
  solver with a damped nonlinear root solve;
- `"interaction_picture_exact_local"` — root-free split solver with exact
  local Kerr/loss/detuning and frozen prepoint-Itô gauge subflows.

If `simulation.sde_solver` is omitted, it inherits the training solver.
Solver controls are backend-specific; the program prints active and inactive
controls at startup. For a new physical regime, verify timestep convergence
and, for multisite interaction-picture runs, affine-action convergence.

## Numerical precision

Precision must be selected before Python starts. The default is
float64/complex128 for the SDE and float64 for the neural network.

Full float64:

```bash
export NSGR_USE_FLOAT64=1
export NSGR_NN_USE_FLOAT64=1
```

Float64/complex128 SDE with a float32 neural network:

```bash
export NSGR_USE_FLOAT64=1
export NSGR_NN_USE_FLOAT64=0
```

Full float32/complex64:

```bash
export NSGR_USE_FLOAT64=0
export NSGR_NN_USE_FLOAT64=0
```

The startup log reports the active training-data and neural-network dtypes.
Checkpoint parameter dtypes must be compatible with the selected neural
precision.

## Multi-device execution

Walker sharding is enabled independently for training and simulation:

```json
"multi_device": {
  "enabled": true,
  "num_devices": "auto"
}
```

`"auto"` uses all visible local JAX devices. The active walker count must be
divisible by the selected device count. For small tests, disable multi-device
execution to avoid unnecessary compilation and sharding overhead.

## Output files

A typical neural-gauge run produces:

```text
<save_dir>/
├── train/
│   ├── model_params.msgpack
│   ├── config_used.json
│   ├── training_history.npz
│   ├── training_history.json
│   ├── training_metadata.json
│   ├── training_history.pdf
│   └── training_history.png
├── simulation/
│   ├── config_used_neural_graph.json
│   ├── simulation_metadata_neural_graph.json
│   ├── simulation_observables_neural_graph.npz
│   ├── simulation_results_neural_graph.npz  # when raw walkers are saved
│   └── ed_benchmark.npz                     # when ED is enabled
└── provenance/
    ├── training/
    └── simulation/
```

Stage-specific training files are additionally written by staged or
overlapping-segment training. The exact output set depends on the storage and
plotting options in the configuration.

## Citation

If this code contributes to published work, please cite:

```bibtex
@article{cao2026neuralgaugep,
  title   = {Neural Gauge-P Representation for Open Quantum Dynamics of Interacting Bosons},
  author  = {Cao, Xiaodong and Zhong, Zhicheng},
  journal = {arXiv preprint arXiv:2607.17534},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.17534}
}
```

Paper: <https://arxiv.org/abs/2607.17534>
