from __future__ import annotations

import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

for candidate in (SCRIPT_DIR, SCRIPT_DIR / "src", SCRIPT_DIR.parent / "src"):
    if (candidate / "nsgr").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

import nsgr as _nsgr
from nsgr.simulation import GaugeSimulator
from nsgr.utility import load_config


NSGR_PACKAGE_ROOT = Path(_nsgr.__file__).resolve().parent


DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "bench_csr.json"
RUN_FILE_NAMES = ("train_gauge.py", "simulation_run.py")


USAGE = """Run the stochastic-gauge simulation and save requested window-end data.

Usage:
  python scripts/simulation_run.py
  python scripts/simulation_run.py configs/bench_csr.json

Current simulation behavior:
  - all gauge modes use the per-window rollout with window progress prints
  - `simulation.sde_solver` selects `semi_implicit_midpoint`,
    `interaction_picture_implicit_midpoint`, or
    `interaction_picture_exact_local`. When omitted it inherits
    `training.sde_solver`, whose backward-compatible default is
    `semi_implicit_midpoint`. `sde_max_iter` controls Picard/Newton; root
    tolerances and damping affect only Newton. The root-free exact-local
    backend reuses only `sde_affine_expm_order/substeps`. Omitted controls
    inherit their validated training values except `simulation.sde_max_iter`,
    whose existing omitted default remains 4.
  - `simulation.apply_neural_gauge_every_steps = r` refreshes the neural gauge
    every `r` SDE microsteps inside each simulation window. Omit the option for
    the old window-start-only behavior. When set, use `1 <= r < simulation.N_steps`.
  - simulation uses the same physical parameters as the `lattice` section
  - staged or segmented training configs do not need an outer
    `training.n_epoch`; the loader can infer the schedule length from
    `training.staged_schedule` or `training.segmented_overlap.stage_schedule`
  - `neural_graph` uses a lightweight sparse real edge-based message passing trunk from the real hopping matrix
  - `neural_graph` uses local phase-space and positive-P drift features together with walker-local `lnOmega`
  - `neural_cnn` uses lightweight 3x3 lattice-stencil blocks on local
    `alpha` and `beta` fields, with walker-local `lnOmega` and time
    entering through global conditioning
  - `model.time_feature = "Raw"` concatenates scalar time directly in the graph/CNN global feature
  - `model.time_feature = "Dense"` applies a learned linear map from scalar time to `time_embed_dim`
  - `model.time_feature = "DampedTrig"` appends learned damped sinusoids
    `exp(-t / tau_i) * cos(w_i t)` and `exp(-t / tau_i) * sin(w_i t)`
  - `model.time_feature = "DampedTrigTrend"` also appends the matching
    `(1 - exp(-t / tau_i)) * cos(w_i t)` and `(1 - exp(-t / tau_i)) * sin(w_i t)`
    channels, so the time basis can represent both decaying and growing envelopes
  - `model.bound_diffusion = true` independently maps the diffusion gauge to
    `diffusion_max * tanh(lambda_raw / diffusion_max)`, preventing the
    `cosh(lambda)` and `sinh(lambda)` noise factors from growing without bound
  - `model.bound_drift = true` independently maps each drift gauge to
    `drift_max * tanh(G_raw / drift_max)`; it does not affect the diffusion gauge
  - `neural_mlp` keeps its configurable time-feature path from the shared `model` config
  - startup prints the active meaning of the shared `model` fields for the selected neural model,
    including which provided fields are ignored by that model
  - `training.neural_gauge_state_gradient = "full"` keeps full pathwise
    gradients through neural-gauge phase-space inputs; `"each_apply"` treats
    `(lnOmega, alpha, beta)` as stop-gradient data at each training gauge
    application
  - the publication training metric is enabled by
    `training.loss_residual_gmm_prefactor`.
    `training.operator_monomials` is the complete onsite residual basis. Every
    left-hand side is one configured bare monomial and every physical
    coefficient occurs only in its exact `L^dagger` right-hand side. At each
    site the raw diagnostic order is trace followed by the configured
    monomials in exact configuration order. No residual is accumulated from
    the rollout start.
    `training.residual_gmm_integrator_nodes` selects 3, 4, 5, or 6 closed
    Newton-Cotes nodes (default 6). `training.N_steps` must be divisible by
    `nodes-1`; 3/4 nodes both have formal degree 3 and 5/6 nodes both have
    formal degree 5.
    With the default `training.residual_gmm_trace_mode="diagnostic"`, a
    population covariance over the configured onsite real and imaginary
    channels is estimated within each site from the self-normalized
    ratio-influence cloud, then averaged over sites to form one shared window
    covariance. The broadcast physical trace equation remains a raw monitor
    but is excluded from that geometry and objective. Set the mode to
    `"joint"` for the legacy trace-inclusive covariance. A lagged covariance
    bank is calibrated during the first 200 accepted optimizer updates of each
    stage and then frozen. Covariance flooring, correlation shrinkage,
    Cholesky whitening, and walkerwise
    Mahalanobis gradient clipping affect training only; they do not change the
    saved simulation rollout
  - `training.N_steps` must be divisible by
    `residual_gmm_integrator_nodes - 1` so that every residual
    window contains the requested quadrature nodes
  - `training.pareto_k_applied_quantities` selects the
    self-normalized onsite operator-monomial clouds for observable Pareto-k:
    `Omega/<Omega> * beta_i^m * alpha_i^n`, corresponding to
    `<(a^\\dagger)^m a^n>`.
    `training.pareto_k_monomials` may instead specify a free list
    `[[m,n], ...]`; `"auto"` selects all non-identity onsite moments appearing
    in the configured projected-residual equations.
  - the supported training-loss families are the covariance-normalized
    windowwise projected residual, Pareto-k, gauge-output
    regularization, and L2 parameter regularization
  - all training objectives and their covariance/EMA histories are
    training-only; simulation loads the learned gauge parameters but does not
    reconstruct or update optimizer-side metric state
  - raw trajectory data are saved only when `simulation.save_raw_walkers = true`
    as `simulation_results_<gauge_mode>.<format>`
  - raw log-weights use the stable split representation
    `lnOmega_absolute[t,w] = lnOmega_history[t,w] + lnOmega_shift_history[t]`;
    this convention is identical for single-batch and chunked simulations
  - `simulation.save_raw_walkers_every_windows` thins raw walker snapshots
    without changing compact observable measurement times by itself; when
    `pareto-k_mean` or `pareto-k_max` is requested, it also controls the
    separate expensive Pareto-k diagnostic grid saved as `pareto-k_times`
  - `simulation.save_observables_every_windows` thins compact observable
    snapshots without changing the SDE window grid
  - `pareto-k_mean` and `pareto-k_max`, when listed in
    `simulation.observables`, diagnose self-normalized observable clouds using
    `training.pareto_k_applied_quantities` and
    `training.pareto_k_applied_quantities_mode` (`P_K<=6`), or
    `training.pareto_k_monomials` when present, including `"auto"` after it
    expands from the operator equations; this selector is
    diagnostic-only for simulation and also saves each selected
    `pareto-k_m*_n*` site-vector plus its `_mean` and `_max`
  - `simulation.walker_batches.enabled = true` can split a large total
    `simulation.num_walker` ensemble into smaller sequential batches; compact
    observables are merged from global weighted numerator/denominator sums
    rather than averaging per-batch ratios
  - compact site-resolved observables are saved when
    `simulation.save_observables = true` as
    `simulation_observables_<gauge_mode>.<format>`
  - `simulation.multi_device.enabled = true` shards walkers over local JAX
    devices with `pmap`; saved outputs keep the same keys and shapes as
    single-device simulation
  - `simulation.multi_device.num_devices = "auto"` uses all visible local
    devices; otherwise set it to an integer, and ensure `num_walker` is
    divisible by that device count
  - multi-device simulation is most useful when each visible device has enough
    walkers to stay compute-bound; as a practical starting point use at least
    about 4000 walkers per device
  - when `simulation.ed.enabled = true`, the ED benchmark uses the same `dt`
    but a denser window-end grid with `N_steps / 10` and `N_windows * 10`
  - the default runtime precision is float64; set `NSGR_USE_FLOAT64=0` and
    `NSGR_NN_USE_FLOAT64=0` before launch to use float32 throughout

Expected folder layout on a cluster:
  Kerr/
    train_gauge.py
    simulation_run.py
    Kerr.json
    nsgr/  # optional when the neural-stochastic-gauge package is installed
"""


def _find_nsgr_dir() -> Path:
    if NSGR_PACKAGE_ROOT.is_dir():
        return NSGR_PACKAGE_ROOT
    raise FileNotFoundError("Could not locate the installed or local nsgr package.")


def _snapshot_run_files(config_path: Path, save_dir: str | Path):
    snapshot_dir = Path(save_dir).expanduser().resolve() / "provenance" / "simulation"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    nsgr_dir = _find_nsgr_dir()
    nsgr_dest = snapshot_dir / "nsgr"
    if nsgr_dest.resolve() != nsgr_dir.resolve():
        if nsgr_dest.exists():
            shutil.rmtree(nsgr_dest)
        shutil.copytree(
            nsgr_dir,
            nsgr_dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"),
        )

    files_to_copy = []
    for file_name in RUN_FILE_NAMES:
        path = SCRIPT_DIR / file_name
        if path.is_file():
            files_to_copy.append(path)
    if config_path.is_file() and config_path.suffix.lower() == ".json":
        files_to_copy.append(config_path)

    copied = set()
    for path in files_to_copy:
        src = path.resolve()
        destination = (snapshot_dir / path.name).resolve()
        if src in copied or src == destination:
            continue
        copied.add(src)
        shutil.copy2(src, destination)

    print(f"run files copied to: {snapshot_dir}", flush=True)


def run_simulation(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(str(config_path))
    try:
        simulator = GaugeSimulator(config, config_path=str(config_path))
        return simulator.run(), config
    finally:
        _snapshot_run_files(config_path, config["io"]["save_dir"])


def main():
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        print(USAGE, flush=True)
        return
    if len(sys.argv) > 2:
        raise SystemExit(USAGE)

    config_path = DEFAULT_CONFIG_PATH if len(sys.argv) == 1 else Path(sys.argv[1])
    artifacts, _ = run_simulation(config_path)

    if artifacts.results_path is not None:
        print(f"saved raw simulation data to: {artifacts.results_path}", flush=True)
    if artifacts.observables_path is not None:
        print(f"saved simulation observables to: {artifacts.observables_path}", flush=True)
    print(f"saved simulation metadata to: {artifacts.metadata_path}", flush=True)
    if artifacts.benchmark_path is not None:
        print(f"saved ED benchmark to: {artifacts.benchmark_path}", flush=True)
    if artifacts.results:
        print(f"saved history shapes = {artifacts.results['history_shapes']}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "run_simulation",
]
