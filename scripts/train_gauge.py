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
from nsgr.training import GaugeTrainer
from nsgr.utility import load_config


NSGR_PACKAGE_ROOT = Path(_nsgr.__file__).resolve().parent


DEFAULT_CONFIG_PATH = SCRIPT_DIR.parent / "configs" / "bench_csr.json"
RUN_FILE_NAMES = ("train_gauge.py", "simulation_run.py")


USAGE = """Train the neural stochastic gauge using JSON-config parameters only.

Usage:
  python scripts/train_gauge.py
  python scripts/train_gauge.py configs/bench_csr.json

Current training behavior:
  - `training.gauge_mode` must be `neural_graph`, `neural_cnn`, or
    `neural_mlp`
  - training is case-by-case and uses the fixed physics in `lattice`
  - `training.dt` is the one training time step used for every epoch
  - `training.sde_solver` selects `semi_implicit_midpoint` (the backward-
    compatible default), `interaction_picture_implicit_midpoint`, or
    `interaction_picture_exact_local`. `sde_max_iter` controls Picard/Newton;
    root tolerances and Newton damping affect only the implicit interaction-
    picture backend. The exact-local backend has no root and reuses only
    `sde_affine_expm_order/substeps` for its hopping-plus-drive half-flow.
    It composes exact onsite Kerr/loss/detuning and exact frozen prepoint-Itô
    gauge/noise/weight maps, but retains noncommuting splitting error. Omitted
    tolerance defaults follow runtime float32/float64 precision.
  - `training.apply_neural_gauge_every_steps = r` refreshes the neural gauge
    every `r` SDE microsteps inside each window. Omit the option for the old
    behavior where the gauge is computed once at the window start. When set,
    use `1 <= r < N_steps` for every training stage.
  - `training.neural_gauge_state_gradient = "full"` keeps full pathwise
    gradients through phase-space inputs to every neural-gauge application;
    `"each_apply"` treats `(lnOmega, alpha, beta)` as stop-gradient data at
    each gauge application while keeping gradients through neural parameters
  - `training.noise_refresh_every` controls how many epochs reuse the same
    stochastic rollout batch before refreshing the training noise
  - `training.operator_monomials = [[m,n], ...]` is the complete onsite
    monomial basis for the projected residual metric. The trace equation is
    added automatically; do not list `(0,0)` explicitly. Channel order is
    trace first, then the configured pairs in exact configuration order.
  - `training.pareto_k_applied_quantities = P_K` and
    `training.pareto_k_applied_quantities_mode` select self-normalized
    observable monomial clouds for `<(a^\\dagger)^m a^n>`, estimated as
    `Omega/<Omega> * beta_i^m * alpha_i^n`, for the Pareto-k envelope loss.
    Pareto-k supports `P_K<=6`; use `"upto"` for `m+n<=P_K`, or `"exact"`
    for only `m+n=P_K`.
    Set `training.pareto_k_monomials = [[m,n], ...]` to select explicit
    Pareto-k clouds instead, or set `training.pareto_k_monomials = "auto"`
    to select all non-identity onsite moments appearing in the selected
    projected-residual equations.
  - `training.loss_residual_gmm_prefactor` enables the covariance-normalized
    projected residual metric. If `training.operator_monomials` is omitted,
    the basis is `[[0,1], [1,0], [1,1], [2,2]]`. Every endpoint map is the
    selected bare monomial `O_q`; all `U`, `gamma`, `F`, `Delta`, and hopping
    coefficients occur only in the exact RHS `L^dagger O_q`.
    In each window, each channel is its bare-monomial endpoint change minus the
    configured closed Newton-Cotes integral of its direct RHS, divided by the
    window duration. `residual_gmm_integrator_nodes` accepts 3, 4, 5, or 6 and
    defaults to 6. The 3/4-node rules both have formal degree 3, while the
    5/6-node rules both have formal degree 5; one more node is not always one
    more formal order. `N_steps` must be divisible by `nodes-1`. Residuals are
    windowwise; there is no prefix accumulation or propagated influence state.
    At each site the complex channel count is
    `1 + len(operator_monomials)`. Real and imaginary parts are stacked into
    one joint covariance geometry, and the one physical trace equation is
    broadcast across sites.
    The forward mean uses the full normalized contribution cloud. Covariance
    and radii use the exact population covariance of the corresponding
    self-normalized ratio-influence cloud. These within-site population
    covariances are averaged over sites to form one shared window covariance,
    which is floored by `residual_gmm_cov_floor`, standardized and shrunk by
    `residual_gmm_cov_shrinkage`, then whitened with a Cholesky solve. Walker
    gradients are clipped at Mahalanobis radius `residual_gmm_d_clip`; the raw
    monitoring mean is not clipped. The lagged covariance bank has shape
    `(window, real_channel, real_channel)`, is updated for the first 200
    accepted optimizer updates of a stage, and then freezes. This bank is
    independent of the optional generic `training.EMA` normalizer.
    `residual_gmm_time_aggregation` accepts `"mean"`, `"log1p"`, `"entropic"`,
    or `"entropic_log1p"`. For every site in every window, define the joint
    covariance-normalized score `q = mu^T P mu`. `"mean"` averages `q`, while
    `"log1p"` transforms each individual site/window score to `log1p(q)` before
    averaging over sites and windows; it is not applied after either average.
    The entropic modes use risk parameter `residual_gmm_time_beta`; the beta is
    inactive for `"mean"` and `"log1p"`. `"mean"` and `2.0` are the respective
    aggregation and beta defaults.
    The metric has one aggregate prefactor and no per-channel prefactors; use
    `loss_residual_gmm` in `training.EMA.terms` for aggregate
    loss-scale normalization.
  - training plots show the projected residual objective and channels, Pareto-k
    diagnostics, gauge penalty, L2 penalty, gradient norm, and learning rate.
    The printed `loss=` value is the actual optimized objective after
    prefactors and any EMA normalization.
  - the observable Pareto-k cloud is always onsite per-site.
  - `training.loss_pareto_k_prefactor` enables the Pareto-k envelope loss on
    self-normalized observable contribution clouds selected by
    `training.pareto_k_applied_quantities` and
    `training.pareto_k_applied_quantities_mode`;
    `pareto_k_threshold` is the target
    tail index, `pareto_k_threshold_tau` is additive log-radius envelope slack,
    `pareto_k_envelope_excess` selects `"log"` violations
    `[log(r/r_allowed)]_+` or `"ratio"` violations `[r/r_allowed - 1]_+`,
    and `pareto_k_envelope_beta` controls entropic focus over the envelope
    violations in the selected top tail.
    EMA normalization may be applied term-wise by listing monomial terms such
    as `loss_pareto_k_m0_n4`, ..., `loss_pareto_k_m4_n0` in
    `training.EMA.terms`; these share the single `loss_pareto_k_prefactor`
    and are normalized site-wise as `(m,n,i)` terms. Use `loss_pareto_k_p4`
    as shorthand for all Pareto-k monomials with `m+n=4`.
    Use `loss_pareto_k_auto` as shorthand for the auto-expanded Pareto-k
    monomial set.
  - `training.q_winsor > 0` controls quantile-winsorized bulk statistics for
    Pareto-k only; it does not enter the projected residual covariance.
  - the supported training-loss families are the covariance-normalized,
    window-integrated projected residual, Pareto-k, gauge-output
    regularization, and L2 parameter regularization. Each is controlled by its
    own prefactor.
  - `training.EMA` optionally applies slow EMA magnitude normalization to
    selected raw losses, using `loss_prefactor * raw_loss / EMA[raw_loss]`;
    the raw losses and EMA scales are still logged separately, and
    the optional `training.EMA.r_max` default prevents one finite spike from
    poisoning the EMA scale estimate
  - `training.multi_device.enabled = true` shards walkers over local JAX
    devices with `pmap`; per-device gradients are averaged before the normal
    optimizer update, and checkpoint/history IO stays unchanged
  - `training.multi_device.num_devices = "auto"` uses all visible local
    devices; otherwise set it to an integer, and ensure the active
    `num_walker` is divisible by that device count
  - `training.load_parameters = true` is a parameter-only warm start. The
    checkpoint is deserialized before `io.clean_start` removes old outputs,
    but optimizer, loss-EMA, residual-covariance, PRNG, and history state are
    initialized anew
  - multi-device training is most useful when each visible device has enough
    walkers to stay compute-bound; as a practical starting point use at least
    about 4000 walkers per device
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
  - `neural_mlp` uses the site-flattened MLP gauge network
  - startup prints the active meaning of the shared `model` fields for the selected neural model,
    including which provided fields are ignored by that model
  - `neural_mlp` keeps its configurable time-feature path from the shared `model` config
  - the default runtime precision is float64; set `NSGR_USE_FLOAT64=0` and
    `NSGR_NN_USE_FLOAT64=0` before launch to use float32 throughout

For staged training:
  if `training.staged_schedule` is present, each stage supplies
  `stage_id`, `n_epoch`, `N_windows`, `num_walker`, and `N_steps`.
  In that case, those four control parameters do not need to be repeated
  in the outer `training` block.
  A stage with `stage_id > 1` loads model parameters. The first listed stage
  uses `training.params_path` when supplied, otherwise the canonical
  `io.save_dir/train/model_params.msgpack`; later stages use the preceding
  canonical checkpoint.

For segmented-overlap training:
  set `training.segmented_overlap.enabled = true` and provide
  `training.segmented_overlap.stage_schedule`. Each stage supplies
  `stage_id`, `n_epoch`, `n_segments`, `n_windows_per_segment`,
  `num_walker`, and `N_steps`; the trainer builds a frozen rollout bank of
  `n_windows_per_segment + (n_segments - 1) * stride` windows, where
  `stride = n_windows_per_segment - segment_overlap_windows`, slices each
  segment from that bank, inverse-weights overlapping windows, averages valid
  segment gradients, and checkpoints at stage end. The same `stage_id > 1`
  parameter-loading rule applies.

Expected folder layout on a cluster:
  Kerr/
    train_gauge.py
    Kerr.json
    simulation_run.py
    nsgr/  # optional when the neural-stochastic-gauge package is installed
"""


def _find_nsgr_dir() -> Path:
    if NSGR_PACKAGE_ROOT.is_dir():
        return NSGR_PACKAGE_ROOT
    raise FileNotFoundError("Could not locate the installed or local nsgr package.")


def _snapshot_run_files(config_path: Path, save_dir: str | Path):
    snapshot_dir = Path(save_dir).expanduser().resolve() / "provenance" / "training"
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


def train_gauge(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path).expanduser().resolve()
    config = load_config(str(config_path))
    try:
        return GaugeTrainer(config, config_path=str(config_path)).fit()
    finally:
        # Training may delete and recreate io.save_dir when clean_start=true,
        # so provenance must be written after that cleanup has taken place.
        _snapshot_run_files(config_path, config["io"]["save_dir"])


def main():
    if len(sys.argv) == 2 and sys.argv[1] in {"-h", "--help"}:
        print(USAGE, flush=True)
        return
    if len(sys.argv) > 2:
        raise SystemExit(USAGE)

    config_path = DEFAULT_CONFIG_PATH if len(sys.argv) == 1 else Path(sys.argv[1])
    artifacts = train_gauge(config_path)

    if isinstance(artifacts, list):
        final_artifact = artifacts[-1]
    else:
        final_artifact = artifacts
    print(f"final params saved to: {final_artifact.params_path}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["train_gauge"]
