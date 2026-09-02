from dataclasses import dataclass

from .analytical_gauge import validate_analytical_gauge_mode
from .dynamics_kernel import (
    compute_observable_pareto_k_indices,
    normalize_applied_quantities,
    normalize_applied_quantities_mode,
    run_one_window_simulation_rollout,
    var_complex_to_real,
)
from .lattice import Lattice
from .lib_preinclude import *
from .model import Model
from .multi_device import (
    MultiDeviceSimulationRollout,
    MultiDeviceSimulationStepper,
    resolve_multi_device_spec,
    shard_walkers,
    split_device_keys,
    unshard_walkers,
    unshard_walkers_device,
)
from .postprocess import (
    compute_equal_time_measurements,
    compute_initial_time_measurements,
    compute_local_equal_time_measurements,
    compute_local_equal_time_measurements_no_error,
    compute_local_initial_time_measurements,
    compute_local_initial_time_measurements_no_error,
    compute_reduced_equal_time_measurements,
    compute_reduced_initial_time_measurements,
    shell_pair_average_samples,
)
from .utility import (
    NEURAL_GAUGE_MODES,
    OPERATOR_MOMENT_DEFAULT_ORDER,
    OPERATOR_MOMENT_MAX_ORDER,
    OPERATOR_MOMENT_SPECS,
    PARETO_K_OBSERVABLE_NAMES,
    PARETO_K_OBSERVABLE_SPECS,
    KeyGenerator,
    archive_extension,
    count_params,
    format_end_time,
    format_sde_solver_controls,
    is_auto_monomial_selector,
    load_config,
    normalize_monomial_pairs,
    normalize_neural_gauge_components,
    normalize_sde_solver,
    onsite_monomials_in_operator_equations,
    prepare_arrays_for_storage,
    require_archive_backend,
    resolve_checkpoint_params_path,
    save_array_archive,
    save_json,
    sde_solver_control_metadata,
    selected_monomial_specs,
)


@dataclass(frozen=True)
class SimulationPhysics:
    n0: float
    U: float
    gamma: float
    F: complex
    Delta: float


@dataclass(frozen=True)
class EDBenchmarkConfig:
    enabled: bool
    n_cut: int


@dataclass(frozen=True)
class ArchiveSavePolicy:
    archive_format: str
    compressed: bool
    precision: str


@dataclass(frozen=True)
class RemoveUnhealthConfig:
    enabled: bool = False
    logstd: float = 15.0


@dataclass
class SimulationArtifacts:
    results: Dict[str, Any]
    results_path: Optional[str]
    observables_path: Optional[str]
    metadata_path: str
    benchmark_path: Optional[str]


INITIAL_TIME_OBSERVABLES = {"G1_initial", "G2_initial", "g2_initial"}
REDUCED_EQUAL_TIME_OBSERVABLES = {
    "G1_local",
    "G1_nn",
    "G1_nnn",
    "G2_local",
    "g2_local",
    "g2_nn",
    "g2_nnn",
}
REDUCED_INITIAL_TIME_OBSERVABLES = {
    "G1_initial_local",
    "G1_initial_nn",
    "G1_initial_nnn",
}
LOCAL_EQUAL_TIME_OBSERVABLES = {"density", "N", "G1_local", "G2_local", "g2_local"}
LOCAL_INITIAL_TIME_OBSERVABLES = {"G1_initial_local"}
PARETO_K_AGGREGATE_OBSERVABLES = {"pareto-k_mean", "pareto-k_max"}
PARETO_K_OBSERVABLES = {*PARETO_K_AGGREGATE_OBSERVABLES, *PARETO_K_OBSERVABLE_NAMES}
DEFAULT_SIMULATION_OBSERVABLES = ("G1", "G1_initial", "density", "g2_local")
SUPPORTED_SIMULATION_OBSERVABLES = {
    "A",
    "B",
    "density",
    "N",
    "G1",
    "G2",
    "g2",
    "G2_local",
    "g2_local",
    "coherence_fraction",
    *INITIAL_TIME_OBSERVABLES,
    *REDUCED_EQUAL_TIME_OBSERVABLES,
    *REDUCED_INITIAL_TIME_OBSERVABLES,
    *PARETO_K_OBSERVABLES,
}
SIMULATION_OBSERVABLE_ALIASES = {
    "g2_iinital": "g2_initial",
    "G2_iinital": "G2_initial",
    "pareto_k_mean": "pareto-k_mean",
    "pareto_k_max": "pareto-k_max",
    **{name.replace("pareto-k", "pareto_k", 1): name for name in PARETO_K_OBSERVABLE_NAMES},
}


def _real_jackknife_error(
    jackknife_values: np.ndarray,
    *,
    full_value=None,
    group_fractions=None,
) -> np.ndarray:
    values = np.asarray(jackknife_values, dtype=float)
    if values.shape[0] <= 1:
        return np.zeros(values.shape[1:], dtype=float)
    if full_value is not None and group_fractions is not None:
        full = np.asarray(full_value, dtype=float)
        fractions = np.asarray(group_fractions, dtype=float)
        coefficient = 1.0 - fractions
        coefficient = coefficient.reshape(
            coefficient.shape + (1,) * (values.ndim - coefficient.ndim)
        )
        variance = np.sum(coefficient * (values - full[None, ...]) ** 2, axis=0)
        return np.sqrt(np.maximum(variance, 0.0))
    mean = np.mean(values, axis=0)
    variance = (values.shape[0] - 1) / values.shape[0] * np.sum((values - mean) ** 2, axis=0)
    return np.sqrt(np.maximum(variance, 0.0))


def _complex_jackknife_error(
    jackknife_values: np.ndarray,
    *,
    full_value=None,
    group_fractions=None,
) -> np.ndarray:
    values = np.asarray(jackknife_values)
    full = None if full_value is None else np.asarray(full_value)
    err_real = _real_jackknife_error(
        np.real(values),
        full_value=None if full is None else np.real(full),
        group_fractions=group_fractions,
    )
    err_imag = _real_jackknife_error(
        np.imag(values),
        full_value=None if full is None else np.imag(full),
        group_fractions=group_fractions,
    )
    return err_real.astype(values.dtype) + 1j * err_imag.astype(values.dtype)


def _safe_real_ratio_np(numerator, denominator, *, occupation_floor):
    denominator = np.asarray(denominator)
    denom_real = np.real(denominator)
    denom_safe = np.maximum(denom_real, float(occupation_floor))
    ratio = np.asarray(numerator, dtype=float) / denom_safe
    return np.where(denom_real >= float(occupation_floor), ratio, np.nan)


def _safe_real_product_ratio_np(numerator, left_denominator, right_denominator, *, occupation_floor):
    left = np.asarray(left_denominator, dtype=float)
    right = np.asarray(right_denominator, dtype=float)
    floor = float(occupation_floor)
    denom_safe = np.maximum(left[:, None], floor) * np.maximum(
        right[None, :],
        floor,
    )
    ratio = np.asarray(numerator, dtype=float) / denom_safe
    reliable = (left[:, None] >= floor) & (right[None, :] >= floor)
    return np.where(reliable, ratio, np.nan)


def _resolve_walker_batch_sizes(total_num_walker: int, batch_cfg) -> list[int]:
    """Resolve batches while honoring an explicitly disabled batch mode.

    Legacy configurations that omit ``enabled`` but provide a batch size or
    count still infer batching.  If ``enabled`` is explicitly false, stale
    sizing fields must not silently reactivate it.
    """

    total_num_walker = int(total_num_walker)
    if total_num_walker < 1:
        raise ValueError("simulation.num_walker must be >= 1")
    if batch_cfg is None:
        batch_cfg = {}
    if not isinstance(batch_cfg, dict):
        raise ValueError("simulation.walker_batches must be a dict when provided")

    requested_num_batches = batch_cfg.get("num_batches")
    requested_batch_size = batch_cfg.get(
        "num_walker_per_batch",
        batch_cfg.get("batch_num_walker"),
    )
    has_sizing_request = (
        requested_num_batches is not None or requested_batch_size is not None
    )
    batch_enabled = (
        bool(batch_cfg["enabled"])
        if "enabled" in batch_cfg
        else has_sizing_request
    )
    if not batch_enabled:
        return [total_num_walker]

    if requested_batch_size is None:
        if requested_num_batches is None:
            batch_size = total_num_walker
        else:
            requested_num_batches = int(requested_num_batches)
            if requested_num_batches < 1:
                raise ValueError("simulation.walker_batches.num_batches must be >= 1")
            batch_size = int(np.ceil(total_num_walker / requested_num_batches))
    else:
        batch_size = int(requested_batch_size)
    if batch_size < 1:
        raise ValueError("simulation.walker_batches.num_walker_per_batch must be >= 1")

    if requested_num_batches is None:
        num_batches = int(np.ceil(total_num_walker / batch_size))
    else:
        num_batches = int(requested_num_batches)
        if num_batches < 1:
            raise ValueError("simulation.walker_batches.num_batches must be >= 1")

    batch_sizes = [
        min(batch_size, total_num_walker - batch_index * batch_size)
        for batch_index in range(num_batches)
        if total_num_walker - batch_index * batch_size > 0
    ]
    if sum(batch_sizes) < total_num_walker:
        raise ValueError(
            "simulation.walker_batches does not cover simulation.num_walker; "
            "increase num_batches or num_walker_per_batch"
        )
    return batch_sizes


def _complex_isfinite_jax(value):
    return jnp.isfinite(jnp.real(value)) & jnp.isfinite(jnp.imag(value))


def _walker_finite_mask_jax(lnOmega, alpha, beta, initial_alpha=None, initial_beta=None):
    lnOmega = jnp.asarray(lnOmega, dtype=CDTYPE)
    alpha = jnp.asarray(alpha, dtype=CDTYPE)
    beta = jnp.asarray(beta, dtype=CDTYPE)
    finite = _complex_isfinite_jax(lnOmega)
    finite = finite & jnp.all(_complex_isfinite_jax(alpha), axis=1)
    finite = finite & jnp.all(_complex_isfinite_jax(beta), axis=1)
    if initial_alpha is not None:
        finite = finite & jnp.all(
            _complex_isfinite_jax(jnp.asarray(initial_alpha, dtype=CDTYPE)),
            axis=1,
        )
    if initial_beta is not None:
        finite = finite & jnp.all(
            _complex_isfinite_jax(jnp.asarray(initial_beta, dtype=CDTYPE)),
            axis=1,
        )
    return finite


def _healthy_walker_mask_jax(
    *,
    lnOmega,
    alpha,
    beta,
    remove_unhealth: RemoveUnhealthConfig,
    initial_alpha=None,
    initial_beta=None,
    eps: float = 1.0e-12,
):
    num_walker = int(jnp.asarray(alpha).shape[0])
    if not remove_unhealth.enabled:
        return jnp.ones((num_walker,), dtype=bool)

    lnOmega = jnp.asarray(lnOmega, dtype=CDTYPE)
    alpha = jnp.asarray(alpha, dtype=CDTYPE)
    beta = jnp.asarray(beta, dtype=CDTYPE)
    finite = _walker_finite_mask_jax(lnOmega, alpha, beta, initial_alpha, initial_beta)
    finite_count = jnp.maximum(jnp.sum(finite), jnp.asarray(1, dtype=jnp.int32))

    lnOmega_safe = jnp.where(finite, lnOmega, jnp.asarray(0.0 + 0.0j, dtype=CDTYPE))
    center_candidates = jnp.where(finite, jnp.real(lnOmega_safe), -jnp.inf)
    center = lax.stop_gradient(jnp.max(center_candidates))
    center = jnp.where(jnp.isfinite(center), center, jnp.asarray(0.0, dtype=DTYPE))
    omega = jnp.where(
        finite,
        jnp.exp(lnOmega_safe - center),
        jnp.asarray(0.0 + 0.0j, dtype=CDTYPE),
    )
    mean_omega = jnp.sum(omega) / finite_count
    eps_complex = jnp.asarray(eps + 0.0j, dtype=CDTYPE)
    mean_omega = jnp.where(jnp.abs(mean_omega) > eps, mean_omega, eps_complex)
    omega_hat = omega / mean_omega

    density = jnp.where(finite[:, None], alpha * beta, jnp.asarray(0.0 + 0.0j, dtype=CDTYPE))
    weighted_density_abs = jnp.max(jnp.abs(omega_hat[:, None] * density), axis=1)
    log_magnitude = jnp.log1p(weighted_density_abs)
    log_magnitude_masked = jnp.where(finite, log_magnitude, jnp.nan)
    median = jnp.nanmedian(log_magnitude_masked)
    mad = jnp.nanmedian(jnp.where(finite, jnp.abs(log_magnitude - median), jnp.nan))
    sigma = jnp.maximum(jnp.asarray(1.4826, dtype=DTYPE) * mad, jnp.asarray(eps, dtype=DTYPE))
    threshold = median + jnp.asarray(remove_unhealth.logstd, dtype=DTYPE) * sigma
    healthy = finite & (log_magnitude <= threshold)
    return jnp.where(jnp.isfinite(threshold), healthy, finite)


def _filter_unhealthy_walkers_np(
    *,
    initial_alpha,
    initial_beta,
    lnOmega,
    alpha,
    beta,
    remove_unhealth: RemoveUnhealthConfig,
):
    if not remove_unhealth.enabled:
        walker_count = int(alpha.shape[0])
        info = {
            "healthy_count": walker_count,
            "walker_count": walker_count,
        }
        return initial_alpha, initial_beta, lnOmega, alpha, beta, info

    healthy_mask = np.asarray(
        _healthy_walker_mask_jax(
            lnOmega=lnOmega,
            alpha=alpha,
            beta=beta,
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            remove_unhealth=remove_unhealth,
        )
    ).astype(bool)
    walker_count = int(healthy_mask.shape[0])
    healthy_count = int(np.sum(healthy_mask))
    if healthy_count == 0:
        finite_mask = np.asarray(
            _walker_finite_mask_jax(
                jnp.asarray(lnOmega, dtype=CDTYPE),
                jnp.asarray(alpha, dtype=CDTYPE),
                jnp.asarray(beta, dtype=CDTYPE),
                jnp.asarray(initial_alpha, dtype=CDTYPE),
                jnp.asarray(initial_beta, dtype=CDTYPE),
            )
        ).astype(bool)
        if np.any(finite_mask):
            healthy_mask = finite_mask
            healthy_count = int(np.sum(healthy_mask))

    info = {"healthy_count": healthy_count, "walker_count": walker_count}
    return (
        np.asarray(initial_alpha)[healthy_mask],
        np.asarray(initial_beta)[healthy_mask],
        np.asarray(lnOmega)[healthy_mask],
        np.asarray(alpha)[healthy_mask],
        np.asarray(beta)[healthy_mask],
        info,
    )


def _filter_finite_walkers_np(
    *,
    initial_alpha,
    initial_beta,
    lnOmega,
    alpha,
    beta,
):
    finite_mask = np.asarray(
        _walker_finite_mask_jax(
            jnp.asarray(lnOmega, dtype=CDTYPE),
            jnp.asarray(alpha, dtype=CDTYPE),
            jnp.asarray(beta, dtype=CDTYPE),
            jnp.asarray(initial_alpha, dtype=CDTYPE),
            jnp.asarray(initial_beta, dtype=CDTYPE),
        )
    ).astype(bool)
    return (
        np.asarray(lnOmega)[finite_mask],
        np.asarray(alpha)[finite_mask],
        np.asarray(beta)[finite_mask],
    )


def _as_shell_pair_arrays(shell_pairs, shell: str):
    if shell_pairs is None:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty
    left, right = shell_pairs.get(shell, ((), ()))
    return jnp.asarray(left, dtype=jnp.int32), jnp.asarray(right, dtype=jnp.int32)


def _weighted_shell_sum(weight, x_left, x_right, left_indices, right_indices):
    samples = shell_pair_average_samples(x_left, x_right, left_indices, right_indices)
    return jnp.sum(weight * samples)


def _compute_observable_sum_snapshot(
    *,
    initial_alpha,
    initial_beta,
    lnOmega,
    lnOmega_shift,
    alpha,
    beta,
    observable_names=(),
    shell_pairs=None,
    remove_unhealth: RemoveUnhealthConfig = RemoveUnhealthConfig(),
):
    """Small sufficient statistics for globally merged weighted observables.

    The solver stores log-weights after subtracting a common real center each
    step.  ``lnOmega_shift`` is the cumulative removed center for this batch, so
    ``exp(lnOmega + lnOmega_shift)`` restores the batch's relative denominator
    scale before a stable per-snapshot centering is applied.
    """

    ln_raw = jnp.asarray(lnOmega, dtype=CDTYPE) + jnp.asarray(lnOmega_shift, dtype=CDTYPE)
    alpha = jnp.asarray(alpha, dtype=CDTYPE)
    beta = jnp.asarray(beta, dtype=CDTYPE)
    initial_alpha = jnp.asarray(initial_alpha, dtype=CDTYPE)
    initial_beta = jnp.asarray(initial_beta, dtype=CDTYPE)
    healthy_mask = _healthy_walker_mask_jax(
        lnOmega=ln_raw,
        alpha=alpha,
        beta=beta,
        initial_alpha=initial_alpha,
        initial_beta=initial_beta,
        remove_unhealth=remove_unhealth,
    )
    center_candidates = jnp.where(
        healthy_mask | (not remove_unhealth.enabled),
        jnp.real(ln_raw),
        -jnp.inf,
    )
    center = lax.stop_gradient(jnp.max(center_candidates))
    center = jnp.where(jnp.isfinite(center), center, jnp.asarray(0.0, dtype=DTYPE))
    ln_weight = jnp.where(
        healthy_mask | (not remove_unhealth.enabled),
        ln_raw,
        jnp.asarray(0.0 + 0.0j, dtype=CDTYPE),
    )
    weight = jnp.exp(ln_weight - center)
    weight = jnp.where(
        healthy_mask | (not remove_unhealth.enabled),
        weight,
        jnp.asarray(0.0 + 0.0j, dtype=CDTYPE),
    )
    density_samples = beta * alpha
    initial_density = initial_beta * initial_alpha
    names = set(str(name) for name in observable_names)
    if not names:
        names = set(SUPPORTED_SIMULATION_OBSERVABLES) - set(PARETO_K_OBSERVABLES)
    full_equal = bool(names & {"G1", "G2", "g2"})
    full_initial = bool(names & {"G1_initial", "G2_initial", "g2_initial"})
    need_density = bool(
        names
        & {
            "density",
            "N",
            "G1_local",
            "G2_local",
            "g2_local",
            "g2_nn",
            "g2_nnn",
            "coherence_fraction",
            "g2_initial",
        }
    )
    need_A = bool(names & {"A", "coherence_fraction"})
    need_B = bool(names & {"B"})
    need_G2_local = bool(names & {"G2_local", "g2_local"})
    need_initial_density = bool(names & {"g2_initial"})
    stats = {
        "log_weight_center": np.asarray(center),
        "walker_count": np.asarray(alpha.shape[0], dtype=np.int64),
        "healthy_count": np.asarray(jnp.sum(healthy_mask), dtype=np.int64),
        "denominator_sum": np.asarray(jnp.sum(weight)),
    }
    if need_A:
        stats["A_sum"] = np.asarray(jnp.einsum("w,wi->i", weight, alpha))
    if need_B:
        stats["B_sum"] = np.asarray(jnp.einsum("w,wi->i", weight, beta))
    if need_density or full_equal:
        stats["density_sum"] = np.asarray(jnp.einsum("w,wi->i", weight, density_samples))
    if need_G2_local:
        stats["G2_local_sum"] = np.asarray(jnp.einsum("w,wi->i", weight, density_samples * density_samples))
    if full_equal:
        stats["G1_sum"] = np.asarray(jnp.einsum("w,wi,wj->ij", weight, beta, alpha))
        stats["G2_sum"] = np.asarray(jnp.einsum("w,wi,wj->ij", weight, density_samples, density_samples))
    if full_initial:
        stats["G1_initial_sum"] = np.asarray(jnp.einsum("w,wi,wj->ij", weight, initial_beta, alpha))
        if names & {"G2_initial", "g2_initial"}:
            stats["G2_initial_sum"] = np.asarray(
                jnp.einsum("w,wi,wj->ij", weight, initial_density, density_samples)
            )
    if need_initial_density:
        stats["N_initial_sum"] = np.asarray(jnp.einsum("w,wi->i", weight, initial_density))
    if names & {"G1_nn", "g2_nn"}:
        left, right = _as_shell_pair_arrays(shell_pairs, "nn")
        if "G1_nn" in names:
            stats["G1_nn_sum"] = np.asarray(_weighted_shell_sum(weight, beta, alpha, left, right))
        if "g2_nn" in names:
            stats["G2_nn_sum"] = np.asarray(_weighted_shell_sum(weight, density_samples, density_samples, left, right))
    if names & {"G1_nnn", "g2_nnn"}:
        left, right = _as_shell_pair_arrays(shell_pairs, "nnn")
        if "G1_nnn" in names:
            stats["G1_nnn_sum"] = np.asarray(_weighted_shell_sum(weight, beta, alpha, left, right))
        if "g2_nnn" in names:
            stats["G2_nnn_sum"] = np.asarray(_weighted_shell_sum(weight, density_samples, density_samples, left, right))
    if "G1_initial_local" in names:
        stats["G1_initial_local_sum"] = np.asarray(jnp.einsum("w,wi->i", weight, initial_beta * alpha))
    if "G1_initial_nn" in names:
        left, right = _as_shell_pair_arrays(shell_pairs, "nn")
        stats["G1_initial_nn_sum"] = np.asarray(_weighted_shell_sum(weight, initial_beta, alpha, left, right))
    if "G1_initial_nnn" in names:
        left, right = _as_shell_pair_arrays(shell_pairs, "nnn")
        stats["G1_initial_nnn_sum"] = np.asarray(_weighted_shell_sum(weight, initial_beta, alpha, left, right))
    return stats


def _stack_observable_sum_history(stat_buffers):
    return {name: np.asarray(values) for name, values in stat_buffers.items()}


def _scaled_sum_from_batch_stats(batch_stats, key: str, center_global: np.ndarray):
    values = np.asarray(batch_stats[key])
    factors = np.exp(np.asarray(batch_stats["log_weight_center"]) - center_global)
    reshape = factors.shape + (1,) * (values.ndim - factors.ndim)
    return np.sum(values * factors.reshape(reshape), axis=0)


def _shell_density_product_np(N, shell_pairs, shell: str):
    if shell_pairs is None:
        return np.full(N.shape[:-1], np.nan, dtype=float)
    left, right = shell_pairs.get(shell, ((), ()))
    left = np.asarray(left, dtype=np.int32)
    right = np.asarray(right, dtype=np.int32)
    if left.size == 0:
        return np.full(N.shape[:-1], np.nan, dtype=float)
    return np.mean(N[..., left] * N[..., right], axis=-1)


def _finalize_observable_sums(batch_stats, observable_names, shell_pairs=None):
    centers = np.asarray(batch_stats["log_weight_center"])
    center_global = np.max(centers, axis=0)
    denom = _scaled_sum_from_batch_stats(batch_stats, "denominator_sum", center_global)

    def ratio(key):
        numerator = _scaled_sum_from_batch_stats(batch_stats, key, center_global)
        return numerator / denom.reshape(denom.shape + (1,) * (numerator.ndim - denom.ndim))

    available = {}
    if "A_sum" in batch_stats:
        available["A"] = ratio("A_sum")
    if "B_sum" in batch_stats:
        available["B"] = ratio("B_sum")
    if "G1_sum" in batch_stats:
        G1 = ratio("G1_sum")
        available["G1"] = G1
        density = np.diagonal(G1, axis1=-2, axis2=-1)
    elif "density_sum" in batch_stats:
        density = ratio("density_sum")
    else:
        density = None
    if density is not None:
        available["density"] = density
        available["N"] = np.real(density).astype(float)
        available["G1_local"] = density
    N = available.get("N")
    if "G2_sum" in batch_stats:
        G2 = ratio("G2_sum")
        available["G2"] = G2
        G2_local = np.real(np.diagonal(G2, axis1=-2, axis2=-1)).astype(float)
        if N is not None:
            available["g2"] = np.asarray(
                [
                    _safe_real_product_ratio_np(
                        np.real(G2_t),
                        N_t,
                        N_t,
                        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
                    )
                    for G2_t, N_t in zip(G2, N)
                ]
            )
    elif "G2_local_sum" in batch_stats:
        G2_local = np.real(ratio("G2_local_sum")).astype(float)
    else:
        G2_local = None
    if G2_local is not None:
        available["G2_local"] = G2_local
        if N is not None:
            available["g2_local"] = _safe_real_ratio_np(
                G2_local,
                N**2,
                occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
            )
    if "A" in available and N is not None:
        available["coherence_fraction"] = _safe_real_ratio_np(
            np.abs(available["A"]) ** 2,
            N,
            occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
        )
    if "G1_initial_sum" in batch_stats:
        available["G1_initial"] = ratio("G1_initial_sum")
    if "G2_initial_sum" in batch_stats:
        G2_initial = ratio("G2_initial_sum")
        available["G2_initial"] = G2_initial
        if "N_initial_sum" in batch_stats and N is not None:
            N_initial = np.real(ratio("N_initial_sum")).astype(float)
            available["g2_initial"] = np.asarray(
                [
                    _safe_real_product_ratio_np(
                        np.real(G2_t),
                        N_initial_t,
                        N_t,
                        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
                    )
                    for G2_t, N_initial_t, N_t in zip(G2_initial, N_initial, N)
                ]
            )
    for name in ("G1_nn", "G1_nnn", "G1_initial_local", "G1_initial_nn", "G1_initial_nnn"):
        key = f"{name}_sum"
        if key in batch_stats:
            available[name] = ratio(key)
    if N is not None:
        for name, shell in (("g2_nn", "nn"), ("g2_nnn", "nnn")):
            key = f"G2_{shell}_sum"
            if key in batch_stats:
                numerator = np.real(ratio(key)).astype(float)
                denominator = _shell_density_product_np(N, shell_pairs, shell)
                available[name] = _safe_real_ratio_np(
                    numerator,
                    denominator,
                    occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
                )
    return {name: available[name] for name in observable_names}


def _observable_jackknife_errors(batch_stats, observable_names, shell_pairs=None):
    num_batches = int(np.asarray(batch_stats["denominator_sum"]).shape[0])
    full_values = _finalize_observable_sums(batch_stats, observable_names, shell_pairs=shell_pairs)
    if num_batches <= 1:
        return {name: np.zeros_like(value) for name, value in full_values.items()}
    group_counts = np.asarray(
        batch_stats.get(
            "walker_count",
            np.ones_like(batch_stats["denominator_sum"], dtype=float),
        ),
        dtype=float,
    )
    total_group_count = np.sum(group_counts, axis=0)
    group_fractions = group_counts / np.maximum(total_group_count, 1.0)
    jackknife_values = {name: [] for name in observable_names}
    for drop_index in range(num_batches):
        reduced = {name: np.delete(np.asarray(value), drop_index, axis=0) for name, value in batch_stats.items()}
        reduced_values = _finalize_observable_sums(reduced, observable_names, shell_pairs=shell_pairs)
        for name in observable_names:
            jackknife_values[name].append(reduced_values[name])
    errors = {}
    for name in observable_names:
        values = np.asarray(jackknife_values[name])
        if np.iscomplexobj(values):
            errors[name] = _complex_jackknife_error(
                values,
                full_value=full_values[name],
                group_fractions=group_fractions,
            )
        else:
            errors[name] = _real_jackknife_error(
                values,
                full_value=full_values[name],
                group_fractions=group_fractions,
            )
    return errors


def _merge_chunked_log_weight_histories(batch_results):
    """Align centered batch log-weights to one stable shared shift history."""

    shifts = np.stack(
        [
            np.asarray(batch_result["lnOmega_shift_history"], dtype=float)
            for batch_result in batch_results
        ],
        axis=0,
    )
    shared_shift = np.max(shifts, axis=0)
    aligned_histories = []
    for batch_result, batch_shift in zip(batch_results, shifts):
        history = np.asarray(batch_result["lnOmega_history"])
        real_dtype = np.real(history).dtype
        relative_shift = np.asarray(
            batch_shift - shared_shift,
            dtype=real_dtype,
        )
        aligned_histories.append(history + relative_shift[:, None])
    return np.concatenate(aligned_histories, axis=1), shared_shift


class GaugeSimulator:
    """Simulation runner that saves raw window-end trajectory data for post-processing."""

    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path
        self.lattice = Lattice.from_config(config["lattice"])
        gauge_mode = config["simulation"]["gauge_mode"]
        self.model = (
            Model(config["model"], self.lattice, gauge_mode=gauge_mode)
            if gauge_mode in NEURAL_GAUGE_MODES
            else None
        )

    @classmethod
    def from_config_path(cls, config_path: str):
        return cls(load_config(config_path), config_path=config_path)

    @property
    def save_dir(self) -> str:
        return self.config["io"]["save_dir"]

    @property
    def simulation_dir(self) -> str:
        return os.path.join(self.save_dir, "simulation")

    @property
    def gauge_tag(self) -> str:
        mode = str(self.config["simulation"]["gauge_mode"]).strip()
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in mode)
        return safe or "unknown_gauge"

    @property
    def params_path(self) -> str:
        return resolve_checkpoint_params_path(self.config, self.save_dir, purpose="simulation")

    @property
    def results_path(self) -> str:
        extension = archive_extension(self._resolve_results_save_policy().archive_format)
        return os.path.join(self.simulation_dir, f"simulation_results_{self.gauge_tag}{extension}")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.simulation_dir, f"simulation_metadata_{self.gauge_tag}.json")

    @property
    def benchmark_path(self) -> str:
        return os.path.join(self.simulation_dir, "ed_benchmark.npz")

    def _reduced_observable_shell_pairs(self):
        return {
            "local": self.lattice.shell_pair_indices("local"),
            "nn": self.lattice.shell_pair_indices("nn"),
            "nnn": self.lattice.shell_pair_indices("nnn"),
        }

    @property
    def observables_path(self) -> str:
        return os.path.join(self.simulation_dir, f"simulation_observables_{self.gauge_tag}.npz")

    @property
    def config_copy_path(self) -> str:
        return os.path.join(self.simulation_dir, f"config_used_{self.gauge_tag}.json")

    def _prepare_io(self):
        os.makedirs(self.simulation_dir, exist_ok=True)
        save_json(self.config, self.config_copy_path)

    def _resolve_physics(self):
        lat_cfg = self.config["lattice"]
        return SimulationPhysics(
            n0=float(lat_cfg["n0"]),
            U=float(lat_cfg["U"]),
            gamma=float(lat_cfg["gamma"]),
            F=complex(lat_cfg["F_real"] + 1j * lat_cfg.get("F_imag", 0.0)),
            Delta=float(lat_cfg["Delta"]),
        )

    def _resolve_ed_config(self):
        ed_cfg = self.config["simulation"].get("ed", {})
        return EDBenchmarkConfig(
            enabled=bool(ed_cfg.get("enabled", False)),
            n_cut=int(ed_cfg.get("n_cut", 0)),
        )

    def _build_ed_times(self):
        sim_cfg = self.config["simulation"]
        subdivision_factor = 10
        n_steps = int(sim_cfg["N_steps"])
        n_windows = int(sim_cfg["N_windows"])
        if n_steps % subdivision_factor != 0:
            raise ValueError(
                "simulation.ed currently requires simulation.N_steps divisible by 10 "
                "because the ED reference uses the same dt with N_steps/10 and N_windows*10"
            )
        ed_n_steps = n_steps // subdivision_factor
        ed_n_windows = n_windows * subdivision_factor
        dt = float(sim_cfg["dt"])
        t0 = float(sim_cfg["t0"])
        ed_times = t0 + dt * ed_n_steps * np.arange(ed_n_windows + 1, dtype=float)
        return ed_times, ed_n_steps, ed_n_windows

    def _resolve_results_save_policy(self):
        sim_cfg = self.config["simulation"]
        return ArchiveSavePolicy(
            archive_format=str(sim_cfg.get("save_format", "npz")).strip().lower(),
            compressed=bool(sim_cfg.get("save_compressed", True)),
            precision=str(sim_cfg.get("save_precision", "float32")).strip().lower(),
        )

    def _resolve_ed_save_policy(self):
        return ArchiveSavePolicy(
            archive_format="npz",
            compressed=False,
            precision="runtime",
        )

    def _resolve_observables_save_policy(self):
        return ArchiveSavePolicy(
            archive_format="npz",
            compressed=False,
            precision="runtime",
        )

    def _resolve_remove_unhealth_config(self):
        sim_cfg = self.config["simulation"]
        raw_cfg = sim_cfg.get("remove_unhealth", {})
        if raw_cfg is None:
            raw_cfg = {}
        if isinstance(raw_cfg, bool):
            raw_cfg = {"enabled": raw_cfg}
        if not isinstance(raw_cfg, dict):
            raise ValueError("simulation.remove_unhealth must be a dict when provided")
        return RemoveUnhealthConfig(
            enabled=bool(raw_cfg.get("enabled", False)),
            logstd=float(raw_cfg.get("logstd", 15.0)),
        )

    def _selected_pareto_k_observable_names(self):
        max_order, mode, monomials = self._pareto_k_observable_selector()
        names = []
        for _total_order, _m_power, _n_power, base_name in selected_monomial_specs(
            PARETO_K_OBSERVABLE_SPECS,
            max_order,
            mode,
            monomials,
        ):
            names.extend((base_name, f"{base_name}_mean", f"{base_name}_max"))
        return tuple(names)

    def _pareto_k_observable_selector(self):
        train_cfg = self.config.get("training", {})
        raw_quantity = train_cfg.get("pareto_k_applied_quantities", 6)
        raw_monomials = train_cfg.get("pareto_k_monomials", ())
        if is_auto_monomial_selector(raw_monomials) or (
            "pareto_k_monomials" not in train_cfg and is_auto_monomial_selector(raw_quantity)
        ):
            raw_operator_quantity = train_cfg.get(
                "operator_applied_quantities",
                OPERATOR_MOMENT_DEFAULT_ORDER,
            )
            operator_monomials = train_cfg.get("operator_monomials", ())
            quantity_is_explicit_operator_selector = (
                not operator_monomials
                and not isinstance(raw_operator_quantity, (int, np.integer))
                and not (isinstance(raw_operator_quantity, str) and raw_operator_quantity.strip().isdigit())
            )
            if quantity_is_explicit_operator_selector:
                operator_monomials = normalize_monomial_pairs(
                    raw_operator_quantity,
                    option_name="training.operator_applied_quantities",
                    max_order=OPERATOR_MOMENT_MAX_ORDER,
                )
                op_order = max(m_power + n_power for m_power, n_power in operator_monomials)
                op_mode = normalize_applied_quantities_mode(
                    train_cfg.get("operator_applied_quantities_mode", "exact")
                )
            else:
                op_order = normalize_applied_quantities(raw_operator_quantity)
                op_mode = normalize_applied_quantities_mode(
                    train_cfg.get("operator_applied_quantities_mode", "exact")
                )
                operator_monomials = normalize_monomial_pairs(
                    operator_monomials,
                    option_name="training.operator_monomials",
                    max_order=OPERATOR_MOMENT_MAX_ORDER,
                )
            operator_specs = selected_monomial_specs(
                OPERATOR_MOMENT_SPECS,
                op_order,
                op_mode,
                operator_monomials,
            )
            auto_monomials = onsite_monomials_in_operator_equations(
                ((m_power, n_power) for _total_order, m_power, n_power, _term in operator_specs),
                max_order=6,
            )
            if not auto_monomials:
                raise ValueError("training.pareto_k_monomials='auto' requires selected operator moments")
            max_order = max(m_power + n_power for m_power, n_power in auto_monomials)
            return max_order, "exact", auto_monomials
        quantity_is_explicit_selector = (
            "pareto_k_monomials" not in train_cfg
            and not isinstance(raw_quantity, (int, np.integer))
            and not (isinstance(raw_quantity, str) and raw_quantity.strip().isdigit())
        )
        if quantity_is_explicit_selector:
            monomials = normalize_monomial_pairs(
                raw_quantity,
                option_name="training.pareto_k_applied_quantities",
                max_order=6,
            )
            max_order = max(m_power + n_power for m_power, n_power in monomials)
            mode = normalize_applied_quantities_mode(
                train_cfg.get("pareto_k_applied_quantities_mode", "exact")
            )
            return max_order, mode, monomials
        max_order = normalize_applied_quantities(raw_quantity)
        mode = normalize_applied_quantities_mode(
            train_cfg.get("pareto_k_applied_quantities_mode", "upto")
        )
        monomials = normalize_monomial_pairs(
            raw_monomials,
            option_name="training.pareto_k_monomials",
            max_order=6,
        )
        return max_order, mode, monomials

    def _pareto_k_summary_block(self, pareto_k_observable_names, save_raw_walkers_every_windows):
        pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials = (
            self._pareto_k_observable_selector()
        )
        return {
            "enabled": bool(pareto_k_observable_names),
            "names": list(pareto_k_observable_names),
            "source": "self-normalized observable clouds",
            "pareto_k_applied_quantities": int(pareto_k_applied_quantities),
            "pareto_k_applied_quantities_mode": str(pareto_k_applied_quantities_mode),
            "pareto_k_monomials": [
                [int(m_power), int(n_power)] for m_power, n_power in pareto_k_monomials
            ],
            "cadence": "save_raw_walkers_every_windows",
            "save_every_windows": int(save_raw_walkers_every_windows),
            "time_grid": "pareto-k_times",
        }

    def _resolve_observable_names(self):
        sim_cfg = self.config["simulation"]
        names = sim_cfg.get("observables", DEFAULT_SIMULATION_OBSERVABLES)
        if names is None:
            return ()
        if isinstance(names, str):
            names = [names]
        normalized = []
        for name in names:
            name = SIMULATION_OBSERVABLE_ALIASES.get(str(name), str(name))
            if name not in SUPPORTED_SIMULATION_OBSERVABLES:
                raise ValueError(
                    f"Unsupported simulation observable '{name}'. "
                    f"Supported: {sorted(SUPPORTED_SIMULATION_OBSERVABLES)}"
                )
            if name not in normalized:
                normalized.append(name)
        if any(name in PARETO_K_AGGREGATE_OBSERVABLES for name in normalized):
            for name in self._selected_pareto_k_observable_names():
                if name not in normalized:
                    normalized.append(name)
        selected_pareto_names = set(self._selected_pareto_k_observable_names())
        unsupported_pareto_names = [
            name
            for name in normalized
            if (
                name in PARETO_K_OBSERVABLES
                and name not in PARETO_K_AGGREGATE_OBSERVABLES
                and name not in selected_pareto_names
            )
        ]
        if unsupported_pareto_names:
            raise ValueError(
                "Requested Pareto-k monomial observables are not selected by "
                "training.pareto_k_monomials or training.pareto_k_applied_quantities/mode: "
                f"{unsupported_pareto_names}. Selected monomial observables: {sorted(selected_pareto_names)}"
            )
        return tuple(normalized)

    def _resolve_analytical_target_time(self, rollout_end_time: float):
        sim_cfg = self.config["simulation"]
        return float(sim_cfg.get("analytic_t_fin", rollout_end_time))

    def _resolve_analytical_gauge_description(self, analytic_target_time: float):
        sim_cfg = self.config["simulation"]
        mode = sim_cfg["gauge_mode"]
        if mode in NEURAL_GAUGE_MODES:
            return None

        info = validate_analytical_gauge_mode(mode=mode)
        return {
            "mode": info.mode,
            "source": info.source,
            "location": info.location,
            "summary": info.summary,
            "best_for": info.best_for,
            "caveats": info.caveats,
            "target_time": float(analytic_target_time),
        }

    def _sample_model_inputs(self):
        lnOmega0, alpha0, beta0 = self.lattice.initialize_phase_space(num_walker=1, n0=self.lattice.n0)
        alpha_real, beta_real = var_complex_to_real(alpha0, beta0)
        lnOmega_real = jnp.stack([jnp.real(lnOmega0), jnp.imag(lnOmega0)], axis=-1)
        return lnOmega_real, alpha_real, beta_real, self.lattice.physical_params()

    def _load_neural_parameters(self):
        if not os.path.exists(self.params_path):
            raise FileNotFoundError(
                f"simulation.gauge_mode='{self.config['simulation']['gauge_mode']}' requires parameters at '{self.params_path}'"
            )

        lnOmega_real, alpha_real, beta_real, physical_params = self._sample_model_inputs()
        state = self.model.create_train_state(
            config=self.config,
            key=random.PRNGKey(0),
            sample_lnOmega_real=lnOmega_real,
            sample_alpha_real=alpha_real,
            sample_beta_real=beta_real,
            sample_t=0.0,
            physical_params=physical_params,
            params_path=self.params_path,
        )
        return state.params

    def _compute_pareto_k_observable_snapshot(self, *, lnOmega, alpha, beta):
        train_cfg = self.config.get("training", {})
        walker_count = int(alpha.shape[0])
        pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials = (
            self._pareto_k_observable_selector()
        )
        if walker_count < 2:
            num_site = int(alpha.shape[-1]) if np.asarray(alpha).ndim >= 2 else int(self.lattice.num_site)
            out = {
                "pareto-k_mean": np.asarray(np.nan, dtype=float),
                "pareto-k_max": np.asarray(np.nan, dtype=float),
            }
            for _total_order, _m_power, _n_power, base_name in selected_monomial_specs(
                PARETO_K_OBSERVABLE_SPECS,
                pareto_k_applied_quantities,
                pareto_k_applied_quantities_mode,
                pareto_k_monomials,
            ):
                out[base_name] = np.full((num_site,), np.nan, dtype=float)
                out[f"{base_name}_mean"] = np.asarray(np.nan, dtype=float)
                out[f"{base_name}_max"] = np.asarray(np.nan, dtype=float)
            return out

        pareto_k_tail_fraction = float(train_cfg.get("pareto_k_tail_fraction", 0.01))
        pareto_k_min_tail_count = int(train_cfg.get("pareto_k_min_tail_count", 32))
        diagnostics = compute_observable_pareto_k_indices(
            jnp.asarray(lnOmega),
            jnp.asarray(alpha),
            jnp.asarray(beta),
            applied_quantities=pareto_k_applied_quantities,
            applied_quantities_mode=pareto_k_applied_quantities_mode,
            monomials=pareto_k_monomials,
            q_winsor=float(train_cfg.get("q_winsor", 0.95)),
            pareto_k_threshold=float(train_cfg.get("pareto_k_threshold", 0.7)),
            pareto_k_threshold_tau=float(train_cfg.get("pareto_k_threshold_tau", 0.1)),
            pareto_k_envelope_beta=float(train_cfg.get("pareto_k_envelope_beta", 0.5)),
            pareto_k_envelope_excess=str(train_cfg.get("pareto_k_envelope_excess", "log")),
            pareto_k_tail_fraction=pareto_k_tail_fraction,
            pareto_k_min_tail_count=pareto_k_min_tail_count,
        )
        return {name: np.asarray(value) for name, value in diagnostics.items()}

    def _compute_observable_snapshot(
        self,
        *,
        observable_names,
        initial_alpha,
        initial_beta,
        lnOmega,
        alpha,
        beta,
        remove_unhealth: RemoveUnhealthConfig = RemoveUnhealthConfig(),
        save_observable_errors: bool = True,
    ):
        pareto_k_names = [name for name in observable_names if name in PARETO_K_OBSERVABLES]
        raw_lnOmega = raw_alpha = raw_beta = None
        if pareto_k_names:
            raw_lnOmega, raw_alpha, raw_beta = _filter_finite_walkers_np(
                initial_alpha=initial_alpha,
                initial_beta=initial_beta,
                lnOmega=lnOmega,
                alpha=alpha,
                beta=beta,
            )
        (
            initial_alpha,
            initial_beta,
            lnOmega,
            alpha,
            beta,
            filter_info,
        ) = _filter_unhealthy_walkers_np(
            initial_alpha=initial_alpha,
            initial_beta=initial_beta,
            lnOmega=lnOmega,
            alpha=alpha,
            beta=beta,
            remove_unhealth=remove_unhealth,
        )
        measurements = {}
        measurement_errors = {}
        reduced_equal_time_names = [
            name
            for name in observable_names
            if name in REDUCED_EQUAL_TIME_OBSERVABLES
            or name in {"A", "B", "density", "N", "coherence_fraction"}
        ]
        equal_time_names = [
            name
            for name in observable_names
            if (
                name not in INITIAL_TIME_OBSERVABLES
                and name not in PARETO_K_OBSERVABLES
                and name not in reduced_equal_time_names
                and name not in REDUCED_INITIAL_TIME_OBSERVABLES
            )
        ]
        reduced_initial_time_names = [
            name for name in observable_names if name in REDUCED_INITIAL_TIME_OBSERVABLES
        ]
        initial_time_names = [
            name
            for name in observable_names
            if name in INITIAL_TIME_OBSERVABLES and name not in reduced_initial_time_names
        ]
        reduced_equal_time_set = set(reduced_equal_time_names)
        if reduced_equal_time_names and reduced_equal_time_set <= LOCAL_EQUAL_TIME_OBSERVABLES:
            if save_observable_errors:
                available, available_errors = compute_local_equal_time_measurements(
                    jnp.asarray(lnOmega),
                    jnp.asarray(alpha),
                    jnp.asarray(beta),
                )
            else:
                available = compute_local_equal_time_measurements_no_error(
                    jnp.asarray(lnOmega),
                    jnp.asarray(alpha),
                    jnp.asarray(beta),
                )
                available_errors = {name: jnp.zeros_like(value) for name, value in available.items()}
            for name in reduced_equal_time_names:
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.asarray(available_errors[name])
        elif reduced_equal_time_names:
            shell_pairs = self._reduced_observable_shell_pairs()
            nn_left, nn_right = _as_shell_pair_arrays(shell_pairs, "nn")
            nnn_left, nnn_right = _as_shell_pair_arrays(shell_pairs, "nnn")
            available, available_errors = compute_reduced_equal_time_measurements(
                jnp.asarray(lnOmega),
                jnp.asarray(alpha),
                jnp.asarray(beta),
                nn_left,
                nn_right,
                nnn_left,
                nnn_right,
            )
            for name in reduced_equal_time_names:
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.asarray(available_errors[name])
        if equal_time_names:
            available, available_errors = compute_equal_time_measurements(
                jnp.asarray(lnOmega),
                jnp.asarray(alpha),
                jnp.asarray(beta),
            )
            for name in equal_time_names:
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.asarray(available_errors[name])
        reduced_initial_time_set = set(reduced_initial_time_names)
        if reduced_initial_time_names and reduced_initial_time_set <= LOCAL_INITIAL_TIME_OBSERVABLES:
            if save_observable_errors:
                available, available_errors = compute_local_initial_time_measurements(
                    jnp.asarray(initial_beta),
                    jnp.asarray(lnOmega),
                    jnp.asarray(alpha),
                )
            else:
                available = compute_local_initial_time_measurements_no_error(
                    jnp.asarray(initial_beta),
                    jnp.asarray(lnOmega),
                    jnp.asarray(alpha),
                )
                available_errors = {name: jnp.zeros_like(value) for name, value in available.items()}
            for name in reduced_initial_time_names:
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.asarray(available_errors[name])
        elif reduced_initial_time_names:
            shell_pairs = self._reduced_observable_shell_pairs()
            nn_left, nn_right = _as_shell_pair_arrays(shell_pairs, "nn")
            nnn_left, nnn_right = _as_shell_pair_arrays(shell_pairs, "nnn")
            available, available_errors = compute_reduced_initial_time_measurements(
                jnp.asarray(initial_beta),
                jnp.asarray(lnOmega),
                jnp.asarray(alpha),
                nn_left,
                nn_right,
                nnn_left,
                nnn_right,
            )
            for name in reduced_initial_time_names:
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.asarray(available_errors[name])
        if initial_time_names:
            available, available_errors = compute_initial_time_measurements(
                jnp.asarray(initial_beta),
                jnp.asarray(initial_alpha),
                jnp.asarray(lnOmega),
                jnp.asarray(alpha),
                jnp.asarray(beta),
            )
            for name in initial_time_names:
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.asarray(available_errors[name])
        if pareto_k_names:
            available = self._compute_pareto_k_observable_snapshot(
                lnOmega=raw_lnOmega,
                alpha=raw_alpha,
                beta=raw_beta,
            )
            for name in pareto_k_names:
                if name not in available:
                    selected = sorted(key for key in available if key.startswith("pareto-k"))
                    raise ValueError(
                        f"Requested Pareto-k observable '{name}' is not selected by the current "
                        f"training Pareto-k monomial selector. Available Pareto-k observables: {selected}"
                    )
                measurements[name] = np.asarray(available[name])
                measurement_errors[name] = np.zeros_like(measurements[name], dtype=float)
        return measurements, measurement_errors, filter_info

    def _run_stochastic_rollout(
        self,
        *,
        rollout_kwargs: Dict[str, Any],
        save_raw_walkers: bool,
        save_raw_walkers_every_windows: int,
        save_observables: bool,
        save_observables_every_windows: int,
        save_observable_errors: bool,
        observable_names,
        save_observable_sums: bool,
        remove_unhealth: RemoveUnhealthConfig,
        progress_every_window: int,
    ):
        n_windows = int(rollout_kwargs["N_windows"])
        num_times = n_windows + 1
        lnOmega0 = rollout_kwargs["lnOmega0"]
        alpha0 = rollout_kwargs["alpha0"]
        beta0 = rollout_kwargs["beta0"]
        times = np.empty((num_times,), dtype=float)

        raw_histories = {}
        raw_times = None
        raw_window_lookup = {}
        pareto_k_names = tuple(name for name in observable_names if name in PARETO_K_OBSERVABLES)
        pareto_k_window_set = set()
        if save_raw_walkers:
            raw_stride = max(1, int(save_raw_walkers_every_windows))
            raw_window_indices = [
                index
                for index in range(num_times)
                if index == 0 or index == n_windows or index % raw_stride == 0
            ]
            raw_window_lookup = {
                window_index: raw_index
                for raw_index, window_index in enumerate(raw_window_indices)
            }
            raw_times = np.empty((len(raw_window_indices),), dtype=float)
            raw_histories = {
                "lnOmega_history": np.empty(
                    (len(raw_window_indices),) + tuple(lnOmega0.shape),
                    dtype=np.asarray(lnOmega0).dtype,
                ),
                "alpha_history": np.empty(
                    (len(raw_window_indices),) + tuple(alpha0.shape),
                    dtype=np.asarray(alpha0).dtype,
                ),
                "beta_history": np.empty(
                    (len(raw_window_indices),) + tuple(beta0.shape),
                    dtype=np.asarray(beta0).dtype,
                ),
                "lnOmega_shift_history": np.empty(
                    (len(raw_window_indices),),
                    dtype=float,
                ),
            }
        if save_observables and pareto_k_names:
            pareto_stride = max(1, int(save_raw_walkers_every_windows))
            pareto_k_window_indices = [
                index
                for index in range(num_times)
                if index == 0 or index == n_windows or index % pareto_stride == 0
            ]
            pareto_k_window_set = set(pareto_k_window_indices)

        observable_window_lookup = {}
        observable_times = None
        pareto_k_window_lookup = {}
        pareto_k_times = None
        if save_observables or save_observable_sums:
            observable_stride = max(1, int(save_observables_every_windows))
            observable_window_indices = [
                index
                for index in range(num_times)
                if index == 0 or index == n_windows or index % observable_stride == 0
            ]
            observable_window_lookup = {
                window_index: observable_index
                for observable_index, window_index in enumerate(observable_window_indices)
            }
            observable_times = np.empty((len(observable_window_indices),), dtype=float)
        if save_observables and pareto_k_names:
            pareto_k_window_indices = sorted(pareto_k_window_set)
            pareto_k_window_lookup = {
                window_index: pareto_index
                for pareto_index, window_index in enumerate(pareto_k_window_indices)
            }
            pareto_k_times = np.empty((len(pareto_k_window_indices),), dtype=float)

        regular_observable_names = tuple(
            name for name in observable_names if name not in PARETO_K_OBSERVABLES
        )
        reduced_shell_pairs = self._reduced_observable_shell_pairs()
        observable_buffers = {name: [] for name in regular_observable_names} if save_observables else {}
        observable_error_buffers = {name: [] for name in regular_observable_names} if save_observables else {}
        pareto_k_buffers = {name: [] for name in pareto_k_names} if save_observables else {}
        pareto_k_error_buffers = {name: [] for name in pareto_k_names} if save_observables else {}
        observable_filter_buffers = {"unhealth_kept_fraction": []} if save_observables and remove_unhealth.enabled else {}
        observable_sum_buffers = {} if save_observable_sums else None

        def record_window(
            index: int,
            t_value,
            lnOmega_value=None,
            alpha_value=None,
            beta_value=None,
            lnOmega_shift_value=0.0,
        ):
            times[index] = float(t_value)
            raw_index = raw_window_lookup.get(index)
            if raw_index is not None:
                raw_times[raw_index] = float(t_value)
                raw_histories["lnOmega_history"][raw_index] = np.asarray(lnOmega_value)
                raw_histories["alpha_history"][raw_index] = np.asarray(alpha_value)
                raw_histories["beta_history"][raw_index] = np.asarray(beta_value)
                raw_histories["lnOmega_shift_history"][raw_index] = float(lnOmega_shift_value)
            observable_index = observable_window_lookup.get(index)
            pareto_k_index = pareto_k_window_lookup.get(index)
            if observable_index is None and pareto_k_index is None:
                return
            if observable_index is not None:
                observable_times[observable_index] = float(t_value)
            if save_observable_sums and observable_index is not None:
                stats = _compute_observable_sum_snapshot(
                    initial_alpha=alpha0,
                    initial_beta=beta0,
                    lnOmega=lnOmega_value,
                    lnOmega_shift=lnOmega_shift_value,
                    alpha=alpha_value,
                    beta=beta_value,
                    observable_names=regular_observable_names,
                    shell_pairs=reduced_shell_pairs,
                    remove_unhealth=remove_unhealth,
                )
                if not observable_sum_buffers:
                    observable_sum_buffers.update({name: [] for name in stats})
                for name, value in stats.items():
                    observable_sum_buffers[name].append(value)
            if save_observables and regular_observable_names and observable_index is not None:
                measurements, errors, filter_info = self._compute_observable_snapshot(
                    observable_names=regular_observable_names,
                    initial_alpha=alpha0,
                    initial_beta=beta0,
                    lnOmega=lnOmega_value,
                    alpha=alpha_value,
                    beta=beta_value,
                    remove_unhealth=remove_unhealth,
                    save_observable_errors=save_observable_errors,
                )
                for name in regular_observable_names:
                    observable_buffers[name].append(measurements[name])
                    observable_error_buffers[name].append(errors[name])
                if observable_filter_buffers:
                    walker_count = max(1, int(filter_info["walker_count"]))
                    observable_filter_buffers["unhealth_kept_fraction"].append(
                        float(filter_info["healthy_count"]) / float(walker_count)
                    )
            if save_observables and pareto_k_names and pareto_k_index is not None:
                pareto_k_times[pareto_k_index] = float(t_value)
                pareto_measurements, pareto_errors, _ = self._compute_observable_snapshot(
                    observable_names=pareto_k_names,
                    initial_alpha=alpha0,
                    initial_beta=beta0,
                    lnOmega=lnOmega_value,
                    alpha=alpha_value,
                    beta=beta_value,
                    remove_unhealth=remove_unhealth,
                )
                for name in pareto_k_names:
                    pareto_k_buffers[name].append(pareto_measurements[name])
                    pareto_k_error_buffers[name].append(pareto_errors[name])

        lnOmega = lnOmega0
        alpha = alpha0
        beta = beta0
        t = rollout_kwargs["t0"]
        lnOmega_shift = 0.0
        key_out = rollout_kwargs["key"]
        progress_every_window = int(progress_every_window)
        multi_device_spec = rollout_kwargs.get("multi_device_spec")
        use_multi_device = bool(multi_device_spec is not None and multi_device_spec.enabled)
        if use_multi_device:
            stepper = MultiDeviceSimulationStepper(
                apply_fn=rollout_kwargs["apply_fn"],
                N_steps=rollout_kwargs["N_steps"],
                apply_neural_gauge_every_steps=rollout_kwargs["apply_neural_gauge_every_steps"],
                sde_max_iter=rollout_kwargs["sde_max_iter"],
                sde_solver=rollout_kwargs["sde_solver"],
                sde_root_rtol=rollout_kwargs["sde_root_rtol"],
                sde_root_atol=rollout_kwargs["sde_root_atol"],
                sde_affine_expm_order=rollout_kwargs["sde_affine_expm_order"],
                sde_affine_expm_substeps=rollout_kwargs[
                    "sde_affine_expm_substeps"
                ],
                sde_newton_damping_steps=rollout_kwargs[
                    "sde_newton_damping_steps"
                ],
                gauge_mode=rollout_kwargs["gauge_mode"],
                neural_gauge_components=rollout_kwargs["neural_gauge_components"],
                spec=multi_device_spec,
            )
            key_devices = split_device_keys(key_out, multi_device_spec)
            lnOmega_sharded = shard_walkers(lnOmega, multi_device_spec)
            alpha_sharded = shard_walkers(alpha, multi_device_spec)
            beta_sharded = shard_walkers(beta, multi_device_spec)

        record_window(0, t, lnOmega, alpha, beta, lnOmega_shift)
        if use_multi_device and not save_raw_walkers and not save_observables and not save_observable_sums:
            rollout = MultiDeviceSimulationRollout(
                apply_fn=rollout_kwargs["apply_fn"],
                N_steps=rollout_kwargs["N_steps"],
                N_windows=n_windows,
                apply_neural_gauge_every_steps=rollout_kwargs["apply_neural_gauge_every_steps"],
                sde_max_iter=rollout_kwargs["sde_max_iter"],
                sde_solver=rollout_kwargs["sde_solver"],
                sde_root_rtol=rollout_kwargs["sde_root_rtol"],
                sde_root_atol=rollout_kwargs["sde_root_atol"],
                sde_affine_expm_order=rollout_kwargs["sde_affine_expm_order"],
                sde_affine_expm_substeps=rollout_kwargs[
                    "sde_affine_expm_substeps"
                ],
                sde_newton_damping_steps=rollout_kwargs[
                    "sde_newton_damping_steps"
                ],
                gauge_mode=rollout_kwargs["gauge_mode"],
                neural_gauge_components=rollout_kwargs["neural_gauge_components"],
                spec=multi_device_spec,
            )
            key_devices, lnOmega_sharded, alpha_sharded, beta_sharded, t_end_devices = rollout(
                key_devices,
                rollout_kwargs["params"],
                lnOmega_sharded,
                alpha_sharded,
                beta_sharded,
                rollout_kwargs["U"],
                rollout_kwargs["gamma"],
                rollout_kwargs["F"],
                rollout_kwargs["Delta"],
                rollout_kwargs["dt"],
                t,
                rollout_kwargs["gauge_weight"],
                rollout_kwargs["n0"],
                rollout_kwargs["J"],
                rollout_kwargs["analytic_target_time"],
            )
            t_end = float(np.asarray(t_end_devices)[0])
            window_dt = float(rollout_kwargs["dt"]) * int(rollout_kwargs["N_steps"])
            times[:] = float(rollout_kwargs["t0"]) + window_dt * np.arange(num_times, dtype=float)
            times[-1] = t_end
            if progress_every_window > 0:
                print(
                    (
                        f"simulation window {n_windows}/{n_windows} complete "
                        f"| t = {t_end:.8g}"
                    ),
                    flush=True,
                )
            return {
                "times": times,
                "raw_times": raw_times,
                **raw_histories,
                "observable_arrays": None,
                "observable_sum_arrays": None,
            }

        for iwin in range(n_windows):
            window_index = iwin + 1
            save_raw_this_window = bool(save_raw_walkers and window_index in raw_window_lookup)
            save_observable_this_window = bool(
                (save_observables or save_observable_sums)
                and window_index in observable_window_lookup
            )
            save_pareto_k_this_window = bool(
                save_observables
                and pareto_k_names
                and window_index in pareto_k_window_lookup
            )
            if use_multi_device:
                sub = jax.vmap(random.fold_in, in_axes=(0, None))(key_devices, iwin)
                key_devices, lnOmega_sharded, alpha_sharded, beta_sharded, aux = stepper(
                    sub,
                    rollout_kwargs["params"],
                    lnOmega_sharded,
                    alpha_sharded,
                    beta_sharded,
                    rollout_kwargs["U"],
                    rollout_kwargs["gamma"],
                    rollout_kwargs["F"],
                    rollout_kwargs["Delta"],
                    rollout_kwargs["dt"],
                    t,
                    rollout_kwargs["gauge_weight"],
                    rollout_kwargs["n0"],
                    rollout_kwargs["J"],
                    rollout_kwargs["analytic_target_time"],
                )
                t = float(np.asarray(aux["t_end"])[0])
                if save_raw_this_window:
                    lnOmega = unshard_walkers(lnOmega_sharded)
                    alpha = unshard_walkers(alpha_sharded)
                    beta = unshard_walkers(beta_sharded)
                elif save_observable_this_window or save_pareto_k_this_window:
                    # Keep walker data on device for compact observable reductions;
                    # avoid host-copying the full walker cloud at every window.
                    lnOmega = unshard_walkers_device(lnOmega_sharded)
                    alpha = unshard_walkers_device(alpha_sharded)
                    beta = unshard_walkers_device(beta_sharded)
                else:
                    lnOmega = alpha = beta = None
            else:
                sub = random.fold_in(key_out, iwin)
                key_out, lnOmega, alpha, beta, aux = run_one_window_simulation_rollout(
                    key=sub,
                    apply_fn=rollout_kwargs["apply_fn"],
                    params=rollout_kwargs["params"],
                    lnOmega=lnOmega,
                    alpha=alpha,
                    beta=beta,
                    U=rollout_kwargs["U"],
                    gamma=rollout_kwargs["gamma"],
                    F=rollout_kwargs["F"],
                    Delta=rollout_kwargs["Delta"],
                    dt=rollout_kwargs["dt"],
                    N_steps=rollout_kwargs["N_steps"],
                    t0=t,
                    apply_neural_gauge_every_steps=rollout_kwargs["apply_neural_gauge_every_steps"],
                    gauge_weight=rollout_kwargs["gauge_weight"],
                    n0=rollout_kwargs["n0"],
                    sde_max_iter=rollout_kwargs["sde_max_iter"],
                    sde_solver=rollout_kwargs["sde_solver"],
                    sde_root_rtol=rollout_kwargs["sde_root_rtol"],
                    sde_root_atol=rollout_kwargs["sde_root_atol"],
                    sde_affine_expm_order=rollout_kwargs[
                        "sde_affine_expm_order"
                    ],
                    sde_affine_expm_substeps=rollout_kwargs[
                        "sde_affine_expm_substeps"
                    ],
                    sde_newton_damping_steps=rollout_kwargs[
                        "sde_newton_damping_steps"
                    ],
                    J=rollout_kwargs["J"],
                    gauge_mode=rollout_kwargs["gauge_mode"],
                    neural_gauge_components=rollout_kwargs["neural_gauge_components"],
                    analytic_target_time=rollout_kwargs["analytic_target_time"],
                )
                t = float(aux["t_end"])
            if use_multi_device:
                center_shift = np.asarray(aux.get("lnOmega_center_shift", 0.0))
                lnOmega_shift = float(center_shift.reshape(-1)[0]) + float(lnOmega_shift)
            else:
                lnOmega_shift = float(lnOmega_shift) + float(np.asarray(aux.get("lnOmega_center_shift", 0.0)))
            record_window(window_index, t, lnOmega, alpha, beta, lnOmega_shift)
            if progress_every_window > 0 and (
                iwin == 0
                or (iwin + 1) % progress_every_window == 0
                or (iwin + 1) == n_windows
            ):
                print(
                    (
                        f"simulation window {iwin + 1}/{n_windows} complete "
                        f"| t = {t:.8g}"
                    ),
                    flush=True,
                )

        observable_arrays = None
        if save_observables:
            observable_arrays = {"times": observable_times}
            for name in regular_observable_names:
                observable_arrays[name] = np.asarray(observable_buffers[name])
                observable_arrays[f"{name}_err"] = np.asarray(observable_error_buffers[name])
            if pareto_k_names:
                observable_arrays["pareto-k_times"] = np.asarray(pareto_k_times)
                for name in pareto_k_names:
                    observable_arrays[name] = np.asarray(pareto_k_buffers[name])
                    observable_arrays[f"{name}_err"] = np.asarray(pareto_k_error_buffers[name])
            for name, values in observable_filter_buffers.items():
                observable_arrays[name] = np.asarray(values, dtype=float)
        observable_sum_arrays = None
        if save_observable_sums:
            observable_sum_arrays = _stack_observable_sum_history(observable_sum_buffers)
            observable_sum_arrays["times"] = observable_times

        return {
            "times": times,
            "raw_times": raw_times,
            **raw_histories,
            "observable_arrays": observable_arrays,
            "observable_sum_arrays": observable_sum_arrays,
        }

    def _run_ed_benchmark(
        self,
        times: np.ndarray,
        physics: SimulationPhysics,
        ed_cfg: EDBenchmarkConfig,
        save_policy: ArchiveSavePolicy,
    ):
        if not ed_cfg.enabled:
            return None

        print(
            (
                f"running ED benchmark with n_cut = {ed_cfg.n_cut} "
                f"on {len(times)} ED reference time point(s)"
            ),
            flush=True,
        )
        initial_alpha = self.lattice.coherent_product_amplitudes(n0=physics.n0)
        rho0 = self.lattice.coherent_product_density_matrix(
            n_cut=int(ed_cfg.n_cut),
            amplitudes=initial_alpha,
        )
        absolute_times = np.asarray(times, dtype=float)
        elapsed_times = absolute_times - absolute_times[0]
        ed_trajectory = self.lattice.time_evolution_observables(
            times=elapsed_times,
            n_cut=int(ed_cfg.n_cut),
            rho0=rho0,
            initial_beta=np.conjugate(initial_alpha),
            U=physics.U,
            gamma=physics.gamma,
            Delta=physics.Delta,
            F=physics.F,
        )
        observable_names = [name for name in ed_trajectory[0].keys() if name != "rho"]
        arrays = {"times": absolute_times}
        for name in observable_names:
            arrays[name] = np.asarray([snapshot[name] for snapshot in ed_trajectory])
        stored_arrays = prepare_arrays_for_storage(arrays, precision=save_policy.precision)
        print(f"saving ED benchmark data to {self.benchmark_path}", flush=True)
        save_array_archive(
            self.benchmark_path,
            stored_arrays,
            archive_format=save_policy.archive_format,
            compressed=save_policy.compressed,
            precision="runtime",
        )
        return {
            "path": self.benchmark_path,
            "n_cut": int(ed_cfg.n_cut),
            "num_times": int(len(times)),
            "save_policy": {
                "format": save_policy.archive_format,
                "compressed": save_policy.compressed,
                "precision": save_policy.precision,
            },
            "saved_arrays": sorted(arrays.keys()),
            "saved_dtypes": {name: str(value.dtype) for name, value in stored_arrays.items()},
        }

    def run(self):
        self._prepare_io()
        sim_cfg = self.config["simulation"]
        keygen = KeyGenerator(sim_cfg["seed"])
        physics = self._resolve_physics()
        ed_cfg = self._resolve_ed_config()
        save_raw_walkers = bool(sim_cfg.get("save_raw_walkers", True))
        save_raw_walkers_every_windows = max(
            1,
            int(sim_cfg.get("save_raw_walkers_every_windows", 1)),
        )
        save_observables = bool(sim_cfg.get("save_observables", False))
        save_observables_every_windows = max(
            1,
            int(sim_cfg.get("save_observables_every_windows", 1)),
        )
        save_observable_errors = bool(sim_cfg.get("save_observable_errors", True))
        observable_names = self._resolve_observable_names() if save_observables else ()
        remove_unhealth = self._resolve_remove_unhealth_config()
        results_save_policy = self._resolve_results_save_policy()
        observables_save_policy = self._resolve_observables_save_policy()
        ed_save_policy = self._resolve_ed_save_policy()
        if save_raw_walkers:
            require_archive_backend(results_save_policy.archive_format)
        if save_observables:
            require_archive_backend(observables_save_policy.archive_format)
        require_archive_backend(ed_save_policy.archive_format)
        progress_every_window = int(
            sim_cfg.get(
                "progress_every_window",
                1 if int(sim_cfg["N_windows"]) <= 20 else max(1, int(sim_cfg["N_windows"]) // 10),
            )
        )
        rollout_end_time = (
            float(sim_cfg["t0"])
            + float(sim_cfg["dt"]) * int(sim_cfg["N_steps"]) * int(sim_cfg["N_windows"])
        )
        analytic_target_time = self._resolve_analytical_target_time(rollout_end_time)
        sde_solver = normalize_sde_solver(sim_cfg.get("sde_solver"))
        sde_max_iter = int(sim_cfg.get("sde_max_iter", 4))
        sde_root_rtol = float(sim_cfg["sde_root_rtol"])
        sde_root_atol = float(sim_cfg["sde_root_atol"])
        sde_affine_expm_order = int(sim_cfg["sde_affine_expm_order"])
        sde_affine_expm_substeps = int(sim_cfg["sde_affine_expm_substeps"])
        sde_newton_damping_steps = int(sim_cfg["sde_newton_damping_steps"])
        sde_control_metadata = sde_solver_control_metadata(
            sde_solver,
            max_iterations=sde_max_iter,
            root_rtol=sde_root_rtol,
            root_atol=sde_root_atol,
            affine_expm_order=sde_affine_expm_order,
            affine_expm_substeps=sde_affine_expm_substeps,
            newton_damping_steps=sde_newton_damping_steps,
        )
        analytical_gauge_description = self._resolve_analytical_gauge_description(analytic_target_time)
        simulation_t_end = format_end_time(
            t0=float(sim_cfg["t0"]),
            dt=float(sim_cfg["dt"]),
            n_steps=int(sim_cfg["N_steps"]),
            n_windows=int(sim_cfg["N_windows"]),
        )
        runtime_real_dtype = str(jnp.asarray(0.0, dtype=DTYPE).dtype)
        runtime_complex_dtype = str(jnp.asarray(0.0 + 0.0j, dtype=CDTYPE).dtype)
        nn_dtype = str(jnp.asarray(0.0, dtype=NNDTYPE).dtype)

        print(f"simulation lattice sites = {self.lattice.num_site}", flush=True)
        print(
            (
                "simulation data precision = "
                f"real {runtime_real_dtype}, complex {runtime_complex_dtype}, nn {nn_dtype}"
            ),
            flush=True,
        )
        if sim_cfg["gauge_mode"] in NEURAL_GAUGE_MODES:
            for line in self.model.config_summary_lines():
                print(line, flush=True)
        print(
            f"simulation with N_steps = {int(sim_cfg['N_steps'])}, N_windows = {int(sim_cfg['N_windows'])}",
            flush=True,
        )
        solver_iteration_suffix = (
            f" (max iterations={sde_max_iter})"
            if "max_iterations" in sde_control_metadata["active_controls"]
            else ""
        )
        print(f"simulation SDE solver = {sde_solver}{solver_iteration_suffix}", flush=True)
        print(
            "simulation SDE solver controls = "
            f"{format_sde_solver_controls(sde_control_metadata)}",
            flush=True,
        )
        apply_neural_gauge_every_steps = int(sim_cfg.get("apply_neural_gauge_every_steps", 0))
        if apply_neural_gauge_every_steps > 0:
            print(
                f"simulation neural gauge refresh = every {apply_neural_gauge_every_steps} SDE step(s)",
                flush=True,
            )
        else:
            print("simulation neural gauge refresh = window start only", flush=True)
        print(f"simulation up to time {simulation_t_end}", flush=True)
        total_num_walker = int(sim_cfg["num_walker"])
        batch_sizes = _resolve_walker_batch_sizes(
            total_num_walker,
            sim_cfg.get("walker_batches", {}),
        )
        chunked_ensemble = len(batch_sizes) > 1
        pareto_k_observable_names = [
            name for name in observable_names if name in PARETO_K_OBSERVABLES
        ]
        if chunked_ensemble and pareto_k_observable_names:
            raise ValueError(
                "simulation Pareto-k observables require the full walker cloud at each "
                "saved time and cannot be merged from chunked compact observable sums. "
                "Use a single walker batch or save raw walkers and postprocess."
            )
        active_num_walker = int(batch_sizes[0])
        print(f"simulation walkers = {total_num_walker}", flush=True)
        if chunked_ensemble:
            print(
                (
                    "chunked walker ensemble = enabled "
                    f"with {len(batch_sizes)} batch(es), "
                    f"batch sizes={batch_sizes}"
                ),
                flush=True,
            )
        else:
            print("chunked walker ensemble = disabled", flush=True)
        multi_device_spec = resolve_multi_device_spec(
            sim_cfg,
            active_num_walker,
            purpose="simulation",
        )
        if multi_device_spec.enabled:
            print(
                (
                    "multi-device simulation = enabled "
                    f"with {multi_device_spec.num_devices} device(s), "
                    f"{multi_device_spec.walkers_per_device} walker(s)/device"
                ),
                flush=True,
            )
        else:
            print("multi-device simulation = disabled", flush=True)
        print(
            (
                "raw save policy: "
                f"enabled={save_raw_walkers}, "
                f"every={save_raw_walkers_every_windows} window(s), "
                f"format={results_save_policy.archive_format}, "
                f"compressed={results_save_policy.compressed}, "
                f"precision={results_save_policy.precision}"
            ),
            flush=True,
        )
        if save_observables:
            print(
                (
                    "observable save policy: "
                    f"format={observables_save_policy.archive_format}, "
                    f"compressed={observables_save_policy.compressed}, "
                    f"precision={observables_save_policy.precision}, "
                    f"every={save_observables_every_windows} window(s), "
                    f"errors={save_observable_errors}, "
                    f"observables={list(observable_names)}"
                ),
                flush=True,
            )
            print(
                (
                    "observable unhealthy-walker filter: "
                    f"enabled={remove_unhealth.enabled}, "
                    f"logstd={remove_unhealth.logstd:g}"
                ),
                flush=True,
            )
            if pareto_k_observable_names:
                pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials = (
                    self._pareto_k_observable_selector()
                )
                if pareto_k_monomials:
                    selector_text = "monomials=" + ",".join(
                        f"({m},{n})" for m, n in pareto_k_monomials
                    )
                else:
                    selector_text = (
                        f"pareto_k_applied_quantities={pareto_k_applied_quantities}, "
                        f"mode={pareto_k_applied_quantities_mode}"
                    )
                print(
                    (
                        "observable Pareto-k diagnostics: "
                        f"enabled for {pareto_k_observable_names}, "
                        f"{selector_text}, "
                        "channels=local onsite, "
                        f"every={save_raw_walkers_every_windows} raw-save window(s)"
                    ),
                    flush=True,
                )

        params = None
        apply_fn = None
        neural_gauge_components = normalize_neural_gauge_components(
            sim_cfg.get(
                "neural_gauge_components",
                self.config["training"].get("neural_gauge_components", "both"),
            )
        )
        if sim_cfg["gauge_mode"] in NEURAL_GAUGE_MODES:
            print(f"model parameters loaded from {self.params_path}", flush=True)
            print(f"neural gauge components = {neural_gauge_components}", flush=True)
            params = self._load_neural_parameters()
            print(f"model parameters = {count_params(params)}", flush=True)
            apply_fn = self.model.module.apply
        elif analytical_gauge_description is not None:
            print(f"analytical gauge mode = {analytical_gauge_description['mode']}", flush=True)
            print(f"source = {analytical_gauge_description['location']}", flush=True)
            print(f"analytical gauge target time = {analytic_target_time:.8g}", flush=True)

        if multi_device_spec.enabled and len(set(batch_sizes)) > 1:
            raise ValueError(
                "chunked multi-device simulation currently requires equal batch sizes; "
                "choose num_walker divisible by num_walker_per_batch or use one device"
            )

        print(
            f"starting stochastic rollout with progress update every {progress_every_window} window(s)",
            flush=True,
        )
        batch_results = []
        batch_observable_sum_arrays = []
        results = None
        for batch_index, batch_num_walker in enumerate(batch_sizes):
            print(
                (
                    f"initializing phase-space walkers for batch "
                    f"{batch_index + 1}/{len(batch_sizes)} "
                    f"with {batch_num_walker} walker(s)"
                ),
                flush=True,
            )
            lnOmega0, alpha0, beta0 = self.lattice.initialize_phase_space(
                num_walker=int(batch_num_walker),
                n0=physics.n0,
            )
            rollout_kwargs = dict(
                key=keygen.next(fold_in_value=1 + batch_index),
                apply_fn=apply_fn,
                params=params,
                lnOmega0=lnOmega0,
                alpha0=alpha0,
                beta0=beta0,
                U=DTYPE(physics.U),
                gamma=DTYPE(physics.gamma),
                F=CDTYPE(physics.F),
                Delta=DTYPE(physics.Delta),
                dt=DTYPE(sim_cfg["dt"]),
                N_steps=int(sim_cfg["N_steps"]),
                N_windows=int(sim_cfg["N_windows"]),
                t0=DTYPE(sim_cfg["t0"]),
                apply_neural_gauge_every_steps=int(sim_cfg.get("apply_neural_gauge_every_steps", 0)),
                gauge_weight=DTYPE(sim_cfg.get("gauge_scale", 1.0)),
                n0=DTYPE(physics.n0),
                sde_max_iter=int(sim_cfg.get("sde_max_iter", 4)),
                sde_solver=sde_solver,
                sde_root_rtol=DTYPE(sde_root_rtol),
                sde_root_atol=DTYPE(sde_root_atol),
                sde_affine_expm_order=sde_affine_expm_order,
                sde_affine_expm_substeps=sde_affine_expm_substeps,
                sde_newton_damping_steps=sde_newton_damping_steps,
                J=self.lattice.hopping_operator(),
                gauge_mode=sim_cfg["gauge_mode"],
                neural_gauge_components=neural_gauge_components,
                analytic_target_time=DTYPE(analytic_target_time),
                multi_device_spec=multi_device_spec,
            )
            batch_result = self._run_stochastic_rollout(
                rollout_kwargs=rollout_kwargs,
                save_raw_walkers=save_raw_walkers,
                save_raw_walkers_every_windows=save_raw_walkers_every_windows,
                save_observables=save_observables and not chunked_ensemble,
                save_observables_every_windows=save_observables_every_windows,
                save_observable_errors=save_observable_errors,
                observable_names=observable_names,
                save_observable_sums=save_observables and chunked_ensemble,
                remove_unhealth=remove_unhealth,
                progress_every_window=progress_every_window,
            )
            if chunked_ensemble:
                if save_observables:
                    batch_observable_sum_arrays.append(batch_result["observable_sum_arrays"])
                if not save_raw_walkers:
                    # Keep only small bookkeeping for chunked compact-observable
                    # runs.  The per-batch walker arrays are intentionally not
                    # retained unless the user explicitly requested raw saves.
                    batch_result = {
                        "times": np.asarray(batch_result["times"]),
                        "raw_times": None,
                    }
                batch_results.append(batch_result)
            else:
                results = batch_result

        if chunked_ensemble:
            results = {
                "times": np.asarray(batch_results[0]["times"]),
                "raw_times": np.asarray(batch_results[0]["raw_times"]) if save_raw_walkers else None,
                "observable_arrays": None,
                "observable_sum_arrays": None,
            }
            if save_raw_walkers:
                for other in batch_results[1:]:
                    if not np.allclose(np.asarray(other["raw_times"]), np.asarray(results["raw_times"])):
                        raise ValueError("chunked simulation produced inconsistent raw save times")
                (
                    results["lnOmega_history"],
                    results["lnOmega_shift_history"],
                ) = _merge_chunked_log_weight_histories(batch_results)
                results["alpha_history"] = np.concatenate(
                    [np.asarray(batch_result["alpha_history"]) for batch_result in batch_results],
                    axis=1,
                )
                results["beta_history"] = np.concatenate(
                    [np.asarray(batch_result["beta_history"]) for batch_result in batch_results],
                    axis=1,
                )
            if save_observables:
                observable_times = np.asarray(batch_observable_sum_arrays[0]["times"])
                for stats in batch_observable_sum_arrays[1:]:
                    if not np.allclose(np.asarray(stats["times"]), observable_times):
                        raise ValueError("chunked simulation produced inconsistent observable save times")
                observable_stat_names = [
                    name for name in batch_observable_sum_arrays[0] if name != "times"
                ]
                batch_stats = {
                    name: np.stack([stats[name] for stats in batch_observable_sum_arrays], axis=0)
                    for name in observable_stat_names
                }
                shell_pairs = self._reduced_observable_shell_pairs()
                observable_values = _finalize_observable_sums(
                    batch_stats,
                    observable_names,
                    shell_pairs=shell_pairs,
                )
                if save_observable_errors:
                    observable_errors = _observable_jackknife_errors(
                        batch_stats,
                        observable_names,
                        shell_pairs=shell_pairs,
                    )
                else:
                    observable_errors = {
                        name: np.zeros_like(value)
                        for name, value in observable_values.items()
                    }
                observable_arrays = {"times": observable_times}
                for name in observable_names:
                    observable_arrays[name] = np.asarray(observable_values[name])
                    observable_arrays[f"{name}_err"] = np.asarray(observable_errors[name])
                if remove_unhealth.enabled and "healthy_count" in batch_stats:
                    healthy = np.sum(np.asarray(batch_stats["healthy_count"], dtype=float), axis=0)
                    total = np.sum(np.asarray(batch_stats["walker_count"], dtype=float), axis=0)
                    observable_arrays["unhealth_kept_fraction"] = healthy / np.maximum(total, 1.0)
                results["observable_arrays"] = observable_arrays
            batch_results = []
            batch_observable_sum_arrays = []

        print("stochastic rollout complete", flush=True)
        arrays_to_save = {}
        stored_arrays = {}
        results_path = None
        if save_raw_walkers:
            print(f"saving raw simulation data to {self.results_path}", flush=True)
            arrays_to_save = {
                "times": np.asarray(results["raw_times"]),
                "lnOmega_history": np.asarray(results["lnOmega_history"]),
                "lnOmega_shift_history": np.asarray(
                    results["lnOmega_shift_history"],
                    dtype=float,
                ),
                "alpha_history": np.asarray(results["alpha_history"]),
                "beta_history": np.asarray(results["beta_history"]),
            }
            stored_arrays = prepare_arrays_for_storage(arrays_to_save, precision=results_save_policy.precision)
            save_array_archive(
                self.results_path,
                stored_arrays,
                archive_format=results_save_policy.archive_format,
                compressed=results_save_policy.compressed,
                precision="runtime",
            )
            results_path = self.results_path
        else:
            print("raw walker data saving disabled", flush=True)

        observables_path = None
        observables_arrays = results.get("observable_arrays")
        stored_observable_arrays = {}
        if save_observables:
            print(f"saving simulation observables to {self.observables_path}", flush=True)
            stored_observable_arrays = prepare_arrays_for_storage(
                observables_arrays,
                precision=observables_save_policy.precision,
            )
            save_array_archive(
                self.observables_path,
                stored_observable_arrays,
                archive_format=observables_save_policy.archive_format,
                compressed=observables_save_policy.compressed,
                precision="runtime",
            )
            observables_path = self.observables_path

        ed_times = None
        ed_n_steps = None
        ed_n_windows = None
        ed_metadata = None
        if ed_cfg.enabled:
            ed_times, ed_n_steps, ed_n_windows = self._build_ed_times()
            print(
                (
                    "ED reference grid uses the same dt with "
                    f"N_steps = {ed_n_steps} and N_windows = {ed_n_windows}"
                ),
                flush=True,
            )
            ed_metadata = self._run_ed_benchmark(
                ed_times,
                physics,
                ed_cfg,
                ed_save_policy,
            )
            if ed_metadata is not None:
                ed_metadata["time_grid"] = {
                    "dt": float(sim_cfg["dt"]),
                    "t0": float(sim_cfg["t0"]),
                    "N_steps": int(ed_n_steps),
                    "N_windows": int(ed_n_windows),
                }

        save_json(
            {
                "config_path": self.config_path,
                "config_copy_path": self.config_copy_path,
                "gauge_mode": sim_cfg["gauge_mode"],
                "gauge_tag": self.gauge_tag,
                "neural_gauge_components": neural_gauge_components,
                "sde_solver": sde_control_metadata,
                "analytic_target_time": analytic_target_time,
                "params_path": self.params_path if sim_cfg["gauge_mode"] in NEURAL_GAUGE_MODES else None,
                "num_walker": int(sim_cfg["num_walker"]),
                "num_site": int(self.lattice.num_site),
                "walker_batches": {
                    "enabled": bool(chunked_ensemble),
                    "num_batches": int(len(batch_sizes)),
                    "batch_sizes": [int(value) for value in batch_sizes],
                    "total_num_walker": int(total_num_walker),
                    "merge_rule": (
                        "global weighted numerator/denominator sums with restored "
                        "per-batch log-weight center shifts"
                    ),
                    "observable_errors": "delete-one-batch jackknife" if chunked_ensemble else "walker delta-method",
                },
                "multi_device": {
                    "enabled": bool(multi_device_spec.enabled),
                    "num_devices": int(multi_device_spec.num_devices),
                    "walkers_per_device": int(multi_device_spec.walkers_per_device),
                    "axis_name": multi_device_spec.axis_name,
                },
                "times": np.asarray(results["times"]).tolist(),
                "raw_times": (
                    np.asarray(results["raw_times"]).tolist()
                    if results.get("raw_times") is not None
                    else []
                ),
                "raw_results_path": results_path,
                "observables_path": observables_path,
                "saved_arrays": sorted(arrays_to_save),
                "save_policy": {
                    "enabled": save_raw_walkers,
                    "format": results_save_policy.archive_format,
                    "compressed": results_save_policy.compressed,
                    "precision": results_save_policy.precision,
                    "save_every_windows": save_raw_walkers_every_windows,
                    "num_saved_times": int(len(results["raw_times"])) if save_raw_walkers else 0,
                    "num_dense_times": int(len(results["times"])),
                },
                "saved_dtypes": {name: str(value.dtype) for name, value in stored_arrays.items()},
                "log_weight_storage": {
                    "convention": (
                        "absolute lnOmega equals lnOmega_history plus "
                        "lnOmega_shift_history at each saved time"
                    ),
                    "shared_shift_restored": False,
                },
                "history_shapes": {
                    name: list(stored_arrays[name].shape)
                    for name in (
                        "lnOmega_history",
                        "lnOmega_shift_history",
                        "alpha_history",
                        "beta_history",
                    )
                    if name in stored_arrays
                },
                "observables": {
                    "enabled": save_observables,
                    "names": list(observable_names),
                    "remove_unhealth": {
                        "enabled": bool(remove_unhealth.enabled),
                        "logstd": float(remove_unhealth.logstd),
                        "criterion": (
                            "remove nonfinite walkers and walkers with "
                            "log1p(max_i |hatOmega alpha_i beta_i|) above "
                            "median + logstd * 1.4826 * MAD"
                        ),
                    },
                    "saved_arrays": sorted(observables_arrays.keys()) if observables_arrays else [],
                    "save_policy": {
                        "format": observables_save_policy.archive_format,
                        "compressed": observables_save_policy.compressed,
                        "precision": observables_save_policy.precision,
                        "save_every_windows": save_observables_every_windows,
                        "save_errors": save_observable_errors,
                        "num_saved_times": (
                            int(len(observables_arrays["times"])) if observables_arrays else 0
                        ),
                    },
                    "saved_dtypes": {
                        name: str(value.dtype) for name, value in stored_observable_arrays.items()
                    },
                    "shapes": {
                        name: list(value.shape) for name, value in stored_observable_arrays.items()
                    },
                    "pareto_k": self._pareto_k_summary_block(
                        pareto_k_observable_names,
                        save_raw_walkers_every_windows,
                    ),
                },
                "physics": {
                    "n0": physics.n0,
                    "U": physics.U,
                    "gamma": physics.gamma,
                    "F_real": float(np.real(physics.F)),
                    "F_imag": float(np.imag(physics.F)),
                    "Delta": physics.Delta,
                },
                "analytical_gauge": analytical_gauge_description,
                "ed_benchmark": ed_metadata,
            },
            self.metadata_path,
        )
        print(f"saving simulation metadata to {self.metadata_path}", flush=True)

        return SimulationArtifacts(
            results={
                "gauge_tag": self.gauge_tag,
                "times": np.asarray(results["times"]),
                "history_shapes": {
                    name: np.asarray(results[name]).shape
                    for name in ("lnOmega_history", "alpha_history", "beta_history")
                    if name in results
                },
                "observable_shapes": {
                    name: value.shape for name, value in stored_observable_arrays.items()
                },
            },
            results_path=results_path,
            observables_path=observables_path,
            metadata_path=self.metadata_path,
            benchmark_path=self.benchmark_path if ed_metadata is not None else None,
        )


__all__ = [
    "EDBenchmarkConfig",
    "GaugeSimulator",
    "RemoveUnhealthConfig",
    "SimulationArtifacts",
    "SimulationPhysics",
]
