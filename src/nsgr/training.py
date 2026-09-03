import copy
import time
from dataclasses import dataclass

from .dynamics_kernel import (
    compute_grads_all_windows,
    normalize_applied_quantities,
    normalize_applied_quantities_mode,
    run_simulation_windows,
    var_complex_to_real,
)
from .lattice import Lattice
from .lib_preinclude import *
from .model import Model
from .multi_device import (
    MultiDeviceGradientComputer,
    MultiDeviceSimulationHistoryRollout,
    resolve_multi_device_spec,
    shard_walkers,
    split_device_keys,
    unshard_walker_history,
)
from .projected_residual import (
    DEFAULT_RESIDUAL_GMM_TRACE_MODE,
    normalize_residual_gmm_trace_mode,
    projected_residual_channel_labels,
    projected_residual_objective_channel_count,
    projected_residual_term_names,
)
from .utility import (
    KeyGenerator,
    LOSS_TERM_TRACE_THRESHOLD,
    PARETO_K_MONOMIAL_SPECS,
    PARETO_K_MONOMIAL_TERMS,
    compute_l2_penalty,
    count_params,
    expand_loss_ema_terms,
    format_end_time,
    format_sde_solver_controls,
    is_auto_monomial_selector,
    load_config,
    make_lr_schedule,
    normalize_neural_gauge_components,
    normalize_sde_solver,
    normalize_monomial_pairs,
    onsite_monomials_in_operator_equations,
    OPERATOR_MOMENT_DEFAULT_ORDER,
    OPERATOR_MOMENT_MAX_ORDER,
    OPERATOR_MOMENT_SPECS,
    prepare_output_dirs,
    resolve_checkpoint_params_path,
    save_json,
    save_npz,
    save_parameters,
    sde_solver_control_metadata,
    selected_monomial_specs,
    selected_monomial_terms,
    to_scalar_float,
    tree_has_nonfinite,
)
from .training_plot import plot_training_history

# Calibrate the covariance geometry before the policy has moved far, then hold
# that geometry fixed so later policies cannot lower the normalized objective
# by progressively inflating residual variance. A newly visited window
# bootstraps from its current site-averaged estimate, so this fixed decay needs
# neither a reference covariance nor bias correction.
RESIDUAL_GMM_COVARIANCE_EMA_DECAY = 0.99
RESIDUAL_GMM_COVARIANCE_UPDATE_EPOCHS = 200

TRAINING_HISTORY_KEYS = (
    "epoch",
    "loss",
    "loss_pareto_k",
    "pareto_k_mean",
    "pareto_k_max",
    "pareto_k_worst",
    "pareto_k_warning_fraction",
    "pareto_k_bad_fraction",
    "loss_gauge",
    "loss_gauge_drift",
    "loss_gauge_diffusion",
    "loss_ess",
    "loss_ess_weighted",
    "log_weight_spread_mean",
    "log_weight_spread_max",
    "log_weight_spread_total",
    "ess_ratio_min",
    "ess_ratio_end",
    "loss_residual_gmm",
    "loss_residual_gmm_time",
    "loss_residual_gmm_raw",
    "residual_gmm_z_mean",
    "residual_gmm_z_max",
    "residual_gmm_z_worst",
    "residual_gmm_radius_mean",
    "residual_gmm_radius_max",
    "residual_gmm_radius_worst",
    "residual_gmm_warning_fraction",
    "residual_gmm_bad_fraction",
    "loss_L2",
    "grads_norm",
    "epoch_time_sec",
    "lr",
    "loss_pareto_k_ema_scale",
    "loss_residual_gmm_ema_scale",
    "loss_pareto_k_normalized",
    "loss_residual_gmm_normalized",
    "loss_pareto_k_raw",
    *PARETO_K_MONOMIAL_TERMS,
    *(f"{term}_ema_scale" for term in PARETO_K_MONOMIAL_TERMS),
    *(f"{term}_normalized" for term in PARETO_K_MONOMIAL_TERMS),
)

PARETO_K_WORST_HISTORY_KEYS = ("pareto_k_worst",)
PARETO_K_WORST_TO_MAX_HISTORY_KEYS = {"pareto_k_worst": "pareto_k_max"}

LOSS_EMA_TERM_SPECS = {
    "loss_residual_gmm": "loss_residual_gmm_prefactor",
    "loss_pareto_k": "loss_pareto_k_prefactor",
}
LOSS_EMA_ALLOWED_TERMS = (
    tuple(LOSS_EMA_TERM_SPECS)
    + PARETO_K_MONOMIAL_TERMS
)


def _initialize_residual_gmm_covariance_ema(
    leading_shape: tuple[int, ...],
    channel_dimension: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a lagged covariance bank and its per-entry initialization mask."""

    shape = tuple(int(size) for size in leading_shape)
    dimension = int(channel_dimension)
    if any(size <= 0 for size in shape):
        raise ValueError(
            "residual covariance EMA leading dimensions must be positive; "
            f"received {shape}"
        )
    if dimension <= 0:
        raise ValueError(
            "residual covariance EMA channel dimension must be positive; "
            f"received {dimension}"
        )
    return (
        jnp.zeros(shape + (dimension, dimension), dtype=DTYPE),
        jnp.zeros(shape, dtype=jnp.bool_),
    )


def _update_residual_gmm_covariance_ema(
    covariance_bank: jnp.ndarray,
    initialized: jnp.ndarray,
    covariance_estimates: jnp.ndarray,
    active_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Commit finite shared covariance estimates after an accepted update."""

    bank = jax.lax.stop_gradient(jnp.asarray(covariance_bank, dtype=DTYPE))
    mask = jax.lax.stop_gradient(jnp.asarray(initialized, dtype=jnp.bool_))
    estimates = jax.lax.stop_gradient(jnp.asarray(covariance_estimates, dtype=DTYPE))
    weights = jnp.asarray(active_weights, dtype=DTYPE)

    if bank.ndim < 2 or bank.shape[-2] != bank.shape[-1]:
        raise ValueError(
            "residual covariance EMA bank must end in a square matrix; "
            f"received shape {bank.shape}"
        )
    if estimates.shape != bank.shape:
        raise ValueError(
            "residual covariance estimates must match the EMA bank shape; "
            f"received estimates {estimates.shape} and bank {bank.shape}"
        )
    leading_shape = bank.shape[:-2]
    if mask.shape != leading_shape:
        raise ValueError(
            "residual covariance EMA initialization mask must match the bank's "
            f"leading shape {leading_shape}; received {mask.shape}"
        )
    if weights.shape != leading_shape:
        raise ValueError(
            "residual covariance EMA update weights must match the bank's "
            f"leading shape {leading_shape}; received {weights.shape}"
        )

    finite = jnp.all(jnp.isfinite(estimates), axis=(-2, -1))
    active = (weights > jnp.asarray(0.0, dtype=DTYPE)) & finite
    decay = jnp.asarray(RESIDUAL_GMM_COVARIANCE_EMA_DECAY, dtype=DTYPE)
    updated = jnp.where(
        mask[..., None, None],
        decay * bank + (jnp.asarray(1.0, dtype=DTYPE) - decay) * estimates,
        estimates,
    )
    updated = DTYPE(0.5) * (updated + jnp.swapaxes(updated, -1, -2))
    bank = jnp.where(active[..., None, None], updated, bank)
    mask = mask | active
    return jax.lax.stop_gradient(bank), jax.lax.stop_gradient(mask)


def _loss_ema_config(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(train_cfg.get("EMA", {}))


def _selector_from_training_config(
    train_cfg: Dict[str, Any],
    *,
    prefix: str,
    default_order: int = 6,
    default_mode: str = "upto",
    max_order: int = 6,
    default_monomials=(),
) -> tuple[int, str, tuple[tuple[int, int], ...]]:
    quantity_key = f"{prefix}_applied_quantities"
    monomial_key = f"{prefix}_monomials"
    raw_quantity = train_cfg.get(quantity_key, default_order)
    if is_auto_monomial_selector(raw_quantity):
        if not default_monomials:
            raise ValueError(f"training.{quantity_key}='auto' requires selected operator moments")
        monomials = tuple((int(m), int(n)) for m, n in default_monomials)
        order = max(m_power + n_power for m_power, n_power in monomials)
        mode = normalize_applied_quantities_mode(
            train_cfg.get(f"{prefix}_applied_quantities_mode", "exact")
        )
        return order, mode, monomials
    quantity_is_explicit_selector = (
        monomial_key not in train_cfg
        and not isinstance(raw_quantity, (int, np.integer))
        and not (isinstance(raw_quantity, str) and raw_quantity.strip().isdigit())
    )
    if quantity_is_explicit_selector:
        monomials = normalize_monomial_pairs(
            raw_quantity,
            option_name=f"training.{quantity_key}",
            max_order=max_order,
        )
        if not monomials:
            raise ValueError(f"training.{quantity_key} explicit monomial selector cannot be empty")
        order = max(m_power + n_power for m_power, n_power in monomials)
        mode = normalize_applied_quantities_mode(
            train_cfg.get(f"{prefix}_applied_quantities_mode", "exact")
        )
        return order, mode, monomials
    order = normalize_applied_quantities(
        raw_quantity
    )
    mode = normalize_applied_quantities_mode(
        train_cfg.get(f"{prefix}_applied_quantities_mode", default_mode)
    )
    raw_monomials = train_cfg.get(f"{prefix}_monomials", default_monomials)
    if is_auto_monomial_selector(raw_monomials):
        if not default_monomials:
            raise ValueError(f"training.{prefix}_monomials='auto' requires selected operator moments")
        monomials = tuple((int(m), int(n)) for m, n in default_monomials)
    else:
        monomials = normalize_monomial_pairs(
            raw_monomials,
            option_name=f"training.{prefix}_monomials",
            max_order=max_order,
        )
    return order, mode, monomials


def _selector_text(order: int, mode: str, monomials=()) -> str:
    monomials = tuple((int(m), int(n)) for m, n in (monomials or ()))
    if monomials:
        pairs = ", ".join(f"({m},{n})" for m, n in monomials)
        return f"monomials = [{pairs}]"
    return f"p = {order}" if mode == "exact" else f"p <= {order}"


def _auto_health_monomials_from_training_config(train_cfg: Dict[str, Any]) -> tuple[tuple[int, int], ...]:
    order, mode, monomials = _selector_from_training_config(
        train_cfg,
        prefix="operator",
        default_order=OPERATOR_MOMENT_DEFAULT_ORDER,
        max_order=OPERATOR_MOMENT_MAX_ORDER,
    )
    operator_specs = selected_monomial_specs(OPERATOR_MOMENT_SPECS, order, mode, monomials)
    return onsite_monomials_in_operator_equations(
        ((m_power, n_power) for _total_order, m_power, n_power, _term in operator_specs),
        max_order=6,
    )


def _residual_gmm_targets(
    train_cfg: Dict[str, Any],
) -> tuple[tuple[int, int], ...]:
    targets = normalize_monomial_pairs(
        train_cfg.get(
            "operator_monomials",
            ((0, 1), (1, 0), (1, 1), (2, 2)),
        ),
        option_name="training.operator_monomials",
        max_order=OPERATOR_MOMENT_MAX_ORDER,
    )
    targets = tuple(
        (int(m_power), int(n_power))
        for m_power, n_power in targets
        if int(m_power) + int(n_power) > 0
    )
    if not targets:
        raise ValueError(
            "training.operator_monomials must contain at least one non-identity "
            "target when loss_residual_gmm_prefactor is active"
        )
    return targets


def _selected_residual_gmm_terms(
    train_cfg: Dict[str, Any],
) -> tuple[str, ...]:
    if not _residual_gmm_branch_requested(train_cfg):
        return ()
    return projected_residual_term_names(_residual_gmm_targets(train_cfg))


def _selected_pareto_k_terms(train_cfg: Dict[str, Any]) -> tuple[str, ...]:
    applied_quantities, applied_quantities_mode, monomials = _selector_from_training_config(
        train_cfg,
        prefix="pareto_k",
        default_order=6,
        max_order=6,
        default_monomials=_auto_health_monomials_from_training_config(train_cfg),
    )
    return selected_monomial_terms(
        PARETO_K_MONOMIAL_SPECS,
        applied_quantities,
        applied_quantities_mode,
        monomials,
    )


def _normalize_loss_ema_terms(terms, train_cfg: Dict[str, Any]) -> tuple[str, ...]:
    if terms is None:
        defaults = []
        if _prefactor_active(
            train_cfg.get("loss_residual_gmm_prefactor", 0.0)
        ):
            defaults.append("loss_residual_gmm")
        if _prefactor_active(train_cfg.get("loss_pareto_k_prefactor", 0.0)):
            defaults.append("loss_pareto_k")
        return tuple(defaults)
    auto_terms = {
        "loss_pareto_k": _selected_pareto_k_terms(train_cfg)
        if is_auto_monomial_selector(train_cfg.get("pareto_k_monomials", ()))
        else (),
    }
    terms = expand_loss_ema_terms(terms, auto_terms=auto_terms)
    normalized = []
    for term in terms:
        key = str(term)
        if key not in LOSS_EMA_ALLOWED_TERMS:
            raise ValueError(
                "training.EMA.terms may only contain loss_residual_gmm, "
                "loss_pareto_k, explicit loss_pareto_k_m*_n* terms, "
                "loss_pareto_k_p* aliases, or loss_pareto_k_auto; "
                f"got {term!r}"
            )
        if key not in normalized:
            normalized.append(key)
    selected_pareto_k_terms = set(_selected_pareto_k_terms(train_cfg))
    inactive_pareto_k_terms = [
        term
        for term in normalized
        if term in PARETO_K_MONOMIAL_TERMS and term not in selected_pareto_k_terms
    ]
    if inactive_pareto_k_terms:
        raise ValueError(
            "training.EMA.terms contains Pareto-k monomial terms not selected by "
            "training.pareto_k_monomials or training.pareto_k_applied_quantities/mode: "
            f"{inactive_pareto_k_terms}"
        )
    if "loss_pareto_k" in normalized and any(term in normalized for term in PARETO_K_MONOMIAL_TERMS):
        raise ValueError(
            "training.EMA.terms cannot mix aggregate loss_pareto_k with "
            "component monomial terms loss_pareto_k_m*_n*"
        )
    return tuple(normalized)


def _resolve_loss_prefactors(train_cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "loss_residual_gmm_prefactor": float(
            train_cfg.get("loss_residual_gmm_prefactor", 0.0)
        ),
    }


def _residual_gmm_loss_enabled(prefactors: Dict[str, float]) -> bool:
    return (
        abs(
            float(
                prefactors.get(
                    "loss_residual_gmm_prefactor",
                    0.0,
                )
            )
        )
        >= LOSS_TERM_TRACE_THRESHOLD
    )


def _prefactor_active(value) -> bool:
    return abs(float(value)) >= LOSS_TERM_TRACE_THRESHOLD


def _residual_gmm_branch_requested(train_cfg: Dict[str, Any]) -> bool:
    return _prefactor_active(
        train_cfg.get("loss_residual_gmm_prefactor", 0.0)
    )


def _active_training_plot_metric_names(
    train_cfg: Dict[str, Any],
    residual_gmm_terms: tuple[str, ...] = (),
) -> set[str]:
    """Return history keys that should appear in training plots."""
    active = {"loss", "grads_norm", "lr"}
    loss_prefactors = _resolve_loss_prefactors(train_cfg)
    if _residual_gmm_loss_enabled(loss_prefactors):
        active.update(
            {
                "loss_residual_gmm_time",
                "loss_residual_gmm_raw",
                "residual_gmm_z_mean",
                "residual_gmm_z_max",
                "residual_gmm_z_worst",
                "residual_gmm_radius_mean",
                "residual_gmm_radius_max",
                "residual_gmm_radius_worst",
                "residual_gmm_warning_fraction",
                "residual_gmm_bad_fraction",
                *residual_gmm_terms,
            }
        )
    if _prefactor_active(train_cfg.get("loss_pareto_k_prefactor", 0.0)):
        active.update(
            {
                "loss_pareto_k_raw",
                "pareto_k_mean",
                "pareto_k_max",
                "pareto_k_worst",
                "pareto_k_warning_fraction",
                "pareto_k_bad_fraction",
                *_selected_pareto_k_terms(train_cfg),
            }
        )
    if _prefactor_active(train_cfg.get("loss_gauge_prefactor", 0.0)):
        active.update({"loss_gauge", "loss_gauge_drift", "loss_gauge_diffusion"})
    if _prefactor_active(train_cfg.get("loss_ess_prefactor", 0.0)):
        active.update(
            {
                "loss_ess",
                "log_weight_spread_mean",
                "log_weight_spread_max",
                "log_weight_spread_total",
                "ess_ratio_min",
                "ess_ratio_end",
            }
        )
    if _prefactor_active(train_cfg.get("loss_L2_prefactor", 0.0)):
        active.add("loss_L2")
    return active


def _format_flat_monomial_losses(
    latest_row: Dict[str, Any],
    *,
    label: str,
    specs: tuple[tuple[int, int, int, str], ...],
    selected_terms: tuple[str, ...],
    key_map: Optional[Dict[str, str]] = None,
) -> str:
    selected = set(selected_terms)
    labels = []
    values = []
    for _total_order, m_power, n_power, term in specs:
        if term not in selected:
            continue
        key = key_map.get(term, term) if key_map is not None else term
        labels.append(f"{int(m_power)}{int(n_power)}")
        values.append(f"{float(latest_row.get(key, 0.0)):.3e}")
    if not labels:
        return ""
    return f"{label}[{','.join(labels)}]=({', '.join(values)})"


def _format_metric_value(latest_row: Dict[str, Any], key: str, default: float = 0.0) -> str:
    return f"{float(latest_row.get(key, default)):.3e}"


_PROGRESS_LABEL_WIDTH = 14
_PROGRESS_LINE_WIDTH = 112


def _format_fraction_value(
    latest_row: Dict[str, Any], key: str, default: float = 0.0
) -> str:
    """Format a unit-interval diagnostic as an immediately readable percentage."""
    return f"{100.0 * float(latest_row.get(key, default)):.2f}%"


def _format_progress_rows(
    label: str,
    items: tuple[str, ...] | list[str],
    *,
    line_width: int = _PROGRESS_LINE_WIDTH,
) -> list[str]:
    """Render a labeled progress section, wrapping fields at natural boundaries."""
    if not items:
        return []
    rows: list[str] = []
    pending: list[str] = []

    def _prefix(row_label: str) -> str:
        return f"  {row_label:<{_PROGRESS_LABEL_WIDTH}} | "

    row_label = str(label)
    for item in (str(value) for value in items):
        prefix = _prefix(row_label)
        candidate = " | ".join((*pending, item))
        if pending and len(prefix) + len(candidate) > int(line_width):
            rows.append(prefix + " | ".join(pending))
            row_label = ""
            pending = [item]
        else:
            pending.append(item)
    if pending:
        rows.append(_prefix(row_label) + " | ".join(pending))
    return rows


def _format_residual_gmm_channel_losses(
    latest_row: Dict[str, Any],
    terms: tuple[str, ...],
) -> list[str]:
    """Render trace and configured onsite equations with natural wrapping."""

    trace_items: list[str] = []
    moment_items: list[str] = []
    for term in terms:
        channel_label = _residual_gmm_term_display(term)
        item = f"{channel_label}={_format_metric_value(latest_row, term)}"
        if channel_label == "(0,0)":
            trace_items.append(f"raw residual square={_format_metric_value(latest_row, term)}")
        else:
            moment_items.append(item)

    lines: list[str] = []
    if trace_items:
        lines.extend(_format_progress_rows("trace equation (0,0)", trace_items))
    if moment_items:
        lines.extend(_format_progress_rows("onsite moment equations", moment_items))
    return lines


def _residual_gmm_term_display(term: str) -> str:
    """Decode one trace-first residual-GMM diagnostic-history key."""

    prefix = "loss_residual_gmm_"
    suffix = term[len(prefix) :] if term.startswith(prefix) else term
    pieces = suffix.split("_")
    if (
        len(pieces) == 2
        and pieces[0].startswith("m")
        and pieces[1].startswith("n")
    ):
        return (
            f"({pieces[0].removeprefix('m')},"
            f"{pieces[1].removeprefix('n')})"
        )
    return suffix


def _format_residual_gmm_setup(
    *,
    targets: tuple[tuple[int, int], ...],
    terms: tuple[str, ...],
    num_site: int,
    prefactor: float,
    integrator_nodes: int,
    d_clip: float,
    cov_floor: float,
    cov_shrinkage: float,
    trace_mode: str,
    time_aggregation: str,
    time_beta: float,
) -> str:
    """Build the shared ordinary/segmented residual-GMM setup block."""
    integrator_nodes = int(integrator_nodes)
    integrator_subintervals = integrator_nodes - 1
    integrator_degree = 3 if integrator_nodes in {3, 4} else 5
    trace_mode = normalize_residual_gmm_trace_mode(trace_mode)
    objective_channel_count = projected_residual_objective_channel_count(
        targets,
        trace_mode=trace_mode,
    )
    time_aggregation = str(time_aggregation)
    if time_aggregation == "log1p":
        site_window_score = "log1p(q), q=mu^T P mu"
        aggregation_order = (
            "transform each site score in each window, then average sites and windows"
        )
    elif time_aggregation == "mean":
        site_window_score = "q=mu^T P mu"
        aggregation_order = "average q over sites and windows"
    elif time_aggregation == "entropic_log1p":
        site_window_score = "q=mu^T P mu"
        aggregation_order = (
            "average q over sites, apply log1p, then aggregate windows entropically"
        )
    else:
        site_window_score = "q=mu^T P mu"
        aggregation_order = "average q over sites, then aggregate windows entropically"
    lines = ["window-integrated site-resolved projected residual GMM"]
    lines.extend(
        _format_progress_rows(
            "equation",
            [
                "d <O_q>/dt = <L^dagger O_q> + V_q",
                "lhs=bare-monomial endpoint difference",
                (
                    f"rhs={integrator_nodes}-node closed Newton-Cotes integral; "
                    "coefficients stay on RHS"
                ),
                (
                    f"grid={integrator_subintervals} equal subintervals; "
                    f"formal polynomial degree={integrator_degree}"
                ),
            ],
        )
    )
    lines.extend(
        _format_progress_rows(
            "moments",
            [f"({int(m_power)},{int(n_power)})" for m_power, n_power in targets],
        )
    )
    lines.extend(
        _format_progress_rows(
            "basis",
            [
                f"sites={int(num_site)}",
                (
                    f"diagnostic channels/site={len(terms)} complex "
                    f"(trace plus {len(targets)} onsite)"
                ),
                (
                    f"objective channels/site={objective_channel_count} complex / "
                    f"{2 * objective_channel_count} real"
                ),
            ],
        )
    )
    lines.extend(
        _format_progress_rows(
            "objective",
            [
                f"trace mode={trace_mode}",
                (
                    "trace retained only as an unnormalized raw diagnostic"
                    if trace_mode == "diagnostic"
                    else "trace included in covariance and normalized objective"
                ),
                f"prefactor={float(prefactor):g}",
                f"time aggregation={time_aggregation}",
                f"per-site/window score={site_window_score}",
                aggregation_order,
                (
                    "time beta=inactive"
                    if time_aggregation in {"mean", "log1p"}
                    else f"time beta={float(time_beta):g}"
                ),
            ],
        )
    )
    lines.extend(
        _format_progress_rows(
            "whitening",
            [
                f"d_clip={float(d_clip):g}",
                f"cov floor={float(cov_floor):g}",
                f"cov shrinkage={float(cov_shrinkage):g}",
            ],
        )
    )
    lines.extend(
        _format_progress_rows(
            "covariance",
            [
                "shared active-channel correlation-space Cholesky whitening",
                "self-normalized ratio-influence covariance",
                "population covariance formed per site, then averaged over sites",
                "lagged EMA per window",
                f"decay={RESIDUAL_GMM_COVARIANCE_EMA_DECAY:g}",
                (
                    f"calibrate first {RESIDUAL_GMM_COVARIANCE_UPDATE_EPOCHS} "
                    "accepted updates/stage, then freeze"
                ),
            ],
        )
    )
    lines.extend(
        _format_progress_rows(
            "channel order",
            [
                "raw diagnostics: trace first",
                "then operator_monomials in configuration order",
                (
                    "covariance/objective: operator_monomials only"
                    if trace_mode == "diagnostic"
                    else "covariance/objective: same trace-first order"
                ),
            ],
        )
    )
    lines.extend(
        _format_progress_rows(
            "channels",
            [
                "(0,0) physical trace",
                *(_residual_gmm_term_display(term) for term in terms[1:]),
            ],
        )
    )
    return "\n".join(lines)


def _format_training_progress_line(
    prefix: str,
    latest_row: Dict[str, Any],
    *,
    selected_residual_gmm_terms: tuple[str, ...],
    selected_pareto_k_terms: tuple[str, ...],
    enable_loss_pareto_k: bool,
    enable_loss_residual_gmm: bool,
    enable_loss_gauge: bool,
    enable_loss_L2: bool,
    enable_loss_ess: bool = False,
) -> str:
    header = [
        prefix,
        f"loss={latest_row['loss']:.4e}",
        f"time={latest_row['epoch_time_sec']:.2f}s",
        f"lr={latest_row['lr']:.3e}",
    ]
    lines = [" | ".join(header)]

    if enable_loss_residual_gmm:
        residual_lines = _format_residual_gmm_channel_losses(
            latest_row,
            selected_residual_gmm_terms,
        )
        lines.extend(
            _format_progress_rows(
                "Residual GMM",
                [
                    "objective="
                    + _format_metric_value(
                        latest_row, "loss_residual_gmm_time"
                    ),
                    "raw residual square="
                    + _format_metric_value(
                        latest_row, "loss_residual_gmm_raw"
                    ),
                ],
            )
        )
        lines.extend(
            _format_progress_rows(
                "site-resolved residual cloud",
                [
                    "z mean/max/worst="
                    + "/".join(
                        (
                            _format_metric_value(
                                latest_row, "residual_gmm_z_mean"
                            ),
                            _format_metric_value(
                                latest_row, "residual_gmm_z_max"
                            ),
                            _format_metric_value(
                                latest_row, "residual_gmm_z_worst"
                            ),
                        )
                    ),
                    "radius mean/max/worst="
                    + "/".join(
                        (
                            _format_metric_value(
                                latest_row,
                                "residual_gmm_radius_mean",
                            ),
                            _format_metric_value(
                                latest_row,
                                "residual_gmm_radius_max",
                            ),
                            _format_metric_value(
                                latest_row,
                                "residual_gmm_radius_worst",
                            ),
                        )
                    ),
                    "warn/bad="
                    + "/".join(
                        (
                            _format_fraction_value(
                                latest_row,
                                "residual_gmm_warning_fraction",
                            ),
                            _format_fraction_value(
                                latest_row,
                                "residual_gmm_bad_fraction",
                            ),
                        )
                    ),
                ],
            )
        )
        lines.extend(residual_lines)

    if enable_loss_pareto_k:
        pareto_items = _format_flat_monomial_losses(
            latest_row,
            label="K",
            specs=PARETO_K_MONOMIAL_SPECS,
            selected_terms=selected_pareto_k_terms,
        )
        lines.append(
            "  K: "
            f"raw={_format_metric_value(latest_row, 'loss_pareto_k_raw')} | "
            f"k[max,worst]=({_format_metric_value(latest_row, 'pareto_k_max')}, "
            f"{_format_metric_value(latest_row, 'pareto_k_worst')})"
        )
        if pareto_items:
            lines.append("     " + pareto_items)

    if enable_loss_gauge:
        lines.append(
            "  gauge: "
            f"drift={_format_metric_value(latest_row, 'loss_gauge_drift')} | "
            f"diff={_format_metric_value(latest_row, 'loss_gauge_diffusion')}"
        )
    if enable_loss_L2:
        lines.append(f"  L2 raw: loss_L2={latest_row['loss_L2']:.4e}")
    if enable_loss_ess:
        lines.append(
            "  ESS: "
            f"hinge={_format_metric_value(latest_row, 'loss_ess')} | "
            f"spread[mean,max]=({_format_metric_value(latest_row, 'log_weight_spread_mean')}, "
            f"{_format_metric_value(latest_row, 'log_weight_spread_max')}) | "
            f"ess/N[min]={_format_metric_value(latest_row, 'ess_ratio_min')}"
        )

    return "\n".join(lines)


@dataclass
class LossEmaNormalizer:
    """Python-side stop-gradient EMA normalization for selected loss terms."""

    enabled: bool
    terms: tuple[str, ...]
    decay: float
    warmup_epochs: int
    eps: float
    floor: float
    ceiling: float
    r_max: Optional[float]
    scales: Dict[str, float]
    pareto_k_site_scales: Optional[np.ndarray] = None
    num_updates: int = 0

    @classmethod
    def from_training_config(cls, train_cfg: Dict[str, Any]):
        cfg = _loss_ema_config(train_cfg)
        enabled = bool(cfg.get("enabled", False))
        terms = _normalize_loss_ema_terms(cfg.get("terms", None), train_cfg) if enabled else ()
        decay = float(cfg.get("decay", 0.995))
        if not (0.0 <= decay < 1.0):
            raise ValueError("training.EMA.decay must satisfy 0 <= decay < 1")
        warmup_epochs = max(0, int(cfg.get("warmup_epochs", 10)))
        eps = float(cfg.get("eps", 1.0e-12))
        floor = float(cfg.get("floor", 1.0e-8))
        ceiling = float(cfg.get("ceiling", 1.0e8))
        r_max_value = cfg.get("r_max", 5.0)
        r_max = None if r_max_value is None else float(r_max_value)
        if floor <= 0.0 or ceiling <= floor:
            raise ValueError("training.EMA requires 0 < floor < ceiling")
        if r_max is not None and r_max < 1.0:
            raise ValueError("training.EMA.r_max must be >= 1, or null to disable relative clipping")
        return cls(
            enabled=enabled,
            terms=terms,
            decay=decay,
            warmup_epochs=warmup_epochs,
            eps=eps,
            floor=floor,
            ceiling=ceiling,
            r_max=r_max,
            scales={term: 1.0 for term in terms},
        )

    def _scale(self, term: str) -> float:
        return float(np.clip(self.scales.get(term, 1.0), self.floor, self.ceiling))

    def effective_prefactors(self, base_prefactors: Dict[str, float]) -> Dict[str, float]:
        """Return prefactors used for the next gradient call.

        During warmup, the raw objective is used while EMA scales are collected.
        After warmup, selected terms are divided by a stop-gradient EMA scale.
        """
        prefactors = dict(base_prefactors)
        if not self.enabled or self.num_updates < self.warmup_epochs:
            return prefactors
        for term in self.terms:
            if term not in LOSS_EMA_TERM_SPECS:
                continue
            pref_key = LOSS_EMA_TERM_SPECS[term]
            prefactors[pref_key] = float(prefactors[pref_key]) / (self._scale(term) + self.eps)
        return prefactors

    def pareto_k_monomial_weight_matrix(self, num_site: int) -> jnp.ndarray:
        normalization_active = self.enabled and self.num_updates >= self.warmup_epochs
        weights = np.ones((len(PARETO_K_MONOMIAL_TERMS), int(num_site)), dtype=np.float64)
        if normalization_active and self.pareto_k_site_scales is not None:
            site_scales = np.asarray(self.pareto_k_site_scales, dtype=np.float64)
            if site_scales.shape == weights.shape:
                for index, term in enumerate(PARETO_K_MONOMIAL_TERMS):
                    if term in self.terms:
                        weights[index, :] = 1.0 / (site_scales[index, :] + self.eps)
        return jnp.asarray(weights, dtype=DTYPE)

    def pareto_k_monomial_weight_vector(self) -> jnp.ndarray:
        return self.pareto_k_monomial_weight_matrix(1)[:, 0]

    def update(self, aux: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        site_terms = aux.get("loss_pareto_k_site_terms")
        if site_terms is not None:
            site_values = np.abs(np.asarray(site_terms, dtype=np.float64))
            if site_values.ndim == 2 and site_values.shape[0] == len(PARETO_K_MONOMIAL_TERMS):
                site_values = np.clip(site_values, self.floor, self.ceiling)
                if self.pareto_k_site_scales is None or self.pareto_k_site_scales.shape != site_values.shape:
                    self.pareto_k_site_scales = np.ones_like(site_values)
                selected = np.asarray(
                    [term in self.terms for term in PARETO_K_MONOMIAL_TERMS],
                    dtype=bool,
                )
                if np.any(selected):
                    if self.r_max is not None and self.num_updates > 0:
                        site_values = np.minimum(
                            site_values,
                            self.pareto_k_site_scales * float(self.r_max),
                        )
                    if self.num_updates == 0:
                        self.pareto_k_site_scales[selected, :] = site_values[selected, :]
                    else:
                        updated = (
                            self.decay * self.pareto_k_site_scales
                            + (1.0 - self.decay) * site_values
                        )
                        self.pareto_k_site_scales[selected, :] = np.clip(
                            updated[selected, :],
                            self.floor,
                            self.ceiling,
                        )
        for term in self.terms:
            if term in PARETO_K_MONOMIAL_TERMS:
                continue
            value = abs(to_scalar_float(aux.get(term, 0.0)))
            if not np.isfinite(value):
                continue
            value = float(np.clip(value, self.floor, self.ceiling))
            if self.r_max is not None and self.num_updates > 0:
                previous_scale = self._scale(term)
                value = min(value, previous_scale * self.r_max)
            if self.num_updates == 0:
                self.scales[term] = value
            else:
                self.scales[term] = float(
                    np.clip(
                        self.decay * self.scales[term] + (1.0 - self.decay) * value,
                        self.floor,
                        self.ceiling,
                    )
                )
        self.num_updates += 1

    def annotate_aux(self, aux: Dict[str, Any]) -> None:
        normalization_active = self.enabled and self.num_updates >= self.warmup_epochs
        for term in LOSS_EMA_ALLOWED_TERMS:
            if term in PARETO_K_MONOMIAL_TERMS:
                index = PARETO_K_MONOMIAL_TERMS.index(term)
                scale = 1.0
                normalized = to_scalar_float(aux.get(term, 0.0))
                if self.enabled and term in self.terms and self.pareto_k_site_scales is not None:
                    site_scales = np.asarray(self.pareto_k_site_scales, dtype=np.float64)
                    if site_scales.ndim == 2 and index < site_scales.shape[0]:
                        row_scale = np.clip(site_scales[index], self.floor, self.ceiling)
                        scale = float(np.mean(row_scale))
                        site_terms = aux.get("loss_pareto_k_site_terms")
                        if normalization_active and site_terms is not None:
                            site_values = np.asarray(site_terms, dtype=np.float64)
                            if site_values.ndim == 2 and site_values.shape == site_scales.shape:
                                normalized = float(np.mean(site_values[index] / (row_scale + self.eps)))
                            else:
                                normalized = normalized / (scale + self.eps)
                aux[f"{term}_ema_scale"] = jnp.asarray(scale, dtype=DTYPE)
                aux[f"{term}_normalized"] = jnp.asarray(normalized, dtype=DTYPE)
                continue
            scale = self._scale(term) if self.enabled and term in self.terms else 1.0
            objective_scale = scale if normalization_active and term in self.terms else 1.0
            aux[f"{term}_ema_scale"] = jnp.asarray(scale, dtype=DTYPE)
            aux[f"{term}_normalized"] = jnp.asarray(
                to_scalar_float(aux.get(term, 0.0)) / (objective_scale + self.eps),
                dtype=DTYPE,
            )

    def summary(self) -> str:
        if not self.enabled:
            return "training EMA loss normalization = disabled"
        return (
            "training EMA loss normalization = enabled "
            f"terms={list(self.terms)}, decay={self.decay:g}, "
            f"warmup_epochs={self.warmup_epochs}, "
            f"floor={self.floor:g}, ceiling={self.ceiling:g}, "
            f"r_max={self.r_max if self.r_max is not None else 'disabled'}"
        )


@dataclass
class TrainingArtifacts:
    state: train_state.TrainState
    history: Dict[str, Any]
    params_path: str
    history_npz_path: str
    history_json_path: str
    metadata_path: str
    history_plot_png_path: str
    history_plot_pdf_path: str
    stage_histories: Optional[list[Dict[str, Any]]] = None


@dataclass(frozen=True)
class StagedStageSpec:
    stage_id: int
    n_epoch: int
    n_windows: int
    num_walker: int
    n_steps: int


@dataclass(frozen=True)
class SegmentedStageSpec:
    stage_id: int
    update_budget: int
    n_segments: int
    num_walker: int
    n_steps: int
    n_windows_per_segment: int


def _should_load_stage_parameters(
    stage_id: int,
    base_load_parameters: bool,
) -> bool:
    """Return whether a schedule entry should initialize from a checkpoint."""

    return bool(int(stage_id) > 1 or base_load_parameters)


@dataclass
class TrainingHistoryBuffer:
    values: Dict[str, list]
    residual_gmm_terms: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        residual_gmm_term_names: tuple[str, ...] = (),
    ):
        residual_terms = tuple(
            str(term) for term in residual_gmm_term_names
        )
        keys = (
            *TRAINING_HISTORY_KEYS,
            *residual_terms,
        )
        return cls(
            values={key: [] for key in keys},
            residual_gmm_terms=residual_terms,
        )

    def append(self, epoch: int, aux: Dict[str, Any], lr_value: float):
        def _aux_float(name: str, default: float = 0.0) -> float:
            return to_scalar_float(aux.get(name, default))

        pareto_k_max = _aux_float("pareto_k_max")
        row = {
            "epoch": int(epoch),
            "loss": to_scalar_float(aux["loss"]),
            "loss_pareto_k": _aux_float("loss_pareto_k"),
            "pareto_k_mean": _aux_float("pareto_k_mean"),
            "pareto_k_max": pareto_k_max,
            "pareto_k_worst": max(_aux_float("pareto_k_worst", pareto_k_max), pareto_k_max),
            "pareto_k_warning_fraction": _aux_float("pareto_k_warning_fraction"),
            "pareto_k_bad_fraction": _aux_float("pareto_k_bad_fraction"),
            "loss_gauge": _aux_float("loss_gauge"),
            "loss_gauge_drift": _aux_float("loss_gauge_drift"),
            "loss_gauge_diffusion": _aux_float("loss_gauge_diffusion"),
            "loss_ess": _aux_float("loss_ess"),
            "loss_ess_weighted": _aux_float("loss_ess_weighted"),
            "log_weight_spread_mean": _aux_float("log_weight_spread_mean"),
            "log_weight_spread_max": _aux_float("log_weight_spread_max"),
            "log_weight_spread_total": _aux_float("log_weight_spread_total"),
            "ess_ratio_min": _aux_float("ess_ratio_min", 1.0),
            "ess_ratio_end": _aux_float("ess_ratio_end", 1.0),
            "loss_residual_gmm": _aux_float(
                "loss_residual_gmm"
            ),
            "loss_residual_gmm_time": _aux_float(
                "loss_residual_gmm_time",
                _aux_float("loss_residual_gmm"),
            ),
            "loss_residual_gmm_raw": _aux_float(
                "loss_residual_gmm_raw"
            ),
            "residual_gmm_z_mean": _aux_float(
                "residual_gmm_z_mean"
            ),
            "residual_gmm_z_max": _aux_float(
                "residual_gmm_z_max"
            ),
            "residual_gmm_z_worst": max(
                _aux_float(
                    "residual_gmm_z_worst",
                    _aux_float("residual_gmm_z_max"),
                ),
                _aux_float("residual_gmm_z_max"),
            ),
            "residual_gmm_radius_mean": _aux_float(
                "residual_gmm_radius_mean"
            ),
            "residual_gmm_radius_max": _aux_float(
                "residual_gmm_radius_max"
            ),
            "residual_gmm_radius_worst": max(
                _aux_float(
                    "residual_gmm_radius_worst",
                    _aux_float("residual_gmm_radius_max"),
                ),
                _aux_float("residual_gmm_radius_max"),
            ),
            "residual_gmm_warning_fraction": _aux_float(
                "residual_gmm_warning_fraction"
            ),
            "residual_gmm_bad_fraction": _aux_float(
                "residual_gmm_bad_fraction"
            ),
            "loss_L2": _aux_float("loss_L2"),
            "grads_norm": to_scalar_float(aux["grads_norm"]),
            "epoch_time_sec": to_scalar_float(aux["epoch_time_sec"]),
            "lr": float(lr_value),
            "loss_pareto_k_ema_scale": _aux_float("loss_pareto_k_ema_scale", 1.0),
            "loss_residual_gmm_ema_scale": _aux_float(
                "loss_residual_gmm_ema_scale",
                1.0,
            ),
            "loss_pareto_k_normalized": _aux_float("loss_pareto_k_normalized", _aux_float("loss_pareto_k")),
            "loss_residual_gmm_normalized": _aux_float(
                "loss_residual_gmm_normalized",
                _aux_float("loss_residual_gmm"),
            ),
        }
        residual_terms_aux = aux.get("loss_residual_gmm_terms")
        residual_terms_array = None
        if residual_terms_aux is not None:
            residual_terms_array = np.asarray(
                residual_terms_aux,
                dtype=np.float64,
            )
            if residual_terms_array.shape != (
                len(self.residual_gmm_terms),
            ):
                residual_terms_array = None
        for index, term in enumerate(self.residual_gmm_terms):
            row[term] = (
                float(residual_terms_array[index])
                if residual_terms_array is not None
                else _aux_float(term)
            )
        for term in PARETO_K_MONOMIAL_TERMS:
            row[term] = _aux_float(term)
            row[f"{term}_ema_scale"] = _aux_float(f"{term}_ema_scale", 1.0)
            row[f"{term}_normalized"] = _aux_float(f"{term}_normalized", row[term])
        row["loss_pareto_k_raw"] = float(sum(row[term] for term in PARETO_K_MONOMIAL_TERMS))
        for key in self.values:
            self.values[key].append(row.get(key, 0.0))
        return row

    def as_dict(self):
        return {key: list(values) for key, values in self.values.items()}

    def final_epoch(self) -> int:
        if not self.values["epoch"]:
            return 0
        return int(self.values["epoch"][-1])

class GaugeTrainer:
    """Minimal trainer for the neural lattice gauge model."""

    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path
        self.lattice = Lattice.from_config(config["lattice"])
        self.model = Model(config["model"], self.lattice, gauge_mode=config["training"]["gauge_mode"])
        self.parameter_initialization = "random"

    def _pareto_k_monomial_channel_summary(
        self,
        max_order: int,
        applied_quantities_mode: str = "upto",
        monomials=(),
    ) -> str:
        monomials = tuple((int(m), int(n)) for m, n in (monomials or ()))
        if monomials:
            site_count = int(self.lattice.num_site)
            counts_by_order: Dict[int, int] = {}
            for m_power, n_power in monomials:
                counts_by_order[int(m_power) + int(n_power)] = counts_by_order.get(
                    int(m_power) + int(n_power),
                    0,
                ) + site_count
            total = len(monomials) * site_count
            counts_text = ", ".join(
                f"p{order}={counts_by_order[order]}" for order in sorted(counts_by_order)
            )
            basis_text = f"selected monomials use {site_count} onsite channel(s)"
            selector_text = _selector_text(max_order, applied_quantities_mode, monomials)
            return (
                "training onsite Pareto-k monomial channels: "
                f"total={total} {selector_text} "
                f"({counts_text}); {basis_text}"
            )

        if normalize_applied_quantities_mode(applied_quantities_mode) == "exact":
            orders = [int(max_order)]
            mode_text = f"exact p={max_order}"
        else:
            orders = list(range(1, int(max_order) + 1))
            mode_text = f"through p<={max_order}"
        site_count = int(self.lattice.num_site)
        counts = [(order + 1) * site_count for order in orders]
        basis_text = f"all monomials use {site_count} onsite channel(s)"
        total = int(sum(counts))
        counts_text = ", ".join(f"p{order}={count}" for order, count in zip(orders, counts))
        return (
            "training onsite Pareto-k monomial channels: "
            f"total={total} {mode_text} "
            f"({counts_text}); {basis_text}"
        )

    @classmethod
    def from_config_path(cls, config_path: str):
        return cls(load_config(config_path), config_path=config_path)

    @property
    def save_dir(self) -> str:
        return self.config["io"]["save_dir"]

    @property
    def train_dir(self) -> str:
        return os.path.join(self.save_dir, "train")

    @property
    def params_path(self) -> str:
        return os.path.join(self.train_dir, "model_params.msgpack")

    @property
    def history_npz_path(self) -> str:
        return os.path.join(self.train_dir, "training_history.npz")

    @property
    def history_json_path(self) -> str:
        return os.path.join(self.train_dir, "training_history.json")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.train_dir, "training_metadata.json")

    @property
    def history_plot_png_path(self) -> str:
        return os.path.join(self.train_dir, "training_history.png")

    @property
    def history_plot_pdf_path(self) -> str:
        return os.path.join(self.train_dir, "training_history.pdf")

    @property
    def config_used_path(self) -> str:
        return os.path.join(self.train_dir, "config_used.json")

    def _prepare_io(self):
        prepare_output_dirs(self.save_dir, self.config["io"]["clean_start"])
        save_json(self.config, self.config_used_path)

    def _sample_model_inputs(self):
        lnOmega0, alpha0, beta0 = self.lattice.initialize_phase_space(num_walker=1, n0=self.lattice.n0)
        alpha_real, beta_real = var_complex_to_real(alpha0, beta0)
        lnOmega_real = jnp.stack([jnp.real(lnOmega0), jnp.imag(lnOmega0)], axis=-1)
        return lnOmega_real, alpha_real, beta_real, self.lattice.physical_params()

    def _build_state(self, key: jax.Array):
        train_cfg = self.config["training"]
        params_path = None
        if train_cfg["load_parameters"]:
            params_path = resolve_checkpoint_params_path(self.config, self.save_dir, purpose="training")
            if not os.path.exists(params_path):
                raise FileNotFoundError(f"Could not find model parameters at '{params_path}'")
            self.parameter_initialization = f"loaded from {params_path}"
        else:
            self.parameter_initialization = "randomly initialized"

        lnOmega_real, alpha_real, beta_real, physical_params = self._sample_model_inputs()
        return self.model.create_train_state(
            config=self.config,
            key=key,
            sample_lnOmega_real=lnOmega_real,
            sample_alpha_real=alpha_real,
            sample_beta_real=beta_real,
            sample_t=0.0,
            physical_params=physical_params,
            params_path=params_path,
        )

    def _build_state_for_config(self, stage_config: Dict[str, Any], key: jax.Array):
        base_config = self.config
        try:
            self.config = stage_config
            return self._build_state(key)
        finally:
            self.config = base_config

    def _history_payload(self, history):
        if isinstance(history, TrainingHistoryBuffer):
            return history.as_dict(), history.final_epoch()
        history_dict = history
        final_epoch = int(history_dict["epoch"][-1]) if history_dict["epoch"] else 0
        return history_dict, final_epoch

    def _persist_checkpoint(self, state):
        save_parameters(state.params, self.params_path)

    def _persist_history_bundle(
        self,
        state,
        history,
        *,
        params_path: str,
        history_npz_path: str,
        history_json_path: str,
        metadata_path: str,
        history_plot_png_path: str,
        history_plot_pdf_path: str,
        config_used_path: Optional[str],
        make_plot: bool = True,
        persist_params: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        history_dict, final_epoch = self._history_payload(history)

        if persist_params:
            save_parameters(state.params, params_path)
        save_npz(history_npz_path, history_dict)
        save_json(history_dict, history_json_path)
        plot_created = False
        if make_plot:
            plot_created = plot_training_history(
                history=history_dict,
                png_path=history_plot_png_path,
                pdf_path=history_plot_pdf_path,
                active_metric_names=_active_training_plot_metric_names(
                    self.config["training"],
                    tuple(
                        key
                        for key in history_dict
                        if key.startswith("loss_residual_gmm_m")
                    ),
                ),
            )
        metadata = {
            "config_path": self.config_path,
            "num_parameter": count_params(state.params),
            "params_path": params_path,
            "parameter_initialization": self.parameter_initialization,
            "checkpoint_semantics": "parameters_only_warm_start",
            "optimizer_state_restored": False,
            "loss_ema_state_restored": False,
            "residual_covariance_state_restored": False,
            "residual_gmm_definition": (
                "site_averaged_covariance_bare_monomial_closed_newton_cotes_v3"
            ),
            "history_npz_path": history_npz_path,
            "history_json_path": history_json_path,
            "history_plot_png_path": (
                history_plot_png_path if (plot_created or os.path.exists(history_plot_png_path)) else None
            ),
            "history_plot_pdf_path": (
                history_plot_pdf_path if (plot_created or os.path.exists(history_plot_pdf_path)) else None
            ),
            "config_used_path": config_used_path,
            "final_epoch": final_epoch,
            "history_keys": list(history_dict.keys()),
        }
        train_cfg = self.config["training"]
        metadata["sde_solver"] = sde_solver_control_metadata(
            normalize_sde_solver(train_cfg.get("sde_solver")),
            max_iterations=int(train_cfg.get("sde_max_iter", 4)),
            root_rtol=float(train_cfg["sde_root_rtol"]),
            root_atol=float(train_cfg["sde_root_atol"]),
            affine_expm_order=int(train_cfg["sde_affine_expm_order"]),
            affine_expm_substeps=int(train_cfg["sde_affine_expm_substeps"]),
            newton_damping_steps=int(train_cfg["sde_newton_damping_steps"]),
        )
        if _residual_gmm_branch_requested(train_cfg):
            trace_mode = normalize_residual_gmm_trace_mode(
                train_cfg.get(
                    "residual_gmm_trace_mode",
                    DEFAULT_RESIDUAL_GMM_TRACE_MODE,
                )
            )
            integrator_nodes = int(
                train_cfg.get("residual_gmm_integrator_nodes", 6)
            )
            targets = _residual_gmm_targets(train_cfg)
            diagnostic_channel_count = 1 + len(targets)
            objective_channel_count = (
                projected_residual_objective_channel_count(
                    targets,
                    trace_mode=trace_mode,
                )
            )
            metadata["residual_gmm_integrator"] = {
                "family": "closed_newton_cotes",
                "nodes": integrator_nodes,
                "equal_subintervals": integrator_nodes - 1,
                "formal_polynomial_degree": (
                    3 if integrator_nodes in {3, 4} else 5
                ),
            }
            metadata["residual_gmm_num_site"] = int(self.lattice.num_site)
            metadata["residual_gmm_covariance"] = {
                "walker_estimator": "population_covariance",
                "site_aggregation": "mean_of_within_site_covariances",
                "shared_across_sites": True,
                "trace_mode": trace_mode,
                "trace_channel_included": trace_mode == "joint",
                "trace_in_covariance": trace_mode == "joint",
                "trace_in_objective": trace_mode == "joint",
                "trace_raw_diagnostic_retained": True,
                "diagnostic_complex_channel_count": diagnostic_channel_count,
                "objective_complex_channel_count": objective_channel_count,
                "objective_real_channel_dimension": (
                    2 * objective_channel_count
                ),
                "bank_has_site_axis": False,
            }
            time_aggregation = str(
                train_cfg.get("residual_gmm_time_aggregation", "mean")
            ).strip().lower()
            if time_aggregation == "log1p":
                site_window_transform = "log1p(q)"
                transform_order = (
                    "apply to each site score in each window, then average "
                    "over sites and windows"
                )
            elif time_aggregation == "mean":
                site_window_transform = "q"
                transform_order = "average q over sites and windows"
            elif time_aggregation == "entropic_log1p":
                site_window_transform = "q"
                transform_order = (
                    "average q over sites, apply log1p, then aggregate "
                    "windows entropically"
                )
            else:
                site_window_transform = "q"
                transform_order = (
                    "average q over sites, then aggregate windows entropically"
                )
            metadata["residual_gmm_aggregation"] = {
                "mode": time_aggregation,
                "site_window_quadratic": "q = mu^T P mu",
                "site_window_transform": site_window_transform,
                "transform_order": transform_order,
                "time_beta_active": time_aggregation
                not in {"mean", "log1p"},
            }
            metadata["residual_gmm_terms"] = list(
                projected_residual_term_names(targets)
            )
            metadata["residual_gmm_channels"] = list(
                projected_residual_channel_labels(targets)
            )
            diagnostic_labels = projected_residual_channel_labels(targets)
            metadata["residual_gmm_objective_channels"] = list(
                diagnostic_labels
                if trace_mode == "joint"
                else diagnostic_labels[1:]
            )
            metadata["residual_gmm_operator_monomials"] = [
                [int(m_power), int(n_power)]
                for m_power, n_power in targets
            ]
        if extra_metadata:
            metadata.update(extra_metadata)
        save_json(metadata, metadata_path)

    def _persist_monitoring_outputs(self, state, history, make_plot: bool = True):
        self._persist_history_bundle(
            state,
            history,
            params_path=self.params_path,
            history_npz_path=self.history_npz_path,
            history_json_path=self.history_json_path,
            metadata_path=self.metadata_path,
            history_plot_png_path=self.history_plot_png_path,
            history_plot_pdf_path=self.history_plot_pdf_path,
            config_used_path=self.config_used_path,
            make_plot=make_plot,
            persist_params=False,
        )

    def _stage_output_paths(self, stage_id: int):
        stage_tag = f"stage_{stage_id:02d}"
        return {
            "params_path": os.path.join(self.train_dir, f"{stage_tag}_model_params.msgpack"),
            "history_npz_path": os.path.join(self.train_dir, f"{stage_tag}_training_history.npz"),
            "history_json_path": os.path.join(self.train_dir, f"{stage_tag}_training_history.json"),
            "metadata_path": os.path.join(self.train_dir, f"{stage_tag}_training_metadata.json"),
            "history_plot_png_path": os.path.join(self.train_dir, f"{stage_tag}_training_history.png"),
            "history_plot_pdf_path": os.path.join(self.train_dir, f"{stage_tag}_training_history.pdf"),
            "config_used_path": os.path.join(self.train_dir, f"{stage_tag}_config.json"),
        }

    def _persist_stage_snapshot(
        self,
        stage_id: int,
        state,
        history,
        stage_config: Dict[str, Any],
        make_plot: bool = True,
    ):
        stage_paths = self._stage_output_paths(stage_id)
        save_json(stage_config, stage_paths["config_used_path"])
        self._persist_history_bundle(
            state,
            history,
            params_path=stage_paths["params_path"],
            history_npz_path=stage_paths["history_npz_path"],
            history_json_path=stage_paths["history_json_path"],
            metadata_path=stage_paths["metadata_path"],
            history_plot_png_path=stage_paths["history_plot_png_path"],
            history_plot_pdf_path=stage_paths["history_plot_pdf_path"],
            config_used_path=stage_paths["config_used_path"],
            make_plot=make_plot,
            persist_params=True,
            extra_metadata={"stage_id": int(stage_id)},
        )

    def _load_segmented_stage_specs(self) -> list[SegmentedStageSpec]:
        segmented_cfg = self.config["training"]["segmented_overlap"]
        default_n_windows = segmented_cfg.get("n_windows_per_segment")
        stage_specs = []
        for idx, item in enumerate(segmented_cfg["stage_schedule"], start=1):
            stage_specs.append(
                SegmentedStageSpec(
                    stage_id=int(item.get("stage_id", idx)),
                    update_budget=int(item["n_epoch"]),
                    n_segments=int(item["n_segments"]),
                    num_walker=int(item["num_walker"]),
                    n_steps=int(item["N_steps"]),
                    n_windows_per_segment=int(item.get("n_windows_per_segment", default_n_windows)),
                )
            )
        return stage_specs

    def _load_staged_stage_specs(self) -> list[StagedStageSpec]:
        stage_specs = []
        for idx, item in enumerate(
            self.config["training"]["staged_schedule"],
            start=1,
        ):
            stage_specs.append(
                StagedStageSpec(
                    stage_id=int(item.get("stage_id", idx)),
                    n_epoch=int(item["n_epoch"]),
                    n_windows=int(item["N_windows"]),
                    num_walker=int(item["num_walker"]),
                    n_steps=int(item["N_steps"]),
                )
            )
        return stage_specs

    def _build_segment_window_weighting(
        self,
        *,
        n_segments: int,
        n_windows_per_segment: int,
        stride_windows: int,
    ):
        effective_total_windows = n_windows_per_segment + max(0, n_segments - 1) * stride_windows
        multiplicities = np.zeros(effective_total_windows, dtype=np.float64)
        for segment_local_index in range(n_segments):
            segment_start_idx = int(segment_local_index * stride_windows)
            multiplicities[segment_start_idx : segment_start_idx + n_windows_per_segment] += 1.0

        segment_window_weights = []
        segment_window_weight_sums = []
        for segment_local_index in range(n_segments):
            segment_start_idx = int(segment_local_index * stride_windows)
            local_weights = 1.0 / multiplicities[segment_start_idx : segment_start_idx + n_windows_per_segment]
            segment_window_weights.append(jnp.asarray(local_weights, dtype=DTYPE))
            segment_window_weight_sums.append(float(np.sum(local_weights)))
        return segment_window_weights, segment_window_weight_sums

    def _fit_one_rollout(self):
        train_cfg = self.config["training"]
        lat_cfg = self.config["lattice"]
        keygen = KeyGenerator(train_cfg["seed"])
        # Deserialize a requested warm-start checkpoint before clean_start can
        # remove an earlier run directory containing that checkpoint.
        state = self._build_state(keygen.next(fold_in_value=1))
        self._prepare_io()

        n_epoch = int(train_cfg["n_epoch"])
        log_every = max(1, int(train_cfg.get("log_every", max(1, n_epoch // 20))))
        save_every = int(train_cfg["save_every"])
        plot_every = int(train_cfg.get("plot_every", save_every if save_every > 0 else 0))
        make_plots = bool(train_cfg.get("make_plots", True))
        num_walker = int(train_cfg["num_walker"])
        sde_max_iter = int(train_cfg.get("sde_max_iter", 4))
        sde_solver = normalize_sde_solver(train_cfg.get("sde_solver"))
        sde_root_rtol = float(train_cfg["sde_root_rtol"])
        sde_root_atol = float(train_cfg["sde_root_atol"])
        sde_affine_expm_order = int(train_cfg["sde_affine_expm_order"])
        sde_affine_expm_substeps = int(train_cfg["sde_affine_expm_substeps"])
        sde_newton_damping_steps = int(train_cfg["sde_newton_damping_steps"])
        sde_control_metadata = sde_solver_control_metadata(
            sde_solver,
            max_iterations=sde_max_iter,
            root_rtol=sde_root_rtol,
            root_atol=sde_root_atol,
            affine_expm_order=sde_affine_expm_order,
            affine_expm_substeps=sde_affine_expm_substeps,
            newton_damping_steps=sde_newton_damping_steps,
        )
        noise_refresh_every = max(1, int(train_cfg.get("noise_refresh_every", 1)))
        auto_health_monomials = _auto_health_monomials_from_training_config(train_cfg)
        pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials = _selector_from_training_config(
            train_cfg,
            prefix="pareto_k",
            default_order=6,
            max_order=6,
            default_monomials=auto_health_monomials,
        )
        pareto_k_threshold = float(train_cfg.get("pareto_k_threshold", 0.7))
        pareto_k_threshold_tau = float(train_cfg.get("pareto_k_threshold_tau", 0.1))
        pareto_k_envelope_beta = float(train_cfg.get("pareto_k_envelope_beta", 0.5))
        pareto_k_envelope_excess = str(train_cfg.get("pareto_k_envelope_excess", "log"))
        pareto_k_tail_fraction = float(train_cfg.get("pareto_k_tail_fraction", 0.01))
        pareto_k_min_tail_count = int(train_cfg.get("pareto_k_min_tail_count", 32))
        residual_gmm_integrator_nodes = int(
            train_cfg.get("residual_gmm_integrator_nodes", 6)
        )
        residual_gmm_d_clip = float(
            train_cfg.get("residual_gmm_d_clip", 10.0)
        )
        residual_gmm_cov_floor = float(
            train_cfg.get("residual_gmm_cov_floor", 1.0e-8)
        )
        residual_gmm_cov_shrinkage = float(
            train_cfg.get("residual_gmm_cov_shrinkage", 0.05)
        )
        residual_gmm_trace_mode = normalize_residual_gmm_trace_mode(
            train_cfg.get(
                "residual_gmm_trace_mode",
                DEFAULT_RESIDUAL_GMM_TRACE_MODE,
            )
        )
        residual_gmm_time_aggregation = str(
            train_cfg.get("residual_gmm_time_aggregation", "mean")
        ).strip().lower()
        residual_gmm_time_beta = float(
            train_cfg.get("residual_gmm_time_beta", 2.0)
        )
        neural_gauge_components = normalize_neural_gauge_components(
            train_cfg.get("neural_gauge_components", "both")
        )
        apply_neural_gauge_every_steps = int(train_cfg.get("apply_neural_gauge_every_steps", 0))
        neural_gauge_state_gradient = str(train_cfg.get("neural_gauge_state_gradient", "full"))
        neural_gauge_each_apply = neural_gauge_state_gradient == "each_apply"
        loss_pareto_k_prefactor = float(train_cfg.get("loss_pareto_k_prefactor", 0.0))
        loss_residual_gmm_prefactor = float(
            train_cfg.get("loss_residual_gmm_prefactor", 0.0)
        )
        loss_L2_prefactor = float(train_cfg.get("loss_L2_prefactor", 0.0))
        loss_gauge_prefactor = float(train_cfg.get("loss_gauge_prefactor", 0.0))
        loss_ess_prefactor = float(train_cfg.get("loss_ess_prefactor", 0.0))
        loss_prefactors = _resolve_loss_prefactors(train_cfg)
        loss_ema = LossEmaNormalizer.from_training_config(train_cfg)
        base_loss_prefactors = {
            "loss_pareto_k_prefactor": loss_pareto_k_prefactor,
            "loss_residual_gmm_prefactor": (
                loss_residual_gmm_prefactor
            ),
            "loss_L2_prefactor": loss_L2_prefactor,
            "loss_gauge_prefactor": loss_gauge_prefactor,
            **loss_prefactors,
        }
        enable_loss_pareto_k = abs(loss_pareto_k_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        enable_loss_residual_gmm = (
            _residual_gmm_loss_enabled(loss_prefactors)
        )
        residual_gmm_targets = (
            _residual_gmm_targets(train_cfg)
            if enable_loss_residual_gmm
            else ()
        )
        enable_loss_L2 = abs(loss_L2_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        enable_loss_gauge = abs(loss_gauge_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        enable_loss_ess = abs(loss_ess_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        selected_pareto_k_terms = _selected_pareto_k_terms(train_cfg) if enable_loss_pareto_k else ()
        lr_schedule = make_lr_schedule(self.config)
        training_t_end = format_end_time(
            t0=float(train_cfg.get("t0", 0.0)),
            dt=float(train_cfg["dt"]),
            n_steps=int(train_cfg["N_steps"]),
            n_windows=int(train_cfg["N_windows"]),
        )
        n0 = float(lat_cfg["n0"])
        U = float(lat_cfg["U"])
        gamma = float(lat_cfg["gamma"])
        F = complex(lat_cfg["F_real"] + 1j * lat_cfg.get("F_imag", 0.0))
        Delta = float(lat_cfg["Delta"])
        dt = float(train_cfg["dt"])
        selected_residual_gmm_terms = (
            _selected_residual_gmm_terms(train_cfg)
            if enable_loss_residual_gmm
            else ()
        )
        history = TrainingHistoryBuffer.create(
            selected_residual_gmm_terms
        )
        runtime_real_dtype = str(jnp.asarray(0.0, dtype=DTYPE).dtype)
        runtime_complex_dtype = str(jnp.asarray(0.0 + 0.0j, dtype=CDTYPE).dtype)
        nn_dtype = str(jnp.asarray(0.0, dtype=NNDTYPE).dtype)

        print(f"training lattice sites = {self.lattice.num_site}", flush=True)
        print(
            (
                "training data precision = "
                f"real {runtime_real_dtype}, complex {runtime_complex_dtype}, nn {nn_dtype}"
            ),
            flush=True,
        )
        for line in self.model.config_summary_lines():
            print(line, flush=True)
        print(f"model parameters = {count_params(state.params)}", flush=True)
        print(f"model initialization = {self.parameter_initialization}", flush=True)
        print(f"checkpoint path = {self.params_path}", flush=True)
        solver_iteration_suffix = (
            f" (max iterations={sde_max_iter})"
            if "max_iterations" in sde_control_metadata["active_controls"]
            else ""
        )
        print(f"training SDE solver = {sde_solver}{solver_iteration_suffix}", flush=True)
        print(
            "training SDE solver controls = "
            f"{format_sde_solver_controls(sde_control_metadata)}",
            flush=True,
        )
        if enable_loss_residual_gmm:
            print(
                _format_residual_gmm_setup(
                    targets=residual_gmm_targets,
                    terms=selected_residual_gmm_terms,
                    num_site=self.lattice.num_site,
                    prefactor=loss_residual_gmm_prefactor,
                    integrator_nodes=(
                        residual_gmm_integrator_nodes
                    ),
                    d_clip=residual_gmm_d_clip,
                    cov_floor=residual_gmm_cov_floor,
                    cov_shrinkage=residual_gmm_cov_shrinkage,
                    trace_mode=residual_gmm_trace_mode,
                    time_aggregation=residual_gmm_time_aggregation,
                    time_beta=residual_gmm_time_beta,
                ),
                flush=True,
            )
        print(
            (
                f"training with N_steps = {int(train_cfg['N_steps'])}, "
                f"N_windows = {int(train_cfg['N_windows'])}"
            ),
            flush=True,
        )
        if apply_neural_gauge_every_steps > 0:
            print(
                f"training neural gauge refresh = every {apply_neural_gauge_every_steps} SDE step(s)",
                flush=True,
            )
        else:
            print("training neural gauge refresh = window start only", flush=True)
        print(
            f"training neural gauge state gradient = {neural_gauge_state_gradient}",
            flush=True,
        )
        if enable_loss_pareto_k:
            print(
                "training Pareto-k quantities = onsite monomial orders "
                f"{_selector_text(pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials)}",
                flush=True,
            )
            print(
                (
                    "training Pareto-k = "
                    f"threshold={pareto_k_threshold:g}, "
                    f"log_radius_slack={pareto_k_threshold_tau:g}, "
                    f"envelope_beta={pareto_k_envelope_beta:g}, "
                    f"envelope_excess={pareto_k_envelope_excess}, "
                    f"tail_fraction={pareto_k_tail_fraction:g}, "
                    f"min_tail_count={pareto_k_min_tail_count}, "
                    "channels=onsite-per-site"
                ),
                flush=True,
            )
        if enable_loss_pareto_k:
            print(
                self._pareto_k_monomial_channel_summary(
                    pareto_k_applied_quantities,
                    pareto_k_applied_quantities_mode,
                    pareto_k_monomials,
                ),
                flush=True,
            )
        print(f"loss-term trace threshold = {LOSS_TERM_TRACE_THRESHOLD:g}", flush=True)
        print(f"training rollout-noise refresh every = {noise_refresh_every} epoch(s)", flush=True)
        print(f"training up to time {training_t_end}", flush=True)
        print(loss_ema.summary(), flush=True)
        multi_device_spec = resolve_multi_device_spec(
            train_cfg,
            num_walker,
            purpose="training",
        )
        if multi_device_spec.enabled:
            print(
                (
                    "multi-device training = enabled "
                    f"with {multi_device_spec.num_devices} device(s), "
                    f"{multi_device_spec.walkers_per_device} walker(s)/device"
                ),
                flush=True,
            )
        else:
            print("multi-device training = disabled", flush=True)

        skipped_epoch_count = 0
        residual_gmm_covariance_update_count = 0
        uniform_window_weights = jnp.full(
            (int(train_cfg["N_windows"]),),
            jnp.asarray(1.0 / max(int(train_cfg["N_windows"]), 1), dtype=DTYPE),
            dtype=DTYPE,
        )
        # Per-window weight-entropy budget: the total complex-ESS budget
        # ln(num_walker) spread uniformly over the trained horizon.
        loss_ess_window_budget = float(
            np.log(max(int(train_cfg["num_walker"]), 2))
            / max(int(train_cfg["N_windows"]), 1)
        )
        if enable_loss_ess:
            print(
                (
                    "training ESS budget = "
                    f"ln({int(train_cfg['num_walker'])}) / "
                    f"{int(train_cfg['N_windows'])} windows = "
                    f"{loss_ess_window_budget:.4f} nats/window"
                ),
                flush=True,
            )
        residual_gmm_channel_dimension = 2 * (
            projected_residual_objective_channel_count(
                residual_gmm_targets,
                trace_mode=residual_gmm_trace_mode,
            )
            if enable_loss_residual_gmm
            else 0
        )
        if enable_loss_residual_gmm:
            (
                residual_gmm_covariance_bank,
                residual_gmm_covariance_initialized,
            ) = _initialize_residual_gmm_covariance_ema(
                (int(train_cfg["N_windows"]),),
                residual_gmm_channel_dimension,
            )
        else:
            residual_gmm_covariance_bank = None
            residual_gmm_covariance_initialized = None
        multi_grad_runner = None
        if multi_device_spec.enabled:
            multi_grad_runner = MultiDeviceGradientComputer(
                apply_fn=state.apply_fn,
                N_steps=int(train_cfg["N_steps"]),
                N_windows=int(train_cfg["N_windows"]),
                apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
                neural_gauge_each_apply=neural_gauge_each_apply,
                sde_max_iter=sde_max_iter,
                sde_solver=sde_solver,
                sde_root_rtol=sde_root_rtol,
                sde_root_atol=sde_root_atol,
                sde_affine_expm_order=sde_affine_expm_order,
                sde_affine_expm_substeps=sde_affine_expm_substeps,
                sde_newton_damping_steps=sde_newton_damping_steps,
                gauge_mode=train_cfg["gauge_mode"],
                neural_gauge_components=neural_gauge_components,
                pareto_k_applied_quantities=pareto_k_applied_quantities,
                pareto_k_applied_quantities_mode=pareto_k_applied_quantities_mode,
                pareto_k_monomials=pareto_k_monomials,
                pareto_k_envelope_excess=pareto_k_envelope_excess,
                operator_monomials=residual_gmm_targets,
                residual_gmm_trace_mode=residual_gmm_trace_mode,
                residual_gmm_time_aggregation=(
                    residual_gmm_time_aggregation
                ),
                residual_gmm_integrator_nodes=(
                    residual_gmm_integrator_nodes
                ),
                pareto_k_tail_fraction=pareto_k_tail_fraction,
                pareto_k_min_tail_count=pareto_k_min_tail_count,
                enable_loss_pareto_k=enable_loss_pareto_k,
                enable_loss_residual_gmm=(
                    enable_loss_residual_gmm
                ),
                enable_loss_gauge=enable_loss_gauge,
                enable_loss_L2=enable_loss_L2,
                enable_loss_ess=enable_loss_ess,
                spec=multi_device_spec,
            )
        batch_lnOmega0 = None
        batch_alpha0 = None
        batch_beta0 = None
        batch_lnOmega0_sharded = None
        batch_alpha0_sharded = None
        batch_beta0_sharded = None
        batch_rollout_key = None
        batch_rollout_keys_sharded = None
        force_refresh_rollout_batch = True
        for epoch in range(n_epoch):
            epoch_start_state = state
            if force_refresh_rollout_batch or (epoch % noise_refresh_every) == 0:
                batch_lnOmega0, batch_alpha0, batch_beta0 = self.lattice.initialize_phase_space(num_walker, n0=n0)
                batch_rollout_key = keygen.next(fold_in_value=1_000_000 + epoch // noise_refresh_every)
                if multi_device_spec.enabled:
                    batch_lnOmega0_sharded = shard_walkers(batch_lnOmega0, multi_device_spec)
                    batch_alpha0_sharded = shard_walkers(batch_alpha0, multi_device_spec)
                    batch_beta0_sharded = shard_walkers(batch_beta0, multi_device_spec)
                    batch_rollout_keys_sharded = split_device_keys(batch_rollout_key, multi_device_spec)
                force_refresh_rollout_batch = False
            epoch_wall_start = time.perf_counter()
            effective_prefactors = loss_ema.effective_prefactors(base_loss_prefactors)
            effective_pareto_k_monomial_weights = loss_ema.pareto_k_monomial_weight_matrix(
                self.lattice.num_site
            )
            if multi_device_spec.enabled:
                grads, aux = multi_grad_runner.gradients(
                    batch_rollout_keys_sharded,
                    state.params,
                    batch_lnOmega0_sharded,
                    batch_alpha0_sharded,
                    batch_beta0_sharded,
                    DTYPE(U),
                    DTYPE(gamma),
                    CDTYPE(F),
                    DTYPE(Delta),
                    DTYPE(dt),
                    uniform_window_weights,
                    DTYPE(0.0),
                    residual_gmm_covariance_bank,
                    residual_gmm_covariance_initialized,
                    DTYPE(train_cfg.get("t0", 0.0)),
                    DTYPE(train_cfg["gauge_scale"]),
                    DTYPE(pareto_k_threshold),
                    DTYPE(pareto_k_threshold_tau),
                    DTYPE(pareto_k_envelope_beta),
                    DTYPE(residual_gmm_d_clip),
                    DTYPE(residual_gmm_cov_floor),
                    DTYPE(residual_gmm_cov_shrinkage),
                    DTYPE(residual_gmm_time_beta),
                    DTYPE(n0),
                    DTYPE(effective_prefactors["loss_pareto_k_prefactor"]),
                    effective_pareto_k_monomial_weights,
                    DTYPE(
                        effective_prefactors[
                            "loss_residual_gmm_prefactor"
                        ]
                    ),
                    DTYPE(effective_prefactors["loss_L2_prefactor"]),
                    DTYPE(effective_prefactors["loss_gauge_prefactor"]),
                    DTYPE(loss_ess_prefactor),
                    DTYPE(loss_ess_window_budget),
                    float(train_cfg.get("q_winsor", 0.95)),
                    self.lattice.hopping_operator(),
                    DTYPE(0.0),
                )
            else:
                grads, aux, _ = compute_grads_all_windows(
                    key=batch_rollout_key,
                    apply_fn=state.apply_fn,
                    params=state.params,
                    lnOmega0=batch_lnOmega0,
                    alpha0=batch_alpha0,
                    beta0=batch_beta0,
                    U=DTYPE(U),
                    gamma=DTYPE(gamma),
                    F=CDTYPE(F),
                    Delta=DTYPE(Delta),
                    dt=DTYPE(dt),
                    N_steps=int(train_cfg["N_steps"]),
                    N_windows=int(train_cfg["N_windows"]),
                    window_loss_weights=uniform_window_weights,
                    lnOmega_shift0=DTYPE(0.0),
                    operator_monomials=residual_gmm_targets,
                    residual_gmm_trace_mode=residual_gmm_trace_mode,
                    residual_gmm_covariance_bank=(
                        residual_gmm_covariance_bank
                    ),
                    residual_gmm_covariance_initialized=(
                        residual_gmm_covariance_initialized
                    ),
                    apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
                    neural_gauge_each_apply=neural_gauge_each_apply,
                    sde_max_iter=sde_max_iter,
                    sde_solver=sde_solver,
                    sde_root_rtol=DTYPE(sde_root_rtol),
                    sde_root_atol=DTYPE(sde_root_atol),
                    sde_affine_expm_order=sde_affine_expm_order,
                    sde_affine_expm_substeps=sde_affine_expm_substeps,
                    sde_newton_damping_steps=sde_newton_damping_steps,
                    t0=DTYPE(train_cfg.get("t0", 0.0)),
                    gauge_weight=DTYPE(train_cfg["gauge_scale"]),
                    pareto_k_threshold=DTYPE(pareto_k_threshold),
                    pareto_k_threshold_tau=DTYPE(pareto_k_threshold_tau),
                    pareto_k_envelope_beta=DTYPE(pareto_k_envelope_beta),
                    pareto_k_envelope_excess=pareto_k_envelope_excess,
                    residual_gmm_d_clip=DTYPE(
                        residual_gmm_d_clip
                    ),
                    residual_gmm_cov_floor=DTYPE(
                        residual_gmm_cov_floor
                    ),
                    residual_gmm_cov_shrinkage=DTYPE(
                        residual_gmm_cov_shrinkage
                    ),
                    residual_gmm_time_aggregation=(
                        residual_gmm_time_aggregation
                    ),
                    residual_gmm_integrator_nodes=(
                        residual_gmm_integrator_nodes
                    ),
                    residual_gmm_time_beta=DTYPE(
                        residual_gmm_time_beta
                    ),
                    pareto_k_tail_fraction=pareto_k_tail_fraction,
                    pareto_k_min_tail_count=pareto_k_min_tail_count,
                    n0=DTYPE(n0),
                    loss_pareto_k_prefactor=DTYPE(effective_prefactors["loss_pareto_k_prefactor"]),
                    loss_pareto_k_monomial_weights=effective_pareto_k_monomial_weights,
                    loss_residual_gmm_prefactor=DTYPE(
                        effective_prefactors[
                            "loss_residual_gmm_prefactor"
                        ]
                    ),
                    loss_L2_prefactor=DTYPE(effective_prefactors["loss_L2_prefactor"]),
                    loss_gauge_prefactor=DTYPE(effective_prefactors["loss_gauge_prefactor"]),
                    loss_ess_prefactor=DTYPE(loss_ess_prefactor),
                    loss_ess_window_budget=DTYPE(loss_ess_window_budget),
                    q_winsor=float(train_cfg.get("q_winsor", 0.95)),
                    pareto_k_applied_quantities=pareto_k_applied_quantities,
                    pareto_k_applied_quantities_mode=pareto_k_applied_quantities_mode,
                    pareto_k_monomials=pareto_k_monomials,
                    J=self.lattice.hopping_operator(),
                    gauge_mode=train_cfg["gauge_mode"],
                    neural_gauge_components=neural_gauge_components,
                    analytic_target_time=DTYPE(0.0),
                    enable_loss_pareto_k=enable_loss_pareto_k,
                    enable_loss_residual_gmm=(
                        enable_loss_residual_gmm
                    ),
                    enable_loss_gauge=enable_loss_gauge,
                    enable_loss_L2=enable_loss_L2,
                    enable_loss_ess=enable_loss_ess,
                )
            state_candidate = state.apply_gradients(grads=grads)
            jax.block_until_ready(aux["loss"])
            epoch_time_sec = time.perf_counter() - epoch_wall_start
            aux["epoch_time_sec"] = jnp.asarray(epoch_time_sec, dtype=DTYPE)

            if tree_has_nonfinite(
                {"params": state_candidate.params, "metrics": aux, "grads": grads}
            ):
                state = epoch_start_state
                skipped_epoch_count += 1
                force_refresh_rollout_batch = True
                print(
                    (
                        f"epoch {epoch + 1:6d} | skipped due to non-finite values; "
                        f"keeping previous parameters"
                    ),
                    flush=True,
                )
                continue

            if (
                residual_gmm_covariance_update_count
                < RESIDUAL_GMM_COVARIANCE_UPDATE_EPOCHS
                and enable_loss_residual_gmm
            ):
                (
                    residual_gmm_covariance_bank,
                    residual_gmm_covariance_initialized,
                ) = _update_residual_gmm_covariance_ema(
                    residual_gmm_covariance_bank,
                    residual_gmm_covariance_initialized,
                    aux["residual_gmm_covariance_estimates"],
                    uniform_window_weights,
                )
            if enable_loss_residual_gmm:
                residual_gmm_covariance_update_count += 1
            state = state_candidate
            loss_ema.annotate_aux(aux)
            loss_ema.update(aux)
            lr_value = float(lr_schedule(epoch))
            latest_row = history.append(epoch + 1, aux, lr_value)

            if (epoch + 1) % log_every == 0 or epoch == 0 or (epoch + 1) == n_epoch:
                print(
                    _format_training_progress_line(
                        f"epoch {epoch + 1:6d}/{n_epoch}",
                        latest_row,
                        selected_residual_gmm_terms=(
                            selected_residual_gmm_terms
                        ),
                        selected_pareto_k_terms=selected_pareto_k_terms,
                        enable_loss_pareto_k=enable_loss_pareto_k,
                        enable_loss_residual_gmm=(
                            enable_loss_residual_gmm
                        ),
                        enable_loss_gauge=enable_loss_gauge,
                        enable_loss_L2=enable_loss_L2,
                        enable_loss_ess=enable_loss_ess,
                    ),
                    flush=True,
                )

            should_save_checkpoint = save_every > 0 and (
                (epoch + 1) % save_every == 0 or (epoch + 1) == n_epoch
            )
            should_plot = make_plots and plot_every > 0 and (
                (epoch + 1) % plot_every == 0 or (epoch + 1) == n_epoch
            )
            should_update_monitoring = should_save_checkpoint or should_plot

            if should_save_checkpoint:
                self._persist_checkpoint(state)
            if should_update_monitoring:
                self._persist_monitoring_outputs(state, history, make_plot=should_plot)

        self._persist_checkpoint(state)
        self._persist_monitoring_outputs(state, history, make_plot=make_plots)
        if skipped_epoch_count > 0:
            print(f"skipped epochs due to non-finite values = {skipped_epoch_count}", flush=True)

        return TrainingArtifacts(
            state=state,
            history=history.as_dict(),
            params_path=self.params_path,
            history_npz_path=self.history_npz_path,
            history_json_path=self.history_json_path,
            metadata_path=self.metadata_path,
            history_plot_png_path=self.history_plot_png_path,
            history_plot_pdf_path=self.history_plot_pdf_path,
        )

    def fit_segmented(self):
        train_cfg = self.config["training"]
        lat_cfg = self.config["lattice"]
        segmented_cfg = train_cfg["segmented_overlap"]
        stage_specs = self._load_segmented_stage_specs()

        total_epoch_budget = sum(stage.update_budget for stage in stage_specs)
        log_every = max(1, int(train_cfg.get("log_every", max(1, total_epoch_budget // 20))))
        save_every = int(train_cfg["save_every"])
        plot_every = int(train_cfg.get("plot_every", save_every if save_every > 0 else 0))
        make_plots = bool(train_cfg.get("make_plots", True))
        sde_max_iter = int(train_cfg.get("sde_max_iter", 4))
        sde_solver = normalize_sde_solver(train_cfg.get("sde_solver"))
        sde_root_rtol = float(train_cfg["sde_root_rtol"])
        sde_root_atol = float(train_cfg["sde_root_atol"])
        sde_affine_expm_order = int(train_cfg["sde_affine_expm_order"])
        sde_affine_expm_substeps = int(train_cfg["sde_affine_expm_substeps"])
        sde_newton_damping_steps = int(train_cfg["sde_newton_damping_steps"])
        sde_control_metadata = sde_solver_control_metadata(
            sde_solver,
            max_iterations=sde_max_iter,
            root_rtol=sde_root_rtol,
            root_atol=sde_root_atol,
            affine_expm_order=sde_affine_expm_order,
            affine_expm_substeps=sde_affine_expm_substeps,
            newton_damping_steps=sde_newton_damping_steps,
        )
        overlap_windows = int(segmented_cfg.get("segment_overlap_windows", 1))
        max_bank_refresh_failures = int(segmented_cfg.get("max_bank_refresh_failures", 8))
        auto_health_monomials = _auto_health_monomials_from_training_config(train_cfg)
        pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials = _selector_from_training_config(
            train_cfg,
            prefix="pareto_k",
            default_order=6,
            max_order=6,
            default_monomials=auto_health_monomials,
        )
        pareto_k_threshold = float(train_cfg.get("pareto_k_threshold", 0.7))
        pareto_k_threshold_tau = float(train_cfg.get("pareto_k_threshold_tau", 0.1))
        pareto_k_envelope_beta = float(train_cfg.get("pareto_k_envelope_beta", 0.5))
        pareto_k_envelope_excess = str(train_cfg.get("pareto_k_envelope_excess", "log"))
        pareto_k_tail_fraction = float(train_cfg.get("pareto_k_tail_fraction", 0.01))
        pareto_k_min_tail_count = int(train_cfg.get("pareto_k_min_tail_count", 32))
        residual_gmm_integrator_nodes = int(
            train_cfg.get("residual_gmm_integrator_nodes", 6)
        )
        residual_gmm_d_clip = float(
            train_cfg.get("residual_gmm_d_clip", 10.0)
        )
        residual_gmm_cov_floor = float(
            train_cfg.get("residual_gmm_cov_floor", 1.0e-8)
        )
        residual_gmm_cov_shrinkage = float(
            train_cfg.get("residual_gmm_cov_shrinkage", 0.05)
        )
        residual_gmm_trace_mode = normalize_residual_gmm_trace_mode(
            train_cfg.get(
                "residual_gmm_trace_mode",
                DEFAULT_RESIDUAL_GMM_TRACE_MODE,
            )
        )
        residual_gmm_time_aggregation = str(
            train_cfg.get("residual_gmm_time_aggregation", "mean")
        ).strip().lower()
        residual_gmm_time_beta = float(
            train_cfg.get("residual_gmm_time_beta", 2.0)
        )
        neural_gauge_components = normalize_neural_gauge_components(
            train_cfg.get("neural_gauge_components", "both")
        )
        apply_neural_gauge_every_steps = int(train_cfg.get("apply_neural_gauge_every_steps", 0))
        neural_gauge_state_gradient = str(train_cfg.get("neural_gauge_state_gradient", "full"))
        neural_gauge_each_apply = neural_gauge_state_gradient == "each_apply"
        loss_pareto_k_prefactor = float(train_cfg.get("loss_pareto_k_prefactor", 0.0))
        loss_residual_gmm_prefactor = float(
            train_cfg.get("loss_residual_gmm_prefactor", 0.0)
        )
        loss_L2_prefactor = float(train_cfg.get("loss_L2_prefactor", 0.0))
        loss_gauge_prefactor = float(train_cfg.get("loss_gauge_prefactor", 0.0))
        loss_ess_prefactor = float(train_cfg.get("loss_ess_prefactor", 0.0))
        loss_prefactors = _resolve_loss_prefactors(train_cfg)
        loss_ema = LossEmaNormalizer.from_training_config(train_cfg)
        base_loss_prefactors = {
            "loss_pareto_k_prefactor": loss_pareto_k_prefactor,
            "loss_residual_gmm_prefactor": (
                loss_residual_gmm_prefactor
            ),
            "loss_L2_prefactor": loss_L2_prefactor,
            "loss_gauge_prefactor": loss_gauge_prefactor,
            **loss_prefactors,
        }
        enable_loss_pareto_k = abs(loss_pareto_k_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        enable_loss_residual_gmm = (
            _residual_gmm_loss_enabled(loss_prefactors)
        )
        residual_gmm_targets = (
            _residual_gmm_targets(train_cfg)
            if enable_loss_residual_gmm
            else ()
        )
        enable_loss_L2 = abs(loss_L2_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        enable_loss_gauge = abs(loss_gauge_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        enable_loss_ess = abs(loss_ess_prefactor) >= LOSS_TERM_TRACE_THRESHOLD
        selected_pareto_k_terms = _selected_pareto_k_terms(train_cfg) if enable_loss_pareto_k else ()

        n0 = float(lat_cfg["n0"])
        U = float(lat_cfg["U"])
        gamma = float(lat_cfg["gamma"])
        F = complex(lat_cfg["F_real"] + 1j * lat_cfg.get("F_imag", 0.0))
        Delta = float(lat_cfg["Delta"])
        dt = float(train_cfg["dt"])
        selected_residual_gmm_terms = (
            _selected_residual_gmm_terms(train_cfg)
            if enable_loss_residual_gmm
            else ()
        )
        keygen = KeyGenerator(train_cfg["seed"])
        history = TrainingHistoryBuffer.create(
            selected_residual_gmm_terms
        )
        stage_histories = []
        state = None
        printed_summary = False
        base_load_parameters = bool(train_cfg.get("load_parameters", False))
        had_params_path = "params_path" in train_cfg
        base_params_path = train_cfg.get("params_path")

        global_epoch = 0
        skipped_update_count = 0
        for stage_index, stage in enumerate(stage_specs, start=1):
            n_windows_per_segment = int(stage.n_windows_per_segment)
            stride_windows = n_windows_per_segment - overlap_windows
            effective_total_windows = n_windows_per_segment + max(0, int(stage.n_segments) - 1) * stride_windows
            anchor_total_windows = max(0, (int(stage.n_segments) - 1) * stride_windows)
            segment_window_weights, segment_window_weight_sums = self._build_segment_window_weighting(
                n_segments=int(stage.n_segments),
                n_windows_per_segment=n_windows_per_segment,
                stride_windows=stride_windows,
            )
            # Per-window weight-entropy budget over the stage's full trained
            # horizon; every segment shares the same per-window allowance.
            loss_ess_window_budget = float(
                np.log(max(int(stage.num_walker), 2))
                / max(int(effective_total_windows), 1)
            )
            if enable_loss_ess:
                print(
                    (
                        f"stage {stage.stage_id} ESS budget = "
                        f"ln({int(stage.num_walker)}) / "
                        f"{int(effective_total_windows)} windows = "
                        f"{loss_ess_window_budget:.4f} nats/window"
                    ),
                    flush=True,
                )
            residual_gmm_channel_dimension = 2 * (
                projected_residual_objective_channel_count(
                    residual_gmm_targets,
                    trace_mode=residual_gmm_trace_mode,
                )
                if enable_loss_residual_gmm
                else 0
            )
            if enable_loss_residual_gmm:
                (
                    residual_gmm_covariance_bank,
                    residual_gmm_covariance_initialized,
                ) = _initialize_residual_gmm_covariance_ema(
                    (
                        int(stage.n_segments),
                        n_windows_per_segment,
                    ),
                    residual_gmm_channel_dimension,
                )
            else:
                residual_gmm_covariance_bank = None
                residual_gmm_covariance_initialized = None
            stage_t_end = format_end_time(
                t0=float(train_cfg.get("t0", 0.0)),
                dt=dt,
                n_steps=int(stage.n_steps),
                n_windows=effective_total_windows,
            )
            stage_cfg = copy.deepcopy(self.config)
            stage_train_cfg = stage_cfg["training"]
            stage_train_cfg["n_epoch"] = int(stage.update_budget)
            stage_train_cfg["num_walker"] = int(stage.num_walker)
            stage_train_cfg["N_steps"] = int(stage.n_steps)
            stage_train_cfg["N_windows"] = int(effective_total_windows)
            stage_train_cfg["lr_schedule_step_divisor"] = 1
            stage_train_cfg["load_parameters"] = _should_load_stage_parameters(
                stage.stage_id,
                base_load_parameters,
            )
            if stage_index > 1:
                stage_train_cfg["params_path"] = self.params_path
            elif had_params_path:
                stage_train_cfg["params_path"] = base_params_path
            else:
                stage_train_cfg.pop("params_path", None)
            stage_train_cfg["segmented_overlap"]["active_stage"] = {
                "stage_id": int(stage.stage_id),
                "n_epoch": int(stage.update_budget),
                "n_segments": int(stage.n_segments),
                "n_windows_per_segment": int(stage.n_windows_per_segment),
                "segment_overlap_windows": int(overlap_windows),
                "stride_windows": int(stride_windows),
                "effective_total_windows": int(effective_total_windows),
                "anchor_total_windows": int(anchor_total_windows),
                "mode": "segmented_overlap",
            }
            state = self._build_state_for_config(
                stage_cfg,
                keygen.next(fold_in_value=10_000 + int(stage.stage_id)),
            )
            if stage_index == 1:
                # The first stage may warm-start from the current save
                # directory. Load it before clean_start removes that directory.
                self._prepare_io()
            stage_lr_schedule = make_lr_schedule(stage_cfg)

            if not printed_summary:
                runtime_real_dtype = str(jnp.asarray(0.0, dtype=DTYPE).dtype)
                runtime_complex_dtype = str(jnp.asarray(0.0 + 0.0j, dtype=CDTYPE).dtype)
                nn_dtype = str(jnp.asarray(0.0, dtype=NNDTYPE).dtype)
                print(f"training lattice sites = {self.lattice.num_site}", flush=True)
                print(
                    (
                        "training data precision = "
                        f"real {runtime_real_dtype}, complex {runtime_complex_dtype}, nn {nn_dtype}"
                    ),
                    flush=True,
                )
                for line in self.model.config_summary_lines():
                    print(line, flush=True)
                print(f"model parameters = {count_params(state.params)}", flush=True)
                print(f"model initialization = {self.parameter_initialization}", flush=True)
                print(f"checkpoint path = {self.params_path}", flush=True)
                print(f"training segmented stages = {len(stage_specs)}", flush=True)
                solver_iteration_suffix = (
                    f" (max iterations={sde_max_iter})"
                    if "max_iterations" in sde_control_metadata["active_controls"]
                    else ""
                )
                print(
                    f"training SDE solver = {sde_solver}{solver_iteration_suffix}",
                    flush=True,
                )
                print(
                    "training SDE solver controls = "
                    f"{format_sde_solver_controls(sde_control_metadata)}",
                    flush=True,
                )
                if enable_loss_residual_gmm:
                    print(
                        _format_residual_gmm_setup(
                            targets=residual_gmm_targets,
                            terms=selected_residual_gmm_terms,
                            num_site=self.lattice.num_site,
                            prefactor=loss_residual_gmm_prefactor,
                            integrator_nodes=(
                                residual_gmm_integrator_nodes
                            ),
                            d_clip=residual_gmm_d_clip,
                            cov_floor=residual_gmm_cov_floor,
                            cov_shrinkage=(
                                residual_gmm_cov_shrinkage
                            ),
                            trace_mode=residual_gmm_trace_mode,
                            time_aggregation=(
                                residual_gmm_time_aggregation
                            ),
                            time_beta=residual_gmm_time_beta,
                        ),
                        flush=True,
                    )
                if enable_loss_pareto_k:
                    print(
                        "training Pareto-k quantities = onsite monomial orders "
                        f"{_selector_text(pareto_k_applied_quantities, pareto_k_applied_quantities_mode, pareto_k_monomials)}",
                        flush=True,
                    )
                    print(
                        (
                            "training Pareto-k = "
                            f"threshold={pareto_k_threshold:g}, "
                            f"log_radius_slack={pareto_k_threshold_tau:g}, "
                            f"envelope_beta={pareto_k_envelope_beta:g}, "
                            f"envelope_excess={pareto_k_envelope_excess}, "
                            f"tail_fraction={pareto_k_tail_fraction:g}, "
                            f"min_tail_count={pareto_k_min_tail_count}, "
                            "channels=onsite-per-site"
                        ),
                        flush=True,
                    )
                if enable_loss_pareto_k:
                    print(
                        self._pareto_k_monomial_channel_summary(
                            pareto_k_applied_quantities,
                            pareto_k_applied_quantities_mode,
                            pareto_k_monomials,
                        ),
                        flush=True,
                    )
                print(f"loss-term trace threshold = {LOSS_TERM_TRACE_THRESHOLD:g}", flush=True)
                print("training rollout-noise refresh every = bank refresh once per outer epoch", flush=True)
                print(loss_ema.summary(), flush=True)
                print(
                    (
                        "segmented overlap training = enabled "
                        f"segment_overlap_windows={overlap_windows}"
                    ),
                    flush=True,
                )
                if apply_neural_gauge_every_steps > 0:
                    print(
                        f"training neural gauge refresh = every {apply_neural_gauge_every_steps} SDE step(s)",
                        flush=True,
                    )
                else:
                    print("training neural gauge refresh = window start only", flush=True)
                print(
                    f"training neural gauge state gradient = {neural_gauge_state_gradient}",
                    flush=True,
                )
                printed_summary = True

            print(
                (
                    f"stage {stage.stage_id}: outer_epochs={stage.update_budget}, "
                    f"n_segments={stage.n_segments}, N_steps={stage.n_steps}, "
                    f"n_windows_per_segment={n_windows_per_segment}, "
                    f"effective_total_windows={effective_total_windows}, "
                    f"t_end={stage_t_end}"
                ),
                flush=True,
            )
            if enable_loss_residual_gmm:
                print(
                    "\n".join(
                        _format_progress_rows(
                            f"stage {stage.stage_id} residual GMM",
                            [
                                f"window duration={dt * int(stage.n_steps):g}",
                                f"sites={int(self.lattice.num_site)}",
                                (
                                    "diagnostic channels/site="
                                    f"{len(selected_residual_gmm_terms)} complex"
                                ),
                                (
                                    "objective channels/site="
                                    f"{residual_gmm_channel_dimension // 2} complex "
                                    f"(trace mode={residual_gmm_trace_mode})"
                                ),
                            ],
                        )
                    ),
                    flush=True,
                )

            multi_device_spec = resolve_multi_device_spec(
                train_cfg,
                int(stage.num_walker),
                purpose="training.segmented_overlap",
            )
            if multi_device_spec.enabled:
                print(
                    (
                        "multi-device segmented training = enabled "
                        f"with {multi_device_spec.num_devices} device(s), "
                        f"{multi_device_spec.walkers_per_device} walker(s)/device"
                    ),
                    flush=True,
                )
            else:
                print("multi-device segmented training = disabled", flush=True)

            multi_bank_rollout = None
            multi_grad_runner = None
            if multi_device_spec.enabled:
                if anchor_total_windows > 0:
                    multi_bank_rollout = MultiDeviceSimulationHistoryRollout(
                        apply_fn=state.apply_fn,
                        N_steps=int(stage.n_steps),
                        N_windows=int(anchor_total_windows),
                        apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
                        sde_max_iter=sde_max_iter,
                        sde_solver=sde_solver,
                        sde_root_rtol=sde_root_rtol,
                        sde_root_atol=sde_root_atol,
                        sde_affine_expm_order=sde_affine_expm_order,
                        sde_affine_expm_substeps=sde_affine_expm_substeps,
                        sde_newton_damping_steps=sde_newton_damping_steps,
                        gauge_mode=train_cfg["gauge_mode"],
                        neural_gauge_components=neural_gauge_components,
                        spec=multi_device_spec,
                    )
                multi_grad_runner = MultiDeviceGradientComputer(
                    apply_fn=state.apply_fn,
                    N_steps=int(stage.n_steps),
                    N_windows=int(n_windows_per_segment),
                    apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
                    neural_gauge_each_apply=neural_gauge_each_apply,
                    sde_max_iter=sde_max_iter,
                    sde_solver=sde_solver,
                    sde_root_rtol=sde_root_rtol,
                    sde_root_atol=sde_root_atol,
                    sde_affine_expm_order=sde_affine_expm_order,
                    sde_affine_expm_substeps=sde_affine_expm_substeps,
                    sde_newton_damping_steps=sde_newton_damping_steps,
                    gauge_mode=train_cfg["gauge_mode"],
                    neural_gauge_components=neural_gauge_components,
                    pareto_k_applied_quantities=pareto_k_applied_quantities,
                    pareto_k_applied_quantities_mode=pareto_k_applied_quantities_mode,
                    pareto_k_monomials=pareto_k_monomials,
                    pareto_k_envelope_excess=pareto_k_envelope_excess,
                    operator_monomials=residual_gmm_targets,
                    residual_gmm_trace_mode=residual_gmm_trace_mode,
                    residual_gmm_time_aggregation=(
                        residual_gmm_time_aggregation
                    ),
                    residual_gmm_integrator_nodes=(
                        residual_gmm_integrator_nodes
                    ),
                    pareto_k_tail_fraction=pareto_k_tail_fraction,
                    pareto_k_min_tail_count=pareto_k_min_tail_count,
                    enable_loss_pareto_k=enable_loss_pareto_k,
                    enable_loss_residual_gmm=(
                        enable_loss_residual_gmm
                    ),
                    enable_loss_gauge=enable_loss_gauge,
                    enable_loss_L2=False,
                    enable_loss_ess=enable_loss_ess,
                    spec=multi_device_spec,
                )

            stage_history = TrainingHistoryBuffer.create(
                selected_residual_gmm_terms
            )
            completed_stage_epochs = 0
            bank_refresh_failures = 0
            while completed_stage_epochs < int(stage.update_budget):
                epoch_start_state = state
                outer_epoch = completed_stage_epochs + 1
                outer_epoch_wall_start = time.perf_counter()
                bank_lnOmega0, bank_alpha0, bank_beta0 = self.lattice.initialize_phase_space(
                    int(stage.num_walker),
                    n0=n0,
                )
                bank_key = keygen.next(fold_in_value=5_000_000 + 10_000 * stage.stage_id + outer_epoch)
                if anchor_total_windows <= 0:
                    bank_results = {
                        "times": np.asarray([float(train_cfg.get("t0", 0.0))], dtype=float),
                        "lnOmega_history": np.asarray(bank_lnOmega0)[None, ...],
                        "lnOmega_shift_history": np.asarray([0.0], dtype=float),
                        "alpha_history": np.asarray(bank_alpha0)[None, ...],
                        "beta_history": np.asarray(bank_beta0)[None, ...],
                    }
                elif multi_device_spec.enabled:
                    (
                        _bank_key_devices,
                        bank_lnOmega_history_sharded,
                        bank_alpha_history_sharded,
                        bank_beta_history_sharded,
                        bank_times_sharded,
                        bank_lnOmega_shift_history_sharded,
                    ) = multi_bank_rollout(
                        split_device_keys(bank_key, multi_device_spec),
                        state.params,
                        shard_walkers(bank_lnOmega0, multi_device_spec),
                        shard_walkers(bank_alpha0, multi_device_spec),
                        shard_walkers(bank_beta0, multi_device_spec),
                        DTYPE(U),
                        DTYPE(gamma),
                        CDTYPE(F),
                        DTYPE(Delta),
                        DTYPE(dt),
                        DTYPE(train_cfg.get("t0", 0.0)),
                        DTYPE(train_cfg["gauge_scale"]),
                        DTYPE(n0),
                        self.lattice.hopping_operator(),
                        DTYPE(0.0),
                    )
                    bank_results = {
                        "times": np.asarray(bank_times_sharded)[0],
                        "lnOmega_history": unshard_walker_history(bank_lnOmega_history_sharded),
                        "lnOmega_shift_history": np.asarray(
                            bank_lnOmega_shift_history_sharded
                        )[0],
                        "alpha_history": unshard_walker_history(bank_alpha_history_sharded),
                        "beta_history": unshard_walker_history(bank_beta_history_sharded),
                    }
                else:
                    bank_results = run_simulation_windows(
                        key=bank_key,
                        apply_fn=state.apply_fn,
                        params=state.params,
                        lnOmega0=bank_lnOmega0,
                        alpha0=bank_alpha0,
                        beta0=bank_beta0,
                        U=DTYPE(U),
                        gamma=DTYPE(gamma),
                        F=CDTYPE(F),
                        Delta=DTYPE(Delta),
                        dt=DTYPE(dt),
                        N_steps=int(stage.n_steps),
                        N_windows=int(anchor_total_windows),
                        t0=DTYPE(train_cfg.get("t0", 0.0)),
                        apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
                        gauge_weight=DTYPE(train_cfg["gauge_scale"]),
                        n0=DTYPE(n0),
                        sde_max_iter=sde_max_iter,
                        sde_solver=sde_solver,
                        sde_root_rtol=DTYPE(sde_root_rtol),
                        sde_root_atol=DTYPE(sde_root_atol),
                        sde_affine_expm_order=sde_affine_expm_order,
                        sde_affine_expm_substeps=sde_affine_expm_substeps,
                        sde_newton_damping_steps=sde_newton_damping_steps,
                        J=self.lattice.hopping_operator(),
                        gauge_mode=train_cfg["gauge_mode"],
                        neural_gauge_components=neural_gauge_components,
                        analytic_target_time=DTYPE(0.0),
                        progress_every_window=0,
                    )
                if not (
                    np.all(np.isfinite(bank_results["times"]))
                    and np.all(np.isfinite(bank_results["lnOmega_history"]))
                    and np.all(np.isfinite(bank_results["lnOmega_shift_history"]))
                    and np.all(np.isfinite(bank_results["alpha_history"]))
                    and np.all(np.isfinite(bank_results["beta_history"]))
                ):
                    bank_refresh_failures += 1
                    print(
                        f"stage {stage.stage_id} outer_epoch {outer_epoch}: bank generation produced non-finite values; retrying",
                        flush=True,
                    )
                    if bank_refresh_failures >= max_bank_refresh_failures:
                        raise RuntimeError(
                            f"Too many non-finite bank refreshes at stage {stage.stage_id}; aborting segmented training."
                        )
                    continue

                frozen_state = state
                effective_prefactors = loss_ema.effective_prefactors(base_loss_prefactors)
                effective_pareto_k_monomial_weights = loss_ema.pareto_k_monomial_weight_matrix(
                    self.lattice.num_site
                )
                segment_aux_rows = []
                segment_grads = []
                valid_segment_indices = []
                valid_segment_weight_sum = 0.0
                skipped_segments_this_epoch = 0
                for segment_local_index in range(int(stage.n_segments)):
                    segment_start_idx = int(segment_local_index * stride_windows)
                    t0_segment = float(bank_results["times"][segment_start_idx])
                    lnOmega_shift_segment = float(
                        bank_results["lnOmega_shift_history"][segment_start_idx]
                    )
                    lnOmega_segment = jnp.asarray(
                        bank_results["lnOmega_history"][segment_start_idx],
                        dtype=CDTYPE,
                    )
                    alpha_segment = jnp.asarray(
                        bank_results["alpha_history"][segment_start_idx],
                        dtype=CDTYPE,
                    )
                    beta_segment = jnp.asarray(
                        bank_results["beta_history"][segment_start_idx],
                        dtype=CDTYPE,
                    )
                    segment_window_weight = segment_window_weights[segment_local_index]
                    segment_weight_sum = float(segment_window_weight_sums[segment_local_index])
                    update_key = keygen.next(
                        fold_in_value=7_000_000 + 100_000 * stage.stage_id + 1_000 * outer_epoch + segment_local_index
                    )
                    update_wall_start = time.perf_counter()
                    if multi_device_spec.enabled:
                        grads, aux = multi_grad_runner.gradients(
                            split_device_keys(update_key, multi_device_spec),
                            frozen_state.params,
                            shard_walkers(lnOmega_segment, multi_device_spec),
                            shard_walkers(alpha_segment, multi_device_spec),
                            shard_walkers(beta_segment, multi_device_spec),
                            DTYPE(U),
                            DTYPE(gamma),
                            CDTYPE(F),
                            DTYPE(Delta),
                            DTYPE(dt),
                            segment_window_weight,
                            DTYPE(lnOmega_shift_segment),
                            (
                                None
                                if residual_gmm_covariance_bank is None
                                else residual_gmm_covariance_bank[
                                    segment_local_index
                                ]
                            ),
                            (
                                None
                                if residual_gmm_covariance_initialized is None
                                else residual_gmm_covariance_initialized[
                                    segment_local_index
                                ]
                            ),
                            DTYPE(t0_segment),
                            DTYPE(train_cfg["gauge_scale"]),
                            DTYPE(pareto_k_threshold),
                            DTYPE(pareto_k_threshold_tau),
                            DTYPE(pareto_k_envelope_beta),
                            DTYPE(residual_gmm_d_clip),
                            DTYPE(residual_gmm_cov_floor),
                            DTYPE(residual_gmm_cov_shrinkage),
                            DTYPE(residual_gmm_time_beta),
                            DTYPE(n0),
                            DTYPE(effective_prefactors["loss_pareto_k_prefactor"]),
                            effective_pareto_k_monomial_weights,
                            DTYPE(
                                effective_prefactors[
                                    "loss_residual_gmm_prefactor"
                                ]
                            ),
                            DTYPE(0.0),
                            DTYPE(effective_prefactors["loss_gauge_prefactor"]),
                            DTYPE(loss_ess_prefactor),
                            DTYPE(loss_ess_window_budget),
                            float(train_cfg.get("q_winsor", 0.95)),
                            self.lattice.hopping_operator(),
                            DTYPE(0.0),
                        )
                    else:
                        grads, aux, _ = compute_grads_all_windows(
                            key=update_key,
                            apply_fn=frozen_state.apply_fn,
                            params=frozen_state.params,
                            lnOmega0=lnOmega_segment,
                            alpha0=alpha_segment,
                            beta0=beta_segment,
                            U=DTYPE(U),
                            gamma=DTYPE(gamma),
                            F=CDTYPE(F),
                            Delta=DTYPE(Delta),
                            dt=DTYPE(dt),
                            N_steps=int(stage.n_steps),
                            N_windows=int(n_windows_per_segment),
                            window_loss_weights=segment_window_weight,
                            lnOmega_shift0=DTYPE(lnOmega_shift_segment),
                            operator_monomials=residual_gmm_targets,
                            residual_gmm_trace_mode=residual_gmm_trace_mode,
                            residual_gmm_covariance_bank=(
                                None
                                if residual_gmm_covariance_bank is None
                                else residual_gmm_covariance_bank[
                                    segment_local_index
                                ]
                            ),
                            residual_gmm_covariance_initialized=(
                                None
                                if residual_gmm_covariance_initialized is None
                                else residual_gmm_covariance_initialized[
                                    segment_local_index
                                ]
                            ),
                            apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
                            neural_gauge_each_apply=neural_gauge_each_apply,
                            sde_max_iter=sde_max_iter,
                            sde_solver=sde_solver,
                            sde_root_rtol=DTYPE(sde_root_rtol),
                            sde_root_atol=DTYPE(sde_root_atol),
                            sde_affine_expm_order=sde_affine_expm_order,
                            sde_affine_expm_substeps=sde_affine_expm_substeps,
                            sde_newton_damping_steps=sde_newton_damping_steps,
                            t0=DTYPE(t0_segment),
                            gauge_weight=DTYPE(train_cfg["gauge_scale"]),
                            pareto_k_threshold=DTYPE(pareto_k_threshold),
                            pareto_k_threshold_tau=DTYPE(pareto_k_threshold_tau),
                            pareto_k_envelope_beta=DTYPE(pareto_k_envelope_beta),
                            pareto_k_envelope_excess=pareto_k_envelope_excess,
                            residual_gmm_d_clip=DTYPE(
                                residual_gmm_d_clip
                            ),
                            residual_gmm_cov_floor=DTYPE(
                                residual_gmm_cov_floor
                            ),
                            residual_gmm_cov_shrinkage=DTYPE(
                                residual_gmm_cov_shrinkage
                            ),
                            residual_gmm_time_aggregation=(
                                residual_gmm_time_aggregation
                            ),
                            residual_gmm_integrator_nodes=(
                                residual_gmm_integrator_nodes
                            ),
                            residual_gmm_time_beta=DTYPE(
                                residual_gmm_time_beta
                            ),
                            pareto_k_tail_fraction=pareto_k_tail_fraction,
                            pareto_k_min_tail_count=pareto_k_min_tail_count,
                            n0=DTYPE(n0),
                            loss_pareto_k_prefactor=DTYPE(effective_prefactors["loss_pareto_k_prefactor"]),
                            loss_pareto_k_monomial_weights=effective_pareto_k_monomial_weights,
                            loss_residual_gmm_prefactor=DTYPE(
                                effective_prefactors[
                                    "loss_residual_gmm_prefactor"
                                ]
                            ),
                            loss_L2_prefactor=DTYPE(0.0),
                            loss_gauge_prefactor=DTYPE(effective_prefactors["loss_gauge_prefactor"]),
                            loss_ess_prefactor=DTYPE(loss_ess_prefactor),
                            loss_ess_window_budget=DTYPE(loss_ess_window_budget),
                            q_winsor=float(train_cfg.get("q_winsor", 0.95)),
                            pareto_k_applied_quantities=pareto_k_applied_quantities,
                            pareto_k_applied_quantities_mode=pareto_k_applied_quantities_mode,
                            pareto_k_monomials=pareto_k_monomials,
                            J=self.lattice.hopping_operator(),
                            gauge_mode=train_cfg["gauge_mode"],
                            neural_gauge_components=neural_gauge_components,
                            analytic_target_time=DTYPE(0.0),
                            enable_loss_pareto_k=enable_loss_pareto_k,
                            enable_loss_residual_gmm=(
                                enable_loss_residual_gmm
                            ),
                            enable_loss_gauge=enable_loss_gauge,
                            enable_loss_L2=False,
                            enable_loss_ess=enable_loss_ess,
                        )
                    jax.block_until_ready(aux["loss"])
                    aux["epoch_time_sec"] = jnp.asarray(time.perf_counter() - update_wall_start, dtype=DTYPE)
                    if tree_has_nonfinite({"grads": grads, "metrics": aux}):
                        skipped_update_count += 1
                        skipped_segments_this_epoch += 1
                        continue
                    segment_grads.append(grads)
                    segment_aux_rows.append(aux)
                    valid_segment_indices.append(segment_local_index)
                    valid_segment_weight_sum += segment_weight_sum
                if not segment_aux_rows or valid_segment_weight_sum <= 0.0:
                    bank_refresh_failures += 1
                    print(
                        f"stage {stage.stage_id} outer_epoch {outer_epoch}: "
                        "no valid frozen segments; retrying",
                        flush=True,
                    )
                    if bank_refresh_failures >= max_bank_refresh_failures:
                        raise RuntimeError(
                            f"Too many failed frozen-bank updates at stage {stage.stage_id}; aborting segmented training."
                        )
                    continue

                def _normalized_tree_sum(trees):
                    return jax.tree_util.tree_map(
                        lambda *values: (
                            jnp.sum(jnp.stack(values, axis=0), axis=0)
                            / DTYPE(valid_segment_weight_sum)
                        ),
                        *trees,
                    )

                # Each segment loss is already a sum over its effective window
                # weights. Sum segment gradients once, then normalize by the
                # total effective window weight.
                data_grads = _normalized_tree_sum(segment_grads)
                zero = jnp.asarray(0.0, dtype=DTYPE)
                if enable_loss_L2:
                    l2_loss, l2_grads = jax.value_and_grad(
                        lambda params: DTYPE(effective_prefactors["loss_L2_prefactor"]) * compute_l2_penalty(params)
                    )(frozen_state.params)
                    grads = jax.tree_util.tree_map(
                        lambda grad, l2_grad: grad + l2_grad,
                        data_grads,
                        l2_grads,
                    )
                else:
                    l2_loss = zero
                    grads = data_grads

                def _avg_aux(name: str, default=0.0):
                    segment_values = [
                        row.get(name, jnp.asarray(default, dtype=DTYPE))
                        for row in segment_aux_rows
                    ]
                    return (
                        jnp.sum(jnp.stack(segment_values, axis=0), axis=0)
                        / DTYPE(valid_segment_weight_sum)
                    )

                aux = {
                    "loss": _avg_aux("loss") + l2_loss,
                    "loss_pareto_k": _avg_aux("loss_pareto_k"),
                    "loss_pareto_k_objective": _avg_aux("loss_pareto_k_objective"),
                    "loss_pareto_k_terms": _avg_aux(
                        "loss_pareto_k_terms",
                        jnp.zeros((len(PARETO_K_MONOMIAL_TERMS),), dtype=DTYPE),
                    ),
                    "loss_pareto_k_site_terms": _avg_aux(
                        "loss_pareto_k_site_terms",
                        jnp.zeros((len(PARETO_K_MONOMIAL_TERMS), self.lattice.num_site), dtype=DTYPE),
                    ),
                    "pareto_k_mean": _avg_aux("pareto_k_mean"),
                    "pareto_k_max": _avg_aux("pareto_k_max"),
                    "pareto_k_worst": jnp.max(
                        jnp.stack([row.get("pareto_k_worst", row.get("pareto_k_max", zero)) for row in segment_aux_rows])
                    ),
                    "pareto_k_warning_fraction": _avg_aux("pareto_k_warning_fraction"),
                    "pareto_k_bad_fraction": _avg_aux("pareto_k_bad_fraction"),
                    "loss_residual_gmm": _avg_aux(
                        "loss_residual_gmm"
                    ),
                    "loss_residual_gmm_time": _avg_aux(
                        "loss_residual_gmm_time"
                    ),
                    "loss_residual_gmm_raw": _avg_aux(
                        "loss_residual_gmm_raw"
                    ),
                    "loss_residual_gmm_terms": _avg_aux(
                        "loss_residual_gmm_terms",
                        jnp.zeros(
                            (len(selected_residual_gmm_terms),),
                            dtype=DTYPE,
                        ),
                    ),
                    "loss_residual_gmm_site_terms": _avg_aux(
                        "loss_residual_gmm_site_terms",
                        jnp.zeros(
                            (
                                len(selected_residual_gmm_terms),
                                self.lattice.num_site,
                            ),
                            dtype=DTYPE,
                        ),
                    ),
                    "residual_gmm_z_mean": _avg_aux(
                        "residual_gmm_z_mean"
                    ),
                    "residual_gmm_z_max": _avg_aux(
                        "residual_gmm_z_max"
                    ),
                    "residual_gmm_z_worst": jnp.max(
                        jnp.stack(
                            [
                                row.get(
                                    "residual_gmm_z_worst",
                                    row.get(
                                        "residual_gmm_z_max",
                                        zero,
                                    ),
                                )
                                for row in segment_aux_rows
                            ]
                        )
                    ),
                    "residual_gmm_radius_mean": _avg_aux(
                        "residual_gmm_radius_mean"
                    ),
                    "residual_gmm_radius_max": _avg_aux(
                        "residual_gmm_radius_max"
                    ),
                    "residual_gmm_radius_worst": jnp.max(
                        jnp.stack(
                            [
                                row.get(
                                    "residual_gmm_radius_worst",
                                    row.get(
                                        "residual_gmm_radius_max",
                                        zero,
                                    ),
                                )
                                for row in segment_aux_rows
                            ]
                        )
                    ),
                    "residual_gmm_warning_fraction": _avg_aux(
                        "residual_gmm_warning_fraction"
                    ),
                    "residual_gmm_bad_fraction": _avg_aux(
                        "residual_gmm_bad_fraction"
                    ),
                    "loss_gauge": _avg_aux("loss_gauge"),
                    "loss_gauge_drift": _avg_aux("loss_gauge_drift"),
                    "loss_gauge_diffusion": _avg_aux("loss_gauge_diffusion"),
                    "loss_ess": _avg_aux("loss_ess"),
                    "loss_ess_weighted": _avg_aux("loss_ess_weighted"),
                    "log_weight_spread_mean": _avg_aux("log_weight_spread_mean"),
                    "log_weight_spread_max": jnp.max(
                        jnp.stack(
                            [
                                row.get("log_weight_spread_max", zero)
                                for row in segment_aux_rows
                            ]
                        )
                    ),
                    # Segments tile the trained horizon (up to the small
                    # overlap), so their per-segment spend totals ADD.
                    "log_weight_spread_total": jnp.sum(
                        jnp.stack(
                            [
                                row.get("log_weight_spread_total", zero)
                                for row in segment_aux_rows
                            ]
                        )
                    ),
                    "ess_ratio_min": jnp.min(
                        jnp.stack(
                            [
                                row.get(
                                    "ess_ratio_min",
                                    jnp.asarray(1.0, dtype=DTYPE),
                                )
                                for row in segment_aux_rows
                            ]
                        )
                    ),
                    # Per-segment scalar: plain mean over segments, not the
                    # window-weight normalization used for summed metrics.
                    "ess_ratio_end": jnp.mean(
                        jnp.stack(
                            [
                                row.get(
                                    "ess_ratio_end",
                                    jnp.asarray(1.0, dtype=DTYPE),
                                )
                                for row in segment_aux_rows
                            ]
                        )
                    ),
                    "loss_L2": l2_loss,
                    "grads_norm": opx.global_norm(grads),
                    "epoch_time_sec": jnp.asarray(time.perf_counter() - outer_epoch_wall_start, dtype=DTYPE),
                }
                for index, term in enumerate(PARETO_K_MONOMIAL_TERMS):
                    aux[term] = aux["loss_pareto_k_terms"][index]
                for index, term in enumerate(
                    selected_residual_gmm_terms
                ):
                    aux[term] = aux[
                        "loss_residual_gmm_terms"
                    ][index]

                lr_value = float(stage_lr_schedule(int(frozen_state.step)))
                state_candidate = frozen_state.apply_gradients(grads=grads)
                jax.block_until_ready(aux["loss"])
                if tree_has_nonfinite({"params": state_candidate.params, "metrics": aux, "grads": grads}):
                    state = epoch_start_state
                    bank_refresh_failures += 1
                    skipped_update_count += 1
                    print(
                        f"stage {stage.stage_id} outer_epoch {outer_epoch}: non-finite update; retrying",
                        flush=True,
                    )
                    if bank_refresh_failures >= max_bank_refresh_failures:
                        raise RuntimeError(
                            f"Too many failed frozen-bank updates at stage {stage.stage_id}; aborting segmented training."
                        )
                    continue

                if (
                    completed_stage_epochs
                    < RESIDUAL_GMM_COVARIANCE_UPDATE_EPOCHS
                    and enable_loss_residual_gmm
                ):
                    for segment_local_index, segment_aux in zip(
                        valid_segment_indices,
                        segment_aux_rows,
                    ):
                        updated_bank, updated_mask = (
                            _update_residual_gmm_covariance_ema(
                                residual_gmm_covariance_bank[
                                    segment_local_index
                                ],
                                residual_gmm_covariance_initialized[
                                    segment_local_index
                                ],
                                segment_aux[
                                    "residual_gmm_covariance_estimates"
                                ],
                                segment_window_weights[
                                    segment_local_index
                                ],
                            )
                        )
                        residual_gmm_covariance_bank = (
                            residual_gmm_covariance_bank.at[
                                segment_local_index
                            ].set(updated_bank)
                        )
                        residual_gmm_covariance_initialized = (
                            residual_gmm_covariance_initialized.at[
                                segment_local_index
                            ].set(updated_mask)
                        )
                state = state_candidate
                completed_stage_epochs += 1
                global_epoch += 1
                loss_ema.annotate_aux(aux)
                latest_row = history.append(global_epoch, aux, lr_value)
                stage_history.append(completed_stage_epochs, aux, lr_value)
                loss_ema.update(aux)

                if global_epoch % log_every == 0 or global_epoch == 1 or global_epoch == total_epoch_budget:
                    print(
                        _format_training_progress_line(
                            (
                                f"stage {stage.stage_id} epoch "
                                f"{completed_stage_epochs:6d}/{stage.update_budget} "
                                f"| segments={len(segment_aux_rows)}/{stage.n_segments} "
                                f"| skipped_segments={skipped_segments_this_epoch}"
                            ),
                            latest_row,
                            selected_residual_gmm_terms=(
                                selected_residual_gmm_terms
                            ),
                            selected_pareto_k_terms=selected_pareto_k_terms,
                            enable_loss_pareto_k=enable_loss_pareto_k,
                            enable_loss_residual_gmm=(
                                enable_loss_residual_gmm
                            ),
                            enable_loss_gauge=enable_loss_gauge,
                            enable_loss_L2=enable_loss_L2,
                            enable_loss_ess=enable_loss_ess,
                        ),
                        flush=True,
                    )

                should_save_checkpoint = save_every > 0 and (
                    global_epoch % save_every == 0 or completed_stage_epochs == stage.update_budget
                )
                should_plot = make_plots and plot_every > 0 and (
                    global_epoch % plot_every == 0 or completed_stage_epochs == stage.update_budget
                )
                should_update_monitoring = should_save_checkpoint or should_plot
                if should_save_checkpoint:
                    self._persist_checkpoint(state)
                if should_update_monitoring:
                    self._persist_monitoring_outputs(state, history, make_plot=should_plot)

            stage_histories.append(stage_history.as_dict())
            self._persist_checkpoint(state)
            self._persist_monitoring_outputs(state, history, make_plot=make_plots)
            self._persist_stage_snapshot(stage.stage_id, state, stage_history, stage_cfg, make_plot=make_plots)

        assert state is not None
        self._persist_checkpoint(state)
        self._persist_monitoring_outputs(state, history, make_plot=make_plots)
        if skipped_update_count > 0:
            print(f"skipped segmented updates due to non-finite values = {skipped_update_count}", flush=True)

        return TrainingArtifacts(
            state=state,
            history=history.as_dict(),
            params_path=self.params_path,
            history_npz_path=self.history_npz_path,
            history_json_path=self.history_json_path,
            metadata_path=self.metadata_path,
            history_plot_png_path=self.history_plot_png_path,
            history_plot_pdf_path=self.history_plot_pdf_path,
            stage_histories=stage_histories,
        )

    def fit_staged(self):
        """Run ordinary staged training through the same public trainer API."""

        stage_specs = self._load_staged_stage_specs()
        base_config = copy.deepcopy(self.config)
        base_train_cfg = base_config["training"]
        base_load_parameters = bool(base_train_cfg.get("load_parameters", False))
        had_params_path = "params_path" in base_train_cfg
        base_params_path = base_train_cfg.get("params_path")
        base_seed = int(base_train_cfg["seed"])
        artifacts = []

        for stage_index, stage in enumerate(stage_specs, start=1):
            stage_config = copy.deepcopy(base_config)
            stage_train_cfg = stage_config["training"]
            stage_train_cfg.pop("staged_schedule", None)
            stage_train_cfg["n_epoch"] = int(stage.n_epoch)
            stage_train_cfg["N_windows"] = int(stage.n_windows)
            stage_train_cfg["num_walker"] = int(stage.num_walker)
            stage_train_cfg["N_steps"] = int(stage.n_steps)
            # A staged continuation must not replay the preceding stage's
            # stochastic rollout stream.
            stage_train_cfg["seed"] = int(base_seed + stage_index - 1)
            stage_train_cfg["load_parameters"] = (
                _should_load_stage_parameters(
                    stage.stage_id,
                    base_load_parameters,
                )
            )
            if stage_index > 1:
                # Every continuation stage must load the preceding canonical
                # checkpoint, not an explicit warm-start path supplied for
                # the first stage.
                stage_train_cfg["params_path"] = self.params_path
            elif had_params_path:
                stage_train_cfg["params_path"] = base_params_path
            else:
                stage_train_cfg.pop("params_path", None)
            stage_config["io"]["clean_start"] = bool(
                base_config["io"]["clean_start"]
            ) if stage_index == 1 else False

            stage_paths = self._stage_output_paths(stage.stage_id)
            print(
                (
                    f"stage {stage.stage_id}: epochs={stage.n_epoch}, "
                    f"N_windows={stage.n_windows}, N_steps={stage.n_steps}, "
                    f"num_walker={stage.num_walker}"
                ),
                flush=True,
            )
            stage_trainer = type(self)(
                stage_config,
                config_path=stage_paths["config_used_path"],
            )
            artifact = stage_trainer._fit_one_rollout()
            stage_trainer._persist_stage_snapshot(
                stage.stage_id,
                artifact.state,
                artifact.history,
                stage_config,
                make_plot=bool(stage_train_cfg.get("make_plots", True)),
            )
            artifacts.append(artifact)

        return artifacts

    def fit(self):
        staged_schedule = self.config["training"].get("staged_schedule")
        segmented_cfg = self.config["training"].get("segmented_overlap")
        segmented_enabled = isinstance(segmented_cfg, dict) and bool(
            segmented_cfg.get("enabled", False)
        )
        if staged_schedule is not None and segmented_enabled:
            raise ValueError(
                "training.staged_schedule cannot be combined with enabled "
                "training.segmented_overlap"
            )
        if staged_schedule is not None:
            return self.fit_staged()
        if segmented_enabled:
            return self.fit_segmented()
        return self._fit_one_rollout()


__all__ = [
    "GaugeTrainer",
    "SegmentedStageSpec",
    "StagedStageSpec",
    "TrainingArtifacts",
    "TrainingHistoryBuffer",
]
