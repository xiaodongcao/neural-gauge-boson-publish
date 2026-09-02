from .analytical_gauge import compute_analytical_gauge_fields
from .lattice import apply_hopping_matrix, broadcast_site_param, conjugate_hopping_operator
from .lib_preinclude import *
from .postprocess import (
    centered_exponentiated_weights_for_axis,
    max_real_log_weight_for_axis,
    safe_complex_denominator,
    self_normalized_weight_ratio,
)
from .projected_residual import (
    CLOSED_NEWTON_COTES_RULES,
    normalize_projected_residual_monomials,
    normalize_residual_gmm_integrator_nodes,
)
from .utility import (
    NEURAL_GAUGE_MODES,
    OPERATOR_MOMENT_MAX_ORDER,
    OPERATOR_MOMENT_SPECS,
    PARETO_K_MONOMIAL_SPECS,
    PARETO_K_MONOMIAL_TERMS,
    PARETO_K_OBSERVABLE_SPECS,
    compute_l2_penalty,
    normalize_neural_gauge_components,
    selected_monomial_specs,
)
from .sde_solver import (
    GaugeFields,
    SolverCoefficients,
    SolverNoise,
    SolverState,
    get_solver,
)

PARETO_K_MONOMIAL_COUNT = len(PARETO_K_MONOMIAL_SPECS)


def _pareto_k_monomial_index(m_power: int, n_power: int) -> int:
    total_order = int(m_power) + int(n_power)
    return total_order * (total_order + 1) // 2 - 1 + int(m_power)


def _operator_moment_index(m_power: int, n_power: int) -> int:
    total_order = int(m_power) + int(n_power)
    return total_order * (total_order + 1) // 2 - 1 + int(m_power)


def _resolve_pareto_k_monomial_weights(weights, num_site: int) -> jnp.ndarray:
    values = jnp.asarray(weights, dtype=DTYPE)
    if values.ndim == 0:
        return jnp.full((PARETO_K_MONOMIAL_COUNT, int(num_site)), values, dtype=DTYPE)
    if values.ndim == 1:
        values = jnp.reshape(values, (PARETO_K_MONOMIAL_COUNT, 1))
        return jnp.broadcast_to(values, (PARETO_K_MONOMIAL_COUNT, int(num_site))).astype(DTYPE)
    return jnp.reshape(values, (PARETO_K_MONOMIAL_COUNT, int(num_site))).astype(DTYPE)


def _zero_residual_gmm_components(
    zero,
    num_site: int,
    term_count: int,
):
    term_values = jnp.zeros((int(term_count),), dtype=DTYPE)
    site_values = jnp.zeros((int(term_count), int(num_site)), dtype=DTYPE)
    return {
        "loss_residual_gmm": jnp.asarray(zero, dtype=DTYPE),
        "loss_residual_gmm_log1p": jnp.asarray(zero, dtype=DTYPE),
        "loss_residual_gmm_raw": jnp.asarray(zero, dtype=DTYPE),
        "loss_residual_gmm_terms": term_values,
        "loss_residual_gmm_site_terms": site_values,
        "residual_gmm_z_mean": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_z_max": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_z_worst": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_radius_mean": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_radius_max": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_radius_worst": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_warning_fraction": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_bad_fraction": jnp.asarray(zero, dtype=DTYPE),
        "residual_gmm_covariance_estimate": jnp.zeros(
            (2 * int(term_count), 2 * int(term_count)),
            dtype=DTYPE,
        ),
    }


def _apply_neural_gauge_component_mask(
    *,
    drift_g: jnp.ndarray,
    drift_f: jnp.ndarray,
    diffusion_g: jnp.ndarray,
    gauge_mode: str,
    neural_gauge_components: str,
):
    """Mask NN gauge outputs while leaving analytical gauges unchanged."""
    if gauge_mode not in NEURAL_GAUGE_MODES:
        return drift_g, drift_f, diffusion_g

    neural_gauge_components = normalize_neural_gauge_components(neural_gauge_components)
    if neural_gauge_components == "both":
        return drift_g, drift_f, diffusion_g
    if neural_gauge_components == "drift":
        return drift_g, drift_f, jnp.zeros_like(diffusion_g)
    if neural_gauge_components == "diffusion":
        return jnp.zeros_like(drift_g), jnp.zeros_like(drift_f), diffusion_g
    raise ValueError("neural_gauge_components must be 'both', 'drift', or 'diffusion'")

@jax.jit
def var_complex_to_real(alpha, beta):
    alpha_real = jnp.concatenate([jnp.real(alpha), jnp.imag(alpha)], axis=-1)
    beta_real = jnp.concatenate([jnp.real(beta), jnp.imag(beta)], axis=-1)
    return alpha_real, beta_real


@partial(jax.jit, static_argnums=(0, 1))
def initialize_phase_space_variables(num_walker: int, num_site: int, n0: DTYPE):
    alpha = (jnp.sqrt(to_r(n0)) * jnp.ones((num_walker, num_site), dtype=DTYPE)).astype(CDTYPE)
    beta = (jnp.sqrt(to_r(n0)) * jnp.ones((num_walker, num_site), dtype=DTYPE)).astype(CDTYPE)
    lnOmega = jnp.zeros((num_walker,), dtype=CDTYPE)
    return lnOmega, alpha, beta


def _complex_mul_safe(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    real = jnp.real(left) * jnp.real(right) - jnp.imag(left) * jnp.imag(right)
    imag = jnp.real(left) * jnp.imag(right) + jnp.imag(left) * jnp.real(right)
    real = jnp.asarray(jnp.real(real), dtype=DTYPE)
    imag = jnp.asarray(jnp.real(imag), dtype=DTYPE)
    return real + jnp.asarray(1.0j, dtype=CDTYPE) * imag


@partial(jax.jit, static_argnames=("axis_name",))
def self_normalized_weight_ratio_for_axis(lnOmega: jnp.ndarray, eps: float = 1e-12, axis_name=None):
    omega_centered = centered_exponentiated_weights_for_axis(lnOmega, eps=eps, axis_name=axis_name)
    sum_omega = jnp.sum(omega_centered)
    walker_count = jnp.asarray(omega_centered.shape[0], dtype=DTYPE)
    if axis_name is not None:
        sum_omega = lax.psum(sum_omega, axis_name)
        walker_count = lax.psum(walker_count, axis_name)
    denom = safe_complex_denominator(sum_omega / walker_count, eps=eps)
    return omega_centered / denom


@partial(jax.jit, static_argnames=("axis_name",))
def trace_normalized_physical_weights_for_axis(
    lnOmega: jnp.ndarray,
    lnOmega_shift: DTYPE,
    reference_lnOmega: jnp.ndarray,
    reference_lnOmega_shift: DTYPE,
    eps: float = 1e-12,
    axis_name=None,
):
    """Return physical weights divided by one stopped trace reference.

    The reference is the complex walker mean of the physical weight at a
    fixed time.  Computing the ratio in centered log coordinates avoids
    reconstructing a potentially enormous common physical-weight scale.
    """

    reference_lnOmega = jnp.asarray(reference_lnOmega, dtype=CDTYPE)
    walker_count = jnp.asarray(reference_lnOmega.shape[0], dtype=DTYPE)
    if axis_name is not None:
        walker_count = lax.psum(walker_count, axis_name)
    reference_center = max_real_log_weight_for_axis(
        reference_lnOmega,
        axis_name=axis_name,
    )

    reference_centered = centered_exponentiated_weights_for_axis(
        reference_lnOmega,
        eps=eps,
        axis_name=axis_name,
    )
    reference_sum = jnp.sum(reference_centered)
    if axis_name is not None:
        reference_sum = lax.psum(reference_sum, axis_name)
    reference_mean = lax.stop_gradient(
        safe_complex_denominator(reference_sum / walker_count, eps=eps)
    )

    relative_shift = lax.stop_gradient(
        jnp.asarray(lnOmega_shift, dtype=DTYPE)
        - jnp.asarray(reference_lnOmega_shift, dtype=DTYPE)
    )
    return (
        jnp.exp(jnp.asarray(lnOmega, dtype=CDTYPE) - reference_center + relative_shift)
        / reference_mean
    )


@partial(jax.jit, static_argnames=("axis_name",))
def trace_reference_ratio_for_axis(
    numerator_lnOmega: jnp.ndarray,
    numerator_lnOmega_shift: DTYPE,
    denominator_lnOmega: jnp.ndarray,
    denominator_lnOmega_shift: DTYPE,
    eps: float = 1e-12,
    axis_name=None,
):
    """Return a stopped ratio of two physical walker-mean trace scales."""

    relative_weights = trace_normalized_physical_weights_for_axis(
        numerator_lnOmega,
        numerator_lnOmega_shift,
        denominator_lnOmega,
        denominator_lnOmega_shift,
        eps=eps,
        axis_name=axis_name,
    )
    ratio_sum = jnp.sum(relative_weights)
    walker_count = jnp.asarray(relative_weights.shape[0], dtype=DTYPE)
    if axis_name is not None:
        ratio_sum = lax.psum(ratio_sum, axis_name)
        walker_count = lax.psum(walker_count, axis_name)
    return lax.stop_gradient(ratio_sum / walker_count)


@partial(jax.jit, static_argnames=("axis_name",))
def weighted_mean_complex_for_axis(x, w, eps=1e-12, axis_name=None):
    weight_shape = (w.shape[0],) + (1,) * (x.ndim - 1)
    weight_view = jnp.asarray(w.reshape(weight_shape), dtype=CDTYPE)
    num = jnp.sum(_complex_mul_safe(weight_view, jnp.asarray(x, dtype=CDTYPE)), axis=0)
    denom_sum = jnp.sum(w)
    if axis_name is not None:
        num = lax.psum(num, axis_name)
        denom_sum = lax.psum(denom_sum, axis_name)
    denom = safe_complex_denominator(denom_sum, eps=eps)
    return num / denom


@partial(jax.jit, static_argnames=("axis_name",))
def log_weight_increment_spread_for_axis(delta_ln_omega, axis_name=None):
    """Return ``Var(Re) + Var(Im)`` of one complex log-weight increment.

    The variance is over the global walker ensemble.  Any walker-common
    shift (for example the solver's real log-weight centering) cancels in
    the variance, so this equals the growth of
    ``Var(Re lnOmega) + Var(Im lnOmega)`` over the window: the window's
    weight-entropy spend in nats.  The value stays differentiable along the
    on-policy path through ``lnOmega``.
    """

    delta = jnp.asarray(delta_ln_omega, dtype=CDTYPE)
    parts = jnp.stack(
        [jnp.real(delta).astype(DTYPE), jnp.imag(delta).astype(DTYPE)],
        axis=-1,
    )
    count = _global_walker_count(int(parts.shape[0]), axis_name=axis_name)
    part_sum = jnp.sum(parts, axis=0)
    if axis_name is not None:
        part_sum = lax.psum(part_sum, axis_name)
    mean = part_sum / count
    centered = parts - mean[None, :]
    centered_sq_sum = jnp.sum(centered * centered, axis=0)
    if axis_name is not None:
        centered_sq_sum = lax.psum(centered_sq_sum, axis_name)
    variances = centered_sq_sum / count
    return jnp.maximum(
        variances[0] + variances[1],
        jnp.asarray(0.0, dtype=DTYPE),
    )


@partial(jax.jit, static_argnames=("axis_name",))
def complex_ess_ratio_for_axis(lnOmega, eps: float = 1e-12, axis_name=None):
    """Return the complex effective-sample-size ratio ``ESS_c / N``.

    ``ESS_c = |sum_w Omega_w|^2 / sum_w |Omega_w|^2`` generalizes the usual
    effective sample size to complex weights: phase spread degrades it
    exactly like modulus spread.  Max-centered exponentiation keeps the
    ratio finite; the common center cancels.  Returned as a stopped
    diagnostic in ``(0, 1]``.
    """

    weights = centered_exponentiated_weights_for_axis(
        lnOmega,
        eps=eps,
        axis_name=axis_name,
    )
    sum_re = jnp.sum(jnp.real(weights))
    sum_im = jnp.sum(jnp.imag(weights))
    abs_sq_sum = jnp.sum(jnp.real(weights * jnp.conj(weights)))
    count = _global_walker_count(int(weights.shape[0]), axis_name=axis_name)
    if axis_name is not None:
        sum_re = lax.psum(sum_re, axis_name)
        sum_im = lax.psum(sum_im, axis_name)
        abs_sq_sum = lax.psum(abs_sq_sum, axis_name)
    numerator = jnp.asarray(sum_re * sum_re + sum_im * sum_im, dtype=DTYPE)
    denominator = jnp.maximum(
        count * jnp.asarray(abs_sq_sum, dtype=DTYPE),
        jnp.asarray(eps, dtype=DTYPE),
    )
    return lax.stop_gradient(numerator / denominator)


def normalize_applied_quantities(applied_quantities) -> int:
    """Normalize the observable-cloud order cutoff.

    The public training option is an integer ``P`` in ``1..6``.
    Observable Pareto-k clouds are always onsite per-site.
    """
    if isinstance(applied_quantities, str):
        value = applied_quantities.strip().lower()
        try:
            order = int(value)
        except ValueError as exc:
            raise ValueError("applied_quantities must be an integer from 1 to 6") from exc
    else:
        order = int(applied_quantities)
    if order < 1 or order > 6:
        raise ValueError("applied_quantities must be an integer from 1 to 6")
    return order


def normalize_applied_quantities_mode(applied_quantities_mode: str = "upto") -> str:
    """Normalize how ``applied_quantities = P`` selects monomial orders.

    ``upto`` preserves the historical behavior and includes all total orders
    ``1 <= m+n <= P``.  ``exact`` includes only the shell ``m+n = P``.
    """
    value = str(applied_quantities_mode).strip().lower().replace("-", "_")
    if value not in {"upto", "exact"}:
        raise ValueError("applied_quantities_mode must be 'upto' or 'exact'")
    return value


def _selected_specs_from_selector(specs, applied_quantities: int, applied_quantities_mode: str, monomials=()):
    return selected_monomial_specs(
        specs,
        normalize_applied_quantities(applied_quantities),
        normalize_applied_quantities_mode(applied_quantities_mode),
        monomials,
    )


def _complex_power_table(base: jnp.ndarray, max_power: int, *, safe: bool) -> tuple[jnp.ndarray, ...]:
    """Return ``(1, base, ..., base**max_power)`` for repeated monomial reuse."""

    powers = [jnp.ones_like(base)]
    for _power in range(int(max_power)):
        if safe:
            powers.append(_complex_mul_safe(powers[-1], base))
        else:
            powers.append(powers[-1] * base)
    return tuple(powers)


def _monomial_from_power_tables(
    alpha_powers: tuple[jnp.ndarray, ...],
    beta_powers: tuple[jnp.ndarray, ...],
    m_power: int,
    n_power: int,
    *,
    safe: bool,
):
    alpha_part = alpha_powers[int(m_power)]
    beta_part = beta_powers[int(n_power)]
    if safe:
        return _complex_mul_safe(alpha_part, beta_part)
    return alpha_part * beta_part


def _global_walker_count(local_walker_count: int, axis_name=None) -> jnp.ndarray:
    walker_count = jnp.asarray(local_walker_count, dtype=DTYPE)
    if axis_name is not None:
        walker_count = lax.psum(walker_count, axis_name)
    return jnp.maximum(walker_count, jnp.asarray(1.0, dtype=DTYPE))


def _time_aggregation_score(value: jnp.ndarray, mode: str) -> jnp.ndarray:
    value = jnp.maximum(jnp.asarray(value, dtype=DTYPE), jnp.asarray(0.0, dtype=DTYPE))
    if mode == "entropic_log1p":
        return jnp.log1p(value)
    # ``log1p`` arrives here after the transform has been applied to each
    # site objective separately; only its window aggregation remains.
    if mode in {"mean", "log1p", "entropic"}:
        return value
    raise ValueError(
        "time aggregation mode must be mean, log1p, entropic, or "
        "entropic_log1p"
    )


def _update_time_logsumexp(
    logsum: jnp.ndarray,
    weight: jnp.ndarray,
    score: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    weight = jnp.asarray(weight, dtype=DTYPE)
    positive_weight = weight > jnp.asarray(0.0, dtype=DTYPE)
    safe_weight = jnp.where(
        positive_weight,
        weight,
        jnp.asarray(1.0, dtype=DTYPE),
    )
    weighted_score = jnp.where(
        positive_weight,
        jnp.log(safe_weight) + beta * score,
        jnp.asarray(-jnp.inf, dtype=DTYPE),
    )
    return jnp.logaddexp(logsum, weighted_score)


def _finalize_time_aggregation(
    weighted_mean_value: jnp.ndarray,
    weighted_score_sum: jnp.ndarray,
    weighted_logsumexp: jnp.ndarray,
    weight_sum: jnp.ndarray,
    beta_value: jnp.ndarray,
    mode: str,
) -> jnp.ndarray:
    """Return a window-weight scaled time aggregate.

    Segmented training divides segment gradients and aux values by the sum of
    segment window weights.  Non-mean risk objectives are therefore returned as
    ``weight_sum * risk`` to preserve the historical outer normalization.
    """

    if mode == "mean":
        return weighted_mean_value
    if mode == "log1p":
        return weighted_score_sum
    safe_weight_sum = jnp.maximum(
        jnp.asarray(weight_sum, dtype=DTYPE),
        jnp.asarray(1.0e-12, dtype=DTYPE),
    )
    beta = jnp.asarray(beta_value, dtype=DTYPE)
    mean_score = weighted_score_sum / safe_weight_sum

    def _risk_nonzero(_):
        return (weighted_logsumexp - jnp.log(safe_weight_sum)) / beta

    risk = lax.cond(
        jnp.abs(beta) > jnp.asarray(1.0e-12, dtype=DTYPE),
        _risk_nonzero,
        lambda _: mean_score,
        operand=jnp.asarray(0.0, dtype=DTYPE),
    )
    return safe_weight_sum * risk


def _resolve_pareto_k_tail_count_static(
    sample_count: int,
    tail_fraction: float,
    min_tail_count: int,
) -> int:
    """Resolve a static PSIS tail count for a traced cloud shape."""

    count = max(2, int(sample_count))
    fraction_count = int(np.floor(float(tail_fraction) * float(count)))
    sqrt_cap_count = int(np.floor(3.0 * np.sqrt(float(count))))
    psis_count = min(fraction_count, sqrt_cap_count)
    requested = max(int(min_tail_count), psis_count)
    return max(1, min(requested, count - 1))


@jax.jit
def robust_mean_cov_2d(
    z_flat: jnp.ndarray,
    q_winsor: float = 0.95,
    eps_cov: float = 1e-8,
    shrinkage: float = 0.05,
):
    q_winsor_raw = jnp.asarray(q_winsor, dtype=DTYPE)
    q_hi = jnp.asarray(jnp.clip(q_winsor_raw, 0.5, 1.0), dtype=DTYPE)
    q_lo = jnp.asarray(1.0, dtype=DTYPE) - q_hi
    shrinkage = jnp.asarray(shrinkage, dtype=DTYPE)
    eps_cov = jnp.asarray(eps_cov, dtype=DTYPE)

    x = jnp.real(z_flat)
    y = jnp.imag(z_flat)

    def _mean_cov_from_xy(x_in, y_in):
        mu_x = lax.stop_gradient(jnp.mean(x_in, axis=0))
        mu_y = lax.stop_gradient(jnp.mean(y_in, axis=0))

        dx = x_in - mu_x[None, :]
        dy = y_in - mu_y[None, :]

        s_xx = jnp.mean(dx * dx, axis=0)
        s_xy = jnp.mean(dx * dy, axis=0)
        s_yy = jnp.mean(dy * dy, axis=0)

        s_bar = 0.5 * (s_xx + s_yy)
        one_minus = jnp.asarray(1.0, dtype=DTYPE) - shrinkage

        s_xx = lax.stop_gradient(one_minus * s_xx + shrinkage * s_bar + eps_cov)
        s_xy = lax.stop_gradient(one_minus * s_xy)
        s_yy = lax.stop_gradient(one_minus * s_yy + shrinkage * s_bar + eps_cov)
        return mu_x, mu_y, s_xx, s_xy, s_yy

    def _winsorized_cov(_):
        x_lo = lax.stop_gradient(jnp.quantile(x, q_lo, axis=0))
        x_hi = lax.stop_gradient(jnp.quantile(x, q_hi, axis=0))
        y_lo = lax.stop_gradient(jnp.quantile(y, q_lo, axis=0))
        y_hi = lax.stop_gradient(jnp.quantile(y, q_hi, axis=0))

        xw = jnp.clip(x, x_lo[None, :], x_hi[None, :])
        yw = jnp.clip(y, y_lo[None, :], y_hi[None, :])
        return _mean_cov_from_xy(xw, yw)

    return lax.cond(
        q_winsor_raw > jnp.asarray(0.0, dtype=DTYPE),
        _winsorized_cov,
        lambda _: _mean_cov_from_xy(x, y),
        operand=jnp.asarray(0.0, dtype=DTYPE),
    )


@jax.jit
def mahalanobis_radius_2d(
    z_flat: jnp.ndarray,
    mu_x: jnp.ndarray,
    mu_y: jnp.ndarray,
    s_xx: jnp.ndarray,
    s_xy: jnp.ndarray,
    s_yy: jnp.ndarray,
    eps_det: float = 1e-12,
):
    eps_det = jnp.asarray(eps_det, dtype=DTYPE)
    x = jnp.real(z_flat)
    y = jnp.imag(z_flat)
    dx = x - mu_x[None, :]
    dy = y - mu_y[None, :]
    det = s_xx * s_yy - s_xy * s_xy + eps_det
    d_sq = (s_yy[None, :] * dx * dx - 2.0 * s_xy[None, :] * dx * dy + s_xx[None, :] * dy * dy) / det[None, :]
    # Floor the squared radius before sqrt. Masked/in-threshold channels can
    # otherwise sit exactly at d_sq=0, where d sqrt(x)/dx is singular and can
    # leak NaNs through autodiff even when the scalar exceedance is zero.
    return jnp.sqrt(jnp.maximum(d_sq, eps_det))


def _pareto_k_tail_channel_stats(
    cloud: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    q_winsor: float,
    eps_cov: float,
    cov_shrinkage: float,
    pareto_k_threshold: float,
    pareto_k_threshold_tau: float,
    pareto_k_envelope_beta: float,
    pareto_k_envelope_excess: str,
    pareto_k_tail_count: int,
    eps_det: float,
    pareto_k_tail_fraction: float = -1.0,
    pareto_k_min_tail_count: int = 32,
    eps: float = 1.0e-12,
    axis_name=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pareto-envelope loss on robust 2D Mahalanobis radii.

    The complex cloud is first converted into a robust covariance-standardized
    radius per walker/channel.  For diagnostics, each channel still reports the
    ordinary Hill-style inverse Pareto tail slope,

        k_hat = mean_j log(r_j / r_{M+1}).

    The optimized loss is instead an entropic risk over rank-wise
    Pareto-envelope violations.  The top ``M`` log-radii should lie below the
    local shoulder reference plus the allowed slope ``pareto_k_threshold``:

        y_j <= u + k0 * log((M + 1) / j) + slack.

    The forward reference ``u`` is the usual cutoff ``y_{M+1}``, while the
    backward pass uses the next shoulder bank as a scale-invariant reference.
    This gives per-rank credit without rewarding uniform shrinkage of the tail.
    ``pareto_k_envelope_excess="log"`` uses the log-radius violation
    ``[y_j - y_j^(k0)]_+``.  ``"ratio"`` uses the equivalent radius ratio
    violation ``[exp(y_j - y_j^(k0)) - 1]_+``.  ``pareto_k_envelope_beta=0``
    recovers the plain mean violation; positive values focus smoothly on the
    ranks that violate the envelope most.
    """
    if axis_name is None:
        stats_cloud = cloud
    else:
        gathered_cloud = lax.all_gather(cloud, axis_name)
        stats_cloud = gathered_cloud.reshape(
            (gathered_cloud.shape[0] * gathered_cloud.shape[1],) + gathered_cloud.shape[2:]
        )
    stats = robust_mean_cov_2d(stats_cloud, q_winsor=q_winsor, eps_cov=eps_cov, shrinkage=cov_shrinkage)
    radius = mahalanobis_radius_2d(cloud, *stats, eps_det=eps_det)

    if axis_name is None:
        fit_radius = radius
    else:
        gathered_radius = lax.all_gather(radius, axis_name)
        fit_radius = gathered_radius.reshape(
            (gathered_radius.shape[0] * gathered_radius.shape[1],) + gathered_radius.shape[2:]
        )

    walker_count = int(fit_radius.shape[0])
    if walker_count < 2:
        raise ValueError(
            "Pareto-tail diagnostics require at least two walkers; "
            f"received {walker_count}"
        )
    if float(pareto_k_tail_fraction) > 0.0:
        requested_tail_count = _resolve_pareto_k_tail_count_static(
            walker_count,
            pareto_k_tail_fraction,
            pareto_k_min_tail_count,
        )
    else:
        requested_tail_count = int(pareto_k_tail_count)
    top_count = max(1, min(requested_tail_count, max(walker_count - 1, 1)))
    log_radius_by_channel = jnp.swapaxes(
        jnp.log(jnp.maximum(fit_radius, jnp.asarray(eps, dtype=DTYPE))),
        0,
        1,
    )
    shoulder_count = max(16, top_count // 4)
    shoulder_count = max(1, min(shoulder_count, max(walker_count - top_count, 1)))
    top_log_radius = lax.top_k(log_radius_by_channel, top_count + shoulder_count)[0]
    top_tail = top_log_radius[:, :top_count]
    shoulder_bank = top_log_radius[:, top_count : top_count + shoulder_count]
    log_tail_cutoff = top_log_radius[:, top_count]
    shoulder_reference = jnp.mean(shoulder_bank, axis=1)
    reference = lax.stop_gradient(log_tail_cutoff - shoulder_reference) + shoulder_reference

    k_hat = jnp.mean(top_tail - log_tail_cutoff[:, None], axis=1)
    k_hat = jnp.maximum(k_hat, jnp.asarray(0.0, dtype=DTYPE))

    rank = jnp.arange(1, top_count + 1, dtype=DTYPE)
    envelope_slope = jnp.asarray(pareto_k_threshold, dtype=DTYPE)
    envelope_slack = jnp.asarray(pareto_k_threshold_tau, dtype=DTYPE)
    rank_envelope = envelope_slope * jnp.log(
        (jnp.asarray(top_count + 1, dtype=DTYPE)) / rank
    )
    envelope_log_excess = jnp.maximum(
        top_tail - reference[:, None] - rank_envelope[None, :] - envelope_slack,
        jnp.asarray(0.0, dtype=DTYPE),
    )
    if pareto_k_envelope_excess == "log":
        envelope_excess = envelope_log_excess
    elif pareto_k_envelope_excess == "ratio":
        envelope_excess = jnp.expm1(envelope_log_excess)
    else:
        raise ValueError('pareto_k_envelope_excess must be "log" or "ratio"')
    envelope_beta = jnp.asarray(pareto_k_envelope_beta, dtype=DTYPE)
    mean_excess = jnp.mean(envelope_excess, axis=1)

    def _entropic_envelope_loss(_):
        log_mean = jsp.special.logsumexp(envelope_beta * envelope_excess, axis=1) - jnp.log(
            jnp.asarray(top_count, dtype=DTYPE)
        )
        return log_mean / envelope_beta

    per_channel_loss = lax.cond(
        jnp.abs(envelope_beta) > jnp.asarray(1.0e-12, dtype=DTYPE),
        _entropic_envelope_loss,
        lambda _: mean_excess,
        operand=jnp.asarray(0.0, dtype=DTYPE),
    )

    mask = jnp.ravel(jnp.asarray(mask, dtype=DTYPE))
    return per_channel_loss, k_hat, mask


@partial(
    jax.jit,
    static_argnames=(
        "applied_quantities",
        "applied_quantities_mode",
        "monomials",
        "pareto_k_envelope_excess",
        "pareto_k_tail_fraction",
        "pareto_k_min_tail_count",
        "axis_name",
    ),
)
def compute_onsite_pareto_k_diagnostics(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    pareto_k_threshold: float = 0.7,
    pareto_k_threshold_tau: float = 0.1,
    pareto_k_envelope_beta: float = 0.5,
    pareto_k_envelope_excess: str = "log",
    pareto_k_tail_fraction: float = 0.01,
    pareto_k_min_tail_count: int = 32,
    loss_pareto_k_monomial_weights: jnp.ndarray = 1.0,
    q_winsor: float = 0.95,
    applied_quantities: int = 6,
    applied_quantities_mode: str = "upto",
    monomials: tuple = (),
    axis_name=None,
):
    r"""Onsite self-normalized Pareto-envelope diagnostics.

    For each selected onsite operator monomial
    ``< (a_i^\dagger)^m a_i^n >`` this fits the tail of the corresponding
    gauge-P cloud ``Omega_hat * beta_i^m * alpha_i^n`` independently by
    site/channel, while batching the selected channels into one JAX
    tail-statistics call.
    """
    zero = jnp.asarray(0.0, dtype=DTYPE)
    omega_hat = self_normalized_weight_ratio_for_axis(lnOmega, axis_name=axis_name)
    num_site = int(alpha.shape[-1])
    pareto_k_loss_terms_sum = jnp.zeros((PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE)
    pareto_k_site_terms_sum = jnp.zeros((PARETO_K_MONOMIAL_COUNT, num_site), dtype=DTYPE)
    pareto_k_mean_terms_sum = jnp.zeros((PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE)
    pareto_k_max_terms_sum = jnp.zeros((PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE)
    pareto_k_site_indices_sum = jnp.zeros((PARETO_K_MONOMIAL_COUNT, num_site), dtype=DTYPE)
    pareto_k_monomial_weights = _resolve_pareto_k_monomial_weights(
        loss_pareto_k_monomial_weights,
        num_site,
    )

    selected_specs = tuple(
        (m_power, n_power, _pareto_k_monomial_index(m_power, n_power))
        for _total_order, m_power, n_power, _term in _selected_specs_from_selector(
            PARETO_K_MONOMIAL_SPECS,
            applied_quantities,
            applied_quantities_mode,
            monomials,
        )
    )
    selected_count = len(selected_specs)

    if selected_count <= 0:
        return {
            "pareto_k_loss": zero,
            "pareto_k_loss_terms": pareto_k_loss_terms_sum,
            "pareto_k_site_terms": pareto_k_site_terms_sum,
            "pareto_k_mean_terms": pareto_k_mean_terms_sum,
            "pareto_k_max_terms": pareto_k_max_terms_sum,
            "pareto_k_site_indices": pareto_k_site_indices_sum,
            "pareto_k_mean": zero,
            "pareto_k_max": zero,
            "pareto_k_warning_fraction": zero,
            "pareto_k_bad_fraction": zero,
        }

    weighted_clouds = []
    masks = []
    selected_indices_np = []
    max_selected_power = max(
        max(int(m_power), int(n_power))
        for m_power, n_power, _term_index in selected_specs
    )
    alpha_powers = _complex_power_table(alpha, max_selected_power, safe=False)
    beta_powers = _complex_power_table(beta, max_selected_power, safe=False)
    for m_power, n_power, term_index in selected_specs:
        monomial_cloud = _monomial_from_power_tables(
            alpha_powers,
            beta_powers,
            n_power,
            m_power,
            safe=False,
        )
        weighted_cloud = omega_hat[:, None] * monomial_cloud
        weighted_clouds.append(weighted_cloud)
        masks.append(jnp.ones((num_site,), dtype=DTYPE))
        selected_indices_np.append(term_index)

    selected_indices = jnp.asarray(np.asarray(selected_indices_np, dtype=np.int32))
    selected_cloud = jnp.concatenate(weighted_clouds, axis=-1)
    selected_mask = jnp.concatenate(masks, axis=0)
    per_channel_loss, k_hat, selected_mask = _pareto_k_tail_channel_stats(
        selected_cloud,
        selected_mask,
        q_winsor=q_winsor,
        eps_cov=1.0e-8,
        cov_shrinkage=0.05,
        pareto_k_threshold=pareto_k_threshold,
        pareto_k_threshold_tau=pareto_k_threshold_tau,
        pareto_k_envelope_beta=pareto_k_envelope_beta,
        pareto_k_envelope_excess=pareto_k_envelope_excess,
        pareto_k_tail_count=1,
        eps_det=1.0e-12,
        pareto_k_tail_fraction=pareto_k_tail_fraction,
        pareto_k_min_tail_count=pareto_k_min_tail_count,
        axis_name=axis_name,
    )
    selected_site_losses = jnp.reshape(per_channel_loss, (selected_count, num_site))
    selected_site_k_indices = jnp.reshape(k_hat, (selected_count, num_site))
    selected_site_mask = jnp.reshape(selected_mask, (selected_count, num_site))
    active_by_term = jnp.maximum(jnp.sum(selected_site_mask, axis=1), jnp.asarray(1.0, dtype=DTYPE))
    selected_term_losses = jnp.sum(selected_site_mask * selected_site_losses, axis=1) / active_by_term
    selected_term_means = jnp.sum(selected_site_mask * selected_site_k_indices, axis=1) / active_by_term
    selected_term_maxes = jnp.max(
        jnp.where(selected_site_mask > jnp.asarray(0.0, dtype=DTYPE), selected_site_k_indices, zero),
        axis=1,
    )
    selected_weights = jnp.take(pareto_k_monomial_weights, selected_indices, axis=0)
    pareto_k_loss_sum = jnp.sum(
        jnp.sum(selected_site_mask * selected_weights * selected_site_losses, axis=1) / active_by_term
    )
    active_channel_count = jnp.maximum(jnp.sum(selected_site_mask), jnp.asarray(1.0, dtype=DTYPE))
    pareto_k_mean = jnp.sum(selected_site_mask * selected_site_k_indices) / active_channel_count
    pareto_k_max_value = jnp.max(
        jnp.where(selected_site_mask > jnp.asarray(0.0, dtype=DTYPE), selected_site_k_indices, zero)
    )
    pareto_k_warning_fraction = (
        jnp.sum(selected_site_mask * (selected_site_k_indices > jnp.asarray(0.5, dtype=DTYPE)).astype(DTYPE))
        / active_channel_count
    )
    pareto_k_bad_fraction = (
        jnp.sum(
            selected_site_mask
            * (selected_site_k_indices > jnp.asarray(pareto_k_threshold, dtype=DTYPE)).astype(DTYPE)
        )
        / active_channel_count
    )
    pareto_k_loss_terms_sum = pareto_k_loss_terms_sum.at[selected_indices].set(selected_term_losses)
    pareto_k_site_terms_sum = pareto_k_site_terms_sum.at[selected_indices, :].set(selected_site_losses)
    pareto_k_mean_terms_sum = pareto_k_mean_terms_sum.at[selected_indices].set(selected_term_means)
    pareto_k_max_terms_sum = pareto_k_max_terms_sum.at[selected_indices].set(selected_term_maxes)
    pareto_k_site_indices_sum = pareto_k_site_indices_sum.at[selected_indices, :].set(selected_site_k_indices)

    return {
        "pareto_k_loss": pareto_k_loss_sum,
        "pareto_k_loss_terms": pareto_k_loss_terms_sum,
        "pareto_k_site_terms": pareto_k_site_terms_sum,
        "pareto_k_mean_terms": pareto_k_mean_terms_sum,
        "pareto_k_max_terms": pareto_k_max_terms_sum,
        "pareto_k_site_indices": pareto_k_site_indices_sum,
        "pareto_k_mean": pareto_k_mean,
        "pareto_k_max": pareto_k_max_value,
        "pareto_k_warning_fraction": pareto_k_warning_fraction,
        "pareto_k_bad_fraction": pareto_k_bad_fraction,
    }


@partial(
    jax.jit,
    static_argnames=(
        "applied_quantities",
        "applied_quantities_mode",
        "monomials",
        "pareto_k_envelope_excess",
        "pareto_k_tail_fraction",
        "pareto_k_min_tail_count",
        "axis_name",
    ),
)
def compute_observable_pareto_k_indices(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    applied_quantities: int = 6,
    applied_quantities_mode: str = "upto",
    monomials: tuple = (),
    q_winsor: float = 0.95,
    pareto_k_threshold: float = 0.7,
    pareto_k_threshold_tau: float = 0.1,
    pareto_k_envelope_beta: float = 0.5,
    pareto_k_envelope_excess: str = "log",
    pareto_k_tail_fraction: float = 0.01,
    pareto_k_min_tail_count: int = 32,
    axis_name=None,
):
    r"""Pareto-k health indices for saved simulation observable clouds.

    Simulation health diagnostics follow the configured Pareto-k operator
    monomial selector. Use ``applied_quantities_mode="upto"`` to scan all
    onsite moments ``< (a^\dagger)^m a^n >`` with
    ``1 <= m+n <= applied_quantities`` and ``"exact"`` to scan only
    ``m+n=applied_quantities``.
    """
    diagnostics = compute_onsite_pareto_k_diagnostics(
        lnOmega,
        alpha,
        beta,
        pareto_k_threshold=pareto_k_threshold,
        pareto_k_threshold_tau=pareto_k_threshold_tau,
        pareto_k_envelope_beta=pareto_k_envelope_beta,
        pareto_k_envelope_excess=pareto_k_envelope_excess,
        pareto_k_tail_fraction=pareto_k_tail_fraction,
        pareto_k_min_tail_count=pareto_k_min_tail_count,
        q_winsor=q_winsor,
        applied_quantities=applied_quantities,
        applied_quantities_mode=applied_quantities_mode,
        monomials=monomials,
        axis_name=axis_name,
    )
    out = {
        "pareto-k_mean": diagnostics["pareto_k_mean"],
        "pareto-k_max": diagnostics["pareto_k_max"],
    }
    for _total_order, m_power, n_power, base_name in _selected_specs_from_selector(
        PARETO_K_OBSERVABLE_SPECS,
        applied_quantities,
        applied_quantities_mode,
        monomials,
    ):
        term_index = _pareto_k_monomial_index(m_power, n_power)
        out[base_name] = diagnostics["pareto_k_site_indices"][term_index]
        out[f"{base_name}_mean"] = diagnostics["pareto_k_mean_terms"][term_index]
        out[f"{base_name}_max"] = diagnostics["pareto_k_max_terms"][term_index]
    return out


def _compute_gauge_fields(
    apply_fn,
    params,
    lnOmega,
    alpha,
    beta,
    time,
    physical_params,
    gauge_mode: str,
    analytic_target_time: DTYPE = 0.0,
):
    W, S = alpha.shape
    if gauge_mode in NEURAL_GAUGE_MODES:
        alpha_real, beta_real = var_complex_to_real(alpha, beta)
        lnOmega_real = jnp.stack([jnp.real(lnOmega), jnp.imag(lnOmega)], axis=-1)
        return apply_fn(params, lnOmega_real, alpha_real, beta_real, time, physical_params)

    drift_g_c, drift_f_c, diffusion_g = compute_analytical_gauge_fields(
        alpha=alpha,
        beta=beta,
        time=time,
        target_time=analytic_target_time,
        U=physical_params[0],
        gauge_mode=gauge_mode,
    )
    drift_g = jnp.concatenate([jnp.real(drift_g_c), jnp.imag(drift_g_c)], axis=-1).astype(DTYPE)
    drift_f = jnp.concatenate([jnp.real(drift_f_c), jnp.imag(drift_f_c)], axis=-1).astype(DTYPE)
    return drift_g, drift_f, diffusion_g.astype(DTYPE)


def _compute_scaled_gauge_fields(
    *,
    apply_fn,
    params,
    lnOmega,
    alpha,
    beta,
    time,
    physical_params,
    gauge_weight: DTYPE,
    gauge_mode: str,
    neural_gauge_components: str,
    neural_gauge_each_apply: bool = False,
    analytic_target_time: DTYPE = 0.0,
):
    """Evaluate gauge fields once and apply the configured neural component mask."""

    if gauge_mode in NEURAL_GAUGE_MODES:
        def stop_state(_):
            return (
                lax.stop_gradient(lnOmega),
                lax.stop_gradient(alpha),
                lax.stop_gradient(beta),
            )

        def keep_state(_):
            return lnOmega, alpha, beta

        lnOmega, alpha, beta = lax.cond(
            neural_gauge_each_apply,
            stop_state,
            keep_state,
            operand=None,
        )

    drift_g, drift_f, diffusion_g = _compute_gauge_fields(
        apply_fn=apply_fn,
        params=params,
        lnOmega=lnOmega,
        alpha=alpha,
        beta=beta,
        time=time,
        physical_params=physical_params,
        gauge_mode=gauge_mode,
        analytic_target_time=analytic_target_time,
    )
    gauge_scale = gauge_weight if gauge_mode in NEURAL_GAUGE_MODES else jnp.asarray(1.0, dtype=DTYPE)
    drift_g = (gauge_scale * drift_g).astype(DTYPE)
    drift_f = (gauge_scale * drift_f).astype(DTYPE)
    diffusion_g = (gauge_scale * diffusion_g).astype(DTYPE)
    return _apply_neural_gauge_component_mask(
        drift_g=drift_g,
        drift_f=drift_f,
        diffusion_g=diffusion_g,
        gauge_mode=gauge_mode,
        neural_gauge_components=neural_gauge_components,
    )


def _selected_operator_moment_specs(applied_quantities: int, applied_quantities_mode: str, monomials=()):
    monomials = tuple((int(m), int(n)) for m, n in (monomials or ()))
    if monomials:
        max_order = max(int(m) + int(n) for m, n in monomials)
    else:
        max_order = normalize_applied_quantities(applied_quantities)
    if max_order > OPERATOR_MOMENT_MAX_ORDER:
        raise ValueError(f"exact moment equations support applied_quantities <= {OPERATOR_MOMENT_MAX_ORDER}")
    return tuple(
        (m_power, n_power, _operator_moment_index(m_power, n_power))
        for _total_order, m_power, n_power, _term in selected_monomial_specs(
            OPERATOR_MOMENT_SPECS,
            max_order,
            normalize_applied_quantities_mode(applied_quantities_mode),
            monomials,
        )
    )


def compute_onsite_exact_moment_snapshot_for_axis(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    U,
    gamma,
    Delta,
    F,
    J=0.0,
    applied_quantities: int = 4,
    applied_quantities_mode: str = "exact",
    monomials: tuple = (),
    eps: float = 1e-12,
    axis_name=None,
):
    r"""Self-normalized snapshots for onsite exact monomial equations.

    The public ``(m,n)`` convention follows the normal-ordered operator
    moment ``< (a_i^\dagger)^m a_i^n >``.  Its gauge-P estimator uses the
    phase-space cloud ``beta_i**m * alpha_i**n``.
    """
    lnOmega = jnp.asarray(lnOmega, dtype=CDTYPE)
    alpha = jnp.asarray(alpha, dtype=CDTYPE)
    beta = jnp.asarray(beta, dtype=CDTYPE)
    num_site = int(alpha.shape[-1])
    selected_specs = _selected_operator_moment_specs(applied_quantities, applied_quantities_mode, monomials)
    if len(selected_specs) <= 0:
        empty = jnp.zeros((0, num_site), dtype=CDTYPE)
        return {
            "M": empty,
            "Phi": empty,
            "selected_indices": jnp.zeros((0,), dtype=jnp.int32),
        }

    weight = centered_exponentiated_weights_for_axis(lnOmega, eps=eps, axis_name=axis_name)
    hop_alpha = apply_hopping_matrix(J, alpha)
    hop_beta = apply_hopping_matrix(conjugate_hopping_operator(J), beta)
    U_site = broadcast_site_param(U, num_site, DTYPE)
    gamma_site = broadcast_site_param(gamma, num_site, DTYPE)
    Delta_site = broadcast_site_param(Delta, num_site, DTYPE)
    F_site = jnp.broadcast_to(jnp.asarray(F, dtype=CDTYPE), (num_site,))
    i_unit = cplx_i()

    moment_cache = {}
    hop_alpha_cache = {}
    hop_beta_cache = {}
    required_power = max(
        max(int(m_power), int(n_power)) + (1 if int(m_power) != int(n_power) else 0)
        for m_power, n_power, _term_index in selected_specs
    )
    alpha_powers = _complex_power_table(alpha, required_power, safe=True)
    beta_powers = _complex_power_table(beta, required_power, safe=True)

    def moment_mean(m_dagger_power: int, n_annihilation_power: int):
        key = (int(m_dagger_power), int(n_annihilation_power))
        if key not in moment_cache:
            moment_cache[key] = weighted_mean_complex_for_axis(
                _monomial_from_power_tables(
                    alpha_powers,
                    beta_powers,
                    key[1],
                    key[0],
                    safe=True,
                ),
                weight,
                eps=eps,
                axis_name=axis_name,
            )
        return moment_cache[key]

    def hop_alpha_mean(m_dagger_power: int, n_annihilation_power: int):
        key = (int(m_dagger_power), int(n_annihilation_power))
        if key not in hop_alpha_cache:
            cloud = _complex_mul_safe(
                _monomial_from_power_tables(
                    alpha_powers,
                    beta_powers,
                    key[1],
                    key[0],
                    safe=True,
                ),
                hop_alpha,
            )
            hop_alpha_cache[key] = weighted_mean_complex_for_axis(
                cloud,
                weight,
                eps=eps,
                axis_name=axis_name,
            )
        return hop_alpha_cache[key]

    def hop_beta_mean(m_dagger_power: int, n_annihilation_power: int):
        key = (int(m_dagger_power), int(n_annihilation_power))
        if key not in hop_beta_cache:
            cloud = _complex_mul_safe(
                _monomial_from_power_tables(
                    alpha_powers,
                    beta_powers,
                    key[1],
                    key[0],
                    safe=True,
                ),
                hop_beta,
            )
            hop_beta_cache[key] = weighted_mean_complex_for_axis(
                cloud,
                weight,
                eps=eps,
                axis_name=axis_name,
            )
        return hop_beta_cache[key]

    moments = []
    rhs_values = []
    selected_indices_np = []
    for m_dagger_power, n_annihilation_power, term_index in selected_specs:
        M_mn = moment_mean(m_dagger_power, n_annihilation_power)
        rhs = (
            (
                i_unit * DTYPE(n_annihilation_power - m_dagger_power) * Delta_site
                - DTYPE(0.5 * (m_dagger_power + n_annihilation_power)) * gamma_site
                + DTYPE(0.5)
                * i_unit
                * U_site
                * DTYPE(
                    m_dagger_power * (m_dagger_power - 1)
                    - n_annihilation_power * (n_annihilation_power - 1)
                )
            )
            * M_mn
        )
        if m_dagger_power != n_annihilation_power:
            rhs = rhs + i_unit * U_site * DTYPE(m_dagger_power - n_annihilation_power) * moment_mean(
                m_dagger_power + 1,
                n_annihilation_power + 1,
            )
        if n_annihilation_power > 0:
            rhs = rhs - i_unit * DTYPE(n_annihilation_power) * F_site * moment_mean(
                m_dagger_power,
                n_annihilation_power - 1,
            )
            rhs = rhs + i_unit * DTYPE(n_annihilation_power) * hop_alpha_mean(
                m_dagger_power,
                n_annihilation_power - 1,
            )
        if m_dagger_power > 0:
            rhs = rhs + i_unit * DTYPE(m_dagger_power) * jnp.conj(F_site) * moment_mean(
                m_dagger_power - 1,
                n_annihilation_power,
            )
            rhs = rhs - i_unit * DTYPE(m_dagger_power) * hop_beta_mean(
                m_dagger_power - 1,
                n_annihilation_power,
            )
        moments.append(M_mn)
        rhs_values.append(rhs)
        selected_indices_np.append(term_index)

    return {
        "M": jnp.stack(moments, axis=0),
        "Phi": jnp.stack(rhs_values, axis=0),
        "selected_indices": jnp.asarray(np.asarray(selected_indices_np, dtype=np.int32)),
    }


def compute_onsite_exact_moment_clouds(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    U,
    gamma,
    Delta,
    F,
    J=0.0,
    applied_quantities: int = 4,
    applied_quantities_mode: str = "exact",
    monomials: tuple = (),
):
    r"""Unweighted per-walker clouds for onsite exact monomial equations.

    Returns the raw estimator cloud ``Y`` for
    ``< (a_i^\dagger)^m a_i^n >`` and the matching unweighted RHS cloud
    ``Phi``.  Self-normalized weights are applied by the caller so the same
    clouds can be used to build residual influence estimates.
    """

    alpha = jnp.asarray(alpha, dtype=CDTYPE)
    beta = jnp.asarray(beta, dtype=CDTYPE)
    num_site = int(alpha.shape[-1])
    selected_specs = _selected_operator_moment_specs(applied_quantities, applied_quantities_mode, monomials)
    if len(selected_specs) <= 0:
        empty = jnp.zeros((0, alpha.shape[0], num_site), dtype=CDTYPE)
        return {
            "Y": empty,
            "Phi": empty,
            "selected_indices": jnp.zeros((0,), dtype=jnp.int32),
        }

    hop_alpha = apply_hopping_matrix(J, alpha)
    hop_beta = apply_hopping_matrix(conjugate_hopping_operator(J), beta)
    U_site = broadcast_site_param(U, num_site, DTYPE)
    gamma_site = broadcast_site_param(gamma, num_site, DTYPE)
    Delta_site = broadcast_site_param(Delta, num_site, DTYPE)
    F_site = jnp.broadcast_to(jnp.asarray(F, dtype=CDTYPE), (num_site,))
    i_unit = cplx_i()

    required_power = max(
        max(int(m_power), int(n_power)) + (1 if int(m_power) != int(n_power) else 0)
        for m_power, n_power, _term_index in selected_specs
    )
    alpha_powers = _complex_power_table(alpha, required_power, safe=True)
    beta_powers = _complex_power_table(beta, required_power, safe=True)

    def moment_cloud(m_dagger_power: int, n_annihilation_power: int):
        return _monomial_from_power_tables(
            alpha_powers,
            beta_powers,
            int(n_annihilation_power),
            int(m_dagger_power),
            safe=True,
        )

    def hop_alpha_cloud(m_dagger_power: int, n_annihilation_power: int):
        return _complex_mul_safe(moment_cloud(m_dagger_power, n_annihilation_power), hop_alpha)

    def hop_beta_cloud(m_dagger_power: int, n_annihilation_power: int):
        return _complex_mul_safe(moment_cloud(m_dagger_power, n_annihilation_power), hop_beta)

    clouds = []
    rhs_clouds = []
    selected_indices_np = []
    for m_dagger_power, n_annihilation_power, term_index in selected_specs:
        Y_mn = moment_cloud(m_dagger_power, n_annihilation_power)
        rhs = (
            (
                i_unit * DTYPE(n_annihilation_power - m_dagger_power) * Delta_site
                - DTYPE(0.5 * (m_dagger_power + n_annihilation_power)) * gamma_site
                + DTYPE(0.5)
                * i_unit
                * U_site
                * DTYPE(
                    m_dagger_power * (m_dagger_power - 1)
                    - n_annihilation_power * (n_annihilation_power - 1)
                )
            )
            * Y_mn
        )
        if m_dagger_power != n_annihilation_power:
            rhs = rhs + i_unit * U_site * DTYPE(m_dagger_power - n_annihilation_power) * moment_cloud(
                m_dagger_power + 1,
                n_annihilation_power + 1,
            )
        if n_annihilation_power > 0:
            rhs = rhs - i_unit * DTYPE(n_annihilation_power) * F_site * moment_cloud(
                m_dagger_power,
                n_annihilation_power - 1,
            )
            rhs = rhs + i_unit * DTYPE(n_annihilation_power) * hop_alpha_cloud(
                m_dagger_power,
                n_annihilation_power - 1,
            )
        if m_dagger_power > 0:
            rhs = rhs + i_unit * DTYPE(m_dagger_power) * jnp.conj(F_site) * moment_cloud(
                m_dagger_power - 1,
                n_annihilation_power,
            )
            rhs = rhs - i_unit * DTYPE(m_dagger_power) * hop_beta_cloud(
                m_dagger_power - 1,
                n_annihilation_power,
            )
        clouds.append(Y_mn)
        rhs_clouds.append(rhs)
        selected_indices_np.append(term_index)

    return {
        "Y": jnp.stack(clouds, axis=0),
        "Phi": jnp.stack(rhs_clouds, axis=0),
        "selected_indices": jnp.asarray(np.asarray(selected_indices_np, dtype=np.int32)),
    }


def _distributed_mean_cov_nd(
    samples: jnp.ndarray,
    *,
    axis_name=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Exact population mean/covariance from distributed sufficient statistics.

    ``samples`` has shape ``(site, local_walker, dim)``.  Only a sitewise
    first moment and cross-product matrix are communicated, so memory and
    collective traffic are independent of the global walker count.
    """

    samples = jnp.asarray(samples, dtype=DTYPE)
    if samples.ndim != 3:
        raise ValueError(
            "covariance samples must have shape (site, walker, dim); "
            f"received {samples.shape}"
        )
    if any(int(size) <= 0 for size in samples.shape):
        raise ValueError(
            "covariance samples require nonempty site, walker, and channel "
            f"dimensions; received {samples.shape}"
        )

    sample_sum = jnp.sum(samples, axis=1)
    sample_count = jnp.asarray(samples.shape[1], dtype=DTYPE)
    if axis_name is not None:
        sample_sum = lax.psum(sample_sum, axis_name)
        sample_count = lax.psum(sample_count, axis_name)
    sample_count = jnp.maximum(sample_count, jnp.asarray(1.0, dtype=DTYPE))
    mean = sample_sum / sample_count
    # A centered second pass avoids catastrophic cancellation when residual
    # influences have a large common offset but a much smaller covariance.
    centered = samples - mean[:, None, :]
    centered_cross = jnp.einsum("swd,swe->sde", centered, centered)
    if axis_name is not None:
        centered_cross = lax.psum(centered_cross, axis_name)
    covariance = centered_cross / sample_count
    covariance = DTYPE(0.5) * (
        covariance + jnp.swapaxes(covariance, -1, -2)
    )
    return (
        lax.stop_gradient(mean),
        lax.stop_gradient(covariance),
        lax.stop_gradient(sample_count),
    )


@partial(
    jax.jit,
    static_argnames=("axis_name",),
)
def _residual_gmm_loss_from_residual_cloud(
    residual_cloud: jnp.ndarray,
    *,
    residual_influence_cloud: jnp.ndarray | None = None,
    d_clip: DTYPE = 10.0,
    cov_floor: DTYPE = 1.0e-8,
    cov_shrinkage: DTYPE = 0.05,
    lagged_covariance: jnp.ndarray | None = None,
    lagged_covariance_initialized: jnp.ndarray | None = None,
    axis_name=None,
):
    """Shared-covariance normalized projected-residual objective.

    The forward mean is the full, untrimmed mean of ``residual_cloud``.  The
    covariance is the site average of the exact population covariance of the
    corresponding self-normalized ratio influence at each lattice site.  This
    one shared covariance preconditions every site residual before the site
    objectives are averaged.  Covariance quantities are stopped-gradient
    preconditioners.  Channels are packed as
    ``Re(trace), Im(trace), Re(O1), Im(O1), ...``; the trace is therefore part
    of both the residual vector and its shared covariance.
    """

    residual_cloud = jnp.asarray(residual_cloud, dtype=CDTYPE)
    if residual_cloud.ndim != 3:
        raise ValueError(
            "residual cloud must have shape (channel, walker, site); "
            f"received {residual_cloud.shape}"
        )
    if residual_influence_cloud is None:
        residual_influence_cloud = residual_cloud
    else:
        residual_influence_cloud = jnp.asarray(
            residual_influence_cloud, dtype=CDTYPE
        )
        if residual_influence_cloud.shape != residual_cloud.shape:
            raise ValueError(
                "residual influence cloud must match the residual cloud shape; "
                f"received {residual_influence_cloud.shape} and "
                f"{residual_cloud.shape}"
            )

    term_count = int(residual_cloud.shape[0])
    walker_count_local = int(residual_cloud.shape[1])
    num_site = int(residual_cloud.shape[2])
    zero = jnp.asarray(0.0, dtype=DTYPE)
    if term_count <= 0:
        return zero, _zero_residual_gmm_components(
            zero,
            num_site,
            term_count,
        )
    if walker_count_local <= 0 or num_site <= 0:
        raise ValueError(
            "residual cloud must contain at least one walker and one site; "
            f"received {residual_cloud.shape}"
        )

    real_channel_count = 2 * term_count
    expected_covariance_shape = (
        real_channel_count,
        real_channel_count,
    )
    if (lagged_covariance is None) != (
        lagged_covariance_initialized is None
    ):
        raise ValueError(
            "lagged residual covariance and its initialization state "
            "must be provided together"
        )
    if (
        lagged_covariance is not None
        and lagged_covariance.shape != expected_covariance_shape
    ):
        raise ValueError(
            "lagged residual covariance must have shared site-averaged shape "
            f"{expected_covariance_shape}; received {lagged_covariance.shape}"
        )
    if lagged_covariance_initialized is None:
        lagged_covariance_initialized = jnp.asarray(False, dtype=jnp.bool_)
    else:
        lagged_covariance_initialized = jnp.asarray(
            lagged_covariance_initialized, dtype=jnp.bool_
        )
        if lagged_covariance_initialized.shape != ():
            raise ValueError(
                "lagged covariance initialization state must be scalar; "
                f"received {lagged_covariance_initialized.shape}"
            )

    def _interleaved_real_samples(cloud: jnp.ndarray) -> jnp.ndarray:
        site_cloud = jnp.transpose(cloud, (2, 1, 0))
        return jnp.reshape(
            jnp.stack((jnp.real(site_cloud), jnp.imag(site_cloud)), axis=-1),
            (num_site, walker_count_local, real_channel_count),
        ).astype(DTYPE)

    samples = _interleaved_real_samples(residual_cloud)
    influence_samples = _interleaved_real_samples(residual_influence_cloud)
    influence_mean, site_covariance_estimate, global_walker_count = (
        _distributed_mean_cov_nd(
            influence_samples,
            axis_name=axis_name,
        )
    )
    covariance_estimate = lax.stop_gradient(
        jnp.mean(site_covariance_estimate, axis=0)
    )

    if lagged_covariance is None:
        covariance = covariance_estimate
    else:
        lagged_covariance = lax.stop_gradient(
            jnp.asarray(lagged_covariance, dtype=DTYPE)
        )
        lagged_covariance = DTYPE(0.5) * (
            lagged_covariance + jnp.swapaxes(lagged_covariance, -1, -2)
        )
        covariance = jnp.where(
            lagged_covariance_initialized,
            lagged_covariance,
            covariance_estimate,
        )
    covariance = lax.stop_gradient(covariance)

    eps_value = jnp.asarray(cov_floor, dtype=DTYPE)
    identity = jnp.eye(real_channel_count, dtype=DTYPE)
    floored_covariance = lax.stop_gradient(
        covariance + eps_value * identity
    )
    channel_scale = lax.stop_gradient(
        jnp.sqrt(
            jnp.maximum(
                jnp.diagonal(floored_covariance, axis1=-2, axis2=-1),
                eps_value,
            )
        )
    )
    inverse_channel_scale = lax.stop_gradient(
        jnp.reciprocal(
            jnp.maximum(channel_scale, jnp.sqrt(eps_value))
        )
    )
    correlation = (
        inverse_channel_scale[:, None]
        * floored_covariance
        * inverse_channel_scale[None, :]
    )
    correlation = lax.stop_gradient(
        DTYPE(0.5) * (correlation + jnp.swapaxes(correlation, -1, -2))
    )
    shrink_value = jnp.asarray(cov_shrinkage, dtype=DTYPE)
    mean_variance = jnp.trace(correlation, axis1=-2, axis2=-1) / jnp.asarray(
        real_channel_count, dtype=DTYPE
    )
    correlation_jitter = jnp.asarray(
        np.finfo(np.dtype(DTYPE)).eps * max(real_channel_count, 1),
        dtype=DTYPE,
    )
    regularized_correlation = lax.stop_gradient(
        (jnp.asarray(1.0, dtype=DTYPE) - shrink_value) * correlation
        + (shrink_value * mean_variance + correlation_jitter) * identity
    )
    cholesky_factor = lax.stop_gradient(
        jnp.linalg.cholesky(regularized_correlation)
    )

    def _whiten_site_values(values: jnp.ndarray) -> jnp.ndarray:
        standardized = values * inverse_channel_scale
        flat_shape = standardized.shape
        standardized_flat = jnp.reshape(
            standardized, (-1, real_channel_count)
        )
        whitened_flat = jnp.swapaxes(
            jsp.linalg.solve_triangular(
                cholesky_factor,
                jnp.swapaxes(standardized_flat, -1, -2),
                lower=True,
            ),
            -1,
            -2,
        )
        return jnp.reshape(whitened_flat, flat_shape)

    sample_sum = jnp.sum(samples, axis=1)
    if axis_name is not None:
        sample_sum = lax.psum(sample_sum, axis_name)
    full_mean = sample_sum / global_walker_count

    centered_influence = influence_samples - influence_mean[:, None, :]
    whitened_centered = _whiten_site_values(centered_influence)
    radius_sq = jnp.sum(whitened_centered * whitened_centered, axis=-1)
    radius = jnp.sqrt(jnp.maximum(radius_sq, zero))
    d_clip_value = jnp.asarray(d_clip, dtype=DTYPE)
    safe_radius = jnp.maximum(radius, jnp.asarray(1.0e-30, dtype=DTYPE))
    clip_scale = lax.stop_gradient(
        jnp.minimum(
            jnp.asarray(1.0, dtype=DTYPE),
            d_clip_value / safe_radius,
        )
    )
    clipped_gradient_sum = jnp.sum(
        clip_scale[:, :, None] * (samples - lax.stop_gradient(samples)),
        axis=1,
    )
    if axis_name is not None:
        clipped_gradient_sum = lax.psum(clipped_gradient_sum, axis_name)
    mean_st = (
        lax.stop_gradient(full_mean)
        + clipped_gradient_sum / global_walker_count
    )

    whitened_mean = _whiten_site_values(mean_st[:, None, :])[:, 0, :]
    z2_site = jnp.maximum(
        jnp.sum(whitened_mean * whitened_mean, axis=-1),
        zero,
    )
    objective = jnp.mean(z2_site)
    objective_log1p = jnp.mean(jnp.log1p(z2_site))
    z_site = jnp.sqrt(z2_site)

    residual_sum_complex = jnp.sum(residual_cloud, axis=1)
    if axis_name is not None:
        residual_sum_complex = lax.psum(residual_sum_complex, axis_name)
    residual_mean_complex = residual_sum_complex / global_walker_count
    raw_site_losses = (
        jnp.real(residual_mean_complex) ** 2
        + jnp.imag(residual_mean_complex) ** 2
    )
    term_losses = jnp.mean(raw_site_losses, axis=1)
    raw_total = jnp.sum(term_losses)

    radius_sum = jnp.sum(radius, axis=1)
    radius_max = lax.stop_gradient(jnp.max(radius, axis=1))
    warning_threshold = DTYPE(0.5) * d_clip_value
    warning_count = lax.stop_gradient(
        jnp.sum((radius > warning_threshold).astype(DTYPE), axis=1)
    )
    bad_count = lax.stop_gradient(
        jnp.sum((radius > d_clip_value).astype(DTYPE), axis=1)
    )
    if axis_name is not None:
        radius_sum = lax.psum(radius_sum, axis_name)
        radius_max = lax.stop_gradient(lax.pmax(radius_max, axis_name))
        warning_count = lax.psum(warning_count, axis_name)
        bad_count = lax.psum(bad_count, axis_name)

    active_sample_count = jnp.maximum(
        jnp.asarray(num_site, dtype=DTYPE) * global_walker_count,
        jnp.asarray(1.0, dtype=DTYPE),
    )
    radius_mean = jnp.sum(radius_sum) / active_sample_count
    radius_max_value = jnp.max(radius_max)
    warning_fraction = jnp.sum(warning_count) / active_sample_count
    bad_fraction = jnp.sum(bad_count) / active_sample_count

    return objective, {
        "loss_residual_gmm": objective,
        "loss_residual_gmm_log1p": objective_log1p,
        "loss_residual_gmm_raw": raw_total,
        "loss_residual_gmm_terms": term_losses,
        "loss_residual_gmm_site_terms": raw_site_losses,
        "residual_gmm_z_mean": jnp.mean(z_site),
        "residual_gmm_z_max": jnp.max(z_site),
        "residual_gmm_z_worst": jnp.max(z_site),
        "residual_gmm_radius_mean": radius_mean,
        "residual_gmm_radius_max": radius_max_value,
        "residual_gmm_radius_worst": radius_max_value,
        "residual_gmm_warning_fraction": warning_fraction,
        "residual_gmm_bad_fraction": bad_fraction,
        "residual_gmm_covariance_estimate": covariance_estimate,
    }


def _projected_residual_node_statistics(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    operator_monomials: tuple,
    U,
    gamma,
    Delta,
    F,
    J=0.0,
    axis_name=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Return direct onsite equation contributions and ratio influences.

    Both arrays have shape ``(2, channel, walker, site)``.  Row zero is the
    bare endpoint monomial ``O`` and row one is its exact ``L^dagger O`` RHS,
    including hopping.  The channel axis follows ``operator_monomials``
    exactly; trace is added separately at window level.
    """

    monomials = normalize_projected_residual_monomials(operator_monomials)
    clouds = compute_onsite_exact_moment_clouds(
        alpha,
        beta,
        U=U,
        gamma=gamma,
        Delta=Delta,
        F=F,
        J=J,
        monomials=monomials,
    )
    polynomials = jnp.stack((clouds["Y"], clouds["Phi"]), axis=0)
    weight_ratio = self_normalized_weight_ratio_for_axis(
        lnOmega,
        axis_name=axis_name,
    )
    contributions = (
        polynomials * weight_ratio[None, None, :, None]
    )
    contribution_sum = jnp.sum(contributions, axis=2)
    global_walker_count = _global_walker_count(
        int(alpha.shape[0]), axis_name=axis_name
    )
    if axis_name is not None:
        contribution_sum = lax.psum(contribution_sum, axis_name)
    contribution_mean = contribution_sum / global_walker_count
    influences = (
        contributions
        - weight_ratio[None, None, :, None]
        * contribution_mean[:, :, None, :]
    )
    return contributions, influences


def _projected_residual_closed_newton_cotes_residual_cloud(
    *node_equations: jnp.ndarray,
    window_duration: DTYPE,
    integrator_nodes: int,
) -> jnp.ndarray:
    r"""Combine direct equation clouds into one window residual cloud.

    The equal-spaced node arrays share shape ``(2,C,W,S)``.  Only row one is
    integrated; the endpoint difference always uses the bare row-zero
    monomial.  The returned shape is ``(C,W,S)``.
    """

    nodes = normalize_residual_gmm_integrator_nodes(integrator_nodes)
    if len(node_equations) != nodes:
        raise ValueError(
            "projected-residual quadrature requires exactly "
            f"{nodes} node clouds; received {len(node_equations)}"
        )
    start_equations = node_equations[0]
    if start_equations.ndim != 4 or int(start_equations.shape[0]) != 2:
        raise ValueError(
            "projected-residual node clouds must share shape "
            "(2,channel,walker,site); received "
            f"{tuple(value.shape for value in node_equations)}"
        )
    if any(
        value.ndim != 4 or value.shape != start_equations.shape
        for value in node_equations[1:]
    ):
        raise ValueError(
            "projected-residual node clouds must share shape "
            "(2,channel,walker,site); received "
            f"{tuple(value.shape for value in node_equations)}"
        )
    if int(start_equations.shape[1]) <= 0:
        raise ValueError(
            "projected-residual quadrature requires at least one channel"
        )

    signed_duration = jnp.asarray(window_duration, dtype=DTYPE)
    duration = jnp.maximum(
        jnp.abs(signed_duration),
        jnp.asarray(1.0e-12, dtype=DTYPE),
    )
    weights, denominator = CLOSED_NEWTON_COTES_RULES[nodes]
    quadrature_scale = signed_duration / jnp.asarray(
        denominator, dtype=DTYPE
    )
    rhs_integral = sum(
        jnp.asarray(weight, dtype=DTYPE) * equations[1]
        for weight, equations in zip(weights, node_equations)
    )
    return (
        node_equations[-1][0]
        - start_equations[0]
        - quadrature_scale * rhs_integral
    ) / duration


def _projected_residual_clouds(
    *,
    operator_monomials: tuple,
    U,
    gamma,
    Delta,
    F,
    J,
    lnOmega_start: jnp.ndarray,
    lnOmega_start_shift: DTYPE,
    alpha_start: jnp.ndarray,
    beta_start: jnp.ndarray,
    lnOmega_end: jnp.ndarray,
    lnOmega_end_shift: DTYPE,
    alpha_end: jnp.ndarray,
    beta_end: jnp.ndarray,
    window_duration: DTYPE,
    residual_node_states: tuple,
    residual_gmm_integrator_nodes: int = 6,
    axis_name=None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Build trace-first site-resolved forward and influence residuals.

    The output order is ``[trace, *operator_monomials]`` and both arrays have
    shape ``(1+len(operator_monomials), walker, site)``.  Trace is one physical
    equation broadcast over sites so every site's walker covariance has the
    same trace-first joint channel basis before those covariances are averaged.
    """

    monomials = normalize_projected_residual_monomials(operator_monomials)
    integrator_nodes = normalize_residual_gmm_integrator_nodes(
        residual_gmm_integrator_nodes
    )
    if len(residual_node_states) != integrator_nodes:
        raise ValueError(
            "residual_node_states must contain exactly "
            f"{integrator_nodes} states; received {len(residual_node_states)}"
        )

    node_statistics = tuple(
        _projected_residual_node_statistics(
            node_lnOmega,
            node_alpha,
            node_beta,
            operator_monomials=monomials,
            U=U,
            gamma=gamma,
            Delta=Delta,
            F=F,
            J=J,
            axis_name=axis_name,
        )
        for node_lnOmega, node_alpha, node_beta in residual_node_states
    )
    forward_residual = (
        _projected_residual_closed_newton_cotes_residual_cloud(
            *(statistics[0] for statistics in node_statistics),
            window_duration=window_duration,
            integrator_nodes=integrator_nodes,
        )
    )
    influence_residual = (
        _projected_residual_closed_newton_cotes_residual_cloud(
            *(statistics[1] for statistics in node_statistics),
            window_duration=window_duration,
            integrator_nodes=integrator_nodes,
        )
    )

    duration = jnp.maximum(
        jnp.abs(jnp.asarray(window_duration, dtype=DTYPE)),
        jnp.asarray(1.0e-12, dtype=DTYPE),
    )
    trace_start = trace_normalized_physical_weights_for_axis(
        lnOmega_start,
        lnOmega_start_shift,
        lnOmega_start,
        lnOmega_start_shift,
        axis_name=axis_name,
    )
    trace_end = trace_normalized_physical_weights_for_axis(
        lnOmega_end,
        lnOmega_end_shift,
        lnOmega_start,
        lnOmega_start_shift,
        axis_name=axis_name,
    )
    trace_forward = (trace_end - trace_start) / duration
    global_walker_count = _global_walker_count(
        int(trace_start.shape[0]), axis_name=axis_name
    )
    trace_start_sum = jnp.sum(trace_start)
    trace_end_sum = jnp.sum(trace_end)
    if axis_name is not None:
        trace_start_sum = lax.psum(trace_start_sum, axis_name)
        trace_end_sum = lax.psum(trace_end_sum, axis_name)
    trace_start_mean = safe_complex_denominator(
        trace_start_sum / global_walker_count
    )
    trace_end_mean = trace_end_sum / global_walker_count
    trace_ratio = trace_end_mean / trace_start_mean
    trace_influence = (
        trace_end - trace_ratio * trace_start
    ) / (duration * trace_start_mean)

    num_site = int(alpha_start.shape[-1])
    trace_forward = jnp.broadcast_to(
        trace_forward[None, :, None],
        (1, trace_forward.shape[0], num_site),
    )
    trace_influence = jnp.broadcast_to(
        trace_influence[None, :, None],
        (1, trace_influence.shape[0], num_site),
    )
    return (
        jnp.concatenate((trace_forward, forward_residual), axis=0),
        jnp.concatenate((trace_influence, influence_residual), axis=0),
    )


def _prepare_rollout_context(
    *,
    alpha,
    dt,
    t0,
    U,
    gamma,
    F,
    Delta,
    J,
    n0,
    center_axis_name=None,
    sde_root_rtol=SDE_ROOT_RTOL_DEFAULT,
    sde_root_atol=SDE_ROOT_ATOL_DEFAULT,
    sde_affine_expm_order: int = 6,
    sde_affine_expm_substeps: int = 1,
    sde_newton_damping_steps: int = 4,
):
    _, num_site = alpha.shape
    t = jnp.asarray(t0, dtype=DTYPE)
    dt = jnp.asarray(dt, dtype=DTYPE)
    solver_coefficients = SolverCoefficients(
        dt=dt,
        U=U,
        gamma=gamma,
        F=F,
        Delta=Delta,
        J=J,
        center_axis_name=center_axis_name,
        root_rtol=jnp.asarray(sde_root_rtol, dtype=DTYPE),
        root_atol=jnp.asarray(sde_root_atol, dtype=DTYPE),
        affine_expm_order=int(sde_affine_expm_order),
        affine_expm_substeps=int(sde_affine_expm_substeps),
        newton_damping_steps=int(sde_newton_damping_steps),
    )
    physical_params = jnp.array([U, gamma, n0, F.real, F.imag, Delta], dtype=NNDTYPE)
    noise_scale = jnp.sqrt(dt).astype(DTYPE)
    return num_site, t, dt, solver_coefficients, physical_params, noise_scale


def _advance_one_step_state_with_gauge_fields(
    *,
    key_base: jax.Array,
    step_index,
    lnOmega_t: jnp.ndarray,
    alpha_t: jnp.ndarray,
    beta_t: jnp.ndarray,
    t_t: DTYPE,
    drift_g: jnp.ndarray,
    drift_f: jnp.ndarray,
    diffusion_g: jnp.ndarray,
    noise_scale: DTYPE,
    solver_coefficients: SolverCoefficients,
    sde_max_iter: int,
    sde_solver: str,
):
    W, S = alpha_t.shape
    step_key = random.fold_in(key_base, step_index)
    noise = random.normal(step_key, (2, W, S), dtype=DTYPE) * noise_scale
    dW_t, dWp_t = noise[0], noise[1]
    g_complex = to_c(drift_g[..., :S], drift_g[..., S:])
    f_complex = to_c(drift_f[..., :S], drift_f[..., S:])
    dln_omega_uncentered = jnp.sum(
        g_complex * dW_t
        + f_complex * dWp_t
        - 0.5 * (g_complex**2 + f_complex**2) * solver_coefficients.dt,
        axis=-1,
    )
    lnOmega_center_shift = jnp.mean(jnp.real(dln_omega_uncentered), axis=0)
    if solver_coefficients.center_axis_name is not None:
        lnOmega_center_shift = lax.pmean(
            lnOmega_center_shift,
            solver_coefficients.center_axis_name,
        )

    solver_cls = get_solver(sde_solver)
    next_state = solver_cls.step(
        state=SolverState(ln_omega=lnOmega_t, alpha=alpha_t, beta=beta_t),
        gauge_fields=GaugeFields(drift_g=drift_g, drift_f=drift_f, diffusion_g=diffusion_g),
        noise=SolverNoise(dW=dW_t, dWp=dWp_t),
        coefficients=solver_coefficients,
        sde_max_iter=sde_max_iter,
    )
    lnOmega_next = next_state.ln_omega
    alpha_next = next_state.alpha
    beta_next = next_state.beta
    return {
        "lnOmega_next": lnOmega_next,
        "alpha_next": alpha_next,
        "beta_next": beta_next,
        "t_next": t_t + solver_coefficients.dt,
        "drift_g": drift_g,
        "drift_f": drift_f,
        "diffusion_g": diffusion_g,
        "lnOmega_center_shift": lnOmega_center_shift,
    }


def _zero_pareto_k_diagnostics(zero, num_site: int):
    return {
        "pareto_k_loss": zero,
        "pareto_k_loss_terms": jnp.zeros((PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE),
        "pareto_k_site_terms": jnp.zeros((PARETO_K_MONOMIAL_COUNT, int(num_site)), dtype=DTYPE),
        "pareto_k_mean_terms": jnp.zeros((PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE),
        "pareto_k_max_terms": jnp.zeros((PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE),
        "pareto_k_site_indices": jnp.zeros((PARETO_K_MONOMIAL_COUNT, int(num_site)), dtype=DTYPE),
        "pareto_k_mean": zero,
        "pareto_k_max": zero,
        "pareto_k_warning_fraction": zero,
        "pareto_k_bad_fraction": zero,
    }


def _gauge_regularizer_components(
    *,
    drift_g: jnp.ndarray,
    drift_f: jnp.ndarray,
    diffusion_g: jnp.ndarray,
):
    drift_g_norm = jnp.linalg.norm(drift_g, axis=-1) ** 2
    drift_f_norm = jnp.linalg.norm(drift_f, axis=-1) ** 2
    diffusion_g_norm = jnp.linalg.norm(diffusion_g, axis=-1) ** 2
    loss_gauge_drift = jnp.mean(drift_g_norm + drift_f_norm)
    loss_gauge_diffusion = jnp.mean(diffusion_g_norm)
    loss_gauge = loss_gauge_drift + loss_gauge_diffusion
    return loss_gauge, loss_gauge_drift, loss_gauge_diffusion


def _advance_frozen_window_segment(
    *,
    key_base: jax.Array,
    start_step: int,
    num_steps: int,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    time: DTYPE,
    lnOmega_shift: DTYPE,
    drift_g: jnp.ndarray,
    drift_f: jnp.ndarray,
    diffusion_g: jnp.ndarray,
    noise_scale: DTYPE,
    solver_coefficients: SolverCoefficients,
    sde_max_iter: int,
    sde_solver: str,
):
    if num_steps <= 0:
        return lnOmega, alpha, beta, time, lnOmega_shift

    def frozen_body(carry, step_index):
        lnOmega_t, alpha_t, beta_t, t_t, lnOmega_shift_t = carry
        step = _advance_one_step_state_with_gauge_fields(
            key_base=key_base,
            step_index=step_index,
            lnOmega_t=lnOmega_t,
            alpha_t=alpha_t,
            beta_t=beta_t,
            t_t=t_t,
            drift_g=drift_g,
            drift_f=drift_f,
            diffusion_g=diffusion_g,
            sde_max_iter=sde_max_iter,
            sde_solver=sde_solver,
            noise_scale=noise_scale,
            solver_coefficients=solver_coefficients,
        )
        return (
            step["lnOmega_next"],
            step["alpha_next"],
            step["beta_next"],
            step["t_next"],
            lnOmega_shift_t + lax.stop_gradient(step["lnOmega_center_shift"]),
        ), None

    (lnOmega_end, alpha_end, beta_end, time_end, lnOmega_shift_end), _ = lax.scan(
        frozen_body,
        (lnOmega, alpha, beta, time, lnOmega_shift),
        jnp.arange(start_step, start_step + num_steps),
        length=num_steps,
    )
    return lnOmega_end, alpha_end, beta_end, time_end, lnOmega_shift_end


def _advance_training_window_segment(
    *,
    key_base: jax.Array,
    start_step: int,
    num_steps: int,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    time: DTYPE,
    lnOmega_shift: DTYPE,
    window_drift_g: jnp.ndarray,
    window_drift_f: jnp.ndarray,
    window_diffusion_g: jnp.ndarray,
    apply_neural_gauge_every_steps: int,
    apply_fn,
    params,
    physical_params,
    gauge_weight: DTYPE,
    gauge_mode: str,
    neural_gauge_components: str,
    neural_gauge_each_apply: bool,
    analytic_target_time: DTYPE,
    noise_scale: DTYPE,
    solver_coefficients: SolverCoefficients,
    sde_max_iter: int,
    sde_solver: str,
):
    """Advance a sub-window, optionally refreshing neural gauge fields by microstep cadence."""

    if num_steps <= 0:
        zero = jnp.asarray(0.0, dtype=DTYPE)
        return (
            lnOmega,
            alpha,
            beta,
            time,
            lnOmega_shift,
            window_drift_g,
            window_drift_f,
            window_diffusion_g,
            zero,
            zero,
            zero,
        )

    should_refresh_within_window = bool(
        gauge_mode in NEURAL_GAUGE_MODES and int(apply_neural_gauge_every_steps) > 0
    )

    if not should_refresh_within_window:
        lnOmega_end, alpha_end, beta_end, time_end, lnOmega_shift_end = _advance_frozen_window_segment(
            key_base=key_base,
            start_step=start_step,
            num_steps=num_steps,
            lnOmega=lnOmega,
            alpha=alpha,
            beta=beta,
            time=time,
            lnOmega_shift=lnOmega_shift,
            drift_g=window_drift_g,
            drift_f=window_drift_f,
            diffusion_g=window_diffusion_g,
            noise_scale=noise_scale,
            solver_coefficients=solver_coefficients,
            sde_max_iter=sde_max_iter,
            sde_solver=sde_solver,
        )
        loss_gauge, loss_gauge_drift, loss_gauge_diffusion = _gauge_regularizer_components(
            drift_g=window_drift_g,
            drift_f=window_drift_f,
            diffusion_g=window_diffusion_g,
        )
        return (
            lnOmega_end,
            alpha_end,
            beta_end,
            time_end,
            lnOmega_shift_end,
            window_drift_g,
            window_drift_f,
            window_diffusion_g,
            loss_gauge,
            loss_gauge_drift,
            loss_gauge_diffusion,
        )

    cadence = max(int(apply_neural_gauge_every_steps), 1)
    lnOmega_t, alpha_t, beta_t, time_t, lnOmega_shift_t = (
        lnOmega,
        alpha,
        beta,
        time,
        lnOmega_shift,
    )
    drift_g_t, drift_f_t, diffusion_g_t = window_drift_g, window_drift_f, window_diffusion_g
    loss_gauge_sum = jnp.asarray(0.0, dtype=DTYPE)
    loss_gauge_drift_sum = jnp.asarray(0.0, dtype=DTYPE)
    loss_gauge_diffusion_sum = jnp.asarray(0.0, dtype=DTYPE)
    step_offset = 0
    while step_offset < int(num_steps):
        absolute_step = int(start_step) + step_offset
        if absolute_step > 0 and absolute_step % cadence == 0:
            drift_g_t, drift_f_t, diffusion_g_t = _compute_scaled_gauge_fields(
                apply_fn=apply_fn,
                params=params,
                lnOmega=lnOmega_t,
                alpha=alpha_t,
                beta=beta_t,
                time=time_t,
                physical_params=physical_params,
                gauge_weight=gauge_weight,
                gauge_mode=gauge_mode,
                neural_gauge_components=neural_gauge_components,
                neural_gauge_each_apply=neural_gauge_each_apply,
                analytic_target_time=analytic_target_time,
            )
        steps_to_refresh = cadence - (absolute_step % cadence)
        chunk_steps = min(int(num_steps) - step_offset, steps_to_refresh)
        chunk_weight = jnp.asarray(chunk_steps / max(int(num_steps), 1), dtype=DTYPE)
        chunk_loss_gauge, chunk_loss_gauge_drift, chunk_loss_gauge_diffusion = (
            _gauge_regularizer_components(
                drift_g=drift_g_t,
                drift_f=drift_f_t,
                diffusion_g=diffusion_g_t,
            )
        )
        loss_gauge_sum = loss_gauge_sum + chunk_weight * chunk_loss_gauge
        loss_gauge_drift_sum = loss_gauge_drift_sum + chunk_weight * chunk_loss_gauge_drift
        loss_gauge_diffusion_sum = loss_gauge_diffusion_sum + chunk_weight * chunk_loss_gauge_diffusion
        lnOmega_t, alpha_t, beta_t, time_t, lnOmega_shift_t = _advance_frozen_window_segment(
            key_base=key_base,
            start_step=absolute_step,
            num_steps=chunk_steps,
            lnOmega=lnOmega_t,
            alpha=alpha_t,
            beta=beta_t,
            time=time_t,
            lnOmega_shift=lnOmega_shift_t,
            drift_g=drift_g_t,
            drift_f=drift_f_t,
            diffusion_g=diffusion_g_t,
            noise_scale=noise_scale,
            solver_coefficients=solver_coefficients,
            sde_max_iter=sde_max_iter,
            sde_solver=sde_solver,
        )
        step_offset += chunk_steps
    return (
        lnOmega_t,
        alpha_t,
        beta_t,
        time_t,
        lnOmega_shift_t,
        drift_g_t,
        drift_f_t,
        diffusion_g_t,
        loss_gauge_sum,
        loss_gauge_drift_sum,
        loss_gauge_diffusion_sum,
    )


@partial(
    jax.jit,
    static_argnames=(
        "apply_fn",
        "N_steps",
        "apply_neural_gauge_every_steps",
        "sde_max_iter",
        "sde_solver",
        "sde_affine_expm_order",
        "sde_affine_expm_substeps",
        "sde_newton_damping_steps",
        "gauge_mode",
        "neural_gauge_components",
        "pareto_k_applied_quantities",
        "pareto_k_applied_quantities_mode",
        "pareto_k_monomials",
        "operator_monomials",
        "residual_gmm_integrator_nodes",
        "pareto_k_envelope_excess",
        "pareto_k_tail_fraction",
        "pareto_k_min_tail_count",
        "enable_loss_pareto_k",
        "enable_loss_residual_gmm",
        "enable_loss_gauge",
        "enable_loss_ess",
        "center_axis_name",
    ),
)
def run_one_window_training_profile(
    key: jax.Array,
    apply_fn,
    params,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    U: DTYPE,
    gamma: DTYPE,
    F: CDTYPE,
    Delta: DTYPE,
    dt: DTYPE,
    N_steps: int,
    t0: DTYPE,
    lnOmega_shift: DTYPE = 0.0,
    apply_neural_gauge_every_steps: int = 0,
    neural_gauge_each_apply: bool = False,
    gauge_weight: DTYPE = 1.0,
    n0: DTYPE = 1.0,
    sde_max_iter: int = 4,
    sde_solver: str = "semi_implicit_midpoint",
    sde_root_rtol: DTYPE = SDE_ROOT_RTOL_DEFAULT,
    sde_root_atol: DTYPE = SDE_ROOT_ATOL_DEFAULT,
    sde_affine_expm_order: int = 6,
    sde_affine_expm_substeps: int = 1,
    sde_newton_damping_steps: int = 4,
    pareto_k_threshold: DTYPE = 0.7,
    pareto_k_threshold_tau: DTYPE = 0.1,
    pareto_k_envelope_beta: DTYPE = 0.5,
    pareto_k_envelope_excess: str = "log",
    operator_monomials: tuple = (),
    residual_gmm_integrator_nodes: int = 6,
    residual_gmm_d_clip: DTYPE = 10.0,
    residual_gmm_cov_floor: DTYPE = 1.0e-8,
    residual_gmm_cov_shrinkage: DTYPE = 0.05,
    residual_gmm_lagged_covariance: jnp.ndarray | None = None,
    residual_gmm_lagged_covariance_initialized: jnp.ndarray | None = None,
    pareto_k_tail_fraction: float = 0.01,
    pareto_k_min_tail_count: int = 32,
    loss_pareto_k_monomial_weights: jnp.ndarray = 1.0,
    q_winsor: float = 0.95,
    pareto_k_applied_quantities: int = 6,
    pareto_k_applied_quantities_mode: str = "upto",
    pareto_k_monomials: tuple = (),
    J=0.0,
    gauge_mode: str = "neural_graph",
    neural_gauge_components: str = "both",
    analytic_target_time: DTYPE = 0.0,
    enable_loss_pareto_k: bool = False,
    enable_loss_residual_gmm: bool = False,
    enable_loss_gauge: bool = True,
    enable_loss_ess: bool = False,
    center_axis_name=None,
):
    _, t, dt, solver_coefficients, physical_params, noise_scale = (
        _prepare_rollout_context(
            alpha=alpha,
            dt=dt,
            t0=t0,
            U=U,
            gamma=gamma,
            F=F,
            Delta=Delta,
            J=J,
            n0=n0,
            center_axis_name=center_axis_name,
            sde_root_rtol=sde_root_rtol,
            sde_root_atol=sde_root_atol,
            sde_affine_expm_order=sde_affine_expm_order,
            sde_affine_expm_substeps=sde_affine_expm_substeps,
            sde_newton_damping_steps=sde_newton_damping_steps,
        )
    )
    zero = jnp.asarray(0.0, dtype=DTYPE)
    lnOmega_shift = lax.stop_gradient(jnp.asarray(lnOmega_shift, dtype=DTYPE))
    window_duration = dt * jnp.asarray(N_steps, dtype=DTYPE)
    window_drift_g, window_drift_f, window_diffusion_g = (
        _compute_scaled_gauge_fields(
            apply_fn=apply_fn,
            params=params,
            lnOmega=lnOmega,
            alpha=alpha,
            beta=beta,
            time=t,
            physical_params=physical_params,
            gauge_weight=gauge_weight,
            gauge_mode=gauge_mode,
            neural_gauge_components=neural_gauge_components,
            neural_gauge_each_apply=neural_gauge_each_apply,
            analytic_target_time=analytic_target_time,
        )
    )

    residual_integrator_nodes = (
        normalize_residual_gmm_integrator_nodes(
            residual_gmm_integrator_nodes
        )
    )
    segment_count = (
        residual_integrator_nodes - 1
        if enable_loss_residual_gmm
        else 1
    )
    if int(N_steps) % segment_count != 0:
        raise ValueError(
            "projected-residual quadrature requires N_steps divisible by "
            f"{segment_count}; received N_steps={int(N_steps)}"
        )

    segment_steps = int(N_steps) // segment_count
    segment_lnOmega = lnOmega
    segment_alpha = alpha
    segment_beta = beta
    segment_time = t
    segment_lnOmega_shift = lnOmega_shift
    segment_drift_g = window_drift_g
    segment_drift_f = window_drift_f
    segment_diffusion_g = window_diffusion_g
    segment_loss_gauge = zero
    segment_loss_gauge_drift = zero
    segment_loss_gauge_diffusion = zero
    segment_snapshots = []

    for segment_index in range(segment_count):
        (
            segment_lnOmega,
            segment_alpha,
            segment_beta,
            segment_time,
            segment_lnOmega_shift,
            segment_drift_g,
            segment_drift_f,
            segment_diffusion_g,
            current_loss_gauge,
            current_loss_gauge_drift,
            current_loss_gauge_diffusion,
        ) = _advance_training_window_segment(
            key_base=key,
            start_step=segment_index * segment_steps,
            num_steps=segment_steps,
            lnOmega=segment_lnOmega,
            alpha=segment_alpha,
            beta=segment_beta,
            time=segment_time,
            lnOmega_shift=segment_lnOmega_shift,
            window_drift_g=segment_drift_g,
            window_drift_f=segment_drift_f,
            window_diffusion_g=segment_diffusion_g,
            apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
            apply_fn=apply_fn,
            params=params,
            physical_params=physical_params,
            gauge_weight=gauge_weight,
            gauge_mode=gauge_mode,
            neural_gauge_components=neural_gauge_components,
            neural_gauge_each_apply=neural_gauge_each_apply,
            analytic_target_time=analytic_target_time,
            noise_scale=noise_scale,
            solver_coefficients=solver_coefficients,
            sde_max_iter=sde_max_iter,
            sde_solver=sde_solver,
        )
        segment_snapshots.append(
            (segment_lnOmega, segment_alpha, segment_beta)
        )
        segment_weight = jnp.asarray(
            segment_steps / max(int(N_steps), 1), dtype=DTYPE
        )
        segment_loss_gauge += segment_weight * current_loss_gauge
        segment_loss_gauge_drift += segment_weight * current_loss_gauge_drift
        segment_loss_gauge_diffusion += (
            segment_weight * current_loss_gauge_diffusion
        )

    lnOmega_end = segment_lnOmega
    alpha_end = segment_alpha
    beta_end = segment_beta
    t_end = segment_time
    lnOmega_shift_end = segment_lnOmega_shift

    if enable_loss_pareto_k:
        diagnostics_end = compute_onsite_pareto_k_diagnostics(
            lnOmega=lnOmega_end,
            alpha=alpha_end,
            beta=beta_end,
            pareto_k_threshold=pareto_k_threshold,
            pareto_k_threshold_tau=pareto_k_threshold_tau,
            pareto_k_envelope_beta=pareto_k_envelope_beta,
            pareto_k_envelope_excess=pareto_k_envelope_excess,
            pareto_k_tail_fraction=pareto_k_tail_fraction,
            pareto_k_min_tail_count=pareto_k_min_tail_count,
            loss_pareto_k_monomial_weights=loss_pareto_k_monomial_weights,
            q_winsor=q_winsor,
            applied_quantities=pareto_k_applied_quantities,
            applied_quantities_mode=pareto_k_applied_quantities_mode,
            monomials=pareto_k_monomials,
            axis_name=solver_coefficients.center_axis_name,
        )
    else:
        diagnostics_end = _zero_pareto_k_diagnostics(zero, alpha.shape[-1])

    if enable_loss_gauge:
        loss_gauge = segment_loss_gauge
        loss_gauge_drift = segment_loss_gauge_drift
        loss_gauge_diffusion = segment_loss_gauge_diffusion
    else:
        loss_gauge = zero
        loss_gauge_drift = zero
        loss_gauge_diffusion = zero

    if enable_loss_ess:
        # The solver's common real centering shift cancels in this walker
        # variance, so the window spread is the physical weight-entropy
        # spend of the window in nats.
        window_log_weight_spread = log_weight_increment_spread_for_axis(
            lnOmega_end - lnOmega,
            axis_name=solver_coefficients.center_axis_name,
        )
        ess_ratio_end = complex_ess_ratio_for_axis(
            lnOmega_end,
            axis_name=solver_coefficients.center_axis_name,
        )
    else:
        window_log_weight_spread = zero
        ess_ratio_end = zero

    if enable_loss_residual_gmm:
        monomials = normalize_projected_residual_monomials(
            operator_monomials
        )
        residual_node_states = (
            (lnOmega, alpha, beta),
            *segment_snapshots,
        )
        (
            residual_gmm_window_cloud,
            residual_gmm_window_influence_cloud,
        ) = _projected_residual_clouds(
            operator_monomials=monomials,
            U=U,
            gamma=gamma,
            Delta=Delta,
            F=F,
            J=J,
            lnOmega_start=lnOmega,
            lnOmega_start_shift=lnOmega_shift,
            alpha_start=alpha,
            beta_start=beta,
            lnOmega_end=lnOmega_end,
            lnOmega_end_shift=lnOmega_shift_end,
            alpha_end=alpha_end,
            beta_end=beta_end,
            window_duration=window_duration,
            residual_node_states=residual_node_states,
            residual_gmm_integrator_nodes=(
                residual_integrator_nodes
            ),
            axis_name=solver_coefficients.center_axis_name,
        )
        (
            loss_residual_gmm_scalar,
            residual_gmm_aux,
        ) = _residual_gmm_loss_from_residual_cloud(
            residual_gmm_window_cloud,
            residual_influence_cloud=(
                residual_gmm_window_influence_cloud
            ),
            d_clip=residual_gmm_d_clip,
            cov_floor=residual_gmm_cov_floor,
            cov_shrinkage=residual_gmm_cov_shrinkage,
            lagged_covariance=residual_gmm_lagged_covariance,
            lagged_covariance_initialized=(
                residual_gmm_lagged_covariance_initialized
            ),
            axis_name=solver_coefficients.center_axis_name,
        )
    else:
        residual_gmm_window_cloud = jnp.zeros(
            (0, alpha.shape[0], alpha.shape[-1]), dtype=CDTYPE
        )
        loss_residual_gmm_scalar = zero
        residual_gmm_aux = (
            _zero_residual_gmm_components(
                zero, alpha.shape[-1], 0
            )
        )

    pareto_k_loss_terms = diagnostics_end["pareto_k_loss_terms"]
    pareto_k_site_terms = diagnostics_end["pareto_k_site_terms"]
    pareto_k_objective = (
        diagnostics_end["pareto_k_loss"] if enable_loss_pareto_k else zero
    )
    residual_gmm_term_count = int(
        residual_gmm_window_cloud.shape[0]
    )
    residual_gmm_loss_terms = residual_gmm_aux.get(
        "loss_residual_gmm_terms",
        jnp.zeros((residual_gmm_term_count,), dtype=DTYPE),
    )
    residual_gmm_site_terms = residual_gmm_aux.get(
        "loss_residual_gmm_site_terms",
        jnp.zeros(
            (
                residual_gmm_term_count,
                residual_gmm_window_cloud.shape[-1],
            ),
            dtype=DTYPE,
        ),
    )
    residual_gmm_objective = (
        loss_residual_gmm_scalar
        if enable_loss_residual_gmm
        else zero
    )

    return key, lnOmega_end, alpha_end, beta_end, {
        "loss_pareto_k": pareto_k_objective,
        "loss_pareto_k_objective": pareto_k_objective,
        "loss_pareto_k_terms": pareto_k_loss_terms,
        "loss_pareto_k_site_terms": pareto_k_site_terms,
        **{
            term: pareto_k_loss_terms[index]
            for index, term in enumerate(PARETO_K_MONOMIAL_TERMS)
        },
        "pareto_k_mean": diagnostics_end["pareto_k_mean"],
        "pareto_k_max": diagnostics_end["pareto_k_max"],
        "pareto_k_warning_fraction": diagnostics_end[
            "pareto_k_warning_fraction"
        ],
        "pareto_k_bad_fraction": diagnostics_end["pareto_k_bad_fraction"],
        "loss_residual_gmm": residual_gmm_objective,
        "loss_residual_gmm_log1p": residual_gmm_aux.get(
            "loss_residual_gmm_log1p", zero
        ),
        "loss_residual_gmm_raw": residual_gmm_aux.get(
            "loss_residual_gmm_raw", zero
        ),
        "loss_residual_gmm_terms": (
            residual_gmm_loss_terms
        ),
        "loss_residual_gmm_site_terms": (
            residual_gmm_site_terms
        ),
        "residual_gmm_z_mean": residual_gmm_aux.get(
            "residual_gmm_z_mean", zero
        ),
        "residual_gmm_z_max": residual_gmm_aux.get(
            "residual_gmm_z_max", zero
        ),
        "residual_gmm_radius_mean": residual_gmm_aux.get(
            "residual_gmm_radius_mean", zero
        ),
        "residual_gmm_radius_max": residual_gmm_aux.get(
            "residual_gmm_radius_max", zero
        ),
        "residual_gmm_warning_fraction": (
            residual_gmm_aux.get(
                "residual_gmm_warning_fraction", zero
            )
        ),
        "residual_gmm_bad_fraction": (
            residual_gmm_aux.get(
                "residual_gmm_bad_fraction", zero
            )
        ),
        "residual_gmm_covariance_estimate": (
            residual_gmm_aux.get(
                "residual_gmm_covariance_estimate",
                jnp.zeros(
                    (alpha.shape[-1], 0, 0),
                    dtype=DTYPE,
                ),
            )
        ),
        "residual_gmm_window_cloud": (
            residual_gmm_window_cloud
        ),
        "loss_gauge": loss_gauge,
        "loss_gauge_drift": loss_gauge_drift,
        "loss_gauge_diffusion": loss_gauge_diffusion,
        "window_log_weight_spread": window_log_weight_spread,
        "ess_ratio_end": ess_ratio_end,
        "lnOmega_shift_end": lnOmega_shift_end,
        "t_end": t_end,
    }


@partial(
    jax.jit,
    static_argnames=(
        "apply_fn",
        "N_steps",
        "apply_neural_gauge_every_steps",
        "sde_max_iter",
        "sde_solver",
        "sde_affine_expm_order",
        "sde_affine_expm_substeps",
        "sde_newton_damping_steps",
        "gauge_mode",
        "neural_gauge_components",
        "center_axis_name",
    ),
)
def run_one_window_simulation_rollout(
    key: jax.Array,
    apply_fn,
    params,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    U: DTYPE,
    gamma: DTYPE,
    F: CDTYPE,
    Delta: DTYPE,
    dt: DTYPE,
    N_steps: int,
    t0: DTYPE,
    apply_neural_gauge_every_steps: int = 0,
    gauge_weight: DTYPE = 1.0,
    n0: DTYPE = 1.0,
    sde_max_iter: int = 4,
    sde_solver: str = "semi_implicit_midpoint",
    sde_root_rtol: DTYPE = SDE_ROOT_RTOL_DEFAULT,
    sde_root_atol: DTYPE = SDE_ROOT_ATOL_DEFAULT,
    sde_affine_expm_order: int = 6,
    sde_affine_expm_substeps: int = 1,
    sde_newton_damping_steps: int = 4,
    J=0.0,
    gauge_mode: str = "neural_graph",
    neural_gauge_components: str = "both",
    analytic_target_time: DTYPE = 0.0,
    center_axis_name=None,
):
    _, t, _, solver_coefficients, physical_params, noise_scale = _prepare_rollout_context(
        alpha=alpha,
        dt=dt,
        t0=t0,
        U=U,
        gamma=gamma,
        F=F,
        Delta=Delta,
        J=J,
        n0=n0,
        center_axis_name=center_axis_name,
        sde_root_rtol=sde_root_rtol,
        sde_root_atol=sde_root_atol,
        sde_affine_expm_order=sde_affine_expm_order,
        sde_affine_expm_substeps=sde_affine_expm_substeps,
        sde_newton_damping_steps=sde_newton_damping_steps,
    )
    window_drift_g, window_drift_f, window_diffusion_g = _compute_scaled_gauge_fields(
        apply_fn=apply_fn,
        params=params,
        lnOmega=lnOmega,
        alpha=alpha,
        beta=beta,
        time=t,
        physical_params=physical_params,
        gauge_weight=gauge_weight,
        gauge_mode=gauge_mode,
        neural_gauge_components=neural_gauge_components,
        analytic_target_time=analytic_target_time,
    )

    should_refresh_within_window = bool(
        gauge_mode in NEURAL_GAUGE_MODES and int(apply_neural_gauge_every_steps) > 0
    )

    if not should_refresh_within_window:
        lnOmega_end, alpha_end, beta_end, t_end, lnOmega_shift_end = (
            _advance_frozen_window_segment(
                key_base=key,
                start_step=0,
                num_steps=N_steps,
                lnOmega=lnOmega,
                alpha=alpha,
                beta=beta,
                time=t,
                lnOmega_shift=jnp.asarray(0.0, dtype=DTYPE),
                drift_g=window_drift_g,
                drift_f=window_drift_f,
                diffusion_g=window_diffusion_g,
                noise_scale=noise_scale,
                solver_coefficients=solver_coefficients,
                sde_max_iter=sde_max_iter,
                sde_solver=sde_solver,
            )
        )
        return key, lnOmega_end, alpha_end, beta_end, {
            "t_end": t_end,
            "lnOmega_center_shift": lnOmega_shift_end,
        }

    cadence = max(int(apply_neural_gauge_every_steps), 1)
    lnOmega_t, alpha_t, beta_t, time_t = lnOmega, alpha, beta, t
    lnOmega_shift_t = jnp.asarray(0.0, dtype=DTYPE)
    drift_g_t, drift_f_t, diffusion_g_t = window_drift_g, window_drift_f, window_diffusion_g
    step_offset = 0
    while step_offset < int(N_steps):
        absolute_step = step_offset
        if absolute_step > 0 and absolute_step % cadence == 0:
            drift_g_t, drift_f_t, diffusion_g_t = _compute_scaled_gauge_fields(
                apply_fn=apply_fn,
                params=params,
                lnOmega=lnOmega_t,
                alpha=alpha_t,
                beta=beta_t,
                time=time_t,
                physical_params=physical_params,
                gauge_weight=gauge_weight,
                gauge_mode=gauge_mode,
                neural_gauge_components=neural_gauge_components,
                analytic_target_time=analytic_target_time,
            )
        steps_to_refresh = cadence - (absolute_step % cadence)
        chunk_steps = min(int(N_steps) - step_offset, steps_to_refresh)
        lnOmega_t, alpha_t, beta_t, time_t, lnOmega_shift_t = (
            _advance_frozen_window_segment(
                key_base=key,
                start_step=absolute_step,
                num_steps=chunk_steps,
                lnOmega=lnOmega_t,
                alpha=alpha_t,
                beta=beta_t,
                time=time_t,
                lnOmega_shift=lnOmega_shift_t,
                drift_g=drift_g_t,
                drift_f=drift_f_t,
                diffusion_g=diffusion_g_t,
                noise_scale=noise_scale,
                solver_coefficients=solver_coefficients,
                sde_max_iter=sde_max_iter,
                sde_solver=sde_solver,
            )
        )
        step_offset += chunk_steps
    return key, lnOmega_t, alpha_t, beta_t, {
        "t_end": time_t,
        "lnOmega_center_shift": lnOmega_shift_t,
    }


@partial(
    jax.jit,
    static_argnames=(
        "apply_fn",
        "N_steps",
        "N_windows",
        "apply_neural_gauge_every_steps",
        "sde_max_iter",
        "sde_solver",
        "sde_affine_expm_order",
        "sde_affine_expm_substeps",
        "sde_newton_damping_steps",
        "gauge_mode",
        "neural_gauge_components",
        "pareto_k_applied_quantities",
        "pareto_k_applied_quantities_mode",
        "pareto_k_monomials",
        "operator_monomials",
        "residual_gmm_time_aggregation",
        "residual_gmm_integrator_nodes",
        "pareto_k_envelope_excess",
        "pareto_k_tail_fraction",
        "pareto_k_min_tail_count",
        "enable_loss_pareto_k",
        "enable_loss_residual_gmm",
        "enable_loss_gauge",
        "enable_loss_L2",
        "enable_loss_ess",
        "center_axis_name",
    ),
)
def _loss_and_aux_all_windows(
    key: jax.Array,
    apply_fn,
    params,
    lnOmega0: jnp.ndarray,
    alpha0: jnp.ndarray,
    beta0: jnp.ndarray,
    U: DTYPE,
    gamma: DTYPE,
    F: CDTYPE,
    Delta: DTYPE,
    dt: DTYPE,
    N_steps: int,
    N_windows: int,
    window_loss_weights: jnp.ndarray,
    lnOmega_shift0: DTYPE = 0.0,
    operator_monomials: tuple = (),
    residual_gmm_covariance_bank: jnp.ndarray | None = None,
    residual_gmm_covariance_initialized: jnp.ndarray | None = None,
    apply_neural_gauge_every_steps: int = 0,
    neural_gauge_each_apply: bool = False,
    sde_max_iter: int = 4,
    sde_solver: str = "semi_implicit_midpoint",
    sde_root_rtol: DTYPE = SDE_ROOT_RTOL_DEFAULT,
    sde_root_atol: DTYPE = SDE_ROOT_ATOL_DEFAULT,
    sde_affine_expm_order: int = 6,
    sde_affine_expm_substeps: int = 1,
    sde_newton_damping_steps: int = 4,
    t0: DTYPE = 0.0,
    gauge_weight: DTYPE = 1.0,
    pareto_k_threshold: DTYPE = 0.7,
    pareto_k_threshold_tau: DTYPE = 0.1,
    pareto_k_envelope_beta: DTYPE = 0.5,
    pareto_k_envelope_excess: str = "log",
    residual_gmm_d_clip: DTYPE = 10.0,
    residual_gmm_cov_floor: DTYPE = 1.0e-8,
    residual_gmm_cov_shrinkage: DTYPE = 0.05,
    residual_gmm_time_aggregation: str = "mean",
    residual_gmm_integrator_nodes: int = 6,
    residual_gmm_time_beta: DTYPE = 0.0,
    pareto_k_tail_fraction: float = 0.01,
    pareto_k_min_tail_count: int = 32,
    n0: DTYPE = 1.0,
    loss_pareto_k_prefactor: DTYPE = 0.0,
    loss_pareto_k_monomial_weights: jnp.ndarray = 1.0,
    loss_residual_gmm_prefactor: DTYPE = 0.0,
    loss_L2_prefactor: DTYPE = 1e-4,
    loss_gauge_prefactor: DTYPE = 1e-2,
    loss_ess_prefactor: DTYPE = 0.0,
    loss_ess_window_budget: DTYPE = 0.0,
    q_winsor: float = 0.95,
    pareto_k_applied_quantities: int = 6,
    pareto_k_applied_quantities_mode: str = "upto",
    pareto_k_monomials: tuple = (),
    J=0.0,
    gauge_mode: str = "neural_graph",
    neural_gauge_components: str = "both",
    analytic_target_time: DTYPE = 0.0,
    enable_loss_pareto_k: bool = False,
    enable_loss_residual_gmm: bool = False,
    enable_loss_gauge: bool = True,
    enable_loss_L2: bool = True,
    enable_loss_ess: bool = False,
    center_axis_name=None,
):
    def _validate_covariance_bank(
        bank: jnp.ndarray | None,
        initialized: jnp.ndarray | None,
    ) -> None:
        if (bank is None) != (initialized is None):
            raise ValueError(
                "windowwise residual covariance bank and initialization mask "
                "must be provided together"
            )
        if bank is None:
            return
        if (
            bank.ndim != 3
            or bank.shape[0] != int(N_windows)
            or bank.shape[-2] != bank.shape[-1]
        ):
            raise ValueError(
                "windowwise residual covariance bank must have shape "
                "(N_windows, channel, channel) for the shared site-averaged "
                f"covariance with N_windows={int(N_windows)}; "
                f"received {bank.shape}"
            )
        if initialized.shape != (int(N_windows),):
            raise ValueError(
                "windowwise residual covariance initialization mask must have "
                f"shape {(int(N_windows),)}; received "
                f"{initialized.shape}"
            )

    _validate_covariance_bank(
        residual_gmm_covariance_bank,
        residual_gmm_covariance_initialized,
    )
    residual_monomials = (
        normalize_projected_residual_monomials(operator_monomials)
        if enable_loss_residual_gmm
        else ()
    )
    if enable_loss_residual_gmm:
        if residual_gmm_covariance_bank is not None:
            expected_real_channels = 2 * (1 + len(residual_monomials))
            expected_shape = (
                expected_real_channels,
                expected_real_channels,
            )
            if (
                residual_gmm_covariance_bank.shape[-2:]
                != expected_shape
            ):
                raise ValueError(
                    "windowwise residual covariance bank is incompatible with "
                    "the trace-first onsite channel basis: expected trailing "
                    f"shape {expected_shape}, received "
                    f"{residual_gmm_covariance_bank.shape[-2:]}"
                )

    zero = jnp.asarray(0.0, dtype=DTYPE)
    neg_inf = jnp.asarray(-jnp.inf, dtype=DTYPE)
    lnOmega_shift0 = lax.stop_gradient(
        jnp.asarray(lnOmega_shift0, dtype=DTYPE)
    )
    zero_pareto_k_terms = jnp.zeros(
        (PARETO_K_MONOMIAL_COUNT,), dtype=DTYPE
    )
    zero_pareto_k_site_terms = jnp.zeros(
        (PARETO_K_MONOMIAL_COUNT, alpha0.shape[-1]), dtype=DTYPE
    )
    residual_gmm_term_count = (
        1 + len(residual_monomials)
        if enable_loss_residual_gmm
        else 0
    )
    zero_residual_gmm_terms = jnp.zeros(
        (residual_gmm_term_count,), dtype=DTYPE
    )
    zero_residual_gmm_site_terms = jnp.zeros(
        (
            residual_gmm_term_count,
            alpha0.shape[-1],
        ),
        dtype=DTYPE,
    )
    residual_gmm_time_weight_sum = jnp.maximum(
        jnp.sum(jnp.asarray(window_loss_weights, dtype=DTYPE)),
        jnp.asarray(1.0e-12, dtype=DTYPE),
    )
    residual_gmm_time_beta_value = jnp.asarray(
        residual_gmm_time_beta, dtype=DTYPE
    )
    ess_window_budget_value = jnp.maximum(
        jnp.asarray(loss_ess_window_budget, dtype=DTYPE),
        jnp.asarray(1.0e-12, dtype=DTYPE),
    )

    def body(carry, i_win):
        (
            key_base,
            lnOmega_t,
            alpha_t,
            beta_t,
            t_t,
            lnOmega_shift_t,
            sumParetoKObjective,
            sumParetoKTerms,
            sumParetoKSiteTerms,
            sumParetoKMean,
            sumParetoKMax,
            sumParetoKWarningFraction,
            sumParetoKBadFraction,
            sumResidualGmmObjective,
            sumResidualGmmTimeScore,
            logResidualGmmTimeRisk,
            sumResidualGmmRaw,
            sumResidualGmmTerms,
            sumResidualGmmSiteTerms,
            sumResidualGmmZMean,
            sumResidualGmmZMax,
            sumResidualGmmRadiusMean,
            sumResidualGmmRadiusMax,
            sumResidualGmmWarningFraction,
            sumResidualGmmBadFraction,
            sumGauge,
            sumGaugeDrift,
            sumGaugeDiffusion,
            sumEssHinge,
            sumLogWeightSpread,
            minEssRatio,
        ) = carry
        window_weight = jnp.asarray(
            window_loss_weights[i_win], dtype=DTYPE
        )
        _, lnOmega_next, alpha_next, beta_next, scalars = (
            run_one_window_training_profile(
                key=random.fold_in(key_base, i_win),
                apply_fn=apply_fn,
                params=params,
                lnOmega=lnOmega_t,
                alpha=alpha_t,
                beta=beta_t,
                U=U,
                gamma=gamma,
                F=F,
                Delta=Delta,
                dt=dt,
                N_steps=N_steps,
                t0=t_t,
                lnOmega_shift=lnOmega_shift_t,
                apply_neural_gauge_every_steps=(
                    apply_neural_gauge_every_steps
                ),
                neural_gauge_each_apply=neural_gauge_each_apply,
                gauge_weight=gauge_weight,
                n0=n0,
                sde_max_iter=sde_max_iter,
                sde_solver=sde_solver,
                sde_root_rtol=sde_root_rtol,
                sde_root_atol=sde_root_atol,
                sde_affine_expm_order=sde_affine_expm_order,
                sde_affine_expm_substeps=sde_affine_expm_substeps,
                sde_newton_damping_steps=sde_newton_damping_steps,
                pareto_k_threshold=pareto_k_threshold,
                pareto_k_threshold_tau=pareto_k_threshold_tau,
                pareto_k_envelope_beta=pareto_k_envelope_beta,
                pareto_k_envelope_excess=pareto_k_envelope_excess,
                operator_monomials=residual_monomials,
                residual_gmm_integrator_nodes=(
                    residual_gmm_integrator_nodes
                ),
                residual_gmm_d_clip=(
                    residual_gmm_d_clip
                ),
                residual_gmm_cov_floor=(
                    residual_gmm_cov_floor
                ),
                residual_gmm_cov_shrinkage=(
                    residual_gmm_cov_shrinkage
                ),
                residual_gmm_lagged_covariance=(
                    None
                    if residual_gmm_covariance_bank is None
                    else residual_gmm_covariance_bank[i_win]
                ),
                residual_gmm_lagged_covariance_initialized=(
                    None
                    if residual_gmm_covariance_initialized is None
                    else residual_gmm_covariance_initialized[i_win]
                ),
                pareto_k_tail_fraction=pareto_k_tail_fraction,
                pareto_k_min_tail_count=pareto_k_min_tail_count,
                loss_pareto_k_monomial_weights=(
                    loss_pareto_k_monomial_weights
                ),
                q_winsor=q_winsor,
                pareto_k_applied_quantities=pareto_k_applied_quantities,
                pareto_k_applied_quantities_mode=(
                    pareto_k_applied_quantities_mode
                ),
                pareto_k_monomials=pareto_k_monomials,
                J=J,
                gauge_mode=gauge_mode,
                neural_gauge_components=neural_gauge_components,
                analytic_target_time=analytic_target_time,
                enable_loss_pareto_k=enable_loss_pareto_k,
                enable_loss_residual_gmm=(
                    enable_loss_residual_gmm
                ),
                enable_loss_gauge=enable_loss_gauge,
                enable_loss_ess=enable_loss_ess,
                center_axis_name=center_axis_name,
            )
        )
        # The plain log1p mode is site-local as well as window-local.  The
        # legacy entropic_log1p mode intentionally transforms the site mean.
        residual_gmm_time_value = (
            scalars["loss_residual_gmm_log1p"]
            if residual_gmm_time_aggregation == "log1p"
            else scalars["loss_residual_gmm"]
        )
        residual_gmm_time_score = _time_aggregation_score(
            residual_gmm_time_value,
            residual_gmm_time_aggregation,
        )
        # Dimensionless one-sided budget violation: zero below the
        # per-window weight-entropy budget, quadratic above it.
        ess_hinge = jnp.square(
            jnp.maximum(
                scalars["window_log_weight_spread"] / ess_window_budget_value
                - jnp.asarray(1.0, dtype=DTYPE),
                jnp.asarray(0.0, dtype=DTYPE),
            )
        )
        return (
            (
                key_base,
                lnOmega_next,
                alpha_next,
                beta_next,
                scalars["t_end"],
                scalars["lnOmega_shift_end"],
                sumParetoKObjective
                + window_weight * scalars["loss_pareto_k_objective"],
                sumParetoKTerms
                + window_weight * scalars["loss_pareto_k_terms"],
                sumParetoKSiteTerms
                + window_weight * scalars["loss_pareto_k_site_terms"],
                sumParetoKMean
                + window_weight * scalars["pareto_k_mean"],
                sumParetoKMax
                + window_weight * scalars["pareto_k_max"],
                sumParetoKWarningFraction
                + window_weight * scalars["pareto_k_warning_fraction"],
                sumParetoKBadFraction
                + window_weight * scalars["pareto_k_bad_fraction"],
                sumResidualGmmObjective
                + window_weight
                * scalars["loss_residual_gmm"],
                sumResidualGmmTimeScore
                + window_weight * residual_gmm_time_score,
                _update_time_logsumexp(
                    logResidualGmmTimeRisk,
                    window_weight,
                    residual_gmm_time_score,
                    residual_gmm_time_beta_value,
                ),
                sumResidualGmmRaw
                + window_weight
                * scalars["loss_residual_gmm_raw"],
                sumResidualGmmTerms
                + window_weight
                * scalars["loss_residual_gmm_terms"],
                sumResidualGmmSiteTerms
                + window_weight
                * scalars["loss_residual_gmm_site_terms"],
                sumResidualGmmZMean
                + window_weight
                * scalars["residual_gmm_z_mean"],
                sumResidualGmmZMax
                + window_weight
                * scalars["residual_gmm_z_max"],
                sumResidualGmmRadiusMean
                + window_weight
                * scalars["residual_gmm_radius_mean"],
                sumResidualGmmRadiusMax
                + window_weight
                * scalars["residual_gmm_radius_max"],
                sumResidualGmmWarningFraction
                + window_weight
                * scalars["residual_gmm_warning_fraction"],
                sumResidualGmmBadFraction
                + window_weight
                * scalars["residual_gmm_bad_fraction"],
                sumGauge + window_weight * scalars["loss_gauge"],
                sumGaugeDrift
                + window_weight * scalars["loss_gauge_drift"],
                sumGaugeDiffusion
                + window_weight * scalars["loss_gauge_diffusion"],
                sumEssHinge + window_weight * ess_hinge,
                sumLogWeightSpread
                + window_weight * scalars["window_log_weight_spread"],
                jnp.minimum(minEssRatio, scalars["ess_ratio_end"]),
            ),
            (
                scalars["pareto_k_max"],
                scalars["residual_gmm_z_max"],
                scalars["residual_gmm_radius_max"],
                scalars[
                    "residual_gmm_covariance_estimate"
                ],
                scalars["window_log_weight_spread"],
                scalars["ess_ratio_end"],
            ),
        )

    body = jax.checkpoint(body)
    init = (
        key,
        lnOmega0,
        alpha0,
        beta0,
        jnp.asarray(t0, dtype=DTYPE),
        lnOmega_shift0,
        zero,
        zero_pareto_k_terms,
        zero_pareto_k_site_terms,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        neg_inf,
        zero,
        zero_residual_gmm_terms,
        zero_residual_gmm_site_terms,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray(1.0, dtype=DTYPE),
    )
    (
        key_out,
        lnOmega_end,
        alpha_end,
        beta_end,
        t_end,
        _lnOmega_shift_end,
        loss_pareto_k_objective_mean,
        loss_pareto_k_terms_mean,
        loss_pareto_k_site_terms_mean,
        pareto_k_mean,
        pareto_k_max,
        pareto_k_warning_fraction,
        pareto_k_bad_fraction,
        loss_residual_gmm_mean,
        loss_residual_gmm_time_score_sum,
        loss_residual_gmm_time_logsumexp,
        loss_residual_gmm_raw_mean,
        loss_residual_gmm_terms_mean,
        loss_residual_gmm_site_terms_mean,
        residual_gmm_z_mean,
        residual_gmm_z_max,
        residual_gmm_radius_mean,
        residual_gmm_radius_max,
        residual_gmm_warning_fraction,
        residual_gmm_bad_fraction,
        loss_gauge_mean,
        loss_gauge_drift_mean,
        loss_gauge_diffusion_mean,
        loss_ess_hinge_mean,
        log_weight_spread_mean,
        ess_ratio_min,
    ), worst_series = lax.scan(body, init, jnp.arange(N_windows))
    (
        pareto_k_worst_series,
        residual_gmm_z_worst_series,
        residual_gmm_radius_worst_series,
        residual_gmm_covariance_estimates,
        window_log_weight_spread_series,
        ess_ratio_series,
    ) = worst_series

    pareto_k_worst = (
        jnp.max(pareto_k_worst_series) if enable_loss_pareto_k else zero
    )
    residual_gmm_z_worst = (
        jnp.max(residual_gmm_z_worst_series)
        if enable_loss_residual_gmm
        else zero
    )
    residual_gmm_radius_worst = (
        jnp.max(residual_gmm_radius_worst_series)
        if enable_loss_residual_gmm
        else zero
    )
    loss_residual_gmm_time = (
        _finalize_time_aggregation(
            loss_residual_gmm_mean,
            loss_residual_gmm_time_score_sum,
            loss_residual_gmm_time_logsumexp,
            residual_gmm_time_weight_sum,
            residual_gmm_time_beta_value,
            residual_gmm_time_aggregation,
        )
        if enable_loss_residual_gmm
        else zero
    )

    loss_pareto_k = (
        loss_pareto_k_prefactor * loss_pareto_k_objective_mean
        if enable_loss_pareto_k
        else zero
    )
    loss_residual_gmm = (
        loss_residual_gmm_prefactor
        * loss_residual_gmm_time
        if enable_loss_residual_gmm
        else zero
    )
    loss_gauge = (
        loss_gauge_prefactor * loss_gauge_mean if enable_loss_gauge else zero
    )
    loss_L2 = (
        loss_L2_prefactor * compute_l2_penalty(params)
        if enable_loss_L2
        else zero
    )
    loss_ess = (
        loss_ess_prefactor * loss_ess_hinge_mean
        if enable_loss_ess
        else zero
    )
    loss = (
        loss_pareto_k
        + loss_residual_gmm
        + loss_gauge
        + loss_L2
        + loss_ess
    )

    aux = {
        "key_out": key_out,
        "lnOmega_end": lnOmega_end,
        "alpha_end": alpha_end,
        "beta_end": beta_end,
        "loss": loss,
        "loss_pareto_k": loss_pareto_k_objective_mean,
        "loss_pareto_k_objective": loss_pareto_k_objective_mean,
        "loss_pareto_k_terms": loss_pareto_k_terms_mean,
        "loss_pareto_k_site_terms": loss_pareto_k_site_terms_mean,
        **{
            term: loss_pareto_k_terms_mean[index]
            for index, term in enumerate(PARETO_K_MONOMIAL_TERMS)
        },
        "pareto_k_mean": pareto_k_mean,
        "pareto_k_max": pareto_k_max,
        "pareto_k_worst": pareto_k_worst,
        "pareto_k_warning_fraction": pareto_k_warning_fraction,
        "pareto_k_bad_fraction": pareto_k_bad_fraction,
        "loss_residual_gmm": loss_residual_gmm_mean,
        "loss_residual_gmm_time": loss_residual_gmm_time,
        "loss_residual_gmm_raw": loss_residual_gmm_raw_mean,
        "loss_residual_gmm_terms": (
            loss_residual_gmm_terms_mean
        ),
        "loss_residual_gmm_site_terms": (
            loss_residual_gmm_site_terms_mean
        ),
        "residual_gmm_z_mean": residual_gmm_z_mean,
        "residual_gmm_z_max": residual_gmm_z_max,
        "residual_gmm_z_worst": residual_gmm_z_worst,
        "residual_gmm_radius_mean": residual_gmm_radius_mean,
        "residual_gmm_radius_max": residual_gmm_radius_max,
        "residual_gmm_radius_worst": residual_gmm_radius_worst,
        "residual_gmm_warning_fraction": (
            residual_gmm_warning_fraction
        ),
        "residual_gmm_bad_fraction": residual_gmm_bad_fraction,
        "residual_gmm_covariance_estimates": (
            residual_gmm_covariance_estimates
        ),
        "loss_gauge": loss_gauge_mean,
        "loss_gauge_drift": loss_gauge_drift_mean,
        "loss_gauge_diffusion": loss_gauge_diffusion_mean,
        "loss_ess": loss_ess_hinge_mean,
        "loss_ess_weighted": loss_ess,
        "log_weight_spread_mean": log_weight_spread_mean,
        "log_weight_spread_max": (
            jnp.max(window_log_weight_spread_series)
            if enable_loss_ess
            else zero
        ),
        "log_weight_spread_total": (
            jnp.sum(window_log_weight_spread_series)
            if enable_loss_ess
            else zero
        ),
        "ess_ratio_min": (
            ess_ratio_min
            if enable_loss_ess
            else jnp.asarray(1.0, dtype=DTYPE)
        ),
        "ess_ratio_end": (
            ess_ratio_series[-1]
            if enable_loss_ess
            else jnp.asarray(1.0, dtype=DTYPE)
        ),
        "loss_L2": loss_L2,
        "t_end": t_end,
    }
    return loss, aux


@partial(
    jax.jit,
    static_argnames=(
        "apply_fn",
        "N_steps",
        "N_windows",
        "apply_neural_gauge_every_steps",
        "sde_max_iter",
        "sde_solver",
        "sde_affine_expm_order",
        "sde_affine_expm_substeps",
        "sde_newton_damping_steps",
        "gauge_mode",
        "neural_gauge_components",
        "pareto_k_applied_quantities",
        "pareto_k_applied_quantities_mode",
        "pareto_k_monomials",
        "operator_monomials",
        "residual_gmm_time_aggregation",
        "residual_gmm_integrator_nodes",
        "pareto_k_envelope_excess",
        "pareto_k_tail_fraction",
        "pareto_k_min_tail_count",
        "enable_loss_pareto_k",
        "enable_loss_residual_gmm",
        "enable_loss_gauge",
        "enable_loss_L2",
        "enable_loss_ess",
        "center_axis_name",
    ),
)
def compute_grads_all_windows(
    key: jax.Array,
    apply_fn,
    params,
    lnOmega0: jnp.ndarray,
    alpha0: jnp.ndarray,
    beta0: jnp.ndarray,
    U: DTYPE,
    gamma: DTYPE,
    F: CDTYPE,
    Delta: DTYPE,
    dt: DTYPE,
    N_steps: int,
    N_windows: int,
    window_loss_weights: jnp.ndarray,
    lnOmega_shift0: DTYPE = 0.0,
    operator_monomials: tuple = (),
    residual_gmm_covariance_bank: jnp.ndarray | None = None,
    residual_gmm_covariance_initialized: jnp.ndarray | None = None,
    apply_neural_gauge_every_steps: int = 0,
    neural_gauge_each_apply: bool = False,
    sde_max_iter: int = 4,
    sde_solver: str = "semi_implicit_midpoint",
    sde_root_rtol: DTYPE = SDE_ROOT_RTOL_DEFAULT,
    sde_root_atol: DTYPE = SDE_ROOT_ATOL_DEFAULT,
    sde_affine_expm_order: int = 6,
    sde_affine_expm_substeps: int = 1,
    sde_newton_damping_steps: int = 4,
    t0: DTYPE = 0.0,
    gauge_weight: DTYPE = 1.0,
    pareto_k_threshold: DTYPE = 0.7,
    pareto_k_threshold_tau: DTYPE = 0.1,
    pareto_k_envelope_beta: DTYPE = 0.5,
    pareto_k_envelope_excess: str = "log",
    residual_gmm_d_clip: DTYPE = 10.0,
    residual_gmm_cov_floor: DTYPE = 1.0e-8,
    residual_gmm_cov_shrinkage: DTYPE = 0.05,
    residual_gmm_time_aggregation: str = "mean",
    residual_gmm_integrator_nodes: int = 6,
    residual_gmm_time_beta: DTYPE = 0.0,
    pareto_k_tail_fraction: float = 0.01,
    pareto_k_min_tail_count: int = 32,
    n0: DTYPE = 1.0,
    loss_pareto_k_prefactor: DTYPE = 0.0,
    loss_pareto_k_monomial_weights: jnp.ndarray = 1.0,
    loss_residual_gmm_prefactor: DTYPE = 0.0,
    loss_L2_prefactor: DTYPE = 1e-4,
    loss_gauge_prefactor: DTYPE = 1e-2,
    loss_ess_prefactor: DTYPE = 0.0,
    loss_ess_window_budget: DTYPE = 0.0,
    q_winsor: float = 0.95,
    pareto_k_applied_quantities: int = 6,
    pareto_k_applied_quantities_mode: str = "upto",
    pareto_k_monomials: tuple = (),
    J=0.0,
    gauge_mode: str = "neural_graph",
    neural_gauge_components: str = "both",
    analytic_target_time: DTYPE = 0.0,
    enable_loss_pareto_k: bool = False,
    enable_loss_residual_gmm: bool = False,
    enable_loss_gauge: bool = True,
    enable_loss_L2: bool = True,
    enable_loss_ess: bool = False,
    center_axis_name=None,
):
    def loss_for_params(params_arg):
        return _loss_and_aux_all_windows(
            key=key,
            apply_fn=apply_fn,
            params=params_arg,
            lnOmega0=lnOmega0,
            alpha0=alpha0,
            beta0=beta0,
            U=U,
            gamma=gamma,
            F=F,
            Delta=Delta,
            dt=dt,
            N_steps=N_steps,
            N_windows=N_windows,
            window_loss_weights=window_loss_weights,
            lnOmega_shift0=lnOmega_shift0,
            operator_monomials=operator_monomials,
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
            sde_root_rtol=sde_root_rtol,
            sde_root_atol=sde_root_atol,
            sde_affine_expm_order=sde_affine_expm_order,
            sde_affine_expm_substeps=sde_affine_expm_substeps,
            sde_newton_damping_steps=sde_newton_damping_steps,
            t0=t0,
            gauge_weight=gauge_weight,
            pareto_k_threshold=pareto_k_threshold,
            pareto_k_threshold_tau=pareto_k_threshold_tau,
            pareto_k_envelope_beta=pareto_k_envelope_beta,
            pareto_k_envelope_excess=pareto_k_envelope_excess,
            residual_gmm_d_clip=(
                residual_gmm_d_clip
            ),
            residual_gmm_cov_floor=(
                residual_gmm_cov_floor
            ),
            residual_gmm_cov_shrinkage=(
                residual_gmm_cov_shrinkage
            ),
            residual_gmm_time_aggregation=(
                residual_gmm_time_aggregation
            ),
            residual_gmm_integrator_nodes=(
                residual_gmm_integrator_nodes
            ),
            residual_gmm_time_beta=(
                residual_gmm_time_beta
            ),
            pareto_k_tail_fraction=pareto_k_tail_fraction,
            pareto_k_min_tail_count=pareto_k_min_tail_count,
            n0=n0,
            loss_pareto_k_prefactor=loss_pareto_k_prefactor,
            loss_pareto_k_monomial_weights=(
                loss_pareto_k_monomial_weights
            ),
            loss_residual_gmm_prefactor=(
                loss_residual_gmm_prefactor
            ),
            loss_L2_prefactor=loss_L2_prefactor,
            loss_gauge_prefactor=loss_gauge_prefactor,
            loss_ess_prefactor=loss_ess_prefactor,
            loss_ess_window_budget=loss_ess_window_budget,
            q_winsor=q_winsor,
            pareto_k_applied_quantities=pareto_k_applied_quantities,
            pareto_k_applied_quantities_mode=(
                pareto_k_applied_quantities_mode
            ),
            pareto_k_monomials=pareto_k_monomials,
            J=J,
            gauge_mode=gauge_mode,
            neural_gauge_components=neural_gauge_components,
            analytic_target_time=analytic_target_time,
            enable_loss_pareto_k=enable_loss_pareto_k,
            enable_loss_residual_gmm=(
                enable_loss_residual_gmm
            ),
            enable_loss_gauge=enable_loss_gauge,
            enable_loss_L2=enable_loss_L2,
            enable_loss_ess=enable_loss_ess,
            center_axis_name=center_axis_name,
        )

    (loss, aux), grads = jax.value_and_grad(
        loss_for_params, has_aux=True
    )(params)
    if center_axis_name is not None:
        grads = jax.tree_util.tree_map(
            lambda value: lax.pmean(value, center_axis_name), grads
        )
    aux["grads_norm"] = opx.global_norm(grads)
    return grads, aux, loss


def run_simulation_windows(
    *,
    key: jax.Array,
    apply_fn,
    params,
    lnOmega0: jnp.ndarray,
    alpha0: jnp.ndarray,
    beta0: jnp.ndarray,
    U: DTYPE,
    gamma: DTYPE,
    F: CDTYPE,
    Delta: DTYPE,
    dt: DTYPE,
    N_steps: int,
    N_windows: int,
    t0: DTYPE,
    gauge_weight: DTYPE,
    n0: DTYPE,
    sde_max_iter: int,
    sde_solver: str = "semi_implicit_midpoint",
    sde_root_rtol: DTYPE = SDE_ROOT_RTOL_DEFAULT,
    sde_root_atol: DTYPE = SDE_ROOT_ATOL_DEFAULT,
    sde_affine_expm_order: int = 6,
    sde_affine_expm_substeps: int = 1,
    sde_newton_damping_steps: int = 4,
    J,
    gauge_mode: str,
    neural_gauge_components: str = "both",
    analytic_target_time: DTYPE = 0.0,
    apply_neural_gauge_every_steps: int = 0,
    progress_every_window: int = 1,
):
    num_times = int(N_windows) + 1
    lnOmega_history = np.empty((num_times,) + tuple(lnOmega0.shape), dtype=np.asarray(lnOmega0).dtype)
    alpha_history = np.empty((num_times,) + tuple(alpha0.shape), dtype=np.asarray(alpha0).dtype)
    beta_history = np.empty((num_times,) + tuple(beta0.shape), dtype=np.asarray(beta0).dtype)
    lnOmega_shift_history = np.empty((num_times,), dtype=float)
    times = np.empty((num_times,), dtype=float)
    lnOmega_history[0] = np.asarray(lnOmega0)
    alpha_history[0] = np.asarray(alpha0)
    beta_history[0] = np.asarray(beta0)
    lnOmega_shift_history[0] = 0.0
    times[0] = float(t0)

    lnOmega = lnOmega0
    alpha = alpha0
    beta = beta0
    t = t0
    lnOmega_shift = 0.0
    key_out = key
    progress_every_window = int(progress_every_window)

    for iwin in range(int(N_windows)):
        sub = random.fold_in(key_out, iwin)
        key_out, lnOmega, alpha, beta, aux = run_one_window_simulation_rollout(
            key=sub,
            apply_fn=apply_fn,
            params=params,
            lnOmega=lnOmega,
            alpha=alpha,
            beta=beta,
            U=U,
            gamma=gamma,
            F=F,
            Delta=Delta,
            dt=dt,
            N_steps=N_steps,
            t0=t,
            apply_neural_gauge_every_steps=apply_neural_gauge_every_steps,
            gauge_weight=gauge_weight,
            n0=n0,
            sde_max_iter=sde_max_iter,
            sde_solver=sde_solver,
            sde_root_rtol=sde_root_rtol,
            sde_root_atol=sde_root_atol,
            sde_affine_expm_order=sde_affine_expm_order,
            sde_affine_expm_substeps=sde_affine_expm_substeps,
            sde_newton_damping_steps=sde_newton_damping_steps,
            J=J,
            gauge_mode=gauge_mode,
            neural_gauge_components=neural_gauge_components,
            analytic_target_time=analytic_target_time,
        )
        t = float(aux["t_end"])
        lnOmega_shift += float(np.asarray(aux.get("lnOmega_center_shift", 0.0)))
        times[iwin + 1] = t
        lnOmega_history[iwin + 1] = np.asarray(lnOmega)
        alpha_history[iwin + 1] = np.asarray(alpha)
        beta_history[iwin + 1] = np.asarray(beta)
        lnOmega_shift_history[iwin + 1] = lnOmega_shift
        if progress_every_window > 0 and (
            iwin == 0
            or (iwin + 1) % progress_every_window == 0
            or (iwin + 1) == int(N_windows)
        ):
            print(
                (
                    f"simulation window {iwin + 1}/{int(N_windows)} complete "
                    f"| t = {t:.8g}"
                ),
                flush=True,
            )

    return {
        "times": times,
        "lnOmega_history": lnOmega_history,
        "lnOmega_shift_history": lnOmega_shift_history,
        "alpha_history": alpha_history,
        "beta_history": beta_history,
    }


__all__ = [
    "compute_grads_all_windows",
    "compute_onsite_pareto_k_diagnostics",
    "compute_onsite_exact_moment_clouds",
    "compute_observable_pareto_k_indices",
    "initialize_phase_space_variables",
    "run_one_window_simulation_rollout",
    "run_one_window_training_profile",
    "run_simulation_windows",
    "self_normalized_weight_ratio",
    "var_complex_to_real",
]
