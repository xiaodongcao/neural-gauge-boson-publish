"""
Analytical gauge definitions used by the active simulation runtime.

Only the gauge family used in

S. Wuster, J. F. Corney, J. M. Rost, and P. Deuar,
"Quantum dynamics of long-range interacting systems using the positive-P and gauge-P representations",
Phys. Rev. E 96, 013309 (2017),

is kept here.

The current solver supports only onsite Bose-Hubbard / Kerr interactions with a
scalar U, not the fully nonlocal interaction matrix W of the paper. Therefore
the implemented analytical gauge is the contact-limit specialization of the
paper's adaptive global gauge family:

- standard drift gauge, Eq. (19)
- global diffusion gauge, Eq. (24)
- adaptive target-time update, Eq. (32)

For the onsite model, each walker carries one scalar adaptive diffusion gauge
parameter a_w(t), broadcast across all sites of that walker. The corresponding
two-component drift gauge is then obtained from the single-mode contact result
of Deuar and Drummond, Eq. (47).
"""

from dataclasses import dataclass

from .lib_preinclude import *


@dataclass(frozen=True)
class AnalyticalGaugeInfo:
    mode: str
    source: str
    location: str
    summary: str
    best_for: str
    caveats: str


def _local_density(alpha: jnp.ndarray, beta: jnp.ndarray):
    return alpha * beta


def _zero_gauge(alpha: jnp.ndarray):
    num_walker, num_site = alpha.shape
    zeros_c = jnp.zeros((num_walker, num_site), dtype=CDTYPE)
    zeros_r = jnp.zeros((num_walker, num_site), dtype=DTYPE)
    return zeros_c, zeros_c, zeros_r


def _adaptive_global_diffusion(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    time,
    target_time,
    U,
):
    """
    Adaptive global diffusion gauge specialized to the onsite/contact model.

    We use the adaptive target-time rule of Wuster et al., Eq. (32), together
    with the global-gauge scalars from Eq. (26):

        I1 = sum_{k,k'} U_{kk'} [n''_k n''_{k'}]
        I2 = sum_{k,k'} U_{kk'}^2 [Re[n_k n_{k'}^*]]


    In the active solver we therefore apply the adaptive gauge trajectory by
    trajectory (walker by walker), not from a walker-mean estimate. For the
    onsite/contact specialization with U_{kk'} = U0 delta_{kk'} and U0 = |U|,
    this gives, for each walker w:

        a_w(t) = (1/6) log[
            4 I2[n_w(t)] (t_fin - t) / U0
            + (1 + 4 I1[n_w(t)] / U0)^(3/2)
        ]

    where, for each walker w,

        I1[n_w] = U0 * sum_i Im[n_{w,i}]^2
        I2[n_w] = U0^2 * sum_i |n_{w,i}|^2

    The resulting scalar a_w is broadcast across all sites.
    """

    num_walker, num_site = alpha.shape
    t_now = jnp.asarray(time, dtype=DTYPE)
    t_fin = jnp.asarray(target_time, dtype=DTYPE)
    t_rem = jnp.maximum(t_fin - t_now, jnp.asarray(0.0, dtype=DTYPE))

    U_abs = jnp.abs(jnp.asarray(U, dtype=DTYPE))
    eps = jnp.asarray(1e-12, dtype=DTYPE)
    U0 = jnp.maximum(U_abs, eps)

    n = _local_density(alpha, beta)
    n_imag_sq = jnp.square(jnp.imag(n).astype(DTYPE))
    n_abs_sq = (jnp.real(n * jnp.conj(n))).astype(DTYPE)

    I1 = U0 * jnp.sum(n_imag_sq, axis=-1)
    I2 = (U0**2) * jnp.sum(n_abs_sq, axis=-1)

    diffusion_inside = (
        4.0 * I2 * t_rem / U0
        + jnp.power(1.0 + 4.0 * I1 / U0, 1.5)
    )
    a_adaptive = (jnp.asarray(1.0 / 6.0, dtype=DTYPE) * jnp.log(jnp.maximum(diffusion_inside, eps)))
    a_adaptive = jnp.where(U_abs > eps, a_adaptive, jnp.zeros_like(a_adaptive))
    return jnp.broadcast_to(a_adaptive[:, None], (num_walker, num_site))


def _wuster_standard_drift(alpha: jnp.ndarray, beta: jnp.ndarray, U, diffusion_g: jnp.ndarray):
    """
    Wuster et al. (2017), Eq. (19), combined with the contact-interaction
    single-mode drift-gauge relation of Deuar and Drummond (2006), Eq. (47),
    converted to the transformed-noise convention used by the current solver.

    Paper notation:
        f_lambda = i Im[n_lambda]

    with n_lambda = alpha_lambda beta_lambda.  In the current solver variables
    this gives the transformed gauge fields

        g = -sqrt(i U) exp(-a) Im[n]
        f = -i g

    where a is the global diffusion parameter from Eq. (24).
    """

    local_density_imag = jnp.imag(_local_density(alpha, beta)).astype(DTYPE)
    sqrt_ik = jnp.sqrt(cplx_i() * jnp.asarray(U, dtype=CDTYPE))
    drift_g = -sqrt_ik * jnp.exp(-jnp.asarray(diffusion_g, dtype=DTYPE)) * local_density_imag
    drift_f = -cplx_i() * drift_g
    return drift_g, drift_f


ANALYTICAL_GAUGE_INFOS = {
    "wuster_adaptive": AnalyticalGaugeInfo(
        mode="wuster_adaptive",
        source=(
            "Wuster et al., Phys. Rev. E 96, 013309 (2017), using the standard "
            "drift gauge with the adaptive global diffusion gauge; drift-field "
            "mapping checked against Deuar and Drummond, J. Phys. A 39, 2723 "
            "(2006)."
        ),
        location="Wuster Eq. (19), p. 5, Eq. (24), p. 6, Eq. (32), p. 7; Deuar-Drummond Eq. (47), p. 15.",
        summary=(
            "Standard drift gauge together with one adaptive global diffusion "
            "parameter a_w(t) per walker, broadcast across all sites of that walker."
        ),
        best_for=(
            "Single-site Kerr tests and onsite lattices when an adaptive "
            "paper-guided global gauge benchmark is wanted."
        ),
        caveats=(
            "The current solver uses the onsite/contact specialization with U0 "
            "= |U| and per-walker global a_w(t). It does not implement the full "
            "nonlocal interaction-matrix version of the paper."
        ),
    ),
    "zero_gauge": AnalyticalGaugeInfo(
        mode="zero_gauge",
        source="Positive-P baseline obtained by setting both drift and diffusion gauges to zero.",
        location="No stochastic gauge: g = 0, f = 0, g'' = 0.",
        summary=(
            "Explicit ungauged positive-P / gauge-P baseline with vanishing "
            "drift and diffusion gauges."
        ),
        best_for=(
            "Reference baseline runs and direct comparison against gauged "
            "simulation strategies."
        ),
        caveats=(
            "This is the ungauged baseline. It is not intended to stabilize "
            "large-occupation or long-time simulations."
        ),
    ),
}


ANALYTICAL_GAUGE_MODES = tuple(ANALYTICAL_GAUGE_INFOS.keys())


def describe_analytical_gauge(mode: str):
    try:
        return ANALYTICAL_GAUGE_INFOS[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown analytical gauge mode '{mode}'") from exc


def validate_analytical_gauge_mode(mode: str):
    return describe_analytical_gauge(mode)


def compute_analytical_gauge_fields(
    *,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    time,
    target_time,
    U,
    gauge_mode: str,
):
    if gauge_mode == "wuster_adaptive":
        diffusion_g = _adaptive_global_diffusion(
            alpha=alpha,
            beta=beta,
            time=time,
            target_time=target_time,
            U=U,
        )
        drift_g, drift_f = _wuster_standard_drift(alpha, beta, U, diffusion_g)
        return drift_g, drift_f, diffusion_g
    if gauge_mode == "zero_gauge":
        return _zero_gauge(alpha)

    raise ValueError(f"Unsupported analytical gauge mode '{gauge_mode}'")


__all__ = [
    "ANALYTICAL_GAUGE_INFOS",
    "ANALYTICAL_GAUGE_MODES",
    "AnalyticalGaugeInfo",
    "compute_analytical_gauge_fields",
    "describe_analytical_gauge",
    "validate_analytical_gauge_mode",
]
