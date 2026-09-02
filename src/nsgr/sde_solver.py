from functools import partial

from .lattice import apply_hopping_matrix, broadcast_site_param, conjugate_hopping_operator
from .lib_preinclude import *


@flax.struct.dataclass
class SolverState:
    """Phase-space state advanced by one stochastic-gauge solver step."""

    ln_omega: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray


@flax.struct.dataclass
class GaugeFields:
    """Gauge fields frozen over one solver step."""

    drift_g: jnp.ndarray
    drift_f: jnp.ndarray
    diffusion_g: jnp.ndarray


@flax.struct.dataclass
class SolverNoise:
    """Pre-sampled Ito Wiener increments for one solver step."""

    dW: jnp.ndarray
    dWp: jnp.ndarray


@flax.struct.dataclass
class SolverStepDiagnostics:
    """Local completion diagnostics for one solver step.

    Every field has shape ``(num_walker, num_site)``.  For an implicit solver,
    ``converged`` is true only when the final nonlinear residual satisfies the
    configured mixed absolute/relative tolerance.  An exact/split backend has
    no root residual and reports zero residual/iterations when all subflows are
    finite.  The standard ``step`` interface fails closed: a walker containing
    any unsuccessful site is returned as non-finite.
    """

    converged: jnp.ndarray
    residual_norm: jnp.ndarray
    newton_iterations: jnp.ndarray
    linear_solve_failed: jnp.ndarray
    finite: jnp.ndarray


@flax.struct.dataclass
class SolverCoefficients:
    """Physical coefficients for one lattice Bose-Hubbard / Kerr step."""

    dt: Any
    U: Any
    gamma: Any
    F: Any
    Delta: Any
    J: Any = 0.0
    center_axis_name: Any = flax.struct.field(pytree_node=False, default=None)
    root_rtol: Any = SDE_ROOT_RTOL_DEFAULT
    root_atol: Any = SDE_ROOT_ATOL_DEFAULT
    newton_damping_steps: Any = flax.struct.field(pytree_node=False, default=4)
    affine_expm_order: Any = flax.struct.field(pytree_node=False, default=6)
    affine_expm_substeps: Any = flax.struct.field(pytree_node=False, default=1)


class GaugeSDESolver:
    """Base interface for single-step stochastic-gauge lattice solvers."""

    solver_name = "base"

    @classmethod
    def step(
        cls,
        state: SolverState,
        gauge_fields: GaugeFields,
        noise: SolverNoise,
        coefficients: SolverCoefficients,
        sde_max_iter: int = 4,
    ) -> SolverState:
        raise NotImplementedError(f"{cls.__name__}.step must be implemented by subclasses.")


class SemiImplicitMidpointSolver(GaugeSDESolver):
    """
    Semi-implicit midpoint solver for the lattice positive-P equations.

    For each site i, the solver advances the site-resolved phase-space variables
    alpha_i and beta_i with:
    - onsite Kerr / Bose-Hubbard nonlinearity
    - linear loss gamma
    - coherent drive F
    - detuning Delta
    - lattice hopping through the dense matrix J
    - frozen gauge fields (g, f, g'')

    The stochastic increments are Ito, evaluated at the old state, while the
    drift is re-evaluated at the midpoint through Picard iterations.
    """

    solver_name = "semi_implicit_midpoint"

    @staticmethod
    def _noise_increments(alpha, beta, ch, sh, U, dW, dWp):
        i = cplx_i()
        sqrt_ik = jnp.sqrt(i * jnp.asarray(U, dtype=CDTYPE))
        Xi_alpha = (i * sqrt_ik) * alpha * (ch * dW + i * sh * dWp)
        Xi_beta = sqrt_ik * beta * (-i * sh * dW + ch * dWp)
        return Xi_alpha, Xi_beta

    @classmethod
    def step(
        cls,
        state: SolverState,
        gauge_fields: GaugeFields,
        noise: SolverNoise,
        coefficients: SolverCoefficients,
        sde_max_iter: int = 4,
    ) -> SolverState:
        return _semi_implicit_midpoint_step_impl(
            state=state,
            gauge_fields=gauge_fields,
            noise=noise,
            coefficients=coefficients,
            sde_max_iter=sde_max_iter,
        )


@partial(jax.jit, static_argnums=(4,))
def _semi_implicit_midpoint_step_impl(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
    sde_max_iter: int,
) -> SolverState:
    """
    Jitted implementation of the active lattice midpoint solver.

    The physical drift contains onsite Kerr evolution plus dense-lattice hopping:
      d alpha_i / dt = alpha_i (-i U alpha_i beta_i - gamma/2 + i Delta)
                       - i F_i + i sum_j J_ij alpha_j
      d beta_i  / dt = beta_i  ( i U alpha_i beta_i - gamma/2 - i Delta)
                       + i F_i* - i sum_j J_ij* beta_j

    Gauge drift and weight updates use the solver-ready fields returned by the
    neural network or the analytical benchmark gauge.
    """

    i = cplx_i()
    alpha = state.alpha
    beta = state.beta
    ln_omega = state.ln_omega
    _, num_site = alpha.shape

    dt = jnp.asarray(coefficients.dt, dtype=DTYPE)
    U = jnp.asarray(coefficients.U, dtype=DTYPE)
    gamma = jnp.asarray(coefficients.gamma, dtype=DTYPE)
    Delta = jnp.asarray(coefficients.Delta, dtype=DTYPE)
    F = jnp.asarray(coefficients.F, dtype=CDTYPE)

    drift_g = jnp.asarray(gauge_fields.drift_g, dtype=DTYPE)
    drift_f = jnp.asarray(gauge_fields.drift_f, dtype=DTYPE)
    diffusion_g = jnp.asarray(gauge_fields.diffusion_g, dtype=DTYPE)
    dW = jnp.asarray(noise.dW, dtype=DTYPE)
    dWp = jnp.asarray(noise.dWp, dtype=DTYPE)

    g = to_c(drift_g[..., :num_site], drift_g[..., num_site:])
    f = to_c(drift_f[..., :num_site], drift_f[..., num_site:])

    ch = jnp.cosh(diffusion_g)
    sh = jnp.sinh(diffusion_g)

    sqrt_ik = jnp.sqrt(i * jnp.asarray(U, dtype=CDTYPE))
    U_site = broadcast_site_param(U, num_site, DTYPE)
    gamma_site = broadcast_site_param(gamma, num_site, DTYPE)
    Delta_site = broadcast_site_param(Delta, num_site, DTYPE)
    F_site = jnp.broadcast_to(F, (num_site,))

    Xi_alpha, Xi_beta = SemiImplicitMidpointSolver._noise_increments(
        alpha=alpha,
        beta=beta,
        ch=ch,
        sh=sh,
        U=U,
        dW=dW,
        dWp=dWp,
    )

    def drift_phys(a, b):
        hop_alpha = apply_hopping_matrix(coefficients.J, a)
        hop_beta = apply_hopping_matrix(conjugate_hopping_operator(coefficients.J), b)
        drift_alpha = (
            a * (-i * U_site * a * b - 0.5 * gamma_site + i * Delta_site)
            - i * F_site
            + i * hop_alpha
        )
        drift_beta = (
            b * (i * U_site * a * b - 0.5 * gamma_site - i * Delta_site)
            + i * jnp.conj(F_site)
            - i * hop_beta
        )
        return drift_alpha, drift_beta

    def drift_gauge(a, b):
        drift_alpha = a * (-i * sqrt_ik * (g * ch + i * f * sh))
        drift_beta = b * (-sqrt_ik * (-i * g * sh + f * ch))
        return drift_alpha, drift_beta

    alpha_base = alpha + Xi_alpha
    beta_base = beta + Xi_beta

    drift_alpha_phys, drift_beta_phys = drift_phys(alpha, beta)
    drift_alpha_gauge, drift_beta_gauge = drift_gauge(alpha, beta)
    alpha_next = alpha_base + dt * (drift_alpha_phys + drift_alpha_gauge)
    beta_next = beta_base + dt * (drift_beta_phys + drift_beta_gauge)

    for _ in range(sde_max_iter):
        alpha_mid = 0.5 * (alpha + alpha_next)
        beta_mid = 0.5 * (beta + beta_next)

        drift_alpha_phys_mid, drift_beta_phys_mid = drift_phys(alpha_mid, beta_mid)
        drift_alpha_gauge_mid, drift_beta_gauge_mid = drift_gauge(alpha_mid, beta_mid)

        alpha_next = alpha_base + dt * (drift_alpha_phys_mid + drift_alpha_gauge_mid)
        beta_next = beta_base + dt * (drift_beta_phys_mid + drift_beta_gauge_mid)

    dln_omega = jnp.sum(g * dW + f * dWp - 0.5 * (g**2 + f**2) * dt, axis=-1)
    # Remove the common real-weight drift each step to keep the stored log-weights
    # numerically centered. This only rescales every walker by the same positive
    # factor, so self-normalized weighted observables remain unchanged.
    local_center = jnp.mean(jnp.real(dln_omega), axis=0)
    if coefficients.center_axis_name is not None:
        local_center = lax.pmean(local_center, coefficients.center_axis_name)
    dln_omega = dln_omega - lax.stop_gradient(local_center)

    return SolverState(
        ln_omega=ln_omega + dln_omega,
        alpha=alpha_next,
        beta=beta_next,
    )


def _complex_finite(value):
    return jnp.isfinite(jnp.real(value)) & jnp.isfinite(jnp.imag(value))


def _nonlinear_midpoint_residual(
    alpha_next,
    beta_next,
    alpha_base,
    beta_base,
    noise_alpha,
    noise_beta,
    U_site,
    gauge_alpha,
    gauge_beta,
    dt,
):
    """Residual of the local Kerr-plus-frozen-gauge midpoint equation."""

    i = cplx_i()
    alpha_mid = 0.5 * (alpha_base + alpha_next)
    beta_mid = 0.5 * (beta_base + beta_next)
    drift_alpha = -i * U_site * alpha_mid**2 * beta_mid + gauge_alpha * alpha_mid
    drift_beta = i * U_site * alpha_mid * beta_mid**2 + gauge_beta * beta_mid
    residual_alpha = alpha_next - alpha_base - noise_alpha - dt * drift_alpha
    residual_beta = beta_next - beta_base - noise_beta - dt * drift_beta
    return residual_alpha, residual_beta


def _nonlinear_midpoint_jacobian(
    alpha_next,
    beta_next,
    alpha_base,
    beta_base,
    U_site,
    gauge_alpha,
    gauge_beta,
    dt,
):
    """Complex 2-by-2 Jacobian of the local holomorphic root equation."""

    i = cplx_i()
    alpha_mid = 0.5 * (alpha_base + alpha_next)
    beta_mid = 0.5 * (beta_base + beta_next)

    drift_aa = -2.0 * i * U_site * alpha_mid * beta_mid + gauge_alpha
    drift_ab = -i * U_site * alpha_mid**2
    drift_ba = i * U_site * beta_mid**2
    drift_bb = 2.0 * i * U_site * alpha_mid * beta_mid + gauge_beta

    jac_aa = 1.0 - 0.5 * dt * drift_aa
    jac_ab = -0.5 * dt * drift_ab
    jac_ba = -0.5 * dt * drift_ba
    jac_bb = 1.0 - 0.5 * dt * drift_bb
    return jac_aa, jac_ab, jac_ba, jac_bb


def _local_residual_norm(residual_alpha, residual_beta):
    return jnp.sqrt(jnp.abs(residual_alpha) ** 2 + jnp.abs(residual_beta) ** 2)


def _jacobian_determinant_is_bad(jac_aa, jac_ab, jac_ba, jac_bb):
    determinant = jac_aa * jac_bb - jac_ab * jac_ba
    real_dtype = jnp.real(determinant).dtype
    determinant_floor_factor = jnp.asarray(
        64.0 * jnp.finfo(real_dtype).eps,
        dtype=real_dtype,
    )
    determinant_scale = jnp.maximum(
        1.0,
        jnp.abs(jac_aa) * jnp.abs(jac_bb) + jnp.abs(jac_ab) * jnp.abs(jac_ba),
    )
    determinant_bad = (
        ~_complex_finite(determinant)
        | (jnp.abs(determinant) <= determinant_floor_factor * determinant_scale)
    )
    return determinant, determinant_bad


def _newton_tolerance(
    alpha_next,
    beta_next,
    alpha_base,
    beta_base,
    noise_alpha,
    noise_beta,
    root_rtol,
    root_atol,
):
    real_dtype = jnp.real(alpha_next).dtype
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    # A requested tolerance below roundoff cannot be a meaningful convergence
    # test, particularly in the optional float32 solver mode.
    rtol = jnp.maximum(jnp.asarray(root_rtol, dtype=real_dtype), 32.0 * eps)
    atol = jnp.maximum(jnp.asarray(root_atol, dtype=real_dtype), 32.0 * eps)
    reference_norm = jnp.sqrt(
        jnp.abs(alpha_base + noise_alpha) ** 2
        + jnp.abs(beta_base + noise_beta) ** 2
    )
    solution_norm = jnp.sqrt(jnp.abs(alpha_next) ** 2 + jnp.abs(beta_next) ** 2)
    scale = jnp.maximum(1.0, jnp.maximum(reference_norm, solution_norm))
    return atol + rtol * scale


def _damped_newton_midpoint(
    alpha_base,
    beta_base,
    noise_alpha,
    noise_beta,
    U_site,
    gauge_alpha,
    gauge_beta,
    dt,
    root_rtol,
    root_atol,
    max_iter,
    damping_steps,
):
    """Solve every walker/site root independently with static damped Newton."""

    i = cplx_i()
    drift_alpha_0 = -i * U_site * alpha_base**2 * beta_base + gauge_alpha * alpha_base
    drift_beta_0 = i * U_site * alpha_base * beta_base**2 + gauge_beta * beta_base
    alpha_next = alpha_base + noise_alpha + dt * drift_alpha_0
    beta_next = beta_base + noise_beta + dt * drift_beta_0

    residual_alpha, residual_beta = _nonlinear_midpoint_residual(
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
    )
    residual_norm = _local_residual_norm(residual_alpha, residual_beta)
    tolerance = _newton_tolerance(
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        root_rtol,
        root_atol,
    )
    finite = (
        _complex_finite(alpha_next)
        & _complex_finite(beta_next)
        & jnp.isfinite(residual_norm)
    )
    converged = finite & (residual_norm <= tolerance)
    iterations = jnp.where(
        converged,
        jnp.zeros_like(residual_norm, dtype=jnp.int32),
        jnp.full_like(residual_norm, max_iter, dtype=jnp.int32),
    )
    linear_solve_failed = jnp.zeros_like(converged)
    real_dtype = jnp.real(alpha_next).dtype

    def newton_condition(carry):
        iteration, _, _, converged, _, _ = carry
        return (iteration < max_iter) & jnp.any(~converged)

    def newton_body(carry):
        (
            iteration,
            alpha_next,
            beta_next,
            converged,
            iterations,
            linear_solve_failed,
        ) = carry
        residual_alpha, residual_beta = _nonlinear_midpoint_residual(
            alpha_next,
            beta_next,
            alpha_base,
            beta_base,
            noise_alpha,
            noise_beta,
            U_site,
            gauge_alpha,
            gauge_beta,
            dt,
        )
        residual_norm = _local_residual_norm(residual_alpha, residual_beta)
        jac_aa, jac_ab, jac_ba, jac_bb = _nonlinear_midpoint_jacobian(
            alpha_next,
            beta_next,
            alpha_base,
            beta_base,
            U_site,
            gauge_alpha,
            gauge_beta,
            dt,
        )
        determinant, determinant_bad = _jacobian_determinant_is_bad(
            jac_aa,
            jac_ab,
            jac_ba,
            jac_bb,
        )
        active = ~converged
        linear_solve_failed = linear_solve_failed | (active & determinant_bad)
        safe_determinant = jnp.where(determinant_bad, jnp.ones_like(determinant), determinant)

        delta_alpha = (jac_bb * residual_alpha - jac_ab * residual_beta) / safe_determinant
        delta_beta = (jac_aa * residual_beta - jac_ba * residual_alpha) / safe_determinant
        solve_finite = _complex_finite(delta_alpha) & _complex_finite(delta_beta)
        linear_solve_failed = linear_solve_failed | (active & ~solve_finite)
        can_update = active & ~determinant_bad & solve_finite

        def evaluate_damped_candidate(damping_index):
            damping = jnp.asarray(0.5, dtype=real_dtype) ** damping_index
            candidate_alpha = alpha_next - damping * delta_alpha
            candidate_beta = beta_next - damping * delta_beta
            candidate_residual_alpha, candidate_residual_beta = _nonlinear_midpoint_residual(
                candidate_alpha,
                candidate_beta,
                alpha_base,
                beta_base,
                noise_alpha,
                noise_beta,
                U_site,
                gauge_alpha,
                gauge_beta,
                dt,
            )
            candidate_norm = _local_residual_norm(
                candidate_residual_alpha,
                candidate_residual_beta,
            )
            candidate_finite = (
                _complex_finite(candidate_alpha)
                & _complex_finite(candidate_beta)
                & jnp.isfinite(candidate_norm)
            )
            return candidate_alpha, candidate_beta, candidate_norm, candidate_finite

        (
            full_alpha,
            full_beta,
            full_norm,
            full_finite,
        ) = evaluate_damped_candidate(jnp.asarray(0, dtype=jnp.int32))
        full_improve = can_update & full_finite & (full_norm < residual_norm)
        best_alpha = jnp.where(full_improve, full_alpha, alpha_next)
        best_beta = jnp.where(full_improve, full_beta, beta_next)
        best_norm = jnp.where(full_improve, full_norm, residual_norm)
        full_tolerance = _newton_tolerance(
            full_alpha,
            full_beta,
            alpha_base,
            beta_base,
            noise_alpha,
            noise_beta,
            root_rtol,
            root_atol,
        )
        full_converged = full_finite & (full_norm <= full_tolerance)
        full_step_finishes_iteration = jnp.all(
            ~active | ~can_update | (full_improve & full_converged)
        )

        def damping_body(damping_index, best):
            best_alpha, best_beta, best_norm = best
            (
                candidate_alpha,
                candidate_beta,
                candidate_norm,
                candidate_finite,
            ) = evaluate_damped_candidate(damping_index)
            improve = can_update & candidate_finite & (candidate_norm < best_norm)
            best_alpha = jnp.where(improve, candidate_alpha, best_alpha)
            best_beta = jnp.where(improve, candidate_beta, best_beta)
            best_norm = jnp.where(improve, candidate_norm, best_norm)
            return best_alpha, best_beta, best_norm

        alpha_next, beta_next, best_norm = lax.cond(
            full_step_finishes_iteration,
            lambda best: best,
            lambda best: lax.fori_loop(
                1,
                damping_steps,
                damping_body,
                best,
            ),
            (best_alpha, best_beta, best_norm),
        )
        tolerance = _newton_tolerance(
            alpha_next,
            beta_next,
            alpha_base,
            beta_base,
            noise_alpha,
            noise_beta,
            root_rtol,
            root_atol,
        )
        finite_after = (
            _complex_finite(alpha_next)
            & _complex_finite(beta_next)
            & jnp.isfinite(best_norm)
        )
        converged_after = finite_after & (best_norm <= tolerance)
        newly_converged = ~converged & converged_after
        iterations = jnp.where(newly_converged, iteration + 1, iterations)
        converged = converged | converged_after
        return (
            iteration + 1,
            alpha_next,
            beta_next,
            converged,
            iterations,
            linear_solve_failed,
        )

    (
        _,
        alpha_next,
        beta_next,
        converged,
        iterations,
        linear_solve_failed,
    ) = lax.while_loop(
        newton_condition,
        newton_body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            alpha_next,
            beta_next,
            converged,
            iterations,
            linear_solve_failed,
        ),
    )

    residual_alpha, residual_beta = _nonlinear_midpoint_residual(
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
    )
    residual_norm = _local_residual_norm(residual_alpha, residual_beta)
    tolerance = _newton_tolerance(
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        root_rtol,
        root_atol,
    )
    finite = (
        _complex_finite(alpha_next)
        & _complex_finite(beta_next)
        & jnp.isfinite(residual_norm)
    )
    converged = finite & (residual_norm <= tolerance)
    final_jacobian = _nonlinear_midpoint_jacobian(
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
    )
    _, final_jacobian_bad = _jacobian_determinant_is_bad(*final_jacobian)
    linear_solve_failed = linear_solve_failed | final_jacobian_bad
    diagnostics = SolverStepDiagnostics(
        converged=lax.stop_gradient(converged),
        residual_norm=lax.stop_gradient(residual_norm),
        newton_iterations=lax.stop_gradient(iterations),
        linear_solve_failed=lax.stop_gradient(linear_solve_failed),
        finite=lax.stop_gradient(finite),
    )
    return alpha_next, beta_next, diagnostics


@partial(jax.custom_vjp, nondiff_argnums=(10, 11))
def _implicit_midpoint_root(
    alpha_base,
    beta_base,
    noise_alpha,
    noise_beta,
    U_site,
    gauge_alpha,
    gauge_beta,
    dt,
    root_rtol,
    root_atol,
    max_iter,
    damping_steps,
):
    """Damped-Newton root with an exact implicit reverse-mode derivative."""

    return _damped_newton_midpoint(
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
        root_rtol,
        root_atol,
        max_iter,
        damping_steps,
    )


def _implicit_midpoint_root_fwd(
    alpha_base,
    beta_base,
    noise_alpha,
    noise_beta,
    U_site,
    gauge_alpha,
    gauge_beta,
    dt,
    root_rtol,
    root_atol,
    max_iter,
    damping_steps,
):
    output = _damped_newton_midpoint(
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
        root_rtol,
        root_atol,
        max_iter,
        damping_steps,
    )
    alpha_next, beta_next, _ = output
    saved = (
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
        root_rtol,
        root_atol,
    )
    return output, saved


def _implicit_midpoint_root_bwd(max_iter, damping_steps, saved, output_cotangent):
    del max_iter, damping_steps
    (
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
        root_rtol,
        root_atol,
    ) = saved
    alpha_cotangent, beta_cotangent, _ = output_cotangent

    jac_aa, jac_ab, jac_ba, jac_bb = _nonlinear_midpoint_jacobian(
        alpha_next,
        beta_next,
        alpha_base,
        beta_base,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
    )
    determinant, determinant_bad = _jacobian_determinant_is_bad(
        jac_aa,
        jac_ab,
        jac_ba,
        jac_bb,
    )
    safe_determinant = jnp.where(
        determinant_bad,
        jnp.ones_like(determinant),
        determinant,
    )
    # JAX's cotangent convention for holomorphic complex maps is the complex
    # transpose (not the Hermitian transpose): solve J^T lambda = z_bar.
    lambda_alpha = (
        jac_bb * alpha_cotangent - jac_ba * beta_cotangent
    ) / safe_determinant
    lambda_beta = (
        jac_aa * beta_cotangent - jac_ab * alpha_cotangent
    ) / safe_determinant
    lambda_alpha = jnp.where(determinant_bad, jnp.zeros_like(lambda_alpha), lambda_alpha)
    lambda_beta = jnp.where(determinant_bad, jnp.zeros_like(lambda_beta), lambda_beta)

    def residual_with_fixed_solution(
        alpha_base_arg,
        beta_base_arg,
        noise_alpha_arg,
        noise_beta_arg,
        U_site_arg,
        gauge_alpha_arg,
        gauge_beta_arg,
        dt_arg,
    ):
        return _nonlinear_midpoint_residual(
            alpha_next,
            beta_next,
            alpha_base_arg,
            beta_base_arg,
            noise_alpha_arg,
            noise_beta_arg,
            U_site_arg,
            gauge_alpha_arg,
            gauge_beta_arg,
            dt_arg,
        )

    _, residual_pullback = jax.vjp(
        residual_with_fixed_solution,
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
    )
    input_cotangents = residual_pullback((-lambda_alpha, -lambda_beta))
    return (
        *input_cotangents,
        jnp.zeros_like(root_rtol),
        jnp.zeros_like(root_atol),
    )


_implicit_midpoint_root.defvjp(
    _implicit_midpoint_root_fwd,
    _implicit_midpoint_root_bwd,
)


def _diagonal_affine_flow(alpha, beta, half_dt, gamma_site, Delta_site, F_site, J):
    """Exact affine flow when hopping is absent (including the one-site case)."""

    i = cplx_i()
    num_site = alpha.shape[-1]
    alpha_rate = -0.5 * gamma_site + i * Delta_site
    J_array = jnp.asarray(J, dtype=CDTYPE)
    if num_site == 1 and J_array.ndim == 2:
        alpha_rate = alpha_rate + i * J_array[0, 0]
    beta_rate = jnp.conj(alpha_rate)
    alpha_drive = -i * F_site
    beta_drive = jnp.conj(alpha_drive)

    def phi1(rate):
        argument = half_dt * rate
        use_direct = jnp.abs(argument) > jnp.asarray(1.0e-5, dtype=DTYPE)
        safe_rate = jnp.where(use_direct, rate, jnp.ones_like(rate))
        direct = jnp.expm1(argument) / safe_rate
        series = half_dt * (
            1.0
            + argument / 2.0
            + argument**2 / 6.0
            + argument**3 / 24.0
            + argument**4 / 120.0
        )
        return jnp.where(use_direct, direct, series)

    alpha_next = alpha * jnp.exp(half_dt * alpha_rate) + alpha_drive * phi1(alpha_rate)
    beta_next = beta * jnp.exp(half_dt * beta_rate) + beta_drive * phi1(beta_rate)
    return alpha_next, beta_next


def _matrix_free_affine_flow(
    alpha,
    beta,
    half_dt,
    gamma_site,
    Delta_site,
    F_site,
    J,
    order,
    substeps,
):
    """Static matrix-free exponential action for a sparse edge-list operator."""

    i = cplx_i()
    alpha_rate = -0.5 * gamma_site + i * Delta_site
    alpha_drive = jnp.broadcast_to(-i * F_site, alpha.shape)
    substep_dt = half_dt / substeps

    # Conjugated beta obeys the same affine equation as alpha.  Stacking the
    # two channels halves the number of dense matmul / sparse scatter launches
    # without assuming a real hopping operator.
    paired = jnp.stack((alpha, jnp.conj(beta)), axis=0)
    paired_drive = jnp.stack((alpha_drive, alpha_drive), axis=0)

    def linear_paired(value):
        return alpha_rate * value + i * apply_hopping_matrix(J, value)

    def substep_body(_, paired):
        paired_result = paired
        paired_term = substep_dt * (linear_paired(paired) + paired_drive)
        paired_result = paired_result + paired_term

        def degree_body(degree, series_state):
            paired_term, paired_result = series_state
            paired_term = (substep_dt / degree) * linear_paired(paired_term)
            paired_result = paired_result + paired_term
            return paired_term, paired_result

        _, paired_result = lax.fori_loop(
            2,
            order + 1,
            degree_body,
            (paired_term, paired_result),
        )
        return paired_result

    paired = lax.fori_loop(0, substeps, substep_body, paired)
    return paired[0], jnp.conj(paired[1])


def _affine_half_flow(
    alpha,
    beta,
    half_dt,
    gamma_site,
    Delta_site,
    F_site,
    J,
    order,
    substeps,
):
    if order < 1:
        raise ValueError("affine_expm_order must be at least 1")
    if substeps < 1:
        raise ValueError("affine_expm_substeps must be at least 1")

    num_site = alpha.shape[-1]
    if isinstance(J, dict):
        if J["edge_src"].shape[0] == 0:
            return _diagonal_affine_flow(
                alpha,
                beta,
                half_dt,
                gamma_site,
                Delta_site,
                F_site,
                0.0,
            )
        return _matrix_free_affine_flow(
            alpha,
            beta,
            half_dt,
            gamma_site,
            Delta_site,
            F_site,
            J,
            order,
            substeps,
        )

    J_array = jnp.asarray(J, dtype=CDTYPE)
    if J_array.ndim == 0 or num_site == 1:
        return _diagonal_affine_flow(
            alpha,
            beta,
            half_dt,
            gamma_site,
            Delta_site,
            F_site,
            J_array,
        )

    return lax.cond(
        jnp.all(J_array == 0),
        lambda _: _diagonal_affine_flow(
            alpha,
            beta,
            half_dt,
            gamma_site,
            Delta_site,
            F_site,
            0.0,
        ),
        lambda _: _matrix_free_affine_flow(
            alpha,
            beta,
            half_dt,
            gamma_site,
            Delta_site,
            F_site,
            J_array,
            order,
            substeps,
        ),
        operand=None,
    )


class InteractionPictureImplicitMidpointSolver(GaugeSDESolver):
    """Affine interaction-picture step with a local implicit Kerr midpoint.

    The loss/detuning/hopping/drive subflow is advanced for half a step.  The
    Ito diffusion is evaluated at the old state, as required by the explicit
    Ito prepoint convention, and that increment is transformed by the same
    homogeneous affine half-flow.  The local Kerr plus frozen gauge drift is
    then solved by damped Newton over a full step.  A second affine half-flow
    completes the symmetric drift splitting.
    """

    solver_name = "interaction_picture_implicit_midpoint"

    @classmethod
    def step(
        cls,
        state: SolverState,
        gauge_fields: GaugeFields,
        noise: SolverNoise,
        coefficients: SolverCoefficients,
        sde_max_iter: int = 4,
    ) -> SolverState:
        return _interaction_picture_implicit_midpoint_step_impl(
            state=state,
            gauge_fields=gauge_fields,
            noise=noise,
            coefficients=coefficients,
            sde_max_iter=sde_max_iter,
        )

    @classmethod
    def step_with_diagnostics(
        cls,
        state: SolverState,
        gauge_fields: GaugeFields,
        noise: SolverNoise,
        coefficients: SolverCoefficients,
        sde_max_iter: int = 4,
    ) -> Tuple[SolverState, SolverStepDiagnostics]:
        return _interaction_picture_implicit_midpoint_step_with_diagnostics_impl(
            state=state,
            gauge_fields=gauge_fields,
            noise=noise,
            coefficients=coefficients,
            sde_max_iter=sde_max_iter,
        )


def _interaction_picture_implicit_midpoint_step_core(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
    sde_max_iter: int,
) -> Tuple[SolverState, SolverStepDiagnostics]:
    i = cplx_i()
    alpha = jnp.asarray(state.alpha, dtype=CDTYPE)
    beta = jnp.asarray(state.beta, dtype=CDTYPE)
    ln_omega = jnp.asarray(state.ln_omega, dtype=CDTYPE)
    _, num_site = alpha.shape

    dt = jnp.asarray(coefficients.dt, dtype=DTYPE)
    half_dt = 0.5 * dt
    U_site = broadcast_site_param(coefficients.U, num_site, DTYPE)
    gamma_site = broadcast_site_param(coefficients.gamma, num_site, DTYPE)
    Delta_site = broadcast_site_param(coefficients.Delta, num_site, DTYPE)
    F_site = jnp.broadcast_to(jnp.asarray(coefficients.F, dtype=CDTYPE), (num_site,))
    root_rtol = jnp.asarray(coefficients.root_rtol, dtype=DTYPE)
    root_atol = jnp.asarray(coefficients.root_atol, dtype=DTYPE)

    drift_g = jnp.asarray(gauge_fields.drift_g, dtype=DTYPE)
    drift_f = jnp.asarray(gauge_fields.drift_f, dtype=DTYPE)
    diffusion_g = jnp.asarray(gauge_fields.diffusion_g, dtype=DTYPE)
    dW = jnp.asarray(noise.dW, dtype=DTYPE)
    dWp = jnp.asarray(noise.dWp, dtype=DTYPE)
    g = to_c(drift_g[..., :num_site], drift_g[..., num_site:])
    f = to_c(drift_f[..., :num_site], drift_f[..., num_site:])
    ch = jnp.cosh(diffusion_g)
    sh = jnp.sinh(diffusion_g)

    sqrt_ik = jnp.sqrt(i * jnp.asarray(U_site, dtype=CDTYPE))
    noise_alpha_old = (i * sqrt_ik) * alpha * (ch * dW + i * sh * dWp)
    noise_beta_old = sqrt_ik * beta * (-i * sh * dW + ch * dWp)
    # Batch the state and noise-vector exponential actions.  The homogeneous
    # noise channel has zero affine drive, while the state channel retains F.
    alpha_pair = jnp.stack((alpha, noise_alpha_old), axis=0)
    beta_pair = jnp.stack((beta, noise_beta_old), axis=0)
    state_drive = jnp.broadcast_to(F_site, alpha.shape)
    pair_drive = jnp.stack((state_drive, jnp.zeros_like(state_drive)), axis=0)
    alpha_pair, beta_pair = _affine_half_flow(
        alpha_pair,
        beta_pair,
        half_dt,
        gamma_site,
        Delta_site,
        pair_drive,
        coefficients.J,
        coefficients.affine_expm_order,
        coefficients.affine_expm_substeps,
    )
    alpha_base, noise_alpha = alpha_pair[0], alpha_pair[1]
    beta_base, noise_beta = beta_pair[0], beta_pair[1]
    gauge_alpha = -i * sqrt_ik * (g * ch + i * f * sh)
    gauge_beta = -sqrt_ik * (-i * g * sh + f * ch)

    alpha_root, beta_root, diagnostics = _implicit_midpoint_root(
        alpha_base,
        beta_base,
        noise_alpha,
        noise_beta,
        U_site,
        gauge_alpha,
        gauge_beta,
        dt,
        root_rtol,
        root_atol,
        sde_max_iter,
        coefficients.newton_damping_steps,
    )
    alpha_next, beta_next = _affine_half_flow(
        alpha_root,
        beta_root,
        half_dt,
        gamma_site,
        Delta_site,
        F_site,
        coefficients.J,
        coefficients.affine_expm_order,
        coefficients.affine_expm_substeps,
    )

    dln_omega = jnp.sum(g * dW + f * dWp - 0.5 * (g**2 + f**2) * dt, axis=-1)
    local_center = jnp.mean(jnp.real(dln_omega), axis=0)
    if coefficients.center_axis_name is not None:
        local_center = lax.pmean(local_center, coefficients.center_axis_name)
    dln_omega = dln_omega - lax.stop_gradient(local_center)
    ln_omega_next = ln_omega + dln_omega

    final_finite = _complex_finite(alpha_next) & _complex_finite(beta_next)
    weight_finite = _complex_finite(ln_omega_next)[..., None]
    local_finite = diagnostics.finite & final_finite & weight_finite
    local_success = (
        diagnostics.converged
        & ~diagnostics.linear_solve_failed
        & local_finite
    )
    diagnostics = SolverStepDiagnostics(
        converged=diagnostics.converged,
        residual_norm=diagnostics.residual_norm,
        newton_iterations=diagnostics.newton_iterations,
        linear_solve_failed=diagnostics.linear_solve_failed,
        finite=lax.stop_gradient(local_finite),
    )

    # Hopping in the second half-flow couples all sites.  Consequently a failed
    # local root invalidates its whole walker, not just the failed site.
    walker_success = jnp.all(local_success, axis=-1)
    complex_nan = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=CDTYPE)
    alpha_next = jnp.where(walker_success[..., None], alpha_next, complex_nan)
    beta_next = jnp.where(walker_success[..., None], beta_next, complex_nan)
    ln_omega_next = jnp.where(walker_success, ln_omega_next, complex_nan)

    return (
        SolverState(
            ln_omega=ln_omega_next,
            alpha=alpha_next,
            beta=beta_next,
        ),
        diagnostics,
    )


@partial(jax.jit, static_argnums=(4,))
def _interaction_picture_implicit_midpoint_step_impl(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
    sde_max_iter: int,
) -> SolverState:
    next_state, _ = _interaction_picture_implicit_midpoint_step_core(
        state,
        gauge_fields,
        noise,
        coefficients,
        sde_max_iter,
    )
    return next_state


@partial(jax.jit, static_argnums=(4,))
def _interaction_picture_implicit_midpoint_step_with_diagnostics_impl(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
    sde_max_iter: int,
) -> Tuple[SolverState, SolverStepDiagnostics]:
    return _interaction_picture_implicit_midpoint_step_core(
        state,
        gauge_fields,
        noise,
        coefficients,
        sde_max_iter,
    )


def _stable_phi1(argument):
    """Return ``(exp(argument) - 1) / argument`` stably near zero."""

    real_dtype = jnp.real(argument).dtype
    use_direct = jnp.abs(argument) > jnp.asarray(1.0e-5, dtype=real_dtype)
    safe_argument = jnp.where(use_direct, argument, jnp.ones_like(argument))
    direct = jnp.expm1(argument) / safe_argument
    series = (
        1.0
        + argument / 2.0
        + argument**2 / 6.0
        + argument**3 / 24.0
        + argument**4 / 120.0
    )
    return jnp.where(use_direct, direct, series)


def _exact_onsite_deterministic_flow(
    alpha,
    beta,
    flow_dt,
    U_site,
    gamma_site,
    Delta_site,
):
    """Apply the exact onsite Kerr/loss/detuning deterministic flow.

    For the local deterministic equations

      d alpha / dt = alpha (-i U alpha beta - gamma/2 + i Delta),
      d beta  / dt = beta  ( i U alpha beta - gamma/2 - i Delta),

    the generally complex phase-space occupation evolves as
    ``n(t) = n(0) exp(-gamma t)``.  Integrating it analytically gives the
    exponent below, including the amplitude change produced when ``n`` is off
    the real physical manifold.
    """

    i = cplx_i()
    occupation_0 = alpha * beta
    gamma_site = jnp.asarray(gamma_site, dtype=DTYPE)
    Delta_site = jnp.asarray(Delta_site, dtype=DTYPE)
    U_site = jnp.asarray(U_site, dtype=CDTYPE)
    decay_argument = -gamma_site * flow_dt
    integrated_occupation = occupation_0 * flow_dt * _stable_phi1(decay_argument)
    alpha_exponent = (
        (-0.5 * gamma_site + i * Delta_site) * flow_dt
        - i * U_site * integrated_occupation
    )
    beta_exponent = (
        (-0.5 * gamma_site - i * Delta_site) * flow_dt
        + i * U_site * integrated_occupation
    )
    return alpha * jnp.exp(alpha_exponent), beta * jnp.exp(beta_exponent)


def _exact_frozen_gauge_diffusion_flow(
    alpha,
    beta,
    dt,
    U_site,
    g,
    f,
    diffusion_g,
    dW,
    dWp,
):
    """Exact geometric-Ito flow for frozen gauge and diffusion fields.

    The two real Wiener processes are shared by the phase-space variables and
    the stochastic weight.  The returned weight increment is the logarithmic
    gauge-weight increment before common-real-part centering.  Even when a
    sampled ``dW`` and ``dWp`` happen to be zero, the state exponent retains
    its ``-B**2 dt / 2`` Ito correction; zero increments do not turn diffusion
    off.
    """

    i = cplx_i()
    sqrt_ik = jnp.sqrt(i * jnp.asarray(U_site, dtype=CDTYPE))
    ch = jnp.cosh(diffusion_g)
    sh = jnp.sinh(diffusion_g)

    # These coefficients reproduce the prepoint-Ito noise convention used by
    # SemiImplicitMidpointSolver.  In particular, the deterministic drift
    # gauge is -B @ (g, f), hence the shifted increments below.
    b_alpha_w = i * sqrt_ik * ch
    b_alpha_wp = -sqrt_ik * sh
    b_beta_w = -i * sqrt_ik * sh
    b_beta_wp = sqrt_ik * ch
    shifted_dW = dW - g * dt
    shifted_dWp = dWp - f * dt

    # Use the analytic identities
    #   b_alpha_w**2 + b_alpha_wp**2 = -i U,
    #   b_beta_w**2  + b_beta_wp**2  =  i U,
    # instead of subtracting cosh(lambda)**2 and sinh(lambda)**2.  The latter
    # is catastrophically ill-conditioned for a large diffusion gauge.
    ito_alpha = 0.5 * i * jnp.asarray(U_site, dtype=CDTYPE) * dt
    ito_beta = -0.5 * i * jnp.asarray(U_site, dtype=CDTYPE) * dt
    alpha_exponent = (
        b_alpha_w * shifted_dW
        + b_alpha_wp * shifted_dWp
        + ito_alpha
    )
    beta_exponent = (
        b_beta_w * shifted_dW
        + b_beta_wp * shifted_dWp
        + ito_beta
    )
    alpha_next = alpha * jnp.exp(alpha_exponent)
    beta_next = beta * jnp.exp(beta_exponent)
    dln_omega = jnp.sum(
        g * dW + f * dWp - 0.5 * (g**2 + f**2) * dt,
        axis=-1,
    )
    return alpha_next, beta_next, dln_omega


class InteractionPictureExactLocalSolver(GaugeSDESolver):
    """Interaction-picture split step with exact frozen onsite subflows.

    One step is composed as H(dt/2) D(dt/2) S(dt) D(dt/2) H(dt/2):

    - H is the existing affine flow restricted to hopping and coherent drive,
    - D is the exact onsite Kerr/loss/detuning deterministic flow, and
    - S is the exact geometric-Ito flow for frozen drift/diffusion gauges and
      their coupled logarithmic weight update.

    Gauge fields are supplied by the caller and are treated as adapted
    prepoint values frozen over the whole step.  The composition removes the
    local implicit root and its midpoint Kerr phase error, while retaining a
    finite-step splitting error between the noncommuting subflows.
    """

    solver_name = "interaction_picture_exact_local"

    @classmethod
    def step(
        cls,
        state: SolverState,
        gauge_fields: GaugeFields,
        noise: SolverNoise,
        coefficients: SolverCoefficients,
        sde_max_iter: int = 4,
    ) -> SolverState:
        del sde_max_iter
        return _interaction_picture_exact_local_step_impl(
            state=state,
            gauge_fields=gauge_fields,
            noise=noise,
            coefficients=coefficients,
        )

    @classmethod
    def step_with_diagnostics(
        cls,
        state: SolverState,
        gauge_fields: GaugeFields,
        noise: SolverNoise,
        coefficients: SolverCoefficients,
        sde_max_iter: int = 4,
    ) -> Tuple[SolverState, SolverStepDiagnostics]:
        del sde_max_iter
        return _interaction_picture_exact_local_step_with_diagnostics_impl(
            state=state,
            gauge_fields=gauge_fields,
            noise=noise,
            coefficients=coefficients,
        )


def _interaction_picture_exact_local_step_core(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
) -> Tuple[SolverState, SolverStepDiagnostics]:
    alpha = jnp.asarray(state.alpha, dtype=CDTYPE)
    beta = jnp.asarray(state.beta, dtype=CDTYPE)
    ln_omega = jnp.asarray(state.ln_omega, dtype=CDTYPE)
    _, num_site = alpha.shape

    dt = jnp.asarray(coefficients.dt, dtype=DTYPE)
    half_dt = 0.5 * dt
    U_site = broadcast_site_param(coefficients.U, num_site, DTYPE)
    gamma_site = broadcast_site_param(coefficients.gamma, num_site, DTYPE)
    Delta_site = broadcast_site_param(coefficients.Delta, num_site, DTYPE)
    F_site = jnp.broadcast_to(jnp.asarray(coefficients.F, dtype=CDTYPE), (num_site,))
    zero_site = jnp.zeros_like(gamma_site)

    drift_g = jnp.asarray(gauge_fields.drift_g, dtype=DTYPE)
    drift_f = jnp.asarray(gauge_fields.drift_f, dtype=DTYPE)
    diffusion_g = jnp.asarray(gauge_fields.diffusion_g, dtype=DTYPE)
    dW = jnp.asarray(noise.dW, dtype=DTYPE)
    dWp = jnp.asarray(noise.dWp, dtype=DTYPE)
    g = to_c(drift_g[..., :num_site], drift_g[..., num_site:])
    f = to_c(drift_f[..., :num_site], drift_f[..., num_site:])

    local_finite = _complex_finite(alpha) & _complex_finite(beta)

    alpha, beta = _affine_half_flow(
        alpha,
        beta,
        half_dt,
        zero_site,
        zero_site,
        F_site,
        coefficients.J,
        coefficients.affine_expm_order,
        coefficients.affine_expm_substeps,
    )
    local_finite = local_finite & _complex_finite(alpha) & _complex_finite(beta)

    alpha, beta = _exact_onsite_deterministic_flow(
        alpha,
        beta,
        half_dt,
        U_site,
        gamma_site,
        Delta_site,
    )
    local_finite = local_finite & _complex_finite(alpha) & _complex_finite(beta)

    alpha, beta, dln_omega = _exact_frozen_gauge_diffusion_flow(
        alpha,
        beta,
        dt,
        U_site,
        g,
        f,
        diffusion_g,
        dW,
        dWp,
    )
    local_finite = local_finite & _complex_finite(alpha) & _complex_finite(beta)

    alpha, beta = _exact_onsite_deterministic_flow(
        alpha,
        beta,
        half_dt,
        U_site,
        gamma_site,
        Delta_site,
    )
    local_finite = local_finite & _complex_finite(alpha) & _complex_finite(beta)

    alpha_next, beta_next = _affine_half_flow(
        alpha,
        beta,
        half_dt,
        zero_site,
        zero_site,
        F_site,
        coefficients.J,
        coefficients.affine_expm_order,
        coefficients.affine_expm_substeps,
    )
    local_finite = (
        local_finite
        & _complex_finite(alpha_next)
        & _complex_finite(beta_next)
    )

    # Preserve the common real log-weight centering used by the existing
    # solvers.  The stopped common shift only rescales all walker weights.
    local_center = jnp.mean(jnp.real(dln_omega), axis=0)
    if coefficients.center_axis_name is not None:
        local_center = lax.pmean(local_center, coefficients.center_axis_name)
    dln_omega = dln_omega - lax.stop_gradient(local_center)
    ln_omega_next = ln_omega + dln_omega
    local_finite = local_finite & _complex_finite(ln_omega_next)[..., None]

    # This backend has no nonlinear equation.  Reuse the common diagnostic
    # structure with zero residual/iterations and convergence equal to finite
    # completion of every exact/split subflow.
    real_dtype = jnp.real(alpha_next).dtype
    residual_norm = jnp.where(
        local_finite,
        jnp.zeros_like(jnp.real(alpha_next), dtype=real_dtype),
        jnp.full_like(jnp.real(alpha_next), jnp.inf, dtype=real_dtype),
    )
    diagnostics = SolverStepDiagnostics(
        converged=lax.stop_gradient(local_finite),
        residual_norm=lax.stop_gradient(residual_norm),
        newton_iterations=jnp.zeros_like(residual_norm, dtype=jnp.int32),
        linear_solve_failed=jnp.zeros_like(local_finite),
        finite=lax.stop_gradient(local_finite),
    )

    # Both affine half-flows couple sites, so any invalid site invalidates the
    # complete walker.  Returning NaNs makes failure visible even to callers
    # that use only the standard step interface.
    walker_success = jnp.all(local_finite, axis=-1)
    complex_nan = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=CDTYPE)
    alpha_next = jnp.where(walker_success[..., None], alpha_next, complex_nan)
    beta_next = jnp.where(walker_success[..., None], beta_next, complex_nan)
    ln_omega_next = jnp.where(walker_success, ln_omega_next, complex_nan)

    return (
        SolverState(
            ln_omega=ln_omega_next,
            alpha=alpha_next,
            beta=beta_next,
        ),
        diagnostics,
    )


@jax.jit
def _interaction_picture_exact_local_step_impl(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
) -> SolverState:
    next_state, _ = _interaction_picture_exact_local_step_core(
        state,
        gauge_fields,
        noise,
        coefficients,
    )
    return next_state


@jax.jit
def _interaction_picture_exact_local_step_with_diagnostics_impl(
    state: SolverState,
    gauge_fields: GaugeFields,
    noise: SolverNoise,
    coefficients: SolverCoefficients,
) -> Tuple[SolverState, SolverStepDiagnostics]:
    return _interaction_picture_exact_local_step_core(
        state,
        gauge_fields,
        noise,
        coefficients,
    )


SOLVER_REGISTRY = {
    InteractionPictureExactLocalSolver.solver_name: InteractionPictureExactLocalSolver,
    InteractionPictureImplicitMidpointSolver.solver_name: InteractionPictureImplicitMidpointSolver,
    SemiImplicitMidpointSolver.solver_name: SemiImplicitMidpointSolver,
}


def get_solver(solver_name: str = SemiImplicitMidpointSolver.solver_name):
    """Return the solver class registered under `solver_name`."""

    try:
        return SOLVER_REGISTRY[solver_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(SOLVER_REGISTRY))
        raise ValueError(f"Unknown solver '{solver_name}'. Valid options: {valid_names}") from exc


__all__ = [
    "GaugeFields",
    "GaugeSDESolver",
    "InteractionPictureExactLocalSolver",
    "InteractionPictureImplicitMidpointSolver",
    "SemiImplicitMidpointSolver",
    "SolverCoefficients",
    "SolverNoise",
    "SolverState",
    "SolverStepDiagnostics",
    "get_solver",
]
