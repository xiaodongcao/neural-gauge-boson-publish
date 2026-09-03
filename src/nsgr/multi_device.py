from dataclasses import dataclass

from .dynamics_kernel import (
    compute_grads_all_windows,
    run_one_window_simulation_rollout,
)
from .lib_preinclude import *
from .projected_residual import DEFAULT_RESIDUAL_GMM_TRACE_MODE
from .utility import PARETO_K_MONOMIAL_TERMS


@dataclass(frozen=True)
class MultiDeviceSpec:
    enabled: bool
    num_devices: int
    walkers_per_device: int
    devices: tuple
    axis_name: str = "device"


# Values in this collection include scalars, per-channel vectors, and
# covariance matrices.  Every entry is reduced across the device axis before
# returning one host-side auxiliary tree.
REDUCED_AUX_KEYS = (
    "loss",
    "loss_pareto_k",
    "loss_pareto_k_objective",
    "loss_pareto_k_terms",
    "loss_pareto_k_site_terms",
    *PARETO_K_MONOMIAL_TERMS,
    "pareto_k_mean",
    "pareto_k_max",
    "pareto_k_worst",
    "pareto_k_warning_fraction",
    "pareto_k_bad_fraction",
    "loss_residual_gmm",
    "loss_residual_gmm_time",
    "loss_residual_gmm_raw",
    "loss_residual_gmm_terms",
    "loss_residual_gmm_site_terms",
    "residual_gmm_z_mean",
    "residual_gmm_z_max",
    "residual_gmm_z_worst",
    "residual_gmm_radius_mean",
    "residual_gmm_radius_max",
    "residual_gmm_radius_worst",
    "residual_gmm_warning_fraction",
    "residual_gmm_bad_fraction",
    "residual_gmm_covariance_estimates",
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
    "loss_L2",
    "t_end",
)

MAX_REDUCED_AUX_KEYS = frozenset(
    key for key in REDUCED_AUX_KEYS if key.endswith("_max") or key.endswith("_worst")
)


def resolve_multi_device_spec(config_section: Dict[str, Any], num_walker: int, *, purpose: str) -> MultiDeviceSpec:
    """Resolve an opt-in walker-sharded pmap configuration."""
    raw = config_section.get("multi_device", {})
    if not isinstance(raw, dict):
        raise ValueError(f"{purpose}.multi_device must be a dictionary")
    enabled = bool(raw.get("enabled", False))
    axis_name = str(raw.get("axis_name", "device"))
    devices = tuple(jax.local_devices())

    if not enabled:
        return MultiDeviceSpec(False, 1, int(num_walker), devices[:1], axis_name)
    if not devices:
        raise RuntimeError(f"{purpose}.multi_device.enabled=true but JAX reports no local devices")

    requested = raw.get("num_devices", "auto")
    if requested is None or str(requested).strip().lower() == "auto":
        num_devices = len(devices)
    else:
        num_devices = int(requested)

    if num_devices < 1:
        raise ValueError(f"{purpose}.multi_device.num_devices must be positive")
    if num_devices > len(devices):
        raise ValueError(
            f"{purpose}.multi_device requested {num_devices} devices, but only {len(devices)} local devices are visible"
        )
    if num_devices == 1:
        return MultiDeviceSpec(False, 1, int(num_walker), devices[:1], axis_name)
    if int(num_walker) % num_devices != 0:
        raise ValueError(
            f"{purpose}.multi_device requires num_walker divisible by num_devices: "
            f"num_walker={num_walker}, num_devices={num_devices}"
        )
    return MultiDeviceSpec(True, num_devices, int(num_walker) // num_devices, devices[:num_devices], axis_name)


def shard_walkers(x: jnp.ndarray, spec: MultiDeviceSpec) -> jnp.ndarray:
    x = jnp.asarray(x)
    if not spec.enabled:
        return x
    return x.reshape((spec.num_devices, spec.walkers_per_device) + x.shape[1:])


def unshard_walkers(x: jnp.ndarray) -> np.ndarray:
    x_host = np.asarray(x)
    if x_host.ndim < 2:
        return x_host
    return x_host.reshape((x_host.shape[0] * x_host.shape[1],) + x_host.shape[2:])


def unshard_walkers_device(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x)
    if x.ndim < 2:
        return x
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def unshard_walker_history(x: jnp.ndarray) -> np.ndarray:
    x_host = np.asarray(x)
    if x_host.ndim < 3:
        return x_host
    x_host = np.swapaxes(x_host, 0, 1)
    return x_host.reshape((x_host.shape[0], x_host.shape[1] * x_host.shape[2]) + x_host.shape[3:])


def split_device_keys(key: jax.Array, spec: MultiDeviceSpec) -> jax.Array:
    if not spec.enabled:
        return key
    return random.split(key, spec.num_devices)


def _reduce_aux_across_devices(aux: Dict[str, Any], axis_name: str) -> Dict[str, Any]:
    out = {}
    for key in REDUCED_AUX_KEYS:
        if key in aux:
            if key in MAX_REDUCED_AUX_KEYS:
                out[key] = lax.pmax(aux[key], axis_name)
            else:
                out[key] = lax.pmean(aux[key], axis_name)
    return out


class MultiDeviceSimulationStepper:
    """Pmap one simulation window over the walker axis."""

    def __init__(
        self,
        *,
        apply_fn,
        N_steps: int,
        apply_neural_gauge_every_steps: int,
        sde_max_iter: int,
        sde_solver: str,
        sde_root_rtol: float,
        sde_root_atol: float,
        sde_affine_expm_order: int,
        sde_affine_expm_substeps: int,
        sde_newton_damping_steps: int,
        gauge_mode: str,
        neural_gauge_components: str,
        spec: MultiDeviceSpec,
    ):
        self.spec = spec

        def _step(
            key,
            params,
            lnOmega,
            alpha,
            beta,
            U,
            gamma,
            F,
            Delta,
            dt,
            t0,
            gauge_weight,
            n0,
            J,
            analytic_target_time,
        ):
            return run_one_window_simulation_rollout(
                key=key,
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
                t0=t0,
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
                center_axis_name=spec.axis_name,
            )

        self._step = jax.pmap(
            _step,
            axis_name=spec.axis_name,
            devices=spec.devices,
            in_axes=(0, None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None),
        )

    def __call__(self, *args):
        return self._step(*args)


class MultiDeviceSimulationRollout:
    """Pmap a complete simulation rollout over the walker axis."""

    def __init__(
        self,
        *,
        apply_fn,
        N_steps: int,
        N_windows: int,
        apply_neural_gauge_every_steps: int,
        sde_max_iter: int,
        sde_solver: str,
        sde_root_rtol: float,
        sde_root_atol: float,
        sde_affine_expm_order: int,
        sde_affine_expm_substeps: int,
        sde_newton_damping_steps: int,
        gauge_mode: str,
        neural_gauge_components: str,
        spec: MultiDeviceSpec,
    ):
        self.spec = spec

        def _rollout(
            key,
            params,
            lnOmega,
            alpha,
            beta,
            U,
            gamma,
            F,
            Delta,
            dt,
            t0,
            gauge_weight,
            n0,
            J,
            analytic_target_time,
        ):
            def body(carry, window_index):
                key_t, lnOmega_t, alpha_t, beta_t, t_t = carry
                sub = random.fold_in(key_t, window_index)
                key_out, lnOmega_next, alpha_next, beta_next, aux = run_one_window_simulation_rollout(
                    key=sub,
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
                    center_axis_name=spec.axis_name,
                )
                return (key_out, lnOmega_next, alpha_next, beta_next, aux["t_end"]), None

            return lax.scan(
                body,
                (key, lnOmega, alpha, beta, t0),
                jnp.arange(N_windows),
                length=N_windows,
            )[0]

        self._rollout = jax.pmap(
            _rollout,
            axis_name=spec.axis_name,
            devices=spec.devices,
            in_axes=(0, None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None),
        )

    def __call__(self, *args):
        return self._rollout(*args)


class MultiDeviceSimulationHistoryRollout:
    """Pmap a complete simulation rollout and return every window-end state."""

    def __init__(
        self,
        *,
        apply_fn,
        N_steps: int,
        N_windows: int,
        apply_neural_gauge_every_steps: int,
        sde_max_iter: int,
        sde_solver: str,
        sde_root_rtol: float,
        sde_root_atol: float,
        sde_affine_expm_order: int,
        sde_affine_expm_substeps: int,
        sde_newton_damping_steps: int,
        gauge_mode: str,
        neural_gauge_components: str,
        spec: MultiDeviceSpec,
    ):
        self.spec = spec

        def _rollout(
            key,
            params,
            lnOmega,
            alpha,
            beta,
            U,
            gamma,
            F,
            Delta,
            dt,
            t0,
            gauge_weight,
            n0,
            J,
            analytic_target_time,
        ):
            def body(carry, window_index):
                key_t, lnOmega_t, alpha_t, beta_t, t_t, lnOmega_shift_t = carry
                sub = random.fold_in(key_t, window_index)
                key_out, lnOmega_next, alpha_next, beta_next, aux = run_one_window_simulation_rollout(
                    key=sub,
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
                    center_axis_name=spec.axis_name,
                )
                lnOmega_shift_next = (
                    lnOmega_shift_t
                    + lax.stop_gradient(aux["lnOmega_center_shift"])
                )
                carry_next = (
                    key_out,
                    lnOmega_next,
                    alpha_next,
                    beta_next,
                    aux["t_end"],
                    lnOmega_shift_next,
                )
                return carry_next, (
                    lnOmega_next,
                    alpha_next,
                    beta_next,
                    aux["t_end"],
                    lnOmega_shift_next,
                )

            (
                key_out,
                _,
                _,
                _,
                _,
                _,
            ), (
                ln_hist,
                alpha_hist,
                beta_hist,
                t_hist,
                lnOmega_shift_hist,
            ) = lax.scan(
                body,
                (
                    key,
                    lnOmega,
                    alpha,
                    beta,
                    t0,
                    jnp.asarray(0.0, dtype=DTYPE),
                ),
                jnp.arange(N_windows),
                length=N_windows,
            )
            return (
                key_out,
                jnp.concatenate((lnOmega[jnp.newaxis, ...], ln_hist), axis=0),
                jnp.concatenate((alpha[jnp.newaxis, ...], alpha_hist), axis=0),
                jnp.concatenate((beta[jnp.newaxis, ...], beta_hist), axis=0),
                jnp.concatenate((jnp.asarray([t0], dtype=t_hist.dtype), t_hist), axis=0),
                jnp.concatenate(
                    (
                        jnp.asarray([0.0], dtype=lnOmega_shift_hist.dtype),
                        lnOmega_shift_hist,
                    ),
                    axis=0,
                ),
            )

        self._rollout = jax.pmap(
            _rollout,
            axis_name=spec.axis_name,
            devices=spec.devices,
            in_axes=(0, None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None),
        )

    def __call__(self, *args):
        return self._rollout(*args)


class MultiDeviceGradientComputer:
    """Pmap walker-sharded losses and average gradients across devices."""

    def __init__(
        self,
        *,
        apply_fn,
        N_steps: int,
        N_windows: int,
        apply_neural_gauge_every_steps: int,
        neural_gauge_each_apply: bool,
        sde_max_iter: int,
        sde_solver: str,
        sde_root_rtol: float,
        sde_root_atol: float,
        sde_affine_expm_order: int,
        sde_affine_expm_substeps: int,
        sde_newton_damping_steps: int,
        gauge_mode: str,
        neural_gauge_components: str,
        pareto_k_applied_quantities: int,
        pareto_k_applied_quantities_mode: str,
        pareto_k_monomials: tuple,
        pareto_k_envelope_excess: str,
        operator_monomials: tuple,
        residual_gmm_trace_mode: str = DEFAULT_RESIDUAL_GMM_TRACE_MODE,
        residual_gmm_time_aggregation: str,
        residual_gmm_integrator_nodes: int = 6,
        pareto_k_tail_fraction: float,
        pareto_k_min_tail_count: int,
        enable_loss_pareto_k: bool,
        enable_loss_residual_gmm: bool,
        enable_loss_gauge: bool,
        enable_loss_L2: bool,
        enable_loss_ess: bool,
        spec: MultiDeviceSpec,
    ):
        self.spec = spec

        def _grads(
            key,
            params,
            lnOmega0,
            alpha0,
            beta0,
            U,
            gamma,
            F,
            Delta,
            dt,
            window_loss_weights,
            lnOmega_shift0,
            residual_gmm_covariance_bank,
            residual_gmm_covariance_initialized,
            t0,
            gauge_weight,
            pareto_k_threshold,
            pareto_k_threshold_tau,
            pareto_k_envelope_beta,
            residual_gmm_d_clip,
            residual_gmm_cov_floor,
            residual_gmm_cov_shrinkage,
            residual_gmm_time_beta,
            n0,
            loss_pareto_k_prefactor,
            loss_pareto_k_monomial_weights,
            loss_residual_gmm_prefactor,
            loss_L2_prefactor,
            loss_gauge_prefactor,
            loss_ess_prefactor,
            loss_ess_window_budget,
            q_winsor,
            J,
            analytic_target_time,
        ):
            grads, aux, _loss = compute_grads_all_windows(
                key=key,
                apply_fn=apply_fn,
                params=params,
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
                residual_gmm_d_clip=residual_gmm_d_clip,
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
                loss_pareto_k_monomial_weights=loss_pareto_k_monomial_weights,
                loss_residual_gmm_prefactor=(
                    loss_residual_gmm_prefactor
                ),
                loss_L2_prefactor=loss_L2_prefactor,
                loss_gauge_prefactor=loss_gauge_prefactor,
                loss_ess_prefactor=loss_ess_prefactor,
                loss_ess_window_budget=loss_ess_window_budget,
                q_winsor=q_winsor,
                pareto_k_applied_quantities=pareto_k_applied_quantities,
                pareto_k_applied_quantities_mode=pareto_k_applied_quantities_mode,
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
                center_axis_name=spec.axis_name,
            )
            reduced_aux = _reduce_aux_across_devices(aux, spec.axis_name)
            reduced_aux["grads_norm"] = opx.global_norm(grads)
            return grads, reduced_aux

        in_axes = (0, None, 0, 0, 0) + (None,) * 29
        assert _grads.__code__.co_argcount == len(in_axes) == 34
        self._grads = jax.pmap(_grads, axis_name=spec.axis_name, devices=spec.devices, in_axes=in_axes)

    def gradients(self, *args) -> Tuple[Any, Dict[str, Any]]:
        grads_replicated, aux_replicated = self._grads(*args)
        grads = jax.tree_util.tree_map(lambda x: x[0], grads_replicated)
        aux = {key: value[0] for key, value in aux_replicated.items()}
        return grads, aux
