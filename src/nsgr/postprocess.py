from dataclasses import dataclass

from .lib_preinclude import *
from .utility import load_array_archive


@dataclass(frozen=True)
class WindowEndDataset:
    times: np.ndarray
    lnOmega_history: np.ndarray
    alpha_history: np.ndarray
    beta_history: np.ndarray
    lnOmega_shift_history: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.lnOmega_shift_history is None:
            object.__setattr__(
                self,
                "lnOmega_shift_history",
                np.zeros_like(self.times, dtype=float),
            )
        if self.times.ndim != 1:
            raise ValueError("times must have shape (num_times,)")
        if self.lnOmega_history.ndim != 2:
            raise ValueError("lnOmega_history must have shape (num_times, num_walker)")
        if self.lnOmega_shift_history.ndim != 1:
            raise ValueError("lnOmega_shift_history must have shape (num_times,)")
        if self.alpha_history.ndim != 3 or self.beta_history.ndim != 3:
            raise ValueError("alpha_history and beta_history must have shape (num_times, num_walker, num_site)")
        if self.lnOmega_history.shape[0] != self.times.shape[0]:
            raise ValueError("times and lnOmega_history must share the same leading dimension")
        if self.lnOmega_shift_history.shape[0] != self.times.shape[0]:
            raise ValueError("times and lnOmega_shift_history must share the same leading dimension")
        if self.alpha_history.shape[0] != self.times.shape[0] or self.beta_history.shape[0] != self.times.shape[0]:
            raise ValueError("times and trajectory histories must share the same leading dimension")
        if self.alpha_history.shape[:2] != self.beta_history.shape[:2]:
            raise ValueError("alpha_history and beta_history must share the same time and walker dimensions")
        if self.alpha_history.shape[1] != self.lnOmega_history.shape[1]:
            raise ValueError("trajectory histories must share the same walker dimension")
        if self.alpha_history.shape[2] != self.beta_history.shape[2]:
            raise ValueError("alpha_history and beta_history must share the same site dimension")

    @classmethod
    def from_path(cls, path: str):
        raw = load_array_archive(path)
        return cls(
            times=np.asarray(raw["times"]),
            lnOmega_history=np.asarray(raw["lnOmega_history"]),
            alpha_history=np.asarray(raw["alpha_history"]),
            beta_history=np.asarray(raw["beta_history"]),
            lnOmega_shift_history=np.asarray(
                raw.get(
                    "lnOmega_shift_history",
                    np.zeros_like(raw["times"], dtype=float),
                )
            ),
        )

    @property
    def num_times(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_walker(self) -> int:
        return int(self.alpha_history.shape[1])

    @property
    def num_site(self) -> int:
        return int(self.alpha_history.shape[2])

    def window_state(self, time_index: int):
        idx = int(time_index)
        return WindowState(
            time=float(np.asarray(self.times[idx])),
            lnOmega=np.asarray(self.lnOmega_history[idx]),
            alpha=np.asarray(self.alpha_history[idx]),
            beta=np.asarray(self.beta_history[idx]),
            lnOmega_shift=float(np.asarray(self.lnOmega_shift_history[idx])),
        )

    def final_state(self):
        return self.window_state(-1)

    def measurement_history(self, measurement_names: Optional[Sequence[str]] = None):
        return compute_measurement_history(self, measurement_names=measurement_names)

    def weighted_magnitude_histogram(
        self,
        time_index: int,
        quantity: str,
        range_min: float,
        range_max: float,
        n_bins: int = 100,
        scale: str = "linear",
        site: Optional[int] = None,
        eps: float = 1e-12,
    ):
        idx = int(time_index)
        if idx < 0:
            idx += self.num_times
        state = self.window_state(idx)
        return state.weighted_magnitude_histogram(
            quantity=quantity,
            range_min=range_min,
            range_max=range_max,
            n_bins=n_bins,
            scale=scale,
            site=site,
            window_index=idx,
            eps=eps,
        )

    def weighted_complex_histogram_2d(
        self,
        time_index: int,
        quantity: str,
        real_range,
        imag_range,
        n_bins=100,
        density: bool = False,
        site: Optional[int] = None,
        eps: float = 1e-12,
    ):
        idx = int(time_index)
        if idx < 0:
            idx += self.num_times
        state = self.window_state(idx)
        return state.weighted_complex_histogram_2d(
            quantity=quantity,
            real_range=real_range,
            imag_range=imag_range,
            n_bins=n_bins,
            density=density,
            site=site,
            window_index=idx,
            eps=eps,
        )


@dataclass(frozen=True)
class MeasurementSnapshot:
    measurements: Dict[str, np.ndarray]
    measurement_errors: Dict[str, np.ndarray]


@dataclass(frozen=True)
class WindowState:
    time: float
    lnOmega: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    lnOmega_shift: float = 0.0

    @property
    def absolute_lnOmega(self):
        """Log-weights with the solver's cumulative shared real shift restored."""

        return np.asarray(self.lnOmega) + float(self.lnOmega_shift)

    def measurement_snapshot(self, measurement_names: Optional[Sequence[str]] = None):
        if measurement_names is None:
            measurements, errors = compute_equal_time_measurements(
                jnp.asarray(self.lnOmega),
                jnp.asarray(self.alpha),
                jnp.asarray(self.beta),
            )
        else:
            measurements, errors = evaluate_measurements_with_error(
                jnp.asarray(self.lnOmega),
                jnp.asarray(self.alpha),
                jnp.asarray(self.beta),
                measurement_names=measurement_names,
            )
        return MeasurementSnapshot(
            measurements={name: np.asarray(value) for name, value in measurements.items()},
            measurement_errors={name: np.asarray(value) for name, value in errors.items()},
        )

    def normalized_contributions(self, observable_samples, eps: float = 1e-12):
        return normalized_weighted_contributions(self.lnOmega, observable_samples, eps=eps)

    def alpha_site_contributions(self, site: int, eps: float = 1e-12):
        return alpha_site_contributions(self.lnOmega, self.alpha, site=site, eps=eps)

    def histogram_alpha_contribution(
        self,
        site: int,
        bins=100,
        density: bool = False,
        real_range=None,
        imag_range=None,
        abs_range=None,
    ):
        return histogram_alpha_contribution(
            self.lnOmega,
            self.alpha,
            site=site,
            bins=bins,
            density=density,
            real_range=real_range,
            imag_range=imag_range,
            abs_range=abs_range,
        )

    def weighted_magnitude_histogram(
        self,
        quantity: str,
        range_min: float,
        range_max: float,
        n_bins: int = 100,
        scale: str = "linear",
        site: Optional[int] = None,
        window_index: Optional[int] = None,
        eps: float = 1e-12,
    ):
        return weighted_magnitude_histogram(
            self.lnOmega,
            self.alpha,
            self.beta,
            quantity=quantity,
            range_min=range_min,
            range_max=range_max,
            n_bins=n_bins,
            scale=scale,
            site=site,
            time=self.time,
            window_index=window_index,
            eps=eps,
        )

    def weighted_complex_histogram_2d(
        self,
        quantity: str,
        real_range,
        imag_range,
        n_bins=100,
        density: bool = False,
        site: Optional[int] = None,
        window_index: Optional[int] = None,
        eps: float = 1e-12,
    ):
        return weighted_complex_histogram_2d(
            self.lnOmega,
            self.alpha,
            self.beta,
            quantity=quantity,
            real_range=real_range,
            imag_range=imag_range,
            n_bins=n_bins,
            density=density,
            site=site,
            time=self.time,
            window_index=window_index,
            eps=eps,
        )


@dataclass(frozen=True)
class MeasurementHistory:
    measurements: Dict[str, np.ndarray]
    measurement_errors: Dict[str, np.ndarray]


@dataclass(frozen=True)
class WeightedMagnitudeHistogram:
    quantity: str
    density: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    range_min: float
    range_max: float
    n_bins: int
    scale_mode: str
    x_scale: str
    y_scale: str
    x_label: str
    y_label: str
    time: Optional[float]
    window_index: Optional[int]
    site: Optional[int]
    sample_count: int


@dataclass(frozen=True)
class WeightedComplexHistogram2D:
    quantity: str
    density: np.ndarray
    real_edges: np.ndarray
    imag_edges: np.ndarray
    real_centers: np.ndarray
    imag_centers: np.ndarray
    real_range: Tuple[float, float]
    imag_range: Tuple[float, float]
    n_bins_real: int
    n_bins_imag: int
    x_label: str
    y_label: str
    time: Optional[float]
    window_index: Optional[int]
    site: Optional[int]
    sample_count: int


@jax.jit
def safe_complex_denominator(denom: jnp.ndarray, eps: float = 1e-12):
    """Floor a complex denominator's magnitude without changing its phase.

    Adding a positive epsilon directly to a complex denominator is not a safe
    regularization: for a small negative-real denominator it can reduce the
    magnitude or cancel the denominator exactly.  Values below the floor are
    instead projected radially onto ``abs(z) == eps``.  The phase at exactly
    zero is defined to be ``+1``.
    """

    denom = jnp.asarray(denom)
    eps = jnp.asarray(eps, dtype=jnp.real(denom).dtype)
    denom_abs = jnp.abs(denom)
    nonzero = denom_abs > jnp.asarray(0.0, dtype=eps.dtype)
    safe_abs = jnp.where(
        nonzero,
        denom_abs,
        jnp.asarray(1.0, dtype=eps.dtype),
    )
    unit = jnp.where(
        nonzero,
        denom / safe_abs,
        jnp.asarray(1.0, dtype=denom.dtype),
    )
    return jnp.where(denom_abs >= eps, denom, eps * unit)


@partial(jax.jit, static_argnames=("axis_name",))
def max_real_log_weight_for_axis(lnOmega: jnp.ndarray, axis_name=None):
    """Return the stopped global maximum real log-weight."""

    # Stop before the collective: JAX has no AD rule for pmax, so stopping only
    # its output still makes value_and_grad attempt to differentiate the pmax.
    center = lax.stop_gradient(jnp.max(jnp.real(jnp.asarray(lnOmega))))
    if axis_name is not None:
        center = lax.pmax(center, axis_name)
    return center


@partial(jax.jit, static_argnames=("axis_name",))
def centered_exponentiated_weights_for_axis(
    lnOmega: jnp.ndarray,
    eps: float = 1e-12,
    axis_name=None,
):
    """Exponentiate log-weights after a stable global max subtraction."""

    del eps
    center = max_real_log_weight_for_axis(lnOmega, axis_name=axis_name)
    return jnp.exp(lnOmega - center)


@jax.jit
def centered_exponentiated_weights(lnOmega: jnp.ndarray, eps: float = 1e-12):
    # The stored log-weight history is only defined up to a common real shift.
    # Max-centering fixes a stable numerical convention; any self-normalized
    # weighted moment is invariant under this shared rescaling.
    return centered_exponentiated_weights_for_axis(
        lnOmega,
        eps=eps,
        axis_name=None,
    )


@jax.jit
def self_normalized_weight_ratio(lnOmega: jnp.ndarray, eps: float = 1e-12):
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    denom = safe_complex_denominator(jnp.mean(Omega_centered), eps=eps)
    return Omega_centered / denom


@jax.jit
def weighted_mean_complex(x, w, eps=1e-12):
    weight_shape = (w.shape[0],) + (1,) * (x.ndim - 1)
    num = jnp.sum(w.reshape(weight_shape) * x, axis=0)
    denom = safe_complex_denominator(jnp.sum(w), eps=eps)
    return num / denom


@jax.jit
def weighted_outer_mean_complex(x_left, x_right, w, eps=1e-12):
    num = jnp.einsum("w,wi,wj->ij", w, x_left, x_right)
    denom = safe_complex_denominator(jnp.sum(w), eps=eps)
    return num / denom


@jax.jit
def weighted_mean_contributions_complex(x, w, eps=1e-12):
    weight_shape = (w.shape[0],) + (1,) * (x.ndim - 1)
    denom = safe_complex_denominator(jnp.sum(w), eps=eps)
    return w.reshape(weight_shape) * x / denom


@jax.jit
def weighted_outer_contributions_complex(x_left, x_right, w, eps=1e-12):
    denom = safe_complex_denominator(jnp.sum(w), eps=eps)
    return w[:, None, None] * x_left[:, :, None] * x_right[:, None, :] / denom


@jax.jit
def contribution_sum_stderr_real(contributions: jnp.ndarray):
    if contributions.shape[0] <= 1:
        return jnp.zeros(contributions.shape[1:], dtype=DTYPE)
    variance = jnp.var(jnp.asarray(contributions, dtype=DTYPE), axis=0, ddof=1)
    return jnp.sqrt(jnp.maximum(variance, 0.0) * DTYPE(contributions.shape[0]))


@jax.jit
def contribution_sum_stderr_complex(contributions: jnp.ndarray):
    err_real = contribution_sum_stderr_real(jnp.real(contributions))
    err_imag = contribution_sum_stderr_real(jnp.imag(contributions))
    return err_real.astype(CDTYPE) + 1j * err_imag.astype(CDTYPE)


@jax.jit
def sample_mean_stderr_real(samples: jnp.ndarray):
    if samples.shape[0] <= 1:
        return jnp.zeros(samples.shape[1:], dtype=DTYPE)
    variance = jnp.var(jnp.asarray(samples, dtype=DTYPE), axis=0, ddof=1)
    sample_count = jnp.asarray(samples.shape[0], dtype=DTYPE)
    return jnp.sqrt(jnp.maximum(variance, 0.0) / sample_count)


@jax.jit
def sample_mean_stderr_complex(samples: jnp.ndarray):
    err_real = sample_mean_stderr_real(jnp.real(samples))
    err_imag = sample_mean_stderr_real(jnp.imag(samples))
    return err_real.astype(CDTYPE) + 1j * err_imag.astype(CDTYPE)


@jax.jit
def sample_mean_complex_with_error(samples: jnp.ndarray):
    mean = jnp.mean(samples, axis=0)
    err = sample_mean_stderr_complex(samples)
    return mean, err


@jax.jit
def weighted_ratio_mean_complex_with_error(numerator_samples, denominator_samples, eps=1e-12):
    """
    Delta-method standard error for a self-normalized ratio estimator.

    If O = mean(X) / mean(Y), then the estimator uses the sample mean of the
    influence variable

        (X_i - O Y_i) / mean(Y),

    which is the CLT / first-order delta-method extension of the sample-mean
    uncertainty estimate in Deuar and Drummond (2006), Eq. (65), to weighted
    gauge-P observables with a fluctuating denominator.
    """

    sample_count = numerator_samples.shape[0]
    denom_mean = jnp.mean(denominator_samples, axis=0)
    denom_safe = safe_complex_denominator(denom_mean, eps=eps)
    numerator_mean = jnp.mean(numerator_samples, axis=0)
    mean = numerator_mean / denom_safe

    expand_shape = (sample_count,) + (1,) * (numerator_samples.ndim - 1)
    denominator_broadcast = denominator_samples.reshape(expand_shape)
    influence = (numerator_samples - mean * denominator_broadcast) / denom_safe
    err = sample_mean_stderr_complex(influence)
    return mean, err


@jax.jit
def weighted_mean_complex_with_error(x, w, eps=1e-12):
    weight_shape = (w.shape[0],) + (1,) * (x.ndim - 1)
    numerator_samples = w.reshape(weight_shape) * x
    return weighted_ratio_mean_complex_with_error(numerator_samples, w, eps=eps)


@jax.jit
def weighted_outer_mean_complex_with_error(x_left, x_right, w, eps=1e-12):
    numerator_samples = w[:, None, None] * x_left[:, :, None] * x_right[:, None, :]
    return weighted_ratio_mean_complex_with_error(numerator_samples, w, eps=eps)


@jax.jit
def shell_pair_average_samples(x_left, x_right, left_indices, right_indices):
    """Average selected directed pair samples without materializing a walker-pair matrix."""

    x_left = jnp.asarray(x_left)
    x_right = jnp.asarray(x_right)
    left_indices = jnp.asarray(left_indices, dtype=jnp.int32)
    right_indices = jnp.asarray(right_indices, dtype=jnp.int32)
    pair_count = int(left_indices.shape[0])
    if pair_count == 0:
        return jnp.full((x_left.shape[0],), jnp.nan + 0.0j, dtype=x_left.dtype)

    def body(edge_index, acc):
        left = left_indices[edge_index]
        right = right_indices[edge_index]
        return acc + x_left[:, left] * x_right[:, right]

    total = lax.fori_loop(
        0,
        pair_count,
        body,
        jnp.zeros((x_left.shape[0],), dtype=x_left.dtype),
    )
    return total / jnp.asarray(pair_count, dtype=x_left.real.dtype)


@jax.jit
def weighted_shell_pair_mean_complex_with_error(
    x_left,
    x_right,
    w,
    left_indices,
    right_indices,
    eps=1e-12,
):
    samples = shell_pair_average_samples(x_left, x_right, left_indices, right_indices)
    return weighted_mean_complex_with_error(samples, w, eps=eps)


@jax.jit
def _compute_first_moment_observables_from_centered_weights(
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    Omega_centered: jnp.ndarray,
    eps: float = 1e-12,
):
    alpha_sq = alpha**2
    beta_sq = beta**2
    alpha2beta = alpha_sq * beta
    alphabeta2 = alpha * beta_sq
    return {
        "A": weighted_mean_complex(alpha, Omega_centered, eps=eps),
        "B": weighted_mean_complex(beta, Omega_centered, eps=eps),
        "na": weighted_mean_complex(alpha2beta, Omega_centered, eps=eps),
        "nb": weighted_mean_complex(alphabeta2, Omega_centered, eps=eps),
    }


@jax.jit
def compute_first_moment_observables(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    return _compute_first_moment_observables_from_centered_weights(
        alpha=alpha,
        beta=beta,
        Omega_centered=Omega_centered,
        eps=eps,
    )


@jax.jit
def compute_density_observables(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    return {
        "A": weighted_mean_complex(alpha, Omega_centered, eps=eps),
        "B": weighted_mean_complex(beta, Omega_centered, eps=eps),
        "G1": weighted_outer_mean_complex(beta, alpha, Omega_centered, eps=eps),
    }


@jax.jit
def compute_operator_observables(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    first_observables = _compute_first_moment_observables_from_centered_weights(
        alpha=alpha,
        beta=beta,
        Omega_centered=Omega_centered,
        eps=eps,
    )
    alpha_sq = alpha**2
    beta_sq = beta**2
    alpha2beta = alpha_sq * beta
    alphabeta2 = alpha * beta_sq
    return {
        **first_observables,
        "G1": weighted_outer_mean_complex(beta, alpha, Omega_centered, eps=eps),
        "Cp": weighted_outer_mean_complex(alphabeta2, alpha, Omega_centered, eps=eps),
        "Cm": weighted_outer_mean_complex(beta, alpha2beta, Omega_centered, eps=eps),
    }


@jax.jit
def safe_real_observable_denominator(x: jnp.ndarray, eps: float = 1e-12):
    eps = jnp.asarray(eps, dtype=DTYPE)
    return jnp.maximum(jnp.real(x).astype(DTYPE), eps)


@jax.jit
def safe_real_observable_ratio(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    eps: float = 1e-12,
    occupation_floor: float = OBSERVABLE_OCCUPATION_FLOOR,
):
    denom_real = jnp.real(denominator).astype(DTYPE)
    denom_safe = safe_real_observable_denominator(denominator, eps=eps)
    occupation_floor = jnp.asarray(occupation_floor, dtype=DTYPE)
    ratio = jnp.asarray(numerator, dtype=DTYPE) / denom_safe
    nan_value = jnp.asarray(jnp.nan, dtype=DTYPE)
    return jnp.where(denom_real >= occupation_floor, ratio, nan_value)


@jax.jit
def propagated_real_ratio_error(
    ratio: jnp.ndarray,
    numerator: jnp.ndarray,
    numerator_err: jnp.ndarray,
    denominator: jnp.ndarray,
    denominator_err: jnp.ndarray,
    denominator_power: float = 1.0,
    eps: float = 1e-12,
    occupation_floor: float = OBSERVABLE_OCCUPATION_FLOOR,
):
    ratio = jnp.asarray(ratio, dtype=DTYPE)
    numerator = jnp.asarray(numerator, dtype=DTYPE)
    numerator_err = jnp.asarray(numerator_err, dtype=DTYPE)
    denominator_real = jnp.asarray(denominator, dtype=DTYPE)
    denominator_err = jnp.asarray(denominator_err, dtype=DTYPE)
    denom_safe = safe_real_observable_denominator(denominator_real, eps=eps)
    num_safe = jnp.maximum(jnp.abs(numerator), jnp.asarray(eps, dtype=DTYPE))
    relative_num = numerator_err / num_safe
    relative_den = jnp.asarray(denominator_power, dtype=DTYPE) * denominator_err / denom_safe
    err = jnp.abs(ratio) * jnp.sqrt(relative_num**2 + relative_den**2)
    occupation_floor = jnp.asarray(occupation_floor, dtype=DTYPE)
    nan_value = jnp.asarray(jnp.nan, dtype=DTYPE)
    return jnp.where(denominator_real >= occupation_floor, err, nan_value)


@jax.jit
def reliable_density_mask(
    density: jnp.ndarray,
    density_err: Optional[jnp.ndarray] = None,
    occupation_floor: float = OBSERVABLE_OCCUPATION_FLOOR,
    sigma_threshold: float = 3.0,
):
    density_real = jnp.asarray(density, dtype=DTYPE)
    floor_ok = density_real >= jnp.asarray(occupation_floor, dtype=DTYPE)
    if density_err is None:
        return floor_ok
    density_err = jnp.maximum(jnp.asarray(density_err, dtype=DTYPE), jnp.asarray(0.0, dtype=DTYPE))
    resolved_ok = density_real >= jnp.asarray(sigma_threshold, dtype=DTYPE) * density_err
    return floor_ok & resolved_ok


@jax.jit
def propagated_real_product_ratio_error(
    ratio: jnp.ndarray,
    numerator: jnp.ndarray,
    numerator_err: jnp.ndarray,
    left_denominator: jnp.ndarray,
    left_denominator_err: jnp.ndarray,
    right_denominator: jnp.ndarray,
    right_denominator_err: jnp.ndarray,
    eps: float = 1e-12,
    occupation_floor: float = OBSERVABLE_OCCUPATION_FLOOR,
):
    ratio = jnp.asarray(ratio, dtype=DTYPE)
    numerator = jnp.asarray(numerator, dtype=DTYPE)
    numerator_err = jnp.asarray(numerator_err, dtype=DTYPE)
    left_denominator = jnp.asarray(left_denominator, dtype=DTYPE)
    left_denominator_err = jnp.asarray(left_denominator_err, dtype=DTYPE)
    right_denominator = jnp.asarray(right_denominator, dtype=DTYPE)
    right_denominator_err = jnp.asarray(right_denominator_err, dtype=DTYPE)

    left_safe = safe_real_observable_denominator(left_denominator, eps=eps)
    right_safe = safe_real_observable_denominator(right_denominator, eps=eps)
    num_safe = jnp.maximum(jnp.abs(numerator), jnp.asarray(eps, dtype=DTYPE))
    relative_num = numerator_err / num_safe
    relative_left = left_denominator_err / left_safe
    relative_right = right_denominator_err / right_safe
    err = jnp.abs(ratio) * jnp.sqrt(
        relative_num**2 + relative_left[:, None] ** 2 + relative_right[None, :] ** 2
    )
    floor = jnp.asarray(occupation_floor, dtype=DTYPE)
    reliable = (left_denominator[:, None] >= floor) & (right_denominator[None, :] >= floor)
    nan_value = jnp.asarray(jnp.nan, dtype=DTYPE)
    return jnp.where(reliable, err, nan_value)


@jax.jit
def compute_equal_time_measurements(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    A, A_err = weighted_mean_complex_with_error(alpha, Omega_centered, eps=eps)
    B, B_err = weighted_mean_complex_with_error(beta, Omega_centered, eps=eps)
    G1, G1_err = weighted_outer_mean_complex_with_error(beta, alpha, Omega_centered, eps=eps)
    density_samples = beta * alpha
    G2, G2_err = weighted_outer_mean_complex_with_error(density_samples, density_samples, Omega_centered, eps=eps)
    density = jnp.diag(G1)
    density_err = jnp.diag(G1_err)
    N = jnp.real(density).astype(DTYPE)
    N_err = jnp.real(density_err).astype(DTYPE)
    G2_local = jnp.real(jnp.diag(G2)).astype(DTYPE)
    G2_local_err = jnp.real(jnp.diag(G2_err)).astype(DTYPE)
    N_safe = safe_real_observable_denominator(density, eps=eps)
    g2 = safe_real_observable_ratio(
        numerator=jnp.real(G2).astype(DTYPE),
        denominator=N_safe[:, None] * N_safe[None, :],
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
    )
    g2_err = propagated_real_product_ratio_error(
        ratio=g2,
        numerator=jnp.real(G2).astype(DTYPE),
        numerator_err=jnp.real(G2_err).astype(DTYPE),
        left_denominator=N,
        left_denominator_err=N_err,
        right_denominator=N,
        right_denominator_err=N_err,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    g2_local = safe_real_observable_ratio(
        numerator=G2_local,
        denominator=N_safe**2,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
    )
    g2_local_err = propagated_real_ratio_error(
        ratio=g2_local,
        numerator=G2_local,
        numerator_err=G2_local_err,
        denominator=N,
        denominator_err=N_err,
        denominator_power=2.0,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    reliable_density = reliable_density_mask(
        N,
        density_err=N_err,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
        sigma_threshold=3.0,
    )
    nan_real = jnp.asarray(jnp.nan, dtype=DTYPE)
    reliable_density_pair = reliable_density[:, None] & reliable_density[None, :]
    g2 = jnp.where(reliable_density_pair, g2, nan_real)
    g2_err = jnp.where(reliable_density_pair, g2_err, nan_real)
    g2_local = jnp.where(reliable_density, g2_local, nan_real)
    g2_local_err = jnp.where(reliable_density, g2_local_err, nan_real)
    A_abs_sq = (jnp.abs(A) ** 2).astype(DTYPE)
    A_abs_sq_err = jnp.sqrt(
        (2.0 * jnp.real(A).astype(DTYPE) * jnp.real(A_err).astype(DTYPE)) ** 2
        + (2.0 * jnp.imag(A).astype(DTYPE) * jnp.imag(A_err).astype(DTYPE)) ** 2
    )
    coherence_fraction = safe_real_observable_ratio(
        numerator=A_abs_sq,
        denominator=N_safe,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    coherence_fraction_err = propagated_real_ratio_error(
        ratio=coherence_fraction,
        numerator=A_abs_sq,
        numerator_err=A_abs_sq_err,
        denominator=N,
        denominator_err=N_err,
        denominator_power=1.0,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    coherence_fraction = jnp.where(reliable_density, coherence_fraction, nan_real)
    coherence_fraction_err = jnp.where(reliable_density, coherence_fraction_err, nan_real)
    measurements = {
        "A": A,
        "B": B,
        "density": density,
        "N": N,
        "G1": G1,
        "G2": G2,
        "g2": g2,
        "G2_local": G2_local,
        "g2_local": g2_local,
        "coherence_fraction": coherence_fraction,
    }
    measurement_errors = {
        "A": A_err,
        "B": B_err,
        "density": density_err,
        "N": N_err,
        "G1": G1_err,
        "G2": G2_err,
        "g2": g2_err,
        "G2_local": G2_local_err,
        "g2_local": g2_local_err,
        "coherence_fraction": coherence_fraction_err,
    }
    return measurements, measurement_errors


@jax.jit
def compute_reduced_equal_time_measurements(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    nn_left_indices: jnp.ndarray,
    nn_right_indices: jnp.ndarray,
    nnn_left_indices: jnp.ndarray,
    nnn_right_indices: jnp.ndarray,
    eps: float = 1e-12,
):
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    A, A_err = weighted_mean_complex_with_error(alpha, Omega_centered, eps=eps)
    B, B_err = weighted_mean_complex_with_error(beta, Omega_centered, eps=eps)
    density_samples = beta * alpha
    density, density_err = weighted_mean_complex_with_error(density_samples, Omega_centered, eps=eps)
    G2_local, G2_local_err = weighted_mean_complex_with_error(
        density_samples * density_samples,
        Omega_centered,
        eps=eps,
    )
    N = jnp.real(density).astype(DTYPE)
    N_err = jnp.real(density_err).astype(DTYPE)
    G2_local_real = jnp.real(G2_local).astype(DTYPE)
    G2_local_err_real = jnp.real(G2_local_err).astype(DTYPE)
    N_safe = safe_real_observable_denominator(N, eps=eps)
    g2_local = safe_real_observable_ratio(
        numerator=G2_local_real,
        denominator=N_safe**2,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
    )
    g2_local_err = propagated_real_ratio_error(
        ratio=g2_local,
        numerator=G2_local_real,
        numerator_err=G2_local_err_real,
        denominator=N,
        denominator_err=N_err,
        denominator_power=2.0,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    reliable = reliable_density_mask(
        N,
        density_err=N_err,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
        sigma_threshold=3.0,
    )
    nan_real = jnp.asarray(jnp.nan, dtype=DTYPE)
    g2_local = jnp.where(reliable, g2_local, nan_real)
    g2_local_err = jnp.where(reliable, g2_local_err, nan_real)
    A_abs_sq = (jnp.abs(A) ** 2).astype(DTYPE)
    A_abs_sq_err = jnp.sqrt(
        (2.0 * jnp.real(A).astype(DTYPE) * jnp.real(A_err).astype(DTYPE)) ** 2
        + (2.0 * jnp.imag(A).astype(DTYPE) * jnp.imag(A_err).astype(DTYPE)) ** 2
    )
    coherence_fraction = safe_real_observable_ratio(
        numerator=A_abs_sq,
        denominator=N_safe,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    coherence_fraction_err = propagated_real_ratio_error(
        ratio=coherence_fraction,
        numerator=A_abs_sq,
        numerator_err=A_abs_sq_err,
        denominator=N,
        denominator_err=N_err,
        denominator_power=1.0,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    coherence_fraction = jnp.where(reliable, coherence_fraction, nan_real)
    coherence_fraction_err = jnp.where(reliable, coherence_fraction_err, nan_real)

    def shell_values(left_indices, right_indices):
        G1_shell, G1_shell_err = weighted_shell_pair_mean_complex_with_error(
            beta,
            alpha,
            Omega_centered,
            left_indices,
            right_indices,
            eps=eps,
        )
        G2_shell, G2_shell_err = weighted_shell_pair_mean_complex_with_error(
            density_samples,
            density_samples,
            Omega_centered,
            left_indices,
            right_indices,
            eps=eps,
        )
        pair_count = int(left_indices.shape[0])
        if pair_count == 0:
            nan_complex = jnp.asarray(jnp.nan + 0.0j, dtype=CDTYPE)
            return nan_complex, nan_complex, nan_real, nan_real
        N_left = N[left_indices]
        N_right = N[right_indices]
        N_left_err = N_err[left_indices]
        N_right_err = N_err[right_indices]
        denom = jnp.mean(N_left * N_right)
        denom_safe = safe_real_observable_denominator(denom, eps=eps)
        G2_shell_real = jnp.real(G2_shell).astype(DTYPE)
        G2_shell_err_real = jnp.real(G2_shell_err).astype(DTYPE)
        g2_shell = safe_real_observable_ratio(
            numerator=G2_shell_real,
            denominator=denom_safe,
            eps=eps,
            occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
        )
        denom_terms = (N_right * N_left_err) ** 2 + (N_left * N_right_err) ** 2
        denom_err = jnp.sqrt(jnp.maximum(jnp.mean(denom_terms), 0.0) / jnp.asarray(pair_count, dtype=DTYPE))
        g2_shell_err = jnp.abs(g2_shell) * jnp.sqrt(
            (G2_shell_err_real / jnp.maximum(jnp.abs(G2_shell_real), jnp.asarray(eps, dtype=DTYPE))) ** 2
            + (denom_err / denom_safe) ** 2
        )
        reliable_shell = denom >= jnp.asarray(OBSERVABLE_OCCUPATION_FLOOR**2, dtype=DTYPE)
        g2_shell = jnp.where(reliable_shell, g2_shell, nan_real)
        g2_shell_err = jnp.where(reliable_shell, g2_shell_err, nan_real)
        return G1_shell, G1_shell_err, g2_shell, g2_shell_err

    G1_nn, G1_nn_err, g2_nn, g2_nn_err = shell_values(nn_left_indices, nn_right_indices)
    G1_nnn, G1_nnn_err, g2_nnn, g2_nnn_err = shell_values(nnn_left_indices, nnn_right_indices)
    measurements = {
        "A": A,
        "B": B,
        "density": density,
        "N": N,
        "G1_local": density,
        "G2_local": G2_local_real,
        "g2_local": g2_local,
        "coherence_fraction": coherence_fraction,
        "G1_nn": G1_nn,
        "G1_nnn": G1_nnn,
        "g2_nn": g2_nn,
        "g2_nnn": g2_nnn,
    }
    measurement_errors = {
        "A": A_err,
        "B": B_err,
        "density": density_err,
        "N": N_err,
        "G1_local": density_err,
        "G2_local": G2_local_err_real,
        "g2_local": g2_local_err,
        "coherence_fraction": coherence_fraction_err,
        "G1_nn": G1_nn_err,
        "G1_nnn": G1_nnn_err,
        "g2_nn": g2_nn_err,
        "g2_nnn": g2_nnn_err,
    }
    return measurements, measurement_errors


@jax.jit
def compute_local_equal_time_measurements(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    """Onsite equal-time observables without any shell-pair work."""

    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    density_samples = beta * alpha
    density, density_err = weighted_mean_complex_with_error(density_samples, Omega_centered, eps=eps)
    G2_local, G2_local_err = weighted_mean_complex_with_error(
        density_samples * density_samples,
        Omega_centered,
        eps=eps,
    )
    N = jnp.real(density).astype(DTYPE)
    N_err = jnp.real(density_err).astype(DTYPE)
    G2_local_real = jnp.real(G2_local).astype(DTYPE)
    G2_local_err_real = jnp.real(G2_local_err).astype(DTYPE)
    N_safe = safe_real_observable_denominator(N, eps=eps)
    g2_local = safe_real_observable_ratio(
        numerator=G2_local_real,
        denominator=N_safe**2,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
    )
    g2_local_err = propagated_real_ratio_error(
        ratio=g2_local,
        numerator=G2_local_real,
        numerator_err=G2_local_err_real,
        denominator=N,
        denominator_err=N_err,
        denominator_power=2.0,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    reliable = reliable_density_mask(
        N,
        density_err=N_err,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
        sigma_threshold=3.0,
    )
    nan_real = jnp.asarray(jnp.nan, dtype=DTYPE)
    g2_local = jnp.where(reliable, g2_local, nan_real)
    g2_local_err = jnp.where(reliable, g2_local_err, nan_real)
    measurements = {
        "density": density,
        "N": N,
        "G1_local": density,
        "G2_local": G2_local_real,
        "g2_local": g2_local,
    }
    measurement_errors = {
        "density": density_err,
        "N": N_err,
        "G1_local": density_err,
        "G2_local": G2_local_err_real,
        "g2_local": g2_local_err,
    }
    return measurements, measurement_errors


@jax.jit
def compute_local_equal_time_measurements_no_error(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    """Mean-only onsite equal-time observables."""

    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    density_samples = beta * alpha
    density = weighted_mean_complex(density_samples, Omega_centered, eps=eps)
    G2_local = weighted_mean_complex(density_samples * density_samples, Omega_centered, eps=eps)
    N = jnp.real(density).astype(DTYPE)
    G2_local_real = jnp.real(G2_local).astype(DTYPE)
    N_safe = safe_real_observable_denominator(N, eps=eps)
    g2_local = safe_real_observable_ratio(
        numerator=G2_local_real,
        denominator=N_safe**2,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
    )
    reliable = N >= jnp.asarray(OBSERVABLE_OCCUPATION_FLOOR, dtype=DTYPE)
    g2_local = jnp.where(reliable, g2_local, jnp.asarray(jnp.nan, dtype=DTYPE))
    return {
        "density": density,
        "N": N,
        "G1_local": density,
        "G2_local": G2_local_real,
        "g2_local": g2_local,
    }


@jax.jit
def compute_initial_time_measurements(
    initial_beta: jnp.ndarray,
    initial_alpha: jnp.ndarray,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eps: float = 1e-12,
):
    """
    Initial-to-current field correlation in the gauge/positive-P convention.

    This uses the saved initial walker values beta_i(t0) together with the
    current alpha_j(t),

        G1_initial_ij(t) = < beta_i(t0) alpha_j(t) >_Omega,

    which reduces to the old single-mode `field_G1` convention for one site.
    """

    initial_beta = jnp.asarray(initial_beta, dtype=alpha.dtype)
    initial_alpha = jnp.asarray(initial_alpha, dtype=alpha.dtype)
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    G1_initial, G1_initial_err = weighted_outer_mean_complex_with_error(
        initial_beta,
        alpha,
        Omega_centered,
        eps=eps,
    )
    initial_density = initial_beta * initial_alpha
    current_density = beta * alpha
    G2_initial, G2_initial_err = weighted_outer_mean_complex_with_error(
        initial_density,
        current_density,
        Omega_centered,
        eps=eps,
    )
    N_initial_complex, N_initial_complex_err = weighted_mean_complex_with_error(
        initial_density,
        Omega_centered,
        eps=eps,
    )
    N_current_complex, N_current_complex_err = weighted_mean_complex_with_error(
        current_density,
        Omega_centered,
        eps=eps,
    )
    N_initial = jnp.real(N_initial_complex).astype(DTYPE)
    N_initial_err = jnp.real(N_initial_complex_err).astype(DTYPE)
    N_current = jnp.real(N_current_complex).astype(DTYPE)
    N_current_err = jnp.real(N_current_complex_err).astype(DTYPE)
    N_initial_safe = safe_real_observable_denominator(N_initial, eps=eps)
    N_current_safe = safe_real_observable_denominator(N_current, eps=eps)
    g2_initial = safe_real_observable_ratio(
        numerator=jnp.real(G2_initial).astype(DTYPE),
        denominator=N_initial_safe[:, None] * N_current_safe[None, :],
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR**2,
    )
    g2_initial_err = propagated_real_product_ratio_error(
        ratio=g2_initial,
        numerator=jnp.real(G2_initial).astype(DTYPE),
        numerator_err=jnp.real(G2_initial_err).astype(DTYPE),
        left_denominator=N_initial,
        left_denominator_err=N_initial_err,
        right_denominator=N_current,
        right_denominator_err=N_current_err,
        eps=eps,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
    )
    reliable_initial = reliable_density_mask(
        N_initial,
        density_err=N_initial_err,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
        sigma_threshold=3.0,
    )
    reliable_current = reliable_density_mask(
        N_current,
        density_err=N_current_err,
        occupation_floor=OBSERVABLE_OCCUPATION_FLOOR,
        sigma_threshold=3.0,
    )
    reliable_pair = reliable_initial[:, None] & reliable_current[None, :]
    nan_real = jnp.asarray(jnp.nan, dtype=DTYPE)
    g2_initial = jnp.where(reliable_pair, g2_initial, nan_real)
    g2_initial_err = jnp.where(reliable_pair, g2_initial_err, nan_real)
    measurements = {
        "G1_initial": G1_initial,
        "G2_initial": G2_initial,
        "g2_initial": g2_initial,
    }
    measurement_errors = {
        "G1_initial": G1_initial_err,
        "G2_initial": G2_initial_err,
        "g2_initial": g2_initial_err,
    }
    return measurements, measurement_errors


@jax.jit
def compute_reduced_initial_time_measurements(
    initial_beta: jnp.ndarray,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    nn_left_indices: jnp.ndarray,
    nn_right_indices: jnp.ndarray,
    nnn_left_indices: jnp.ndarray,
    nnn_right_indices: jnp.ndarray,
    eps: float = 1e-12,
):
    initial_beta = jnp.asarray(initial_beta, dtype=alpha.dtype)
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    G1_initial_local, G1_initial_local_err = weighted_mean_complex_with_error(
        initial_beta * alpha,
        Omega_centered,
        eps=eps,
    )
    G1_initial_nn, G1_initial_nn_err = weighted_shell_pair_mean_complex_with_error(
        initial_beta,
        alpha,
        Omega_centered,
        nn_left_indices,
        nn_right_indices,
        eps=eps,
    )
    G1_initial_nnn, G1_initial_nnn_err = weighted_shell_pair_mean_complex_with_error(
        initial_beta,
        alpha,
        Omega_centered,
        nnn_left_indices,
        nnn_right_indices,
        eps=eps,
    )
    measurements = {
        "G1_initial_local": G1_initial_local,
        "G1_initial_nn": G1_initial_nn,
        "G1_initial_nnn": G1_initial_nnn,
    }
    measurement_errors = {
        "G1_initial_local": G1_initial_local_err,
        "G1_initial_nn": G1_initial_nn_err,
        "G1_initial_nnn": G1_initial_nnn_err,
    }
    return measurements, measurement_errors


@jax.jit
def compute_local_initial_time_measurements(
    initial_beta: jnp.ndarray,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    eps: float = 1e-12,
):
    """Onsite initial-to-current field correlation without shell-pair work."""

    initial_beta = jnp.asarray(initial_beta, dtype=alpha.dtype)
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    G1_initial_local, G1_initial_local_err = weighted_mean_complex_with_error(
        initial_beta * alpha,
        Omega_centered,
        eps=eps,
    )
    measurements = {"G1_initial_local": G1_initial_local}
    measurement_errors = {"G1_initial_local": G1_initial_local_err}
    return measurements, measurement_errors


@jax.jit
def compute_local_initial_time_measurements_no_error(
    initial_beta: jnp.ndarray,
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    eps: float = 1e-12,
):
    """Mean-only onsite initial-to-current field correlation."""

    initial_beta = jnp.asarray(initial_beta, dtype=alpha.dtype)
    Omega_centered = centered_exponentiated_weights(lnOmega, eps=eps)
    return {
        "G1_initial_local": weighted_mean_complex(
            initial_beta * alpha,
            Omega_centered,
            eps=eps,
        )
    }


def evaluate_measurements(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    measurement_names: Sequence[str],
):
    available, _ = compute_equal_time_measurements(lnOmega, alpha, beta)
    result = {}
    for name in measurement_names:
        if name not in available:
            raise ValueError(f"Unsupported measurement '{name}'. Supported: {sorted(available)}")
        result[name] = available[name]
    return result


def evaluate_measurements_with_error(
    lnOmega: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    measurement_names: Sequence[str],
):
    available, available_errors = compute_equal_time_measurements(lnOmega, alpha, beta)
    result = {}
    result_errors = {}
    for name in measurement_names:
        if name not in available:
            raise ValueError(f"Unsupported measurement '{name}'. Supported: {sorted(available)}")
        result[name] = available[name]
        result_errors[name] = available_errors[name]
    return result, result_errors


@jax.jit
def _compute_measurement_history_all(
    lnOmega_history: jnp.ndarray,
    alpha_history: jnp.ndarray,
    beta_history: jnp.ndarray,
):
    return jax.vmap(compute_equal_time_measurements, in_axes=(0, 0, 0))(
        lnOmega_history,
        alpha_history,
        beta_history,
    )


@jax.jit
def _compute_initial_time_measurement_history(
    initial_alpha: jnp.ndarray,
    initial_beta: jnp.ndarray,
    lnOmega_history: jnp.ndarray,
    alpha_history: jnp.ndarray,
    beta_history: jnp.ndarray,
):
    return jax.vmap(
        lambda lnOmega_t, alpha_t, beta_t: compute_initial_time_measurements(
            initial_beta,
            initial_alpha,
            lnOmega_t,
            alpha_t,
            beta_t,
        ),
        in_axes=(0, 0, 0),
    )(
        lnOmega_history,
        alpha_history,
        beta_history,
    )


def load_window_end_dataset(path: str):
    return WindowEndDataset.from_path(path)


def compute_measurement_history(
    lnOmega_history,
    alpha_history=None,
    beta_history=None,
    measurement_names: Optional[Sequence[str]] = None,
):
    if isinstance(lnOmega_history, WindowEndDataset):
        dataset = lnOmega_history
        lnOmega_history = dataset.lnOmega_history
        alpha_history = dataset.alpha_history
        beta_history = dataset.beta_history
    elif alpha_history is None or beta_history is None:
        raise ValueError(
            "compute_measurement_history requires either a WindowEndDataset or explicit "
            "(lnOmega_history, alpha_history, beta_history) arrays"
        )

    lnOmega_history = np.asarray(lnOmega_history)
    alpha_history = np.asarray(alpha_history)
    beta_history = np.asarray(beta_history)
    if lnOmega_history.ndim != 2:
        raise ValueError("lnOmega_history must have shape (num_times, num_walker)")
    if alpha_history.ndim != 3 or beta_history.ndim != 3:
        raise ValueError("alpha_history and beta_history must have shape (num_times, num_walker, num_site)")
    if alpha_history.shape[0] != lnOmega_history.shape[0] or beta_history.shape[0] != lnOmega_history.shape[0]:
        raise ValueError("history arrays must share the same leading time dimension")

    measurement_history, measurement_error_history = _compute_measurement_history_all(
        jnp.asarray(lnOmega_history),
        jnp.asarray(alpha_history),
        jnp.asarray(beta_history),
    )
    initial_measurement_names = {"G1_initial", "G2_initial", "g2_initial"}
    if measurement_names is None or any(name in initial_measurement_names for name in measurement_names):
        initial_measurement_history, initial_measurement_error_history = _compute_initial_time_measurement_history(
            jnp.asarray(alpha_history[0]),
            jnp.asarray(beta_history[0]),
            jnp.asarray(lnOmega_history),
            jnp.asarray(alpha_history),
            jnp.asarray(beta_history),
        )
        measurement_history = dict(measurement_history)
        measurement_error_history = dict(measurement_error_history)
        measurement_history.update(initial_measurement_history)
        measurement_error_history.update(initial_measurement_error_history)
    if measurement_names is None:
        selected_names = tuple(measurement_history.keys())
    else:
        selected_names = tuple(measurement_names)
    for name in selected_names:
        if name not in measurement_history:
            raise ValueError(f"Unsupported measurement '{name}'. Supported: {sorted(measurement_history)}")

    return MeasurementHistory(
        measurements={name: np.asarray(measurement_history[name]) for name in selected_names},
        measurement_errors={name: np.asarray(measurement_error_history[name]) for name in selected_names},
    )


def normalized_weighted_contributions(
    lnOmega,
    observable_samples,
    eps: float = 1e-12,
):
    contributions = weighted_mean_contributions_complex(
        jnp.asarray(observable_samples),
        centered_exponentiated_weights(jnp.asarray(lnOmega), eps=eps),
        eps=eps,
    )
    return np.asarray(contributions)


def alpha_site_contributions(
    lnOmega,
    alpha,
    site: int,
    eps: float = 1e-12,
):
    alpha = np.asarray(alpha)
    return normalized_weighted_contributions(lnOmega, alpha[:, int(site)], eps=eps)


def histogram_1d(values, bins=100, value_range=None, density: bool = False):
    samples = np.asarray(values, dtype=float).reshape(-1)
    hist, bin_edges = np.histogram(samples, bins=bins, range=value_range, density=density)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return {
        "hist": hist,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
    }


def _histogram_scale_axes(scale: str):
    scale_key = str(scale).strip().lower()
    allowed = {"linear", "logx", "logy", "logxlogy"}
    if scale_key not in allowed:
        raise ValueError(f"Unsupported histogram scale '{scale}'. Supported: {sorted(allowed)}")
    x_scale = "log" if scale_key in {"logx", "logxlogy"} else "linear"
    y_scale = "log" if scale_key in {"logy", "logxlogy"} else "linear"
    return scale_key, x_scale, y_scale


def _safe_weight_ratio(lnOmega, eps: float = 1e-12):
    """
    Build the self-normalized weight ratio Omega / <Omega> from the stored lnOmega.

    The stored solver history may differ from the raw log-weight by a common real
    shift, so this routine first safely exponentiates lnOmega and then forms the
    invariant ratio Omega / <Omega>.
    """

    ratio = np.asarray(self_normalized_weight_ratio(jnp.asarray(lnOmega), eps=eps))
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError("Self-normalized Omega / <Omega> is numerically unstable for these samples")
    return ratio


def _safe_centered_weight(lnOmega, eps: float = 1e-12):
    centered = np.asarray(centered_exponentiated_weights(jnp.asarray(lnOmega), eps=eps))
    if not np.all(np.isfinite(centered)):
        raise FloatingPointError("Centered exponentiated Omega is numerically unstable for these samples")
    return centered


def _resolve_site_index(site: Optional[int], num_site: int) -> Optional[int]:
    if site is None:
        return None
    site_index = int(site)
    if site_index < 0:
        site_index += int(num_site)
    if site_index < 0 or site_index >= int(num_site):
        raise IndexError(f"site index {site} is out of range for num_site={num_site}")
    return site_index


def _weighted_complex_samples(
    lnOmega,
    alpha,
    beta,
    quantity: str,
    site: Optional[int] = None,
    eps: float = 1e-12,
):
    quantity_key = str(quantity).strip().lower()
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    if alpha.ndim != 2 or beta.ndim != 2:
        raise ValueError("alpha and beta must have shape (num_walker, num_site)")
    if alpha.shape != beta.shape:
        raise ValueError("alpha and beta must share the same shape")

    centered_weight = _safe_centered_weight(lnOmega, eps=eps)
    normalized_weight = _safe_weight_ratio(lnOmega, eps=eps)
    site_index = _resolve_site_index(site, alpha.shape[1])
    if quantity_key == "alpha":
        weighted_samples = centered_weight[:, None] * alpha
        if site_index is None:
            samples = weighted_samples.reshape(-1)
            label_core = r"$\Omega\alpha$"
        else:
            samples = weighted_samples[:, site_index]
            label_core = rf"$\Omega\alpha_{{{site_index}}}$"
    elif quantity_key == "beta":
        weighted_samples = centered_weight[:, None] * beta
        if site_index is None:
            samples = weighted_samples.reshape(-1)
            label_core = r"$\Omega\beta$"
        else:
            samples = weighted_samples[:, site_index]
            label_core = rf"$\Omega\beta_{{{site_index}}}$"
    elif quantity_key == "omega":
        if site_index is not None:
            raise ValueError("quantity='omega' does not support site-resolved selection")
        samples = centered_weight.reshape(-1)
        label_core = r"$\Omega$"
    elif quantity_key == "alpha_omega":
        weighted_samples = normalized_weight[:, None] * alpha
        if site_index is None:
            samples = weighted_samples.reshape(-1)
            label_core = r"$\Omega\alpha / \langle \Omega \rangle$"
        else:
            samples = weighted_samples[:, site_index]
            label_core = rf"$\Omega\alpha_{{{site_index}}} / \langle \Omega \rangle$"
    elif quantity_key == "beta_omega":
        weighted_samples = normalized_weight[:, None] * beta
        if site_index is None:
            samples = weighted_samples.reshape(-1)
            label_core = r"$\Omega\beta / \langle \Omega \rangle$"
        else:
            samples = weighted_samples[:, site_index]
            label_core = rf"$\Omega\beta_{{{site_index}}} / \langle \Omega \rangle$"
    elif quantity_key == "omega_omega":
        if site_index is not None:
            raise ValueError("quantity='omega_omega' does not support site-resolved selection")
        samples = normalized_weight.reshape(-1)
        label_core = r"$\Omega/\langle\Omega\rangle$"
    else:
        raise ValueError(
            "quantity must be one of "
            "['alpha', 'beta', 'omega', 'alpha_omega', 'beta_omega', 'omega_omega']"
        )

    return quantity_key, np.asarray(samples), label_core, site_index


def weighted_magnitude_histogram(
    lnOmega,
    alpha,
    beta,
    *,
    quantity: str,
    range_min: float,
    range_max: float,
    n_bins: int = 100,
    scale: str = "linear",
    site: Optional[int] = None,
    time: Optional[float] = None,
    window_index: Optional[int] = None,
    eps: float = 1e-12,
):
    quantity_key, weighted_samples, label_core, site_index = _weighted_complex_samples(
        lnOmega,
        alpha,
        beta,
        quantity=quantity,
        site=site,
        eps=eps,
    )
    samples = np.abs(weighted_samples).astype(float, copy=False)
    x_label = rf"$|{label_core[1:-1]}|$"
    y_label = rf"$P(|{label_core[1:-1]}|)$"
    scale_mode, x_scale, y_scale = _histogram_scale_axes(scale)

    range_min = float(range_min)
    range_max = float(range_max)
    if not np.isfinite(range_min) or not np.isfinite(range_max):
        raise ValueError("range_min and range_max must be finite")
    if range_max <= range_min:
        raise ValueError("range_max must be strictly greater than range_min")

    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    if x_scale == "log":
        if range_min <= 0.0:
            raise ValueError("log-x histograms require range_min > 0")
        bin_edges = np.logspace(np.log10(range_min), np.log10(range_max), n_bins + 1, dtype=float)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    else:
        bin_edges = np.linspace(range_min, range_max, n_bins + 1, dtype=float)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    density, _ = np.histogram(samples, bins=bin_edges, density=True)

    return WeightedMagnitudeHistogram(
        quantity=quantity_key,
        density=np.asarray(density, dtype=float),
        bin_edges=np.asarray(bin_edges, dtype=float),
        bin_centers=np.asarray(bin_centers, dtype=float),
        range_min=range_min,
        range_max=range_max,
        n_bins=n_bins,
        scale_mode=scale_mode,
        x_scale=x_scale,
        y_scale=y_scale,
        x_label=x_label,
        y_label=y_label,
        time=None if time is None else float(time),
        window_index=None if window_index is None else int(window_index),
        site=site_index,
        sample_count=int(samples.size),
    )


def _parse_histogram_2d_bins(n_bins) -> Tuple[int, int]:
    if isinstance(n_bins, Sequence) and not isinstance(n_bins, (str, bytes)):
        if len(n_bins) != 2:
            raise ValueError("2D histogram n_bins must be an int or a length-2 sequence")
        n_bins_real = int(n_bins[0])
        n_bins_imag = int(n_bins[1])
    else:
        n_bins_real = int(n_bins)
        n_bins_imag = int(n_bins)
    if n_bins_real <= 0 or n_bins_imag <= 0:
        raise ValueError("2D histogram bin counts must be positive")
    return n_bins_real, n_bins_imag


def _validate_histogram_2d_range(name: str, value_range) -> Tuple[float, float]:
    if value_range is None or len(value_range) != 2:
        raise ValueError(f"{name} must be a length-2 range")
    vmin = float(value_range[0])
    vmax = float(value_range[1])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"{name} must contain finite bounds")
    if vmax <= vmin:
        raise ValueError(f"{name} upper bound must be strictly greater than lower bound")
    return vmin, vmax


def weighted_complex_histogram_2d(
    lnOmega,
    alpha,
    beta,
    *,
    quantity: str,
    real_range,
    imag_range,
    n_bins=100,
    density: bool = False,
    site: Optional[int] = None,
    time: Optional[float] = None,
    window_index: Optional[int] = None,
    eps: float = 1e-12,
):
    quantity_key, weighted_samples, label_core, site_index = _weighted_complex_samples(
        lnOmega,
        alpha,
        beta,
        quantity=quantity,
        site=site,
        eps=eps,
    )
    real_min, real_max = _validate_histogram_2d_range("real_range", real_range)
    imag_min, imag_max = _validate_histogram_2d_range("imag_range", imag_range)
    n_bins_real, n_bins_imag = _parse_histogram_2d_bins(n_bins)
    density_grid, real_edges, imag_edges = np.histogram2d(
        np.real(weighted_samples).reshape(-1),
        np.imag(weighted_samples).reshape(-1),
        bins=(n_bins_real, n_bins_imag),
        range=((real_min, real_max), (imag_min, imag_max)),
        density=bool(density),
    )
    real_centers = 0.5 * (real_edges[:-1] + real_edges[1:])
    imag_centers = 0.5 * (imag_edges[:-1] + imag_edges[1:])
    return WeightedComplexHistogram2D(
        quantity=quantity_key,
        density=np.asarray(density_grid, dtype=float),
        real_edges=np.asarray(real_edges, dtype=float),
        imag_edges=np.asarray(imag_edges, dtype=float),
        real_centers=np.asarray(real_centers, dtype=float),
        imag_centers=np.asarray(imag_centers, dtype=float),
        real_range=(real_min, real_max),
        imag_range=(imag_min, imag_max),
        n_bins_real=n_bins_real,
        n_bins_imag=n_bins_imag,
        x_label=rf"$\mathrm{{Re}}({label_core[1:-1]})$",
        y_label=rf"$\mathrm{{Im}}({label_core[1:-1]})$",
        time=None if time is None else float(time),
        window_index=None if window_index is None else int(window_index),
        site=site_index,
        sample_count=int(np.asarray(weighted_samples).size),
    )


def histogram_alpha_contribution(
    lnOmega,
    alpha,
    site: int,
    bins=100,
    density: bool = False,
    real_range=None,
    imag_range=None,
    abs_range=None,
):
    contributions = alpha_site_contributions(lnOmega, alpha, site=site)
    return {
        "contributions": contributions,
        "real": histogram_1d(np.real(contributions), bins=bins, value_range=real_range, density=density),
        "imag": histogram_1d(np.imag(contributions), bins=bins, value_range=imag_range, density=density),
        "abs": histogram_1d(np.abs(contributions), bins=bins, value_range=abs_range, density=density),
    }


__all__ = [
    "MeasurementSnapshot",
    "MeasurementHistory",
    "WeightedComplexHistogram2D",
    "WeightedMagnitudeHistogram",
    "WindowEndDataset",
    "WindowState",
    "alpha_site_contributions",
    "compute_equal_time_measurements",
    "compute_density_observables",
    "compute_first_moment_observables",
    "compute_initial_time_measurements",
    "compute_local_equal_time_measurements",
    "compute_local_equal_time_measurements_no_error",
    "compute_local_initial_time_measurements",
    "compute_local_initial_time_measurements_no_error",
    "compute_measurement_history",
    "compute_operator_observables",
    "compute_reduced_equal_time_measurements",
    "compute_reduced_initial_time_measurements",
    "contribution_sum_stderr_complex",
    "contribution_sum_stderr_real",
    "centered_exponentiated_weights",
    "centered_exponentiated_weights_for_axis",
    "evaluate_measurements",
    "evaluate_measurements_with_error",
    "histogram_1d",
    "histogram_alpha_contribution",
    "weighted_complex_histogram_2d",
    "weighted_magnitude_histogram",
    "load_window_end_dataset",
    "max_real_log_weight_for_axis",
    "normalized_weighted_contributions",
    "propagated_real_ratio_error",
    "safe_complex_denominator",
    "safe_real_observable_denominator",
    "safe_real_observable_ratio",
    "sample_mean_complex_with_error",
    "sample_mean_stderr_complex",
    "sample_mean_stderr_real",
    "self_normalized_weight_ratio",
    "shell_pair_average_samples",
    "weighted_mean_complex",
    "weighted_mean_complex_with_error",
    "weighted_mean_contributions_complex",
    "weighted_outer_contributions_complex",
    "weighted_outer_mean_complex",
    "weighted_outer_mean_complex_with_error",
    "weighted_ratio_mean_complex_with_error",
    "weighted_shell_pair_mean_complex_with_error",
]
