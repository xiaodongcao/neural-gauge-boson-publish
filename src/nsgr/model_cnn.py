from .lib_preinclude import *
from .model import DenseLayer


DEFAULT_CNN_NUM_TAU_ENCODE = 4
DEFAULT_CNN_NUM_FREQUENCY_ENCODE = 4
DEFAULT_CNN_TIME_TAU_RANGE = (1.0e-2, 1.0e1)
DEFAULT_CNN_TIME_W_RANGE = (1.0e-2, 1.0e1)


def _normalize_cnn_time_feature_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key == "none":
        return "no_time"
    if key in {"raw", "raw_time"}:
        return "raw_time"
    if key == "dense":
        return "dense_time"
    if key in {"dampedtrig", "learneddampedtrig"}:
        key = key.replace("dampedtrig", "damped_trig")
    if key in {"dampedtrigtrend", "learneddampedtrigtrend"}:
        key = key.replace("dampedtrigtrend", "damped_trig_trend")
    if key in {"damped_trig", "learned_damped_trig"}:
        return "damped_trig"
    if key in {"damped_trig_trend", "learned_damped_trig_trend"}:
        return "damped_trig_trend"
    raise ValueError(f"Invalid CNN time feature: {name}")


def _inverse_softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("inverse-softplus inputs must be finite and positive")
    return x + np.log(-np.expm1(-x))


def _positive_raw_init(values: Sequence[float], min_value: float, dtype: Any):
    values_np = np.asarray(tuple(values), dtype=np.float64)
    if values_np.ndim != 1 or values_np.size == 0:
        raise ValueError("CNN time initialization values must be a non-empty 1D sequence")
    offset = values_np - float(min_value)
    if np.any(~np.isfinite(offset)) or np.any(offset <= 0.0):
        raise ValueError(
            "CNN time initialization values must be finite and strictly "
            "larger than their configured minimum"
        )
    raw = _inverse_softplus_np(offset)
    return jnp.asarray(raw, dtype=dtype)


def _linear_positive_grid_np(start: float, stop: float, count: int) -> np.ndarray:
    count = int(count)
    if count < 1:
        raise ValueError("time-feature grid counts must be >= 1")
    if stop <= start or start <= 0.0:
        raise ValueError("time-feature grid range must satisfy 0 < start < stop")
    return np.linspace(float(start), float(stop), count, dtype=np.float64)


def _logabs_from_realimag(re, im, eps=1e-12):
    return 0.5 * jnp.log(re * re + im * im + eps)


def _shift_open_axis(x, shift: int, axis: int):
    if shift == 0:
        return x
    if axis == 1:
        if shift > 0:
            return jnp.concatenate([jnp.zeros_like(x[:, :shift]), x[:, :-shift]], axis=1)
        return jnp.concatenate([x[:, -shift:], jnp.zeros_like(x[:, : -shift])], axis=1)
    if shift > 0:
        return jnp.concatenate([jnp.zeros_like(x[:, :, :shift]), x[:, :, :-shift]], axis=2)
    return jnp.concatenate([x[:, :, -shift:], jnp.zeros_like(x[:, :, : -shift])], axis=2)


def _shift_axis(x, shift: int, axis: int, boundary: str):
    if shift == 0:
        return x
    if boundary == "periodic":
        return jnp.roll(x, shift=shift, axis=axis)
    return _shift_open_axis(x, shift, axis)


def _shift_grid(x, shift_x: int, shift_y: int, boundary_x: str, boundary_y: str):
    x = _shift_axis(x, shift_x, axis=1, boundary=boundary_x)
    x = _shift_axis(x, shift_y, axis=2, boundary=boundary_y)
    return x


def _stencil_stack(x, boundary_x: str, boundary_y: str):
    patches = []
    for shift_x in (-1, 0, 1):
        for shift_y in (-1, 0, 1):
            patches.append(_shift_grid(x, shift_x, shift_y, boundary_x, boundary_y))
    return jnp.concatenate(patches, axis=-1)


def _apply_global_modulation(x, scale_shift):
    scale, shift = jnp.split(scale_shift, 2, axis=-1)
    return x * (1.0 + jnp.tanh(scale)) + shift


class LightweightCNNBlock(nn.Module):
    """Small periodic-stencil block for lattice-aware gauge features."""

    model_dim: int
    stencil_dim: int
    boundary_x: str = "periodic"
    boundary_y: str = "periodic"
    mapping_dtype: jnp.dtype = NNDTYPE
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.norm = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.condition_proj = DenseLayer(
            feature_dim=2 * self.model_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.stencil_proj = DenseLayer(
            feature_dim=self.stencil_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.out_proj = DenseLayer(
            feature_dim=self.model_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, h, global_hidden):
        condition = self.condition_proj(global_hidden)[:, None, None, :]
        x = self.norm(h)
        x = _apply_global_modulation(x, condition)
        x = _stencil_stack(x, self.boundary_x, self.boundary_y)
        x = nn.gelu(self.stencil_proj(x))
        return h + nn.gelu(self.out_proj(x))


class GaugeNN(nn.Module):
    """Lightweight CNN/stencil gauge network for regular 2D lattices."""

    num_site: int
    embed_dim: int
    time_embed_dim: int
    feature_dims: Sequence[int]
    use_bias: bool = True
    mapping_dtype: jnp.dtype = NNDTYPE
    kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    drift_max: float = 10.0
    diffusion_max: float = 10.0

    time_feature: str = "Dense"
    num_tau_encode: int = DEFAULT_CNN_NUM_TAU_ENCODE
    num_frequency_encode: int = DEFAULT_CNN_NUM_FREQUENCY_ENCODE
    graph_time_tau_init: Any = None
    graph_time_w_init: Any = None
    graph_time_tau_min: float = 1.0e-3
    graph_time_w_min: float = 1.0e-4

    lattice_shape: Tuple[int, int] = (1, 1)
    boundary_x: str = "periodic"
    boundary_y: str = "periodic"
    num_cnn_layers: int = 2
    bound_drift: bool = False
    bound_diffusion: bool = False

    def setup(self):
        self.cnn_time_feature_mode = _normalize_cnn_time_feature_name(self.time_feature)
        stencil_dim = self.feature_dims[0] if len(self.feature_dims) > 0 else self.embed_dim
        global_hidden_dim = self.feature_dims[1] if len(self.feature_dims) > 1 else max(self.embed_dim // 2, 32)

        if int(self.lattice_shape[0]) * int(self.lattice_shape[1]) != int(self.num_site):
            raise ValueError(
                f"neural_cnn lattice_shape={self.lattice_shape} is incompatible with num_site={self.num_site}"
            )

        if self.cnn_time_feature_mode == "dense_time":
            self.time_dense_proj = DenseLayer(
                feature_dim=self.time_embed_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
        elif self.cnn_time_feature_mode in {"damped_trig", "damped_trig_trend"}:
            tau_init = (
                _linear_positive_grid_np(
                    DEFAULT_CNN_TIME_TAU_RANGE[0],
                    DEFAULT_CNN_TIME_TAU_RANGE[1],
                    self.num_tau_encode,
                )
                if self.graph_time_tau_init is None
                else np.asarray(tuple(self.graph_time_tau_init), dtype=np.float64)
            )
            w_init = (
                _linear_positive_grid_np(
                    DEFAULT_CNN_TIME_W_RANGE[0],
                    DEFAULT_CNN_TIME_W_RANGE[1],
                    self.num_frequency_encode,
                )
                if self.graph_time_w_init is None
                else np.asarray(tuple(self.graph_time_w_init), dtype=np.float64)
            )
            if tau_init.shape != (int(self.num_tau_encode),):
                raise ValueError(
                    "graph_time_tau_init length must match num_tau_encode"
                )
            if w_init.shape != (int(self.num_frequency_encode),):
                raise ValueError(
                    "graph_time_w_init length must match num_frequency_encode"
                )
            self.time_damped_tau_raw = self.param(
                "time_damped_tau_raw",
                lambda key, shape: _positive_raw_init(tau_init, self.graph_time_tau_min, self.mapping_dtype),
                (int(self.num_tau_encode),),
            )
            self.time_w_raw = self.param(
                "time_w_raw",
                lambda key, shape: _positive_raw_init(w_init, self.graph_time_w_min, self.mapping_dtype),
                (int(self.num_frequency_encode),),
            )

        self.local_embed = DenseLayer(
            feature_dim=self.embed_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.input_norm = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.global_proj = DenseLayer(
            feature_dim=global_hidden_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.global_to_site = DenseLayer(
            feature_dim=self.embed_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.cnn_blocks = [
            LightweightCNNBlock(
                model_dim=self.embed_dim,
                stencil_dim=stencil_dim,
                boundary_x=self.boundary_x,
                boundary_y=self.boundary_y,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for _ in range(int(self.num_cnn_layers))
        ]
        self.final_norm = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )

        head_init = nn.initializers.normal(stddev=0.01)
        self.drift_g_head = DenseLayer(
            feature_dim=2,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=head_init,
            bias_init=self.bias_init,
        )
        self.drift_f_head = DenseLayer(
            feature_dim=2,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=head_init,
            bias_init=self.bias_init,
        )
        self.diffusion_head = DenseLayer(
            feature_dim=1,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=head_init,
            bias_init=self.bias_init,
        )

    def _encode_time(self, time, batch_size: int):
        t = jnp.asarray(time, self.mapping_dtype)
        t = jnp.atleast_1d(t)
        if t.shape[0] == 1:
            t = jnp.broadcast_to(t, (batch_size,))
        elif t.shape[0] != batch_size:
            raise ValueError(f"time must be scalar or have shape ({batch_size},), got {t.shape}")
        t = t[:, None]

        if self.cnn_time_feature_mode == "no_time":
            return jnp.zeros((batch_size, 0), dtype=self.mapping_dtype)
        if self.cnn_time_feature_mode == "raw_time":
            return t
        if self.cnn_time_feature_mode == "dense_time":
            return self.time_dense_proj(t)

        damped_tau = (
            jnp.asarray(self.graph_time_tau_min, dtype=self.mapping_dtype)
            + nn.softplus(self.time_damped_tau_raw)[None, :, None]
        )
        angular_freq = (
            jnp.asarray(self.graph_time_w_min, dtype=self.mapping_dtype)
            + nn.softplus(self.time_w_raw)[None, None, :]
        )
        t_grid = t[:, None, None]
        decay = jnp.exp(-t_grid / damped_tau)
        phase = t_grid * angular_freq
        cos_feat = decay * jnp.cos(phase)
        sin_feat = decay * jnp.sin(phase)
        features = [
            cos_feat.reshape(batch_size, -1),
            sin_feat.reshape(batch_size, -1),
        ]
        if self.cnn_time_feature_mode == "damped_trig_trend":
            rise = jnp.asarray(1.0, dtype=self.mapping_dtype) - decay
            rising_cos_feat = rise * jnp.cos(phase)
            rising_sin_feat = rise * jnp.sin(phase)
            features.extend(
                [
                    rising_cos_feat.reshape(batch_size, -1),
                    rising_sin_feat.reshape(batch_size, -1),
                ]
            )
        return jnp.concatenate(features, axis=-1)

    def __call__(self, lnOmega_real, alpha_real, beta_real, time, physical_params):
        """Map walker phase-space states to real-valued gauge fields."""
        del physical_params
        batch_size = alpha_real.shape[0]
        num_site = alpha_real.shape[1] // 2
        if num_site != self.num_site:
            raise ValueError(f"Expected num_site={self.num_site}, but got {num_site} from alpha_real.")

        Nx, Ny = int(self.lattice_shape[0]), int(self.lattice_shape[1])
        lnOmega_real = jnp.asarray(lnOmega_real, dtype=self.mapping_dtype)
        alpha_real = jnp.asarray(alpha_real, dtype=self.mapping_dtype)
        beta_real = jnp.asarray(beta_real, dtype=self.mapping_dtype)

        a_re = alpha_real[:, :num_site]
        a_im = alpha_real[:, num_site:]
        b_re = beta_real[:, :num_site]
        b_im = beta_real[:, num_site:]

        local_feat = jnp.stack(
            [
                a_re,
                a_im,
                b_re,
                b_im,
            ],
            axis=-1,
        ).reshape((batch_size, Nx, Ny, -1))

        time_feat = self._encode_time(time, batch_size)
        global_feat = jnp.concatenate([lnOmega_real, time_feat], axis=-1)
        global_hidden = nn.gelu(self.global_proj(global_feat))
        global_context = nn.gelu(self.global_to_site(global_hidden))[:, None, None, :]

        h = self.local_embed(local_feat)
        h = self.input_norm(h)
        h = nn.gelu(h + global_context)

        for block in self.cnn_blocks:
            h = block(h, global_hidden)

        h = nn.gelu(self.final_norm(h))
        h = h.reshape((batch_size, num_site, -1))

        drift_g_pair = self.drift_g_head(h)
        drift_f_pair = self.drift_f_head(h)
        diffusion_g = jnp.squeeze(self.diffusion_head(h), axis=-1)

        drift_g = jnp.concatenate([drift_g_pair[..., 0], drift_g_pair[..., 1]], axis=-1)
        drift_f = jnp.concatenate([drift_f_pair[..., 0], drift_f_pair[..., 1]], axis=-1)

        if self.bound_drift:
            drift_max = jnp.asarray(self.drift_max, dtype=self.mapping_dtype)
            drift_g = drift_max * jnp.tanh(drift_g / drift_max)
            drift_f = drift_max * jnp.tanh(drift_f / drift_max)

        if self.bound_diffusion:
            diffusion_max = jnp.asarray(self.diffusion_max, dtype=self.mapping_dtype)
            diffusion_g = diffusion_max * jnp.tanh(diffusion_g / diffusion_max)

        return drift_g, drift_f, diffusion_g


__all__ = ["GaugeNN"]
