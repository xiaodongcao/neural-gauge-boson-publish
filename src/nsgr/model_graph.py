from .lib_preinclude import *
from .model import DenseLayer


DEFAULT_GRAPH_NUM_TAU_ENCODE = 4
DEFAULT_GRAPH_NUM_FREQUENCY_ENCODE = 4
DEFAULT_GRAPH_TIME_TAU_RANGE = (1.0e-2, 1.0e1)
DEFAULT_GRAPH_TIME_W_RANGE = (1.0e-2, 1.0e1)


def _normalize_graph_time_feature_name(name: str) -> str:
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
    raise ValueError(f"Invalid graph time feature: {name}")


def _inverse_softplus_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
        raise ValueError("inverse-softplus inputs must be finite and positive")
    return x + np.log(-np.expm1(-x))


def _positive_raw_init(values: Sequence[float], min_value: float, dtype: Any):
    values_np = np.asarray(tuple(values), dtype=np.float64)
    if values_np.ndim != 1 or values_np.size == 0:
        raise ValueError("graph time initialization values must be a non-empty 1D sequence")
    offset = values_np - float(min_value)
    if np.any(~np.isfinite(offset)) or np.any(offset <= 0.0):
        raise ValueError(
            "graph time initialization values must be finite and strictly "
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


@jax.jit
def _logabs_from_realimag(re, im, eps=1e-12):
    # log|z| = 0.5 * log(re^2 + im^2)
    return 0.5 * jnp.log(re * re + im * im + eps)


@jax.jit
def _phase_sincos_from_realimag(re, im, eps=1e-12):
    # stable sin/cos(angle(z)) via z/|z| (avoids arctan2 instability near 0)
    r = jnp.sqrt(re * re + im * im + eps)
    cos = (re / r) 
    sin = (im / r)
    return sin, cos

def _edge_aggregate_messages(x, edge_dst, edge_src, edge_weight, num_site: int):
    """Aggregate weighted neighbor features using a sparse real-valued edge list."""
    agg = jnp.zeros((x.shape[0], num_site, x.shape[-1]), dtype=x.dtype)
    if edge_src.shape[0] == 0:
        return agg
    edge_messages = edge_weight[None, :, None] * x[:, edge_src, :]
    return agg.at[:, edge_dst, :].add(edge_messages)


def _edge_hopping(x, edge_dst, edge_src, edge_weight):
    """Apply the real sparse hopping edge list to batched site vectors."""
    if edge_src.shape[0] == 0:
        return jnp.zeros_like(x)
    messages = edge_weight[None, :].astype(x.dtype) * x[:, edge_src]
    return jnp.zeros_like(x).at[:, edge_dst].add(messages)


def _apply_global_modulation(x, scale_shift):
    scale, shift = jnp.split(scale_shift, 2, axis=-1)
    return x * (1.0 + jnp.tanh(scale)) + shift


def _lnOmega_features(lnOmega_real, dtype):
    """Return raw log-weight features for graph conditioning."""
    return jnp.asarray(lnOmega_real, dtype=dtype)


class LightweightGraphBlock(nn.Module):
    """Lightweight residual graph block for repeated gauge-network application."""

    model_dim: int
    message_dim: int
    mapping_dtype: jnp.dtype = NNDTYPE
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.message_norm = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.message_in = DenseLayer(
            feature_dim=self.message_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.self_proj = DenseLayer(
            feature_dim=self.model_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.message_out = DenseLayer(
            feature_dim=self.model_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.local_norm = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.local_proj = DenseLayer(
            feature_dim=self.model_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.condition_proj = DenseLayer(
            feature_dim=2 * self.model_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, h, global_hidden, global_context, edge_dst, edge_src, edge_weight):
        num_site = h.shape[1]
        context = global_context[:, None, :]
        condition = self.condition_proj(global_hidden)[:, None, :]

        x = self.message_norm(h + context)
        x = _apply_global_modulation(x, condition)
        agg = _edge_aggregate_messages(self.message_in(x), edge_dst, edge_src, edge_weight, num_site)
        msg = self.self_proj(x) + self.message_out(agg)
        h = h + nn.gelu(msg)

        y = self.local_norm(h + context)
        y = _apply_global_modulation(y, condition)
        return h + nn.gelu(self.local_proj(y))


class GaugeNN(nn.Module):
    """Shared lattice graph network producing sitewise drift and diffusion gauges."""

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
    num_tau_encode: int = DEFAULT_GRAPH_NUM_TAU_ENCODE
    num_frequency_encode: int = DEFAULT_GRAPH_NUM_FREQUENCY_ENCODE
    graph_time_tau_init: Any = None
    graph_time_w_init: Any = None
    graph_time_tau_min: float = 1.0e-3
    graph_time_w_min: float = 1.0e-4

    hopping_matrix: Any = 0.0
    num_message_passing_layers: int = 3
    bound_drift: bool = False
    bound_diffusion: bool = False
    u1_invariant: bool = False

    def setup(self):
        self.graph_time_feature_mode = _normalize_graph_time_feature_name(self.time_feature)
        message_dim = self.feature_dims[0] if len(self.feature_dims) > 0 else self.embed_dim
        global_hidden_dim = self.feature_dims[1] if len(self.feature_dims) > 1 else max(self.embed_dim // 2, 32)

        if self.graph_time_feature_mode == "dense_time":
            self.time_dense_proj = DenseLayer(
                feature_dim=self.time_embed_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
        elif self.graph_time_feature_mode in {"damped_trig", "damped_trig_trend"}:
            tau_init = (
                _linear_positive_grid_np(
                    DEFAULT_GRAPH_TIME_TAU_RANGE[0],
                    DEFAULT_GRAPH_TIME_TAU_RANGE[1],
                    self.num_tau_encode,
                )
                if self.graph_time_tau_init is None
                else np.asarray(tuple(self.graph_time_tau_init), dtype=np.float64)
            )
            w_init = (
                _linear_positive_grid_np(
                    DEFAULT_GRAPH_TIME_W_RANGE[0],
                    DEFAULT_GRAPH_TIME_W_RANGE[1],
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

        self.node_embed = DenseLayer(
            feature_dim=self.embed_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.node_norm = nn.LayerNorm(
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
        self.global_to_node = DenseLayer(
            feature_dim=self.embed_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        self.graph_blocks = [
            LightweightGraphBlock(
                model_dim=self.embed_dim,
                message_dim=message_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for _ in range(self.num_message_passing_layers)
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
        self.degree, self.edge_dst, self.edge_src, self.edge_weight = self._graph_operators()

    def _encode_time(self, time, batch_size: int):
        """Encode scalar time into graph-global features."""
        t = jnp.asarray(time, self.mapping_dtype)
        t = jnp.atleast_1d(t)
        if t.shape[0] == 1:
            t = jnp.broadcast_to(t, (batch_size,))
        elif t.shape[0] != batch_size:
            raise ValueError(f"time must be scalar or have shape ({batch_size},), got {t.shape}")
        t = t[:, None]

        if self.graph_time_feature_mode == "no_time":
            return jnp.zeros((batch_size, 0), dtype=self.mapping_dtype)
        if self.graph_time_feature_mode == "raw_time":
            return t
        if self.graph_time_feature_mode == "dense_time":
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
        if self.graph_time_feature_mode == "damped_trig_trend":
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

    def _graph_operators(self):
        """Build degree features and sparse real edge operators once during module setup."""
        hopping = np.asarray(self.hopping_matrix)
        if hopping.ndim == 0:
            degree = jnp.zeros((self.num_site,), dtype=self.mapping_dtype)
            empty_int = jnp.zeros((0,), dtype=jnp.int32)
            empty_float = jnp.zeros((0,), dtype=self.mapping_dtype)
            return degree, empty_int, empty_int, empty_float

        if hopping.shape != (self.num_site, self.num_site):
            raise ValueError(
                f"hopping_matrix must have shape ({self.num_site}, {self.num_site}), got {hopping.shape}"
            )
        if not np.allclose(np.imag(hopping), 0.0, atol=1e-12):
            raise ValueError("neural_graph expects a real-valued hopping_matrix")

        # Preserve the configured hopping precision until the final cast to
        # mapping_dtype.  An unconditional float32 conversion silently
        # degraded graph features in float64 runs.
        hopping_real = np.real(hopping)
        hopping_abs = np.abs(hopping_real)

        degree = np.sum(hopping_abs, axis=-1)
        degree_scale = max(float(np.max(degree)), 1.0)
        degree = degree / degree_scale

        edge_dst_np, edge_src_np = np.nonzero(hopping_abs > 0.0)
        edge_weight_np = hopping_real[edge_dst_np, edge_src_np]

        return (
            jnp.asarray(degree, dtype=self.mapping_dtype),
            jnp.asarray(edge_dst_np, dtype=jnp.int32),
            jnp.asarray(edge_src_np, dtype=jnp.int32),
            jnp.asarray(edge_weight_np, dtype=self.mapping_dtype),
        )

    def __call__(self, lnOmega_real, alpha_real, beta_real, time, physical_params):
        """Map walker phase-space states to real-valued gauge fields."""
        batch_size = alpha_real.shape[0]
        num_site = alpha_real.shape[1] // 2
        if num_site != self.num_site:
            raise ValueError(f"Expected num_site={self.num_site}, but got {num_site} from alpha_real.")

        lnOmega_real = jnp.asarray(lnOmega_real, dtype=self.mapping_dtype)
        alpha_real = jnp.asarray(alpha_real, dtype=self.mapping_dtype)
        beta_real = jnp.asarray(beta_real, dtype=self.mapping_dtype)


        # local alpha, beta, and n
        a_re = alpha_real[:, :num_site]
        a_im = alpha_real[:, num_site:]
        b_re = beta_real[:, :num_site]
        b_im = beta_real[:, num_site:]
        n_re = a_re * b_re - a_im * b_im
        n_im = a_re * b_im + a_im * b_re
        logabs_a = _logabs_from_realimag(a_re, a_im, )
        logabs_b = _logabs_from_realimag(b_re, b_im, )
        logabs_n = _logabs_from_realimag(n_re, n_im, )
        sin_a, cos_a = _phase_sincos_from_realimag(a_re, a_im, )
        sin_b, cos_b = _phase_sincos_from_realimag(b_re, b_im, )
        sin_n, cos_n = _phase_sincos_from_realimag(n_re, n_im, )

        params = jnp.ravel(jnp.asarray(physical_params, dtype=self.mapping_dtype))

        # local drift alpha and beta
        alpha_c = lax.complex(a_re, a_im)
        beta_c = lax.complex(b_re, b_im)
        n_c = lax.complex(n_re, n_im)
        hop_alpha = _edge_hopping(alpha_c, self.edge_dst, self.edge_src, self.edge_weight)
        hop_beta = _edge_hopping(beta_c, self.edge_dst, self.edge_src, self.edge_weight)

        zero = jnp.asarray(0.0, dtype=self.mapping_dtype)
        U = params[0] if params.shape[0] > 0 else zero
        gamma = params[1] if params.shape[0] > 1 else zero
        F_re = params[3] if params.shape[0] > 3 else zero
        F_im = params[4] if params.shape[0] > 4 else zero
        Delta = params[5] if params.shape[0] > 5 else zero
        i_c = lax.complex(zero, jnp.asarray(1.0, dtype=self.mapping_dtype))
        F_c = lax.complex(F_re, F_im)
        onsite_alpha = i_c * Delta - i_c * U * n_c - 0.5 * gamma 
        onsite_beta = - i_c * Delta + i_c * U * n_c - 0.5 * gamma 
        drift_alpha = alpha_c * onsite_alpha - i_c * F_c + i_c * hop_alpha
        drift_beta = beta_c * onsite_beta + i_c * jnp.conj(F_c) - i_c * hop_beta

        drift_alpha_re = jnp.real(drift_alpha).astype(self.mapping_dtype)
        drift_alpha_im = jnp.imag(drift_alpha).astype(self.mapping_dtype)
        drift_beta_re = jnp.real(drift_beta).astype(self.mapping_dtype)
        drift_beta_im = jnp.imag(drift_beta).astype(self.mapping_dtype)
        logabs_drift_alpha = _logabs_from_realimag(drift_alpha_re, drift_alpha_im)
        logabs_drift_beta = _logabs_from_realimag(drift_beta_re, drift_beta_im)
        sin_drift_alpha, cos_drift_alpha = _phase_sincos_from_realimag(drift_alpha_re, drift_alpha_im)
        sin_drift_beta, cos_drift_beta = _phase_sincos_from_realimag(drift_beta_re, drift_beta_im)

        degree_feat = jnp.broadcast_to(self.degree[None, :, None], (batch_size, num_site, 1))
        if self.u1_invariant:
            # Joint-U(1)-invariant features. The physics is exactly covariant
            # under alpha -> e^{i th} alpha, beta -> e^{-i th} beta,
            # F -> e^{i th} F (verified pathwise for both solver backends),
            # and the optimal gauge outputs are invariant scalars. Every
            # feature below is invariant under that joint rotation, so the
            # network cannot spend capacity on the unphysical global phase:
            # - n = alpha*beta and the drift/hopping bilinears carry all
            #   relative-phase information;
            # - the drive enters only through the spurions F*beta and
            #   conj(F)*alpha (relative phase between drive and field);
            # - log-moduli carry the amplitudes.
            bilinear_drift_alpha = beta_c * drift_alpha
            bilinear_drift_beta = alpha_c * drift_beta
            bilinear_hop_alpha = beta_c * hop_alpha
            bilinear_hop_beta = alpha_c * hop_beta
            spurion_alpha = jnp.conj(F_c) * alpha_c
            spurion_beta = F_c * beta_c
            node_feat = jnp.stack(
                [
                    n_re,
                    n_im,
                    jnp.real(bilinear_drift_alpha).astype(self.mapping_dtype),
                    jnp.imag(bilinear_drift_alpha).astype(self.mapping_dtype),
                    jnp.real(bilinear_drift_beta).astype(self.mapping_dtype),
                    jnp.imag(bilinear_drift_beta).astype(self.mapping_dtype),
                    jnp.real(bilinear_hop_alpha).astype(self.mapping_dtype),
                    jnp.imag(bilinear_hop_alpha).astype(self.mapping_dtype),
                    jnp.real(bilinear_hop_beta).astype(self.mapping_dtype),
                    jnp.imag(bilinear_hop_beta).astype(self.mapping_dtype),
                    jnp.real(spurion_alpha).astype(self.mapping_dtype),
                    jnp.imag(spurion_alpha).astype(self.mapping_dtype),
                    jnp.real(spurion_beta).astype(self.mapping_dtype),
                    jnp.imag(spurion_beta).astype(self.mapping_dtype),
                    logabs_a,
                    logabs_b,
                    logabs_n,
                    logabs_drift_alpha,
                    logabs_drift_beta,
                ],
                axis=-1,
            )
        else:
            node_feat = jnp.stack(
                [
                    a_re,
                    a_im,
                    b_re,
                    b_im,
                    n_re,
                    n_im,
                    drift_alpha_re,
                    drift_alpha_im,
                    drift_beta_re,
                    drift_beta_im,
                    logabs_a,
                    logabs_b,
                    logabs_n,
                    logabs_drift_alpha,
                    logabs_drift_beta,
                    sin_a,
                    cos_a,
                    sin_b,
                    cos_b,
                    sin_n,
                    cos_n,
                    sin_drift_alpha,
                    cos_drift_alpha,
                    sin_drift_beta,
                    cos_drift_beta,
                ],
                axis=-1,
            )
        node_feat = jnp.concatenate([node_feat, degree_feat], axis=-1)

        time_feat = self._encode_time(time, batch_size)
        lnOmega_feat = _lnOmega_features(lnOmega_real, self.mapping_dtype)
        global_feat = jnp.concatenate([lnOmega_feat, time_feat], axis=-1)

        global_hidden = nn.gelu(self.global_proj(global_feat))
        global_context = nn.gelu(self.global_to_node(global_hidden))

        h = self.node_embed(node_feat)
        h = self.node_norm(h)
        h = nn.gelu(h + global_context[:, None, :])

        for block in self.graph_blocks:
            h = block(h, global_hidden, global_context, self.edge_dst, self.edge_src, self.edge_weight)

        h = self.final_norm(h)
        h = nn.gelu(h)

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
