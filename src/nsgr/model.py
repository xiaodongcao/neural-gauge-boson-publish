from .lib_preinclude import *
from .utility import create_train_state


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
    return "unknown"


def multiscale_fourier_features(
    t: jnp.ndarray,
    scales: Sequence[float] = (1.0,),
    num_freqs: int = 8,
    fmin: float = 1.0,
    fmax: float = 10.0,
    logspace: bool = True,
    dtype: jnp.dtype = DTYPE,
):
    t = jnp.asarray(t, dtype)
    if t.ndim == 0:
        t = t[None]

    if logspace:
        base = jnp.exp(jnp.linspace(jnp.log(fmin), jnp.log(fmax), num_freqs, dtype=dtype))
    else:
        base = jnp.linspace(fmin, fmax, num_freqs, dtype=dtype)

    feats = []
    for scale in scales:
        scaled_t = t / jnp.asarray(scale, dtype)
        phase = scaled_t[:, None] * base[None, :]
        feats.append(jnp.sin(phase))
        feats.append(jnp.cos(phase))
    return jnp.concatenate(feats, axis=-1)


class DenseLayer(nn.Module):
    """Typed dense layer shared by the graph and MLP gauge models."""

    feature_dim: int
    use_bias: bool = True
    mapping_dtype: jnp.dtype = DTYPE
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        return nn.Dense(
            features=self.feature_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.mapping_dtype,
            param_dtype=self.mapping_dtype,
        )(jnp.asarray(inputs, dtype=self.mapping_dtype))


class FANLayer(nn.Module):
    feature_dim: int
    mapping_dtype: jnp.dtype = DTYPE
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        # Each periodic latent contributes both sin and cos channels.  Allocate
        # the remaining channels to the non-periodic branch so the layer always
        # returns exactly feature_dim outputs, including odd and very small
        # widths.
        self.periodic_dim = int(self.feature_dim) // 4
        self.nonperiodic_dim = int(self.feature_dim) - 2 * self.periodic_dim
        self.lin_feature_embed_q = (
            nn.Dense(
                features=self.periodic_dim,
                param_dtype=self.mapping_dtype,
                dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            if self.periodic_dim > 0
            else None
        )
        self.lin_feature_embed_g = nn.Dense(
            features=self.nonperiodic_dim,
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

    def __call__(self, x):
        g = nn.gelu(self.lin_feature_embed_g(x))
        if self.lin_feature_embed_q is None:
            return g
        q = self.lin_feature_embed_q(x)
        return jnp.concatenate([jnp.sin(q), jnp.cos(q), g], axis=-1)


class FANLayers(nn.Module):
    num_layers: int
    feature_dim: int
    mapping_dtype: jnp.dtype = DTYPE
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.layers = [
            FANLayer(
                feature_dim=self.feature_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MLPBlock(nn.Module):
    hidden_dim: int
    out_dim: int
    mapping_dtype: jnp.dtype = DTYPE
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    def setup(self):
        self.fc1 = DenseLayer(
            feature_dim=self.hidden_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            mapping_dtype=self.mapping_dtype,
        )
        self.ln1 = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.fc2 = DenseLayer(
            feature_dim=self.out_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            mapping_dtype=self.mapping_dtype,
        )
        self.ln2 = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        x = self.ln1(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.ln2(x)
        x = self.fc2(x)
        x = nn.gelu(x)
        return x


class Model:
    """Factory and checkpoint wrapper for neural-network gauge models."""

    def __init__(self, config: Dict[str, Any], lattice, gauge_mode: str = "neural_graph"):
        self.config = config
        self.lattice = lattice
        self.gauge_mode = str(gauge_mode).strip()
        self.module = self._build_module()

    def _common_kwargs(self):
        model_cfg = self.config
        return dict(
            num_site=self.lattice.num_site,
            mapping_dtype=NNDTYPE,
            use_bias=True,
            embed_dim=model_cfg["embed_dim"],
            time_embed_dim=model_cfg["time_embed_dim"],
            feature_dims=model_cfg["feature_dims"],
            time_feature=model_cfg.get("time_feature", "Dense"),
            drift_max=model_cfg.get("drift_max", 10.0),
            diffusion_max=model_cfg.get("diffusion_max", 10.0),
            bound_drift=model_cfg.get("bound_drift", False),
            bound_diffusion=model_cfg.get("bound_diffusion", False),
        )

    def config_summary_lines(self) -> list[str]:
        cfg = self.config
        lines = [f"selected neural model = {self.gauge_mode}"]
        if self.gauge_mode == "neural_graph":
            graph_time_feature = cfg.get("time_feature", "Dense")
            graph_time_mode = _normalize_graph_time_feature_name(graph_time_feature)
            if graph_time_mode in {"damped_trig", "damped_trig_trend"}:
                num_tau_encode = int(
                    cfg.get(
                        "num_tau_encode",
                        len(cfg.get("graph_time_tau_init", (0.1, 0.4, 1.2, 3.6))),
                    )
                )
                num_frequency_encode = int(
                    cfg.get(
                        "num_frequency_encode",
                        len(cfg.get("graph_time_w_init", (0.25, 0.5, 1.0, 2.0))),
                    )
                )
                tau_initialization = cfg.get("graph_time_tau_init")
                frequency_initialization = cfg.get("graph_time_w_init")
                tau_initialization_text = (
                    "default [1e-2, 10] linear grid"
                    if tau_initialization is None
                    else str(list(tau_initialization))
                )
                frequency_initialization_text = (
                    "default [1e-2, 10] linear grid"
                    if frequency_initialization is None
                    else str(list(frequency_initialization))
                )
                time_initialization_summary = (
                    "damped-trig initialization: "
                    f"tau={tau_initialization_text}, "
                    f"w={frequency_initialization_text}"
                )
                if graph_time_mode == "damped_trig_trend":
                    num_time_features = 4 * num_tau_encode * num_frequency_encode
                    graph_time_meaning = (
                        "use learned damped and rising sinusoidal time features "
                        "exp(-t/tau_i) and (1 - exp(-t/tau_i)) times cos(w_j t), sin(w_j t) "
                        "on a learnable tau x frequency grid"
                    )
                    graph_time_embed_meaning = "ignored by neural_graph in DampedTrigTrend mode"
                else:
                    num_time_features = 2 * num_tau_encode * num_frequency_encode
                    graph_time_meaning = (
                        "use learned damped sinusoidal time features "
                        "exp(-t/tau_i) * cos(w_j t) and exp(-t/tau_i) * sin(w_j t) "
                        "on a learnable tau x frequency grid"
                    )
                    graph_time_embed_meaning = "ignored by neural_graph in DampedTrig mode"
            elif graph_time_mode == "dense_time":
                graph_time_meaning = (
                    "apply a learned linear map from scalar time to a "
                    f"{cfg['time_embed_dim']}-dimensional time feature"
                )
                graph_time_embed_meaning = (
                    "output width of the learned linear graph time projection"
                )
            elif graph_time_mode == "no_time":
                graph_time_meaning = "omit time from the graph-network inputs"
                graph_time_embed_meaning = "ignored by neural_graph when no time feature is used"
            else:
                graph_time_meaning = "concatenate raw scalar time only"
                graph_time_embed_meaning = "ignored by neural_graph in Raw mode"
            lines.extend(
                [
                    "active model config:",
                    f"  - embed_dim = {cfg['embed_dim']}: hidden site-state width in the graph trunk",
                    (
                        f"  - time_embed_dim = {cfg['time_embed_dim']}: "
                        f"{graph_time_embed_meaning}"
                    ),
                    (
                        f"  - feature_dims = {list(cfg['feature_dims'])}: "
                        "[sparse message width, shared global-conditioning hidden width]"
                    ),
                    f"  - time_feature = {graph_time_feature}: {graph_time_meaning}",
                    (
                        f"  - num_message_passing_layers = {cfg.get('num_message_passing_layers', 3)}: "
                        "number of lightweight sparse real-edge message-passing blocks"
                    ),
                    (
                        f"  - u1_invariant = {cfg.get('u1_invariant', False)}: "
                        "whether node features are joint-U(1)-invariant "
                        "(n, drift/hopping bilinears, drive spurions, log-moduli) "
                        "instead of raw phase-carrying components"
                    ),
                    (
                        f"  - bound_drift = {cfg.get('bound_drift', False)}: "
                        "whether to tanh-bound the two drift gauges"
                    ),
                    (
                        f"  - bound_diffusion = {cfg.get('bound_diffusion', False)}: "
                        "whether to independently tanh-bound the diffusion gauge"
                    ),
                    (
                        f"  - drift_max = {cfg.get('drift_max', 10.0)}, "
                        f"diffusion_max = {cfg.get('diffusion_max', 10.0)}: "
                        "drift and diffusion bounds for their active switches"
                    ),
                    "model notes:",
                    "  - the graph model uses local phase-space features, positive-P drift features, raw walker-local lnOmega, and the selected time features",
                    "  - each graph block reuses that conditioner to modulate both the message path and the local residual path",
                    "  - physical_params provide the fixed physical scalars used to build the positive-P drift node features",
                ]
            )
            if graph_time_mode in {"damped_trig", "damped_trig_trend"}:
                lines[4:4] = [
                    (
                        f"  - num_tau_encode = {num_tau_encode}, "
                        f"num_frequency_encode = {num_frequency_encode}: "
                        "learnable tau and frequency grid sizes"
                    ),
                    (
                        f"  - {time_initialization_summary}, giving "
                        f"{num_time_features} time features"
                    ),
                ]
            return lines

        if self.gauge_mode == "neural_cnn":
            cnn_time_feature = cfg.get("time_feature", "Dense")
            cnn_time_mode = _normalize_graph_time_feature_name(cnn_time_feature)
            if cnn_time_mode in {"damped_trig", "damped_trig_trend"}:
                num_tau_encode = int(
                    cfg.get(
                        "num_tau_encode",
                        len(cfg.get("graph_time_tau_init", (0.1, 0.4, 1.2, 3.6))),
                    )
                )
                num_frequency_encode = int(
                    cfg.get(
                        "num_frequency_encode",
                        len(cfg.get("graph_time_w_init", (0.25, 0.5, 1.0, 2.0))),
                    )
                )
                tau_initialization = cfg.get("graph_time_tau_init")
                frequency_initialization = cfg.get("graph_time_w_init")
                tau_initialization_text = (
                    "default [1e-2, 10] linear grid"
                    if tau_initialization is None
                    else str(list(tau_initialization))
                )
                frequency_initialization_text = (
                    "default [1e-2, 10] linear grid"
                    if frequency_initialization is None
                    else str(list(frequency_initialization))
                )
                time_initialization_summary = (
                    "damped-trig initialization: "
                    f"tau={tau_initialization_text}, "
                    f"w={frequency_initialization_text}"
                )
                if cnn_time_mode == "damped_trig_trend":
                    num_time_features = 4 * num_tau_encode * num_frequency_encode
                    cnn_time_meaning = (
                        "use learned damped and rising sinusoidal time features "
                        "exp(-t/tau_i) and (1 - exp(-t/tau_i)) times cos(w_j t), sin(w_j t)"
                    )
                    cnn_time_embed_meaning = "ignored by neural_cnn in DampedTrigTrend mode"
                else:
                    num_time_features = 2 * num_tau_encode * num_frequency_encode
                    cnn_time_meaning = (
                        "use learned damped sinusoidal time features "
                        "exp(-t/tau_i) * cos(w_j t) and exp(-t/tau_i) * sin(w_j t)"
                    )
                    cnn_time_embed_meaning = "ignored by neural_cnn in DampedTrig mode"
            elif cnn_time_mode == "dense_time":
                num_tau_encode = int(cfg.get("num_tau_encode", 4))
                num_frequency_encode = int(cfg.get("num_frequency_encode", 4))
                num_time_features = cfg["time_embed_dim"]
                cnn_time_meaning = (
                    "apply a learned linear map from scalar time to a "
                    f"{cfg['time_embed_dim']}-dimensional time feature"
                )
                cnn_time_embed_meaning = (
                    "output width of the learned linear CNN time projection"
                )
            elif cnn_time_mode == "no_time":
                num_tau_encode = int(cfg.get("num_tau_encode", 4))
                num_frequency_encode = int(cfg.get("num_frequency_encode", 4))
                num_time_features = 0
                cnn_time_meaning = "omit time from the CNN global conditioner"
                cnn_time_embed_meaning = "ignored by neural_cnn when no time feature is used"
            else:
                num_tau_encode = int(cfg.get("num_tau_encode", 4))
                num_frequency_encode = int(cfg.get("num_frequency_encode", 4))
                num_time_features = 1
                cnn_time_meaning = "concatenate raw scalar time in the CNN global conditioner"
                cnn_time_embed_meaning = "ignored by neural_cnn in Raw mode"

            lines.extend(
                [
                    "active model config:",
                    f"  - embed_dim = {cfg['embed_dim']}: hidden site-state width in the CNN/stencil trunk",
                    (
                        f"  - time_embed_dim = {cfg['time_embed_dim']}: "
                        f"{cnn_time_embed_meaning}"
                    ),
                    (
                        f"  - feature_dims = {list(cfg['feature_dims'])}: "
                        "[3x3 stencil mixing width, shared global-conditioning hidden width]"
                    ),
                    f"  - time_feature = {cnn_time_feature}: {cnn_time_meaning}",
                    (
                        f"  - num_cnn_layers = {cfg.get('num_cnn_layers', cfg.get('num_message_passing_layers', 2))}: "
                        "number of lightweight 3x3 lattice-stencil blocks"
                    ),
                    (
                        f"  - bound_drift = {cfg.get('bound_drift', False)}: "
                        "whether to tanh-bound the two drift gauges"
                    ),
                    (
                        f"  - bound_diffusion = {cfg.get('bound_diffusion', False)}: "
                        "whether to independently tanh-bound the diffusion gauge"
                    ),
                    (
                        f"  - drift_max = {cfg.get('drift_max', 10.0)}, "
                        f"diffusion_max = {cfg.get('diffusion_max', 10.0)}: "
                        "drift and diffusion bounds for their active switches"
                    ),
                    "model notes:",
                    "  - the CNN model applies 3x3 lattice stencil mixing to local alpha and beta fields",
                    "  - raw walker-local lnOmega and the selected time features enter only through global FiLM-style conditioning",
                    "  - physical_params stay in the public model call signature but are ignored inside the CNN network",
                ]
            )
            if cnn_time_mode in {"damped_trig", "damped_trig_trend"}:
                lines[4:4] = [
                    (
                        f"  - num_tau_encode = {num_tau_encode}, "
                        f"num_frequency_encode = {num_frequency_encode}: "
                        "learnable tau and frequency grid sizes"
                    ),
                    (
                        f"  - {time_initialization_summary}, giving "
                        f"{num_time_features} time features"
                    ),
                ]
            return lines

        time_feature = cfg.get("time_feature", "None")
        if time_feature == "None":
            time_meaning = "omit time from the MLP inputs"
        elif time_feature == "Dense":
            time_meaning = "append a learned dense embedding of raw time"
        elif time_feature == "FAN":
            time_meaning = "append a FAN time embedding"
        else:
            time_meaning = "append fixed Fourier features followed by a learned projection"

        lines.extend(
            [
                "active model config:",
                f"  - embed_dim = {cfg['embed_dim']}: input embedding width before the fusion MLP",
                (
                    f"  - time_embed_dim = {cfg['time_embed_dim']}: shared embedding width "
                    "used for learned time features"
                ),
                (
                    f"  - feature_dims = {list(cfg['feature_dims'])}: "
                    "[fusion hidden width, fusion output width]"
                ),
                f"  - time_feature = {time_feature}: {time_meaning}",
                "model notes:",
                (
                    f"  - num_message_passing_layers = {cfg.get('num_message_passing_layers', 3)}: "
                    "graph setting; also the CNN default when num_cnn_layers is absent"
                ),
                (
                    f"  - bound_drift = {cfg.get('bound_drift', False)}: "
                    "whether to tanh-bound the two drift gauges"
                ),
                (
                    f"  - bound_diffusion = {cfg.get('bound_diffusion', False)}: "
                    "whether to tanh-bound the diffusion gauge"
                ),
                (
                    f"  - drift_max = {cfg.get('drift_max', 10.0)}, "
                    f"diffusion_max = {cfg.get('diffusion_max', 10.0)}: "
                    "drift and diffusion bounds for their independent switches"
                ),
                "  - physical_params stay in the public model call signature but are ignored inside the network",
            ]
        )
        return lines

    def _build_module(self):
        common_kwargs = self._common_kwargs()
        if self.gauge_mode == "neural_mlp":
            from .model_MLP import GaugeNN as MLPGaugeNN

            return MLPGaugeNN(**common_kwargs)

        if self.gauge_mode == "neural_cnn":
            from .model_cnn import GaugeNN as CNNGaugeNN

            return CNNGaugeNN(
                **common_kwargs,
                lattice_shape=(int(self.lattice.Nx), int(self.lattice.Ny)),
                boundary_x=self.lattice.boundary_x,
                boundary_y=self.lattice.boundary_y,
                num_cnn_layers=int(
                    self.config.get("num_cnn_layers", self.config.get("num_message_passing_layers", 2))
                ),
                num_tau_encode=int(
                    self.config.get(
                        "num_tau_encode",
                        len(self.config.get("graph_time_tau_init", (0.1, 0.4, 1.2, 3.6))),
                    )
                ),
                num_frequency_encode=int(
                    self.config.get(
                        "num_frequency_encode",
                        len(self.config.get("graph_time_w_init", (0.25, 0.5, 1.0, 2.0))),
                    )
                ),
                graph_time_tau_init=(
                    tuple(self.config["graph_time_tau_init"])
                    if "graph_time_tau_init" in self.config
                    else None
                ),
                graph_time_w_init=(
                    tuple(self.config["graph_time_w_init"])
                    if "graph_time_w_init" in self.config
                    else None
                ),
                graph_time_tau_min=float(self.config.get("graph_time_tau_min", 1.0e-3)),
                graph_time_w_min=float(self.config.get("graph_time_w_min", 1.0e-4)),
            )

        if self.gauge_mode != "neural_graph":
            raise ValueError(
                "Model only constructs neural gauge modes; expected one of "
                "'neural_mlp', 'neural_cnn', or 'neural_graph', "
                f"got {self.gauge_mode!r}"
            )

        from .model_graph import GaugeNN as GraphGaugeNN

        return GraphGaugeNN(
            **common_kwargs,
            hopping_matrix=self.lattice.hopping_matrix,
            num_message_passing_layers=self.config.get("num_message_passing_layers", 3),
            u1_invariant=bool(self.config.get("u1_invariant", False)),
            num_tau_encode=int(
                self.config.get(
                    "num_tau_encode",
                    len(self.config.get("graph_time_tau_init", (0.1, 0.4, 1.2, 3.6))),
                )
            ),
            num_frequency_encode=int(
                self.config.get(
                    "num_frequency_encode",
                    len(self.config.get("graph_time_w_init", (0.25, 0.5, 1.0, 2.0))),
                )
            ),
            graph_time_tau_init=(
                tuple(self.config["graph_time_tau_init"])
                if "graph_time_tau_init" in self.config
                else None
            ),
            graph_time_w_init=(
                tuple(self.config["graph_time_w_init"])
                if "graph_time_w_init" in self.config
                else None
            ),
            graph_time_tau_min=float(self.config.get("graph_time_tau_min", 1.0e-3)),
            graph_time_w_min=float(self.config.get("graph_time_w_min", 1.0e-4)),
        )

    def create_train_state(
        self,
        config: Dict[str, Any],
        key: jax.Array,
        sample_lnOmega_real: jnp.ndarray,
        sample_alpha_real: jnp.ndarray,
        sample_beta_real: jnp.ndarray,
        sample_t: float,
        physical_params: jnp.ndarray,
        params_path: Optional[str] = None,
    ):
        return create_train_state(
            config=config,
            key=key,
            model=self.module,
            sample_lnOmega_real=jnp.asarray(sample_lnOmega_real, dtype=NNDTYPE),
            sample_alpha_real=jnp.asarray(sample_alpha_real, dtype=NNDTYPE),
            sample_beta_real=jnp.asarray(sample_beta_real, dtype=NNDTYPE),
            sample_t=jnp.asarray(sample_t, dtype=NNDTYPE),
            physical_params=jnp.asarray(physical_params, dtype=NNDTYPE),
            params_path=params_path,
        )


__all__ = [
    "DenseLayer",
    "FANLayer",
    "FANLayers",
    "MLPBlock",
    "Model",
    "multiscale_fourier_features",
]
