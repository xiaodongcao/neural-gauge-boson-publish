from .lib_preinclude import *
from .model import DenseLayer, FANLayers, MLPBlock, multiscale_fourier_features


class GaugeNN(nn.Module):
    num_site: int
    embed_dim: int
    time_embed_dim: int
    feature_dims: Sequence[int]
    use_bias: bool = True
    mapping_dtype: jnp.dtype = DTYPE
    kernel_init: Callable = nn.initializers.normal(stddev=0.1)
    bias_init: Callable = nn.initializers.zeros

    drift_max: float = 10.0
    diffusion_max: float = 10.0
    bound_drift: bool = False
    bound_diffusion: bool = False

    time_feature: str = "None"
    time_scales: Sequence[float] = (0.5, 1.0, 10.0)
    time_num_freqs: int = 8
    time_fmin: float = 0.5
    time_fmax: float = 5.0

    def setup(self):
        if self.time_feature == "Dense":
            self.embeding_time = DenseLayer(
                feature_dim=self.time_embed_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
        elif self.time_feature == "FAN":
            self.embeding_time = FANLayers(
                num_layers=1,
                feature_dim=self.time_embed_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
        elif self.time_feature == "FixedFourier":
            self.embeding_time = DenseLayer(
                feature_dim=self.time_embed_dim,
                mapping_dtype=self.mapping_dtype,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
        elif self.time_feature != "None":
            raise ValueError(f"Invalid time feature: {self.time_feature}")

        self.input_layer_norm = nn.LayerNorm(
            param_dtype=self.mapping_dtype,
            dtype=self.mapping_dtype,
            epsilon=1e-5,
            scale_init=nn.initializers.ones,
            bias_init=nn.initializers.zeros,
        )
        self.embed_layer = DenseLayer(
            feature_dim=self.embed_dim,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        self.fusion = MLPBlock(
            hidden_dim=self.feature_dims[0],
            out_dim=self.feature_dims[1],
            mapping_dtype=self.mapping_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        head_init = nn.initializers.normal(stddev=0.01)
        self.drift_g_fit = DenseLayer(
            feature_dim=2 * self.num_site,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=head_init,
            bias_init=self.bias_init,
        )
        self.drift_f_fit = DenseLayer(
            feature_dim=2 * self.num_site,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=head_init,
            bias_init=self.bias_init,
        )
        self.diffusion_g_fit = DenseLayer(
            feature_dim=self.num_site,
            mapping_dtype=self.mapping_dtype,
            use_bias=self.use_bias,
            kernel_init=head_init,
            bias_init=self.bias_init,
        )

    def __call__(self, lnOmega_real, alpha_real, beta_real, time, physical_params):
        """Map flattened walker features to real-valued gauge fields."""
        del physical_params
        num_walker = alpha_real.shape[0]
        num_site = alpha_real.shape[1] // 2
        if num_site != self.num_site:
            raise ValueError(f"Expected num_site={self.num_site}, but got {num_site} from alpha_real.")

        omega_feat = jnp.asarray(lnOmega_real, dtype=self.mapping_dtype)
        a_re = jnp.asarray(alpha_real[:, :num_site], dtype=self.mapping_dtype)
        a_im = jnp.asarray(alpha_real[:, num_site:], dtype=self.mapping_dtype)
        b_re = jnp.asarray(beta_real[:, :num_site], dtype=self.mapping_dtype)
        b_im = jnp.asarray(beta_real[:, num_site:], dtype=self.mapping_dtype)
        var_feat = jnp.concatenate([a_re, a_im, b_re, b_im], axis=1)

        time_feat = jnp.asarray(time, self.mapping_dtype)
        time_feat = jnp.atleast_1d(time_feat)[:, None]
        time_feat = jnp.broadcast_to(time_feat, (num_walker, time_feat.shape[1]))

        if self.time_feature == "None":
            x = jnp.concatenate([omega_feat, var_feat, ], axis=1)
        elif self.time_feature == "Dense" or self.time_feature == "FAN":
            x = jnp.concatenate([omega_feat, var_feat, self.embeding_time(time_feat), ], axis=1)
        else:
            fourier_feat = multiscale_fourier_features(
                t=jnp.asarray(time, self.mapping_dtype),
                scales=self.time_scales,
                num_freqs=self.time_num_freqs,
                fmin=self.time_fmin,
                fmax=self.time_fmax,
                logspace=True,
                dtype=self.mapping_dtype,
            )
            fourier_feat = jnp.broadcast_to(fourier_feat, (num_walker, fourier_feat.shape[1]))
            x = jnp.concatenate([omega_feat, var_feat, self.embeding_time(fourier_feat), ], axis=1)

        x = self.embed_layer(x)
        x = self.input_layer_norm(x)
        x = self.fusion(x)

        drift_g = self.drift_g_fit(x)
        drift_f = self.drift_f_fit(x)
        diffusion_g = self.diffusion_g_fit(x)
        if self.bound_drift:
            drift_max = jnp.asarray(self.drift_max, dtype=self.mapping_dtype)
            drift_g = drift_max * jnp.tanh(drift_g / drift_max)
            drift_f = drift_max * jnp.tanh(drift_f / drift_max)
        if self.bound_diffusion:
            diffusion_max = jnp.asarray(self.diffusion_max, dtype=self.mapping_dtype)
            diffusion_g = diffusion_max * jnp.tanh(diffusion_g / diffusion_max)
        return drift_g, drift_f, diffusion_g
