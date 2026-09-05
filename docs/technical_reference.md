# NSGR Technical Reference

**Neural Stochastic-Gauge Representation for Quantum Many-Body Systems**

The maintained trainer has five loss families:

1. A covariance-normalized, window-integrated projected residual.
2. An optional Pareto-k envelope penalty for selected observable clouds.
3. An optional neural-gauge output penalty.
4. An optional neural-parameter L2 penalty.
5. An optional weight-entropy (complex-ESS) budget penalty.

Their prefactors are, respectively,

```text
loss_residual_gmm_prefactor
loss_pareto_k_prefactor
loss_gauge_prefactor
loss_L2_prefactor
loss_ess_prefactor
```

The optimized scalar is the sum of enabled, prefactor-weighted terms. A zero
prefactor disables its family. Trace preservation is part of the projected
residual and is not a separate selectable objective. Unsupported training keys
are rejected by `utility.load_config` rather than silently ignored.

## SDE solver selection

Training accepts

```json
"sde_solver": "semi_implicit_midpoint",
"sde_max_iter": 3,
"sde_affine_expm_order": 6,
"sde_affine_expm_substeps": 1,
"sde_newton_damping_steps": 4
```

inside the `training` section. The valid solver names are
`"semi_implicit_midpoint"`, `"interaction_picture_implicit_midpoint"`, and
`"interaction_picture_exact_local"`; the first remains the default for
backward compatibility. The same keys are accepted in `simulation`. If the
simulation solver is omitted, validation inherits the training solver rather
than silently changing the discretization. The selected solver is a static JAX
argument, is shared by ordinary, staged, segmented-bank, and multi-device
rollouts, and is recorded in training and simulation metadata.

The controls have backend-specific applicability:

- `semi_implicit_midpoint`: `sde_max_iter` is its static Picard count.
- `interaction_picture_implicit_midpoint`: `sde_max_iter`, positive mixed
  `sde_root_rtol/atol`, `sde_newton_damping_steps`, and
  `sde_affine_expm_order/substeps` are active.
- `interaction_picture_exact_local`: only
  `sde_affine_expm_order/substeps` are active; it has no nonlinear iteration,
  root tolerance, or Newton damping.

Root-tolerance defaults are dtype-aware: `1e-9` and `1e-11` in float64, and
`1e-5` and `1e-7` in float32. The positive integer affine order, affine
substeps, and Newton damping controls default to 6, 1, and 4. Simulation
inherits an omitted solver name and each omitted root-tolerance, affine-flow,
or damping control from the validated training section. The pre-existing
`simulation.sde_max_iter` behavior is unchanged: when omitted, its default is
4. Solver names and integer controls are static JAX arguments, so changing one
can create a distinct compiled executable even when that control is inactive.
Startup reports active values and names inactive controls. Metadata preserves
all configured values for provenance and records authoritative
`active_controls` and `inactive_controls` lists.

`interaction_picture_implicit_midpoint` does not silently accept an
unconverged nonlinear root. Its standard `step` interface replaces the entire
affected walker by NaNs when any site fails convergence, encounters a failed
linear solve, or becomes non-finite; the whole walker is invalidated because
the second affine half-flow couples sites. The existing training finite checks
therefore reject the affected update or frozen-bank refresh. For direct solver
validation, `InteractionPictureImplicitMidpointSolver.step_with_diagnostics`
also returns per-walker, per-site convergence flags, final residual norms,
Newton iteration counts, linear-solve failure flags, and finite flags. These
diagnostics are not added to every training metric payload.

For this solver, one step first advances the affine
loss/detuning/hopping/drive subflow by half a step. The Itô diffusion is
evaluated from the old phase-space state (the explicit prepoint convention),
and its increment is transported through the corresponding homogeneous affine
half-flow. A damped Newton solve then advances the local Kerr plus frozen
gauge-drift part for one full step, followed by the second affine half-step.
The no-hopping and one-site affine flow is analytic. Multisite hopping uses a
static, matrix-free truncated-Taylor exponential action; its order and number
of substeps are the two affine controls above. It avoids dense matrix
exponentials and keeps sparse lattice scaling, but it is not algebraically
exact. Scientific runs must verify timestep and affine-order/substep
convergence. This backend targets more reliable nonlinear solves and gradients;
it is not guaranteed to be faster than the fixed-Picard backend on every
lattice.

### Exact-local interaction-picture backend

`interaction_picture_exact_local` composes a timestep `h` as

```text
H(h/2) -> D(h/2) -> S(h) -> D(h/2) -> H(h/2).
```

`H` contains hopping and coherent drive. It reuses the same affine action as
the implicit interaction-picture backend, with loss and detuning set to zero
inside this subflow. It is analytic for one site or zero hopping; general
hopping uses the static matrix-free Taylor action controlled by
`sde_affine_expm_order/substeps`.

`D` is the exact onsite deterministic Kerr/loss/detuning flow. For
`n_0 = alpha beta`, substep length `tau`, and
`phi_1(z) = (exp(z)-1)/z`, evaluated by a stable near-zero series,

```text
n(tau) = n_0 exp(-gamma tau)
I(tau) = n_0 tau phi_1(-gamma tau)
alpha' = alpha exp[(-gamma/2 + i Delta) tau - i U I(tau)]
beta'  = beta  exp[(-gamma/2 - i Delta) tau + i U I(tau)].
```

These algebraic complex expressions remain valid away from the physical
manifold; a complex `n_0` can therefore change amplitudes as well as phases.

`S` evaluates no neural network internally. It consumes the adapted gauge
fields supplied by the rollout and freezes them over the timestep. Let
`G=(g,f)`, `s=sqrt(i U)`, `c=cosh(lambda)`, `q=sinh(lambda)`, and define the
two-noise coefficient vectors

```text
b_alpha = ( i s c, -s q)
b_beta  = (-i s q,  s c).
```

For `x` equal to `alpha` or `beta`, its exact frozen-coefficient geometric-Itô
map is

```text
x' = x exp[b_x . (Delta W - G h) - (b_x . b_x) h/2],
```

and the coupled logarithmic weight update is

```text
Delta lnOmega = sum_sites[G . Delta W - (G . G) h/2].
```

Every dot and square here is algebraic, without complex conjugation. State and
weight use the same two real Wiener increments. The existing stopped,
walker-common real log-weight centering is then applied. This construction is
the explicit prepoint-Itô convention conditional on gauge fields frozen over
the step; a coarser neural-gauge refresh cadence keeps the most recently
adapted fields frozen across its intervening steps.

Each subflow above is exact under its frozen coefficients (apart from the
configured multisite affine Taylor approximation), but the full SDE step is
not exact because `H`, `D`, and `S` do not commute and the neural gauge is
state-dependent. The palindromic composition alone does not establish
stochastic order two. Until same-Wiener coupled-`dt` tests establish stronger
behavior, no claim is made beyond the conservative weak-order-one and
strong-order-one-half target. Validate phase and amplitude, weak observables,
pathwise gradients, tail diagnostics, and affine-action convergence at fixed
wall time before preferring this backend for a new regime.

## Covariance-Normalized Projected Residual

A typical configuration is

```json
{
  "training": {
    "operator_monomials": [[0,1], [1,0], [1,1], [2,2]],
    "loss_residual_gmm_prefactor": 1.0,
    "residual_gmm_integrator_nodes": 6,
    "residual_gmm_d_clip": 10.0,
    "residual_gmm_cov_floor": 1.0e-8,
    "residual_gmm_cov_shrinkage": 0.05,
    "residual_gmm_trace_mode": "diagnostic",
    "residual_gmm_time_aggregation": "mean",
    "residual_gmm_time_beta": 2.0
  }
}
```

`training.operator_monomials` is the complete scientific selector for this
loss. `(m,n)` denotes the onsite normal-ordered monomial
`(a^dagger)^m a^n`. The identity `(0,0)` is reserved for the automatic trace
equation and must not be listed. If the selector is omitted while the loss is
active, validation inserts the four pairs shown above.

There is no generated channel hierarchy. The persistent raw-diagnostic channel
order at every lattice site is exactly

```text
[trace, (m_0,n_0), (m_1,n_1), ...]
```

where the pairs retain configuration order. Let `M` be the number of configured
monomials. The raw diagnostic count and active covariance/objective count are

```text
C_diagnostic = 1 + M
C_objective  = M + (1 if residual_gmm_trace_mode == "joint" else 0)
D            = 2 C_objective
```

These are the complex diagnostic/objective counts and active real channel
dimension per site.

The default `residual_gmm_trace_mode="diagnostic"` keeps the trace equation in
raw histories but excludes it from covariance estimation, whitening,
Mahalanobis clipping, `q`, and the optimized objective. The explicit `"joint"`
mode restores the legacy trace-first joint geometry. The endpoint observable
is always the selected bare onsite monomial. Physical coefficients such as
`U`, `gamma`, `F`, `Delta`, and hopping amplitudes occur only in its exact
adjoint-Lindblad right-hand side `L^dagger O`.

### Window integration

For one window `[t_a,t_b]` with signed duration `Delta T`, let
`N=residual_gmm_integrator_nodes` and use equally spaced nodes
`t_j=t_a+j Delta T/(N-1)`. The supported closed Newton--Cotes rules are

| `N` | normalized weights | subintervals | formal degree |
|---:|---|---:|---:|
| 3 | `(1,4,1)/6` | 2 | 3 |
| 4 | `(1,3,3,1)/8` | 3 | 3 |
| 5 | `(7,32,12,32,7)/90` | 4 | 5 |
| 6 | `(19,75,50,50,75,19)/288` | 5 | 5 |

The default is six nodes. `training.N_steps` must be divisible by `N-1` in
ordinary, staged, and segmented schedules. Three and four nodes have the same
formal degree, as do five and six nodes.

For selected monomial `O_q` at site `i`, define normalized walker
contributions

```text
C^L_(q,w,i)(t) = u_w(t) P_[O_(q,i)],w(t),
C^R_(q,w,i)(t) = u_w(t) P_[L^dagger O_(q,i)],w(t),
u_w(t)          = Omega_phys,w(t) / mean_v Omega_phys,v(t).
```

The window residual contribution is

```text
r_(q,w,i) = [C^L_(q,w,i)(t_(N-1)) - C^L_(q,w,i)(t_0)
             - Delta T * sum_j w_j C^R_(q,w,i)(t_j)] / |Delta T|.
```

Each window is a local consistency test. Residuals are not accumulated from
the rollout start or propagated between windows.

### Trace preservation

The SDE solver stores the physical weight in split form,

```text
Omega_phys = exp(lnOmega_centered + accumulated_center_shift).
```

For a window starting at `t_k`, the identity contribution uses one fixed
start-of-window reference. Its residual is therefore the physical endpoint
trace drift divided by the window duration; it does not become identically zero
through snapshot-by-snapshot normalization.

There is one physical trace equation. It is broadcast across the site axis so
the raw `(channel,walker,site)` diagnostic cloud has a stable trace-first shape;
broadcasting does not create independent physical trace equations. In default
`diagnostic` mode it is sliced from the active cloud before covariance and
objective calculations. In `joint` mode it participates in the residual mean,
site-averaged covariance, and cross-covariances with the monomial channels.

### Site-averaged covariance geometry

The residual clouds have shape

```text
(complex_channel, walker, site).
```

At each site, active complex channels are packed in interleaved order. In the
default mode this is

```text
[Re(O_0), Im(O_0), Re(O_1), Im(O_1), ...].
```

In `joint` mode `[Re(trace), Im(trace)]` is prepended. Raw diagnostic histories
remain trace first in either mode.

The covariance uses the exact population covariance of self-normalized
ratio-estimator influence contributions. The same endpoint-minus-integral
operation used for the forward cloud is applied to the influence cloud, while
paired walker identity is retained across quadrature nodes. Let
`x_(k,w,i)` be the resulting active real influence vector for window `k`,
walker `w`, and site `i`, and let `mu_(k,i)` be the corresponding active
forward residual mean. With `N_w` walkers and `N_s` sites, the within-site and
shared covariances are

```text
xbar_(k,i)  = (1/N_w) sum_w x_(k,w,i),
Sigma_(k,i) = (1/N_w) sum_w
                (x_(k,w,i)-xbar_(k,i))(x_(k,w,i)-xbar_(k,i))^T,
SigmaBar_k  = (1/N_s) sum_i Sigma_(k,i).
```

Thus `SigmaBar_k` is an average of within-site population covariances. It is
not a covariance over pooled site samples, so differences between site means
do not add a between-site scatter term. The active forward residual mean and
covariance estimate both use all walkers; `q_winsor` does not enter this loss.
There is no additional division by `N_w` after forming `SigmaBar_k`.

One shared covariance whitens every site in a window. With the active real
channel dimension `D` defined above, the regularization is

```text
SigmaBar_floor,k = SigmaBar_k + cov_floor I,
A_k              = diag(SigmaBar_floor,k)^(1/2),
CBar_k           = A_k^(-1) SigmaBar_floor,k A_k^(-1),
CBar_reg,k       = (1-shrinkage) CBar_k
                   + shrinkage trace(CBar_k)/D I + jitter I.
```

The implementation Cholesky-whitens with `CBar_reg,k` after scaling by
`A_k^(-1)`. Equivalently, this defines one shared precision matrix `PBar_k`.
All configured onsite monomial channels remain in the objective. The trace is
also active only in `joint` mode. Covariance quantities are stop-gradient
preconditioners.

A lagged covariance bank is updated throughout each stage after every accepted
optimizer update. An uninitialized window is scored with its current stopped
`SigmaBar_k` estimate; after initialization, the score uses the preceding
accepted bank. Before the EMA update (decay `0.95`), generalized eigenvalues of
the new estimate relative to that preceding bank, regularized by
`residual_gmm_cov_floor`, are clipped to `[0.25, 4]`.
This limits the effect of one anomalous rollout while allowing the covariance
geometry to continue adapting as the policy changes. Skipped updates do not
modify the bank. Shapes are

```text
ordinary:  (window, D, D)
segmented: (segment, window, D, D)
```

with matching Boolean initialization masks that omit the final matrix axes.

The covariance-update controls are internal defaults and are not part of the
training configuration. The active forward mean is not clipped. Its gradient
contribution is clipped walkerwise at Mahalanobis radius
`residual_gmm_d_clip`, using a
straight-through construction so the reported forward residual still includes
every walker in every active channel. Lagging removes an instantaneous
incentive to inflate covariance, while the spectral-ratio bound limits movement
of the normalizer. The full raw residual history (including trace),
active-channel radii, and physical observables should still be inspected
alongside the normalized objective.

For each window, every site mean uses the same `PBar_k`. Define the individual
site/window quadratic

```text
q_(k,i) = mu_(k,i)^T PBar_k mu_(k,i),
L_k     = (1/N_s) sum_i q_(k,i).
```

Window scores are then reduced using

```text
residual_gmm_time_aggregation = "mean" | "log1p" | "entropic" | "entropic_log1p"
residual_gmm_time_beta = 2.0
```

The modes have different, intentional orders of operation:

```text
mean:           average L_k over windows
log1p:          average (1/N_s) sum_i log1p(q_(k,i)) over windows
entropic:       entropic window aggregate of L_k
entropic_log1p: entropic window aggregate of log1p(L_k)
```

Thus the plain `"log1p"` mode transforms every individual site score in every
window before averaging either sites or windows. It is not `log1p` of a
site-averaged window score and is not `log1p` of the complete loss. The legacy
`"entropic_log1p"` mode instead first averages `q_(k,i)` over sites to form
`L_k`, then applies `log1p(L_k)` before its entropic window aggregation.
`residual_gmm_time_beta` is active only for the two entropic modes.

`loss_residual_gmm` records the weighted mean of the untransformed quadratic
window scores `L_k` in every mode.
`loss_residual_gmm_time` records the configured time aggregate and is the value
multiplied by `loss_residual_gmm_prefactor`. The loss has one aggregate
prefactor and no per-channel prefactors.

### Multi-device semantics

Walker sharding must not turn devices into independent residual ensembles.
Distributed sums and cross-products first produce the global walker mean and
population covariance at each site. The site covariances are then averaged to
obtain the single `SigmaBar_k` used by every device. Maxima use global maximum
reductions. The shared covariance and diagnostics are identical on every
device before gradients are reduced. Covariance banks remain unsharded and
retain the shapes above, so running on six devices changes walker placement but
not the scientific definition.

## Pareto-k Envelope Penalty

Pareto-k acts on onsite self-normalized observable contribution clouds

```text
C_(w,i)^(m,n)
  = [Omega_w / mean_v Omega_v] beta_(w,i)^m alpha_(w,i)^n,
```

corresponding to `<(a_i^dagger)^m a_i^n>`. Select clouds by total order,

```text
pareto_k_applied_quantities = P_K
pareto_k_applied_quantities_mode = "upto" | "exact"
```

or with a free list:

```text
pareto_k_monomials = [[0,1], [1,0], [1,1], [2,2]]
```

`pareto_k_monomials = "auto"` selects nonidentity onsite moments appearing in
the configured projected-residual equations and their direct right-hand sides.
Explicit lists override the total-order selector.

Each selected site cloud is robustly centered and covariance-normalized. The
upper radial sample determines a Pareto-k envelope penalty. The aggregate is

```text
loss_pareto_k_prefactor * sum_(selected (m,n)) L_K(m,n),
```

where each `L_K(m,n)` is formed sitewise before averaging. Controls include

```text
pareto_k_threshold
pareto_k_threshold_tau
pareto_k_tail_fraction
pareto_k_min_tail_count
pareto_k_envelope_beta
pareto_k_envelope_excess = "log" | "ratio"
q_winsor
```

`q_winsor` belongs to the Pareto-k cloud statistics only. Pareto-k may be used
as a training regularizer or, with a zero prefactor, as a simulation health
diagnostic for requested observable clouds.

## Gauge and Parameter Regularization

The gauge-output penalty is one objective with two reported components:

```text
loss_gauge
loss_gauge_drift
loss_gauge_diffusion
```

`loss_gauge_prefactor` multiplies the total. Drift and diffusion values are
diagnostic decompositions, not separately configurable loss families.

`loss_L2_prefactor` multiplies the squared neural-parameter norm reported as
`loss_L2`. Setting the prefactor to zero disables it explicitly.

Drift and diffusion gauges can also have independent smooth output bounds:

```json
"bound_drift": true,
"bound_diffusion": true
```

which apply

```text
G      = drift_max * tanh(G_raw / drift_max),
lambda = diffusion_max * tanh(lambda_raw / diffusion_max).
```

These architectural bounds are distinct from the gauge-output loss.

## Weight-Entropy (Complex-ESS) Budget Penalty

The drift gauge is a Girsanov control: its log-weight update

```text
d lnOmega = g.dW + f.dW' - (g^2 + f^2)/2 dt
```

spends walker weight entropy at the rate `E[|g|^2 + |f|^2]`.  The complex
effective sample size

```text
ESS_c = |sum_w Omega_w|^2 / sum_w |Omega_w|^2
      ~= N exp(-Var(Re lnOmega) - Var(Im lnOmega))
```

decays exponentially in the accumulated spread, so the total budget before
ensemble degeneracy is `ln(num_walker)` nats.  The projected-residual
objective is covariance-whitened and therefore scale-invariant in its own
sampling noise: it enforces boundary-term bias but cannot price this
statistical cost.  The ESS penalty adds the missing price as a one-sided
budget constraint.

Per training window `k`, the spend is

```text
S_k = Var_w(Re Delta lnOmega_k) + Var_w(Im Delta lnOmega_k)
```

(the walker variance of the window's complex log-weight increment; the
solver's common real centering shift cancels in the variance).  With the
uniform per-window budget

```text
b = ln(num_walker) / N_windows_trained
```

the penalty is the dimensionless hinge

```text
loss_ess = mean_k [ max(0, S_k / b - 1) ]^2,
loss     = ... + loss_ess_prefactor * loss_ess.
```

Below budget the hinge is exactly zero and the penalty cannot interfere
with the residual physics; above budget `loss_ess_prefactor` acts as an
enforcement stiffness rather than a trade-off weight.  The diffusion gauge
never enters `d lnOmega`, so the penalty automatically prices only the
drift gauge and leaves diffusion-based confinement free.

Activation follows the standard rule: the family is inactive by default
(`loss_ess_prefactor = 0.0`) and enabled only when the prefactor magnitude
reaches the shared trace threshold (`>= 1e-8`).  In the segmented trainer
the budget denominator is the stage's effective total trained windows and
`num_walker` is the stage walker count.

Reported metrics (history and dashboard):

```text
loss_ess                  windowwise-mean hinge (unweighted objective)
loss_ess_weighted         prefactor-weighted contribution to the loss
log_weight_spread_mean    window-mean spend S_k in nats
log_weight_spread_max     worst-window spend
log_weight_spread_total   summed spend over the trained horizon
ess_ratio_min             minimum complex ESS_c / N over windows (stopped)
ess_ratio_end             final-window complex ESS_c / N (stopped)
```

The spend `S_k` stays differentiable along the on-policy path through
`lnOmega`; the ESS ratios are stop-gradient diagnostics.

## Joint-U(1)-Invariant Node Features (`model.u1_invariant`)

At any drive the physics is exactly covariant under the joint global
rotation

```text
alpha_i -> e^{i th} alpha_i,   beta_i -> e^{-i th} beta_i,   F -> e^{i th} F,
```

with invariant gauge outputs and invariant lnOmega (verified pathwise for
both solver backends, including noise, hopping, and site-dependent drive).
The optimal drift and diffusion gauges are therefore invariant scalars of
this rotation, while the legacy node features (raw Re/Im and sin/cos of
alpha, beta and the drifts) are covariant: the network must spend capacity
canceling the unphysical global phase, which rotates at rate ~ U n along
trajectories.

`model.u1_invariant` (optional bool, default `false` = the legacy feature
set, so existing checkpoints keep loading unchanged) switches the graph
model's node features to an exactly invariant set:

```text
Re/Im n_i                       n_i = alpha_i beta_i
Re/Im (beta_i A^P_alpha,i)      drift bilinears
Re/Im (alpha_i A^P_beta,i)
Re/Im (beta_i (J alpha)_i)      hopping bilinears
Re/Im (alpha_i (J* beta)_i)
Re/Im (conj(F) alpha_i)         drive spurions (vanish at F = 0)
Re/Im (F beta_i)
log|alpha_i|, log|beta_i|, log|n_i|,
log|A^P_alpha,i|, log|A^P_beta,i|,
degree_i
```

These 20 per-site features determine (alpha_i, beta_i) up to the global
phase — the only direction the rotation removes — and the drive enters
solely through the spurions, i.e. the relative phase between drive and
field. Global features (lnOmega, time) are unchanged; both Re and Im of
lnOmega are themselves invariant. Output heads are unchanged: the gauge
outputs are invariant scalars, so nothing is rotated back.

The two modes have different first-layer widths (20 vs 26 input features),
so loading a checkpoint across modes fails immediately with a parameter
shape error rather than silently misreading weights.

## Neural Gauge Application

By default, a neural gauge is evaluated at the beginning of a training window
and held fixed during that window. Set

```json
"apply_neural_gauge_every_steps": 20
```

to refresh it every 20 SDE microsteps. The state-gradient mode is

```json
"neural_gauge_state_gradient": "full"
```

or

```json
"neural_gauge_state_gradient": "each_apply"
```

`full` retains the recurrent pathwise gradient through both neural parameters
and phase-space inputs. `each_apply` evaluates the network with
`stop_gradient(X)` at every application, retaining direct parameter gradients
but cutting recurrent state-input gradients. The forward trajectory is
unchanged.

`noise_refresh_every` controls how many consecutive epochs reuse one frozen
stochastic rollout batch. It changes training-noise refresh cadence, not the
number of walkers or the residual covariance definition.

## EMA Loss-Scale Normalization

`training.EMA` optionally applies Python-side, stop-gradient magnitude
normalization before enabled loss terms are combined. The projected residual
is normalized only as aggregate `loss_residual_gmm`. Pareto-k can be normalized
as an aggregate or through selected monomial components.

```json
"EMA": {
  "enabled": true,
  "terms": [
    "loss_residual_gmm",
    "loss_pareto_k_p2"
  ],
  "decay": 0.995,
  "warmup_epochs": 10,
  "floor": 1.0e-8,
  "r_max": 5.0
}
```

Supported Pareto aliases are

```text
loss_pareto_k_pP   -> selected Pareto-k monomials with m+n=P
loss_pareto_k_auto -> the auto-selected Pareto-k monomial set
```

An aggregate Pareto-k term and its component terms cannot be normalized
simultaneously. Raw losses and EMA scales remain in history. The internal
residual covariance bank is independent of this optional loss-scale EMA.

## Training History and Plots

The maintained history contains the active subset of

```text
loss
loss_residual_gmm
loss_residual_gmm_time
loss_residual_gmm_raw
loss_residual_gmm_m0_n0
loss_residual_gmm_m*_n*
residual_gmm_z_mean/max/worst
residual_gmm_radius_mean/max/worst
residual_gmm_warning_fraction
residual_gmm_bad_fraction
loss_pareto_k_raw
loss_pareto_k_m*_n*
pareto_k_mean/max/worst
pareto_k_warning_fraction
pareto_k_bad_fraction
loss_gauge
loss_gauge_drift
loss_gauge_diffusion
loss_L2
grads_norm
lr
```

Residual per-channel histories are raw residual squares averaged over sites in
trace-first configuration order. The transient auxiliary tree may contain a
`(channel,site)` breakdown for aggregation, but this array is not persisted in
epoch history. `loss_residual_gmm_raw` is the sum of these full raw diagnostic
terms and therefore includes trace in both modes; it is not the optimized
normalized objective in `diagnostic` mode. This avoids history growth with
lattice size.

EMA-enabled terms additionally record matching scale and normalized values.
`loss` is always the optimized objective after prefactors and optional EMA
normalization. Raw channel terms are diagnostics; their sum is not expected to
equal the covariance-normalized objective.

## Reduced Simulation Observables

`G1`, `G1_initial`, and `g2` save full site-by-site matrices. For larger
lattices, reduced shell observables avoid materializing a
`num_walker x num_site x num_site` tensor:

```json
"observables": [
  "density",
  "G1_local", "G1_nn", "G1_nnn",
  "G1_initial_local", "G1_initial_nn", "G1_initial_nnn",
  "g2_local", "g2_nn", "g2_nnn"
]
```

`local` keeps onsite values; `nn` and `nnn` are directed nearest- and
next-nearest-neighbor shell averages. These estimators are compatible with
chunked walker batches. Set `simulation.save_observable_errors=false` when
error estimates are not required.

## Validation

From the repository root:

```bash
python -m py_compile \
  src/nsgr/dynamics_kernel.py \
  src/nsgr/training.py \
  src/nsgr/multi_device.py \
  src/nsgr/projected_residual.py \
  src/nsgr/simulation.py \
  src/nsgr/utility.py \
  src/nsgr/training_plot.py \
  src/nsgr/lattice.py

python -m unittest discover -s tests -v
```
