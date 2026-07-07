# Primal-Dual Dynamics: Convergence, Autocorrelation, and Variational Structure

*Notes from theoretical discussion, 2026-03-04*

---

## 1. The Ergodic Dual Stationarity Criterion

### 1.1 Setup

The primal-dual trainer solves the Lagrangian relaxation:

```
maximise  D(λ) = min_θ L(θ, λ)
  over    λ ≥ 0
```

where `L(θ, λ) = mean_b[−f_b(θ) + λ_b^T g_b(θ)]`, `g_b = r_min − R_b` are constraint slacks (positive = violated), and `λ ≥ 0` is enforced by projection onto the non-negative orthant. The dual is maximised by projected subgradient ascent:

```
λ_{t+1} = Π_+(λ_t + α · g_t)
```

### 1.2 The Pointwise Criterion and Its Failure Mode

The original convergence criterion averaged the per-epoch **pointwise** projected dual residual:

```
r_t = (1/α) · mean_i |Π_+(λ_t + α·g_t) − λ_t|
criterion = mean(r_t over convergence_window) < threshold
```

This fails in the **cyclo-oscillatory regime**: the per-epoch residual stays large (dual is actively moving) even though the *time-averaged* trajectory satisfies KKT / complementary slackness. The criterion sees "dual is still moving" and refuses to declare convergence, despite the ergodic trajectory being effectively converged.

### 1.3 The Ergodic Criterion

Evaluate the projected residual on **windowed ergodic averages** of λ and g:

```
λ̄ = (1/W) Σ_{t ∈ window} λ_t
ḡ = (1/W) Σ_{t ∈ window} g_t

r_ergodic = (1/α) · mean_i |Π_+(λ̄_i + α·ḡ_i) − λ̄_i|
```

This is motivated by the ergodic convergence guarantee of projected subgradient methods (Nedić & Ozdaglar; Arrow–Hurwicz–Uzawa): `λ̄_T → λ*` even when pointwise iterates cycle.

### 1.4 What the Projected Residual Measures in Each Case

For a single element `i`, the ergodic residual decomposes as:

| Condition | Formula | Interpretation |
|-----------|---------|----------------|
| `λ̄_i + α·ḡ_i ≥ 0` (interior) | `r_i = \|ḡ_i\|` | Residual = mean constraint slack magnitude; independent of `α` |
| `λ̄_i + α·ḡ_i < 0` (boundary clip) | `r_i = λ̄_i / α` | Residual ∝ `λ̄_i/α`; at boundary `λ̄_i ∝ α·σ_g`, so again ≈ `σ_g` |

The `1/α` normalisation makes the criterion **scale-free** at steady state. This is its main strength — and its main limitation as a stability diagnostic.

---

## 2. Role of the Dual Step Size α

### 2.1 Steady-State Regime (Long Window)

In steady state, the `1/α` normalisation absorbs the amplitude scaling of the dual oscillations:

- **Interior points**: residual = `|ḡ|` exactly, independent of `α`.
- **Boundary points**: amplitude of `λ_t` scales as `α·σ_g`; after `1/α` normalisation, residual ≈ `σ_g`. Also approximately independent of `α`.

**Conclusion**: for sufficiently long windows, the ergodic residual converges to the same value regardless of oscillation amplitude. The criterion is a first-moment diagnostic; it cannot distinguish a system quietly sitting near the saddle from one swinging wildly around it.

### 2.2 Transient Regime (Short Window)

With more aggressive `α`, the dual iterates converge faster toward the saddle-point neighbourhood (convergence time scale `τ(α) ∝ 1/α`). For a finite window `W`:

- If `τ(α) ≪ W`: window is mostly in steady state; residual ≈ independent of `α`.
- If `τ(α) ~ W`: significant transient fraction contaminates the window; larger `α` → smaller transient bias → smaller ergodic residual.

**Conclusion**: the hypothesis that "more aggressive α yields smaller ergodic residuals" holds in the transient regime through faster convergence, but vanishes as `W → ∞`.

### 2.3 Higher-Order Moments and Finite-Window Variance

The scale-free property applies only to the **first moment** (mean) of the ergodic residual. The **variance** and higher moments depend on `α` through the boundary rectification effect:

- At the projection boundary (`λ ≈ 0`), `λ_t` is a half-rectified process: upswings ∝ `α` are preserved, downswings are clipped. This introduces skewness and excess kurtosis in `λ̄` that grow with `α`.
- The `1/α` normalisation cancels the mean bias but does **not** cancel the higher-moment structure.
- Result: larger `α` → heavier-tailed residual distributions → more variable convergence detection across runs.

**Bias–variance tradeoff for finite windows:**
- Small oscillations (small `α`): small ergodic residual variance; slow transient convergence.
- Large oscillations (large `α`): faster transient convergence, better mean residual in short windows; but higher variance in residual estimates and more variable convergence detection.

---

## 3. Autocorrelation Profile of Primal-Dual Dynamics

### 3.1 Linearised Dynamics Near the Saddle Point

Expanding around `(θ*, λ*)` and ignoring the projection (interior, `λ* > 0`), the coupled update is:

```
δθ_{t+1} = (I − ηH) δθ_t − η J^T δλ_t
δλ_{t+1} =      α J δθ_t  +       δλ_t
```

where `H = ∇²_θθ L` is the Lagrangian Hessian and `J = ∂g/∂θ` is the constraint Jacobian. In block form:

```
M = [ I − ηH,  −ηJ^T ]
    [    αJ,       I  ]
```

`M` is **non-symmetric**, reflecting the min-max (saddle-point) structure. Its eigenvalues are generically **complex**, producing oscillatory dynamics.

### 3.2 Scalar Case

For `θ, λ ∈ ℝ`:

```
μ = (2 − ηh)/2  ±  i · √(4αηj² − η²h²) / 2
```

Complex eigenvalues arise when `αη‖J‖² > η²‖H‖²/4`, i.e., when the dual step is aggressive relative to the primal curvature.

**Stability**: `|μ|² = 1 − ηh + αηj² < 1` requires `α < h/j²`.

### 3.3 Autocorrelation Function Shape

Since the linearised dynamics have eigenvalues `μ = |μ| e^{±iω}`, the autocorrelation function of the dual deviations `δλ_t` is:

```
R(τ) ∝ |μ|^τ · cos(ωτ)
```

This is a **damped cosine** — exponential decay modulated by oscillation — with:

| Quantity | Expression | Dependence on α |
|----------|-----------|-----------------|
| Oscillation frequency | `ω ∝ √(αη)` | Increases with α |
| Decay rate | `1/τ_c ∝ ηh − αηj²` | Larger α → slower decay |
| Correlation time | `τ_c ∝ 1/(ηh − αηj²)` | Increases with α |
| Stability condition | `α < h/j²` | — |

### 3.4 Key Consequences for Ergodic Averaging

**Negative autocorrelation lobes** (the cosine going negative) are not a pathology — they help the ergodic average. Anti-correlated consecutive samples partially cancel, accelerating convergence of `λ̄` relative to a purely positively-correlated process.

**Effective sample size** in a window `W`: approximately `W_eff ≈ W / τ_c`. Since `τ_c` grows with `α`, more aggressive step sizes produce *fewer* independent samples per window, increasing variance of the ergodic average.

**At the projection boundary**, the linear analysis breaks down. The rectification `Π_+` produces a half-rectified process with:
- Strictly positive autocorrelation (no negative lobes) — the asymmetry kills anti-correlation.
- Faster initial decay but heavier tail.
- Loss of the ergodic-averaging benefit from anti-correlation.

This is where `α`-dependent higher-order effects are most pronounced, consistent with the second-moment analysis above.

---

## 4. Variational Formulation Over Joint Distributions

### 4.1 The Stochastic Primal-Dual System

Adding noise to regularise:

```
dθ = −∇_θ L(θ, λ) dt + √(2/β_θ) dW_θ
dλ = +∇_λ L(θ, λ) dt + √(2/β_λ) dW_λ
```

where `β_θ = η/σ²` and `β_λ = α/σ²` are inverse temperatures (the dual temperature is `σ²/α`).

### 4.2 Factored (Mean-Field) Variational Problem

Restricting to product distributions `q = q_θ ⊗ q_λ`, the stationary marginals are the saddle point of:

```
min_{q_θ} max_{q_λ}  E_{q_θ ⊗ q_λ}[L(θ, λ)]  +  (σ²/η) H[q_θ]  −  (σ²/α) H[q_λ]
```

where `H[q] = −∫ q log q` is differential entropy. The saddle-point conditions give:

```
q_θ*(θ) ∝ exp(−β_θ · E_{q_λ*}[L(θ, λ)])
q_λ*(λ) ∝ exp(+β_λ · E_{q_θ*}[L(θ, λ)])
```

**Interpretation**: this is the original min-max problem with **entropic regularisation**. The step sizes `(η, α)` appear as inverse temperatures governing the spread of the primal and dual Gibbs distributions. Larger `α` → larger `β_λ` → `q_λ*` concentrates more sharply around `λ*` → stronger constraint enforcement, but larger oscillation amplitude.

### 4.3 The Linearised (Gaussian) Regime

In the Gaussian approximation near the saddle, the stationary distribution is `N(0, Σ)` where `Σ` satisfies the **discrete Lyapunov equation**:

```
M Σ M^T − Σ = −Q
```

`M` is the linearisation matrix (containing `η` and `α`); `Q` is the noise covariance. The corresponding variational problem is **maximum entropy subject to the Lyapunov constraint**:

```
max_{q}  H[q]    subject to    E_q[zz^T] = Σ(η, α)
```

The solution is the Gaussian with covariance `Σ(η, α)`. The parameters `(η, α)` enter through `M` and determine which second-moment structures are admissible — i.e., `α` deforms the feasible set of the variational problem, not just the objective.

### 4.4 Why the Full Joint Distribution Is Harder

The full joint stationary distribution `q(θ, λ)` does **not** generally admit a clean variational characterisation. The reason: the primal-dual drift `F = (−∇_θ L, +∇_λ L)` lacks **detailed balance** — at stationarity there is a persistent probability current circulating around the saddle point. This current is the continuous-time analogue of the oscillatory behaviour visible in the autocorrelation function. Systems with persistent currents cannot be minimisers of any standard free energy functional.

The linearised Gaussian case is special because the Lyapunov constraint absorbs the current structure into the admissible covariance set, restoring a clean maximum-entropy characterisation.

### 4.5 Summary: Role of α Across the Three Formulations

| Formulation | Role of α |
|-------------|-----------|
| Factored min-max | Dual inverse temperature `β_λ = α/σ²`; controls concentration of `q_λ*` |
| Gaussian / Lyapunov | Enters `M`; deforms the feasible second-moment set |
| Full Fokker-Planck | Scales the persistent probability current; no clean variational dual |

---

## 5. Connections and Open Questions

1. **Ergodic criterion as first-moment diagnostic**: the windowed projected residual `r_ergodic` captures whether `(λ̄, ḡ)` is near a KKT point, but is blind to the stability (variance) of that neighbourhood. A natural complement would be a second-moment criterion based on the variance of `λ_t` within the window — effectively a spectral test against the expected Lyapunov-equation covariance.

2. **Optimal window size**: the bias-variance tradeoff for the ergodic criterion is governed by `τ_c(α)`. The optimal window `W* ∝ τ_c` trades off transient bias against estimation variance. In the current implementation `convergence_window` is a fixed hyperparameter; in principle it could be adapted based on an online estimate of `τ_c`.

3. **Boundary effects and the projection**: the half-rectification of `Π_+` is the main source of `α`-dependent higher-order behaviour. A projected Ornstein–Uhlenbeck analysis (reflecting boundary at 0) would give closed-form expressions for the asymmetric autocorrelation and the positive bias `E[λ̄] − λ*` at boundary-active constraints.

4. **Non-factored variational problem**: characterising the full joint stationary distribution via a variational principle for systems with persistent currents is an active area (e.g., GENERIC framework, entropy production minimisation). Connecting this to primal-dual optimisation dynamics is an open question.
