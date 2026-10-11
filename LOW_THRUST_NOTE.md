# Low-thrust legs in Star

Star treats continuous thrust as a small perturbation of a ballistic Lambert trajectory. It estimates changes in departure and arrival velocities while retaining the same encounter positions and times. Each leg assumes two-body gravity; encounters are connected through patched conics.

The method chooses a thrust direction from an endpoint-speed objective and computes linear and quadratic responses for reuse across acceleration samples. Analytic reference fields and finite-node quadrature provide the estimates; nonlinear propagation replays selected controls to measure endpoint errors.

## Problem and endpoint estimate

Fix the departure and arrival positions and epochs, $(\mathbf R_D,t_0)$ and $(\mathbf R_A,t_f)$. Let $\mathbf r_{\mathrm{ref}}(t)$ be the ballistic Lambert trajectory connecting them. Subscripts D and A denote departure and arrival, with $t_D=t_0$ and $t_A=t_f$.

Apply thrust with a constant acceleration magnitude $a\geq0$ over $T=t_f-t_0$, with a time-dependent unit direction $\mathbf u(t)$: $\mathbf a(t)=a\mathbf u(t)$. Departure velocity adjusts to preserve the fixed encounters through quadratic order. Nonlinear replay can still have a terminal position error.

The quantity stored for each endpoint is the spacecraft velocity relative to its encounter body:

$$
\mathbf v_{\infty,k}(a)=\dot{\mathbf r}(t_k,a)-\mathbf v_{\mathrm{body},k},
\qquad k\in\{D,A\}.
$$

The encounter-body velocities are fixed, so changes in spacecraft endpoint velocities are also changes in $\mathbf v_\infty$. The search estimates these vectors as

$$
\mathbf v_{\infty,k}(a)\approx\mathbf v_{\infty,k}^{(0)}
+a\,\mathbf s_k+a^2\,\mathbf q_k,
$$

where $\mathbf v_{\infty,k}^{(0)}$ is the ballistic value, $\mathbf s_k$ is the linear coefficient, and $\mathbf q_k$ is the quadratic coefficient. The quadratic term captures gravity's nonlinear response to displacement from the reference. [In tested cases, we empirically observe much smaller terminal position errors in nonlinear replay when departure velocity includes the quadratic correction, compared with the linear estimate alone.]

Both coefficients are computed once per reference and thrust-direction history; each acceleration sample then requires only vector arithmetic.

## Choosing the thrust direction

### Endpoint-speed objective

The direction is chosen to maximize a signed difference of endpoint speeds. Star retains the following public sign convention:

| Endpoint-speed preference | Objective J to maximize |
| --- | --- |
| `+-`: departure speed minus arrival speed | $\|\mathbf v_{\infty,D}\|-\|\mathbf v_{\infty,A}\|$ |
| `-+`: arrival speed minus departure speed | $\|\mathbf v_{\infty,A}\|-\|\mathbf v_{\infty,D}\|$ |

The `+-` preference is often useful for pump-down tours, while `-+` favors the reverse speed trade for pump-up tours.

Define the endpoint gradients on the ballistic reference by

$$
\boldsymbol\gamma_D=\frac{\partial J}{\partial\mathbf v_{\infty,D}},
\qquad
\boldsymbol\gamma_A=\frac{\partial J}{\partial\mathbf v_{\infty,A}}.
$$

For nonzero endpoint speeds, `+-` gives $\boldsymbol\gamma_D=\mathbf v_{\infty,D}^{(0)}/\|\mathbf v_{\infty,D}^{(0)}\|$ and $\boldsymbol\gamma_A=-\mathbf v_{\infty,A}^{(0)}/\|\mathbf v_{\infty,A}^{(0)}\|$; `-+` reverses both signs. Either preference can trade improvement at one endpoint against deterioration at the other. This objective does not establish minimum-fuel optimality.

### Sensitivity along the arc: the primer vector

Two-body gravitational acceleration is $\mathbf g(\mathbf r)=-\mu\mathbf r/\|\mathbf r\|^3$. A small thrust acceleration produces a first-order position variation about the Lambert reference:

$$
\delta\ddot{\mathbf r}(t)=G(t)\delta\mathbf r(t)+\mathbf a(t),
\qquad \delta\mathbf r(t_0)=\delta\mathbf r(t_f)=0.
$$

Here $G(t)=\left.\partial\mathbf g/\partial\mathbf r\right|_{\mathbf r_{\mathrm{ref}}(t)}$ is the $3\times3$ gravity Jacobian.

The **primer vector** $\mathbf p(t)$ expresses the endpoint objective's sensitivity to thrust along the arc. Define it by the adjoint boundary-value problem

$$
\ddot{\mathbf p}(t)=G(t)^T\mathbf p(t),
\qquad \mathbf p(t_0)=-\boldsymbol\gamma_D,
\qquad \mathbf p(t_f)=\boldsymbol\gamma_A.
$$

Integration by parts cancels the interior terms through the adjoint equation. Using the fixed-position boundary conditions gives

$$
\begin{aligned}
\int_{t_0}^{t_f}\mathbf p(t)^T\mathbf a(t)\,dt
&=\left[\mathbf p(t)^T\delta\dot{\mathbf r}(t)
-\dot{\mathbf p}(t)^T\delta\mathbf r(t)\right]_{t_0}^{t_f}\\
&=\boldsymbol\gamma_D^T\delta\dot{\mathbf r}(t_0)
+\boldsymbol\gamma_A^T\delta\dot{\mathbf r}(t_f)
=\delta J.
\end{aligned}
$$

At a prescribed magnitude $a$, aligning thrust with $\mathbf p(t)$ maximizes the integral and hence the first-order objective change $\delta J$:

$$
\mathbf u(t)=\frac{\mathbf p(t)}{\|\mathbf p(t)\|},
\qquad \mathbf a(t)=a\mathbf u(t).
$$

The normalization applies where $\|\mathbf p(t)\|>0$; an isolated zero has no preferred direction. The direction history is computed once on the Lambert reference and held fixed across acceleration samples: $\mathbf u(t)$ changes with time, but does not change when $a$ is varied.

## Position response and gravity curvature

### Taylor expansion of position and gravity

For this fixed direction history, write the perturbed position as a local expansion in the scalar acceleration magnitude:

$$
\mathbf r(t,a)=\mathbf r_{\mathrm{ref}}(t)
+a\,\mathbf r_1(t)+a^2\,\mathbf r_2(t)+O(a^3),
$$

where

$$
\mathbf r_1(t)=\left.\frac{\partial\mathbf r(t,a)}{\partial a}\right|_{a=0},
\qquad
\mathbf r_2(t)=\frac12\left.\frac{\partial^2\mathbf r(t,a)}{\partial a^2}\right|_{a=0}.
$$

These are time-dependent vector coefficients of an expansion in $a$. The actual position corrections are $a\mathbf r_1(t)$ and $a^2\mathbf r_2(t)$.

Gravity's linear response to displacement is described by G. Its quadratic response uses **H, a $3\times3\times3$ tensor of second position derivatives**. Each gravity component $g_i$ has a $3\times3$ Hessian $H_i(t)$, with entries $[H_i(t)]_{j\ell}=H_{ij\ell}(t)$. "Third-order tensor" counts the three indices; the derivatives are second order.

Contracting both position indices with $\mathbf x$ produces a vector of three quadratic forms:

$$
H(t)[\mathbf x,\mathbf x]=
\begin{bmatrix}
\mathbf x^T H_1(t)\mathbf x\\
\mathbf x^T H_2(t)\mathbf x\\
\mathbf x^T H_3(t)\mathbf x
\end{bmatrix},
\qquad
\bigl(H(t)[\mathbf x,\mathbf x]\bigr)_i
=\sum_{j,\ell}H_{ij\ell}(t)x_jx_\ell.
$$

The gravity Taylor expansion in a displacement $\delta\mathbf r$ at a fixed reference time is

$$
\mathbf g(\mathbf r_{\mathrm{ref}}(t)+\delta\mathbf r)
=\mathbf g(\mathbf r_{\mathrm{ref}}(t))
+G(t)\delta\mathbf r
+\tfrac12 H(t)[\delta\mathbf r,\delta\mathbf r]
+O(\|\delta\mathbf r\|^3).
$$

### Response equations and endpoint velocities

Substitute the position and gravity expansions into $\ddot{\mathbf r}(t,a)=\mathbf g(\mathbf r(t,a))+a\mathbf u(t)$. The ballistic terms cancel; collecting the coefficients of $a$ and $a^2$ gives

$$
\begin{aligned}
\ddot{\mathbf r}_1(t)&=G(t)\mathbf r_1(t)+\mathbf u(t),\\
\ddot{\mathbf r}_2(t)&=G(t)\mathbf r_2(t)+\mathbf f_2(t),
\qquad \mathbf f_2(t)=\tfrac12 H(t)[\mathbf r_1(t),\mathbf r_1(t)],\\
\mathbf r_m(t_0)&=\mathbf r_m(t_f)=0\quad(m=1,2).
\end{aligned}
$$

The first equation gives the linear response; the second adds gravity's curvature. No thrust-direction derivative appears because $\mathbf u(t)$ is fixed as $a$ varies.

The fixed encounter positions make both position coefficients vanish at the endpoints. Their time derivatives give the velocity coefficients:

$$
\mathbf s_k=\dot{\mathbf r}_1(t_k),
\qquad \mathbf q_k=\dot{\mathbf r}_2(t_k),
\qquad k\in\{D,A\}.
$$

The API returns true first and second derivatives at $a=0$: $\mathbf s_{1,k}=\mathbf s_k$ and $\mathbf s_{2,k}=2\mathbf q_k$. The leg builder uses $\mathbf v_{\infty,k}^{(0)}+a\mathbf s_{1,k}+\tfrac12a^2\mathbf s_{2,k}$. The factor of two follows from the definition of $\mathbf r_2(t)$.

## Computing the response with quadrature

The generator evaluates the response using an analytic conic basis and finite-node quadrature, without numerical ODE propagation. Physical replay uses nonlinear ODE propagation separately.

[`star/low_thrust.py`](star/low_thrust.py) reuses the homogeneous conic solutions in [`star/maneuver_placement.py`](star/maneuver_placement.py). A 4×4 in-plane system and a 2×2 normal system impose the endpoint values for each primer. Shared geometry supplies the objective primer and six endpoint-component primers. G is implicit in the basis; the contracted Hessian forcing is evaluated explicitly. The implementation requires nondegenerate elliptic references with $10^{-8}<e\leq0.99$.

### Quadrature nodes and endpoint responses

Let $\nu$ be the reference's true anomaly, unwrapped across additional revolutions, and let h be its conserved specific angular momentum magnitude. Time integration uses

$$
dt=\frac{\|\mathbf r_{\mathrm{ref}}(t)\|^2}{h}\,d\nu,
\qquad h=\|\mathbf r_{\mathrm{ref}}(t)\times\dot{\mathbf r}_{\mathrm{ref}}(t)\|.
$$

The default grid has $N=400(1+|n_{\mathrm{rev}}|)$ intervals and $N+1$ equally spaced true-anomaly nodes:

$$
\nu_\ell=\nu_0+\ell\Delta\nu,
\qquad \Delta\nu=\frac{\nu_f-\nu_0}{N},
\qquad \rho_\ell=\frac{p_{\mathrm{orb}}}{1+e\cos\nu_\ell}.
$$

Here e is the reference eccentricity, $p_{\mathrm{orb}}$ is its semilatus rectum, and $\rho_\ell$ is its radius at the node. The orbital quantity $p_{\mathrm{orb}}$ is distinct from the primer vector $\mathbf p(t)$.

Let $t_\ell$ be the reference time corresponding to $\nu_\ell$, and let $w_\ell$ be the composite Simpson weights in true anomaly: $\Delta\nu/3$ times $(1,4,2,4,\ldots,2,4,1)$ for this even N. For a vector or scalar integrand F,

$$
\int_{t_0}^{t_f}F(t)\,dt
\approx\sum_{\ell=0}^{N}w_\ell\,\frac{\rho_\ell^2}{h}\,F(t_\ell).
$$

For example, let $\mathbf p_{k,j}(t)$ be the primer for the objective $J=v_{\infty,k,j}$, Cartesian component j of the velocity at endpoint $k\in\{D,A\}$. Applying the same sensitivity identity to the first- and second-order forcing gives

$$
\begin{aligned}
s_{k,j}&=\int_{t_0}^{t_f}\mathbf p_{k,j}(t)^T\mathbf u(t)\,dt
\approx\sum_{\ell=0}^{N}w_\ell\frac{\rho_\ell^2}{h}
\mathbf p_{k,j}(t_\ell)^T\mathbf u(t_\ell),\\
q_{k,j}&=\int_{t_0}^{t_f}\mathbf p_{k,j}(t)^T\mathbf f_2(t)\,dt
\approx\sum_{\ell=0}^{N}w_\ell\frac{\rho_\ell^2}{h}
\mathbf p_{k,j}(t_\ell)^T\mathbf f_2(t_\ell).
\end{aligned}
$$

The first pass uses cumulative quadrature and an endpoint correction to reconstruct $\mathbf r_1(t_\ell)$. The second evaluates $\mathbf f_2(t_\ell)$ and integrates its endpoint response. The fields are analytic and the integrals numerical; neither pass reconstructs $\mathbf r_2(t)$.

### Analytic gravity derivatives used by the implementation

At a given reference time, write $\mathbf R=\mathbf r_{\mathrm{ref}}(t)$ and $\rho=\|\mathbf R\|$. Differentiating $g_i=-\mu R_i\rho^{-3}$ gives

$$
\begin{aligned}
G_{ij}(t)&=-\frac{\mu}{\rho^3}\delta_{ij}
+\frac{3\mu}{\rho^5}R_iR_j,\\
G(t)&=\mu\left(\frac{3\mathbf R\mathbf R^T}{\rho^5}
-\frac{I_3}{\rho^3}\right).
\end{aligned}
$$

Here $I_3$ is the identity matrix and $\delta_{ij}$ is the Kronecker delta (1 if $i=j$, otherwise 0). G is symmetric and depends only on the reference position. Its units are $\mathrm{s}^{-2}$ when position is in km and $\mu$ is in $\mathrm{km}^3/\mathrm{s}^2$. Differentiating once more gives the tensor entries

$$
H_{ij\ell}(t)=\frac{\partial G_{ij}}{\partial R_\ell}
=\frac{3\mu}{\rho^5}
\left(\delta_{ij}R_\ell+\delta_{i\ell}R_j+\delta_{j\ell}R_i\right)
-\frac{15\mu}{\rho^7}R_iR_jR_\ell.
$$

With $\mathbf x=\mathbf r_1(t)$, the curvature forcing can be evaluated directly without forming the full tensor:

$$
\begin{aligned}
\mathbf f_2(t)
&=\tfrac12 H(t)[\mathbf x,\mathbf x]\\
&=\frac{3\mu}{\rho^5}\mathbf x(\mathbf R^T\mathbf x)
+\frac{3\mu}{2\rho^5}\mathbf R\|\mathbf x\|^2
-\frac{15\mu}{2\rho^7}\mathbf R(\mathbf R^T\mathbf x)^2.
\end{aligned}
$$

The kernel evaluates this vector in the reference's radial, transverse, and normal (RTN) frame. At this time, let $(x_R,x_T,x_N)$ be the components of $\mathbf r_1(t)$. Since $\mathbf R=(\rho,0,0)$ in that frame,

$$
\begin{aligned}
f_{2,R}(t)&=\frac{3\mu}{2\rho^4}\left(x_T^2+x_N^2-2x_R^2\right),\\
f_{2,T}(t)&=\frac{3\mu}{\rho^4}x_Rx_T,\\
f_{2,N}(t)&=\frac{3\mu}{\rho^4}x_Rx_N.
\end{aligned}
$$

These are the `fr`, `ft`, and `fn` expressions in `_integrate_lt_quadrature_jit`; its scalar `gravity` is $\mu/\rho^4$, using $h^2=\mu p_{\mathrm{orb}}$. G and H are differentiated analytically, so no finite-difference approximation is needed.

### Basis reconstruction and fixed-endpoint response

Since G is symmetric, the conic primer basis also spans homogeneous position variations. Its constant Wronskian, formed from basis functions and their time derivatives, converts cumulative thrust integrals into a forced $\mathbf r_1(t)$ with zero initial displacement. Adding a homogeneous solution from the arrival-component primers cancels the final displacement and supplies the departure-velocity adjustment.

Projecting the curvature forcing through the same endpoint primers gives $\mathbf q_D$ and $\mathbf q_A$ without another boundary solve.

### Code path and batching

Internally, `Jlev = -J` and `dJlev_da_s` represents $-dJ/da$. The objective-gradient signs give the same thrust-direction primer. The code uses opposite-sign endpoint-component primers with compensating negative integration signs, yielding the velocity coefficients defined above.

| Step | Implementation | Result |
| --- | --- | --- |
| Reference and primers | `prepare_primer_geometry`, `solve_primer_coefficients` in `maneuver_placement.py` | Shared conic geometry and six-constant primer fields |
| Endpoint responses | `solve_arc_low_thrust` → `_integrate_lt_quadrature_jit` in `low_thrust.py` | Linear sensitivities and second derivatives for each seed and steering choice |
| Throttle samples | `_process_lambert_chunk` in [`leg_database.py`](star/leg_database.py) | Vectorized quadratic endpoint velocities, speed filtering, and replay metadata |
| Patching and search | [`flyby_database.py`](star/flyby_database.py), [`combo.py`](star/combo.py) | Existing flyby feasibility and additive DSM + aT + patch cost |
| Physical replay | `propagate_binned_low_thrust`, [`plotter.py`](star/plotter.py) | Nonlinear two-body propagation of selected controls and reported endpoint misses |

`solve_arc_low_thrust` groups seeds by revolution count into bounded batches; the compiled kernel uses Numba `prange` across seeds.

The builder retains one ballistic option per seed. It uses a relaxed V∞ prefilter to admit recoverable references, then checks the resulting vector magnitudes against the actual limits. Samples whose aT alone exceeds the mission Δv budget are skipped before row expansion. The search adds no low-thrust root finding or propagation to the Lambert solve.

Both LT examples specify `acc_lt_max` and `dacc_lt` and allow finite flyby patches.

## Patched-conic costs and physical replay

The propulsion expenditure is

$$
\Delta v_{\mathrm{LT,eq}}=\int\|\mathbf a(t)\|\,dt=aT.
$$

It is distinct from the endpoint-speed benefit or net vector impulse. Star's existing additive accounting charges each leg and encounter once:

$$
\Delta v_{\mathrm{search}}
=\sum_{\mathrm{legs}}(\Delta v_{\mathrm{DSM}}+aT)
+\sum_{\mathrm{flybys}}\Delta v_{\mathrm{patch}}.
$$

For constant specific impulse, the LT mass ratio is exp(−aT/(g₀ Isp)). Escape and insertion costs remain separate.

Quadratic endpoint velocities leave an $O(a^3)$ position residual for sufficiently small acceleration, apart from reference and quadrature errors. The local expansion imposes no closure tolerance and can worsen estimates outside its useful range, especially on long multi-revolution arcs. Constant full-duration thrust, two endpoint-speed preferences, and a fixed reference restrict the candidate family; absent candidates do not prove infeasibility. The search does not enforce unpowered flybys or physical endpoint closure.

Saved `lt_control` contains the reference velocities, primer weight angle, and predicted endpoint speeds; `lt_n_rev` preserves the reference branch. For positive acceleration, `lt_lev_type = ±2` identifies quadratic legs; `±1` retains the legacy first-order meaning. The existing `propagate_binned_low_thrust` helper replays either control once, without correction. The plotter labels the approximation and shows position and velocity misses. Rebuild Phase 1 to get quadratic candidates; old caches retain their original estimates.

## Related Works

The present method mirrors the impulsive maneuver-placement approach of [Landau (2018)](https://doi.org/10.2514/1.G003172) and is inspired by [Landau (2019)](https://www.jpl.nasa.gov/site/research/media/posters/2019/R19221p.pdf), who treats low thrust as a linearized perturbation to conic arcs. We extend this perturbative description with a quadratic endpoint-velocity response. For each Lambert seed, we grid the allowable leveraging expenditure $\Delta v_{\mathrm{LT,eq}}=aT$ and use the reference primer direction that maximizes the first-order improvement in the signed departure–arrival speed difference $J$. This provides structured sampling of departure/arrival $v_\infty$ pairs within the chosen steering family, preserving alternatives for subsequent flyby matching and tour assembly.

[Iijima, Oguri, and Kuwahara (2026)](https://arxiv.org/html/2609.09515v1) recently augmented Star using a direct transcription of linearized low-thrust dynamics into a convex program. Their convex-pruning step minimizes control expenditure while leaving both endpoint $\mathbf v_\infty$ vectors free within magnitude bounds (Eq. 53). The published algorithm retains at most one corrected leg per eligible nonresonant Lambert seed, with compatibility between neighboring legs evaluated subsequently by the Flyby Module (Algorithm 1). This introduces a potential undersampling risk: the minimum-effort correction may fail a flyby constraint even when a more expensive correction from the same seed could satisfy it. Representing the endpoint-condition family by a single minimum-effort leg can therefore discard useful alternatives before tour assembly.
