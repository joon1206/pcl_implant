# Governing equations and numerical method

## Scope and units

The supported model resolves a symmetric slab half-thickness, solid cylinder radius, or solid sphere radius. Canonical units are millimetres (mm), days (d), kilodaltons (kDa), and megapascals (MPa). Acid concentration is normalized by a user-defined reference concentration. Every parameter name carries its unit where practical.

The state in each finite-volume cell is

- \(I=1/M_n\), inverse number-average molecular weight (kDa\(^{-1}\));
- \(a\), normalized retained-acid/soluble-oligomer concentration (dimensionless);
- \(X_c\), crystalline volume fraction (dimensionless);
- \(s\), retained solid fraction (dimensionless).

## Random scission derivation

Let a specimen contain polymer mass \(m\), initially distributed among \(N_0=m/M_{n,0}\) chains. Every main-chain scission creates one additional chain while mass remains in the specimen. If \(q\) is the scission rate in chains per mass per time,

\[
N(t)=N_0+qmt,
\qquad
M_n(t)=\frac{m}{N(t)}.
\]

Therefore

\[
\frac{1}{M_n(t)}=\frac{1}{M_{n,0}}+q t.
\]

This constant random-scission result is hyperbolic in \(M_n\), not exponential. It applies before substantial soluble-mass escape and assumes equal bond susceptibility.

## Coupled hydrolysis and acid transport

The local scission equation is

\[
\frac{\partial I}{\partial t}
=k_s f_T f_{\mathrm{pH}}
\frac{1-X_c}{1-X_{c,0}}
(1+\beta a),
\]

or compactly

\[
\frac{\partial I}{\partial t}=k_s f_T f_{\mathrm{pH}} f_a(X_c)(1+\beta a).
\]

Here \(k_s\) has units kDa\(^{-1}\) d\(^{-1}\), \(\beta\) is inverse normalized acid, and \(f_a\) is amorphous accessibility. The Arrhenius factor is

\[
f_T=\exp\left[-\frac{E_a}{R}\left(\frac{1}{T}-\frac{1}{T_{\mathrm{ref}}}\right)\right].
\]

The acid balance in radial dimension \(q=0,1,2\) is

\[
\frac{\partial a}{\partial t}
=\frac{1}{r^q}\frac{\partial}{\partial r}
\left(r^q D_a\frac{\partial a}{\partial r}\right)
+Y_a\frac{\partial I}{\partial t}-k_{\mathrm{cl}}a.
\]

Diffusivity changes with crystalline and void fractions:

\[
D_a=D_{a,0}\exp[-b_X(X_c-X_{c,0})]\,[1+b_p(1-s)].
\]

Symmetry imposes zero flux at \(r=0\). At the exposed surface \(r=L\), outward flux is

\[
-D_a\nabla a\cdot\mathbf n=h(a-a_\infty).
\]

The sign is implemented as an outward finite-volume face flux; the discrete total inventory rate equals sources minus bulk clearance minus the surface flux.

## Uniform autocatalytic limit

With uniform material, no clearance, constant crystallinity, and \(a=Y_a(I-I_0)\),

\[
\frac{dI}{dt}=k_s[1+g(I-I_0)],\qquad g=\beta Y_a.
\]

The exact solution used for empirical calibration is

\[
I(t)=I_0+\frac{\exp(k_sgt)-1}{g},
\qquad M_n(t)=I(t)^{-1}.
\]

This connects the independently testable kinetic law to the spatial PDE.

## Ideal molecular-weight distribution moments

For an initially monodisperse chain of \(n\) repeat units, let every one of its \(n-1\) bonds break independently with probability \(p\). The expected fragment count is \(1+(n-1)p\), giving

\[
M_n=\frac{M_{n,0}}{1+(n-1)p}.
\]

Two repeat units separated by \(d\) bonds remain in one fragment with probability \((1-p)^d\). The expected sum of squared fragment lengths is therefore

\[
\mathbb E\!\left[\sum_j \ell_j^2\right]
=n+2\sum_{d=1}^{n-1}(n-d)(1-p)^d,
\]

and

\[
M_w=\frac{M_{n,0}}{n^2}\mathbb E\!\left[\sum_j \ell_j^2\right],
\qquad Đ=\frac{M_w}{M_n}.
\]

This is an exact distribution-level consequence of the ideal cleavage assumptions. It is not a full population balance: real initial dispersity, preferential amorphous cleavage, soluble-fragment escape, branching, and SEC measurement effects are excluded.

## Morphology and delayed mass loss

Chemicrystallization is represented as relaxation toward a degradation-dependent target:

\[
X_c^*=X_{c,0}+\Delta X_{c,\max}\left(1-\frac{M_n}{M_{n,0}}\right),
\]

\[
\frac{\partial X_c}{\partial t}
=k_X(X_c^*-X_c)-k_{X,h}(M_{n,0}\,\partial_t I)X_c.
\]

The second term permits crystalline loss but is disabled by default. Soluble mass is activated smoothly near \(M_{n,\mathrm{sol}}\):

\[
\frac{\partial s}{\partial t}
=-k_{\mathrm{diss}}s
\left[1+\exp\left(\frac{M_n-M_{n,\mathrm{sol}}}{w_M}\right)\right]^{-1}.
\]

This is a regularized hypothesis, not a molecular population balance. It avoids claiming mass loss as soon as a single average chain crosses a hard threshold.

## Mechanical relations

Let \(r_E=E_c/E_a\). A crystalline/amorphous mixture factor is

\[
g_X=\frac{(1-X_c)+r_E X_c}{(1-X_{c,0})+r_E X_{c,0}}.
\]

Tie-chain retention above critical entanglement molecular weight \(M_e\) is

\[
g_M=\max\left(\frac{M_n-M_e}{M_{n,0}-M_e},0\right).
\]

Local normalized modulus and strength are

\[
\frac{E}{E_0}=g_X g_M^{p_E}s^{n_E},
\qquad
\frac{\sigma}{\sigma_0}=g_M^{p_\sigma}s^{n_\sigma}.
\]

The volume-weighted arithmetic mean is reported as a parallel/Voigt stiffness, the harmonic mean as a series/Reuss stiffness, and the minimum local strength as a conservative weakest-link metric. These are reduced-order structural indicators, not a replacement for a load- and boundary-condition-specific mechanical solve.

## Geometry and dimensionless groups

For slab, cylinder, and sphere respectively,

\[
\frac{SA}{V}=\frac{1}{L},\quad \frac{2}{R},\quad \frac{3}{R}.
\]

The model reports

\[
\mathrm{Da}=k_sM_{n,0}\frac{L^2}{D_{a,0}},
\qquad
\mathrm{Bi}=\frac{hL}{D_{a,0}}.
\]

\(\mathrm{Da}\ll1\) suggests nearly uniform reaction-limited behavior; \(\mathrm{Da}\gg1\) warns that transport can generate spatial gradients. These are screening criteria, not rigorous error bounds.

## Global sensitivity estimators

The sensitivity workflow draws scrambled Sobol points for two base matrices \(A\) and \(B\), evaluates column-replacement matrices \(A_B^{(i)}\), and reports the Saltelli first-order and Jansen total-order estimators. Parameter ranges and linear/logarithmic measures are written into the result file. A 64-versus-128 base-sample comparison is retained as a convergence diagnostic. These are range-dependent screening indices, not posterior probabilities or measured causal effects.

## Discretization

Cell-centered finite volumes integrate the radial metric exactly over each control volume. Harmonic diffusivity is used at internal faces. The semi-discrete stiff system is integrated by SciPy's variable-order BDF method with specified relative/absolute tolerances and maximum step. No post-step clipping is used to conceal instability; material bounds are checked and only solver-scale roundoff is projected in reported output.

