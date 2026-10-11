# Research log

## Iteration 0 — untouched baseline

**Hypothesis:** existing programs provide a usable numerical baseline.  
**Test:** run both default CLIs.  
**Result:** rejected. The scalar program exhausted memory in the 3-D plot; the hybrid program overflowed under an explicit diffusion number of 2.5 and nevertheless exited successfully because negative/invalid states were clipped. Runtime was 32.451 s to failure and 109.723 s respectively.  
**Decision:** preserve the legacy files for traceability, but build a tested package around a stiff conservative solver.

## Iteration 1 — kinetic law audit

**Hypothesis:** the original exponential is the random-scission solution.  
**Result:** rejected analytically. Constant random scission makes chain count, hence \(1/M_n\), linear in time. Exponential \(M_n\) requires additional feedback or is empirical.  
**Implementation:** inverse-\(M_n\) state plus exact retained-acid autocatalytic uniform limit.  
**Validation:** fit days 0--400 of independent long-term PCL data; hold out days 500 and 650.  
**Result:** retained. Held-out RMSE 0.475 kDa versus 2.445 kDa exponential and 6.862 kDa constant-scission.

## Iteration 2 — conservative spatial model

**Hypothesis:** a radial finite-volume form can resolve acid retention without boundary-sign ambiguity.  
**Implementation:** cell-centered slab/cylinder/sphere volumes, harmonic face diffusion, explicit outward Robin flux, BDF time integration.  
**Verification:** exact discrete inventory balance, zero-field limit, positivity checks, 20--160 cell refinement, and maximum-step refinement.  
**Result:** retained. Spatial error decreased monotonically; 80-to-160-cell change was 0.001865 kDa in the selected test.

## Iteration 3 — morphology-aware mechanics

**Hypothesis:** crystallinity can preserve or increase small-strain stiffness while molecular-weight-dependent strength declines.  
**Implementation:** chemicrystallization target, crystalline/amorphous mixture factor, tie-chain retention, porosity factor, Voigt/Reuss stiffness bounds, weakest-link strength.  
**Result:** numerically plausible and consistent with qualitative literature trends, but not empirically validated. Retained as an explicitly exploratory capability. A universal \(E\propto M_n^\alpha\) claim was rejected.

## Iteration 4 — geometry beyond SA/V

**Hypothesis:** ideal shapes having equal global SA/V show materially different mean degradation.  
**Test:** slab, cylinder, sphere at 1 mm\(^{-1}\), with lengths 1:2:3.  
**Result:** supported numerically but not empirically. Dimensionless transport regimes and local paths differ; final global \(M_n\) spanned 6.053--6.800 kDa in the selected experiment.  
**Decision:** retain the geometry-aware solver and dimensionless diagnostic; state the modest result and avoid claiming a universal large effect.

## Iteration 5 — supplied STL

**Hypothesis:** the example mesh supports reliable scalar geometry.  
**Result:** retained with an explicit millimetre assumption. The mesh is watertight and consistently wound. Its negative Euler number warns that topology and internal accessible surfaces cannot be represented by a single half-thickness.  
**Decision:** report scalar descriptors only; do not display invented mesh-resolved degradation fields.

## Iteration 6 — medium-shift and uncertainty audit

**Hypothesis:** the water-calibrated autocatalytic law retains its advantage under a PBS environment shift.
**Test:** freeze water day-0--400 parameters and score every nonzero PBS observation, with days 500/650 reported separately; also fit PBS through day 400 as a secondary check.
**Result:** supported within this study. Without refitting, late PBS RMSE was 1.058 kDa, versus 3.673 kDa exponential and 8.010 kDa constant scission. PBS-specific refitting did not improve autocatalytic late prediction (2.563 kDa), illustrating that a better training fit is not guaranteed to extrapolate better.
**Uncertainty:** 2,000 multiplicative residual bootstrap fits kept both late water observations inside 95% predictive intervals, but the kinetic parameters were strongly anticorrelated.
**Decision:** retain the predictive law while explicitly treating its individual parameters as weakly separated.

## Iteration 7 — distribution-level random-scission consequence

**Hypothesis:** independent bond cleavage from initially monodisperse chains produces an auditable prediction for \(M_w\) and dispersity from \(M_n\).
**Test:** derive the exact connected-repeat-unit pair sum and verify unbroken-chain and monotonic-cleavage limits.
**Result:** retained analytically. Mapping the selected water series predicts dispersity from 1.000 initially to 1.952 at day 650.
**Decision:** expose the calculation as a falsifiable hypothesis, not validation; the dataset has no full distribution measurements and real PCL is initially polydisperse.

## Iteration 8 — global sensitivity

**Hypothesis:** transport and reaction parameters can be ranked reproducibly over declared ranges.
**Test:** 1,024 coupled simulations from a scrambled Sobol/Saltelli design, with 64-versus-128 base-sample changes retained.
**Result:** intrinsic scission rate dominated two-year \(M_n\) (total-order index 0.619) and weakest strength (0.562); autocatalysis, diffusivity, and mass transfer contributed. Mass-retention rankings were unstable because the first-order estimate for diffusivity shifted 0.406 between sample sizes.
**Decision:** accept \(M_n\)/strength as screening-grade and reject a mass-retention ranking at this sample size.

