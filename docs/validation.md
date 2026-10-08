# Verification and validation report

All values below are reproduced by `python -m experiments.run_validation`. Machine-readable results are in `results/validation/validation_metrics.json`.

## Baseline audit

Untouched default programs were run before modification on 7 October 2026.

| Program | Exact command (output path aside) | Result | Runtime |
|---|---|---|---:|
| scalar legacy model | `python pcl_deg_model.py --outdir <temporary>` | failed while rendering its 26,298 × 120 3-D grid; NumPy requested an additional 382 MiB | 32.451 s |
| hybrid legacy model | `python pcl_implant_hybrid_model.py --outdir <temporary>` | exited 0 but emitted overflow and invalid-value warnings; clipping masked unstable explicit diffusion | 109.723 s |

Before dependencies were installed, both also failed at import because no dependency specification existed. The new project includes `requirements.txt` and `pyproject.toml`.

The hybrid default has \(\Delta x=0.02\), \(D_0=0.01\), and \(\Delta t=0.1\), so the initial explicit diffusion number is

\[
D\Delta t/\Delta x^2=2.5,
\]

already above the one-dimensional forward-Euler limit of 0.5. Its diffusivity can increase further by \(e^4\). The boundary ghost formula also updates full boundary nodes as though they had full cell volumes. The supported solver replaces this path with BDF integration of half-cell conservative balances.

## Level A: mathematical correctness

- Constant random scission reproduces a line in \(1/M_n\) to floating-point tolerance.
- The autocatalytic closed form tends to constant random scission as feedback tends to zero.
- A uniform concentration equal to the exterior produces zero Robin flux and zero diffusion.
- For slab, cylinder, and sphere grids, the discrete concentration inventory changes by exactly the negative outward boundary flux (test tolerance \(10^{-12}\)).
- Vanishing hydrolysis preserves initial \(M_n\), crystallinity, and mass.
- State and parameter validation rejects invalid units/ranges.

## Level B: numerical convergence

For a 180-day transport-sensitive case, final volume-mean \(M_n\) converged as follows:

| Cells | Final \(M_n\) (kDa) | Absolute difference from 160-cell result (kDa) |
|---:|---:|---:|
| 20 | 27.603825 | 0.012934 |
| 40 | 27.596464 | 0.005574 |
| 80 | 27.592753 | 0.001863 |
| 160 | 27.590890 | 0 |

At 80 cells, changing maximum BDF step from 4 to 0.5 days changed final \(M_n\) by only \(1.34\times10^{-7}\) kDa. The representative configuration uses a 2-day maximum step and stricter local error control.

## Level C: physical limiting cases

- Setting hydrolysis effectively to zero changed mean \(M_n\) by 0.0 kDa over 100 days at printed precision.
- Removing autocatalysis gave 35.154 kDa at 100 days.
- With otherwise identical inputs, strong surface clearance retained 34.725 kDa versus 33.415 kDa with no clearance; clearance therefore slowed acid-catalyzed degradation.
- Mass loss remained negligible while \(M_n\) stayed well above the configured soluble threshold, matching the intended bulk-degradation ordering.

## Level D: empirical validation

Dataset: Gil-Castell et al. (2019), electrospun PCL immersed in ultra-pure water at 37 °C. Calibration points were fixed in advance at days 0, 50, 100, 200, 300, and 400. Days 500 and 650 were held out.

| Model | Free kinetic parameters | Training RMSE (kDa) | Held-out RMSE (kDa) | Held-out MAE (kDa) |
|---|---:|---:|---:|---:|
| constant random scission | 1 | 3.657 | 6.862 | 6.795 |
| exponential | 1 | 2.712 | 2.445 | 2.416 |
| autocatalytic random scission | 2 | **2.374** | **0.475** | **0.387** |

The autocatalytic fit parameters were \(k_s=4.7213\times10^{-5}\) kDa\(^{-1}\) d\(^{-1}\) and feedback \(g=114.713\) kDa. This validates only the uniform molecular-weight kinetic limit. Two held-out points are too few for a general performance claim, and the richer model has one additional parameter.

## Level E: geometry/generalization probe

An equal-SA/V comparison used ideal slab, cylinder, and sphere domains, each with 1 mm\(^{-1}\) global SA/V. Their characteristic transport lengths were 1, 2, and 3 mm and their reported Damk\"ohler numbers were 0.992, 3.968, and 8.927. Final global \(M_n\) values after 365 days were 6.800, 6.189, and 6.053 kDa. Thus global SA/V fails to determine transport regime or molecular-weight history, although this single exploratory case does not establish a universal lifetime-effect magnitude. Multi-thickness spatial measurements are required.

## Level F: cross-medium prediction

The water-calibrated parameters were frozen and applied to the separate PBS series from the same publication. This is an environment-shift test, not an independent-material or independent-laboratory validation.

| Model | PBS RMSE, all nonzero times (kDa) | PBS RMSE, days 500/650 (kDa) |
|---|---:|---:|
| constant random scission | 5.187 | 8.010 |
| exponential | 3.077 | 3.673 |
| autocatalytic random scission | **2.616** | **1.058** |

For comparison, fitting the autocatalytic law to PBS through day 400 yielded late RMSE 2.563 kDa. That worse extrapolation is retained as a negative result: environment-specific refitting did not guarantee better future prediction.

## Level G: uncertainty and identifiability

A fixed-initial-condition, multiplicative residual bootstrap in log-\(M_n\) used 2,000 deterministic resamples of the six water calibration observations. The 95% parameter percentile intervals were broad: \(k_s=1.79\times10^{-5}\) to \(1.22\times10^{-4}\) kDa\(^{-1}\) d\(^{-1}\), and feedback reached the optimizer's near-zero lower regime at its lower limit and 546.8 kDa at its upper limit. Both late observations fell inside the 95% predictive intervals:

| Day | Observed \(M_n\) (kDa) | 95% confidence interval (kDa) | 95% predictive interval (kDa) |
|---:|---:|---:|---:|
| 500 | 7.52 | 3.17--11.56 | 3.03--12.23 |
| 650 | 3.34 | 0.79--9.19 | 0.77--9.60 |

The local log-parameter correlation was −0.974 and bootstrap log-parameter correlation was −0.685. Thus the combined trajectory is predictive on this small holdout, but \(k_s\) and feedback are not cleanly separable as physical quantities.

## Level H: distribution and sensitivity checks

The exact ideal random-scission moment calculation passes the unbroken-chain limit \(M_w=M_n\), \(Đ=1\), preserves requested \(M_n\), and maintains \(1\le Đ<2\) for the tested cleavage range. Mapping the water series predicts \(Đ=1.952\) by day 650. This is a falsifiable analytical output, not an empirical validation because the dataset has no full SEC distribution.

The global screen used 128 base Sobol samples and 1,024 coupled two-year cylinder simulations. Stable total-order indices were:

| Parameter | Final \(M_n\) | Weakest strength |
|---|---:|---:|
| intrinsic scission rate | **0.619** | **0.562** |
| autocatalysis | 0.225 | 0.242 |
| acid diffusivity | 0.159 | 0.237 |
| surface mass transfer | 0.137 | 0.096 |
| chemicrystallization gain | 0.011 | 0.009 |
| dissolution rate | 0.001 | <0.001 |

The largest total-order change from 64 to 128 base samples was 0.018 for \(M_n\) and 0.010 for strength. Mass-retention indices were labeled unstable because its largest first-order change was 0.406. Sensitivity rankings depend on the declared parameter ranges and do not establish empirical causality.

## CAD audit

Assuming the STL coordinates are millimetres, `Snap-Fit v5.stl` is a single, watertight, consistently wound mesh with 3,348 vertices and 6,756 faces. Surface area is 7,868.084 mm², volume is 7,733.711 mm³, global SA/V is 1.017375 mm\(^{-1}\), and \(V/A=0.982922\) mm. Bounding-box dimensions are 45.0 × 17.574 × 45.0 mm. Euler number −30 indicates nontrivial topology/through-holes. The STL is used only for audited scalar descriptors; no field is painted onto the mesh.

## Remaining validation gaps

- no matched dataset jointly measures \(M_n\), full MWD, crystallinity, mass, modulus, strength, and local profiles;
- the morphology/mechanics and soluble-mass equations are uncalibrated;
- the empirical validation uses electrospun material, not the supplied dense CAD implant;
- uncertainty is propagated only for the uniform water kinetic fit, not the coupled PDE or mechanics;
- in vivo translation is unvalidated;
- no full 3-D structural mechanics is performed.

