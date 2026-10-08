# Novelty assessment

## What is not novel

Random chain scission, acid autocatalysis, reaction--diffusion degradation, chemicrystallization, Voigt/Reuss bounds, and Damk\"ohler/Biot scaling all have substantial prior art. A one-dimensional PCL reaction--diffusion model is not by itself novel. Recent open-source work by Nansak et al. (2026) is more detailed for lipase-mediated enzymatic degradation and explicitly tracks crystalline/amorphous enzyme complexes.

## Potential contribution

The repository now provides a compact hydrolytic-PCL workflow that combines four elements which were absent from the original code:

1. a random-scission inverse-molecular-weight state with an exact autocatalytic uniform limit;
2. a conservative finite-volume embedding for slab, cylinder, and sphere geometries with a verified outward Robin balance;
3. a morphology-aware separation of modulus, strength, and structural homogenization bounds;
4. a pre-declared held-out comparison against tabulated long-duration PCL data, an environment-shift test, bootstrap uncertainty/identifiability, and convergence/limiting-case verification;
5. exact ideal random-scission distribution moments and a reproducible range-based global sensitivity screen.

The integration and evidence workflow may be useful and potentially distinctive as research software, but the underlying physics are combinations of established ideas. No claim of new polymer chemistry is justified.

## Evidence

Fitting days 0--400 of the Gil-Castell water dataset and predicting days 500 and 650 gives held-out RMSE:

- autocatalytic random scission: **0.475 kDa**;
- empirical exponential: **2.445 kDa**;
- constant random scission: **6.862 kDa**.

This is a predictive improvement on one small held-out set. It does not prove generality, and the two-parameter autocatalytic law is more flexible than the one-parameter baselines. The use of future points and explicit holdout partially mitigates, but does not eliminate, that concern.

With parameters frozen at the water calibration, late PBS RMSE was 1.058 kDa for autocatalytic random scission, 3.673 kDa for the exponential, and 8.010 kDa for constant scission. This strengthens the selected-dataset result across media but is not independent-laboratory validation. A 2,000-sample multiplicative residual bootstrap placed both late water observations inside 95% predictive intervals. It also exposed weak parameter separation: the two log parameters were strongly anticorrelated, so the rate and feedback should not be interpreted independently from this dataset.

The 1,024-run global screen identifies intrinsic scission rate as the dominant input for two-year \(M_n\) and weakest-link strength over the declared ranges. Acid diffusivity, autocatalysis, and surface mass transfer also contribute. Mass-retention indices failed the half-sample stability criterion and are not used for ranking. These findings prioritize measurements; they do not validate the uncalibrated mass/mechanics equations.

## Claim boundary

The defensible claim is: *for the selected long-duration hydrolytic PCL dataset, a mechanistically constrained autocatalytic random-scission law predicts two pre-declared late water observations more accurately than constant random scission or an exponential fitted to the same training observations, and retains the lowest late-time error when those water parameters are transferred without refitting to the study's PBS arm.*

The spatial, crystallinity, mass-loss, and mechanical portions are numerically verified hypotheses, not empirically validated predictors. A broader novelty claim should wait for prospective multi-thickness experiments and comparison against current PCL-specific models.

