# Novelty assessment

## What is not novel

Random chain scission, acid autocatalysis, reaction--diffusion degradation, chemicrystallization, Voigt/Reuss bounds, and Damk\"ohler/Biot scaling all have substantial prior art. A one-dimensional PCL reaction--diffusion model is not by itself novel. Recent open-source work by Nansak et al. (2026) is more detailed for lipase-mediated enzymatic degradation and explicitly tracks crystalline/amorphous enzyme complexes.

## Potential contribution

The repository now provides a compact hydrolytic-PCL workflow that combines four elements which were absent from the original code:

1. a random-scission inverse-molecular-weight state with an exact autocatalytic uniform limit;
2. a conservative finite-volume embedding for slab, cylinder, and sphere geometries with a verified outward Robin balance;
3. a morphology-aware separation of modulus, strength, and structural homogenization bounds;
4. a pre-declared held-out comparison against tabulated long-duration PCL data, plus convergence and limiting-case verification.

The integration and evidence workflow may be useful and potentially distinctive as research software, but the underlying physics are combinations of established ideas. No claim of new polymer chemistry is justified.

## Evidence

Fitting days 0--400 of the Gil-Castell water dataset and predicting days 500 and 650 gives held-out RMSE:

- autocatalytic random scission: **0.475 kDa**;
- empirical exponential: **2.445 kDa**;
- constant random scission: **6.862 kDa**.

This is a predictive improvement on one small held-out set. It does not prove generality, and the two-parameter autocatalytic law is more flexible than the one-parameter baselines. The use of future points and explicit holdout partially mitigates, but does not eliminate, that concern.

## Claim boundary

The defensible claim is: *for the selected long-duration hydrolytic PCL dataset, a mechanistically constrained autocatalytic random-scission law predicts two pre-declared late observations more accurately than constant random scission or an exponential fitted to the same training observations.*

The spatial, crystallinity, mass-loss, and mechanical portions are numerically verified hypotheses, not empirically validated predictors. A broader novelty claim should wait for prospective multi-thickness experiments and comparison against current PCL-specific models.

