# Research gaps and hypothesis matrix

Scores are qualitative (low/medium/high) and reflect the evidence reviewed through 7 October 2026.

| Hypothesis | Prior art / closest formulation | Proposed testable contribution | Required data | Identifiability risk | Difficulty | Expected value | Novelty confidence | Falsification criterion |
|---|---|---|---|---|---|---|---|---|
| H1: retained-acid random scission out-predicts one-rate laws | Antheunis 2010; Lykins 2022 | conservative spatial embedding of the mechanistic uniform limit and held-out comparison | time-resolved \(M_n\), buffer renewal, thickness | medium | low | high | low as chemistry; medium as integrated implementation | no held-out gain over exponential across datasets |
| H2: evolving crystallinity explains stiffness retention despite \(M_n\) loss | Bosworth 2010; Nansak 2026 (enzymatic) | joint hydrolytic chain-scission/morphology/mechanics model | matched GPC, DSC, tensile tests | high | medium | high | medium | morphology-aware model fails held-out mechanics or fitted gain is non-identifiable |
| H3: equal SA/V geometries can develop different local degradation | Wang 2008 and transport literature | exact slab/cylinder/sphere finite-volume comparison at equal SA/V | spatial pH/\(M_n\) profiles across shapes | medium | medium | high | profiles converge over the full relevant Damk\"ohler range |
| H4: distribution moments predict strength better than \(M_n\) | kinetic scission/population-balance literature | exact ideal random-scission \(M_w\)/dispersity moments, then calibrated population balance | full SEC distributions and tensile failure | high | high | high | no held-out gain over \(M_n\)+crystallinity |
| H5: dimensionless groups choose model fidelity | degradation maps in Wang 2008 | use Damk\"ohler/Biot criteria to select uniform vs spatial solver | multi-thickness experiments | medium | medium | medium | selected low-fidelity solver exceeds declared error tolerance |
| H6: load changes hydrolysis through measurable morphology/transport, not an ad hoc stress exponent | Ferreira 2025 | discriminate stress-assisted chemistry from strain-induced transport/crystallinity | loaded/unloaded GPC, DSC, water uptake | high | high | high | a common chemistry model with measured morphology explains both groups |
| H7: threshold-aware inverse design can target functional lifetime | broad inverse-design prior art | constrained geometry/material search with applicability checks | calibrated posterior and structural loading data | high | medium | high | posterior predictive intervals miss prospective experiments |

## Selection

H1 + H2 + H3 were selected as a coherent, minimal advance. H1 has direct independent data and is empirically tested. H2 is physically motivated but only numerically demonstrated. H3 is mathematically verified; the current example produced modest differences in global \(M_n\), so the strong claim that equal SA/V *necessarily* gives practically important lifetime differences is rejected. Local-profile measurements across thicknesses are needed.

The analytical portion of H4 is now implemented and verified: ideal random scission predicts dispersity rising from 1 toward 2. It is deliberately not called empirically validated because the selected dataset lacks full SEC distributions and is not initially monodisperse. H5 now has a global sensitivity screen but still lacks the multi-thickness error study needed to turn dimensionless groups into model-selection thresholds. H6--H7 remain future work. The exact prospective measurements and decision rules needed for H2--H5 are frozen in `experimental_protocol.md`.

