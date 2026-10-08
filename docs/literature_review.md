# Literature review and evidence audit

Search updated 7 October 2026. Bibliographic identity and DOI metadata were cross-checked through publisher pages, Crossref, PubMed/PMC, OpenAlex, Semantic Scholar, arXiv, and public code/data repositories where available. Google Scholar links exposed by publisher/index pages were used for citation chaining; inaccessible full texts were not treated as read. The verified bibliography is in `references.bib`.

## Hydrolysis and molecular-weight kinetics

PCL is a semicrystalline aliphatic polyester. Water-accessible ester bonds undergo random cleavage, creating hydroxyl and carboxyl end groups. Hydrolytic PCL commonly shows a long interval of falling molecular weight with little dry-mass loss, followed by fragmentation and resorption once sufficiently short oligomers become mobile [Bartnikowski et al., 2019; Lykins et al., 2022]. Enzymatic degradation is a distinct regime: adsorption, enzyme accessibility, and surface or penetrative action can dominate, so a hydrolysis calibration must not silently be transferred to lipase exposure [Shi et al., 2020; Nansak et al., 2026].

For constant random scission while polymer mass remains in the specimen, the chain count rises linearly. Consequently,

\[
\frac{1}{M_n(t)}=\frac{1}{M_{n,0}}+k_s t,
\]

not an exponential in \(M_n\). An exponential can be an effective law when the cleavage rate grows with the number of acid end groups or when it is simply fitted over a limited interval. Lykins et al. explicitly compared constant-scission and autocatalytic descriptions and reported that neither fully described all accelerated PCL blend data. Antheunis et al. derived an end-group-based autocatalytic law and tested it across aliphatic polyesters including PCL. The present implementation therefore treats the exponential as a baseline, not a mechanistic default.

Acid retention is geometry and environment dependent. The same polymer can appear closer to constant random scission in a well-buffered thin sample and more strongly autocatalytic in a thick or poorly cleared domain. Temperature acceleration is represented with an Arrhenius factor, but the default activation energy is an illustrative configuration value rather than a universal PCL constant. The pH effect is exposed as a documented empirical multiplier because a single symmetric pH-rate formula is not supported across acid, neutral, and alkaline mechanisms.

## Semicrystalline morphology and mechanics

Hydrolysis preferentially attacks more accessible amorphous material. Shortened chains can reorganize, producing chemicrystallization. Bosworth and Downes observed increased PCL crystallinity together with increased stiffness/strength over 90 days despite molecular-weight loss. Gil-Castell et al. reported long-term molecular-weight trajectories in water and PBS. Electrospun-mesh studies also show non-monotone modulus and substantial loss of elongation at break. These observations directly contradict a universal \(E/E_0=(M_n/M_{n,0})^\alpha\) law: small-strain modulus can rise while molecular weight falls.

The revised model separates:

- local small-strain modulus, influenced by crystalline fraction, tie-chain retention, and porosity;
- strength, made more sensitive to entanglement/tie-chain loss;
- parallel (Voigt) and series (Reuss) effective stiffness bounds;
- weakest-local-strength retention as a conservative functional metric.

These constitutive relations remain hypotheses until jointly calibrated against molecular weight, DSC crystallinity, density/porosity, and mechanical measurements from the same specimens. The software labels normalized predictions and does not claim structural finite-element analysis.

## Transport, geometry, and erosion

Reaction--diffusion models for degradable polyesters are established prior art [Wang et al., 2008]. Recent PCL work includes a six-species enzymatic reaction--diffusion model with separate crystalline/amorphous states and porosity-dependent enzyme transport [Nansak et al., 2026]. The current contribution is narrower: hydrolytic random scission and soluble-acid transport in slab, cylinder, or sphere symmetry, discretized by conservative finite volumes.

Global SA/V is not a complete transport descriptor. For ideal shapes with the same SA/V, slab half-thickness, cylinder radius, and sphere radius differ by factors 1:2:3. Their diffusion time \(L^2/D\), radial metric, and local fields therefore differ even when a scalar SA/V law predicts identical degradation. The numerical example produced distinct global molecular-weight histories and markedly different Damk\"ohler numbers, but a single parameter regime cannot establish a universal effect size.

PCL hydrolysis is generally bulk molecular degradation before appreciable mass loss. Enzymatic erosion may be surface dominated. A shrinking mesh without an arbitrary-Lagrangian/Eulerian or mapped-coordinate transport term is inconsistent. The legacy moving-boundary option is therefore not used by the supported solver. The revised model instead uses a fixed domain and a clearly identified, threshold-activated local soluble-solid fraction. It is exploratory and requires gravimetric calibration.

## Environment, processing, and in vivo translation

Processing changes molecular-weight distribution, orientation, crystallinity, residual stress, and pore architecture. Ferreira et al. found applied load accelerated molecular-weight loss in electrospun filaments at 45 °C while modulus and strength increased and elongation decreased. This supports mechanics--chemistry interaction as a research question but does not identify a general stress-assisted rate law. The implementation therefore has no speculative stress coupling.

In vivo PCL degradation adds enzymes, cells, phagocytosis, fluid renewal, mechanical load, and spatially varying tissue contact. Accelerated acidic or elevated-temperature tests do not establish a unique time-shift factor to physiological service. Model parameters must be calibrated for the intended environment, and extrapolation should carry uncertainty.

## Evidence quality and contradictions

| Claim | Evidence | Limit |
|---|---|---|
| Constant random cleavage gives linear \(1/M_n\) | polymer population balance; Lykins et al. discussion | assumes retained mass and length-independent bond susceptibility |
| Acid end groups can accelerate hydrolysis | Antheunis et al.; broader polyester reaction--diffusion literature | buffer exchange and diffusion may suppress local feedback |
| PCL often loses molecular weight before mass | multiple hydrolytic studies/reviews | enzymes and porous specimens can lose mass early |
| Crystallinity may rise during early hydrolysis | Bosworth & Downes; Gil-Castell et al. | processing and enzyme type can reverse the trend |
| Modulus is not determined by \(M_n\) alone | simultaneous morphology/mechanics studies | quantitative constitutive law remains dataset-specific |
| Geometry cannot be reduced universally to SA/V | transport scaling and scaffold studies | global averages can still be similar in reaction-limited regimes |
| In vitro rates transfer directly in vivo | contradicted by reviews and mixed protocols | no general mapping is currently defensible |

## Dataset selected for validation

Gil-Castell et al. provide tabulated \(M_n\) for electrospun PCL in water and PBS through 650 days. The values are transcribed in `data/gil_castell_2019_pcl_mn.csv`. The water series was chosen before fitting: days 0--400 are calibration data and days 500 and 650 are held out. This tests molecular-weight kinetics only; it does not validate local profiles, mass loss, or mechanics.

