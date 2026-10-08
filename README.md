# PCL implant degradation and functional-retention model

A reproducible Python model for investigating how PCL chemistry, morphology, environment, geometry, and time influence molecular degradation, mass loss, and mechanical-property retention.

The supported model replaces the original unstable explicit solver with a conservative finite-volume reaction--diffusion formulation integrated by a stiff BDF method. It derives molecular-weight evolution from random chain scission, adds retained-acid autocatalysis, tracks evolving crystallinity and delayed soluble mass, and distinguishes stiffness from strength. The supplied STL is audited for geometry but is **not** presented as a mesh-resolved field simulation.

## Main scientific result

On long-duration electrospun-PCL data from Gil-Castell et al. (2019), models were fitted through day 400 and evaluated on pre-declared days 500 and 650:

| Kinetic law | Held-out RMSE |
|---|---:|
| autocatalytic random scission | **0.475 kDa** |
| empirical exponential | 2.445 kDa |
| constant random scission | 6.862 kDa |

This is evidence for the kinetic core on one dataset, not validation of the spatial, mass-loss, or mechanical submodels. See [validation](docs/validation.md) and [novelty assessment](docs/novelty_assessment.md).

## Model summary

The local inverse number-average molecular weight \(I=1/M_n\) evolves as

\[
\frac{\partial I}{\partial t}
=k_s f_T f_{\mathrm{pH}}f_a(X_c)(1+\beta a),
\]

while retained acid/oligomer concentration follows

\[
\frac{\partial a}{\partial t}
=\nabla\cdot(D_a\nabla a)+Y_a\frac{\partial I}{\partial t}-k_{\mathrm{cl}}a.
\]

The exposed boundary uses the outward Robin law

\[
-D_a\nabla a\cdot\mathbf n=h(a-a_\infty).
\]

Crystallinity changes both accessibility and transport. A smooth molecular-weight threshold activates soluble-mass loss. Local modulus combines crystalline/amorphous stiffness, tie-chain retention, and porosity; strength uses a more molecular-weight-sensitive law. The code reports Voigt and Reuss effective-stiffness bounds and weakest-local-strength retention rather than confusing a material average with implant structural stiffness.

Full equations, definitions, units, and applicability limits are in [theory](docs/theory.md).

## Install

Use Python 3.10 or newer:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux/macOS, the environment interpreter is normally `.venv/bin/python`.

## Quick start

Run the documented two-year slab example:

```text
python -m pcl_model.cli simulate --config configs/pcl_hydrolysis.yaml --outdir results/simulation
```

Outputs:

- `summary.json`: parameters, dimensionless groups, solver diagnostics, failure time, and final values;
- `trajectory.csv`: reproducible scalar time series;
- `simulation_summary.png`: molecular weight, mass, mechanics, and local profiles;
- `spatial_profiles.png`: acid, crystallinity, and retained-solid profiles.

Audit the example mesh, declaring its coordinate unit:

```text
python -m pcl_model.cli mesh "Snap-Fit v5.stl" --unit mm
```

Reproduce all validation figures and metrics:

```text
python -m experiments.run_validation --outdir results/validation
python -m pytest -q
```

## Inputs and units

Configuration is YAML, but all computation is Python. Canonical units are:

| Quantity | Unit |
|---|---|
| length | mm |
| time | day |
| molecular weight | kDa |
| diffusivity | mm²/day |
| surface mass transfer | mm/day |
| modulus and strength | MPa |
| crystallinity, solid fraction, normalized acid | dimensionless |

`configs/pcl_hydrolysis.yaml` is an auditable demonstration, not a universal PCL parameter set. In particular, geometry, processing history, pH protocol, buffer renewal, temperature, initial molecular-weight distribution, and crystallinity must match the experiment being represented.

## Geometry

The spatial solver supports symmetric slab, cylinder, and sphere domains. A scalar global SA/V does not uniquely specify transport length:

\[
(SA/V)_{\mathrm{slab}}=1/L,\qquad
(SA/V)_{\mathrm{cylinder}}=2/R,\qquad
(SA/V)_{\mathrm{sphere}}=3/R.
\]

The STL audit reports closure, winding, connected components, area, volume, bounding box, SA/V, and \(V/A\). STL files have no inherent unit, so the unit must be supplied. Pores only count if they are represented as accessible triangulated surfaces; `V/A` is not a minimum wall-thickness measurement.

## Repository layout

```text
pcl_model/                 supported Python package
configs/                   reproducible model inputs
data/                      provenance-documented validation data
experiments/               validation and comparison workflow
tests/                     analytical and numerical regression tests
docs/                      theory, literature, gaps, validation, novelty, log
pcl_deg_model.py           preserved legacy scalar implementation
pcl_implant_hybrid_model.py preserved legacy explicit implementation
Snap-Fit v5.stl            example CAD mesh
```

The two legacy scripts are retained for traceability. Their baseline failures are documented and they are not the supported scientific solver.

## Capabilities and claim boundaries

Supported and verified:

- constant and autocatalytic random-scission kinetics;
- conservative 1-D radial acid transport with Robin clearance;
- slab/cylinder/sphere metric factors;
- stiff integration, convergence tests, positivity checks, and limiting cases;
- explicit units and strict parameter validation;
- reproducible empirical comparison and CAD scalar audit.

Exploratory, not yet empirically validated:

- crystallinity evolution;
- molecular-weight-triggered mass loss;
- morphology/tie-chain mechanical constitutive relations;
- functional-failure time;
- geometry extrapolation beyond ideal radial domains.

Not implemented:

- full molecular-weight distributions;
- enzymatic binding kinetics;
- moving-boundary erosion;
- stress-assisted chemistry;
- full 3-D transport or finite-element structural mechanics;
- Bayesian posterior inference or inverse design.

## Documentation

- [Literature review](docs/literature_review.md)
- [Theory](docs/theory.md)
- [Research gaps and hypotheses](docs/research_gaps.md)
- [Research log, including negative results](docs/research_log.md)
- [Verification and validation](docs/validation.md)
- [Novelty assessment](docs/novelty_assessment.md)
- [BibTeX references](docs/references.bib)

## Citation and reproducibility

This repository is research software, not a clinically validated design tool. Cite the underlying experimental source when using the bundled data (Gil-Castell et al., 2019, DOI `10.3390/nano9050786`) and report the configuration file, commit hash, Python version, dependency versions, and generated `summary.json`.
