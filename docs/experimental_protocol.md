# Prospective PCL degradation validation protocol

This protocol converts the remaining model gaps into a preregisterable experiment. It is written as an executable design, not as evidence that the physical work has already been performed.

## Primary questions and frozen decisions

1. Does a water-calibrated retained-acid random-scission model predict PBS and thickness-shifted \(M_n\) better than constant-scission and exponential baselines?
2. Do measured crystallinity and retained mass improve held-out modulus and strength prediction beyond \(M_n\) alone?
3. At matched global SA/V, do slab, cylinder, and sphere specimens produce measurably different internal degradation profiles?

Primary molecular endpoint: SEC number-average molecular weight. Primary mechanical endpoint: tensile strength retention. Primary comparison metric: specimen-level held-out RMSE. No model form may be changed after the final two time points are unblinded.

## Specimens and allocation

- Polymer: one traceable medical-grade PCL lot; record supplier, lot, initial SEC distribution, density, and processing history.
- Manufacture dense specimens using one thermal history. Randomize print/mold order across groups.
- Thickness series: slabs with total thickness 0.5, 1.0, and 2.0 mm.
- Equal-SA/V series: slab half-thickness 1 mm, cylinder radius 2 mm, and sphere radius 3 mm, giving nominal SA/V 1 mm\(^{-1}\). Keep every other material/process variable fixed.
- Media: ultrapure water and PBS at 37.0 ± 0.5 °C.
- Destructive time points: 0, 30, 90, 180, 365, 540, and 730 days.
- Biological/processing replicates: five independently manufactured specimens per geometry × medium × time cell. Treat subsamples from one specimen as technical replicates, not independent observations.
- Minimum core design: 3 slab thicknesses × 2 media × 7 times × 5 specimens = 210 specimens. Run the equal-SA/V shape series as a separate 3 × 2 × 7 × 5 block (210 specimens). Add 10% manufacturing reserve before randomization.

Generate the randomization list with a fixed NumPy random seed, stratified by manufacturing batch. Analysts receive coded specimen IDs until the analysis script and exclusion log are frozen.

## Immersion and recorded covariates

Use individually sealed vessels with 20.0 mL medium per gram initial dry polymer. Incubate without light at 37 °C and fixed orbital agitation. Renew medium every seven days; record exact renewal time, bath temperature, pH immediately before renewal, and any lost volume. Retain 1 mL of every pre-renewal medium sample at −20 °C for total organic carbon or oligomer analysis. Include medium-only blanks for every renewal batch.

Before immersion, vacuum-dry specimens to constant mass: two consecutive measurements 24 h apart differing by less than 0.1%. Record dimensions at three positions, dry mass, and density. At harvest, measure wet mass, blot using a fixed 30 s protocol, then vacuum-dry to the same constant-mass criterion.

## Measurements from each harvested specimen

Perform measurements in this order and preserve raw instrument exports:

1. Photograph and record visible cracking or fragmentation before handling.
2. Record wet and final dry mass; calculate water uptake and retained dry mass.
3. Section the specimen into surface and core fractions using a documented depth threshold equal to the outer 20% and central 20% of the transport length. Record recovered mass from each fraction.
4. SEC/GPC: report the full calibrated chromatogram, \(M_n\), \(M_w\), dispersity, column set, solvent, flow, calibration standard, injection concentration, and detection method for bulk, surface, and core fractions where mass permits.
5. DSC: first heating, controlled cooling, and second heating with raw heat-flow traces; calculate crystallinity using the declared PCL heat of fusion and correct for polymer mass fraction.
6. Mechanical coupons from matched specimens: modulus from a preregistered low-strain interval, yield/maximum stress, strain at break, crosshead rate, gauge length, dimensions, and failure location. Exclude grip failures only by the frozen rule.
7. Optional but high-value: micro-CT porosity before immersion and at harvest; spatial pH or Raman/FTIR line scans registered to section depth.

Use separate matched specimens for destructive sectioning and tensile failure if both cannot be obtained without altering the test. Their IDs must share the same randomized manufacturing block.

## Quality control and exclusions

- Calibrate balance, temperature probe, pH meter, DSC, and SEC on each run day; retain calibration records.
- Inject an SEC pooled quality-control sample every ten injections and repeat the batch if QC \(M_n\) drifts more than 5%.
- Analyze samples in randomized order, not chronological order.
- Predeclare exclusions: manufacturing dimension outside ±5%, documented vessel leak, temperature excursion longer than 4 h, insufficient recovered mass, or instrument QC failure. Never exclude because a result disagrees with the model.
- Preserve excluded observations in the dataset with a reason code.

## Frozen train/test split and falsification rules

- Fit kinetic and transport parameters using days 0--365 from the 1.0 mm water slabs only.
- Validate time extrapolation on days 540 and 730 from those slabs.
- Validate environment shift on all PBS slabs with parameters frozen except measured external pH/renewal inputs.
- Validate thickness transfer on the 0.5 and 2.0 mm slabs without changing chemistry parameters.
- Validate shape transfer on the equal-SA/V cylinder and sphere without changing chemistry or transport parameters.

Reject H1 if autocatalytic random scission does not lower late-time RMSE relative to both baselines in at least two of the three transfer tests (time, medium, thickness). Reject H2 if morphology-aware mechanics does not lower held-out strength RMSE by at least 10% over the \(M_n\)-only baseline or if its 95% interval includes no improvement. Reject H3 if surface-to-core \(M_n\) contrasts and global trajectories are indistinguishable across shapes within the preregistered smallest effect of interest: 10% of initial \(M_n\).

## Data and reproducibility package

Store one tidy CSV row per specimen/assay plus unmodified instrument exports. Required metadata include specimen ID, parent batch, geometry, dimensions, medium, bath volume, every renewal/pH record, exact exposure duration, wet/dry mass, section depth, SEC outputs, DSC outputs, mechanical outputs, exclusion state, and reason. Commit the frozen Python analysis script and environment lock before unblinding days 540/730. Publish raw data, processed data, calibration files, and a machine-readable manifest with checksums.
