# Paper notes — DGP description, cohort marginals, rule layer
 
Accumulated from the R7 append and the E0 band correction. These are drafting notes for
the DGP section and its appendix, plus scope statements that have to appear somewhere
specific rather than being generally true.
 
Everything here is measured or derived. Where a number is quoted, its source and spread
are stated; where something is unmeasured, it says so.
 
---
 
## 1. Binding rates fall in three classes — say so in the appendix
 
The rule table currently reads as seven measured quantities. It isn't. Presenting it
undifferentiated invites a reviewer to ask what R7's 0.198 is evidence *of*, and the
honest answer is "nothing — it is `p_smoker` read back."
 
| class | rules | centre comes from | what it tells a reader |
|---|---|---|---|
| **Derived from config** | R1 (`1−p_smoker`), R5 (`p_male`), R6 (composite), R7 (`p_smoker`) | closed form | nothing empirical; it is a restatement of a declared marginal |
| **Solved to a target** | R3 (0.080), R4 (0.200) | `solve_binding_rates` | boundary pressure is a *stated design parameter*, not an accident |
| **Emergent** | R2 (0.379) | measurement only | a genuine property of the structural chain |
 
R6's closed form, worth stating because it looks emergent and isn't:
 
```
P(Is_Pregnant = 1) = (1 − p_male) · (preg_age_max − age_lo)/(age_hi − age_lo) · p_pregnant
                   = 0.5 · (55−30)/(79−30) · 0.04 = 0.010204
```
 
Verified across five parameter settings, all within 1.4 se of the derivation.
 
R2 was checked rather than assumed emergent — it moves with `p_male` (0.3786 → 0.3854),
`sd_glucose` (→ 0.4254) and `b_bmi_glucose` (→ 0.3246), and is a Gaussian mixture over a
clipped BMI over a uniform age, so no closed form is available.
 
**Why this earns its space.** The R3/R4 row is the one that matters: it says boundary
pressure was *chosen*, so the paper can't be accused of having discovered a result that
was an artifact of how often the rules happened to bind. That argument is invisible if
all seven rates are presented as measurements.
 
---
 
## 2. Cohort marginals — the framing to use, and the citation not to add
 
**Three categories, not two.** An earlier draft of these notes split the DGP's constants
into "sourced structural coefficients" and "declared cohort marginals." `age_range` fits
neither, and it is the most load-bearing constant in the model, so the split has to be
three-way:
 
| category | members | what a citation licenses |
|---|---|---|
| **Sourced coefficients** | `b_bmi_glucose` (0.70), `b_hyperglyc_sbp` (1.76) | the magnitude of a structural edge |
| **Sourced window, stipulated shape** | `age_range` (30, 79) | the *window* only — see below |
| **Stipulated marginals** | `p_male`, `p_smoker`, `p_pregnant`, `preg_age_max`, `sd_*`, `p_diabetic_midband`, `meds_age_scale` | nothing; these are declared |
 
**Use for the sourced coefficients:** drawn from published Mendelian randomization
estimates where available, so the causal structure is not arbitrary. Keep them framed as
*sourced to avoid arbitrary structure*, not as *estimates the simulation inherits
validity from*.
 
**Use for the marginals:** declared stipulations that fix the composition of the
simulated cohort. No claim depends on a marginal's epidemiological accuracy; marginals
set how many rows populate each rule's binding stratum, which affects precision in that
stratum (`se ∝ 1/√m`) and not the substance of any result.
 
**Do not** cite a smoking-prevalence source for `p_smoker = 0.20`. US adult cigarette
smoking was 9.9% in 2024, down from 10.8% in 2023 — the configured value is roughly
double it. A citation converts a stipulation nobody can attack into a claim that is
straightforwardly wrong, and invites an audit of the DGP as a cardiovascular model, which
it is not and does not need to be.
 
### The `age_range` exception — state it explicitly
 
The AHA PREVENT citation (Khan et al., *Circulation* 2024) says the equations are
**validated for ages 30–79**. It says nothing about how age is distributed inside that
window. The DGP draws `Age ~ Uniform(30, 79)`. The window is sourced; the shape is
stipulated.
 
This matters because age is the most load-bearing variable in the model. At n=50000,
seed 0: `corr(Age, z) = 0.4294`, `corr(Age, p_true) = 0.2817`, and the Age term alone
carries 31.8% of `s_core` variance. It propagates onward — `corr(Age, BMI) = 0.4063`,
`corr(Age, Systolic_BP) = 0.5652`, `corr(Age, On_BP_Medication) = 0.4393`. The uniform
assumption shapes roughly a third of the label signal.
 
**And it sits upstream of the calibration**, unlike every other marginal. Varying the
window while holding the seed-0 calibration constants fixed (frozen implementation,
sandbox):
 
```
age_range      R6_derived  R6_obs   R2_obs  R4_obs   prevalence
(30, 79)        0.010204   0.00996  0.3789  0.2017   0.1003
(40, 79)        0.007692   0.00758  0.3697  0.2467   0.1003
(30, 69)        0.012821   0.01246  0.3873  0.1328   0.1011
(30, 90)        0.008333   0.00824  0.3693  0.2867   0.0994
```
 
R4's declared band is 0.200 ± 0.005; a ten-year change puts it ~14 band widths out.
Prevalence and Bayes AUC hold, because those are solved — the orthogonal knobs work as
advertised, but the *rule layer* was never orthogonalized against the age window.
Contrast `p_smoker`, which leaves R2 at 0.378860 exactly across its entire range.
 
**The two-sentence fix.** The cohort spans 30–79 because that is the range over which
PREVENT is validated; within that window age is drawn uniformly by design, for even
coverage of the validated range rather than population realism, and the calibration
constants are solved conditional on the window.
 
That converts a weakness into a stated design property. Uniform-for-coverage is a
*better* property for a designed DGP than approximate realism — but only if said.
Unexplained, it reads as carelessness.
 
**Two risks it closes.** Over-claim by association: put the PREVENT citation next to the
cohort description and a reader infers the age distribution is realistic. And the reverse:
say nothing about uniformity and a reviewer supplies the uncharitable reading.
 
**Sensitivity is unmeasured.** Nothing in the record varies the age distribution. The
sweep above shows the *rule layer* responds; it does not show whether any finding does.
Do not claim robustness to it.
 
**For NeurIPS D&B this becomes a packaging requirement.** If the DGP ships as an artifact,
`age_range` is a knob users will turn, and turning it silently invalidates the declared
binding rates unless `solve_binding_rates` is re-run. That belongs in the documentation,
not in a footnote.
 
The DGP's own header already says these are "declared, not tuned for dramatic results."
That phrase, or something close, should survive into the paper — it is a
pre-registration-flavoured commitment and costs nothing to keep.
 
---
 
## 3. R7 and the one-directional-rule story
 
R1 (`Current_Smoker == 0 → Cigs_Per_Day == 0`) is one-directional: it catches
`smoker=0 & cigs>0` and not `smoker=1 & cigs=0`. That single gap produced three separate
findings under three different names before it was diagnosed — it is why `rules_tol`
exists, why E9's rotation test showed a spurious off-manifold jump, and why 12.6% of
R1–R6-clean copula rows had exactly zero density.
 
This is worth telling as a short methods anecdote rather than burying: it is a concrete
demonstration that a hand-written constraint list can be *silently incomplete in one
direction*, which is precisely the failure mode a reader will suspect of any rule-based
manifold definition. Telling it pre-empts the objection and shows the instrument that
caught it.
 
**R7's consequent is exact `>= 1`, and the reason is geometric.** Under the SCM,
`Cigs ∈ (0,1)` has zero density for every row — non-smokers sit on the atom at 0, smokers
on `[1,40]` after clipping. Exact `>= 1.0` places the rule edge exactly on the support
edge; a tolerance would forgive a strip that is structurally impossible for anyone.
Measured: min `Cigs_Per_Day` among smokers is exactly `1.0`, 27 rows of 50000 sit
precisely on it, none below.
 
Contrast with R1, whose tolerance (`rules_tol`, `tol=0.5`) exists because its consequent
is exact equality to a *point mass* on a continuous column, which no generator
reproduces. Two rules on the same column, one tolerated and one exact, for a stated
reason. A reviewer who notices the asymmetry should find the answer already there.
 
**R1/R7 partition the cohort exactly** — R1 binds on non-smokers, R7 on smokers, disjoint,
summing to n (40100 + 9900 = 50000 at n=50000). Useful as a one-line consistency check in
the appendix.
 
---
 
## 4. Appending a rule cannot perturb the sampler — footnote this
 
`evaluate_rules` consumes no RNG and runs after all sampling; `solve_binding_rates` only
indexes `RULES` by name for the R3/R4 antecedents. Adding R7 therefore left every
pre-existing quantity **bit-identical**: `X`, `p_true`, both calibration constants,
`solved_gamma`, `solved_alpha`, realised prevalence, expected Bayes AUC, all six original
binding rates. Asserted by exact equality, not tolerance.
 
Worth a footnote because the natural reader assumption is that changing the rule set
invalidates prior numbers, and the paper reports results from before and after the
change. One sentence closes it: the rule layer is a read-only view over the sampled
cohort, so extending it appends a column and moves nothing.
 
This also happens to be a small argument for the DGP's architecture — the layered design
(cohort → label → rules) is what makes the invariance provable rather than merely likely.
 
---
 
## 5. Scope statements that must appear at specific points, not generally
 
**R7's adequacy is measured for the Gaussian copula only.** ARF is untested on the
density axis and its failure mode, if it has one, need not be R1's one-directionality.
This belongs next to the on-manifold definition, not in a general limitations paragraph,
because the definition is what inherits the scope.
 
**The corruption harness constructs structurally impossible rows, not off-manifold rows
in general.** All four mutations land in deterministic or structural regions, so every
corrupted row has exactly zero density. A3's finding is "structurally impossible rows
protect the recipient." State it that way every time.
 
**The size of the gap that leaves is unmeasured.** Low-but-nonzero-density off-manifold
rows are not covered by the harness, and the density distribution of rule-*violating*
copula rows has never been measured — the record has no cell for it. Do not characterise
the gap as small, large, or typical; none of those is evidenced. (§2's 12.6% is
rule-*clean* rows at zero density and does not speak to it.)
 
What *can* be said: the corner the harness occupies is generator-reachable rather than
invented, since 12.6% of rule-clean copula rows sat at exactly zero density before R7.
That defends the arm's realism without needing the unmeasured quantity.
 
**This is the paper's most exposed seam.** The headline inversion — 0.236 vs 0.034 —
compares on-manifold contradiction against structurally impossible rows. Narrowing the
claim is correct but doesn't narrow its *role*: a reviewer asking "isn't your
off-manifold arm a strawman?" is asking about the central result. Measuring the
violating-row density distribution is one cheap runner and would bound the gap directly.
Worth doing before submission rather than after a reviewer names it.
 
**The estimator cannot see the Bayes-level residual.** At the E0 config, resolving real
data's own residual of ~0.001 at 3 se would need m ≈ 260000. State this as a bound on
what the real-data baseline can claim, in the same place the baseline is introduced.
 
---
 
## 6. Two tensions to state rather than smooth over
 
**Per-record scoring against a per-assignment ground truth.** Harm is defined over a
whole labelling — the maximally harmful assignment places positives at the lowest `p*` —
while the metrics under test score individual rows. A single row has no `p*`-rank in
isolation. A reviewer will find this; stating it first costs a paragraph and reads as
rigour rather than as a hole.
 
**Task B's ground-truth positive is an arm label, not a row property.** "This row came
from the harmful arm" is a fact about the construction, not about the row. Same shape as
the tension above and should be acknowledged in the same place.
 
---
 
## 7. Running list of numbers to re-verify before they go in prose
 
Not a to-do for anyone else — a note to myself that each of these has a specific
provenance and none should be quoted without it.
 
- On-manifold excess share 76.7% [sd 0.28] → 71.7% [sd 0.50] post-R7. Five DGP × five
  copula seeds.
- Across-DGP-seed sd of the on-manifold share: 0.28 canonical vs 0.86 frozen, variance
  ratio 9.4 against F(4,4)₀.₉₇₅ ≈ 9.60. **Quote the wider band, record as unresolved.**
  Ten canonical seeds would close it; the 0.86 is not derivable from anything in the
  repo.
- ARF on-manifold excess 0.0022, se 0.00171 — an *equivalence* claim, never a difference
  claim. It fails the difference branch by construction and that is correct.
- Every band centre in the rule table: state which class it belongs to (§1) before
  quoting it as a measurement.
- Age's share of the label signal (31.8% of `s_core` variance; `corr(Age, z) = 0.4294`)
  is one DGP seed at n=50000 on the frozen implementation. Re-measure on canonical across
  seeds before it goes in prose.
- The `age_range` sweep in §2 held the seed-0 calibration fixed deliberately — it shows
  what a downstream user turning the knob would get, not what a recalibrated DGP gives.
  Say which, every time it is quoted.
 
