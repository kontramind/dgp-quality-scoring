# Decisions

Choices that outlive the conversation that produced them, and that are not derivable
from the code, the manifests or the git history. One entry per decision, newest last.

An entry records what was decided, the arithmetic or evidence behind it, and how to
apply it. Where a figure comes from a committed artifact the entry names the artifact;
where it comes from outside this repo the entry says so, because the repo's convention
(see README, "Numbers live in artifacts, not in prose") is that a quoted number is
either derivable from `results/` or explicitly marked as not being from here.

---

## 2026-08-19 — Quote the wider band on the on-manifold share spread

**Decision.** When quoting the across-DGP-seed spread of the R1-R6 on-manifold share of
excess contradiction, use the wider figure, 0.86. Do not quote 0.28 as the settled
value, and do not describe the difference between them as noise.

**The numbers.** `occ_cop_r7.py` measures `across_dgp_seed_sd` on that share as **0.28**
over five DGP seeds — `manifests/occupancy_copula_r7.json`, key
`shares."R1-R6".on_manifold_share_pct_sd`, run at commit `eb5c040`. The frozen
implementation's corresponding figure is **0.86**. That 0.86 is *not* measured in this
repo: it was reported alongside the targets in the brief that commissioned the five-seed
run, and no artifact here derives it.

**The arithmetic.** The variance ratio is 0.86² / 0.28² = 9.4. The 97.5th percentile of
F(4, 4) is about 9.60. The ratio therefore sits just inside the critical value rather
than comfortably within sampling variability. Two five-sample standard deviations are
weak evidence in either direction: this neither establishes a real difference in spread
nor rules one out.

**Why the wider one.** Calling the difference noise on the strength of one five-sample
comparison is the same move as quoting a single draw as converged, which this project
has already paid for twice — the reference `gamma = 2.64`, which was the n=5000 seed-0
draw quoted as if converged (`manifests/s0_e0.json` records the resolution), and the
77.5% headline on-manifold share, which was seed 0 and the maximum of five. Both were
single draws presented as characteristic. Preferring the wider band costs nothing here:
the R7 gap is 4.6 points, so every conclusion in this line of work survives either
figure.

**How to apply.** Treat the spread as unresolved. Use 0.86 in any band or error bar, and
write "unresolved" rather than "noise" or "not a finding".

**Why 0.86 cannot simply be made derivable.** It came from a sandbox script that was
never committed and that runs against `reference/dgp_frozen.py`. Committing that script
would make the frozen reference a runtime dependency of something outside
`tests/test_differential.py` and `scripts/export_differential.py`, which is a freeze-line
breach — and this is a nuisance parameter, not worth one.

**Exit.** Canonical's own 0.28 is the operative number; 0.86 is cited only because both
are five-sample sds and neither settles the other. Ten canonical DGP seeds would make
the frozen citation unnecessary altogether, at which point this entry is closed rather
than merely carried. Not worth commissioning a run for on its own: the coupling sweep
will supply more seeds as a by-product, and this entry should be revisited then.

---

## 2026-08-20 — R7 is part of `dgp.RULES`, and adding it re-baselined nothing

**Decision.** `R7_smoker_cigs_min` (`Current_Smoker == 1 -> Cigs_Per_Day >= 1`) is an
**append-only** addition to `dgp.RULES`. Every pre-existing E0 golden value is
bit-invariant under it: the E0 table gains one row (R7 binding, 0.198 measured, band
+/-0.01 to match the other emergent rates) and **no existing cell was edited**. The E0
record was deliberately **not** re-run.

**Why it belongs in `dgp.RULES` and not in a runner.** R1 is one-directional: it catches
`smoker == 0 & cigs > 0` and never `smoker == 1 & cigs == 0`. That gap is the single
cause of three findings in this project, most recently that 12.6% of R1-R6-clean copula
rows carried *exactly zero* density (`results/r7_decomp.csv`). R7 closes the other
direction and removes every such row, so it is load-bearing for the definition of
on-manifold rather than a filter over it.

**Why the consequent carries no tolerance.** Under the SCM,
`cigs = where(smoker == 1, clip(N(15, 5), 1, 40), 0.0)`, so the clip lower bound returns
literal `1.0` and a smoker's `Cigs_Per_Day` is exactly `>= 1.0` with no float hazard
(canonical, n=50000, seed 0: minimum among smokers exactly 1.0, 22 rows at the edge,
none below -- `manifests/s0_e0_r7_invariance.json`, `r7_support_edge`). `Cigs in (0, 1)`
has zero density for *every* row, so the rule edge sits exactly on the support edge and a
tolerance would forgive a strip that is structurally impossible for anyone. `rules_tol.py`
therefore stays R1-only; it picks R7 up automatically through its `else` branch with the
exact consequent.

**Why nothing was re-baselined.** `evaluate_rules` consumes no RNG and runs after all
sampling; `solve_binding_rates` only indexes `RULES` by name for the R3/R4 antecedents.
A rule *addition* cannot perturb the sampler. `scripts/export_e0_r7_invariance.py` checks
this rather than asserting it, by exact `==` / `DataFrame.equals` / `np.array_equal` and
never `np.isclose`: `glucose_base`, `sbp_base`, `solved_gamma`, `solved_alpha`,
`realised_prevalence`, `expected_bayes_auc`, all six R1-R6 binding rates, `X` elementwise
and `p_true` elementwise are identical with and without R7. It also asserts the
structural identity that is stronger than any band on either rate: R1 binds on
non-smokers and R7 on smokers, so every row binds exactly one of them
(40100 + 9900 = 50000, disjoint).

**Differential contract, narrowed.** `tests/test_differential.py` now compares canonical
against `reference/dgp_frozen.py` over `set(dgp.RULES) & set(ref.RULES)` -- the six
shared rules, on which agreement stays exact -- and asserts that the set difference is
exactly `{"R7_smoker_cigs_min"}` one way and empty the other. R7 is the *sole permitted*
divergence, and the assertion still fails on an eighth divergence or on R7 silently
disappearing (both verified by mutation). A carve-out that cannot fail is worse than no
differential test.

**`r7_decomp.py` and `occ_cop_r7.py` are reproducible only at their committed SHAs**
(`133a889` and `571c444` respectively). Both contrast an R1-R6 row against an R1-R7 row,
and both obtain the R1-R6 row from `rules_tol.check_rules_tol`, which reads `dgp.RULES`
with a late lookup. With R7 in `dgp.RULES` that "R1-R6" row silently becomes an R1-R7
row, the two rows collapse, and the headline decomposition (12.6% -> 0.0%; 76.7% ->
71.7%) degenerates -- while the scripts keep running and keep emitting plausible numbers.
Their committed results stand. **Do not re-run either script in place at HEAD.**

**How to apply.** Treat `dgp.RULES` as seven rules. Quote R7's binding rate as ~0.198
(= `p_smoker`). When a script needs a genuine R1-R6 row alongside an R1-R7 one, it must
name the six explicitly rather than reading `dgp.RULES` and calling the result R1-R6.

---

## 2026-08-20 — Binding-rate band centres come from the derivation, never from a draw

**Decision.** Every binding rate in the E0 golden table falls into exactly one of three
classes, and the class fixes where its band centre comes from:

| class | rules | centre |
|---|---|---|
| **derived from config** | R1 `1 - p_smoker`, R5 `p_male`, R7 `p_smoker`, R6 the pregnancy composite | the closed form |
| **solved to a declared target** | R3, R4 | the target passed to `solve_binding_rates` |
| **emergent** | R2 | a measured value; there is no alternative |

R6's closed form is
`(1 - p_male) * (preg_age_max - age_lo)/(age_hi - age_lo) * p_pregnant`, because
`Is_Pregnant` is drawn only for women at or below `preg_age_max` over a uniform age
range. On the defaults it is `0.5 * 25/49 * 0.04 = 0.0102040816`.

**What was wrong.** Four centres in `tests/test_dgp.py::EMERGENT_BANDS` had been set from
single observed draws: R1 0.803, R5 0.497, R6 0.0100, R7 0.197. Recentred on their
derivations: **0.800, 0.500, 0.010204, 0.200**. Widths are unchanged, and every one of
the seven rates stays inside its band across seeds 0-4 (worst case 0.50 of a width, R3).

**Why it matters beyond tidiness.** Canonical measures R5 at 0.502520. Against the old
centre that consumed 55% of the +/-0.01 before any sampling error was spent; against
0.500 it consumes 25%. Centring a tolerance on one realisation of a quantity that has a
closed form is the same single-draw failure mode this project has already paid for twice
(the reference `gamma = 2.64`, and the 77.5% on-manifold headline — see the entries
above and `manifests/s0_e0.json`).

**R2's status was checked, not assumed.** It moves with several parameters at once and is
a Gaussian mixture over a clipped BMI over a uniform age: `p_male=0.3 -> 0.385`,
`sd_glucose=25 -> 0.425`, `b_bmi_glucose=1.4 -> 0.325`. A measured centre is the only one
available for it.

**How to apply.** When adding a rule, classify it first. If its antecedent reads a config
parameter back, the centre is the derivation — do not run the DGP and write down what came
out. `tests/test_dgp.py::test_derived_band_centres_match_their_closed_forms` enforces this
for the four derived rules so the rule survives as more than a comment. (The table was
called `EMERGENT_BANDS` when this entry was written, which was a misnomer for four of its
seven entries; it and `SOLVED_BANDS` were later merged into `BINDING_BANDS` — see the
classed-bands-table entry below.)

---

## 2026-08-20 — `s0_e0_golden` regenerated as a pure column addition

**Decision.** `results/s0_e0_golden.csv` and `manifests/s0_e0_golden.json` were
regenerated after R7 joined `dgp.RULES`, by re-running
`scripts/export_s0.py::export_e0_golden` alone. This is **not** a re-baseline.

**The licence.** The R7 addition is bit-invariant for every pre-existing E0 quantity —
proved by construction and by execution in `manifests/s0_e0_r7_invariance.json` (commit
`6282f5a`). Regeneration therefore cannot alter a cell; it can only append the two
columns `export_s0.py` derives from `list(dgp.RULES)`.

**The verification.** The regenerated CSV was diffed against the committed one before
being committed, comparing the raw CSV text of every pre-existing cell rather than
parsed floats. All 24 pre-existing columns bit-identical; row count 5 -> 5; exactly two
columns added, `R7_smoker_cigs_min__binding_rate` and
`R7_smoker_cigs_min__violation_rate`, each landing at the end of its own block per the
script's grouping; R7's violation rate exactly 0.0 on all five seeds.

**Why bother.** The committed CSV was no longer schema-identical to what its own script
produces at HEAD, so the next person to run `export_s0.py` would have seen a column diff
and had to reconstruct why. **Still outstanding:** `manifests/s0_e0.json` is written by
`write_run_manifest`, which also derives `binding_rates` / `violation_rates` from
`m["rules"]` and so is stale in the same way. It was left alone deliberately — it embeds
a pytest summary line, so regenerating it is not a pure column addition and does not
carry this entry's licence.

---

## 2026-08-20 — Three provenance categories, not two (supersedes the two-way split)

**Decision.** This entry replaces an earlier two-way sourced/stipulated split, which was
wrong: `age_range` fits neither side. There are **three** categories, and canonical
`dgp.py` now carries both structures that make the boundary checkable rather than
asserted.

**1. Sourced structural coefficients.** `b_bmi_glucose = 0.70` and
`b_hyperglyc_sbp = 1.76`. The citation licenses the magnitude. Recorded in
`dgp.COEFFICIENT_PROVENANCE`, copied **verbatim** from
`reference/dgp_frozen.py::COEFFICIENT_PROVENANCE` — the frozen wording is the only
authority for these strings, and copying text is not a runtime dependency on the frozen
reference, so it does not breach the freeze line. Do not reword, reformat or "improve" a
citation.

**2. Sourced window, stipulated shape.** `age_range` alone. The AHA PREVENT citation
licenses the **30-79 window** — the equations are validated over it — and says nothing
about the **distribution inside it**. The DGP draws `Age ~ Uniform(30, 79)`, chosen for
even coverage of the validated range, not for population realism; no cohort is uniform in
age. A reader who sees only the citation will infer otherwise, so the split is written
down at both places `age_range` appears in `dgp.py`.

`age_range` is also **not a free knob**. `glucose_base` and `sbp_base` are solved by
`solve_binding_rates()` conditional on the window, and the rule layer is not
orthogonalized against it. Sweeping the window on the frozen implementation while holding
the seed-0 calibration fixed moves R4 from 0.2017 at (30, 79) to 0.2467 at (40, 79) and
0.1328 at (30, 69) — against a declared band of 0.200 +/- 0.005, roughly nine and fourteen
band widths out. Prevalence and Bayes AUC are unmoved, because those are *solved*; the
binding rates are not. Contrast `p_smoker`, which leaves R2 at 0.378860 across its whole
range. (Frozen's digits, quoted as mechanism, not as targets.) **Changing `age_range`
without re-running `solve_binding_rates` silently invalidates every declared binding
rate.**

**3. Stipulated marginals.** `p_male`, `p_smoker`, `p_pregnant`, `preg_age_max`, the
`sd_*` family, `p_diabetic_midband`, `meds_age_scale`. Declared, not estimated; not to be
cited to any prevalence source. Enumerated in `dgp.COHORT_STIPULATIONS`, because the claim
that only *some* quantities are sourced is checkable only if the rest are listed rather
than left to inference from silence. They exist to give each rule enough binding mass to
carry signal — a design requirement, not a modelling claim.

`p_smoker = 0.20` is roughly double contemporary US adult cigarette-smoking prevalence
(9.9% in 2024 — an external figure, not derivable from this repo). It must not be sourced:
doing so would invite an audit of this DGP as a cardiovascular model, which it is not and
does not need to be.

**Inertness and drift.** Both structures are module-level data that nothing reads at
runtime; they cannot perturb generation. Verified rather than assumed — after adding them,
the E0 golden CSV regenerates byte-identical to the committed one.
`COEFFICIENT_PROVENANCE` duplicates each coefficient's magnitude next to its citation, and
that duplication is only defensible because it cannot drift:
`tests/test_dgp.py::test_coefficient_provenance_matches_config` asserts the cited values
equal the `DGPConfig` defaults. A citation attached to a number the model no longer uses
is worse than no citation, because it reads as evidence for whatever the field now holds.
`COHORT_STIPULATIONS` carries notes only, no values — there is no citation there for a
value to drift away from, so a second copy of the defaults would be a drift surface with
nothing to gain.

**How to apply.** In prose, say "stipulated cohort composition" for category 3, and for
`age_range` say "the validated 30-79 window, sampled uniformly by stipulation". If a
reviewer asks why smoking prevalence is 20%, the answer is that R1 and R7 need binding
mass on both sides of the smoker split.

---

## 2026-08-20 — One classed bands table; `EMERGENT_BANDS` and `SOLVED_BANDS` are gone

**Decision.** `tests/test_dgp.py`'s two band tables are merged into one,
`BINDING_BANDS`, mapping each rule to `(centre, width, class)` with `class` in
`derived` / `solved` / `emergent`. `EMERGENT_BANDS` was a misnomer once four of its seven
entries turned out to be derived, and the split between the two dicts encoded the
classification implicitly — a rule's class was "which dict it sits in, plus a comment".
Now it is data.

**Centres and widths are unchanged.** This is a restructure, not a recalibration: the
seed-0 fractions of band used are identical to before — R1 0.200, R2 0.216, R3 0.096,
R4 0.156, R5 0.252, R6 0.302, R7 0.200, worst case over seeds 0-4 is 0.500 (R3).

**What the class field now buys.**
`test_derived_band_centres_match_their_closed_forms` iterates the entries marked
`derived` instead of carrying its own hard-coded list, so a rule added as `derived`
cannot silently skip the closed-form check, and one added as `derived` without a closed
form in `_derived_centres` fails loudly rather than being skipped. The same test asserts
that the `solved` centres are exactly `E0_TARGETS` and that `set(BINDING_BANDS)` equals
`set(dgp.RULES)`, so an unbanded rule is caught too. All three failure modes were
verified by mutation.

**How to apply.** When adding a rule, add its band with a class. `derived` obliges you to
add its closed form to `_derived_centres`; the test will tell you if you forget.

---

## 2026-08-20 — Historical `git_dirty: false` stamps stand; the audit is closed

**Decision.** The provenance defect fixed in `5533c70` — `_write` splatting
`_provenance()` after `frame.to_csv`, so the `@cache` first evaluated on a tree the run
had already dirtied — was flagged as casting doubt on every manifest `export_s0.py` wrote
before that commit. **It does not.** No re-audit of `s0_config_grid` or
`s0_coupling1_ceiling` is needed.

**Why the defect is direction-safe.**

| tree before run | run changed an output | stamp | correct? |
|---|---|---|---|
| clean | no | false | yes |
| clean | yes | **true** | no — false alarm |
| dirty | no | true | yes |
| dirty | yes | true | yes |

Uncommitted **code** is visible to `git status` before the run and stays visible no matter
when `_provenance()` is evaluated. So `git_dirty: false` still implies no uncommitted code
at run time, which is the only question the field exists to answer. The defect can
manufacture a false *positive* and cannot manufacture a false *clearance*.

**How to apply.** Historical `false` stamps are trustworthy as they stand. Only historical
`true` stamps are suspect, and only in the narrow sense that some may be false alarms from
a run that changed an output on an otherwise clean tree — which matters only if a run was
ever discarded on that basis. None was. Do not reopen this.
