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
for the four derived rules so the rule survives as more than a comment. Note that
`EMERGENT_BANDS` is now a misnomer: four of its seven entries are derived, not emergent.
Renaming it touches every call site and was left for a follow-up.

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

## 2026-08-20 — Cohort marginals are stipulations, not epidemiological estimates

**Decision.** `p_smoker`, `p_male`, `p_pregnant`, `preg_age_max` and `age_range` fix the
composition of the simulated cohort. They are **declared stipulations**. They are not
estimates of any real population and must never be cited to a prevalence source.

**What is sourced, and where.** Only the structural coefficients are literature-backed:
`b_bmi_glucose = 0.70` and `b_hyperglyc_sbp = 1.76` (Mendelian-randomization estimates)
and the 30-79 age range's origin in the AHA PREVENT validated range. In canonical `dgp.py`
these carry inline `# MR estimate` comments on the `DGPConfig` fields; the full citations
live in `reference/dgp_frozen.py::COEFFICIENT_PROVENANCE`, which is comparison-only and
must not become a runtime dependency. Note that the age range is doing two jobs — it is
sourced *and* it enters R6's derived band centre as a cohort stipulation.

**Why this is worth writing down.** `p_smoker = 0.20` is roughly double contemporary US
adult cigarette-smoking prevalence (9.9% in 2024 — an external figure, not derivable from
this repo). Sourcing it would invite an audit of the DGP as a cardiovascular model. It is
not one and does not need to be: no claim in this line of work depends on a marginal's
epidemiological accuracy. The marginals exist to give each rule enough binding mass to
carry signal, which is a design requirement, not a modelling claim.

**How to apply.** In prose, call these "stipulated cohort composition". If a reviewer asks
why smoking prevalence is 20%, the answer is that R1 and R7 need binding mass on both
sides of the smoker split, not that any population looks like this.
