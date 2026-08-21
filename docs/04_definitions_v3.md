# S1 Definitions — v3, FROZEN
 
Status: **frozen.** This is S2 task 1, complete. Every runner brief (S2 task 2 onward)
is written against this document and audited against it. A defect here becomes a defect
in every rebuilt runner with no friction, so changes after this point are amendments
with a stated reason, not edits.
 
S1 is closed; nothing below relitigates an adopted decision.
 
**Changes in v2 (retained):**
1. §1B's rejection cap and fallback — adopted (500 attempts, margin-correction
   fallback, `fallback_used` diagnostic).
2. §0/§3's `y` / `y_det` notation split — adopted.
3. §5's gap claim corrected once: v1 had asserted low-but-nonzero-density rows were
   "most of what a real generator produces," citing a figure that does not support it.
**Change in v3 — §5 rewritten, and this one is substantive.** v1 and v2 both named
the harness's gap as "low-but-nonzero-density off-manifold rows." **Under §2's
definition that set is empty.** Every rule's violation region maps to a `-inf` branch
in `density.log_density_X`, and the SCM assigns each region zero occupancy (verified:
0 rows in all seven regions, 5 seeds × n=200,000). Rule violation implies zero density
analytically, so the harness is fully representative of off-manifold rows in this DGP
and no measurement is needed to establish it.
 
A different and narrower limitation replaces it: every rule here is stated over a
deterministic or structural region, so **this DGP cannot exhibit an
implausible-but-possible violation at all.** That is a property of the DGP's design,
not of the harness, and it belongs in the paper as such. See §5.
 
---
 
## 0. Shared notation
 
Two label spaces are used across these items and the source prose calls both of them
`y`. They are kept apart here because conflating them is exactly the kind of defect
this document exists to prevent:
 
- **`y`** — the classification-task label, `Heart_Disease_Risk`. Used in §1A, §1B,
  §1C, §4, §5.
- **`y_det`** — a detection-task ground truth: binary arm or manifold membership
  (e.g. "this row is in the harmful arm", "this row is off-manifold"). Used in §3
  only. `y_det` is never an input to the excess estimator.
Other notation, held constant throughout:
 
- `p*` — `TrueRisk(X_ref, manifest).p(X)`, per `corrupt.py`. `X_ref` is whichever
  frame's moments were frozen at construction; a runner brief must name `X_ref`
  explicitly and use the same one for every `p*` call inside that runner, since
  `TrueRisk`'s moments are fixed at `__init__` and mixing reference frames across
  arms of one comparison silently rescales `p*` between them.
- A **subset** is any row selection: `all`, `subset ⊆ pool`, `m = |subset|`.
- A subset is **feature-defined** if membership is a function of `X` alone (or of
  `p*(X)`, which is a function of `X`); **label-defined** if membership is a
  function of `y` (or `y_det`); **detector-score-defined** if membership is a
  function of a fitted model's output. See §4.1's untested middle case.
- **On-manifold** / **off-manifold** is defined in §2 and is a property of `X`
  alone, independent of `y`.
---
 
## 1A. The formal statement of harm
 
Fix a pool of `n` rows with true risks `p*(x_i)` and `n_pos = round(π·n)` for
declared prevalence `π`. Over all `y ∈ {0,1}^n` with `Σy = n_pos`:
 
> **The maximally harmful labelling at matched prevalence on matched features is
> the one placing positives at the lowest true risk** — `argsort(p*)[:n_pos] → y=1`.
 
This is a property of the labelling, stated over the whole assignment. **A single
row has no p\*-rank in isolation** and no per-row implementation may compute or
report a "harm score" for one row without reference to the pool it was drawn from.
 
`argmax_y mean|y − p*|` at fixed `Σy = n_pos` is the same object as
`argmax_y excess(y)` (§4) at the same constraint — proof: `Σ|y−p*| = n_pos + Σp* −
2Σ_{y=1}p*`, the first two terms fixed by the construction, so maximising
`mean|y−p*|` is exactly minimising `Σ_{y=1}p*`, which the local Bayes floor term
(depending only on features, also fixed) does not disturb. Nothing computes this
maximisation directly; it licenses reading `harmful_rank`'s construction in §1B
below as *the* harm-maximising arm, not *an* approximation to it.
 
The paper's per-record framing sits in tension with this per-assignment definition.
State the tension in the writing, do not paper over it: the metrics under test score
individual rows; the ground truth they are scored against is a property only of the
set they were drawn into.
 
---
 
## 1B. The agreeing arm — conditional Poisson via exponential tilting
 
`inject2.py::_place(weights=p)` currently does
`rng.choice(n, size=n_pos, replace=False, p=w)` — successive sampling, not the
design the docstring implies. Measured (n=20000, π=0.10, 200 replicates): top-decile
inclusion 0.358 against a 0.589 target, mean p\* among positives 0.553 against 0.702.
This is the arm to be replaced. It is the one built with `weights=p` in
`_place` — the "creative" / agreeing construction in `inject2.py`'s taxonomy, and
the general-purpose "agreeing at π" construction anywhere else it is needed
(κ sweep endpoints, §1C).
 
**Replacement algorithm.** Given per-row `p*_i` for `i` in the pool and a target
`n_pos`:
 
1. Clip `p*_i` to `[ε, 1−ε]`, `ε = 1e-12` (matches `dgp.py`'s own clipping
   convention), and work in logits: `ℓ_i = logit(p*_i)`.
2. Solve the scalar `t` such that `Σ_i expit(ℓ_i + t) = n_pos`, via `brentq` on
   `f(t) = Σ_i expit(ℓ_i + t) − n_pos`, bracket `[-60, 60]` (matches
   `dgp.py::_solve_alpha`'s bracket — the function is monotonic in `t` so a wider
   bracket costs nothing and a narrower one risks a `brentq` failure on an extreme
   pool).
3. Compute tilted probabilities `q_i = expit(ℓ_i + t)`.
4. Draw `y_i ~ Bernoulli(q_i)` independently.
5. If `Σy ≠ n_pos`: reject the draw and repeat step 4 (**not** step 2 — `t` is
   fixed once solved; only the independent Bernoulli draws are redrawn).
**Cap and fallback — adopted.** S1 said only "needs a cap and a fallback; at n≈24000
the sum concentrates tightly so acceptance is workable." Decided:
 
- Cap: 500 rejection attempts.
- Fallback on cap exhaustion: take the last rejected draw and correct it at the
  margin — repeatedly flip the 0→1 with `q_i` closest to 1 among current negatives,
  or 1→0 with `q_i` closest to 0 among current positives, whichever direction
  closes the gap between `Σy` and `n_pos`, until they're equal. This is a targeted
  minimal correction rather than a fresh uniform redraw, so it perturbs the
  achieved inclusion probabilities as little as possible relative to the tilted
  design.
- Every call reports whether the cap was hit (`fallback_used: bool`) and the number
  of rejection attempts consumed, in the returned diagnostics — a run that hits the
  fallback on a nontrivial fraction of draws is a sign the tilting itself is
  misbehaving (e.g. `n_pos` too close to 0 or `n`) and should be surfaced, not
  silently absorbed.
**Justification for the paper** (already settled, not open): conditional Poisson at
the solved tilt is the maximum-entropy design for a fixed sample size given
first-order inclusion probabilities — the least committal labelling consistent with
the boundary at the required prevalence.
 
**The free reference row.** The **untilted** draw — `y_i ~ Bernoulli(p*_i)`
independently, no conditioning on `Σy`, no tilting — is reported alongside as a
free reference, not as an experimental arm. It is what real data does, at
whatever prevalence it happens to realise, and it exists to show what the
prevalence-matching constraint (tilting) costs relative to the unconstrained
process. It is never substituted into a recipient-model comparison expecting a
matched-prevalence arm, since its prevalence is random by construction.
 
---
 
## 1C. κ
 
Defined on the *induced statistic* of an observed or constructed placement, not on
the tilt parameter `t` — this is what makes κ computable for generator output,
where there is no `t` to read off.
 
Given a pool with mean true risk `p̄`, a placement with `n_pos` positives and
`S = mean_{i: y_i=1} p*_i`, and `S_min`, `S_max` the achievable extremes of `S` at
that `n_pos` within the same pool (`S_min` = mean `p*` of the `n_pos` lowest-`p*`
rows in the pool; `S_max` = mean `p*` of the `n_pos` highest):
 
```
κ = (p̄ − S) / (p̄ − S_min)     if S ≤ p̄   (contradicting placements)
κ = (S − p̄) / (S_max − p̄)     if S ≥ p̄   (agreeing placements)
κ = 0                          if S == p̄  (both formulas are 0/0 here; define directly)
```
 
`κ = −1` at the rank extremum (`harmful_rank`, §1B's harm-maximising arm),
`κ = +1` at perfect agreement (`S = S_max`), `κ ≈ 0` at random placement.
 
For an observed generator sample (copula, ARF): the pool is that generator's own
synthetic sample (or whatever sub-pool the runner defines — named explicitly in the
brief), `p*` computed via `TrueRisk` against the runner's declared `X_ref`, `S`
computed from the generator's own labels. No construction is needed; κ is read
directly off what the generator produced. This is what places generator output on
the same axis as the constructed arms.
 
The κ **sweep** (varying `t` and reporting κ across the range) is deferred per the
priority list — item 2 on the enrichment list. This definition is adopted and usable
now regardless (definitions cost no evidence); only the sweep experiment is
deferred. The extremum claim (κ=−1 is the harm-maximising placement, by §1A's proof)
does not depend on the sweep.
 
---
 
## 2. On-manifold — operational definition
 
**On-manifold is defined by rule membership, not by density.** A row is on-manifold
if and only if it violates none of R1–R7, where:
 
- R1–R6 are `dgp.RULES` as currently defined (canonical `dgp.py`, `evaluate_rules`).
- **R7** (`Current_Smoker == 1 → Cigs_Per_Day >= 1`) is a **freeze-line addition** to
  `dgp.RULES` — not a post-hoc filter as it was in `r7_decomp.py`, where it was
  deliberately kept out of `dgp.RULES` and applied only inside that script. Per S1,
  it goes into canonical `dgp.RULES` for real, with the E0 golden re-run that
  implies. This is S2 brief #2's job; once done, `evaluate_rules` carries R7
  natively and no runner needs to apply it as a separate filter.
- R1's consequent is evaluated with the tolerance in `rules_tol.py`
  (`Cigs_Per_Day < tol`, `tol = 0.5`) rather than exact equality to 0. Exact
  equality on a continuous column is not a meaningful violation test for any
  generator that doesn't reproduce point masses exactly, and inflates off-manifold
  counts for reasons that have nothing to do with the boundary. **This tolerance
  applies whether or not R7 has yet been added to `dgp.RULES`** — it is a numerical
  patch to R1's exact-equality check, orthogonal to R7's separate logical extension
  of rule coverage.
- Violation = binding & not-satisfied, per rule, exactly as `evaluate_rules` /
  `check_rules_tol` already compute it. A row is on-manifold iff it is not flagged
  `__violated` under any of the seven.
**Density is the instrument that validated this, not the definition.** It is not
recomputed at runtime as part of the on-manifold test in any rebuilt runner. It
stays in the paper as the method that established R7's adequacy (§ evidence below),
because that derivation is reusable by another metric-builder in a way a bare
definition is not.
 
**Evidence, already committed, restated for reference (not to be rerun as part of
this document, only cited):**
 
- 12.6–12.7% of R1–R6-clean copula rows had exactly zero density under
  `density.log_density_X`, traced entirely to R1 being one-directional (catches
  `smoker=0 & cigs>0`, not `smoker=1 & cigs=0`).
- R7 removes every zero-density row: `frac_density_zero` exactly 0.0000 on every
  seed tested, verified as exact array equality.
- On-manifold excess share moves 76.7% [sd 0.28] → 71.7% [sd 0.50] under R7.
**Scope, stated explicitly wherever this definition is used — and load-bearing, not
pro forma (see §5's closing thread).** Adequacy is measured for the copula generator
only. ARF is untested on this axis (its zero-density
failure mode, if any, may not be R1's one-directionality at all) and must not be
assumed to inherit the copula's numbers. Testing ARF (and later CTGAN) against this
same instrument is on the enrichment list, not required for the A2/A3/A4 rebuild.
 
---
 
## 3. Per-stratum reporting — detection metrics only
 
**Applies to:** any metric scoring how well a per-row score or fitted detector
ranks/separates rows against a ground-truth `y_det` (arm membership, on/off-manifold
membership, or any other binary detection target) — Task B AUROC-type quantities,
and anything of that shape (average precision, etc., if used).
 
**Does not apply to:** contradiction-magnitude quantities (`excess`, `mean|y-p*|`,
and anything built from them). Those are governed by §4, whose trap (§4.1) is
violated by exactly the stratification this item requires for detection metrics —
the two conventions are both live at once and must not be swapped.
 
**Requirement:** report pooled, `y_det=1`-stratum, and `y_det=0`-stratum. Headline
stratum by prevalence of the *detection* target: `y_det=1` when its prevalence is
below 0.5, `y_det=0` when above, both at exactly 0.5.
 
**Rationale (carries into the paper, not just the runner):** for the classification
task's own label `y` at contradiction, `|y − p*| ≤ p*` when `y=0` and `≤ 1−p*` when
`y=1`; at classification prevalence `π` with `mean p* ≈ π`, max achievable
contradiction is `≈ π` in the `y=0` stratum and `≈ 1−π` in the `y=1` stratum — a
ratio of `(1−π)/π`, 9× at `π=0.10`. This is about the classification label `y`, and
explains why pooled Task-B-style numbers involving `y`-conditioned contradiction are
structurally muted, not about `y_det`. Do not conflate the two label spaces when
writing this justification into the paper — see §0.
 
Stratifying is validating, not cherry-picking: on the existing A2 numbers, weak
detectors stay weak in the headline stratum (DCR 0.5229 pooled → 0.5332 on
`y_det=1`) while the oracle separates sharply (0.5936 → 0.9966).
 
---
 
## 4. The excess estimator
 
```
excess(subset) = mean_{i ∈ subset} |y_i − p*_i|  −  2 · mean_{i ∈ subset} [p*_i·(1−p*_i)]
```
 
Both terms computed **within** the same subset — this is what makes the floor
local. Corruption raises `p*` and raises the floor with it; comparing a subset
against real data's global floor overstates contradiction wherever `p*` is
elevated (the E12 failure, superseded, do not repeat its construction).
 
**Unbiasedness and closed-form se.** Under `y_i ~ Bernoulli(p*_i)` row-independent
(the "null" / real-data-like generating process), `excess` is exactly unbiased at
every subset size, not asymptotically. Standard error:
 
```
se(subset) = sqrt( mean_{i ∈ subset}[ p*_i·(1−p*_i)·(1−2p*_i)² ] / m ),   m = |subset|
```
 
This must be computed from **that subset's own `p*` values**, every time. The
`se ≈ 0.173/√m` shortcut quoted in the S1 record is an E0-config-specific
back-of-envelope number (0.173 is `sqrt(mean[p*(1−p*)(1−2p*)²])` evaluated on real
data's `p*` distribution at the E0 config) — it is not a general identity and must
not be substituted for the formula above in any rebuilt runner. Using it on a
cell whose `p*` distribution differs from real data's (any corrupted or off-manifold
subset will differ) is exactly the error S1 flagged as moving numbers by 10–20%; the
fix is simply always evaluating the formula on the cell in hand, which this document
requires unconditionally.
 
**Two-branch claim rule.** The branch is a property of **the claim being made about
a cell**, not an intrinsic property of the cell — the same `(excess, se, m)` triple
can support a difference claim against one comparison value and an equivalence claim
against another. A runner cannot auto-assign a status; the brief for any experiment
producing claim-bearing cells must state, per cell, which claim is being tested and
against what comparison value. Given that:
 
- **Difference claim** ("nonzero", "larger than `V`"): let
  `effect = |excess − V|` (`V = 0` for a bare "nonzero" claim). Supported iff
  `se < effect / 3`.
- **Equivalence claim** ("indistinguishable from `V`", typically `V=0`, optionally
  also "and distinguishable from `W`"): construct `[excess − 3·se, excess + 3·se]`.
  Supported iff this interval contains `V` and — when a contrasted alternative `W`
  is named — excludes `W`.
  Worked reference (already measured, cited not rerun): ARF on-manifold excess
  0.0022, se 0.00171 — ratio 0.78, fails the difference branch (as it should; the
  smallness *is* the claim here, not a nonzero effect). Under the equivalence
  branch: `3·se = 0.0051`, interval `[−0.0029, 0.0073]` contains 0 and excludes the
  copula's 0.0353. Passes.
**Suppression.** A cell whose applicable branch is not satisfied is shown in tables
with its `(excess, se, m)` — never dropped — but is not available for interpretation
anywhere in the paper's prose. Flagging a cell as underpowered in a table and then
calling it suggestive in text is the exact claim-evidence mismatch this rule exists
to prevent.
 
**Output contract for every runner producing excess cells:** store `excess`, `se`,
`m`, and per-seed values (so cross-seed sd is a column, not a re-derivation), for
every cell — whether or not that cell currently supports a claim. This is what makes
the threshold-sensitivity check ("which cells change status at a ⅓→½ or ⅓→⅕
threshold") a post-hoc read of the stored table at write-up time rather than a
rerun.
 
### 4.1 The trap — carry into every excess-producing brief
 
**Excess must never be computed within a `y`-defined (or `y_det`-defined) subset.**
Measured on data with zero injected contradiction (n=16000):
 
| subset            | n     | excess    |
|---|---|---|
| all rows          | 16000 | −0.00069  |
| `y==1` stratum     | 1550  | **+0.37394** |
| `y==0` stratum     | 14450 | **−0.04087** |
| `p* below median` | 8000  | −0.00039  |
| `Age < 55`         | 8115  | +0.00137  |
 
Conditioning on `y` changes `E[y | p*]` within the subset, so the estimator stops
measuring contradiction and starts measuring the conditioning itself. **Feature-
defined subsets are fine** (rows 4–5 above); label-defined subsets are not (rows
2–3). This is exactly the split defined in §0.
 
This collides in practice with §3, which *requires* label-defined (`y_det`-defined)
stratification for detection metrics on the very same experimental runs. A runner
computing both a Task-B-style AUC (§3, stratify by `y_det`) and an excess value
(§4, never stratify by `y` or `y_det`) on the same population must apply each rule
to its own metric independently. State the mechanism in code comments at the point
of computation, not just follow the rule silently — a future reader (or a future
brief) copying the stratification loop from the detection-metric code path into the
excess code path is the failure mode this guards against.
 
**Open, untested (not blocking the A2/A3/A4 rebuild, noted for the enrichment
list):** subsets defined by a fitted detector's score. Such a subset is a function
of features, but the detector was fit against something `y`-correlated, so whether
this leaks the same way a `y`-defined subset does is unresolved. Detector-score-
defined subsets should not be used for excess reporting until this is checked.
 
---
 
## 5. The corruption harness
 
`corrupt.py`'s four mutations, applied one per selected row (uniformly among
whichever are eligible for that row):
 
| rule broken       | eligibility                                    | mutation |
|---|---|---|
| R2_glucose_floor  | `Fasting_Glucose < 100 & Diabetic == 0`         | `Diabetic ← 1` |
| R4_bp_mandatory   | `Systolic_BP > 140 & On_BP_Medication == 1`     | `On_BP_Medication ← 0` |
| R5_anatomical     | `Biological_Sex == 1 & Is_Pregnant == 0`        | `Is_Pregnant ← 1` |
| R1_smoker_cigs    | `Current_Smoker == 0 & Cigs_Per_Day == 0`       | `Cigs_Per_Day ← 12.0` |
 
All four land in deterministic or structural regions of the DGP, so every corrupted
row has **exactly zero density** under `density.log_density_X` — not merely low
density. (R1's mutation here is the mirror image of the mechanism §2 found the
copula producing naturally: R1 is one-directional, and this harness exploits the
direction R1 itself does not catch on its own consequent side.)
 
**Claim scope, narrowed and adopted:** the harness demonstrates "structurally
impossible rows protect the recipient," not "off-manifold rows do" in general.
 
**The gap this leaves is not the one v1 and v2 named.** Both said the gap was
"low-but-nonzero-density off-manifold rows." Under §2's definition — off-manifold
means violating a rule — **that set is empty**, and the emptiness is analytic rather
than a measurement:
 
Every rule's violation region maps to a `-inf` branch in `density.log_density_X`.
R1's `smoker=0 & cigs!=0` hits the non-smoker atom; R2 and R3 hit the `lo_band` /
`hi_band` diabetic determinism; R4 hits the `meds==1` forcing above 140; R5 and R6
hit pregnancy ineligibility; R7 falls strictly outside `_clip_atoms(15, 5, 1, 40)`.
Confirmed upstream on the SCM: **zero occupancy in all seven violation regions**
across 5 seeds at n=200,000. `Diabetic` is stochastic only inside the glucose
midband, `On_BP_Medication` only at SBP ≤ 140, and `Cigs_Per_Day` has support
`{0} ∪ [1,40]` with positive minimum exactly 1.0.
 
So **rule violation implies zero density, by construction**. The harness is not
unrepresentative of off-manifold rows; in this DGP, off-manifold rows are *all*
structurally impossible. The strawman objection dissolves rather than being bounded,
and no runner is needed to bound it.
 
**The limitation that replaces it is a property of the DGP, not the harness.** Every
rule here is stated over a deterministic or structural region, so **this DGP cannot
exhibit an implausible-but-possible violation at all**. Some real constraint
violations are of that kind (a pregnant male); many are not — a BMI of 12 with
otherwise normal covariates is implausible, not impossible, and no rule catches it.
State this as a design property in the paper. It is narrower to defend than the old
framing and it is the honest one.
 
**One live thread, generator-specific.** `rules_tol` tolerates R1 at `tol = 0.5`, so
a non-smoker with `Cigs_Per_Day ∈ (0, 0.5)` is rule-clean yet zero-density. That band
is **empty for `GaussianCopulaDT`**, which resamples empirical quantiles and so cannot
emit a value absent from training — confirmed 0 across all 25 fits in the S2 occupancy
run. A generator that smooths continuous columns (SDV's parametric marginals, CTGAN,
TabDDPM) can land there. So "rule violation ⟺ zero density" holds exactly under exact
rules, and holds under tolerated R1 only because the tolerance band is unreachable for
this particular generator. **§2's copula-only scope caveat is load-bearing, not pro
forma.** The `n_cigs_in_tol_band` column exists to measure it per generator.
 
Enrichment item 8 (extending the harness to perturb continuous columns away from
their conditional means) is unaffected and remains out of scope for the A2/A3/A4
rebuild. It is now better motivated: it is the only route to an
implausible-but-possible arm, which this DGP cannot otherwise produce.
 
**Implementation status:** status:** canonical `corrupt.py` currently has `TrueRisk` only.
`VIOLATIONS` and `apply_violations` are absent from canonical and are the first
piece of S2 brief #2 — ported from the reference implementation (eligibility masks,
per-row single-violation choice via `rng.integers` over eligible options, in-place
column mutation) rather than rewritten from a blank page, since the reference
implementation is exactly the specification above already expressed in code.
 
---
 
## 6. Cross-reference: what each item governs, at a glance
 
| item | governs | never | primary file |
|---|---|---|---|
| 1A | definition of a maximally harmful *labelling* | scoring a single row in isolation | — (proof only) |
| 1B | constructing the agreeing/creative arm at a target prevalence | using `_place(weights=p)` unmodified | `inject2.py` |
| 1C | placing an observed or constructed placement on [−1,1] | reading κ off `t` directly | — |
| 2 | on/off-manifold test | using density as the definition | `dgp.RULES`, `rules_tol.py` |
| 3 | stratifying **detection** metrics by `y_det` | applying this rule to excess | Task B runners |
| 4 | computing excess + its se + claim status | computing excess inside a `y`/`y_det` subset | — |
| 5 | what the corruption harness does and does not prove | claiming it covers low-density off-manifold rows | `corrupt.py` |
 
---
 
## 7. Decisions taken in S2 that were not in S1
 
Both closed. Recorded here and belonging in `decisions.md`.
 
1. **§1B's rejection cap and fallback.** 500 attempts; margin-correction fallback
   (flip the row nearest its threshold, one at a time, until `Σy = n_pos`) rather
   than a fresh redraw; `fallback_used` and attempt count in returned diagnostics.
   Rationale for the fallback shape: it perturbs the achieved inclusion
   probabilities less than a uniform redraw would, so a run that hits it is still
   close to the tilted design rather than silently degraded to something else.
2. **§0/§3's `y` vs `y_det` notation split.** S1's prose used `y` for both the
   classification label and the detection ground truth. They are distinct variables
   and are named distinctly from here on. This is a clarification, not a new
   research decision, but it is load-bearing: §4.1's trap and §3's requirement pull
   in opposite directions on the same runs, and a shared variable name is one
   keystroke from reproducing the +0.374 bug.
Everything else above restates an S1-adopted decision in implementable form, or
narrows an S1-flagged scope statement (§2's copula-only adequacy, §5's
structural-impossibility-only claim) without changing what was decided.
 
