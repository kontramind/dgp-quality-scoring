# Experimental record v2 — per-record quality scoring for synthetic tabular data
 
All numbers measured, not estimated.
 
**v2 revision note (audit session).** Experiments A1–A6 below were run against the
existing code base and force three changes to the record:
 
1. **E3, E6, E10 and E12 are measured under the definition of harm that E5 was
   superseded for** (`corrupt.build_populations` still assigns
   `y = 1[p*(x) < 0.5]`). They are marked SUPERSEDED and must not be cited.
2. **The 2×2's support axis is refuted as the harm mechanism.** On-manifold label
   contradiction does 2–8× the downstream damage of off-manifold contradiction at
   matched prevalence, volume and contradiction magnitude (A3).
3. **Every "fitted" number in the record was computed with an untuned holder model.**
   Tuning roughly halves the oracle gap (A5).
Superseded results are retained so the correction is auditable.
 
**Global caveats.** One DGP configuration; two generators (Gaussian copula with the
distributional-transform fix, ARF); seed counts stated per experiment; DCR / kNN /
adversarial validation are reimplementations, α-precision is synthcity's own for E10.
A1–A6 were run on a single-core box at reduced sample size and seed count — they are
direction-setting, not publication numbers.
 
---
 
## E0 — DGP verification (unchanged)
 
Config: `solve_binding_rates({"R3_glucose_ceil":0.08, "R4_bp_mandatory":0.20})`,
solved `glucose_base = 102.2 mg/dL`, `sbp_base = 109.3 mmHg`.
 
| Property | Target | Realised |
|---|---|---|
| Prevalence | 0.10 | 0.1003 |
| Bayes-optimal AUC | 0.85 | 0.8500 |
| Rule violations on real data | 0 | **0** |
 
Rule binding rates: R1 0.803 · R2 0.379 · R3 0.080 · R4 0.202 · R5 0.497 · R6 0.0100.
λ leaves features bit-identical and changes only the labelling function. R1 is an exact
equality on a continuous column; use `rules_tol.check_rules_tol(df, tol=0.5)`.
 
---
 
## E1 — Leaf residual vs `predict_proba` (unchanged conclusion)
 
`residual` is a deterministic function of `(y, p̂_leaf)`; the 38% "unexplained variance"
comes from the non-monotone `−H(p̂)` subtraction (Pearson +0.787, Spearman −0.332). The
aleatoric correction costs 6.4 AUROC points on every task measured. `leafscore.py`
leaves the method and stays only as C5's negative result.
 
---
 
## E2 — Label-flip detection (features untouched, 20% flipped) — SINGLE SEED
 
`|y − predict_proba|` 0.9613 · leaf axis 2 raw 0.9609 · adversarial 0.930 · leaf
corrected 0.8970 · kNN 0.851 · IF 0.837 · α-precision 0.822 · DCR 0.801. Feature-only
variants at chance (0.503–0.509).
 
The "existing metrics are blind" claim holds only for feature-only variants. Do not
overclaim. **Needs error bars and re-running under the A2 construction.**
 
---
 
## E3 — Four-quadrant test — **SUPERSEDED by A1/A2**
 
Retained numbers: adversarial 0.9996 Task A / 0.549 Task B; DCR 0.979 / 0.603; kNN
0.977 / 0.632; `|y − predict_proba|` 0.812 / 0.927; α-precision `_OC` 0.9280 / 0.5526;
IF 0.788 / 0.756; leaf axis 2 raw 0.780 / 0.908; leaf support 0.716 / 0.503.
 
**Why superseded:** the harmful arm is 84.7% positive and the creative arm 22.4%
(A1), so Task B is 0.811-separable by the label alone. The mechanism findings survive
(objective mismatch explains leaf support's failure; the point-mass artifact check
stands) but every Task B figure must be re-measured.
 
---
 
## E4 — Downstream utility via filtering — SUPERSEDED by E8
 
Artifact of the E7 copula bug. Axis-2 deletion at 1.1% prevalence was a positive-class
deletion operator, not self-distillation.
 
---
 
## E5 — C1 via injection — SUPERSEDED by E11
 
`inject.py` assigned harmful labels as `(p_new < 0.5)`, making them 91% positive and
rebalancing training prevalence 0.012 → 0.163. The patch in `inject.py` is also broken
(supply-limited, delivers 38% of requested rows). **Do not use `inject.py`.**
 
---
 
## E6 — Oracle gap — **SUPERSEDED by A1/A2/A5**
 
Retained: oracle 0.9566 ± 0.0008, fitted 0.9110 ± 0.0007, leaf axis 2 0.8903, DCR
0.6130, adversarial 0.5746. Reproduced exactly in A1.
 
**Why superseded:** built on the threshold construction (A1), and computed with an
untuned holder model (A5). Both corrections move the headline number.
 
---
 
## E7 — The Gaussian copula tie-handling bug (stands)
 
`downstream.GaussianCopula.fit` gaussianises average ranks; for a binary column `Z`
collapses to two atoms and the empirical-quantile inverse returns the modal category.
`Heart_Disease_Risk` 0.1015 → 0.0111; `Is_Pregnant` 0.0105 → 0.0000. Fixed in
`copula_fix.py` (distributional transform). Worth a paragraph in the paper
independently: this pattern is copied widely.
 
---
 
## E8 — Downstream utility, corrected copula and ARF (stands; see A4 for mechanism)
 
Real ceiling LogReg 0.8457 / MLP 0.8423 / LGBM 0.8313. Copula no-filter 0.8233 /
0.7961 / 0.7060; axis-2 filter 10% → 0.8446 / 0.8349 / 0.8280; positives-matched random
control 0.8121 / 0.7271 / 0.5586. ARF: everything flat, no intervention differs from
no-filter beyond noise. Relabelling matches or beats every filter on both generators.
 
**A4 supplies the missing mechanism.** Filtering has headroom only when the generator
is poor because only a poor generator produces the contradiction, and support-based
filtering fails on it because the contradiction sits on-manifold.
 
*Recipients were never tuned* — LGBM at `150/31`. Part of the copula's 0.706 is likely
fittable away. Denominators need re-running.
 
---
 
## E9 — Rotation test (stands; hypothesis refuted)
 
ARF's downstream utility is invariant to a change of basis (LGBM 0.8290 → 0.8266 across
three rotation matrices). Axis alignment is not the explanation for ARF's advantage.
The off-manifold jump under rotation is an artifact of R1's exact equality — no dose
response, saturates at θ = 0.05 rad. Argue decoupling from the θ=0 data.
 
---
 
## E10 — Baselines against synthcity — **Task B figure SUPERSEDED**
 
synthcity's `_OC` α-precision (the variant Alaa et al. specify) reads 0.9280 Task A /
0.5526 Task B; `baselines.py` is a faithful `_naive` at 0.7865 / 0.6890. The published
implementation is better on support and closer to chance on the boundary. The Task A
comparison stands; the Task B figure inherits E3's construction problem.
 
---
 
## E11 — C1 via injection, rebuilt (stands, but see A3)
 
`inject2.py`. Every arm delivers the same row count at the same class prevalence; the
four corrupted arms share a feature distribution. Diagnostics confirm exact matching
(injected prevalence 0.1014 across arms, mean `p*` 0.223 across all four corrupted
arms).
 
Copula, 20% injection, 3 seeds: harmful_rank −0.1697 / −0.0596 / −0.0380; creative
+0.0134 / −0.0006 / +0.0346; clean, balance, random all at zero. Monotone dose response
on LogReg: −0.0127, −0.0301, −0.0612, −0.1697 at 2/5/10/20%. Reproduces in sign on ARF
but much smaller.
 
**A3 shows the effect is not about off-manifold-ness.** C1 must be restated as a claim
about label-vs-boundary structure, with support as an attenuating factor.
 
**Note on the `creative` arm.** Prevalence matching forces creative to be a top-decile
threshold on `p*`, not a draw from `Bernoulli(p*)` — the corrupted pool's mean `p*` is
0.223 while injected prevalence is 0.101, so coherent labels are arithmetically
impossible. `creative` is E8's relabelling operator under another name.
 
---
 
## E12 — Quadrant occupancy — **SUPERSEDED by A4**
 
Retained: copula off-manifold 0.2645, harmful quadrant 0.0217; ARF 0.1110 / 0.0037.
 
**Why superseded:** measures occupancy of the cell A3 shows is not the dangerous one,
and compares each subset's mean `|y − p*|` against real data's global 0.1375 without
correcting for the fact that corruption raises `p*` and raises the Bayes floor with it.
A4 replaces it.
 
---
 
# A-series: audit session
 
## A1 — Task B is separable by the label alone (`diag1.py`, `diag3.py`)
 
`corrupt.build_populations` quadrant composition, 3 seeds, n=60 000:
 
| quadrant | n | prevalence | mean `p*` | mean `\|y−p*\|` |
|---|---|---|---|---|
| clean | 8166 | 0.117 | 0.117 | 0.152 |
| creative | 7164 | **0.224** | 0.226 | 0.235 |
| harmful | 7170 | **0.847** | 0.224 | 0.838 |
| label | 7500 | 0.899 | 0.101 | 0.861 |
 
**A detector that ignores the features and reports `y` scores 0.8112 ± 0.0041 on
Task B.** E6 reproduces exactly (fitted 0.9110 ± 0.0007, oracle 0.9566 ± 0.0008), so the
protocol is sound and the leak is in the construction.
 
Decomposition by label stratum:
 
| detector | pooled | y=1 only | y=0 only |
|---|---|---|---|
| oracle `\|y−p*\|` | 0.9566 | 0.8647 | 0.9786 |
| fitted `\|y−p̂\|` | 0.9110 | 0.7775 | 0.8975 |
| DCR | 0.6130 | **0.4364** | 0.7055 |
 
The pooled fitted score exceeds both of its strata — aggregation gain from the
62-point class-balance difference, not boundary information.
 
## A2 — Task B rebuilt on the rank construction (`diag2.py`, `diag3.py`)
 
One corrupted pool split in half, same prevalence (0.10), positives at lowest `p*`
(harmful) or ∝ `p*` (creative). Label prevalence uninformative by construction.
 
| detector | pooled | y=1 only |
|---|---|---|
| oracle `\|y−p*\|` | 0.5936 ± 0.0038 | **0.9966 ± 0.0028** |
| fitted `\|y−p̂\|` | 0.5807 ± 0.0013 | **0.9715 ± 0.0020** |
| leaf axis 2 | 0.5522 | — |
| kNN density | 0.5290 | — |
| DCR | 0.5229 | 0.5332 |
| α-precision (naive) | 0.5166 | — |
| adversarial validation | 0.5012 | — |
| label alone | 0.5000 | 0.5000 |
 
**At 10% prevalence, per-row harm is a positive-label phenomenon.** A `y=0` row at
moderately elevated `p*` is barely contradictory, so pooled Task B is uninformative.
Report per stratum.
 
## A3 — The support axis is not the harm mechanism (`diag4.py`, `diag6.py`, `diag7.py`)
 
The paper's 2×2 has never had its high-support × label-disagrees cell measured
downstream. Three constructions of increasing rigour, copula base, 20% injection:
 
| construction | ON pool | matching | ΔLGBM OFF harmful | ΔLGBM ON harmful |
|---|---|---|---|---|
| `diag4` | uncorrupted, no resampling | prevalence, volume | −0.040 | **−0.111** |
| `diag6` | 36% violating, `p*`-matched | + `p*` distribution | −0.034 | **−0.215** |
| `diag7` | **0% violating**, `p*`-matched | + `p*` distribution | −0.034 | **−0.236** |
 
`diag7` pools, 5 seeds (base LogReg 0.8275 / MLP 0.7947 / LGBM 0.7016):
 
| pool | mean `p*` | `p*` of positives | mean `\|y−p*\|` | violation rate |
|---|---|---|---|---|
| OFF (corrupted) | 0.2278 | 0.0135 | 0.3276 | 1.00 |
| ON (strict) | 0.2245 | 0.0129 | 0.3245 | 0.00 |
 
| arm | ΔLogReg | ΔMLP | ΔLGBM |
|---|---|---|---|
| OFF creative | +0.0154 ± 0.0008 | +0.0006 ± 0.0149 | +0.0298 ± 0.0131 |
| ON creative | +0.0075 ± 0.0078 | +0.0151 ± 0.0076 | +0.0297 ± 0.0073 |
| OFF harmful | −0.1639 ± 0.0270 | −0.0421 ± 0.0130 | −0.0338 ± 0.0144 |
| **ON harmful** | **−0.4046 ± 0.0759** | **−0.3507 ± 0.0257** | **−0.2362 ± 0.0237** |
 
Contradiction magnitude is matched to three decimals, and by every magnitude measure
the OFF arm is the *stronger* contradiction (`diag5.py`: mean `|y−p*|` 0.3292 vs
0.2028; excess over Bayes floor 0.0874 vs 0.0636; corr(y, `p*`) −0.3108 vs −0.2239).
It does less damage anyway.
 
**Being off-manifold protects the recipient.** Creative is identical in both pools
(+0.0298 vs +0.0297 LGBM), so neither half of C1 is about off-manifold-ness.
 
*Caveat:* the ON pool in `diag7` has ~30% duplicated rows from with-replacement
`p*`-matching. `diag4` has no duplication and shows the same direction.
 
## A4 — Where generator label error actually lives (`occ_cop.py`, `occ_arf.py`)
 
`excess = mean|y − p*| − 2·E[p*(1−p*)]`, computed **within each subset** so the Bayes
floor tracks the local `p*`. Symmetric noise gives excess ≈ 0. Copula 5 seeds, ARF 3
seeds (fitted on a 12k subsample for runtime).
 
| generator | subset | mass | excess | AUC(`p*`, y) | share of excess |
|---|---|---|---|---|---|
| **REAL** | all | 1.000 | **0.0020** | 0.8525 | — |
| **copula** | all | 1.000 | **0.0346** | **0.5873** | — |
| | on-manifold | 0.737 | 0.0363 | 0.5886 | **77.5%** |
| | off-manifold | 0.263 | 0.0296 | 0.5623 | 22.5% |
| **ARF** | all | 1.000 | **0.0047** | 0.8365 | — |
| | on-manifold | 0.852 | **0.0022** | 0.8380 | 39.3% |
| | off-manifold | 0.149 | 0.0192 | 0.8212 | 60.7% |
 
Real data's excess of 0.0020 calibrates the estimator.
 
**Three findings.** (1) The copula's label error is structured, not symmetric noise —
17× real data's residual, and its labels retain 0.587 AUC against the true boundary
where real data has 0.853. (2) 77.5% of it sits on-manifold, where A3 says it does ~7×
the per-row damage. (3) ARF's on-manifold excess (0.0022) is indistinguishable from
real data's own residual; what little it has is off-manifold.
 
**This explains E8.** Where a generator's label error lives depends on generator
quality. For ARF, contradiction is proportionally off-manifold but negligible in
absolute terms, so a support filter is aimed roughly right and finds nothing. For the
copula it is on-manifold and abundant, so the filter is aimed at the wrong three
quarters.
 
## A5 — The oracle gap is a property of the fit, not the boundary (`tune_phat.py`)
 
28 configs × 3 seeds, tuned by early stopping on a validation split; model quality
measured on a held-out half never used for population construction.
 
| config | model AUC | Task B (threshold) | oracle gap | Task B (rank, y=1) | oracle gap |
|---|---|---|---|---|---|
| paper default 300/63/.05 | 0.8310 | 0.9086 | **+0.0476** | 0.9733 | +0.0249 |
| best tuned (rand02) | 0.8454 | 0.9374 | **+0.0188** | 0.9901 | +0.0081 |
| best model (rand23) | 0.8520 | 0.9293 | +0.0269 | 0.9853 | +0.0130 |
| crippled 5 trees/2 leaves | 0.7892 | 0.8611 | +0.0951 | 0.6759 | +0.3223 |
 
The paper default is undertuned — 24 of 28 configs beat it, and it sits 0.019 below the
Bayes ceiling. **"Estimating the boundary costs 4.6 AUROC points" is ~2× too
pessimistic;** the tuned figure is ~2 points, or under 1 on the rank construction.
 
**Model quality does not determine detection quality** (Spearman +0.41). The driver is
`min_child_samples` (Spearman −0.60):
 
| min_child_samples | model AUC | Task B (rank, y=1) |
|---|---|---|
| 5 | 0.8465 | 0.9780 |
| 20 | 0.8453 | 0.9755 |
| 50 | 0.8456 | 0.9746 |
| 200 | 0.8461 | 0.9462 |
| 500 | 0.8450 | **0.8896** |
 
Discrimination is flat across the range; detection falls 9 points. A leaf with a
500-row floor cannot resolve `p̂` far below prevalence, and the harmful rows live in
exactly that tail. **The score depends on the holder model's resolution in the
low-probability tail, not its discrimination.**
 
## A6 — Detector suite, tuned model + rank construction (`rerun2.py`)
 
n=30 000, 3 seeds, single-core budget. Clean rows carry real labels at natural
prevalence; corrupted rows are prevalence-matched.
 
| detector | Task A off-vs-clean | Task B pooled | **Task B y=1 only** |
|---|---|---|---|
| ORACLE `\|y−p*\|` | 0.7225 ± 0.0064 | 0.5916 ± 0.0052 | **0.9974 ± 0.0018** |
| FITTED, leaf-constrained | 0.6780 ± 0.0126 | 0.5674 ± 0.0051 | **0.9871 ± 0.0024** |
| FITTED, val-selected | 0.6708 ± 0.0436 | 0.5683 ± 0.0142 | 0.9851 ± 0.0119 |
| FITTED, paper default | 0.6479 ± 0.0085 | 0.5690 ± 0.0034 | 0.9799 ± 0.0038 |
| leaf axis 2 | 0.6273 ± 0.0228 | 0.5571 ± 0.0074 | 0.9608 ± 0.0081 |
| adversarial (default) | **0.9998 ± 0.0001** | 0.5014 ± 0.0063 | 0.6364 ± 0.0309 |
| **adversarial, ORACLE-TUNED** | **0.9998 ± 0.0001** | 0.5044 ± 0.0031 | **0.6230 ± 0.0345** |
| DCR (feat+label) | 0.9717 ± 0.0007 | 0.5270 ± 0.0052 | 0.5476 ± 0.0159 |
| kNN density | 0.9661 ± 0.0013 | 0.5299 ± 0.0077 | 0.5169 ± 0.0170 |
| label alone | 0.5000 | 0.5000 | 0.5000 |
| Isolation Forest | 0.7021 ± 0.0171 | 0.5236 ± 0.0074 | **0.1930 ± 0.0540** |
| α-precision (naive) | 0.7050 ± 0.0046 | 0.5186 ± 0.0074 | **0.1536 ± 0.0127** |
 
**Three findings.**
 
1. **The steelman fails.** Adversarial validation with hyperparameters chosen to
   maximise Task B directly — oracle access no honest baseline gets — moves from
   0.5014 to 0.5044 pooled and *down* from 0.636 to 0.623 on the positive stratum,
   while holding 0.9998 on Task A. C2 is now as strong as it can be made.
2. **Two baselines are inverted, not blind.** α-precision 0.154 and Isolation Forest
   0.193 on the positive stratum rank harmful rows as *better* than creative ones.
   Anti-correlation is a stronger claim than blindness and follows from A3: harmful
   positives sit at low `p*`, i.e. at unremarkable feature values, which is what a
   support metric rewards. Report signed.
3. **Neither validation AUC nor validation log-loss selects for tail resolution.** In
   an earlier 20-config run both criteria picked the identical model at
   `min_child_samples = 200`; here validation log-loss picked 500 in one of three
   seeds while AUC picked 50. The harmful rows are too small a fraction of validation
   mass to move a global scoring rule. **No standard selection criterion will find
   this property** — it needs an explicit leaf-size constraint or a criterion
   restricted to the low-`p̂` stratum.
Realised tuning benefit here is small (0.9871 vs 0.9799) because this grid's worst
option was reachable but rarely selected. The sensitivity in A5 is real; the *realised*
penalty depends on the grid. Needs ~10 seeds.
 
## A7 — Reproducibility defect in the ARF path
 
`arf_compat.py` documents the arfpy patch (`self.factor_cols[j]` → `.iloc[j]`) but
never applies it — it was hand-edited into the installed package in an earlier session.
**The ARF half of E8, E9, E11 and E12 cannot be rebuilt from the repository.** Fixed by
`arf_patch.py`, which applies both patches idempotently at import. Import it before
`arf_compat`.
 
---
 
## Status after v2
 
**Established.**
- The failure class exists in unmodified generator output, is structured rather than
  symmetric, and is quantified (A4).
- Contradiction location depends on generator quality; this explains E8's
  generator-conditional filtering result (A4 + E8).
- On-manifold contradiction dominates off-manifold contradiction in downstream harm
  (A3).
- Every deployed per-record metric fails on the boundary question, including under a
  steelman, and two are inverted (A6). **C2 stands and is stronger than recorded.**
- The boundary-relative score works: 0.987 fitted against a 0.997 oracle (A6). Near
  definitional — say so.
- Detection quality depends on tail resolution, not discrimination, and no standard
  model-selection criterion finds it (A5, A6). **New result.**
- Leaf geometry is refuted as both instruments (E1, E3). Retain as C5.
- The Gaussian copula tie-handling bug (E7).
**Refuted.**
- Low support as the dangerous quadrant (A3). The 2×2 must be reoriented.
- Axis alignment as ARF's advantage (E9).
- The aleatoric correction (E1).
**Superseded, do not cite.** E3 Task B, E4, E5, E6, E10 Task B, E12.
 
**Open, in priority order.**
1. **Is the generator-quality axis real or a two-family artifact?** Degrade ARF via
   `num_trees` / `min_node_size` and re-measure A4. If contradiction migrates
   on-manifold as ARF degrades, the axis is continuous and the paper's spine holds. If
   a crippled ARF only emits more off-manifold rows while its in-support labels stay
   Bayes-clean, the claim shrinks to a statement about Gaussian copulas. **Decides the
   introduction. Run first.**
2. A3 hardened into the headline figure: both generators, ≥10 seeds, rate sweep,
   `p*`-matching without replacement.
3. A2/A6 at ≥10 seeds and full sample size; E2 re-run on the rank construction.
4. CTGAN as a third point on the quality axis.
5. Recipient tuning so E8's and A3's denominators are honest.
6. Prevalence sweep 0.05–0.50; coupling sweep λ 0→1 (C5's only direct test).
7. TabDDPM (GPU); RF proximity as a second tree-based support method (C5).
8. A written answer to the relabelling objection, not just a table.
 
