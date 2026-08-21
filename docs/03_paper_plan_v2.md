# Paper plan v2
 
**v2 note.** The v1 plan is superseded. Its thesis — that the dangerous synthetic
records are the low-support ones with contradicting labels — was refuted by A3. The
paper survives with the emphasis moved one cell, and gets stronger in the process.
 
---
 
## Working title direction
 
*Where synthetic label error lives: boundary contradiction is not an off-manifold
phenomenon, and per-record metrics are looking in the wrong place.*
 
Avoid "hallucination". Avoid any title built on "off-manifold", which was v1's error.
 
---
 
## The thesis in one paragraph
 
A generator's label error is not symmetric noise around the true risk — for a Gaussian
copula it is structured contradiction at 17× real data's Bayes residual, and its labels
retain 0.587 AUC against the true boundary where real data has 0.853. Three quarters of
that contradiction sits **on-manifold**, in rows every support-based per-record metric
scores as good, and on-manifold contradiction does 2–8× the downstream damage of
off-manifold contradiction at matched prevalence, volume and contradiction magnitude.
Where the error sits depends on generator quality: for ARF the in-support labels are
Bayes-clean and what little contradiction exists is off-manifold. Support-based metrics
are therefore aligned with harm only in the regime where harm is negligible.
 
---
 
## Claim structure
 
**C1. Label-vs-boundary structure determines downstream consequence; support does not.**
At matched prevalence, volume and contradiction magnitude, on-manifold contradiction
costs a recipient 0.236 LGBM AUC against 0.034 for off-manifold contradiction, and
coherent labels help identically in both (+0.0298 vs +0.0297).
*Evidence:* E11 (dose response, controls at zero), A3 (three constructions of
increasing rigour).
*Load-bearing.* Everything else is interesting only if this holds.
 
**C2. Every deployed per-record metric fails on the boundary question, and two are
inverted.** Adversarial validation holds 0.9998 on off-manifold detection and 0.62 on
the boundary **even when its hyperparameters are chosen to maximise the boundary task
directly** — oracle access no honest baseline gets. α-precision (0.154) and Isolation
Forest (0.193) rank harmful rows as better than creative ones.
*Evidence:* A6, E10.
*This is the paper's strongest claim.* The steelman is what makes it unattackable.
 
**C3. A boundary-relative score works, and it is close to definitional.** Fitted
`|y − p̂|` reaches 0.987 against a 0.997 oracle on the positive-label stratum. Harm is
defined by disagreement with `p*` and detected by disagreement with an estimate of
`p*`. Say so before a reviewer does. Its role is to show the question is answerable,
which is what makes C2 damning rather than merely curious.
*Evidence:* A6, A5.
 
**C4. Where the error lives depends on generator quality — which explains why filtering
helps a poor generator and does nothing to a good one.** Copula: 77.5% of excess
contradiction on-manifold, axis-2 filtering gains +0.122 LGBM. ARF: on-manifold excess
0.0022 against real data's own 0.0020, and no intervention beats no-filter.
*Evidence:* A4 + E8.
*Contingent on the open question below.*
 
**C5. Detection quality depends on the holder model's resolution in the low-probability
tail, not its discrimination — and no standard model-selection criterion finds it.**
`min_child_samples` moves Task B by 9 points while holdout AUC stays flat at 0.845.
Validation AUC and validation log-loss select the same tail-degraded model.
*Evidence:* A5, A6.
*New in v2, and the most directly practitioner-facing result in the paper.*
 
**C6 (methodological). A predictive model's partition is not a general-purpose support
estimate.** Leaf support 0.716 vs DCR 0.979 on off-manifold detection while adversarial
validation — also a LightGBM — reaches 0.9996. A tree splits only where splitting
reduces loss on its own objective.
*Evidence:* E3 (Task A figures survive supersession).
*Placement:* after the main result, as the justification for a label residual over leaf
geometry.
 
---
 
## The open question that decides the introduction
 
The generator-quality axis rests on two points from two model families. A Gaussian
copula couples the label to the features through a single latent correlation vector, so
its 0.587 may be a **capacity limit of the family** rather than a position on a
continuum ARF sits higher on.
 
**Test:** degrade ARF deliberately via `num_trees` and `min_node_size` and re-measure
A4's excess and its on/off split.
- Contradiction migrates on-manifold as ARF degrades → the axis is continuous, C4 is a
  general claim, write the introduction around it.
- A crippled ARF only emits more off-manifold rows while its in-support labels stay
  Bayes-clean → the axis is about model families, and the paper shrinks to "Gaussian
  copula labels are badly miscalibrated against the boundary and every per-record
  metric is blind to it." Still true, still TMLR, smaller claim.
**Run this before writing anything.**
 
---
 
## Why a designed DGP (unchanged)
 
`p*(x)` cannot exist in observational data, so without it no row can be graded and the
failure class cannot be demonstrated at all. Ground truth grades scores; it never
computes them. Every score reported as practitioner-available comes from fitted models.
 
Scope sentence: *we do not validate on observational data, where the reference boundary
must be estimated rather than known; we quantify that cost (~2 AUROC points with a
tuned holder model) but do not verify it transfers.*
 
---
 
## Objections and where they are answered
 
1. **"Just use a better generator."** A4 answers it quantitatively: a better generator
   buys 0.0346 → 0.0047 of excess contradiction. E9 already refuted the axis-alignment
   explanation for ARF's advantage, so this is not hand-waving about representation.
2. **"Just relabel."** Relabelling with `p̂` matches or beats every filter on both
   generators and reaches the real-data ceiling (E8). Near-tautological given the
   definition of harm — overwrite the labels and the harmful cell is empty by
   construction. The defensible arguments are that relabelling is task-specific
   (destroys the release for anything but the declared task) and turns the release into
   a distillation of the holder's model. **These must be argued in prose, not implied
   by a table.**
3. **"Your score just recovers `p*`."** Concede it (C3) and redirect to C2 and C5.
4. **"Does this hold on real data?"** No answer by deliberate scope choice. State it in
   the abstract, not the limitations.
---
 
## Remaining work
 
**Blocking, in order:**
0. The ARF-degradation test above.
1. A3 hardened: both generators, ≥10 seeds, rate sweep, `p*`-matching without
   replacement.
2. A2/A6 at ≥10 seeds and full sample size; E2 re-run on the rank construction.
3. CTGAN as a third point on the quality axis.
4. Recipient tuning so E8's and A3's denominators are honest.
**Strengthening:**
5. Prevalence sweep 0.05–0.50 — everything is conditional on 10%, including the
   positive-stratum framing.
6. Coupling sweep λ 0→1 — C5/C6's only direct mechanism test. Use the feat+label DCR
   variant; feature-only is invariant across λ by construction.
7. Robustness of the on/off split against an exact support criterion rather than the
   declared rule list, which is a lower bound on true off-manifold-ness.
8. RF proximity as a second tree-based support method.
9. TabDDPM (GPU).
 
**Freeze the definitions section before any of the above.** Harm has been
operationalised three times in this project — `1[p* < 0.5]`, rank at matched
prevalence, and now the support-axis question — and each redefinition invalidated a
block of experiments. The code has been fine; the definitions have not.
 
---
 
## Venue
 
**TMLR — primary.** Judges claims against evidence. The negative results (C2's
steelman, C6, the inverted baselines) strengthen rather than weaken. Unlimited length
absorbs the sweeps.
 
**NeurIPS Datasets & Benchmarks — parallel, if the DGP is packaged.** Requires a
pinned environment and the `arf_patch.py` fix; A7 showed the ARF path was not
reproducible from the repository at all.
 
**ICLR / NeurIPS main track — no.** They gate on significance and the transfer question
has no answer by design.
 
**Fallbacks:** ACM JDIQ, ACM TKDD. Both more likely to demand a real dataset.
 
**arXiv once the ARF-degradation test resolves and A3 is hardened.**
 
---
 
## Honest assessment
 
A solid measurement paper with a genuine negative result and a mechanism. Not a paper
anyone will build on immediately: on a good generator none of this changes a decision,
and the practical urgency claim cannot be made. Lead with the phenomenon, not the
score.
 
