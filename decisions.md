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
write "unresolved" rather than "noise" or "not a finding". It is cheap to settle later —
a coupling sweep that supplies more DGP seeds would resolve it as a by-product, with no
run commissioned specifically for it.
