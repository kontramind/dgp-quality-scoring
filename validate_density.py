import numpy as np
import pandas as pd
from scipy.stats import norm
from dgp import DGPConfig, generate, solve_binding_rates, evaluate_rules
from density import log_density_X, _clip_atoms

rng = np.random.default_rng(0)

print("=" * 70)
print("CHECK 1: per-conditional Monte Carlo integration (each should be ~1.0)")
print("=" * 70)


def mc_integral_clip(mu, sd, lo, hi, n=4_000_000, seed=0):
    """MC-integrate the clip(mu+N(0,sd),lo,hi) density: uniform proposal on the
    interior for the continuous part, plus the two atom masses in closed form
    (the atom masses themselves are exact tail probabilities, not something to
    Monte-Carlo -- what we're checking is that _clip_atoms' interior term,
    integrated, plus those two atoms, sums to 1)."""
    r = np.random.default_rng(seed)
    u = r.uniform(lo, hi, n)
    logpdf = _clip_atoms(mu, sd, lo, hi, u)  # interior branch since u in (lo,hi) a.s.
    interior_integral = np.mean(np.exp(logpdf)) * (hi - lo)
    atom_lo = norm.cdf((lo - mu) / sd)
    atom_hi = 1 - norm.cdf((hi - mu) / sd)
    return interior_integral + atom_lo + atom_hi, interior_integral, atom_lo, atom_hi


# Sample a spread of realistic parent contexts to check the conditional at.
cfg = solve_binding_rates({"R3_glucose_ceil": 0.08, "R4_bp_mandatory": 0.20})
cfg.n_samples = 40_000
X, G, M = generate(cfg)

print("\n-- Cigs_Per_Day | Smoker=1 : clip(N(15,5),1,40) --")
tot, interior, a_lo, a_hi = mc_integral_clip(15.0, 5.0, 1.0, 40.0)
print(f"  interior={interior:.6f}  atom@1={a_lo:.6f}  atom@40={a_hi:.6e}  TOTAL={tot:.6f}")

print("\n-- BMI | Age, Sex : clip(mu_bmi + N(0,3), 15, 50), 5 sampled contexts --")
sample_idx = rng.choice(len(X), 5, replace=False)
for i in sample_idx:
    age_i, sex_i = X.Age.iloc[i], X.Biological_Sex.iloc[i]
    mu = 25 + cfg.b_age_bmi * (age_i - 30) + cfg.b_male_bmi * sex_i
    tot, interior, a_lo, a_hi = mc_integral_clip(mu, cfg.sd_bmi, 15.0, 50.0)
    print(f"  age={age_i:5.1f} sex={sex_i:.0f} mu={mu:6.2f}  "
          f"interior={interior:.6f} atom@15={a_lo:.2e} atom@50={a_hi:.2e}  TOTAL={tot:.6f}")

print("\n-- Systolic_BP | Age,BMI,Glucose,Smoker : clip(mu_sbp + N(0,12), 70, 200), 5 contexts --")
for i in sample_idx:
    age_i, bmi_i = X.Age.iloc[i], X.BMI.iloc[i]
    glu_i, sm_i = X.Fasting_Glucose.iloc[i], X.Current_Smoker.iloc[i]
    hi_band = float(glu_i >= 126.0)
    mu = (cfg.sbp_base + cfg.b_age_sbp * (age_i - 30) + cfg.b_bmi_sbp * (bmi_i - 25)
          + cfg.b_hyperglyc_sbp * hi_band + cfg.b_smoker_sbp * sm_i)
    tot, interior, a_lo, a_hi = mc_integral_clip(mu, cfg.sd_sbp, 70.0, 200.0)
    print(f"  mu={mu:6.2f}  interior={interior:.6f} atom@70={a_lo:.2e} atom@200={a_hi:.2e}  TOTAL={tot:.6f}")

print("\n-- Fasting_Glucose | BMI : unclipped Normal, sanity quad over wide range --")
from scipy.integrate import quad
mu = cfg.glucose_base + cfg.b_bmi_glucose * (X.BMI.iloc[0] - 25)
val, _ = quad(lambda g: norm.pdf(g, mu, cfg.sd_glucose), mu - 15 * cfg.sd_glucose, mu + 15 * cfg.sd_glucose)
print(f"  mu={mu:.2f}  quad integral over +/-15sd = {val:.8f}")

print("\n(Age uniform, Sex/Smoker/Preg/Diabetic-midband/Meds-below Bernoulli, and the")
print(" Diabetic/Meds deterministic branches are point-mass or uniform by construction")
print(" and sum to 1 trivially -- omitted from MC checks.)")

print()
print("=" * 70)
print("CHECK 2: joint log-density on real generated rows -- sanity range")
print("=" * 70)
logp_real = log_density_X(X, cfg)
print(f"n={len(X)}  finite: {np.isfinite(logp_real).sum()}/{len(X)}  "
      f"mean={logp_real.mean():.3f}  sd={logp_real.std():.3f}  "
      f"min={logp_real.min():.3f}  max={logp_real.max():.3f}")
assert np.isfinite(logp_real).all(), "real generated rows must have finite density"

print()
print("=" * 70)
print("CHECK 3: rank check -- real rows vs the worked implausible row")
print("(Age 30, Systolic_BP 190, violates no rule -- R4 only requires meds=1)")
print("=" * 70)

# Build the implausible row: take a real row's other fields, force Age=30, SBP=190,
# and set On_BP_Medication=1 so it satisfies R4 (the only rule that could fire).
base = X.iloc[0].copy()
bad = base.copy()
bad["Age"] = 30.0
bad["Systolic_BP"] = 190.0
bad["On_BP_Medication"] = 1.0
bad_df = pd.DataFrame([bad])

# confirm it is rule-clean
summaries, flags, total = evaluate_rules(bad_df)
assert total == 0, f"constructed row must violate no rule of R1-R6, got {total}"
print("rule violations on constructed row:", total, "(expect 0)")

logp_bad = log_density_X(bad_df, cfg)[0]
pct = (logp_real < logp_bad).mean() * 100
print(f"log p(implausible row) = {logp_bad:.3f}")
print(f"log p(real rows): mean={logp_real.mean():.3f}, "
      f"1st pct={np.percentile(logp_real,1):.3f}, min={logp_real.min():.3f}")
print(f"implausible row sits below {100-pct:.4f}% ... i.e. at percentile {pct:.6f} of the real distribution")
print(f"fraction of real rows with LOWER density than the implausible row: {pct:.4f}%")
