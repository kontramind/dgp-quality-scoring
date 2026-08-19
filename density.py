"""
Closed-form per-row log-density of X under the DGP's structural causal model.

Follows the exact ancestral order in dgp.py::generate. Ten conditionals, chained by
the product rule:

    p(x) = p(Age) p(Sex) p(Smoker) p(Preg|Sex,Age) p(Cigs|Smoker) p(BMI|Age,Sex)
           p(Glucose|BMI) p(Diabetic|Glucose) p(SBP|Age,BMI,Glucose,Smoker) p(Meds|SBP,Age)

This is a density w.r.t. a fixed mixed reference measure per variable: Lebesgue on the
continuous interior, counting (point) mass at each clip/determinism atom. Each factor
is individually normalised by construction (uniform, Bernoulli, clipped-Gaussian with
exact tail-mass atoms, or a deterministic point mass) so the product is a valid density
w.r.t. that fixed reference measure — this is what the Monte Carlo / quadrature checks
in validate_density.py confirm, since a term that were mis-normalised would fail those
checks even though "each factor is individually normalised by construction" is true of
the model, not automatically true of an implementation of it.

Does NOT include Heart_Disease_Risk. The label is a separate downstream variable
(Y | X via p_true(z)); "on-manifold" per item 2 is a question about X only.
"""
import numpy as np
from scipy.stats import norm

EPS = 1e-12


def _clip_atoms(mu, sd, lo, hi, x):
    """log-density for x = clip(mu + N(0,sd), lo, hi), evaluated at observed x.

    Returns array of log-densities: interior points get the raw Normal logpdf
    (clipping does not distort the density on the interior, only redistributes tail
    mass onto the two boundary atoms); x==lo or x==hi get the tail-mass atom's log
    probability instead of a density.
    """
    x = np.asarray(x, dtype=float)
    z_lo = (lo - mu) / sd
    z_hi = (hi - mu) / sd
    atom_lo = norm.cdf(z_lo)          # P(raw <= lo)
    atom_hi = 1.0 - norm.cdf(z_hi)    # P(raw >= hi)
    interior = norm.logpdf(x, mu, sd)
    at_lo = np.isclose(x, lo, atol=1e-6)
    at_hi = np.isclose(x, hi, atol=1e-6)
    # clip(mu+N(0,sd),lo,hi) can NEVER land strictly outside [lo,hi]. A value observed
    # there (e.g. a row assembled from mismatched marginals, as GaussianCopulaDT can
    # produce) is off the support entirely -- not a small raw-Normal density, -inf.
    strictly_inside = (x > lo) & (x < hi) & ~at_lo & ~at_hi
    out = np.full_like(x, -np.inf)
    out = np.where(strictly_inside, interior, out)
    out = np.where(at_lo, np.log(np.maximum(atom_lo, EPS)), out)
    out = np.where(at_hi, np.log(np.maximum(atom_hi, EPS)), out)
    return out


def log_density_X(df, cfg):
    """Per-row log p(x) in nats. -inf where the row is structurally impossible
    under the SCM (e.g. a pregnant male) or contradicts a deterministic flag rule
    (e.g. glucose>=126 with Diabetic=0). This subsumes rule violation as a special
    (infinite) case of low density, which is exactly item 2's unification question.
    """
    age = df["Age"].to_numpy(float)
    sex = df["Biological_Sex"].to_numpy(float)
    smoker = df["Current_Smoker"].to_numpy(float)
    preg = df["Is_Pregnant"].to_numpy(float)
    cigs = df["Cigs_Per_Day"].to_numpy(float)
    bmi = df["BMI"].to_numpy(float)
    glucose = df["Fasting_Glucose"].to_numpy(float)
    diabetic = df["Diabetic"].to_numpy(float)
    sbp = df["Systolic_BP"].to_numpy(float)
    meds = df["On_BP_Medication"].to_numpy(float)
    n = len(df)
    logp = np.zeros(n)

    # 1. Age ~ Uniform(age_range)
    lo_a, hi_a = cfg.age_range
    in_range = (age >= lo_a) & (age <= hi_a)
    logp = logp + np.where(in_range, -np.log(hi_a - lo_a), -np.inf)

    # 2. Sex ~ Bernoulli(p_male)
    logp = logp + np.where(sex == 1, np.log(cfg.p_male), np.log(1 - cfg.p_male))

    # 3. Smoker ~ Bernoulli(p_smoker)
    logp = logp + np.where(smoker == 1, np.log(cfg.p_smoker), np.log(1 - cfg.p_smoker))

    # 4. Is_Pregnant | Sex, Age  (deterministic 0 off support)
    eligible = (sex == 0) & (age <= cfg.preg_age_max)
    lp_preg = np.where(
        eligible,
        np.where(preg == 1, np.log(cfg.p_pregnant), np.log(1 - cfg.p_pregnant)),
        np.where(preg == 0, 0.0, -np.inf))
    logp = logp + lp_preg

    # 5. Cigs_Per_Day | Current_Smoker
    lp_cigs_smoker = _clip_atoms(15.0, 5.0, 1.0, 40.0, cigs)
    lp_cigs = np.where(
        smoker == 1, lp_cigs_smoker,
        np.where(cigs == 0.0, 0.0, -np.inf))
    logp = logp + lp_cigs

    # 6. BMI | Age, Sex
    mu_bmi = 25 + cfg.b_age_bmi * (age - 30) + cfg.b_male_bmi * sex
    lp_bmi = _clip_atoms(mu_bmi, cfg.sd_bmi, 15.0, 50.0, bmi)
    logp = logp + lp_bmi

    # 7. Fasting_Glucose | BMI  (not clipped -> plain Gaussian)
    mu_glu = cfg.glucose_base + cfg.b_bmi_glucose * (bmi - 25)
    lp_glu = norm.logpdf(glucose, mu_glu, cfg.sd_glucose)
    logp = logp + lp_glu

    # 8. Diabetic | Fasting_Glucose  (deterministic outside the midband)
    hi_band = glucose >= 126.0
    lo_band = glucose < 100.0
    mid_band = ~hi_band & ~lo_band
    lp_diab = np.where(
        hi_band, np.where(diabetic == 1, 0.0, -np.inf),
        np.where(lo_band, np.where(diabetic == 0, 0.0, -np.inf),
                 np.where(diabetic == 1, np.log(cfg.p_diabetic_midband),
                          np.log(1 - cfg.p_diabetic_midband))))
    logp = logp + np.where(mid_band | hi_band | lo_band, lp_diab, lp_diab)  # exhaustive

    # 9. Systolic_BP | Age, BMI, Fasting_Glucose, Current_Smoker
    mu_sbp = (cfg.sbp_base + cfg.b_age_sbp * (age - 30) + cfg.b_bmi_sbp * (bmi - 25)
              + cfg.b_hyperglyc_sbp * hi_band.astype(float) + cfg.b_smoker_sbp * smoker)
    lp_sbp = _clip_atoms(mu_sbp, cfg.sd_sbp, 70.0, 200.0, sbp)
    logp = logp + lp_sbp

    # 10. On_BP_Medication | Systolic_BP, Age  (deterministic above 140)
    p_meds_below = np.clip(cfg.meds_age_scale * (age - 30) / 100.0, 0.0, 1.0)
    above = sbp > 140.0
    lp_meds = np.where(
        above, np.where(meds == 1, 0.0, -np.inf),
        np.where(meds == 1, np.log(np.maximum(p_meds_below, EPS)),
                 np.log(np.maximum(1 - p_meds_below, EPS))))
    logp = logp + lp_meds

    return logp
