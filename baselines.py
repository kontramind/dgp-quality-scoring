"""Faithful reimplementations of the standard per-record synthetic-data metrics.
For publication these must be re-run against the synthcity/SDMetrics implementations."""
import numpy as np, lightgbm as lgb
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def _prep(ref, qry, cols):
    sc = StandardScaler().fit(ref[cols].to_numpy(float))
    return sc.transform(ref[cols].to_numpy(float)), sc.transform(qry[cols].to_numpy(float))

def dcr(ref, qry, cols):
    """Distance to closest record. Lower = closer to real data = 'better quality'."""
    R, Q = _prep(ref, qry, cols)
    nn = NearestNeighbors(n_neighbors=1).fit(R)
    return nn.kneighbors(Q)[0][:, 0]

def knn_density(ref, qry, cols, k=10):
    R, Q = _prep(ref, qry, cols)
    nn = NearestNeighbors(n_neighbors=k).fit(R)
    return nn.kneighbors(Q)[0].mean(axis=1)

def alpha_precision(ref, qry, cols):
    """Per-sample alpha-level (Alaa et al. 2022). Score = smallest alpha-support
    hypersphere containing the point, i.e. its quantile among real radii.
    Low = deep inside the real support."""
    R, Q = _prep(ref, qry, cols)
    c = R.mean(axis=0)
    dR = np.linalg.norm(R - c, axis=1)
    dQ = np.linalg.norm(Q - c, axis=1)
    return np.searchsorted(np.sort(dR), dQ, side="right") / len(dR)

def isolation(ref, qry, cols, seed=0):
    R, Q = _prep(ref, qry, cols)
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(R)
    return -iso.score_samples(Q)

def adversarial(ref, qry, cols, seed=0, folds=4):
    """Discriminator real vs synthetic; out-of-fold P(synthetic) per row."""
    R = ref[cols].to_numpy(float); Q = qry[cols].to_numpy(float)
    X = np.vstack([R, Q]); lab = np.r_[np.zeros(len(R)), np.ones(len(Q))]
    oof = np.zeros(len(X))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, lab):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               verbose=-1, random_state=seed)
        m.fit(X[tr], lab[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof[len(R):]
