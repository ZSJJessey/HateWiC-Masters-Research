"""Paired cluster inference conditional on already fitted models.

The same cluster draw/swap is applied to every seed and every protocol in a
contrast. These intervals do not quantify model-retraining uncertainty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONDITIONAL_SCOPE = "Conditional on fitted models; excludes retraining uncertainty. Folds are not independent replicates."


def _inputs(y, *predictions):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected nonempty binary truth.")
    arrays = [np.atleast_2d(np.asarray(p)) for p in predictions]
    if any(not len(p) or p.shape[1] != len(y) or p.shape != arrays[0].shape or not np.isin(p, [0, 1]).all() for p in arrays):
        raise ValueError("Predictions must share seed-by-sample shape and binary labels.")
    return y, arrays


def _macro(y, p, weights):
    if any(weights[y == label].sum() == 0 for label in [0, 1]):
        return None
    scores = []
    for row in p:
        f = []
        for label in [0, 1]:
            tp = weights[(y == label) & (row == label)].sum()
            fp = weights[(y != label) & (row == label)].sum()
            fn = weights[(y == label) & (row != label)].sum()
            f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        scores.append(np.mean(f))
    return float(np.mean(scores))


def paired_statistic(y_true, pred_a, pred_b, statistic="macro_f1", subgroup=None, weights=None):
    """Positive = B improves over A; heterogeneity statistic = gap(A)-gap(B)."""
    y, (a, b) = _inputs(y_true, pred_a, pred_b)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Invalid sample weights.")
    if statistic == "macro_f1":
        sa, sb = _macro(y, a, w), _macro(y, b, w)
        return None if sa is None or sb is None else sb - sa
    if statistic != "heterogeneity_gap_reduction":
        raise ValueError("Unknown paired statistic.")
    h = np.asarray(subgroup)
    if h.shape != y.shape or not np.isin(h, [0, 1]).all():
        raise ValueError("A binary heterogeneity mask is required.")
    h = h.astype(bool)
    vals = [_macro(y, p, w * mask) for p in [a, b] for mask in [~h, h]]
    return None if any(v is None for v in vals) else (vals[0] - vals[1]) - (vals[2] - vals[3])


def generalization_statistic(y_true, a_random, a_term, b_random, b_term, weights=None):
    """Gap reduction: (A random - A term) - (B random - B term)."""
    y, arrays = _inputs(y_true, a_random, a_term, b_random, b_term)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Invalid sample weights.")
    vals = [_macro(y, p, w) for p in arrays]
    return None if any(v is None for v in vals) else (vals[0]-vals[1])-(vals[2]-vals[3])


def _clusters(clusters, n):
    c = np.asarray(clusters)
    if c.ndim != 1 or len(c) != n or pd.isna(c).any():
        raise ValueError("Each sample needs a nonmissing lexical/duplicate cluster.")
    codes, unique = pd.factorize(c, sort=False)
    return codes, len(unique)


def _report(observed, boot, null, n_clusters, seeds, n_bootstrap, n_permutations):
    valid = np.asarray([x for x in boot if x is not None and np.isfinite(x)])
    null_valid = np.asarray([x for x in null if x is not None and np.isfinite(x)])
    estimable = observed is not None and n_clusters >= 2
    # Missing-class resamples are counted explicitly. Insufficient supported
    # draws produce descriptive results rather than a significance claim.
    infer = estimable and len(valid) >= .9*n_bootstrap and len(null_valid) == n_permutations
    return {"estimate": observed, "estimable": estimable, "inferential": bool(infer),
            "ci_low": float(np.quantile(valid, .025)) if infer else None,
            "ci_high": float(np.quantile(valid, .975)) if infer else None,
            "p_value": float((1 + (np.abs(null_valid) >= abs(observed)-1e-15).sum())/(1+len(null_valid))) if infer else None,
            "valid_bootstrap": len(valid), "requested_bootstrap": n_bootstrap,
            "valid_permutations": len(null_valid), "requested_permutations": n_permutations,
            "n_clusters": n_clusters, "n_seeds": seeds, "scope": CONDITIONAL_SCOPE}


def paired_cluster_inference(y_true, pred_a, pred_b, clusters, statistic="macro_f1", subgroup=None,
                             n_bootstrap=10000, n_permutations=10000, seed=13):
    y, (a, b) = _inputs(y_true, pred_a, pred_b)
    codes, k = _clusters(clusters, len(y))
    if n_bootstrap < 1 or n_permutations < 1:
        raise ValueError("Positive resampling counts required.")
    fn = lambda aa, bb, w=None: paired_statistic(y, aa, bb, statistic, subgroup, w)
    observed, boot, null = fn(a, b), [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        weights = rng.multinomial(k, np.full(k, 1/k))[codes]
        boot.append(fn(a, b, weights))
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, k).astype(bool)[codes][None, :]
        null.append(fn(np.where(swap, b, a), np.where(swap, a, b)))
    return _report(observed, boot, null, k, len(a), n_bootstrap, n_permutations)


def generalization_gap_inference(y_true, a_random, a_term, b_random, b_term, clusters,
                                  n_bootstrap=10000, n_permutations=10000, seed=13):
    y, (ar, at, br, bt) = _inputs(y_true, a_random, a_term, b_random, b_term)
    codes, k = _clusters(clusters, len(y))
    if n_bootstrap < 1 or n_permutations < 1:
        raise ValueError("Positive resampling counts required.")
    fn = lambda aa, ab, ba, bb, w=None: generalization_statistic(y, aa, ab, ba, bb, w)
    observed, boot, null = fn(ar, at, br, bt), [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        boot.append(fn(ar, at, br, bt, rng.multinomial(k, np.full(k, 1/k))[codes]))
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, k).astype(bool)[codes][None, :]
        null.append(fn(np.where(swap, br, ar), np.where(swap, bt, at), np.where(swap, ar, br), np.where(swap, at, bt)))
    return _report(observed, boot, null, k, len(ar), n_bootstrap, n_permutations)


def paired_cluster_mean_inference(differences, clusters, n_bootstrap=10000, n_permutations=10000, seed=13):
    """Equal-weight group differences, clustered by lexical/duplicate component."""
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError("Finite group differences required.")
    if n_bootstrap < 1 or n_permutations < 1:
        raise ValueError("Positive resampling counts required.")
    codes, k = _clusters(clusters, len(d))
    rng = np.random.default_rng(seed)
    boot, null = [], []
    for _ in range(n_bootstrap):
        w = rng.multinomial(k, np.full(k, 1/k))[codes]
        boot.append(float(np.average(d, weights=w)))
    for _ in range(n_permutations):
        signs = (2*rng.integers(0, 2, k)-1)[codes]
        null.append(float(np.mean(d*signs)))
    return _report(float(d.mean()), boot, null, k, 1, n_bootstrap, n_permutations)


def holm_family(results, expected_tests=10):
    """Missing tests block claims; registered nonestimable tests retain p=1.

    Their original p_value remains missing. p_for_holm=1 is a conservative
    non-rejection marker, never a computed inferential p-value.
    """
    rows = pd.DataFrame(results).copy()
    if "test_id" not in rows or "p_value" not in rows or rows.test_id.duplicated().any():
        raise ValueError("Unique test_id and p_value required.")
    numeric = pd.to_numeric(rows.p_value, errors="coerce")
    if ((numeric.dropna() < 0) | (numeric.dropna() > 1)).any() or len(rows) > expected_tests:
        raise ValueError("Invalid primary test family.")
    unavailable = pd.Series(False, index=rows.index)
    for column in ["estimable", "inferential"]:
        if column in rows:
            unavailable |= rows[column].eq(False)
    numeric = numeric.mask(unavailable, 1.)
    rows["p_for_holm"] = numeric
    rows["non_rejection_marker"] = unavailable
    complete = len(rows) == expected_tests and numeric.notna().all()
    rows["family_complete"] = bool(complete)
    rows["p_holm"] = np.nan
    rows["significant_0_05"] = None
    if complete:
        order = np.argsort(numeric.to_numpy(), kind="stable")
        adjusted = np.minimum(1., np.maximum.accumulate(numeric.to_numpy()[order] * np.arange(expected_tests, 0, -1)))
        rows.loc[rows.index[order], "p_holm"] = adjusted
        rows["significant_0_05"] = rows.p_holm.le(.05)
    return rows
