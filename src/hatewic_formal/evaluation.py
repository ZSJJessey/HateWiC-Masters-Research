"""Matched out-of-fold evaluation. No text or annotator details are exported."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

OOF_COLUMNS = {"example_id", "majority_label", "prediction", "probability", "fold", "protocol", "model", "seed"}


def binary_metrics(y_true, y_pred, require_both_classes=True):
    """Macro F1 uses fixed labels [0,1]; unsupported comparisons are explicit."""
    y, p = np.asarray(y_true), np.asarray(y_pred)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty, equally sized one-dimensional labels.")
    if not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Labels and predictions must be binary.")
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    estimable = bool((support > 0).all())
    return {"macro_f1": float(f1.mean()) if estimable or not require_both_classes else None,
            "estimable": estimable, "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(recall.mean()) if estimable else None,
            "per_class": {str(k): {"precision": float(precision[k]), "recall": float(recall[k]),
                                    "f1": float(f1[k]), "support": int(support[k])} for k in [0, 1]},
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(), "n": len(y)}


def validate_oof(predictions, fold_manifest, expected_ids=None):
    """Validate complete pooled OOF coverage against the manifest's test rows.

    Manifest: example_id, protocol, fold (or rotation), split (or partition).
    Every model/seed must have exactly the test assignment of its protocol.
    """
    p, m = predictions.copy(), fold_manifest.copy()
    if OOF_COLUMNS - set(p):
        raise ValueError(f"Missing OOF columns: {sorted(OOF_COLUMNS-set(p))}")
    split = "split" if "split" in m else "partition"
    if "fold" not in m and "rotation" in m:
        m = m.rename(columns={"rotation": "fold"})
    if not {"example_id", "protocol", "fold", split}.issubset(m):
        raise ValueError("Incomplete fold manifest.")
    p["example_id"], m["example_id"] = p.example_id.astype(str), m.example_id.astype(str)
    if p[list(OOF_COLUMNS)].isna().any().any():
        raise ValueError("OOF fields cannot be missing.")
    if not np.isin(p.majority_label, [0, 1]).all() or not np.isin(p.prediction, [0, 1]).all():
        raise ValueError("OOF labels must be binary.")
    prob = p.probability.to_numpy(dtype=float)
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Invalid probabilities.")
    if not np.array_equal(p.prediction.to_numpy(), (prob >= .5).astype(int)):
        raise ValueError("Predictions must use the fixed probability threshold 0.5.")
    if p.groupby("example_id").majority_label.nunique().gt(1).any():
        raise ValueError("True labels differ across runs.")
    if "majority_label" in m:
        label_ref = m[["example_id", "majority_label"]].drop_duplicates()
        if label_ref.example_id.duplicated().any():
            raise ValueError("Manifest labels differ across partitions.")
        joined = p.merge(label_ref, on="example_id", how="left", suffixes=("", "_manifest"), validate="many_to_one")
        if not joined.majority_label.eq(joined.majority_label_manifest).all():
            raise ValueError("OOF truth differs from frozen manifest labels.")
    tests = m.loc[m[split].eq("test"), ["example_id", "protocol", "fold"]]
    if tests.duplicated(["example_id", "protocol"]).any():
        raise ValueError("Each manifest ID must be tested exactly once per protocol.")
    if not len(tests) or not len(p):
        raise ValueError("Empty predictions or manifest.")
    for (protocol, model, seed), run in p.groupby(["protocol", "model", "seed"], dropna=False):
        ref = tests.loc[tests.protocol.eq(protocol)].set_index("example_id").fold.sort_index()
        if run.example_id.duplicated().any():
            raise ValueError("Duplicate OOF prediction for a sample.")
        actual = run.set_index("example_id").fold.sort_index()
        if not actual.index.equals(ref.index) or not np.array_equal(actual.to_numpy(), ref.to_numpy()):
            raise ValueError(f"OOF IDs or folds do not match manifest: {protocol}/{model}/{seed}")
        if expected_ids is not None and set(actual.index) != set(map(str, expected_ids)):
            raise ValueError("OOF coverage differs from expected cohort.")
    return {"validated": True, "runs": int(p.groupby(["protocol", "model", "seed"]).ngroups), "rows": len(p)}


def align_paired_predictions(frame, model_a, model_b, protocol):
    """Return (truth, A, B, IDs, seeds), with predictions shaped seeds × IDs."""
    f = frame.copy()
    f["example_id"] = f.example_id.astype(str)
    blocks = []
    for model in [model_a, model_b]:
        b = f.loc[f.protocol.eq(protocol) & f.model.eq(model)]
        if b.empty or b.duplicated(["seed", "example_id"]).any():
            raise ValueError("Missing model or duplicate paired predictions.")
        blocks.append(b)
    a, b = blocks
    seeds = sorted(a.seed.unique())
    if set(seeds) != set(b.seed.unique()):
        raise ValueError("Paired models require identical seed sets.")
    ids = sorted(a.example_id.unique())
    key = pd.MultiIndex.from_product([seeds, ids], names=["seed", "example_id"])
    ordered = []
    for block in blocks:
        indexed = block.set_index(["seed", "example_id"])
        if set(indexed.index) != set(key):
            raise ValueError("Paired runs must cover identical IDs for every seed.")
        ordered.append(indexed.reindex(key))
    truth = ordered[0].majority_label.to_numpy().reshape(len(seeds), -1)
    other = ordered[1].majority_label.to_numpy().reshape(len(seeds), -1)
    if not np.array_equal(truth, other) or not np.all(truth == truth[0]):
        raise ValueError("Paired true labels are inconsistent.")
    return truth[0], *(x.prediction.to_numpy().reshape(len(seeds), -1) for x in ordered), ids, seeds


def validate_score_metadata(metadata, ids):
    required = {"example_id", "sense_heterogeneous", "lexical_cluster_id"}
    if required - set(metadata):
        raise ValueError("Incomplete score-only metadata.")
    m = metadata.copy()
    m["example_id"] = m.example_id.astype(str)
    if m.example_id.duplicated().any() or set(m.example_id) != set(map(str, ids)):
        raise ValueError("Score metadata must match IDs one-to-one.")
    if m[list(required)].isna().any().any() or not np.isin(m.sense_heterogeneous, [False, True, 0, 1]).all():
        raise ValueError("Invalid score metadata.")
    return m.set_index("example_id").loc[list(map(str, ids))]


def summarize_oof(predictions, score_metadata=None):
    """Pool test predictions across folds; never interpret folds as replicates."""
    rows = []
    for keys, run in predictions.groupby(["protocol", "model", "seed"]):
        if run.example_id.duplicated().any():
            raise ValueError("Duplicate sample in OOF run.")
        groups = {"overall": np.ones(len(run), dtype=bool)}
        if score_metadata is not None:
            meta = validate_score_metadata(score_metadata, run.example_id)
            het = meta.sense_heterogeneous.to_numpy(dtype=bool)
            groups.update(homogeneous=~het, heterogeneous=het)
        for subgroup, mask in groups.items():
            metrics = binary_metrics(run.majority_label.to_numpy()[mask], run.prediction.to_numpy()[mask]) if mask.any() else {"macro_f1": None, "estimable": False, "n": 0}
            rows.append(dict(zip(["protocol", "model", "seed"], keys), subgroup=subgroup, **metrics))
    return pd.DataFrame(rows)


def ro3_contrasts(predictions, score_metadata, full="full", no_gloss="no_gloss", random_protocol="random", term_protocol="term_disjoint"):
    """Point contrasts of paired seed means; inferential tests live in statistics."""
    from .statistics import paired_statistic, generalization_statistic
    rows, aligned = [], {}
    for protocol in [random_protocol, term_protocol]:
        y, a, b, ids, seeds = align_paired_predictions(predictions, no_gloss, full, protocol)
        meta = validate_score_metadata(score_metadata, ids)
        het = meta.sense_heterogeneous.to_numpy(dtype=bool)
        for metric in ["macro_f1", "heterogeneity_gap_reduction"]:
            rows.append({"protocol": protocol, "contrast": metric, "estimate": paired_statistic(y, a, b, statistic=metric, subgroup=het), "seeds": seeds})
        aligned[protocol] = (y, a, b, ids, seeds)
    r, t = aligned[random_protocol], aligned[term_protocol]
    if r[3:] != t[3:] or not np.array_equal(r[0], t[0]):
        raise ValueError("Generalization contrast needs the same IDs and seeds across protocols.")
    rows.append({"protocol": "cross_protocol", "contrast": "generalization_gap_reduction", "estimate": generalization_statistic(r[0], r[1], t[1], r[2], t[2]), "seeds": r[4]})
    return pd.DataFrame(rows)


def primary_classification_tests(predictions, score_metadata, random_protocol="random", term_protocol="term_disjoint", **resampling):
    """Seven predeclared RO2/RO3 tests, to combine with three RO1 tests.

    Five-seed core analyses and single-seed all-configuration analyses must be
    supplied separately. Bootstrap repeats sample clusters, not seed identities.
    """
    from .statistics import paired_cluster_inference, generalization_gap_inference
    rows, gloss_pairs = [], {}
    for protocol in [random_protocol, term_protocol]:
        for a_model, b_model, ro in [("bert_context", "hatebert_context", "ro2"), ("no_gloss", "full", "ro3")]:
            y, a, b, ids, seeds = align_paired_predictions(predictions, a_model, b_model, protocol)
            meta = validate_score_metadata(score_metadata, ids)
            clusters = meta.lexical_cluster_id.to_numpy()
            result = paired_cluster_inference(y, a, b, clusters, **resampling)
            rows.append({"test_id": f"{ro}_overall_{protocol}", "protocol": protocol, "baseline": a_model, "candidate": b_model, **result})
            if ro == "ro3":
                result = paired_cluster_inference(y, a, b, clusters, statistic="heterogeneity_gap_reduction", subgroup=meta.sense_heterogeneous.to_numpy(dtype=bool), **resampling)
                rows.append({"test_id": f"ro3_heterogeneity_gap_{protocol}", "protocol": protocol, "baseline": a_model, "candidate": b_model, **result})
                gloss_pairs[protocol] = y, a, b, ids, seeds, clusters
    r, t = gloss_pairs[random_protocol], gloss_pairs[term_protocol]
    if r[3:5] != t[3:5] or not np.array_equal(r[0], t[0]) or not np.array_equal(r[5], t[5]):
        raise ValueError("Cross-protocol inference needs matched IDs, labels, seeds and clusters.")
    result = generalization_gap_inference(r[0], r[1], t[1], r[2], t[2], r[5], **resampling)
    rows.append({"test_id": "ro3_generalization_gap", "protocol": "cross_protocol", "baseline": "no_gloss", "candidate": "full", **result})
    return pd.DataFrame(rows)
