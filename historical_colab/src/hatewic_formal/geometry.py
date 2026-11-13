"""RO1 geometry on L2-normalized frozen target representations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

METRICS = ["cosine_separation", "silhouette_cosine", "davies_bouldin"]


def geometry_scores(vectors, labels):
    x, y = np.asarray(vectors, dtype=float), np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or y.ndim != 1 or not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite sample-by-feature vectors and aligned binary labels.")
    norms = np.linalg.norm(x, axis=1)
    if (norms == 0).any():
        raise ValueError("Zero vectors cannot be cosine-normalized.")
    counts = [int((y == label).sum()) for label in [0, 1]]
    result = {"n": len(y), "n0": counts[0], "n1": counts[1], "estimable": min(counts) >= 2}
    if min(counts) < 2:
        return {**result, **{m: None for m in METRICS}}
    x = x / norms[:, None]
    # The pooled within-class pair mean weights every within-class pair equally.
    sums, pairs = 0., 0
    for label in [0, 1]:
        block = x[y == label]
        sums += (np.square(block.sum(axis=0)).sum() - len(block))/2
        pairs += len(block)*(len(block)-1)//2
    across = float(x[y == 0].mean(axis=0) @ x[y == 1].mean(axis=0))
    result.update(cosine_separation=float(sums/pairs-across),
                  silhouette_cosine=float(silhouette_score(x, y, metric="cosine")),
                  davies_bouldin=float(davies_bouldin_score(x, y)))
    return result


def compare_geometry(vectors_by_backbone, samples, term_pos_col="term_pos_key"):
    """Matched rows must correspond to samples in exactly their given order.

    The caller supplies representation arrays extracted against the same frozen
    ID manifest. Group keys encode normalized term + POS (not local sense_id).
    Returns per-group metrics, equal-weight summaries and supplementary global
    metrics. No raw sentence fields are copied.
    """
    required = {"example_id", "majority_label", term_pos_col}
    if required - set(samples) or samples.example_id.astype(str).duplicated().any():
        raise ValueError("Unique sample IDs, labels and term/POS keys required.")
    if samples[list(required)].isna().any().any() or not vectors_by_backbone:
        raise ValueError("Missing geometry inputs.")
    rows, global_rows = [], []
    positions = samples.reset_index(drop=True).groupby(term_pos_col, sort=True).indices
    for backbone, vectors in vectors_by_backbone.items():
        x = np.asarray(vectors)
        if x.ndim != 2 or len(x) != len(samples):
            raise ValueError("Representations do not match ID manifest length.")
        global_rows.append({"backbone": backbone, "analysis": "global_supplementary", **geometry_scores(x, samples.majority_label.to_numpy())})
        for key, indices in positions.items():
            scores = geometry_scores(x[indices], samples.majority_label.to_numpy()[indices])
            rows.append({"group_key": key, "backbone": backbone, **scores})
    groups = pd.DataFrame(rows)
    summary = []
    for backbone, block in groups.groupby("backbone"):
        supported = block.loc[block.estimable]
        summary.append({"backbone": backbone, "analysis": "equal_weight_term_pos", "supported_groups": len(supported),
                        "total_groups": len(block), **{m: float(supported[m].mean()) if len(supported) else None for m in METRICS}})
    return groups, pd.DataFrame(summary), pd.DataFrame(global_rows)


def geometry_inference(group_scores, group_clusters, baseline="bert", candidate="hatebert", **resampling):
    """Paired supported group metrics; positive differences denote improvement.

    group_clusters maps group_key to the union lexical/duplicate component.
    Lexical grouping excludes POS, so related POS groups are resampled together.
    """
    from .statistics import paired_cluster_mean_inference
    if group_scores.duplicated(["group_key", "backbone"]).any():
        raise ValueError("Duplicate geometry group/backbone.")
    a = group_scores.loc[group_scores.backbone.eq(baseline)].set_index("group_key")
    b = group_scores.loc[group_scores.backbone.eq(candidate)].set_index("group_key")
    if set(a.index) != set(b.index):
        raise ValueError("Backbones need the same geometry groups.")
    b = b.reindex(a.index)
    if not np.array_equal(a.estimable, b.estimable):
        raise ValueError("Backbone support differs.")
    ids = a.index[a.estimable].tolist()
    if any(key not in group_clusters for key in ids):
        raise ValueError("Missing geometry group cluster.")
    rows = []
    for metric in METRICS:
        if not ids:
            result = {"estimate": None, "p_value": None, "estimable": False, "inferential": False}
        else:
            d = b.loc[ids, metric].to_numpy(dtype=float)-a.loc[ids, metric].to_numpy(dtype=float)
            if metric == "davies_bouldin":
                d = -d
            result = paired_cluster_mean_inference(d, [group_clusters[key] for key in ids], **resampling)
        rows.append({"test_id": "ro1_"+metric, "metric": metric, "direction": "baseline_minus_candidate" if metric == "davies_bouldin" else "candidate_minus_baseline", **result})
    return pd.DataFrame(rows)


def save_geometry_figure(summary, output_path):
    """Save a standalone empirical summary figure for export."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        ax.bar(summary.backbone, summary[metric])
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
