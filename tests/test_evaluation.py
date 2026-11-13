import numpy as np
import pandas as pd
import pytest

from hatewic_formal.evaluation import binary_metrics, validate_oof, align_paired_predictions, validate_score_metadata, primary_classification_tests
from hatewic_formal.statistics import paired_cluster_inference, generalization_gap_inference, paired_statistic, holm_family
from hatewic_formal.geometry import geometry_scores, compare_geometry, geometry_inference


def test_fixed_labels_and_unsupported_subgroup():
    m = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["macro_f1"] == pytest.approx((2/3 + 4/5)/2)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["balanced_accuracy"] == .75
    d = binary_metrics([1, 1], [1, 1])
    assert d["macro_f1"] is None and not d["estimable"]
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0, 2])


def fixture_oof():
    manifest = pd.DataFrame({"example_id": ["a", "b", "c", "d"], "protocol": "random", "fold": [0, 0, 1, 1], "split": "test"})
    blocks = []
    for model in ["full", "no_gloss"]:
        for seed in [13, 37]:
            blocks.append(manifest.drop(columns="split").assign(model=model, seed=seed, majority_label=[0, 1, 0, 1], prediction=[0, 1, 0, 1], probability=[.1, .9, .2, .8]))
    return pd.concat(blocks, ignore_index=True), manifest


def test_oof_matching_and_missing_ids():
    p, manifest = fixture_oof()
    assert validate_oof(p, manifest)["runs"] == 4
    for bad in [pd.concat([p, p.iloc[[0]]]), p.drop(index=0), p.assign(fold=4)]:
        with pytest.raises(ValueError):
            validate_oof(bad, manifest)
    y, a, b, ids, seeds = align_paired_predictions(p.sample(frac=1, random_state=3), "no_gloss", "full", "random")
    assert ids == ["a", "b", "c", "d"] and seeds == [13, 37]
    assert a.shape == b.shape == (2, 4)
    with pytest.raises(ValueError):
        align_paired_predictions(p.loc[~(p.model.eq("full") & p.seed.eq(37))], "no_gloss", "full", "random")
    with pytest.raises(ValueError):
        validate_score_metadata(pd.DataFrame({"example_id": ids[:-1], "sense_heterogeneous": False, "lexical_cluster_id": "x"}), ids)


def test_oof_frozen_truth_and_threshold():
    p, manifest = fixture_oof()
    manifest = manifest.assign(majority_label=[0, 1, 0, 1])
    assert validate_oof(p, manifest)["validated"]
    # A consistent change across all models and seeds still violates frozen truth.
    tampered = p.copy()
    tampered.loc[tampered.example_id.eq("a"), "majority_label"] = 1
    with pytest.raises(ValueError, match="frozen manifest"):
        validate_oof(tampered, manifest)
    wrong_threshold = p.copy()
    wrong_threshold.loc[0, "probability"] = .5
    with pytest.raises(ValueError, match="threshold"):
        validate_oof(wrong_threshold, manifest)
    tie = p.copy()
    tie.loc[tie.example_id.eq("b"), "probability"] = .5
    assert validate_oof(tie, manifest)["validated"]


def test_paired_resampling_and_seed_invariance():
    y = np.tile([0, 1], 8)
    clusters = np.repeat(np.arange(8), 2)
    a = np.tile(y, (2, 1))
    r = paired_cluster_inference(y, a, a, clusters, n_bootstrap=99, n_permutations=99, seed=4)
    assert r["estimate"] == r["ci_low"] == r["ci_high"] == 0
    assert r["p_value"] == 1
    b = a.copy(); b[:, ::4] = 1-b[:, ::4]
    r1 = paired_cluster_inference(y, b, a, clusters, n_bootstrap=99, n_permutations=99, seed=4)
    r2 = paired_cluster_inference(y, b[:1], a[:1], clusters, n_bootstrap=99, n_permutations=99, seed=4)
    assert r1["estimate"] > 0
    for key in ["estimate", "ci_low", "ci_high", "p_value"]:
        assert r1[key] == r2[key]
    assert r1 == paired_cluster_inference(y, b, a, clusters, n_bootstrap=99, n_permutations=99, seed=4)


def test_degenerate_groups_and_gap_directions():
    y = np.tile([0, 1], 4)
    a, b = y.copy(), y.copy()
    a[4:] = 1-a[4:]
    h = np.repeat([False, True], 4)
    assert paired_statistic(y, a, b, "heterogeneity_gap_reduction", h) == 1
    assert paired_statistic(y, a, b, "heterogeneity_gap_reduction", y) is None
    r = paired_cluster_inference(y, a, b, np.zeros(8), n_bootstrap=9, n_permutations=9)
    assert not r["inferential"] and r["p_value"] is None
    r = generalization_gap_inference(y, y, a, y, y, np.repeat(np.arange(4), 2), n_bootstrap=9, n_permutations=9)
    assert r["estimate"] > 0


def test_holm_requires_complete_family():
    partial = holm_family([{"test_id": "one", "p_value": .001}])
    assert not partial.family_complete.any() and partial.p_holm.isna().all()
    full = holm_family([{"test_id": str(i), "p_value": .001 if i == 0 else .8} for i in range(10)])
    assert full.iloc[0].p_holm == .01
    assert full.iloc[0].significant_0_05
    registered = [{"test_id": str(i), "p_value": .001 if i == 0 else .8, "estimable": True} for i in range(10)]
    registered[-1].update(p_value=None, estimable=False)
    conservative = holm_family(registered)
    assert conservative.family_complete.all()
    assert conservative.iloc[-1].p_for_holm == 1 and conservative.iloc[-1].non_rejection_marker
    assert pd.isna(conservative.iloc[-1].p_value)
    assert not conservative.iloc[-1].significant_0_05


def test_seven_predeclared_classification_tests_and_protocol_matching():
    p, _ = fixture_oof()
    neural = p.loc[p.model.eq("full")]
    runs = pd.concat([neural.assign(model=model, protocol=protocol)
                      for model in ["full", "no_gloss", "bert_context", "hatebert_context"]
                      for protocol in ["random", "term_disjoint"]], ignore_index=True)
    meta = pd.DataFrame({"example_id": ["a", "b", "c", "d"], "sense_heterogeneous": [0, 0, 1, 1], "lexical_cluster_id": ["a", "b", "c", "d"]})
    tests = primary_classification_tests(runs, meta, n_bootstrap=9, n_permutations=9)
    assert len(tests) == tests.test_id.nunique() == 7
    assert tests.estimate.eq(0).all()
    bad = runs.copy()
    bad.loc[bad.protocol.eq("term_disjoint") & bad.model.eq("full"), "seed"] = 999
    with pytest.raises(ValueError):
        primary_classification_tests(bad, meta, n_bootstrap=9, n_permutations=9)


def test_geometry_normalizes_and_matches_groups():
    x = np.array([[1, .1], [1, -.1], [-1, .1], [-1, -.1]])
    y = np.array([0, 0, 1, 1])
    a = geometry_scores(x, y)
    b = geometry_scores(x*np.arange(1, 5)[:, None], y)
    assert a["cosine_separation"] > 1.9
    for key in ["cosine_separation", "silhouette_cosine", "davies_bouldin"]:
        assert a[key] == pytest.approx(b[key])
    assert not geometry_scores(x[:3], y[:3])["estimable"]
    samples = pd.DataFrame({"example_id": np.arange(8), "majority_label": np.tile(y, 2), "term_pos_key": np.repeat(["a|noun", "b|noun"], 4)})
    groups, summary, global_summary = compare_geometry({"bert": np.tile(x, (2, 1)), "hatebert": np.tile(x, (2, 1))}, samples)
    assert summary.supported_groups.tolist() == [2, 2]
    result = geometry_inference(groups, {"a|noun": "a", "b|noun": "b"}, n_bootstrap=9, n_permutations=9)
    assert result.estimate.eq(0).all() and result.p_value.eq(1).all()
    with pytest.raises(ValueError):
        geometry_scores(np.zeros((4, 2)), y)
