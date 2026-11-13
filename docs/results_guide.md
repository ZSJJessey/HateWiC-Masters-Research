# Results and thesis figure guide

The public CSV files are aggregate outputs copied from the audited Chapter 4 delivery, except the baseline metrics, which come from the original private baseline output directories. No example-level predictions are published.

## Reading the tables

- `core_repeats/`: five-seed results for BERT context, HateBERT context, full and no-gloss configurations. `metrics.csv` gives seed-level subgroup aggregates, including class support and confusion counts; `seed_summary.csv` gives means and sample standard deviations. The standard deviation is not a confidence interval.
- `single_seed/`: seed-13 comparisons, including the three supporting ablations. These are not five-seed stability results.
- `ro1/`: eligible within-target geometry summaries and uncorrected tests; the completed ten-test Holm results are in the classification phase tables. Global supplementary geometry is reported separately.
- `baselines/`: classical and frozen-feature MLP metrics. Vocabulary, priors and scaling are fitted on each training partition only.
- `audit/`: independently recomputed aggregate summaries and fold support checks retained from the thesis audit. These are numerical verification outputs, not additional training seeds.

## Figures

| Thesis figure | Published file | Aggregate source and reproduction |
|---|---|---|
| 4.1 paired geometry effects | `ro1_paired_effects.png` | `ro1/primary_tests_uncorrected.csv`; `geometry.py` and `statistics.py` |
| 4.2 fixed baseline comparison | `baseline_macro_f1.png` | `baselines/*_metrics.csv`; `baselines.py` |
| 4.3 five-seed core comparison | `core_macro_f1.png` | `core_repeats/metrics.csv` and `seed_summary.csv` |
| 4.4 source-sense subgroup comparison | `subgroup_macro_f1.png` | `core_repeats/metrics.csv`; subgroup registry is private |
| 4.5 mean confusion counts | `mean_confusion.png` | `core_repeats/metrics.csv`, overall subgroup; arithmetic mean across five seeds |

The figures are the audited thesis PNG originals. `scripts/plot_public_results.py` provides an aggregate-only reconstruction with independent styling. It writes previews separately and does not overwrite the originals.

## Interpretation

The full model has mean macro F1 0.808814 on random folds and 0.794578 on lexical-disjoint folds. Paired full-minus-no-gloss gains are approximately 3.34 and 2.88 percentage points. For example, 0.808814 minus 0.775399 is about 0.033415, or 3.3415 percentage points, not a 3.3415 percent relative increase.

Overall improvement coexists with wider homogeneous-versus-heterogeneous gaps. Both subgroup absolute scores and gap changes must be read together. A widening gap cannot be described as improved heterogeneous-sense robustness. The cross-protocol gap compares the same examples under different training conditions.

The ten-test primary family is preserved. The RO2 tests compare HateBERT context with BERT context; they do not directly test full-model superiority over the context baselines. No new post-hoc confirmatory comparison is substituted here. Intervals condition on fitted models and do not include all sources of retraining uncertainty.

Counts in error analysis are quantitative. Semantic interpretations of the private sampled cases are not presented as human-verified findings.
