# Chapter 4 Results and Discussion

## 4.1 Overview

This chapter evaluates whether domain adaptation separates hateful from non-hateful target representations, whether a target-focused classifier improves contextual classification, and whether linked sense definitions improve classification and subgroup performance. The evaluation separates representation geometry from predictive performance. It also distinguishes random grouping from lexical-disjoint evaluation, where target terms do not overlap across training, validation and test partitions.

The evaluation uses a fixed conservative cohort and matched out-of-fold predictions. This stage reports representation geometry and five baselines. Fine-tuned neural and linked-gloss comparisons remain pending.

## 4.2 Dataset and experimental setup

### 4.2.1 Data audit and analysis population

The fresh local audit reconstructs 4,029 examples from 12,442 individual annotation rows. Of these, 3,845 have a qualifying binary majority and 184 do not. The conservative main set contains 3,493 examples: 1,815 non-hateful and 1,678 hateful. Table 4.1 records the audit population and evaluation supports. Its values are taken from `artifacts/data_audit.json`.

| Table 4.1 Audit quantity | Verified count |
|---|---|
| Individual annotation rows | 12,442 |
| Original examples | 4,029 |
| Majority-label candidates | 3,845 |
| Excluded without a majority | 184 |
| Conservative main set | 3,493 |
| Non-hateful and hateful main examples | 1,815 and 1,678 |
| Unresolved position candidates | 100 |
| Rule-selected position candidates | 242 |
| Majority candidates without three annotations | 11 |
| Source senses in original data and main set | 1,888 and 1,767 |
| Heterogeneous source senses and their examples | 278 and 863 |
| Main lexical groups and joint components | 799 and 790 |
| RO1 eligible target term and POS groups | 133 |

The exclusion categories overlap: one of the non-three-annotator examples is already a rule-selected case. Their union removes 352 majority candidates from the main set. The original source file is fixed by its SHA-256 fingerprint in the audit. The main table was also rebuilt from the private Drive source in Colab. Its SHA-256 matched the locally frozen main CSV, and both cloud fold checks passed before model fitting. The saved cloud checkpoint records this verification in `verification/cloud_checks.json`.

Each example's binary label is reconstructed from individual annotations. Weakly and strongly hateful responses map to the hateful class, while cannot-decide responses remain outside the two class vote totals. The reproduced candidate rule requires at least two votes for the winning binary class and more votes for it than for the other binary class. In the final main set, every example has exactly three annotations, so this corresponds to a two-of-three majority. Source contexts, glosses and individual labels remain unchanged. The analysis ledger preserves exclusions and their reasons, including overlapping reasons.

The conservative analysis population excludes unresolved target positions, examples whose occurrence was selected by the last-occurrence rule, and examples without exactly three annotations. This defines the population to which the formal results apply. A unique literal or inflection-based candidate remains a documented evidence category rather than an original span annotation. Any recovery of an excluded example requires recorded evidence and a new frozen dataset version before scoring.

Source senses are identified by the original target term, part of speech and locally scoped sense identifier, retaining case. Lexical grouping separately normalizes the target term through NFKC, case folding and whitespace normalization and omits part of speech. These keys serve different purposes: the former preserves source meaning identities and the latter prevents target-term overlap.

### 4.2.2 Evaluation protocols

Both protocols use ten frozen folds and split seed 20261026. Rotation k tests on fold k, validates on the next fold modulo ten, and trains on the remaining eight folds. The random protocol keeps known duplicate-context components together. The lexical-disjoint protocol combines lexical identity and duplicate-context relations into connected components and keeps each component in one fold. Input hashes and complete assignments accompany the results.

The saved local fold audit passes both protocol checks and records nonzero support for both classes in every training, validation and test partition. Fold sizes need not match exactly because linked examples remain together. Evidence: `artifacts/fold_audit.json` and its recorded manifest fingerprint.

The full tokenization audit passes for all 3,493 main examples. BERT-base and HateBERT produce identical context token IDs and target masks for every example under the matched extraction procedure. Target-preserving windows clip 20 contexts for each backbone; 22 linked glosses are clipped at the gloss limit. All main examples have a gloss. These are input-processing findings rather than representation or classification results. Evidence: `artifacts/tokenizer_audit.json`.

Each model produces one test prediction per example, protocol and seed. Preprocessing that learns from data is fitted on the training partition for each rotation. Validation macro F1 controls epoch selection. The test partition does not determine thresholds, early stopping, settings or benchmark choices. These controls make the out-of-fold comparisons use the same examples without treating overlapping training folds as independent experiments.

### 4.2.3 Models and training

The full model uses an independent HateBERT context encoder and BERT-base linked-gloss encoder. The context branch averages the final-layer subwords belonging to the target. The gloss branch averages final-layer non-special, non-padding tokens. Their interaction concatenates the two representations, their absolute difference and their elementwise product. A 256-unit ReLU layer with dropout 0.1 produces a single classification logit.

| Setting | Fixed first pass |
|---|---|
| Context and gloss maximum lengths | 128 and 64 tokens including special tokens |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Warmup | 10 percent of optimizer steps |
| Actual batch and accumulation | 8 examples and 2 batches |
| Maximum epochs and patience | 8 and 2 |
| Selection | Validation macro F1, retaining earlier epoch on ties |
| Classification and threshold | BCE with logits and probability threshold 0.5 |
| Contrastive weight and temperature | 0.1 and 0.1 |
| First seed | 13 |
| Core repeated seeds | 13, 37, 73, 101, 149 |

The supervised contrastive objective operates on normalized target representations within the actual batch. Anchors with no same-class positive are omitted from its mean but retain their BCE contribution. Gradient accumulation changes the optimizer update batch without creating additional contrastive positives. The no-gloss ablation retains the contrastive term; contextual baselines use BCE. The remaining ablations remove structured interaction, remove contrastive loss or freeze the context branch.

These fixed settings replace the candidate-grid selection described in Chapter 3 for the initial pass. The first pass comprises seven neural configurations across both protocols and ten folds, requiring 140 fits. Repeating four core configurations with four further seeds requires 320 additional fits. Model revisions, package versions, selected epochs, elapsed time and run status are saved. Execution uses the local Apple GPU after the training and validation resource pilot.

### 4.2.4 Metrics and inference

Macro F1 is the arithmetic mean of the hateful and non-hateful class F1 scores. Per-class precision, recall, F1 and support, accuracy and confusion matrices accompany it. Results use fixed class order and the same evaluated IDs within a comparison. Missing or degenerate subgroup support is reported rather than silently changing the comparison population.

RO1 uses L2-normalized frozen target vectors. Cosine separation is the pooled mean same-label pairwise cosine similarity minus mean different-label similarity within a target-term and POS group. Silhouette uses cosine distance, and Davies-Bouldin uses Euclidean distance on normalized vectors. Eligible groups contain at least two examples of each class and receive equal weight in the primary aggregate. Global geometry is supplementary. Higher separation and Silhouette and lower Davies-Bouldin indicate clearer label separation under their respective definitions.

Paired inference resamples lexical and duplicate joint components, preserving the prediction pairing. Cluster bootstrap confidence intervals and paired permutation tests use 10,000 replicates. Inference requires at least two independent components, at least 90 percent valid bootstrap draws and all requested permutation draws to be valid. Comparisons outside these conditions remain descriptive. Holm correction applies to the ten prespecified primary tests when the corresponding results are available. The intervals condition on the fitted models; repeated seeds characterize training variation separately. Single-seed results do not establish stability across training randomness.

## 4.3 RO1 Target representation separation

The primary representation analysis covers 133 target-term and POS groups with at least two examples of each class. Each group receives equal weight. BERT-base and HateBERT were extracted over the same 3,493 main examples using the same target-preserving inputs and last-layer target mean pooling. Extraction used float32 on Apple MPS with both encoders frozen in evaluation mode. Saved feature archives verify the example order, model revisions, tokenizer identity and checksums.

| Table 4.2 Within-target geometry | BERT-base | HateBERT |
|---|---:|---:|
| Cosine separation | 0.021820 | 0.017148 |
| Cosine Silhouette | 0.057497 | 0.056017 |
| Davies-Bouldin index | 1.953350 | 1.982007 |

The three point estimates favor BERT-base: HateBERT has lower cosine separation and Silhouette and a higher Davies-Bouldin index. This pattern does not support the proposed representation-separation benefit in this analysis. The paired intervals below include zero, so the estimates also leave uncertainty about the direction and size of the population difference.

| Table 4.3 Paired geometry effect | Effect and 95% cluster interval | Unadjusted p |
|---|---|---:|
| Cosine separation | -0.004672 [-0.011125, 0.001502] | 0.147885 |
| Cosine Silhouette | -0.001480 [-0.015065, 0.011974] | 0.832917 |
| Davies-Bouldin index | -0.028657 [-0.072603, 0.014153] | 0.195980 |

Effects equal HateBERT minus BERT-base for cosine separation and Silhouette, and BERT-base minus HateBERT for Davies-Bouldin. Positive values therefore favor HateBERT in all three rows. Inference uses 123 joint lexical and duplicate components, 10,000 cluster-bootstrap draws and 10,000 paired permutations. These p values are unadjusted; the prespecified ten-test Holm family remains pending until the seven classification comparisons are available. Global geometry is saved as a supplementary analysis and does not replace the within-target comparison.

Evidence: `results/ro1/summary.csv`, `results/ro1/group_scores.csv`, `results/ro1/primary_tests_uncorrected.csv`, `results/ro1/global_supplementary.csv` and `frozen_features/extraction_run.json`.

![Figure 4.1. Paired geometry effects and 95% cluster-bootstrap intervals; displayed p values are unadjusted.](results/figures/ro1_paired_effects.png)

## 4.4 RO2 Target hateful sense classification

The completed baseline evaluation comprises 100 runs across both ten-fold protocols: 60 classical baseline runs and 40 frozen-feature MLP runs. Each model contributes one out-of-fold test prediction for every main example. The table pools the 3,493 predictions per protocol and model; these are seed-13 results on the formal frozen cohort.

| Table 4.4 Baseline | Random macro F1 | Lexical-disjoint macro F1 |
|---|---:|---:|
| Training majority | 0.341937 | 0.341937 |
| Training lexical prior | 0.686030 | 0.341937 |
| TF-IDF and logistic regression | 0.668101 | 0.647929 |
| BERT-base frozen features + MLP | 0.747192 | 0.725894 |
| HateBERT frozen features + MLP | 0.745802 | 0.716457 |

BERT-base frozen features with an MLP produce the highest observed baseline macro F1 under both protocols. HateBERT frozen features are close under random grouping and have a lower point score under lexical-disjoint evaluation. These results describe the fixed frozen-feature classifiers; the matched fine-tuned contextual comparison remains pending.

The lexical prior uses normalized target term and POS. Ties and unseen keys return the current training majority class. Every test term is unseen under the lexical-disjoint protocol, so this baseline returns the training majority there. TF-IDF uses unigrams and bigrams, minimum document frequency 2, at most 50,000 features, sublinear term frequency, Unicode accent stripping and a token pattern that includes single-character words. Logistic regression uses C = 1, balanced class weights, LBFGS and at most 1,000 iterations. Vocabulary and estimator fitting use the current training partition alone.

The frozen-feature MLPs use four hidden layers of 300, 200, 100 and 50 units, learning rate 0.0005 and ten iterations. Standardization is fitted independently within each training partition. Both backbones use the same MLP settings, sample IDs and folds. Convergence warnings are retained in the individual run records. No test-guided setting changes were made.

Fine-tuned contextual classifiers, the full dual encoder and the ablations: Pending experimental results. The formal RO2 primary tests compare the matched fine-tuned contextual models and are not supplied by the frozen-feature table.

Evidence: `baselines/classical/metrics.csv`, `baselines/frozen_mlp/metrics.csv`, their complete out-of-fold predictions and the corresponding run configurations.

![Figure 4.2. Pooled out-of-fold macro F1 for five fixed baselines under both protocols; seed 13.](results/figures/baseline_macro_f1.png)

## 4.5 RO3 Linked definitions and subgroup performance

Pending experimental results.

The definition gain is full-model macro F1 minus no-gloss macro F1 on identical examples. Sense heterogeneity is defined by the presence of both majority labels within one source-sense group of the frozen main dataset. Its registry is used for scoring alone and never enters the model, training objective, early stopping or parameter selection.

The heterogeneity gap equals homogeneous-group macro F1 minus heterogeneous-group macro F1. Report each absolute subgroup score alongside the change in the gap. The cross-protocol generalization gap equals random-protocol macro F1 minus lexical-disjoint macro F1 on the same test example IDs. It compares models trained under different conditions; report this qualification with the comparison. A smaller gap supports improvement in a difficult group only when its absolute score also supports that interpretation. Linked gold-sense gloss availability defines this evaluation setting and does not demonstrate automatic sense retrieval at deployment.

## 4.6 Ablation analysis

Pending experimental results.

Compare the full model with no gloss, simple concatenation, no contrastive loss and frozen context configurations on matched predictions. The no-gloss comparison retains contrastive training to isolate the gloss branch. Simple concatenation tests structured interaction. Removing contrastive loss tests its incremental role. Freezing context tests the contribution of adapting the context branch while the other trainable components remain available. Record parameter counts and training time when measured, without attributing a difference to an unisolated mechanism.

## 4.7 Error analysis and discussion

Pending experimental results.

Select error categories from out-of-fold predictions, including target morphology, multiple occurrences, context dependence, gloss alignment and lexical novelty. Review the same error sample across matched models and retain category definitions and sample-selection rules. Present restricted-data examples only within the permitted research audience. Connect the eventual conclusions to the observed absolute scores, paired differences and dataset population. Include non-positive findings and uncompleted runs where they affect the answers to the research questions.

## Evidence sources

Dataset counts and provenance are linked to `artifacts/data_audit.json`, fold consistency to `artifacts/fold_audit.json`, tokenization to `artifacts/tokenizer_audit.json` and classical test metrics to `baselines/classical/metrics.csv`. The number and source field of each reported result are recorded in `artifacts/result_source_index.json`. The earlier grouped validation baselines and synthetic implementation checks remain preparation evidence and are excluded from formal result tables. This file contains no estimated model results.
