# Experimental protocol

## Data and partitions

The supplied source contains 12,442 individual annotation rows and 4,029 examples. Binary majority voting retains 3,845 candidates and excludes 184 without a qualifying majority. The conservative main set has 3,493 examples: 1,815 non-hateful and 1,678 hateful. Unresolved positions, rule-selected repeated occurrences and non-three-annotator records are excluded by the recorded evidence rules; original data are retained privately.

The source-sense key preserves original target spelling and case and combines target, POS and local sense ID. The lexical grouping key applies NFKC, case folding and whitespace normalization without POS. Source-sense heterogeneity is used only for evaluation grouping, not as a model input or training signal. The main set has 2,630 homogeneous and 863 heterogeneous examples.

Random folds isolate duplicate-text components. Lexical-disjoint folds isolate joint lexical and duplicate components. The fold seed is 20261026. Rotation k uses fold k for test, (k+1) mod 10 for validation and eight remaining folds for training. Test predictions occur once per example, model, protocol and seed.

## Models and training

The context representation is the last-layer mean over target subwords. The independent BERT-base gloss encoder pools last-layer non-special gloss tokens. The full model concatenates target, gloss, absolute difference and element-wise product, followed by a 256-unit ReLU head, dropout 0.1 and one logit. Sigmoid is used for the 0.5 prediction threshold; BCE is calculated from logits. The contrastive objective uses normalized target representations.

| Setting | Value |
|---|---|
| Context/gloss maximum length | 128/64 including special tokens |
| Actual batch / accumulation | 8 / 2 |
| Optimizer | AdamW, weight decay 0.01 |
| Peak learning rate | 2e-5 |
| Schedule | Linear warmup over 10% of planned updates, then linear decay |
| Maximum epochs / patience | 8 / 2 |
| Selection | Validation macro F1; ties keep the earlier epoch |
| Supervised contrastive weight / temperature | 0.1 / 0.1 where enabled |
| Seeds | 13, 37, 73, 101, 149 |
| Formal backend | Apple M5 MPS float32 |

The no-gloss ablation retains contrastive loss. Both context baselines and the no-contrastive ablation use BCE alone. Contrastive positives are limited to the actual eight-example batch; accumulation does not enlarge that batch. Anchors without a same-class positive are omitted from the contrastive term and retain BCE.

Seven variants at seed 13 across two protocols and ten folds give 140 fits. Four core variants (BERT context, HateBERT context, full, no gloss) at four additional seeds give 320 more fits. The other three ablations remain single-seed comparisons.

## Evaluation and inference

Macro F1 is primary. Per-class precision, recall, F1, support, accuracy and confusion counts are also saved. RO1 uses eligible target/POS groups with at least two examples per class, equal group weighting and L2-normalized representations: cosine separation, cosine-distance Silhouette and Euclidean Davies-Bouldin.

Inference uses paired lexical/duplicate-component resampling, 10,000 bootstrap draws and 10,000 paired permutations. Ten prespecified tests receive a joint Holm correction: three RO1, two RO2 backbone contrasts, two overall gloss gains, two heterogeneity-gap changes and one cross-protocol gap change. Intervals condition on fitted models; overlapping training folds are not independent replicates. The cross-protocol comparison evaluates the same examples under different training conditions.
