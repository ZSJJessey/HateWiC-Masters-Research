# HateWiC Master's Research

Code and aggregate experimental results for Shijie Zhou's Master's research in Artificial Intelligence at Universiti Kebangsaan Malaysia.

This is a personal research repository, not the official HateWiC dataset repository. The project studies target-word hateful-sense classification with BERT-base, HateBERT and dataset-linked sense glosses.

## Start here

1. Read [the experiment protocol](docs/protocol.md) and [the results guide](docs/results_guide.md).
2. Inspect [the foundation Colab notebook](notebooks/01_Foundation_Colab.ipynb), which includes all twelve preparation steps.
3. Inspect [the formal local notebook](notebooks/04_Local_Formal_Experiments.ipynb), [`src/hatewic_formal/models.py`](src/hatewic_formal/models.py) and [`src/hatewic_formal/training.py`](src/hatewic_formal/training.py).
4. Review [the aggregate tables](results/tables) and [the thesis figures](results/figures).
5. Follow [the reproduction instructions](docs/reproduction.md) for tests or an authorised data-based reproduction.

The thesis results come from the local Apple MPS float32 implementation: **140 first-pass fits and 320 additional core repeats, 460 in total**. The [historical Colab notebook](notebooks/02_Historical_Formal_Colab.ipynb) loads its own reviewed implementation from `historical_colab/`. It documents the earlier CUDA stage; its neural outputs are not pooled with the final MPS matrix.

## Findings

The conservative analysis set contains 3,493 examples. Both protocols use ten rotations with eight folds for training, one for validation and one for testing.

- RO1: the within-target representation analysis does not establish a HateBERT separation advantage. Non-significance is not evidence of equivalence.
- RO2: the full dual encoder has higher observed mean macro F1 than the context baselines. The prespecified RO2 tests compare HateBERT context with BERT context, so they do not directly confirm the broader full-model superiority hypothesis.
- RO3: adding dataset-linked sense glosses improves overall macro F1 by approximately 3.34 percentage points on random folds and 2.88 on lexical-disjoint folds. The homogeneous-versus-heterogeneous gaps widen. The cross-protocol gap does not significantly narrow.

The linked gloss is supplied as part of the evaluated input. These experiments do not establish an automatic sense-selection or gloss-retrieval system.

## Repository map

| Location | Purpose |
|---|---|
| `notebooks/` | Foundation Colab, historical formal Colab, local pilot and formal local entry points |
| `src/hatewic_formal/` | Authoritative formal model, data, folds, tokenization, training, evaluation and statistics |
| Root Python scripts | Original local orchestration layout, retained for imports, recovery and implementation tests |
| `scripts/` | Public preparation, validation and plotting utilities |
| `historical_colab/` | Exact expanded source and synthetic tests from the original embedded Colab archive |
| `tests/` | Formal implementation tests with synthetic fixtures |
| `configs/` | Pinned model revisions, environment record and private-input configuration example |
| `results/tables/` | Audited aggregate results only |
| `results/figures/` | Five figures used in the thesis results chapter |
| `docs/` | Protocol, source mapping, reproduction, result lineage and publication checks |

## Installation and synthetic checks

Use Python 3.12 and an isolated environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest tests -q --ignore=tests/test_mps_gpu.py
python scripts/validate_publication.py
```

The tests above use synthetic fixtures and do not need the private HateWiC CSV or pretrained model downloads. Hardware-specific checks are separate. Package availability and GPU support depend on the platform; the full original environment record is `requirements-local.lock`.

## Data access and publication boundary

HateWiC: Hoeken et al., EMNLP 2024. [Paper](https://aclanthology.org/2024.emnlp-main.10/) and [official repository](https://github.com/SanneHoeken/HateWiC).

The dataset was supplied for research use with a request not to redistribute it publicly. Obtain authorisation and request the full data from the authors. The public repository contains no source CSV, real contexts, linked gloss records, individual annotator records, example-level predictions, feature vectors or trained weights. The supervisor can review these through the separately authorised private research folder.

No open-source licence is granted for the original project code in this first release. See [third-party notices](THIRD_PARTY_NOTICES.md). Public visibility does not change the dataset's access conditions.
