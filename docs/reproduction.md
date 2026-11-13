# Reproduction instructions

## 1. Review without data

Install the environment and run the synthetic tests from the README. Open `results/tables/` and `results/figures/` to review saved aggregate results. No real-data notebook needs to be executed for this review.

The foundation notebook includes all twelve historical steps. Interactive reviews write private files and require human decisions. Do not use Run All on it. Later steps read the saved review records. The historical review dates and folder names refer to the private project layout and should be configured for an authorised project.

## 2. Prepare an independent private reproduction

Clone into a new private working directory on your computer; generated artifacts must never be committed. Use a fresh checkout, rather than the original thesis run directory. Obtain the authorised source CSV and the private target-position/duplicate evidence files from the researcher through the approved private channel. The raw CSV alone does not reproduce historical manual span decisions.

Copy `configs/private_inputs.example.json` to the ignored `configs/private_inputs.json` and set the six private input paths. Then run:

```bash
python scripts/prepare_reproduction.py --config configs/private_inputs.json --stage prepare
python prepare_models.py
python scripts/prepare_reproduction.py --stage validate
```

Preparation verifies evidence, rebuilds the conservative cohort and recreates folds. Validation runs synthetic CPU checks and the full tokenizer audit. Missing private inputs stop with an English message before preparation proceeds. Artifacts and verification files are private outputs, excluded from Git.

## 3. Check the Apple GPU and seal fresh gates

Only on a compatible Mac with MPS available, run:

```bash
python scripts/prepare_reproduction.py --stage pilot
python scripts/prepare_reproduction.py --stage seal
```

The pilot stage explicitly runs synthetic hardware/recovery tests and the real train/validation resource pilot. It is a training action, so it is separate from validation. It uses no test examples. The seal stage checks the launch gates and writes a new execution identity for this checkout. If the environment, code, data or fold hashes differ from a previous reproduction, stop and inspect; do not overwrite existing run markers.

The original `requirements-local.lock` is the exact environment record, not a promise of identical results on other platforms. The model revisions are pinned. MPS availability and floating-point behavior must be checked on the reproduction machine. CUDA runs belong in a separate output directory and are not interchangeable with the thesis MPS runs.

## 4. Explicitly launch training

From the repository root, open `notebooks/04_Local_Formal_Experiments.ipynb` in JupyterLab or run:

```bash
python start_formal_queue.py
```

Set `HATEWIC_CODE_ROOT` to the checkout root if opening local notebooks from another directory. The queue resumes matching checkpoints and skips verified completed predictions. It does not start just by importing the modules. Keep the Mac powered and awake. Do not run a second pilot while the formal queue is active.

`run_schedule.csv` lists all 460 jobs. Predictions, checkpoints, status and recovery files are generated privately in the checkout. The queue checks coverage, labels, fingerprints and identity before counting a fit as complete.

## 5. Classical baselines, frozen features and inference

Use the functions in `hatewic_formal.baselines`, `feature_extraction` and `reporting`, following Sections 6-7 of the historical Colab as a call-sequence reference, but import the final package for the final implementation. Keep all outputs in the private reproduction directory. Run `scripts/reproduce_auxiliary.py` explicitly to calculate classical baselines, frozen features, RO1 geometry and feature MLPs using the final package. RO1 and final matched predictions are prerequisites for the ten-test family. Final reporting in `update_formal_chapter.py` validates execution provenance before generating the phase summaries.

The full inference uses 10,000 resamples/permutations and can take substantial CPU time; it is not part of a quick smoke check. Single-seed ablations and five-seed core summaries are reported separately. Never reuse the public aggregate files as if they were newly generated predictions.

## 6. Public plotting

`python scripts/plot_public_results.py` creates fresh aggregate-only plots under the ignored `public_plot_preview/` directory. Published thesis PNG files in `results/figures/` are retained as audited originals. Recreated plot styling may differ while the tabulated numbers remain the source of truth.
