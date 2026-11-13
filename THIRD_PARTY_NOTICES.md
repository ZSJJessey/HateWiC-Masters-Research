# Third-party notices

This repository is an independent Master's research implementation. It does not include the HateWiC source dataset or claim ownership of it.

- HateWiC: Hoeken et al., *HateWiC: Hateful Word in Context Classification*, EMNLP 2024. https://aclanthology.org/2024.emnlp-main.10/ . Dataset access and redistribution conditions remain with the authors.
- BERT: Devlin et al. The public model is `google-bert/bert-base-uncased`; the exact revision is in `configs/model_revisions.json`. Follow the model repository's licence and notices when downloading or redistributing model assets.
- HateBERT: Caselli et al. The public model is `GroNLP/hateBERT`; the exact revision is recorded in the same configuration. Follow the model repository's terms and notices.
- The foundation notebook references the HateWiC source implementation as the origin of an occurrence-selection convention. Those references are retained; rule-selected repeated occurrences are excluded from the conservative main set.
- PyTorch, Transformers, scikit-learn, pandas, NumPy, SciPy, Matplotlib, pytest, Jupyter and python-docx are external dependencies. They are not vendored here and retain their respective licences.

No blanket open-source licence is added for the original code in this release. This notice does not grant rights over third-party data, model weights or other protected material.
