# Implementation lineage and publication edits

The final `src/hatewic_formal/` package and local runner come from the implementation used for all 460 thesis neural fits. Model, training, tokenization, evaluation and statistics modules are preserved byte for byte. The source-to-public-file mapping and SHA-256 values are in `source_mapping.json`.

The earlier Colab archive is expanded exactly into `historical_colab/`; its original archive hash and member hashes are in `historical_archive_manifest.json`. The archive contained Python source and synthetic tests only. The historical model/training implementation differs from the final MPS version, including device and recovery support. No cross-backend equality is asserted.

Publication edits remove notebook outputs, widget metadata and personal paths; replace the Colab embedded archive with visible repository source; make local notebook roots configurable; and select the report-building Python through `HATEWIC_DOC_PYTHON` (default: the current interpreter). The quantitative stage template is moved to `docs/chapter4_report_template.md`. Its pending fields belong to a report-generation template and are not completed-results claims.

Original frozen source fingerprints must not be bypassed. A new checkout must generate fresh validation identities for its own reproduction directory; it cannot claim to be an original thesis run by copying old markers. The original data, source code, predictions, weights and review files remain untouched.

Root orchestration scripts retain their original placement because their imports, source hashes and tests depend on that structure. `scripts/` contains the added preparation and publication utilities. All such utilities are distinguished from the original scientific implementation.
