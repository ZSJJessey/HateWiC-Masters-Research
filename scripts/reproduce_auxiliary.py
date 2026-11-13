"""Explicit private baseline/feature/geometry reproduction. May take hours."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def main():
    from hatewic_formal.runner import load_artifacts
    from hatewic_formal.config import sha256
    from hatewic_formal.tokenization import create_tokenizers
    from hatewic_formal.feature_extraction import extract_frozen_features,load_features
    from hatewic_formal.baselines import run_baselines
    from hatewic_formal.evaluation import summarize_oof
    from hatewic_formal.reporting import geometry_report
    if not (ROOT/'artifacts/data_audit.json').is_file():raise SystemExit('Private artifacts missing; follow docs/reproduction.md first.')
    main,manifest,audit,folds=load_artifacts(ROOT)
    scoring=__import__('pandas').read_csv(ROOT/'artifacts/scoring_registry.csv')
    fold_hash=sha256(ROOT/'artifacts/fold_manifest.csv')
    classical=run_baselines(main,manifest,ROOT/'baselines/classical',audit['data_version'],fold_hash)
    summarize_oof(classical,scoring).to_csv(ROOT/'baselines/classical/metrics.csv',index=False)
    tokens=create_tokenizers(cache_dir=str(ROOT/'public_model_cache'),local_files_only=True)
    paths=extract_frozen_features(main,tokens,str(ROOT/'public_model_cache'),ROOT/'frozen_features')
    geometry_report(ROOT,paths)
    features={'bert':load_features(paths['bert_base']),'hatebert':load_features(paths['hatebert'])}
    predictions=run_baselines(main,manifest,ROOT/'baselines/frozen_mlp',audit['data_version'],fold_hash,models=['bert_frozen_mlp','hatebert_frozen_mlp'],features_by_backbone=features)
    summarize_oof(predictions,scoring).to_csv(ROOT/'baselines/frozen_mlp/metrics.csv',index=False)
    print('Private auxiliary comparisons complete.')
if __name__=='__main__':main()
