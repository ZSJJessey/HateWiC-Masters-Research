"""Tables and ten prespecified tests from complete, verified OOF runs only."""
from pathlib import Path
import json
import pandas as pd
from .config import sha256, write_json, CORE_VARIANTS, VARIANTS
from .evaluation import validate_oof, summarize_oof, ro3_contrasts, primary_classification_tests
from .geometry import compare_geometry, geometry_inference
from .statistics import holm_family
from .feature_extraction import load_features
from .runner import load_artifacts

def geometry_report(root,feature_paths):
    root=Path(root);main,_,_,_=load_artifacts(root)
    arrays={}
    for name,key in [('bert','bert_base'),('hatebert','hatebert')]:
        features=load_features(feature_paths[key])
        if set(features.index)!=set(main.example_id): raise ValueError('Frozen feature coverage differs.')
        arrays[name]=features.loc[main.example_id].to_numpy()
    groups,summary,global_scores=compare_geometry(arrays,main)
    out=root/'results/ro1';out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('group_scores',groups),('summary',summary),('global_supplementary',global_scores)]:frame.to_csv(out/f'{name}.csv',index=False)
    clusters=main.groupby('term_pos_key').lexical_cluster_id.agg(lambda x: str(x.iloc[0])).to_dict()
    if main.groupby('term_pos_key').lexical_cluster_id.nunique().max()!=1: raise ValueError('Term/POS group crosses inference clusters.')
    tests=geometry_inference(groups,clusters,n_bootstrap=10000,n_permutations=10000,seed=13)
    tests.to_csv(out/'primary_tests_uncorrected.csv',index=False)
    write_json(out/'evidence.json',{'feature_files':{k:sha256(v) for k,v in feature_paths.items()},'summary_sha256':sha256(out/'summary.csv'),'primary_test_count':len(tests),'holm_status':'awaiting seven classification comparisons'})
    return summary

def collect_predictions(root):
    root=Path(root);main,manifest,_,_=load_artifacts(root)
    frames=[]
    for marker in (root/'runs').glob('*/prediction_verified.json'):
        saved=json.loads(marker.read_text());p=marker.parent/'predictions.csv'
        if sha256(p)!=saved['predictions_sha256']:raise ValueError('Prediction file checksum mismatch.')
        frames.append(pd.read_csv(p))
    if not frames:return pd.DataFrame()
    predictions=pd.concat(frames,ignore_index=True)
    # Partial model/seed/protocol blocks remain in their run folders, not in complete OOF tables.
    complete=[g for _,g in predictions.groupby(['protocol','model','seed']) if len(g)==len(main)]
    if not complete:return pd.DataFrame()
    predictions=pd.concat(complete,ignore_index=True)
    validate_oof(predictions,manifest,main.example_id)
    return predictions

def final_reports(root,phase='single_seed'):
    root=Path(root);main,manifest,_,_=load_artifacts(root)
    predictions=collect_predictions(root)
    seeds=[13] if phase=='single_seed' else [13,37,73,101,149]
    if phase not in ['single_seed','core_repeats']:raise ValueError('Unknown reporting phase.')
    if predictions.empty:raise ValueError('No complete neural OOF predictions are available.')
    predictions=predictions[predictions.seed.isin(seeds)]
    if phase=='core_repeats':predictions=predictions[predictions.model.isin(CORE_VARIANTS)]
    for protocol in ['random','term_disjoint']:
        for model in (VARIANTS if phase=='single_seed' else CORE_VARIANTS):
            for seed in seeds:
                if len(predictions[predictions.protocol.eq(protocol)&predictions.model.eq(model)&predictions.seed.eq(seed)])!=len(main):raise ValueError('Core comparison is incomplete; do not compute a partial primary family.')
    scoring=pd.read_csv(root/'artifacts/scoring_registry.csv')
    out=root/'results'/phase;out.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(out/'neural_oof_predictions.csv',index=False)
    metric_rows=summarize_oof(predictions,scoring)
    metric_rows.to_csv(out/'metrics.csv',index=False)
    seed_summary=metric_rows.groupby(['protocol','model','subgroup'],as_index=False).agg(
        completed_seeds=('seed','nunique'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_seed_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),
        accuracy_seed_sd=('accuracy','std'),balanced_accuracy_mean=('balanced_accuracy','mean'))
    seed_summary['expected_seeds']=len(seeds)
    if not seed_summary.completed_seeds.eq(len(seeds)).all():raise ValueError('A summary is missing a required seed.')
    seed_summary.to_csv(out/'seed_summary.csv',index=False)
    ro3_contrasts(predictions,scoring).to_csv(out/'ro3_contrasts.csv',index=False)
    classify=primary_classification_tests(predictions,scoring,n_bootstrap=10000,n_permutations=10000,seed=13)
    ro1path=root/'results/ro1/primary_tests_uncorrected.csv'
    if not ro1path.is_file():raise ValueError('RO1 results are required before ten-test Holm correction.')
    alltests=pd.concat([pd.read_csv(ro1path),classify],ignore_index=True)
    corrected=holm_family(alltests,expected_tests=10)
    corrected.to_csv(out/'primary_tests_holm.csv',index=False)
    write_json(out/'evidence.json',{'status':'complete_for_requested_phase','phase':phase,'seeds':seeds,'main_examples':len(main),'oof_validation':validate_oof(predictions,manifest,main.example_id),'files':{p.name:sha256(p) for p in out.glob('*.csv')}})
    return corrected
