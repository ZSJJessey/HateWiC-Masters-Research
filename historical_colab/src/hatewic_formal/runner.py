"""Explicit gates, resource pilot, resumable jobs and verified test predictions."""
from pathlib import Path
import gc
import importlib.metadata
import json
import shutil
import time
import numpy as np
import pandas as pd
import torch
from .config import VARIANTS, CORE_VARIANTS, SEEDS, code_hash, sha256, object_hash, write_json
from .training import DEFAULT_TRAINING_CONFIG, run_training, predict_dataframe
from .models import build_model
from .evaluation import binary_metrics, validate_oof, summarize_oof

def environment():
    packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','torch','transformers']}
    return {'packages':packages,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}

def load_artifacts(root):
    root=Path(root)
    audit=json.loads((root/'artifacts/data_audit.json').read_text())
    for name,digest in audit['files'].items():
        if sha256(root/'artifacts'/name)!=digest: raise ValueError('Data artifact fingerprint changed: '+name)
    folds=json.loads((root/'artifacts/fold_audit.json').read_text())
    if sha256(root/'artifacts/fold_manifest.csv')!=folds['manifest_sha256']: raise ValueError('Fold manifest changed.')
    return pd.read_csv(root/'artifacts/main.csv',keep_default_na=False),pd.read_csv(root/'artifacts/fold_manifest.csv'),audit,folds

def partitions(main,manifest,protocol,fold):
    rows=manifest[manifest.protocol.eq(protocol)&manifest.rotation.eq(fold)]
    if len(rows)!=len(main): raise ValueError('Incomplete rotation.')
    return {split:main[main.example_id.isin(rows.loc[rows.split.eq(split),'example_id'])].sort_values('example_id').assign(split=split).reset_index(drop=True) for split in ['train','validation','test']}

def make_schedule(root):
    jobs=[]
    for phase,seeds,variants in [('single_seed',[13],VARIANTS),('core_repeats',SEEDS[1:],CORE_VARIANTS)]:
        for seed in seeds:
            for protocol in ['random','term_disjoint']:
                for fold in range(10):
                    for model in variants:
                        jobs.append({'run_id':f'{protocol}__{model}__s{seed}__f{fold:02d}','phase':phase,'model':model,'seed':seed,'protocol':protocol,'fold':fold})
    frame=pd.DataFrame(jobs)
    frame.to_csv(Path(root)/'run_schedule.csv',index=False)
    return frame

def _identity(audit,folds):
    return {'code_sha256':code_hash(),'data_version':audit['data_version'],'fold_sha256':folds['manifest_sha256']}

def resource_pilot(root,tokenizers,cache_dir=None):
    """A timing experiment on train/dev only. No test IDs enter the model."""
    if not torch.cuda.is_available(): raise RuntimeError('A free Colab GPU is required for the resource pilot.')
    root=Path(root);main,manifest,audit,folds=load_artifacts(root)
    parts=partitions(main,manifest,'random',0)
    free=shutil.disk_usage(root).free
    if free<8*1024**3: raise RuntimeError('At least 8 GiB free storage is required for active recovery and model caches.')
    prior=root/'benchmark/resource_pilot.json'
    if prior.is_file():
        cached=json.loads(prior.read_text())
        if cached.get('status')=='passed' and cached.get('identity')==_identity(audit,folds) and cached.get('environment',{}).get('device')==environment()['device']:
            return cached
    configs=[]
    for variant in ['hatebert_context','full']:
        train=parts['train'].groupby('majority_label',group_keys=False).head(32).sort_values('example_id')
        validation=parts['validation'].groupby('majority_label',group_keys=False).head(16).sort_values('example_id')
        config={**DEFAULT_TRAINING_CONFIG,'max_epochs':1,'seed':13,'checkpoint_every_updates':2}
        folder=root/'benchmark'/code_hash()[:12]/variant
        measurement=folder/'measurement.json'
        if measurement.is_file():
            cached=json.loads(measurement.read_text())
            if cached.get('identity')!=_identity(audit,folds):raise ValueError('Pilot measurement identity differs.')
            configs.append(cached['measurement'])
            continue
        model=build_model(variant,seed=13,cache_dir=cache_dir)
        started=time.monotonic()
        result=run_training(train,validation,tokenizers,model,config,folder,_identity(audit,folds),mode='benchmark')
        torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        bytes_saved=sum(p.stat().st_size for p in folder.glob('*.pt'))
        timing={'training_elapsed_seconds':result['elapsed_seconds'],'peak_cuda_bytes':result['peak_cuda_bytes'],'updates':result['updates']}
        # Deliberately conservative projection includes checkpoint and validation overhead.
        projected=elapsed*len(parts['train'])/len(train)*8
        measured={'model':variant,'train_examples':len(train),'validation_examples':len(validation),'elapsed_seconds':elapsed,'training_timing':timing,'checkpoint_bytes':bytes_saved,'projected_max_8epoch_seconds_per_full_fold':projected}
        configs.append(measured)
        write_json(measurement,{'identity':_identity(audit,folds),'measurement':measured})
        del model;gc.collect();torch.cuda.empty_cache()
        for p in folder.glob('*.pt'): p.unlink()  # Pilot checkpoints are disposable after timing is recorded.
    report={'status':'passed','identity':_identity(audit,folds),'environment':environment(),'free_bytes':free,'benchmarks':configs,'projection_note':'Conservative subset extrapolation includes pilot checkpoint frequency of every 2 updates; formal training saves every 50 updates. Actual full-run early stopping, storage throughput and free-GPU availability may differ.'}
    write_json(root/'benchmark/resource_pilot.json',report)
    return report

def check_gates(root):
    root=Path(root);main,manifest,audit,folds=load_artifacts(root)
    tests=json.loads((root/'verification/tests.json').read_text())
    evidence=json.loads((root/'artifacts/position_evidence_audit.json').read_text())
    tokens=json.loads((root/'artifacts/tokenizer_audit.json').read_text())
    resource=json.loads((root/'benchmark/resource_pilot.json').read_text())
    if tests.get('status')!='passed' or tests.get('code_sha256')!=code_hash(): raise ValueError('Code tests are missing or stale.')
    if evidence.get('status')!='passed' or evidence.get('ledger_sha256')!=audit['input_hashes']['ledger']: raise ValueError('Position evidence verification is missing or stale.')
    if tokens.get('status')!='passed' or tokens.get('main_examples')!=len(main): raise ValueError('Full tokenizer audit is missing or incomplete.')
    if tokens.get('data_version')!=audit['data_version']: raise ValueError('Tokenizer audit uses another dataset.')
    if resource.get('status')!='passed' or resource.get('identity')!=_identity(audit,folds): raise ValueError('Resource pilot is missing or stale.')
    return True

def run_job(root,job,tokenizers,cache_dir=None):
    root=Path(root);check_gates(root)
    main,manifest,audit,folds=load_artifacts(root)
    parts=partitions(main,manifest,job['protocol'],int(job['fold']))
    folder=root/'runs'/job['run_id'];folder.mkdir(parents=True,exist_ok=True)
    prediction_path=folder/'predictions.csv'
    config={**DEFAULT_TRAINING_CONFIG,'seed':int(job['seed']),'checkpoint_every_updates':50}
    identity={**_identity(audit,folds),'run_id':job['run_id']}
    if (folder/'prediction_verified.json').is_file():
        record=json.loads((folder/'prediction_verified.json').read_text())
        if record['identity']!=identity or record['predictions_sha256']!=sha256(prediction_path): raise ValueError('Saved prediction provenance mismatch.')
        return record
    if shutil.disk_usage(root).free<8*1024**3: raise RuntimeError('Insufficient free storage for active checkpoints; no run was removed.')
    write_json(folder/'status.json',{'status':'running','job':job,'identity':identity,'environment':environment()})
    try:
        model=build_model(job['model'],seed=int(job['seed']),cache_dir=cache_dir)
        result=run_training(parts['train'],parts['validation'],tokenizers,model,config,folder,identity,mode='formal',resume=True)
        frame=predict_dataframe(model,parts['test'],tokenizers,config).rename(columns={'y_true':'majority_label','y_pred':'prediction'})
        for key in ['fold','protocol','model','seed']: frame[key]=job[key]
        if set(frame.example_id)!=set(parts['test'].example_id) or frame.example_id.duplicated().any(): raise ValueError('Test predictions are incomplete.')
        if not frame.set_index('example_id').majority_label.sort_index().equals(parts['test'].set_index('example_id').majority_label.sort_index()): raise ValueError('Test labels changed.')
        frame.to_csv(prediction_path,index=False)
        restored=pd.read_csv(prediction_path)
        if not np.array_equal(restored.prediction,frame.prediction): raise ValueError('Prediction readback mismatch.')
        record={'status':'completed','identity':identity,'predictions_sha256':sha256(prediction_path),'metrics':binary_metrics(frame.majority_label,frame.prediction),'training':result}
        write_json(folder/'prediction_verified.json',record)
        write_json(folder/'status.json',{'status':'completed','job':job,'identity':identity})
        # Preserve representative core weights; all other completed checkpoints are temporary.
        preserve=job['model'] in CORE_VARIANTS and int(job['seed'])==13 and int(job['fold'])==0 and job['protocol']=='random'
        for name in ['restore.pt']+([] if preserve else ['best.pt']):
            p=folder/name
            if p.exists(): p.unlink()
        del model;gc.collect();torch.cuda.empty_cache()
        return record
    except Exception as exc:
        write_json(folder/'status.json',{'status':'failed','job':job,'identity':identity,'error_type':type(exc).__name__,'error':str(exc)})
        raise

def progress(root):
    root=Path(root);schedule=make_schedule(root)
    completed=[x for x in schedule.run_id if (root/'runs'/x/'prediction_verified.json').is_file()]
    report={'planned_single_seed':140,'planned_core_repeats':320,'completed_neural_runs':len(completed),'remaining_neural_runs':460-len(completed),'completed_ids':completed}
    pilot=root/'benchmark/resource_pilot.json'
    if pilot.is_file():
        b=json.loads(pilot.read_text())['benchmarks'];timings={x['model']:x['projected_max_8epoch_seconds_per_full_fold'] for x in b}
        pending=schedule[~schedule.run_id.isin(completed)]
        report['projected_remaining_gpu_hours']=float(sum(timings['hatebert_context' if m in ['bert_context','hatebert_context','no_gloss'] else 'full'] for m in pending.model)/3600)
    else: report['projected_remaining_gpu_hours']=None
    write_json(root/'run_progress.json',report)
    return report

def run_pending(root,tokenizers,phase='single_seed',max_jobs=1,cache_dir=None):
    if phase not in ['single_seed','core_repeats']:raise ValueError('Unknown schedule phase.')
    if max_jobs is not None and max_jobs<1:raise ValueError('max_jobs must be positive or None.')
    check_gates(root)
    schedule=make_schedule(root);completed=0
    if phase=='core_repeats':
        missing=[rid for rid in schedule.loc[schedule.phase.eq('single_seed'),'run_id'] if not (Path(root)/'runs'/rid/'prediction_verified.json').is_file()]
        if missing:raise RuntimeError(f'{len(missing)} first-pass neural runs remain; complete them before core repeats.')
    for job in schedule[schedule.phase.eq(phase)].to_dict('records'):
        if (Path(root)/'runs'/job['run_id']/'prediction_verified.json').exists():
            run_job(root,job,tokenizers,cache_dir=cache_dir)
            continue
        run_job(root,job,tokenizers,cache_dir=cache_dir);completed+=1
        progress(root)
        if max_jobs is not None and completed>=max_jobs:break
    return progress(root)
