"""One isolated MPS formal fit, with unchanged model settings and frozen partitions."""
from pathlib import Path
import os,sys,json,time,gc,fcntl,platform
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK',None)
import numpy as np
import pandas as pd
import torch
from hatewic_formal.config import VARIANTS,CORE_VARIANTS,SEEDS,code_hash,sha256,write_json
from hatewic_formal.runner import load_artifacts,partitions
from hatewic_formal.models import build_model
from hatewic_formal.training import DEFAULT_TRAINING_CONFIG,run_training,predict_dataframe
from hatewic_formal.tokenization import create_tokenizers
from hatewic_formal.evaluation import binary_metrics


def stamp():return datetime.now(timezone.utc).isoformat()


def schedule_rows():
    return [dict(run_id=f'{protocol}__{model}__s{seed}__f{fold:02d}',phase=phase,model=model,seed=seed,protocol=protocol,fold=fold)
      for phase,seeds,variants in [('single_seed',[13],VARIANTS),('core_repeats',SEEDS[1:],CORE_VARIANTS)]
      for seed in seeds for protocol in ['random','term_disjoint'] for fold in range(10) for model in variants]


def verify_prediction_frame(frame,test,job):
    needed={'example_id','majority_label','prediction','probability','protocol','model','seed','fold'}
    if not needed.issubset(frame):raise ValueError('Prediction fields missing.')
    if frame.example_id.duplicated().any() or set(frame.example_id)!=set(test.example_id):raise ValueError('Test ID coverage differs.')
    if not frame.set_index('example_id').majority_label.sort_index().equals(test.set_index('example_id').majority_label.sort_index()):raise ValueError('Test labels differ.')
    if not np.isfinite(frame.probability).all() or not frame.probability.between(0,1).all():raise ValueError('Invalid probabilities.')
    if not np.array_equal(frame.prediction,(frame.probability>=.5).astype(int)):raise ValueError('Threshold differs.')
    for key in ['protocol','model','seed','fold']:
        if not frame[key].eq(job[key]).all():raise ValueError('Prediction metadata differs: '+key)
    return {'status':'passed','test_examples':len(test)}


def identity_for(audit,folds,job):
    return dict(code_sha256=code_hash(),runner_sha256=sha256(Path(__file__)),data_version=audit['data_version'],
        fold_sha256=folds['manifest_sha256'],run_id=job['run_id'],backend='mps',precision='float32',torch_version=torch.__version__)


def check_gates(root=ROOT):
    root=Path(root);main,folds,audit,fold_audit=load_artifacts(root)
    tests=json.loads((root/'verification/tests.json').read_text())
    pilot=json.loads((root/'verification/local_pilot.json').read_text())
    tokenizer=json.loads((root/'artifacts/tokenizer_audit.json').read_text())
    evidence=json.loads((root/'artifacts/position_evidence_audit.json').read_text())
    if tests.get('status')!='passed' or tests.get('code_sha256')!=code_hash():raise ValueError('Current implementation checks required.')
    if tests.get('local_runner_sha256')!=sha256(Path(__file__)):raise ValueError('Local scheduler checks missing or stale.')
    if pilot.get('status')!='passed' or pilot['identity']['code_sha256']!=code_hash():raise ValueError('Current MPS pilot required.')
    if pilot['identity']['data_version']!=audit['data_version'] or pilot['identity']['fold_sha256']!=fold_audit['manifest_sha256']:raise ValueError('Pilot data/folds differ.')
    if pilot.get('torch_version')!=torch.__version__:raise ValueError('Pilot PyTorch version differs.')
    if tokenizer.get('status')!='passed' or tokenizer.get('data_version')!=audit['data_version'] or tokenizer.get('main_examples')!=len(main):raise ValueError('Full tokenizer audit required.')
    if evidence.get('status')!='passed' or evidence.get('ledger_sha256')!=audit['input_hashes']['ledger']:raise ValueError('Position evidence mismatch.')
    if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU unavailable; no CPU fallback.')
    if torch.cuda.is_available():raise RuntimeError('This queue is explicitly MPS-only.')
    return main,folds,audit,fold_audit


def run_one(run_id):
    rows={row['run_id']:row for row in schedule_rows()}
    if run_id not in rows:raise ValueError('Unknown formal run ID.')
    job=rows[run_id];main,folds,audit,fold_audit=check_gates()
    parts=partitions(main,folds,job['protocol'],job['fold'])
    identity=identity_for(audit,fold_audit,job)
    folder=ROOT/'runs'/run_id;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=folder/'prediction_verified.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if record['identity']!=identity or sha256(folder/'predictions.csv')!=record['predictions_sha256']:raise ValueError('Saved result provenance mismatch.')
            verify_prediction_frame(pd.read_csv(folder/'predictions.csv'),parts['test'],job)
            return record
        if __import__('shutil').disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Less than 12 GiB free disk; preserve results and stop.')
        resumed=(folder/'restore.pt').exists()
        if (folder/'run_config.json').exists() and not resumed and not (folder/'completed.json').exists():
            raise RuntimeError('Existing unfinished run has no recovery checkpoint; retain evidence and inspect before restarting.')
        started=time.monotonic();config={**DEFAULT_TRAINING_CONFIG,'seed':job['seed'],'checkpoint_every_updates':50}
        write_json(folder/'status.json',dict(status='running',stage='loading',job=job,identity=identity,started_at=stamp(),pid=os.getpid()))
        try:
            torch.set_num_threads(4)
            tokens=create_tokenizers(cache_dir=str(ROOT/'public_model_cache'))
            model=build_model(job['model'],seed=job['seed'],cache_dir=str(ROOT/'public_model_cache'),local_files_only=True)
            write_json(folder/'status.json',dict(status='running',stage='training',job=job,identity=identity,updated_at=stamp(),pid=os.getpid()))
            result=run_training(parts['train'],parts['validation'],tokens,model,config,folder,identity,mode='formal',resume=True,device='mps')
            write_json(folder/'status.json',dict(status='running',stage='test_prediction',job=job,identity=identity,updated_at=stamp(),pid=os.getpid()))
            frame=predict_dataframe(model,parts['test'],tokens,config).rename(columns={'y_true':'majority_label','y_pred':'prediction'})
            for key in ['protocol','model','seed','fold']:frame[key]=job[key]
            verify_prediction_frame(frame,parts['test'],job)
            pending=folder/'predictions.csv.pending';frame.to_csv(pending,index=False)
            verify_prediction_frame(pd.read_csv(pending),parts['test'],job);pending.replace(folder/'predictions.csv')
            record=dict(status='completed',identity=identity,job=job,predictions_sha256=sha256(folder/'predictions.csv'),metrics=binary_metrics(frame.majority_label,frame.prediction),training=result,
                wall_seconds_this_invocation=time.monotonic()-started,resumed_from_checkpoint=resumed,completed_at=stamp())
            write_json(marker,record)
            write_json(folder/'status.json',dict(status='completed',job=job,identity=identity,updated_at=stamp()))
            # These are documented disposable recovery files, cleared only after verified predictions.
            preserve=job['model'] in CORE_VARIANTS and job['seed']==13 and job['fold']==0 and job['protocol']=='random'
            for name in ['restore.pt']+([] if preserve else ['best.pt']):
                p=folder/name
                if p.is_file():p.unlink()
            del model;gc.collect();torch.mps.empty_cache()
            print(json.dumps({'completed':run_id,'wall_seconds':record['wall_seconds_this_invocation'],'epochs':len(result['history'])}),flush=True)
            return record
        except BaseException as exc:
            write_json(folder/'status.json',dict(status='failed',job=job,identity=identity,error_type=type(exc).__name__,error=str(exc),updated_at=stamp()))
            raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args()
    run_one(args.run_id)
