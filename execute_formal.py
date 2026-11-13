"""Single local supervisor: isolated fits, truthful heartbeat and phase-complete reports."""
from pathlib import Path
import csv,json,os,sys,time,fcntl,subprocess,hashlib,signal,math
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent

def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows():
    with (ROOT/'run_schedule.csv').open() as f:items=list(csv.DictReader(f))
    for r in items:
        for k in ['seed','fold']:r[k]=int(r[k])
    if len(items)!=460 or len({r['run_id'] for r in items})!=460:raise RuntimeError('Full frozen 460-fit schedule required.')
    return items

def completed(items):
    expected=json.loads((ROOT/'verification/formal_execution_identity.json').read_text())
    audit=json.loads((ROOT/'artifacts/data_audit.json').read_text())
    if audit['data_version']!=expected['data_version'] or sha(ROOT/'artifacts/main.csv')!=audit['files']['main.csv'] or sha(ROOT/'artifacts/fold_manifest.csv')!=expected['fold_sha256']:
        raise RuntimeError('Frozen cohort or folds changed.')
    with (ROOT/'artifacts/main.csv').open() as f:
        labels={r['example_id']:int(r['majority_label']) for r in csv.DictReader(f)}
    test_ids={}
    with (ROOT/'artifacts/fold_manifest.csv').open() as f:
        for r in csv.DictReader(f):
            if r['split']=='test':test_ids.setdefault((r['protocol'],int(r['rotation'])),set()).add(r['example_id'])
    known={job['run_id'] for job in items}
    if any(p.parent.name not in known for p in (ROOT/'runs').glob('*/prediction_verified.json')):
        raise RuntimeError('Unscheduled result directory cannot enter this matrix.')
    found=[]
    for job in items:
        p=ROOT/'runs'/job['run_id']/'prediction_verified.json'
        if p.exists():
            record=json.loads(p.read_text());csv_path=p.parent/'predictions.csv'
            if record.get('status')!='completed' or sha(csv_path)!=record['predictions_sha256']:raise RuntimeError('Completed prediction fingerprint differs.')
            if record['identity'].get('backend')!='mps' or record['identity'].get('precision')!='float32':raise RuntimeError('Foreign backend result in local matrix.')
            if record['identity']!={**expected,'run_id':job['run_id']} or record.get('job')!=job:
                raise RuntimeError('Result belongs to a different run, code, data or fold version.')
            with csv_path.open() as f:
                reader=csv.DictReader(f)
                required={'example_id','majority_label','prediction','probability','protocol','model','seed','fold'}
                if not required.issubset(reader.fieldnames or []):raise RuntimeError('Prediction fields missing.')
                frame=list(reader)
            ids=[r['example_id'] for r in frame]
            if len(set(ids))!=len(ids) or set(ids)!=test_ids.get((job['protocol'],job['fold']),set()):raise RuntimeError('Test coverage differs.')
            for r in frame:
                probability=float(r['probability'])
                if not math.isfinite(probability) or not 0<=probability<=1 or int(r['prediction'])!=int(probability>=.5):raise RuntimeError('Prediction threshold or probability differs.')
                if int(r['majority_label'])!=labels[r['example_id']]:raise RuntimeError('Frozen test label differs.')
                if any(r[k]!=str(job[k]) for k in ['protocol','model','seed','fold']):raise RuntimeError('Prediction fold or model differs.')
            found.append(job['run_id'])
    return found

def progress(items,active=None,status='running',detail=None):
    done=completed(items)
    value={'updated_at':stamp(),'supervisor_pid':os.getpid(),'status':status,'active_run_id':active,
        'planned_single_seed':140,'planned_core_repeats':320,'completed_single_seed':sum(r['phase']=='single_seed' and r['run_id'] in done for r in items),
        'completed_core_repeats':sum(r['phase']=='core_repeats' and r['run_id'] in done for r in items),
        'completed_neural_runs':len(done),'remaining_neural_runs':460-len(done),'completed_ids':done,'backend':'mps','precision':'float32','detail':detail}
    if active:
        p=ROOT/'runs'/active/'live_progress.json'
        if p.exists():value['active_training_progress']=json.loads(p.read_text());value['active_progress_file_mtime']=p.stat().st_mtime
    # Estimates use complete fresh fits of the same configuration when available.
    per_model={}
    for rid in done:
        record=json.loads((ROOT/'runs'/rid/'prediction_verified.json').read_text())
        if not record['resumed_from_checkpoint']:
            per_model.setdefault(record['job']['model'],[]).append(record['wall_seconds_this_invocation'])
    value['measured_full_fit_seconds_by_model']={m:sum(v)/len(v) for m,v in per_model.items()}
    unknown=[r for r in items if r['run_id'] not in done and r['model'] not in per_model]
    value['remaining_gpu_hours_same_model_extrapolation']=None if unknown else sum(value['measured_full_fit_seconds_by_model'][r['model']] for r in items if r['run_id'] not in done)/3600
    value['estimate_note']='Completed full-fit means by model; not a guaranteed completion time. Missing model timings keep total null.'
    write(ROOT/'run_progress.json',value);return value

def postprocess(phase=None):
    args=[sys.executable,str(ROOT/'update_formal_chapter.py')]
    if phase:args+=['--phase',phase]
    with (ROOT/'.runtime/postprocess.log').open('a') as log:
        result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=log)
    if result.returncode:raise RuntimeError('Postprocessing failed; inspect local log, preserve all completed fits.')

def main(max_jobs=None):
    runtime=ROOT/'.runtime';runtime.mkdir(exist_ok=True)
    with (runtime/'formal_queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=rows();count=0
        try:
            progress(jobs)
            for phase in ['single_seed','core_repeats']:
                phase_jobs=[r for r in jobs if r['phase']==phase]
                for job in phase_jobs:
                    if job['run_id'] in completed(jobs):continue
                    progress(jobs,job['run_id'],detail='Launching one isolated Apple GPU process.')
                    log_path=ROOT/'runs'/job['run_id']/'process.log';log_path.parent.mkdir(parents=True,exist_ok=True)
                    with log_path.open('a') as log:
                        child=subprocess.Popen([sys.executable,str(ROOT/'local_runner.py'),'--run-id',job['run_id']],cwd=ROOT,stdout=log,stderr=log)
                        while child.poll() is None:
                            progress(jobs,job['run_id'],detail={'worker_pid':child.pid,'log':str(log_path.relative_to(ROOT))})
                            time.sleep(15)
                    if child.returncode:raise RuntimeError(f"Formal fit failed: {job['run_id']}; recovery files preserved.")
                    count+=1;progress(jobs)
                    postprocess()
                    if max_jobs is not None and count>=max_jobs:
                        progress(jobs,status='bounded_batch_complete');return
                postprocess(phase)
            progress(jobs,status='training_and_reports_complete',detail='460 fits verified; final document rendering and substantive interpretation still require verification.')
        except BaseException as exc:
            progress(jobs,status='needs_attention',detail={'error_type':type(exc).__name__,'error':str(exc)})
            raise

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--max-jobs',type=int);args=parser.parse_args()
    if args.max_jobs is not None and args.max_jobs<1:raise ValueError('max-jobs must be positive.')
    main(args.max_jobs)
