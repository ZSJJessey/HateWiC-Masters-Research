"""Prepare and validate a fresh private reproduction; training requires --stage pilot."""
from pathlib import Path
import argparse,json,os,sys,subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','validate','pilot','seal'],required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/private_inputs.json')
    args=parser.parse_args()
    from hatewic_formal.config import write_json,code_hash,sha256
    if args.stage=='prepare':
        if not args.config.is_file():
            parser.error('Private input configuration missing. Copy configs/private_inputs.example.json to configs/private_inputs.json and supply authorised input paths.')
        paths={k:Path(v).expanduser() for k,v in json.loads(args.config.read_text()).items()}
        required=['raw','candidates','records','rules','ledger','duplicate_pairs']
        if any(k not in paths or not paths[k].is_file() for k in required):
            parser.error('Authorised private inputs are missing. Obtain the dataset and historical span/duplicate evidence through the private research channel.')
        if (ROOT/'artifacts/data_audit.json').exists():
            parser.error('Prepared artifacts already exist. Use a fresh checkout; do not overwrite a frozen reproduction.')
        from hatewic_formal.data import verify_position_sources,rebuild
        from hatewic_formal.folds import create_folds
        evidence=verify_position_sources(*(paths[k] for k in ['raw','candidates','records','rules','ledger']))
        main,scoring,audit=rebuild(paths['raw'],paths['ledger'],paths['duplicate_pairs'],ROOT/'artifacts')
        write_json(ROOT/'artifacts/position_evidence_audit.json',evidence)
        create_folds(main,ROOT/'artifacts')
        print('Private conservative cohort and both fold protocols prepared:',len(main))
        return
    if not (ROOT/'artifacts/data_audit.json').is_file():
        parser.error('Private artifacts are missing. Run --stage prepare with authorised inputs first.')
    from hatewic_formal.runner import load_artifacts
    main,manifest,audit,folds=load_artifacts(ROOT)
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'HF_HUB_OFFLINE':'1'}
    if args.stage=='validate':
        subprocess.run([sys.executable,'-m','pytest','tests','-q','--ignore=tests/test_mps_gpu.py'],cwd=ROOT,env=env,check=True)
        write_json(ROOT/'verification/tests.json',dict(status='passed',code_sha256=code_hash(),local_runner_sha256=sha256(ROOT/'local_runner.py'),scope='Synthetic CPU implementation tests'))
        from hatewic_formal.tokenization import create_tokenizers
        from hatewic_formal.feature_extraction import audit_tokenizers
        tokens=create_tokenizers(cache_dir=str(ROOT/'public_model_cache'),local_files_only=True)
        report=audit_tokenizers(main,tokens,ROOT/'artifacts')
        report.update(data_version=audit['data_version'],main_examples=len(main))
        write_json(ROOT/'artifacts/tokenizer_audit.json',report)
        print('Implementation and full tokenizer validation passed.')
    elif args.stage=='pilot':
        import torch
        if not torch.backends.mps.is_available():parser.error('An actual Apple GPU is required for the explicit pilot stage.')
        (ROOT/'verification').mkdir(exist_ok=True)
        subprocess.run([sys.executable,'-m','pytest','tests/test_mps_gpu.py','-q'],cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,'local_pilot.py'],cwd=ROOT,env=env,check=True)
    else:
        from local_runner import check_gates,identity_for
        from hatewic_formal.runner import make_schedule
        check_gates()
        identity=identity_for(audit,folds,dict(run_id='new-reproduction'))
        identity.pop('run_id')
        output=ROOT/'verification/formal_execution_identity.json'
        if output.exists() and json.loads(output.read_text())!=identity:
            parser.error('Existing execution identity differs; inspect it rather than replacing historical provenance.')
        write_json(output,identity)
        make_schedule(ROOT)
        print('Fresh execution identity sealed. Training has not been launched.')

if __name__=='__main__':main()
