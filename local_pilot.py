"""Offline Apple-GPU pilot on train/dev only, with a checkpoint resume."""
from pathlib import Path
import os,sys,time,gc,platform,json,resource
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ.pop('PYTORCH_ENABLE_MPS_FALLBACK',None)
import torch
from hatewic_formal.config import code_hash,write_json,sha256
from hatewic_formal.runner import load_artifacts,partitions
from hatewic_formal.models import build_model
from hatewic_formal.tokenization import create_tokenizers
from hatewic_formal.training import DEFAULT_TRAINING_CONFIG,run_training,predict_dataframe

def main():
    if not torch.backends.mps.is_available():raise RuntimeError('Apple GPU unavailable; no CPU fallback.')
    torch.set_num_threads(4)
    cohort,folds,audit,fold_audit=load_artifacts(ROOT)
    parts=partitions(cohort,folds,'random',0)
    train=parts['train'].groupby('majority_label',group_keys=False).head(32).sort_values('example_id')
    dev=parts['validation'].groupby('majority_label',group_keys=False).head(16).sort_values('example_id')
    assert not set(train.example_id)&set(dev.example_id)
    cache=ROOT/'public_model_cache';tokens=create_tokenizers(cache_dir=str(cache))
    identity=dict(code_sha256=code_hash(),data_version=audit['data_version'],fold_sha256=fold_audit['manifest_sha256'],backend='mps',precision='float32',pilot_script_sha256=sha256(Path(__file__)))
    config={**DEFAULT_TRAINING_CONFIG,'max_epochs':1,'checkpoint_every_updates':2}
    previous=ROOT/'verification/local_pilot.json'
    if previous.is_file():
        cached=json.loads(previous.read_text())
        if cached.get('status')=='passed' and cached.get('identity')==identity and cached.get('torch_version')==torch.__version__:
            for row in cached['benchmarks']:
                assert sha256(ROOT/'pilot'/row['model']/'best.pt')==row['best_checkpoint_sha256']
            print('Reused verified local GPU pilot; no training repeated.',flush=True)
            return cached
    report=dict(status='running',identity=identity,device='Apple M5 MPS',physical_memory_bytes=17179869184,torch_version=torch.__version__,precision='float32',train_examples=len(train),validation_examples=len(dev),test_examples_used=0,benchmarks=[],numerical_resume_evidence=json.loads((ROOT/'verification/mps_recovery.json').read_text()),notes='Separate backend and precision from CUDA AMP; no cross-device checkpoint continuation or pooled neural scoring.')
    out=ROOT/'verification/local_pilot.json';write_json(out,report)
    for variant in ['hatebert_context','full']:
        folder=ROOT/'pilot'/variant
        model=build_model(variant,seed=13,cache_dir=str(cache),local_files_only=True)
        started=time.monotonic()
        print('Starting',variant,'64 train / 32 dev on MPS',flush=True)
        if not (folder/'restore.pt').exists():
            paused=run_training(train,dev,tokens,model,config,folder,identity,mode='benchmark',device='mps',interrupt_after_updates=1)
            assert paused['status']=='interrupted'
            saved=torch.load(folder/'restore.pt',map_location='cpu',weights_only=False)
            assert saved['state']['updates']==1 and saved['rng']['mps'] is not None
            del saved;gc.collect()
            print('Checkpoint and Apple GPU RNG saved; resuming',variant,flush=True)
        result=run_training(train,dev,tokens,model,config,folder,identity,mode='benchmark',device='mps')
        torch.mps.synchronize()
        elapsed=time.monotonic()-started
        predictions=predict_dataframe(model,dev,tokens,config)
        assert len(predictions)==len(dev) and predictions.probability.between(0,1).all()
        predictions.to_csv(folder/'validation_predictions.csv',index=False)
        report['benchmarks'].append(dict(best_checkpoint_sha256=sha256(folder/'best.pt'),model=variant,elapsed_including_resume_and_validation_seconds=elapsed,result=result,checkpoint_bytes=sum(p.stat().st_size for p in folder.glob('*.pt')),mps_allocated_bytes_after=torch.mps.current_allocated_memory(),mps_driver_allocated_bytes_after=torch.mps.driver_allocated_memory(),process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write_json(out,report)
        print('Completed',variant,'seconds',round(elapsed,2),'allocated GB',round(torch.mps.current_allocated_memory()/1024**3,2),flush=True)
        del model;gc.collect();torch.mps.empty_cache()
    report['status']='passed';report['scope']='Local train/dev resource and execution pilot only; no formal test predictions or batch training.'
    write_json(out,report)
    print('Local MPS pilot passed for both models.',flush=True)
if __name__=='__main__':main()
