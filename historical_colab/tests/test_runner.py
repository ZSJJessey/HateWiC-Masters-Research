import pytest
import pandas as pd
from hatewic_formal.runner import make_schedule,partitions,run_pending

def test_schedule_has_exact_required_jobs(tmp_path):
    frame=make_schedule(tmp_path)
    assert len(frame)==460 and frame.run_id.nunique()==460
    assert frame.phase.value_counts().to_dict()=={'core_repeats':320,'single_seed':140}
    assert frame[frame.phase.eq('single_seed')].seed.unique().tolist()==[13]
    assert frame[frame.phase.eq('core_repeats')].model.nunique()==4

def test_missing_gates_prevent_training(tmp_path):
    with pytest.raises(FileNotFoundError):run_pending(tmp_path,{},max_jobs=1)

def test_invalid_queue_settings_fail_before_work(tmp_path):
    with pytest.raises(ValueError):run_pending(tmp_path,{},phase='reduced_folds')
    with pytest.raises(ValueError):run_pending(tmp_path,{},max_jobs=0)

def test_pilot_reuses_completed_measurement_after_later_failure(tmp_path,monkeypatch):
    import hatewic_formal.runner as runner
    monkeypatch.setattr(runner.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(runner.torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(runner.torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(runner,'environment',lambda:{'device':'fake GPU'})
    main=pd.DataFrame({'example_id':range(40),'majority_label':[0,1]*20})
    manifest=pd.DataFrame({'example_id':range(40),'protocol':'random','rotation':0,'split':['train']*32+['validation']*4+['test']*4})
    monkeypatch.setattr(runner,'load_artifacts',lambda root:(main,manifest,{'data_version':'data'},{'manifest_sha256':'fold'}))
    calls=[]
    def build(variant,**kwargs):
        calls.append(variant)
        if calls==['hatebert_context','full']:raise RuntimeError('Simulated later-model cache failure')
        return object()
    def train(*args,**kwargs):
        folder=args[5];folder.mkdir(parents=True,exist_ok=True)
        (folder/'best.pt').write_bytes(b'fixture')
        return {'elapsed_seconds':1.,'peak_cuda_bytes':10,'updates':2}
    monkeypatch.setattr(runner,'build_model',build);monkeypatch.setattr(runner,'run_training',train)
    with pytest.raises(RuntimeError,match='later-model'):runner.resource_pilot(tmp_path,{})
    result=runner.resource_pilot(tmp_path,{})
    assert calls==['hatebert_context','full','full']
    assert len(result['benchmarks'])==2 and result['status']=='passed'
