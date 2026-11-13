import importlib.util,json,hashlib
from pathlib import Path
import pytest

def queue_module():
 p=Path(__file__).resolve().parents[1]/'execute_formal.py';s=importlib.util.spec_from_file_location('local_queue',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture(tmp_path,monkeypatch):
 m=queue_module();monkeypatch.setattr(m,'ROOT',tmp_path)
 job=dict(run_id='random__full__s13__f00',phase='single_seed',model='full',seed=13,protocol='random',fold=0)
 folder=tmp_path/'runs'/job['run_id'];folder.mkdir(parents=True)
 data=b'example_id,majority_label,prediction,probability,protocol,model,seed,fold\n1,0,0,0.2,random,full,13,0\n2,1,1,0.8,random,full,13,0\n';(folder/'predictions.csv').write_bytes(data)
 identity=dict(code_sha256='code',runner_sha256='runner',data_version='data',fold_sha256='fold',backend='mps',precision='float32',torch_version='version')
 (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts/main.csv').write_text('example_id,majority_label\n1,0\n2,1\n');(tmp_path/'artifacts/fold_manifest.csv').write_text('example_id,protocol,rotation,split\n1,random,0,test\n2,random,0,test\n');identity['fold_sha256']=m.sha(tmp_path/'artifacts/fold_manifest.csv');(tmp_path/'artifacts/data_audit.json').write_text(json.dumps({'data_version':'data','files':{'main.csv':m.sha(tmp_path/'artifacts/main.csv')}}));(tmp_path/'verification').mkdir();(tmp_path/'verification/formal_execution_identity.json').write_text(json.dumps(identity))
 record=dict(status='completed',identity={**identity,'run_id':job['run_id']},job=job,predictions_sha256=hashlib.sha256(data).hexdigest(),wall_seconds_this_invocation=50.,resumed_from_checkpoint=False)
 p=folder/'prediction_verified.json';p.write_text(json.dumps(record))
 return m,job,p,record

def test_verified_result_counts_once(tmp_path,monkeypatch):
 m,job,p,record=fixture(tmp_path,monkeypatch);assert m.completed([job])==[job['run_id']]

@pytest.mark.parametrize('key',['code_sha256','data_version','fold_sha256','run_id','backend'])
def test_other_run_or_version_never_counts(tmp_path,monkeypatch,key):
 m,job,p,record=fixture(tmp_path,monkeypatch);record['identity'][key]='different';p.write_text(json.dumps(record))
 with pytest.raises(RuntimeError):m.completed([job])

def test_changed_prediction_never_counts(tmp_path,monkeypatch):
 m,job,p,record=fixture(tmp_path,monkeypatch);(p.parent/'predictions.csv').write_text('changed')
 with pytest.raises(RuntimeError):m.completed([job])

@pytest.mark.parametrize('content',[b'example_id,prediction\n1,0\n', b'example_id,majority_label,prediction,probability,protocol,model,seed,fold\n1,1,0,0.2,random,full,13,0\n2,1,1,0.8,random,full,13,0\n', b'example_id,majority_label,prediction,probability,protocol,model,seed,fold\n1,0,1,0.2,random,full,13,0\n2,1,1,0.8,random,full,13,0\n'])
def test_fingerprint_matched_but_invalid_contents_never_count(tmp_path,monkeypatch,content):
 m,job,p,record=fixture(tmp_path,monkeypatch);(p.parent/'predictions.csv').write_bytes(content);record['predictions_sha256']=hashlib.sha256(content).hexdigest();p.write_text(json.dumps(record))
 with pytest.raises(RuntimeError):m.completed([job])
