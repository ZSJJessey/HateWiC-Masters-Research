import json,hashlib,importlib.util
from pathlib import Path
import pytest

def module(tmp_path,monkeypatch):
    p=Path(__file__).resolve().parents[1]/'update_formal_chapter.py';s=importlib.util.spec_from_file_location('chapter_stage',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);monkeypatch.setattr(m,'ROOT',tmp_path);return m

def fixture(tmp_path):
    folder=tmp_path/'results/single_seed';folder.mkdir(parents=True)
    data=b'model,macro_f1\nfull,0.5\n';(folder/'metrics.csv').write_bytes(data)
    digest=hashlib.sha256(data).hexdigest();(folder/'evidence.json').write_text(json.dumps({'files':{'metrics.csv':digest},'status':'complete_for_requested_phase'}))
    identity={'backend':'mps'};markers={'run':'prediction-sha'}
    (folder/'local_provenance.json').write_text(json.dumps({'identity':identity,'accepted_prediction_hashes':markers}))
    return folder,identity,markers

def test_cache_requires_matching_source_markers(tmp_path,monkeypatch):
    folder,identity,markers=fixture(tmp_path);m=module(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError):m.verify_phase_cache('single_seed',{'run':'changed'},identity)

def test_cache_requires_actual_result_fingerprints(tmp_path,monkeypatch):
    folder,identity,markers=fixture(tmp_path);m=module(tmp_path,monkeypatch);(folder/'metrics.csv').write_text('changed')
    with pytest.raises(RuntimeError):m.verify_phase_cache('single_seed',markers,identity)

def test_matching_cache_is_accepted(tmp_path,monkeypatch):
    folder,identity,markers=fixture(tmp_path);m=module(tmp_path,monkeypatch)
    assert m.verify_phase_cache('single_seed',markers,identity)
