import importlib.util
from pathlib import Path
import pandas as pd
import pytest


def module():
    path=Path(__file__).resolve().parents[1]/'local_runner.py'
    spec=importlib.util.spec_from_file_location('local_formal_runner',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def fixture():
    job=dict(run_id='random__full__s13__f00',model='full',protocol='random',seed=13,fold=0)
    truth=pd.DataFrame({'example_id':[1,2,3,4],'majority_label':[0,1,0,1]})
    pred=truth.assign(prediction=[0,1,0,1],probability=[.2,.8,.3,.9],model='full',protocol='random',seed=13,fold=0)
    return job,truth,pred


def test_valid_complete_test_predictions():
    job,truth,pred=fixture();assert module().verify_prediction_frame(pred,truth,job)['status']=='passed'

@pytest.mark.parametrize('problem',['wrong_label','wrong_fold','duplicate_id','nan_probability','threshold_mismatch'])
def test_corrupt_predictions_are_not_counted(problem):
    job,truth,pred=fixture()
    if problem=='wrong_label':pred.loc[0,'majority_label']=1
    if problem=='wrong_fold':pred.loc[0,'fold']=1
    if problem=='duplicate_id':pred.loc[0,'example_id']=2
    if problem=='nan_probability':pred.loc[0,'probability']=float('nan')
    if problem=='threshold_mismatch':pred.loc[0,'prediction']=1
    with pytest.raises(ValueError):module().verify_prediction_frame(pred,truth,job)


def test_queue_contains_every_required_job_once():
    rows=module().schedule_rows()
    assert len(rows)==460 and len({x['run_id'] for x in rows})==460
    assert sum(x['phase']=='single_seed' for x in rows)==140
    assert sum(x['phase']=='core_repeats' for x in rows)==320
    assert {x['seed'] for x in rows}=={13,37,73,101,149}
