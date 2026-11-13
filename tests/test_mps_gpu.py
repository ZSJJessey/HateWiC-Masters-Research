"""Hardware checks; skipped unless an actual Apple GPU is accessible."""
import copy
import pytest
import torch
from test_neural import tiny,frame,tokenizers
from hatewic_formal.training import run_training,predict_dataframe
from hatewic_formal.models import VARIANTS
from hatewic_formal.tokenization import tokenize_dataframe
pytestmark=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual Apple GPU required')

@pytest.mark.parametrize('variant',VARIANTS)
def test_actual_mps_forward_and_backward(tokenizers,variant):
    model=tiny(variant).to('mps');model.train()
    data=tokenize_dataframe(frame(n=4),tokenizers,model.context_name,16,8,model.uses_gloss)
    batch={k:torch.stack([row[k] for row in data]).to('mps') for k in data[0]}
    output=model(**batch);output['loss'].backward();torch.mps.synchronize()
    assert torch.isfinite(output['loss']).item()
    assert model.head[0].weight.grad is not None
    if variant=='frozen_context':assert all(p.grad is None for p in model.context_encoder.parameters())
    del model,output,batch;torch.mps.empty_cache()


def test_actual_mps_resume(tokenizers,tmp_path):
    torch.manual_seed(23);initial=tiny('full');a=copy.deepcopy(initial);b=copy.deepcopy(initial)
    cfg=dict(batch_size=2,gradient_accumulation_steps=2,max_epochs=2,patience=2,context_max_length=16,gloss_max_length=8,learning_rate=.001,seed=13)
    identity=dict(code_hash='mps-fixture',data_hash='synthetic',fold_hash='synthetic')
    completed=run_training(frame(),frame(30,4),tokenizers,a,cfg,tmp_path/'a',identity,mode='smoke',device='mps')
    interrupted=run_training(frame(),frame(30,4),tokenizers,b,cfg,tmp_path/'b',identity,mode='smoke',device='mps',interrupt_after_updates=1)
    assert interrupted['status']=='interrupted'
    b=tiny('full')
    recovered=run_training(frame(),frame(30,4),tokenizers,b,cfg,tmp_path/'b',identity,mode='smoke',device='mps')
    assert len(completed['history'])==len(recovered['history'])
    history_delta=max(abs(x['train_loss']-y['train_loss']) for x,y in zip(completed['history'],recovered['history']))
    for x,y in zip(completed['history'],recovered['history']):
        assert x['epoch']==y['epoch'] and x['updates']==y['updates']
        assert x['validation_macro_f1']==y['validation_macro_f1']
    assert history_delta<=1e-7
    differences={k:float((t.cpu()-b.state_dict()[k].cpu()).abs().max()) for k,t in a.state_dict().items() if not torch.equal(t.cpu(),b.state_dict()[k].cpu())}
    assert all(torch.allclose(t.cpu(),b.state_dict()[k].cpu(),rtol=0,atol=1e-7) for k,t in a.state_dict().items()),differences
    pa=predict_dataframe(a,frame(30,4),tokenizers,cfg)
    pb=predict_dataframe(b,frame(30,4),tokenizers,cfg)
    assert pa.y_pred.equals(pb.y_pred)
    delta=float((pa.probability-pb.probability).abs().max())
    assert delta<=1e-7
    from pathlib import Path
    from hatewic_formal.config import write_json
    write_json(Path(__file__).resolve().parents[1]/'verification/mps_recovery.json',dict(
        status='passed_with_declared_float32_tolerance',weight_absolute_tolerance=1e-7,
        bitwise_equal=not bool(differences),weight_max_absolute_difference=max(differences.values(),default=0),
        prediction_probability_max_absolute_difference=delta,identical_predicted_labels=True,
        identical_training_history=completed['history']==recovered['history'],training_loss_max_absolute_difference=history_delta,history_absolute_tolerance=1e-7,scope='Synthetic tiny full dual encoder; does not establish cross-device bitwise equivalence.'))
