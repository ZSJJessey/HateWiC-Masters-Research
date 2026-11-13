"""No-download tests of alignment, objective, frozen branch and exact recovery."""
import copy
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from hatewic_formal.models import build_model, masked_mean, supervised_contrastive_loss, VARIANTS
from hatewic_formal.tokenization import tokenize_context, tokenize_dataframe
from hatewic_formal.training import run_training, predict_dataframe

@pytest.fixture
def tokenizers(tmp_path):
    vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','the','cat','cats','run','runs','a','small','animal','.','!','x']
    path=tmp_path/'vocab.txt'; path.write_text('\n'.join(vocab))
    tok=BertTokenizerFast(vocab={word:i for i,word in enumerate(vocab)},do_lower_case=True)
    return dict(bert_base=tok,hatebert=tok,gloss=tok)


def tiny(variant='full'):
    config=BertConfig(vocab_size=16,hidden_size=12,num_hidden_layers=1,num_attention_heads=3,
                      intermediate_size=24,hidden_dropout_prob=.1,attention_probs_dropout_prob=.1)
    return build_model(variant,context_encoder=BertModel(config),gloss_encoder=BertModel(config),hidden_size=8)


def frame(start=0,n=8):
    return pd.DataFrame([dict(example_id=i,example='the cat runs .',term='cat',target_start=4,target_end=7,
            majority_label=i%2,needs_U01C3_input_substitution=False,wiktionary_definition='a small animal') for i in range(start,start+n)])


def test_pool_and_contrastive():
    hidden=torch.tensor([[[1.,2.],[3.,4.],[9.,9.]]])
    assert torch.equal(masked_mean(hidden,torch.tensor([[1,1,0]])),torch.tensor([[2.,3.]]))
    x=torch.randn(2,5,requires_grad=True)
    zero=supervised_contrastive_loss(x,torch.tensor([0,1])); zero.backward()
    assert zero.item()==0 and torch.equal(x.grad,torch.zeros_like(x))
    x=torch.tensor([[1.,0.],[.9,.1],[0.,1.]],requires_grad=True)
    loss=supervised_contrastive_loss(x,torch.tensor([0,0,1]))
    assert torch.isfinite(loss) and loss.item()>=0
    loss.backward(); assert x.grad.abs().sum()>0


def test_alignment_and_window(tokenizers):
    tok=tokenizers['bert_base']; text='x '*30+'cat runs'
    result=tokenize_context(text,60,63,tok,max_length=8)
    assert len(result['input_ids'])==8 and sum(result['target_mask'])==1
    assert result['input_ids'][result['target_mask'].index(1)]==tok.convert_tokens_to_ids('cat')
    multi=tokenize_context('the cat runs',4,12,tok,max_length=8)
    assert sum(multi['target_mask'])==2
    with pytest.raises(ValueError):
        tokenize_context('the cat runs',5,7,tok)
    derived=tokenize_context('cat\u01c3',0,3,tok,max_length=8,substitute_u01c3=True)
    assert sum(derived['target_mask'])==1


@pytest.mark.parametrize('variant',VARIANTS)
def test_variants_gradient_and_freeze(tokenizers,variant):
    model=tiny(variant); model.train()
    data=tokenize_dataframe(frame(n=4),tokenizers,model.context_name,16,8,model.uses_gloss)
    batch={k:torch.stack([row[k] for row in data]) for k in data[0]}
    output=model(**batch); output['loss'].backward()
    assert output['logits'].shape==(4,) and torch.isfinite(output['loss'])
    assert model.head[0].weight.grad is not None
    if variant=='frozen_context':
        assert not model.context_encoder.training
        assert all(p.grad is None and not p.requires_grad for p in model.context_encoder.parameters())
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        context_ids={id(p) for p in model.context_encoder.parameters()}
        assert not context_ids.intersection(id(p) for group in optimizer.param_groups for p in group['params'])
    else:
        assert any(p.grad is not None for p in model.context_encoder.parameters())
    assert model.contrastive_weight==(0. if variant in ('bert_context','hatebert_context','no_contrastive') else .1)


def test_exact_resume_and_provenance(tokenizers,tmp_path):
    torch.set_num_threads(1); torch.manual_seed(11)
    initial=tiny('no_gloss'); full=copy.deepcopy(initial); resumed=copy.deepcopy(initial)
    config=dict(batch_size=2,gradient_accumulation_steps=2,max_epochs=2,patience=2,
                context_max_length=16,gloss_max_length=8,learning_rate=.001,seed=13)
    identity=dict(code_hash='fixture',data_hash='fixture',fold_hash='fixture')
    a=run_training(frame(),frame(20,4),tokenizers,full,config,tmp_path/'full',identity,mode='smoke')
    interrupted=run_training(frame(),frame(20,4),tokenizers,resumed,config,tmp_path/'resume',identity,mode='smoke',interrupt_after_updates=1)
    assert interrupted['status']=='interrupted'
    torch.manual_seed(927); resumed=tiny('no_gloss')
    b=run_training(frame(),frame(20,4),tokenizers,resumed,config,tmp_path/'resume',identity,mode='smoke')
    assert a['history']==b['history']
    selected=min(a['history'],key=lambda item:(-item['validation_macro_f1'],item['epoch']))
    assert a['best_epoch']==b['best_epoch']==selected['epoch']
    assert a['best_validation_macro_f1']==selected['validation_macro_f1']
    assert all(torch.equal(value,resumed.state_dict()[key]) for key,value in full.state_dict().items())
    pd.testing.assert_frame_equal(predict_dataframe(full,frame(20,4),tokenizers,config),predict_dataframe(resumed,frame(20,4),tokenizers,config))
    with pytest.raises(ValueError,match='differs'):
        run_training(frame(),frame(20,4),tokenizers,resumed,{**config,'seed':37},tmp_path/'resume',identity,mode='smoke')


def test_formal_requires_gpu(tokenizers,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='GPU'):
        run_training(frame(),frame(20,4),tokenizers,tiny(),{},tmp_path,dict(code_hash='x',data_hash='x',fold_hash='x'))
