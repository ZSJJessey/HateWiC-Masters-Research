"""No-download extraction, cache identity, and exhaustive tokenizer auditing."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig,BertModel,BertTokenizerFast
from hatewic_formal.feature_extraction import extract_frozen_features,load_features,audit_tokenizers


def inputs():
    words=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','the','cat','runs','a','small','animal','x','!']
    tokenizer=BertTokenizerFast(vocab={w:i for i,w in enumerate(words)},do_lower_case=True)
    tokenizers={key:tokenizer for key in ('bert_base','hatebert','gloss')}
    frame=pd.DataFrame([dict(example_id=i,example='the cat runs',term='cat',target_start=4,target_end=7,
                            majority_label=i%2,wiktionary_definition='a small animal',needs_U01C3_input_substitution=False) for i in range(3)])
    cfg=BertConfig(vocab_size=len(words),hidden_size=12,num_hidden_layers=1,num_attention_heads=3,intermediate_size=24)
    encoders={key:BertModel(cfg) for key in ('bert_base','hatebert')}
    return frame,tokenizers,encoders


def test_extract_and_cache(tmp_path):
    frame,tokenizers,encoders=inputs(); torch.set_num_threads(1)
    original=frame.copy(deep=True)
    paths=extract_frozen_features(frame,tokenizers,None,tmp_path,encoders=encoders,device='cpu',batch_size=2)
    features=load_features(paths['bert_base'])
    assert features.shape==(3,12) and features.index.tolist()==[0,1,2]
    assert np.isfinite(features.values).all()
    assert all(not p.requires_grad for encoder in encoders.values() for p in encoder.parameters())
    assert all(not encoder.training for encoder in encoders.values())
    pd.testing.assert_frame_equal(frame,original)
    cached=extract_frozen_features(frame,tokenizers,None,tmp_path,encoders=encoders,device='cpu')
    assert paths==cached
    with pytest.raises(ValueError,match='identity differs'):
        extract_frozen_features(frame.iloc[::-1],tokenizers,None,tmp_path,encoders=encoders,device='cpu')
    metadata=json.loads((tmp_path/'bert_base_target_features.json').read_text())
    assert metadata['labels_used'] is False
    with open(paths['bert_base'],'ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_features(paths['bert_base'])


def test_audit_all_rows_and_fail_closed(tmp_path):
    frame,tokenizers,_=inputs(); original=frame.copy(deep=True)
    summary=audit_tokenizers(frame,tokenizers,tmp_path/'good',context_max_length=8,gloss_max_length=6)
    assert summary['examples']==3 and summary['source_unchanged'] and summary['context_input_ids_equal']==3
    pd.testing.assert_frame_equal(frame,original)
    frame.loc[1,'target_start']=5
    with pytest.raises(ValueError,match='audit failed'):
        audit_tokenizers(frame,tokenizers,tmp_path/'bad')
    errors=pd.read_csv(tmp_path/'bad'/'tokenizer_audit_errors.csv')
    assert len(errors)==2 and set(errors.example_id)=={1}
    assert not (tmp_path/'bad'/'tokenizer_audit.json').exists()
