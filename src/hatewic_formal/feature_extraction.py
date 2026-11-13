"""Reusable frozen target features and tokenizer audits with explicit provenance."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .models import BERT_NAME, BERT_REVISION, HATEBERT_NAME, HATEBERT_REVISION, masked_mean
from .tokenization import tokenize_dataframe, tokenize_context, tokenize_gloss, _flag

BACKBONES={'bert_base':(BERT_NAME,BERT_REVISION),'hatebert':(HATEBERT_NAME,HATEBERT_REVISION)}


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_hash():
    folder=Path(__file__).parent
    return _digest({name:_file_hash(folder/name) for name in ('feature_extraction.py','tokenization.py','models.py')})


def _tokenizer_hash(tokenizer):
    return _digest(dict(vocabulary=tokenizer.get_vocab(),special_tokens=tokenizer.special_tokens_map,
                        tokenizer_class=type(tokenizer).__name__))


def _ids(main):
    values=[value.item() if isinstance(value,np.generic) else value for value in main.example_id.tolist()]
    if not values:
        raise ValueError('A nonempty input set is required.')
    if len(values)!=len(set(values)):
        raise ValueError('Feature extraction requires unique example IDs.')
    return values


def _write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(temporary,path)


def load_features(path):
    """Verify an archive against its manifest before exposing its indexed features."""
    path=Path(path); manifest_path=path.with_suffix('.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if _file_hash(path)!=manifest['archive_sha256']:
        raise ValueError('Feature archive checksum differs from its manifest.')
    with np.load(path,allow_pickle=False) as saved:
        ids=saved['example_ids']; features=saved['features']
    if ids.tolist()!=manifest['ordered_example_ids'] or features.shape!=tuple(manifest['feature_shape']):
        raise ValueError('Feature IDs or dimensions differ from the manifest.')
    if len(ids)!=len(set(ids.tolist())) or features.ndim!=2 or not np.isfinite(features).all():
        raise ValueError('Feature IDs or values are invalid.')
    return pd.DataFrame(features,index=pd.Index(ids,name='example_id'))


def extract_frozen_features(main,tokenizers,cache_dir,out_dir,encoders=None,batch_size=8,device=None,
                            context_max_length=128,local_files_only=False):
    """Extract each backbone once. Injected encoders are intended for no-download tests.

    Labels may be present in the table but never enter an encoder or objective.
    Cache reuse requires an exact match of data, order, code and tokenizer identity.
    """
    if batch_size<1:
        raise ValueError('Batch size must be positive.')
    ordered_ids=_ids(main); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    shared=dict(ordered_example_ids=ordered_ids,data_sha256=hashlib.sha256(main.to_json(orient='records').encode()).hexdigest(),
                code_sha256=_code_hash(),context_max_length=context_max_length)
    result={}
    for key,(name,revision) in BACKBONES.items():
        path=out/f'{key}_target_features.npz'; metadata_path=path.with_suffix('.json')
        injected_sha256=None
        if encoders and key in encoders:
            state_digest=hashlib.sha256()
            for tensor_name,tensor in sorted(encoders[key].state_dict().items()):
                state_digest.update(tensor_name.encode())
                state_digest.update(str((tuple(tensor.shape),tensor.dtype)).encode())
                state_digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            injected_sha256=state_digest.hexdigest()
        provenance={**shared,'backbone':name,'revision':revision,'tokenizer_sha256':_tokenizer_hash(tokenizers[key]),
                    'injected_encoder':bool(encoders and key in encoders),'injected_weights_sha256':injected_sha256}
        if path.exists() or metadata_path.exists():
            if not (path.exists() and metadata_path.exists()):
                raise ValueError('Incomplete feature cache; preserve it and use a new output directory.')
            previous=json.loads(metadata_path.read_text())
            if any(previous.get(k)!=v for k,v in provenance.items()):
                raise ValueError('Feature cache input/code/model identity differs; use a new output directory.')
            load_features(path)
            result[key]=str(path)
            continue
        if encoders and key in encoders:
            encoder=encoders[key]
        else:
            from transformers import AutoModel
            encoder=AutoModel.from_pretrained(name,revision=revision,cache_dir=cache_dir,local_files_only=local_files_only)
        encoder.to(device).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        frame=main.copy()
        # The tokenizer dataset API carries labels; extraction supplies placeholders.
        frame['majority_label']=0
        dataset=tokenize_dataframe(frame,tokenizers,key,context_max_length,include_gloss=False)
        pieces=[]
        with torch.inference_mode():
            for begin in range(0,len(dataset),batch_size):
                records=[dataset[i] for i in range(begin,min(begin+batch_size,len(dataset)))]
                batch={field:torch.stack([r[field] for r in records]).to(device) for field in ('input_ids','attention_mask','target_mask')}
                hidden=encoder(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).last_hidden_state
                pieces.append(masked_mean(hidden,batch['target_mask']).float().cpu().numpy())
        if not pieces:
            raise ValueError('Feature extraction requires a nonempty input set.')
        features=np.concatenate(pieces,axis=0)
        ids=np.asarray(ordered_ids)
        if ids.dtype.kind=='O':
            raise ValueError('Example IDs must be homogeneous integers or strings.')
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle,example_ids=ids,features=features)
        os.replace(temporary,path)
        _write_json(metadata_path,{**provenance,'feature_shape':list(features.shape),'archive_sha256':_file_hash(path),
                                  'device':str(device),'batch_size':batch_size,'pooling':'last-layer target-subword mean',
                                  'labels_used':False})
        load_features(path); result[key]=str(path)
        del encoder
        if device.type=='cuda':
            torch.cuda.empty_cache()
    return result


def audit_tokenizers(main,tokenizers,out_dir,context_max_length=128,gloss_max_length=64):
    """Validate every target and report clipping without changing source records."""
    before=hashlib.sha256(main.to_json(orient='records').encode()).hexdigest(); _ids(main)
    rows=[]; errors=[]
    for row in main.to_dict('records'):
        record={'example_id':row['example_id']}; contexts={}
        for key in BACKBONES:
            tokenizer=tokenizers[key]
            try:
                text=str(row['example']); substituted=_flag(row.get('needs_U01C3_input_substitution',False))
                model_text=text.replace('\u01c3','!') if substituted else text
                full=tokenizer(model_text,add_special_tokens=True,truncation=False)['input_ids']
                context=tokenize_context(text,row['target_start'],row['target_end'],tokenizer,context_max_length,substituted)
                contexts[key]=context
                record.update({f'{key}_full_context_tokens':len(full),f'{key}_window_tokens':sum(context['attention_mask']),
                               f'{key}_context_truncated':len(full)>context_max_length,
                               f'{key}_target_subwords':sum(context['target_mask'])})
            except (ValueError,TypeError,KeyError) as error:
                errors.append({'example_id':row['example_id'],'tokenizer':key,'error':str(error)})
        gloss=row.get('wiktionary_definition','')
        if isinstance(gloss,str) and gloss.strip():
            try:
                full=tokenizers['gloss'](gloss,add_special_tokens=True,truncation=False)['input_ids']
                encoded=tokenize_gloss(gloss,tokenizers['gloss'],gloss_max_length)
                record.update(gloss_available=True,gloss_full_tokens=len(full),gloss_window_tokens=sum(encoded['gloss_attention_mask']),
                              gloss_truncated=len(full)>gloss_max_length)
            except (ValueError,TypeError,KeyError) as error:
                errors.append({'example_id':row['example_id'],'tokenizer':'gloss','error':str(error)})
        else:
            record.update(gloss_available=False,gloss_full_tokens=0,gloss_window_tokens=0,gloss_truncated=False)
        if len(contexts)==2:
            record['context_input_ids_equal']=contexts['bert_base']['input_ids']==contexts['hatebert']['input_ids']
            record['context_target_masks_equal']=contexts['bert_base']['target_mask']==contexts['hatebert']['target_mask']
        rows.append(record)
    after=hashlib.sha256(main.to_json(orient='records').encode()).hexdigest()
    if before!=after:
        raise RuntimeError('Source dataframe was modified during tokenizer auditing.')
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(errors,columns=['example_id','tokenizer','error']).to_csv(out/'tokenizer_audit_errors.csv',index=False)
    if errors:
        _write_json(out/'tokenizer_audit_failed.json',dict(status='failed',errors=len(errors),data_sha256=before,code_sha256=_code_hash()))
        raise ValueError(f'Tokenizer audit failed for {len(errors)} records; formal extraction is blocked.')
    table=pd.DataFrame(rows); table.to_csv(out/'tokenizer_audit.csv',index=False)
    summary=dict(status='passed',examples=len(table),source_unchanged=True,data_sha256=before,code_sha256=_code_hash(),
                 tokenizer_sha256={key:_tokenizer_hash(tokenizers[key]) for key in (*BACKBONES,'gloss')},
                 context_max_length=context_max_length,gloss_max_length=gloss_max_length,
                 context_input_ids_equal=int(table.context_input_ids_equal.sum()),context_target_masks_equal=int(table.context_target_masks_equal.sum()),
                 truncated_contexts={key:int(table[f'{key}_context_truncated'].sum()) for key in BACKBONES},
                 gloss_available=int(table.gloss_available.sum()),gloss_truncated=int(table.gloss_truncated.sum()))
    _write_json(out/'tokenizer_audit.json',summary)
    return summary
