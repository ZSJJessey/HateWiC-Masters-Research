"""Validation-selected training with exact optimizer-boundary recovery; no test fitting."""
from __future__ import annotations
import hashlib, json, math, os, random, time
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from .tokenization import tokenize_dataframe

DEFAULT_TRAINING_CONFIG = dict(learning_rate=2e-5,weight_decay=0.01,warmup_ratio=0.1,batch_size=8,
    gradient_accumulation_steps=2,max_epochs=8,patience=2,context_max_length=128,gloss_max_length=64,
    seed=13,checkpoint_every_updates=1,gradient_clip=1.0)


def _json_write(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
    os.replace(temporary,path)


def _torch_write(path,value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def _rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def _seed(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def _macro_f1(truth,prediction):
    scores=[]
    for label in (0,1):
        tp=sum(int(y==label and p==label) for y,p in zip(truth,prediction))
        fp=sum(int(y!=label and p==label) for y,p in zip(truth,prediction))
        fn=sum(int(y==label and p!=label) for y,p in zip(truth,prediction))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores)/2


def _batch(dataset,indices,device):
    return {key:torch.stack([dataset[i][key] for i in indices]).to(device) for key in dataset[0]}


def _autocast(device):
    return torch.autocast(device_type='cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()


def _predict(model,dataset,batch_size,device):
    model.eval(); labels=[]; probabilities=[]
    with torch.no_grad():
        for start in range(0,len(dataset),batch_size):
            batch=_batch(dataset,range(start,min(start+batch_size,len(dataset))),device)
            labels.extend(batch.pop('labels').long().cpu().tolist())
            with _autocast(device):
                logits=model(**batch)['logits']
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return labels,probabilities


def predict_dataframe(model,df,tokenizers,config=None,device=None):
    import pandas as pd
    config={**DEFAULT_TRAINING_CONFIG,**(config or {})}
    device=torch.device(device or next(model.parameters()).device)
    dataset=tokenize_dataframe(df,tokenizers,model.context_name,config['context_max_length'],config['gloss_max_length'],model.uses_gloss)
    truth,probabilities=_predict(model,dataset,config['batch_size'],device)
    return pd.DataFrame(dict(example_id=dataset.example_ids,y_true=truth,y_pred=[int(p>=.5) for p in probabilities],probability=probabilities))


def run_training(train_df,val_df,tokenizers,model,config,out_dir,identity,mode='formal',resume=True,
                 interrupt_after_updates=None):
    """Fit on train, select on validation. An interruption count is relative to this call.

    ``identity`` must describe code version, data hash, and fold hash. It is bound to
    checkpoints together with exact ordered input ID/label/text/span hashes.
    """
    if mode not in ('formal','benchmark','smoke'):
        raise ValueError('Unknown training mode.')
    if mode=='formal' and not torch.cuda.is_available():
        raise RuntimeError('Formal neural training requires a free Colab CUDA runtime.')
    config={**DEFAULT_TRAINING_CONFIG,**config}
    if min(config['batch_size'],config['gradient_accumulation_steps'],config['max_epochs'],config['patience']) < 1:
        raise ValueError('Batch, accumulation, epoch and patience settings must be positive.')
    if config['checkpoint_every_updates'] < 1:
        raise ValueError('Checkpoint frequency must be positive.')
    if not identity or not all(any(term in key.lower() for key in identity) for term in ('code','data','fold')):
        raise ValueError('Identity must contain code, data and fold provenance.')
    if len(train_df)==0 or len(val_df)==0 or set(train_df.example_id)&set(val_df.example_id):
        raise ValueError('Nonempty, disjoint training and validation IDs are required.')
    for frame in (train_df,val_df):
        if 'split' in frame and frame['split'].astype(str).str.lower().eq('test').any():
            raise ValueError('Test rows must not enter training or validation.')
    data_hashes={name:hashlib.sha256(frame.to_json(orient='records').encode()).hexdigest() for name,frame in [('train',train_df),('validation',val_df)]}
    manifest=dict(config=config,identity=identity,model=model.spec,data_hashes=data_hashes,mode=mode)
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True,default=str).encode()).hexdigest()
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    config_path=out/'run_config.json'; restore_path=out/'restore.pt'; best_path=out/'best.pt'; completed_path=out/'completed.json'
    if config_path.exists():
        previous=json.loads(config_path.read_text())
        if previous['config_fingerprint']!=fingerprint:
            raise ValueError('Run directory provenance/configuration differs; use a new run ID.')
        if not resume:
            raise FileExistsError('An existing run must be resumed or assigned a new directory.')
    else:
        _json_write(config_path,{**manifest,'config_fingerprint':fingerprint})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    if completed_path.exists():
        result=json.loads(completed_path.read_text())
        model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model'])
        return result
    _seed(config['seed'])
    train=tokenize_dataframe(train_df,tokenizers,model.context_name,config['context_max_length'],config['gloss_max_length'],model.uses_gloss)
    validation=tokenize_dataframe(val_df,tokenizers,model.context_name,config['context_max_length'],config['gloss_max_length'],model.uses_gloss)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
    batch_count=math.ceil(len(train)/config['batch_size']); accumulation=config['gradient_accumulation_steps']
    total_updates=math.ceil(batch_count/accumulation)*config['max_epochs']; warmup=math.ceil(total_updates*config['warmup_ratio'])
    def lr_factor(step):
        if warmup and step<warmup:
            return (step+1)/warmup
        return max(0.,(total_updates-step)/max(1,total_updates-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    state=dict(epoch=0,next_batch=0,updates=0,best_score=-1.,bad_epochs=0,history=[],epoch_loss_sum=0.,epoch_examples=0,elapsed_seconds=0.)
    if restore_path.exists():
        saved=torch.load(restore_path,map_location=device,weights_only=False)
        if saved['config_fingerprint']!=fingerprint:
            raise ValueError('Recovery checkpoint provenance differs.')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        state=saved['state']; _restore_rng(saved['rng'])
    start_updates=state['updates']; started=time.monotonic(); base_elapsed=state['elapsed_seconds']
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    def save():
        state['elapsed_seconds']=base_elapsed+time.monotonic()-started
        _torch_write(restore_path,dict(config_fingerprint=fingerprint,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),state=state,rng=_rng()))
    while state['epoch']<config['max_epochs'] and state['bad_epochs']<config['patience']:
        epoch=state['epoch']; generator=torch.Generator().manual_seed(config['seed']+epoch*100003)
        order=torch.randperm(len(train),generator=generator).tolist()
        batches=[order[i:i+config['batch_size']] for i in range(0,len(order),config['batch_size'])]
        model.train(); optimizer.zero_grad(set_to_none=True)
        for first in range(state['next_batch'],batch_count,accumulation):
            group=batches[first:first+accumulation]; group_size=sum(map(len,group))
            for indices in group:
                batch=_batch(train,indices,device)
                with _autocast(device):
                    output=model(**batch); loss=output['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss.')
                scaler.scale(loss*(len(indices)/group_size)).backward()
                state['epoch_loss_sum']+=float(loss.detach())*len(indices); state['epoch_examples']+=len(indices)
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['gradient_clip'])
            old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale()>=old_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state['updates']+=1; state['next_batch']=first+len(group)
            if state['updates']%config['checkpoint_every_updates']==0:
                save()
            if interrupt_after_updates is not None and state['updates']-start_updates>=interrupt_after_updates:
                save()
                return dict(status='interrupted',best_checkpoint=str(best_path) if best_path.exists() else None,
                            config_fingerprint=fingerprint,history=state['history'],updates=state['updates'])
        truth,probabilities=_predict(model,validation,config['batch_size'],device)
        score=_macro_f1(truth,[int(p>=.5) for p in probabilities])
        record=dict(epoch=epoch+1,train_loss=state['epoch_loss_sum']/state['epoch_examples'],validation_macro_f1=score,updates=state['updates'])
        state['history'].append(record)
        if score>state['best_score']:
            state['best_score']=score; state['bad_epochs']=0
            _torch_write(best_path,dict(model=model.state_dict(),config_fingerprint=fingerprint,epoch=epoch+1,validation_macro_f1=score,identity=identity))
        else:
            state['bad_epochs']+=1
        state.update(epoch=epoch+1,next_batch=0,epoch_loss_sum=0.,epoch_examples=0)
        save(); _json_write(out/'history.json',state['history'])
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model'])
    selected=min(state['history'],key=lambda item:(-item['validation_macro_f1'],item['epoch']))
    result=dict(status='completed',best_checkpoint=str(best_path),best_epoch=selected['epoch'],best_validation_macro_f1=selected['validation_macro_f1'],selection_rule='highest validation macro F1; earliest epoch on ties',config_fingerprint=fingerprint,history=state['history'],
                updates=state['updates'],elapsed_seconds=state['elapsed_seconds'],peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
                device=str(device),identity=identity)
    _json_write(completed_path,result)
    return result
