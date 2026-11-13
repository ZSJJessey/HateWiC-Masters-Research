"""Training-only classical estimators and frozen-feature MLP comparators."""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .config import code_hash, sha256, write_json, object_hash
from .evaluation import binary_metrics, validate_oof, summarize_oof

def majority_class(y):
    counts=np.bincount(np.asarray(y,dtype=int),minlength=2)
    return int(counts[1]>counts[0])

def lexical_prior(train, query):
    fallback=majority_class(train.majority_label)
    key='term_pos_key' if 'term_pos_key' in train else 'lexical_key'
    counts=train.groupby([key,'majority_label']).size().unstack(fill_value=0).reindex(columns=[0,1],fill_value=0)
    table={term:(fallback if row[0]==row[1] else int(row[1]>row[0])) for term,row in counts.iterrows()}
    prediction=np.array([table.get(term,fallback) for term in query[key]],dtype=int)
    return prediction,prediction.astype(float)

def fit_predict(model_name, train, query, seed=13, features=None):
    if model_name=='majority':
        pred=np.full(len(query),majority_class(train.majority_label),dtype=int)
        return pred,pred.astype(float),[]
    if model_name=='lexical_prior':
        pred,prob=lexical_prior(train,query)
        return pred,prob,[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if model_name=='tfidf_lr':
            estimator=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True,strip_accents='unicode',token_pattern=r'(?u)\b\w+\b')),
                                ('classifier',LogisticRegression(C=1.,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=seed))])
            estimator.fit(train.example,train.majority_label)
            probability=estimator.predict_proba(query.example)[:,1]
        elif model_name in ('bert_frozen_mlp','hatebert_frozen_mlp'):
            if features is None: raise ValueError('Verified frozen target features are required.')
            # Scaling is fitted on the current training partition only.
            estimator=Pipeline([('scale',StandardScaler()),('classifier',MLPClassifier(hidden_layer_sizes=(300,200,100,50),learning_rate_init=.0005,max_iter=10,early_stopping=False,random_state=seed))])
            estimator.fit(features.loc[train.example_id].to_numpy(),train.majority_label)
            probability=estimator.predict_proba(features.loc[query.example_id].to_numpy())[:,1]
        else: raise ValueError('Unknown baseline: '+model_name)
    return (probability>=.5).astype(int),probability,[str(w.message) for w in caught]

def run_baselines(main, manifest, out_dir, data_version, fold_hash, models=None, features_by_backbone=None):
    models=models or ['majority','lexical_prior','tfidf_lr']
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for protocol in ['random','term_disjoint']:
        for fold in range(10):
            part=manifest[manifest.protocol.eq(protocol)&manifest.rotation.eq(fold)]
            train=main[main.example_id.isin(part.loc[part.split.eq('train'),'example_id'])]
            test=main[main.example_id.isin(part.loc[part.split.eq('test'),'example_id'])]
            for model in models:
                run_id=f'{protocol}__{model}__s13__f{fold:02d}'
                folder=out/run_id;folder.mkdir(exist_ok=True)
                identity={'run_id':run_id,'model':model,'seed':13,'fold':fold,'protocol':protocol,'data_version':data_version,'fold_sha256':fold_hash,'code_sha256':code_hash(),'train_ids':train.example_id.tolist(),'test_ids':test.example_id.tolist(),'selection':'fixed settings; no test-guided selection'}
                identity_hash=object_hash(identity)
                if (folder/'completed.json').is_file():
                    saved=pd.read_csv(folder/'predictions.csv')
                    info=__import__('json').loads((folder/'completed.json').read_text())
                    if info['identity_sha256']!=identity_hash or info['predictions_sha256']!=sha256(folder/'predictions.csv'): raise ValueError('Completed baseline provenance changed.')
                    outputs.append(saved);continue
                start=time.monotonic()
                features=None
                if '_frozen_mlp' in model:
                    features=(features_by_backbone or {}).get(model.split('_')[0])
                prediction,probability,notes=fit_predict(model,train,test,features=features)
                frame=pd.DataFrame({'example_id':test.example_id.to_numpy(),'majority_label':test.majority_label.to_numpy(),'prediction':prediction,'probability':probability,'fold':fold,'protocol':protocol,'model':model,'seed':13})
                frame.to_csv(folder/'predictions.csv',index=False)
                write_json(folder/'run_config.json',identity)
                write_json(folder/'completed.json',{'status':'completed','identity_sha256':identity_hash,'predictions_sha256':sha256(folder/'predictions.csv'),'elapsed_seconds':time.monotonic()-start,'warnings':notes,'metrics':binary_metrics(frame.majority_label,frame.prediction)})
                outputs.append(frame)
    combined=pd.concat(outputs,ignore_index=True)
    validate_oof(combined,manifest,main.example_id)
    combined.to_csv(out/'oof_predictions.csv',index=False)
    write_json(out/'oof_validation.json',validate_oof(combined,manifest,main.example_id))
    return combined
