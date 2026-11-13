"""Two fixed ten-fold protocols; no training or score-guided split search."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .config import FOLD_SEED, sha256, write_json

def create_folds(main, out_dir):
    manifests=[]; checks=[]
    for protocol, group_col in [('random','duplicate_cluster_id'),('term_disjoint','lexical_cluster_id')]:
        assign = pd.Series(-1,index=main.index,dtype=int)
        splitter=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=FOLD_SEED)
        for fold,(_,test_idx) in enumerate(splitter.split(main,main.majority_label,groups=main[group_col])):
            assign.iloc[test_idx]=fold
        if assign.lt(0).any(): raise ValueError('Incomplete folds.')
        for rotation in range(10):
            part=main[['example_id','majority_label','duplicate_cluster_id','lexical_key','lexical_cluster_id']].copy()
            part['protocol']=protocol;part['rotation']=rotation;part['assigned_fold']=assign
            part['split']=['test' if x==rotation else 'validation' if x==(rotation+1)%10 else 'train' for x in assign]
            if part.groupby(group_col).split.nunique().max()!=1: raise ValueError('Group leakage.')
            if part.groupby('duplicate_cluster_id').split.nunique().max()!=1: raise ValueError('Duplicate leakage.')
            if protocol=='term_disjoint' and part.groupby('lexical_key').split.nunique().max()!=1: raise ValueError('Term leakage.')
            train_terms=set(part.loc[part.split.eq('train'),'lexical_key'])
            part['term_seen_in_train']=part.lexical_key.isin(train_terms)
            for split, rows in part.groupby('split'):
                counts=rows.majority_label.value_counts()
                if set(counts.index)!={0,1}: raise ValueError('A partition is missing one class.')
                checks.append({'protocol':protocol,'rotation':rotation,'split':split,'examples':len(rows),'non_hateful':int(counts[0]),'hateful':int(counts[1])})
            manifests.append(part)
    manifest=pd.concat(manifests,ignore_index=True).sort_values(['protocol','rotation','example_id'])
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(out/'fold_manifest.csv',index=False);pd.DataFrame(checks).to_csv(out/'fold_support.csv',index=False)
    write_json(out/'fold_audit.json',{'seed':FOLD_SEED,'protocols':['random','term_disjoint'],'rotations_per_protocol':10,'main_examples':len(main),'manifest_sha256':sha256(out/'fold_manifest.csv'),'status':'passed','class_support':checks})
    return manifest
