import pandas as pd
from hatewic_formal.baselines import lexical_prior
from hatewic_formal.data import lexical_key, Components
from hatewic_formal.folds import create_folds

def test_prior_tie_uses_training_majority():
    train=pd.DataFrame({'lexical_key':['tie','tie','b','c','d'],'majority_label':[0,1,1,1,0]})
    pred,_=lexical_prior(train,pd.DataFrame({'lexical_key':['tie','unseen']}))
    assert pred.tolist()==[1,1]

def test_lexical_key_omits_pos_and_case():
    assert lexical_key('  Ｔｅｒｍ  WORD ')=='term word'

def test_transitive_components():
    components=Components([1,2,3,4]);components.join(1,2);components.join(2,3)
    assert components.labels([1,2,3,4])==['1','1','1','4']

def test_rotations_and_no_term_leakage(tmp_path):
    frame=pd.DataFrame({'example_id':range(80),'majority_label':[0,1]*40,'duplicate_cluster_id':[str(i) for i in range(80)],'lexical_key':[str(i//2) for i in range(80)],'lexical_cluster_id':[str(i//2) for i in range(80)]})
    manifest=create_folds(frame,tmp_path)
    for protocol,rows in manifest.groupby('protocol'):
        assert len(rows)==800
        assert rows[rows.split.eq('test')].example_id.value_counts().eq(1).all()
    oo=manifest[manifest.protocol.eq('term_disjoint')]
    assert oo.groupby(['rotation','lexical_key']).split.nunique().max()==1
