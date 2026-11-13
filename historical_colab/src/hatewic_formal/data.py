"""Rebuild labels and preserve the provenance of every inclusion decision."""
from pathlib import Path
import json
import unicodedata
import pandas as pd
from .config import RAW_SHA256, sha256, object_hash, write_json

def lexical_key(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())

class Components:
    def __init__(self, ids):
        self.parent = {int(x): int(x) for x in ids}
    def root(self, x):
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def join(self, a, b):
        if int(a) in self.parent and int(b) in self.parent:
            a, b = self.root(a), self.root(b)
            self.parent[max(a,b)] = min(a,b)
    def join_values(self, frame, column):
        for ids in frame.groupby(column, sort=True).example_id:
            values = list(ids[1]); first = values[0]
            for other in values[1:]: self.join(first, other)
    def labels(self, ids):
        return [str(self.root(x)) for x in ids]

def verify_position_sources(raw_path, candidates_path, records_path, rules_path, ledger_path):
    """Read back historical evidence; never rerun or overwrite interactive reviews."""
    raw=pd.read_csv(raw_path,sep=';',keep_default_na=False).drop_duplicates('example_id').set_index('example_id')
    candidates=pd.read_csv(candidates_path,keep_default_na=False).set_index('example_id')
    records=json.loads(Path(records_path).read_text())
    if records['source_sha256']!=RAW_SHA256: raise ValueError('Review records refer to another source.')
    manual=records['records'];rules=pd.read_csv(rules_path,keep_default_na=False).set_index('example_id')
    ledger=pd.read_csv(ledger_path,keep_default_na=False)
    if len(manual)!=177 or len(rules)!=242: raise ValueError('Historical review counts differ.')
    for row in ledger.itertuples():
        example_id=int(row.example_id);status=row.position_status
        if status.startswith('unique_'):
            source=candidates.loc[example_id]
            if source.alignment_status_03!=status or int(source.candidate_count)!=1: raise ValueError('Unique-candidate evidence mismatch.')
            expected=(int(source.target_start_candidate),int(source.target_end_candidate))
        elif status=='manual_confirmed':
            source=manual[str(example_id)]
            if source['decision']!='confirmed': raise ValueError('Manual decision mismatch.')
            expected=(int(source['start']),int(source['end']))
            if raw.loc[example_id,'example'][expected[0]:expected[1]]!=source['surface']: raise ValueError('Manual target surface mismatch.')
        elif status=='repeated_same_surface_rule_candidate':
            source=rules.loc[example_id];expected=(int(source.target_start_candidate),int(source.target_end_candidate))
        else:
            expected_decision={'manual_uncertain':'uncertain','manual_no_identifiable_target':'no_identifiable_target'}.get(status)
            if expected_decision is None or manual[str(example_id)]['decision']!=expected_decision: raise ValueError('Unresolved decision mismatch.')
            continue
        if expected!=(int(row.target_start_candidate),int(row.target_end_candidate)): raise ValueError('Position ledger span differs from evidence.')
    return {'status':'passed','candidate_sha256':sha256(candidates_path),'records_sha256':sha256(records_path),'rules_sha256':sha256(rules_path),'ledger_sha256':sha256(ledger_path)}

def rebuild(raw_path, ledger_path, duplicate_pairs_path, out_dir, *, expected_main=3493):
    paths = {k: Path(v) for k,v in {'raw':raw_path,'ledger':ledger_path,'duplicate_pairs':duplicate_pairs_path}.items()}
    hashes = {k: sha256(v) for k,v in paths.items()}
    if hashes['raw'] != RAW_SHA256: raise ValueError('Raw data fingerprint differs from the approved source.')
    raw = pd.read_csv(paths['raw'], sep=';', keep_default_na=False)
    required = ['example_id','annotator_id','example','term','pos','sense_id','wiktionary_definition','annotation_label','binary_label']
    if not set(required).issubset(raw): raise ValueError('Required source fields are missing.')
    if raw.duplicated(['example_id','annotator_id']).any(): raise ValueError('Repeated example/annotator pair.')
    mapping = {'Not hateful':'0','Weakly hateful':'1','Strongly hateful':'1','Cannot decide':'None'}
    if not raw.annotation_label.map(mapping).eq(raw.binary_label.astype(str)).all(): raise ValueError('Source label mapping mismatch.')
    content = ['example','term','pos','sense_id','wiktionary_definition']
    if raw.groupby('example_id')[content].nunique(dropna=False).gt(1).any().any(): raise ValueError('Inconsistent example contents.')
    votes = pd.crosstab(raw.example_id, raw.binary_label.astype(str)).reindex(columns=['0','1','None'], fill_value=0)
    # Approved reproduction: >=2 agreeing votes, strict winner over the other binary class.
    keep = votes['0'].ne(votes['1']) & (votes['0'].ge(2) | votes['1'].ge(2))
    if 'agree_with_majority_binary' in raw:
        labels=(votes['1']>votes['0']).astype(int)
        expected=[bool(keep.loc[eid]) and str(label)==str(labels.loc[eid]) for eid,label in zip(raw.example_id,raw.binary_label)]
        supplied=raw.agree_with_majority_binary.astype(str).str.lower()
        if not supplied.isin(['true','false']).all() or not supplied.eq('true').eq(expected).all(): raise ValueError('Source majority agreement flags differ.')
    examples = raw.drop_duplicates('example_id')[['example_id']+content].copy()
    examples['annotation_count'] = examples.example_id.map(raw.groupby('example_id').size()).astype(int)
    examples['votes_non_hateful'] = examples.example_id.map(votes['0']).astype(int)
    examples['votes_hateful'] = examples.example_id.map(votes['1']).astype(int)
    examples['majority_label'] = examples.example_id.map((votes['1']>votes['0']).astype(int))
    examples['has_majority'] = examples.example_id.map(keep)
    ledger = pd.read_csv(paths['ledger'], keep_default_na=False)
    retained = examples[examples.has_majority].copy()
    if set(ledger.example_id) != set(retained.example_id) or ledger.example_id.duplicated().any(): raise ValueError('Position ledger ID mismatch.')
    retained = retained.merge(ledger[['example_id','majority_label','annotation_count','position_status','target_start_candidate','target_end_candidate']], on='example_id', validate='one_to_one', suffixes=('','_ledger'))
    for col in ['majority_label','annotation_count']:
        if not retained[col].eq(retained[col+'_ledger']).all(): raise ValueError('Position ledger label/count mismatch.')
    unresolved = retained.position_status.isin(['uncertain','no_identifiable_target','unresolved_no_candidate','manual_uncertain','manual_no_identifiable_target'])
    rules = retained.position_status.eq('rule_assigned_candidate')
    # Allow the exact historical ledger spelling, while refusing any unrecognised evidence tier.
    rule_statuses = [x for x in retained.position_status.unique() if 'rule' in x]
    rules = retained.position_status.isin(rule_statuses)
    accepted = {'unique_literal_candidate','unique_regular_form_candidate','confirmed','manually_confirmed','manual_confirmed'}
    known = accepted | set(rule_statuses) | {'uncertain','no_identifiable_target','unresolved_no_candidate','manual_uncertain','manual_no_identifiable_target'}
    if set(retained.position_status)-known: raise ValueError('Unrecognised position evidence status: '+str(set(retained.position_status)-known))
    retained['exclusion_reasons'] = [json.dumps((["unresolved_position"] if u else [])+(["rule_selected_position"] if r else [])+(["non_three_annotators"] if n!=3 else [])) for u,r,n in zip(unresolved,rules,retained.annotation_count)]
    retained['in_main'] = ~unresolved & ~rules & retained.annotation_count.eq(3)
    main = retained[retained.in_main].copy().sort_values('example_id').reset_index(drop=True)
    if len(main)!=expected_main: raise ValueError(f'Main cohort size {len(main)} differs from approved {expected_main}.')
    main['target_start'] = main.target_start_candidate.astype(int)
    main['target_end'] = main.target_end_candidate.astype(int)
    for row in main.itertuples():
        if not 0<=row.target_start<row.target_end<=len(row.example): raise ValueError('Invalid character span: '+str(row.example_id))
        if not row.example[row.target_start:row.target_end].strip(): raise ValueError('Empty target surface.')
    main['lexical_key'] = main.term.map(lexical_key)
    main['term_pos_key'] = [json.dumps([t,p],ensure_ascii=False) for t,p in zip(main.lexical_key,main.pos)]
    # Source sense IDs are local to the original case-sensitive term and POS.
    for frame in [examples, main]:
        frame['source_sense_key'] = [json.dumps([str(t),str(p),str(s)],ensure_ascii=False) for t,p,s in zip(frame.term,frame.pos,frame.sense_id)]
    main['needs_U01C3_input_substitution'] = main.example_id.eq(1707)
    pairs = pd.read_csv(paths['duplicate_pairs'], keep_default_na=False)
    # Preserve transitive links even when an excluded example bridges two retained examples.
    duplicate = Components(retained.example_id)
    duplicate.join_values(retained, 'example')
    for pair in pairs.itertuples(): duplicate.join(pair.example_id_a,pair.example_id_b)
    main['duplicate_cluster_id'] = duplicate.labels(main.example_id)
    joint = Components(main.example_id)
    joint.join_values(main, 'lexical_key'); joint.join_values(main, 'duplicate_cluster_id')
    main['lexical_cluster_id'] = joint.labels(main.example_id)
    # Scoring-only registry is deliberately written separately from model inputs.
    heterogeneous = main.groupby('source_sense_key').majority_label.nunique().eq(2)
    scoring = main[['example_id','source_sense_key','lexical_cluster_id']].copy()
    scoring['sense_heterogeneous'] = main.source_sense_key.map(heterogeneous).astype(bool)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    retained.to_csv(out/'inclusion_ledger.csv',index=False)
    examples.to_csv(out/'all_examples.csv',index=False)
    main.to_csv(out/'main.csv',index=False)
    scoring.to_csv(out/'scoring_registry.csv',index=False)
    pairs.to_csv(out/'duplicate_pairs_source.csv',index=False)
    audit = {
        'source_sha256':hashes['raw'],'input_hashes':hashes,'input_paths':{k:str(v) for k,v in paths.items()},
        'annotation_rows':len(raw),'original_examples':len(examples),'majority_examples':len(retained),'excluded_no_majority':len(examples)-len(retained),
        'raw_source_senses':int(examples.source_sense_key.nunique()),'main_examples':len(main),'main_label_counts':{str(k):int(v) for k,v in main.majority_label.value_counts().items()},
        'position_status_counts':{str(k):int(v) for k,v in retained.position_status.value_counts().items()},
        'unresolved_positions':int(unresolved.sum()),'rule_selected_positions':int(rules.sum()),'non_three_majority':int(retained.annotation_count.ne(3).sum()),
        'main_source_senses':int(main.source_sense_key.nunique()),'heterogeneous_source_senses':int(heterogeneous.sum()),'heterogeneous_examples':int(scoring.sense_heterogeneous.sum()),
        'main_lexical_groups':int(main.lexical_key.nunique()),'joint_components':int(main.lexical_cluster_id.nunique()),
        'ro1_eligible_term_pos_groups':int(main.groupby(['term_pos_key','majority_label']).size().unstack(fill_value=0).reindex(columns=[0,1],fill_value=0).ge(2).all(axis=1).sum()),
        'files':{name:sha256(out/name) for name in ['inclusion_ledger.csv','all_examples.csv','main.csv','scoring_registry.csv','duplicate_pairs_source.csv']},'audit_status':'passed',
    }
    audit['data_version'] = object_hash(audit['files'])
    write_json(out/'data_audit.json',audit)
    return main, scoring, audit
