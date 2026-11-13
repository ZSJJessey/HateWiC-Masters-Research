"""Update only evidence-supported chapter tables; absent experiments remain pending."""
from pathlib import Path
import os,sys,json,csv,re,subprocess,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from hatewic_formal.reporting import collect_predictions,final_reports
from hatewic_formal.evaluation import summarize_oof
from hatewic_formal.config import write_json,sha256
from hatewic_formal.runner import load_artifacts
NAMES={'bert_context':'BERT context','hatebert_context':'HateBERT context','full':'Full dual encoder','no_gloss':'No gloss','concat':'Simple concatenation','no_contrastive':'No contrastive loss','frozen_context':'Frozen context encoder'}
DOC_PYTHON=os.environ.get('HATEWIC_DOC_PYTHON',sys.executable)

def verify_phase_cache(phase,accepted,identity):
    folder=ROOT/'results'/phase
    lineage=json.loads((folder/'local_provenance.json').read_text())
    if lineage.get('identity')!=identity or lineage.get('accepted_prediction_hashes')!=accepted:
        raise RuntimeError('Cached phase belongs to different accepted predictions or execution identity.')
    evidence=json.loads((folder/'evidence.json').read_text())
    if evidence.get('status')!='complete_for_requested_phase':raise RuntimeError('Phase evidence is incomplete.')
    for name,digest in evidence['files'].items():
        if sha256(folder/name)!=digest:raise RuntimeError('Phase result fingerprint differs: '+name)
    return True

def phase_marker_hashes(phase,accepted):
    from execute_formal import rows
    from hatewic_formal.config import CORE_VARIANTS
    wanted=[job['run_id'] for job in rows() if (job['phase']=='single_seed' if phase=='single_seed' else job['model'] in CORE_VARIANTS)]
    if any(rid not in accepted for rid in wanted):raise RuntimeError('The requested phase does not have complete matched predictions.')
    return {rid:sha256(ROOT/'runs'/rid/'predictions.csv') for rid in sorted(wanted)}

def replace_section(text,a,b,new):
    start=text.index(a);end=text.index(b,start);return text[:start]+new.rstrip()+'\n\n'+text[end:]

def score(rows,model,protocol,subgroup='overall'):
    found=rows[rows.model.eq(model)&rows.protocol.eq(protocol)&rows.subgroup.eq(subgroup)]
    if found.empty:return 'Pending'
    values=found.macro_f1.dropna()
    if len(values)!=len(found):return 'Insufficient support'
    return f'{values.mean():.6f}' if len(values)==1 else f'{values.mean():.6f} ± {values.std(ddof=1):.6f}'

def mean(rows,model,protocol,subgroup='overall'):
    v=rows[rows.model.eq(model)&rows.protocol.eq(protocol)&rows.subgroup.eq(subgroup)].macro_f1
    return None if v.empty or v.isna().any() else float(v.mean())

def error_material(predictions,main,phase):
    errors=[];summary=[]
    for protocol in ['random','term_disjoint']:
        part=predictions[predictions.protocol.eq(protocol)&predictions.model.isin(['full','no_gloss'])]
        if set(part.model)!=set(['full','no_gloss']):continue
        for (model,seed),block in part.groupby(['model','seed']):
            for truth in [0,1]:
                group=block[block.majority_label.eq(truth)]
                summary.append(dict(protocol=protocol,model=model,seed=int(seed),truth=truth,support=len(group),errors=int(group.prediction.ne(group.majority_label).sum())))
        outcomes=part.assign(correct=part.prediction.eq(part.majority_label)).groupby(['example_id','model']).correct.mean().unstack('model')
        for category,mask in [('gloss_consistently_corrects',outcomes.full.eq(1)&outcomes.no_gloss.eq(0)),('gloss_consistently_harms',outcomes.full.eq(0)&outcomes.no_gloss.eq(1)),('both_consistently_wrong',outcomes.full.eq(0)&outcomes.no_gloss.eq(0))]:
            ids=sorted(outcomes.index[mask].tolist())[:10]
            cases=main[main.example_id.isin(ids)].copy()
            for record in cases[['example_id','example','term','pos','wiktionary_definition','majority_label','target_start','target_end']].to_dict('records'):
                errors.append({**record,'protocol':protocol,'category':category,'review_status':'pending_qualitative_review','review_note':''})
    out=ROOT/'results'/phase;out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(summary).to_csv(out/'error_counts_by_seed.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'restricted_error_review.csv',index=False)
    return summary

def make_neural_figure(metrics):
    if metrics.empty:return False
    overall=metrics[metrics.subgroup.eq('overall')]
    models=[m for m in NAMES if set(overall[overall.model.eq(m)].protocol)=={'random','term_disjoint'}]
    if not models:return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(len(models));width=.36
    multiple=overall.seed.nunique()>1
    for j,(protocol,color,label) in enumerate([('random','#23668c','Random'),('term_disjoint','#d58a04','Lexical-disjoint')]):
        groups=[overall[overall.model.eq(m)&overall.protocol.eq(protocol)].macro_f1 for m in models]
        vals=[g.mean() for g in groups];errors=[g.std(ddof=1) if len(g)>1 else 0 for g in groups]
        bars=ax.bar(x+(j-.5)*width,vals,width,label=label,color=color,yerr=errors if multiple else None,capsize=3)
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
    ax.set_xticks(x,[NAMES[m].replace(' ','\n',1) for m in models]);ax.set_ylim(0,1.05);ax.set_ylabel('Pooled out-of-fold macro F1')
    ax.set_title('Complete neural comparisons on the frozen 3,493-example main set');ax.legend(frameon=False,ncol=2)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    folder=ROOT/'results/figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(folder/('neural_oof_macro_f1.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig);return True


def main(phase=None):
    from execute_formal import rows,completed
    # The generic OOF library does not know this local backend identity.
    # Validate every marker and its full frozen-test contents before collection.
    accepted=completed(rows())
    identity=json.loads((ROOT/'verification/formal_execution_identity.json').read_text())
    main_df,_,audit,_=load_artifacts(ROOT)
    if phase:
        marker_hashes=phase_marker_hashes(phase,accepted)
        final_reports(ROOT,phase=phase)
        write_json(ROOT/'results'/phase/'local_provenance.json',dict(identity=identity,accepted_prediction_hashes=marker_hashes))
    predictions=collect_predictions(ROOT)
    chosen='core_repeats' if (ROOT/'results/core_repeats/evidence.json').exists() else ('single_seed' if (ROOT/'results/single_seed/evidence.json').exists() else None)
    if chosen:
        verify_phase_cache(chosen,phase_marker_hashes(chosen,accepted),identity)
        predictions=pd.read_csv(ROOT/'results'/chosen/'neural_oof_predictions.csv')
        metrics=pd.read_csv(ROOT/'results'/chosen/'metrics.csv')
    elif predictions.empty:
        metrics=pd.DataFrame(columns=['protocol','model','seed','subgroup','macro_f1'])
    else:
        # Only seed 13 is shown until the full five-seed core matrix is complete.
        predictions=predictions[predictions.seed.eq(13)]
        metrics=summarize_oof(predictions,pd.read_csv(ROOT/'artifacts/scoring_registry.csv'))
    interim=ROOT/'results/progress';interim.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(interim/'complete_oof_metrics.csv',index=False)
    has_neural_figure=make_neural_figure(metrics)
    text=(ROOT/'docs/chapter4_report_template.md').read_text()
    text=text.replace('This stage reports representation geometry and five baselines. Fine-tuned neural and linked-gloss comparisons remain pending.',
        'This stage reports verified representation geometry, five baselines and any completed full-protocol neural predictions. A neural configuration appears only after all ten test folds have been verified. Remaining results are explicitly marked pending.')
    marker='### 4.2.4 Metrics and inference'
    method='Formal neural comparisons use the Apple M5 MPS backend in float32 with PyTorch 2.14.1. The earlier CUDA autocast pilot and two CUDA fold results are retained separately and excluded from the primary local neural matrix. Fixed data, folds, model revisions, batch settings and validation selection are unchanged. MPS recovery saves device RNG state and is checked with an explicit 1e-7 absolute numerical tolerance; this does not assert bitwise identity across devices. A mathematically equivalent masked eligible-anchor mean replaces boolean indexing in the contrastive term for deterministic MPS compatibility.\n\n'
    text=text.replace(marker,method+marker)
    classical=pd.concat([pd.read_csv(ROOT/'baselines/classical/metrics.csv'),pd.read_csv(ROOT/'baselines/frozen_mlp/metrics.csv')])
    labels={'majority':'Training majority','lexical_prior':'Training lexical prior','tfidf_lr':'TF-IDF and LR','bert_frozen_mlp':'BERT frozen features and MLP','hatebert_frozen_mlp':'HateBERT frozen features and MLP'}
    lines=['## 4.4 RO2 Target hateful sense classification','',
      'Five baseline configurations have complete matched out-of-fold predictions on all 3,493 main examples under both protocols. Scaling and vocabulary fitting use each training partition alone. The frozen-feature baselines retain the original extraction and execution provenance.','',
      '| Baseline | Random macro F1 | Lexical-disjoint macro F1 |','|---|---:|---:|']
    for model,name in labels.items():lines.append(f'| {name} | {score(classical,model,"random")} | {score(classical,model,"term_disjoint")} |')
    lines+=['','Neural results use matched full-protocol predictions. Values are single-seed pooled macro F1 until the five-seed core matrix is complete; five-seed entries are means ± seed standard deviations. Seed standard deviation is not a confidence interval.','',
      '| Neural model | Random macro F1 | Lexical-disjoint macro F1 |','|---|---:|---:|']
    for model,name in NAMES.items():
        source=metrics
        if chosen=='core_repeats' and model not in ['bert_context','hatebert_context','full','no_gloss']:
            source=pd.read_csv(ROOT/'results/single_seed/metrics.csv')
        lines.append(f'| {name} | {score(source,model,"random")} | {score(source,model,"term_disjoint")} |')
    lines+=['','Single-seed ablations remain supporting results when the core models have five seeds. No per-fold partial score is substituted for a complete out-of-fold result. The fixed threshold is 0.5 and epoch selection uses validation macro F1 only.','',
       'Evidence: `baselines/classical/metrics.csv`, `baselines/frozen_mlp/metrics.csv`, `results/progress/complete_oof_metrics.csv` and the per-run prediction fingerprints.']
    lines+=['','![Figure 4.2. Fixed baseline macro F1 on matched out-of-fold predictions.](results/figures/baseline_macro_f1.png)']
    if has_neural_figure:lines+=['','![Figure 4.3. Complete neural out-of-fold macro F1; error bars, when present, show seed standard deviations.](results/figures/neural_oof_macro_f1.png)']
    if metrics.empty:lines+=['','Fine-tuned model results: Pending experimental results.']
    text=replace_section(text,'## 4.4 ','## 4.5 ','\n'.join(lines))
    lines=['## 4.5 RO3 Linked definitions and subgroup performance','',
      'The full model and no-gloss ablation share the context encoder and supervised contrastive loss. Their difference isolates the linked-gloss branch under matched training settings. Sense heterogeneity is defined by both majority labels occurring within the original term, POS and local sense identifier in the frozen main set. The registry is scoring-only and never enters model inputs, loss, early stopping or parameter selection.','']
    for protocol,label in [('random','Random'),('term_disjoint','Lexical-disjoint')]:
        lines += [f'### {label} protocol','', '| Group | No-gloss macro F1 | Full-model macro F1 |','|---|---:|---:|']
        for subgroup in ['overall','homogeneous','heterogeneous']:lines.append(f'| {subgroup.title()} | {score(metrics,"no_gloss",protocol,subgroup)} | {score(metrics,"full",protocol,subgroup)} |')
        values={m:{g:mean(metrics,m,protocol,g) for g in ['overall','homogeneous','heterogeneous']} for m in ['no_gloss','full']}
        if all(v is not None for vals in values.values() for v in vals.values()):
            a,b=values['no_gloss'],values['full'];gap_a=a['homogeneous']-a['heterogeneous'];gap_b=b['homogeneous']-b['heterogeneous']
            lines+=['',f'The overall gloss gain is {b["overall"]-a["overall"]:.6f}. The homogeneous-minus-heterogeneous gap is {gap_a:.6f} without glosses and {gap_b:.6f} with glosses; the gap reduction is {gap_a-gap_b:.6f}. The heterogeneous-group absolute change is {b["heterogeneous"]-a["heterogeneous"]:.6f}, and the homogeneous-group absolute change is {b["homogeneous"]-a["homogeneous"]:.6f}. These absolute changes accompany the gap to distinguish difficult-group improvement from deterioration of the reference group.']
        else:lines+=['','Matched gloss comparison: Pending experimental results.']
        lines+=['']
    vals=[mean(metrics,m,p) for m in ['no_gloss','full'] for p in ['random','term_disjoint']]
    if all(v is not None for v in vals):
        ar,at,br,bt=vals;lines +=[f'The cross-protocol gap is {ar-at:.6f} without glosses and {br-bt:.6f} with glosses, a reduction of {(ar-at)-(br-bt):.6f}. This comparison uses the same tested example IDs but different training conditions; it is not an intrinsic seen-versus-unseen contrast within one fitted model.','']
    lines+=['All main examples have a linked source gloss. These results therefore concern classification with an available linked sense definition; they do not establish automatic sense retrieval at deployment.']
    text=replace_section(text,'## 4.5 ','## 4.6 ','\n'.join(lines))
    single_path=ROOT/'results/single_seed/metrics.csv'
    single=pd.read_csv(single_path) if single_path.exists() else metrics
    lines=['## 4.6 Ablation analysis','',
      'The fixed single-seed ablations remove the gloss branch, replace structured interaction with concatenation, remove supervised contrastive loss or freeze the context encoder. The no-gloss configuration retains contrastive loss. A positive difference below favors the full model.','',
      '| Full minus ablation | Random macro F1 difference | Lexical-disjoint difference |','|---|---:|---:|']
    for model in ['no_gloss','concat','no_contrastive','frozen_context']:
        values=[]
        for protocol in ['random','term_disjoint']:
            full,a=mean(single,'full',protocol),mean(single,model,protocol)
            values.append('Pending' if full is None or a is None else f'{full-a:.6f}')
        lines.append(f'| {NAMES[model]} | {values[0]} | {values[1]} |')
    lines+=['','Ablations are matched descriptive comparisons. They are not additional uncorrected confirmatory tests.']
    text=replace_section(text,'## 4.6 ','## 4.7 ','\n'.join(lines))
    error_phase=chosen or 'progress';err=error_material(predictions,main_df,error_phase) if not predictions.empty else []
    lines=['## 4.7 Error analysis and discussion','']
    if err:
        errors=pd.DataFrame(err)
        lines+=['Error counts are recorded separately for each seed and true class. The following values are mean counts across fitted seeds, not independent sample replicates.','',
          '| Protocol and model | False positives | False negatives |','|---|---:|---:|']
        for (protocol,model),group in errors.groupby(['protocol','model']):
            fp=group[group.truth.eq(0)].errors.mean();fn=group[group.truth.eq(1)].errors.mean()
            lines.append(f'| {protocol} {NAMES[model]} | {fp:.2f} | {fn:.2f} |')
        lines+=['','A deterministic restricted review sheet samples up to ten examples per protocol for gloss-corrected errors, gloss-introduced errors and persistent shared errors. For multi-seed reports these categories require consistent outcomes across every fitted seed. Semantic interpretation of these cases remains pending qualitative review; no automatically inferred explanation is presented as an annotated finding.']
    else:lines+=['Pending experimental results. Error interpretation awaits complete matched out-of-fold predictions.']
    if chosen:
        tests=pd.read_csv(ROOT/'results'/chosen/'primary_tests_holm.csv')
        lines+=['','## 4.8 Prespecified paired inference','',
          'The primary inferential report uses the five-seed core matrix when available. The first-pass seed-13 report is provisional; ablation scores remain descriptive. Each phase preserves the ten prespecified tests, matched cluster resampling and conditional fitted-model interpretation.','',
          '| Prespecified comparison | Effect and 95 percent cluster interval | Holm adjusted p |','|---|---|---:|']
        for row in tests.to_dict('records'):
            supported=row.get('inferential',False)
            ci=f"{row['estimate']:.6f} [{row['ci_low']:.6f}, {row['ci_high']:.6f}]" if supported else 'Descriptive or unsupported'
            adjusted=row.get('p_holm',row.get('p_value_holm',row.get('adjusted_p_value')))
            p='Not estimated' if not supported else ('Pending' if adjusted is None or pd.isna(adjusted) else f'{adjusted:.6f}')
            lines.append(f"| {row['test_id'].replace('_',' ')} | {ci} | {p} |")
        lines+=['','The confidence intervals condition on the fitted models and exclude retraining uncertainty. Folds with overlapping training sets are not treated as independent replicates. Non-positive point estimates and unsupported comparisons are retained.']
    text=replace_section(text,'## 4.7 ','## Evidence sources','\n'.join(lines))
    text+='\n\nLocal execution evidence: `verification/local_pilot.json`, `verification/tests.json`, `artifacts/reused_results_provenance.json`, `run_progress.json` and the per-run identities.\n'
    (ROOT/'chapter4.md').write_text(text)
    # Source-index fingerprints accompany every data table; specific row numbers are in the CSVs.
    source_files=[p for p in (ROOT/'results').rglob('*.csv') if p.name!='restricted_error_review.csv']+[ROOT/'artifacts/data_audit.json',ROOT/'artifacts/fold_audit.json',ROOT/'artifacts/tokenizer_audit.json']
    write_json(ROOT/'results/chapter4_source_files.json',{str(p.relative_to(ROOT)):sha256(p) for p in source_files})
    subprocess.run([DOC_PYTHON,str(ROOT/'build_chapter4.py')],cwd=ROOT,check=True,capture_output=True,text=True)
    print('Chapter 4 stage updated; completed OOF blocks:',0 if metrics.empty else metrics[['protocol','model','seed']].drop_duplicates().shape[0],flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['single_seed','core_repeats']);args=parser.parse_args();main(args.phase)
