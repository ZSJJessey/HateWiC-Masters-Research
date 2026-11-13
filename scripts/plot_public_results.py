"""Recreate figures from public aggregate tables only; never read private examples."""
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'public_plot_preview';out.mkdir(exist_ok=True)
    core=pd.read_csv(ROOT/'results/tables/core_repeats/metrics.csv')
    labels={'bert_context':'BERT context','hatebert_context':'HateBERT context','full':'Full','no_gloss':'No gloss'}
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(labels))
    for i,protocol in enumerate(['random','term_disjoint']):
        groups=[core[(core.protocol==protocol)&(core.model==m)&(core.subgroup=='overall')].macro_f1 for m in labels]
        ax.bar(x+(i-.5)*.36,[g.mean() for g in groups],.36,yerr=[g.std(ddof=1) for g in groups],capsize=3,label=protocol)
    ax.set_xticks(x,list(labels.values()));ax.set_ylabel('Macro F1');ax.legend();fig.tight_layout();fig.savefig(out/'core_macro_f1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,protocol in zip(axes,['random','term_disjoint']):
        for i,m in enumerate(['no_gloss','full']):
            values=[core[(core.protocol==protocol)&(core.model==m)&(core.subgroup==g)].macro_f1.mean() for g in ['homogeneous','heterogeneous']]
            ax.bar(np.arange(2)+(i-.5)*.36,values,.36,label=m)
        ax.set_xticks([0,1],['Homogeneous','Heterogeneous']);ax.set_title(protocol);ax.legend()
    axes[0].set_ylabel('Macro F1');fig.tight_layout();fig.savefig(out/'subgroup_macro_f1.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7,6))
    for row,protocol in enumerate(['random','term_disjoint']):
        for col,m in enumerate(['no_gloss','full']):
            block=core[(core.protocol==protocol)&(core.model==m)&(core.subgroup=='overall')]
            matrix=np.mean([np.array(ast.literal_eval(v)) for v in block.confusion_matrix],axis=0)
            ax=axes[row,col];ax.imshow(matrix,cmap='Blues');ax.set_title(protocol+' / '+m)
            ax.set_xlabel('Predicted');ax.set_ylabel('True');ax.set_xticks([0,1]);ax.set_yticks([0,1])
            for a in range(2):
                for b in range(2):ax.text(b,a,f'{matrix[a,b]:.1f}',ha='center',va='center')
    fig.tight_layout();fig.savefig(out/'mean_confusion.png',dpi=180);plt.close(fig)
    geometry=pd.read_csv(ROOT/'results/tables/ro1/primary_tests_uncorrected.csv')
    fig,ax=plt.subplots(figsize=(7,3));y=np.arange(len(geometry));ax.errorbar(geometry.estimate,y,xerr=np.stack([geometry.estimate-geometry.ci_low,geometry.ci_high-geometry.estimate]),fmt='o',capsize=3);ax.axvline(0,color='gray',ls='--');ax.set_yticks(y,geometry.metric);ax.set_xlabel('Effect favouring HateBERT');fig.tight_layout();fig.savefig(out/'ro1_paired_effects.png',dpi=180);plt.close(fig)
    baseline=pd.concat([pd.read_csv(p) for p in (ROOT/'results/tables/baselines').glob('*.csv')]);baseline=baseline[baseline.subgroup=='overall']
    models=baseline.model.unique();fig,ax=plt.subplots(figsize=(9,4))
    for i,protocol in enumerate(['random','term_disjoint']):
        values=[baseline[(baseline.protocol==protocol)&(baseline.model==m)].macro_f1.mean() for m in models]
        ax.bar(np.arange(len(models))+(i-.5)*.36,values,.36,label=protocol)
    ax.set_xticks(np.arange(len(models)),models,rotation=15,ha='right');ax.set_ylabel('Macro F1');ax.legend();fig.tight_layout();fig.savefig(out/'baseline_macro_f1.png',dpi=180);plt.close(fig)
    print('Aggregate-only preview figures saved:',out)
if __name__=='__main__':main()
