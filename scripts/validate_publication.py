"""Validate the public file collection without private data or training."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','__pycache__','.pytest_cache','.ipynb_checkpoints','artifacts','runs','baselines','frozen_features','verification','pilot','public_model_cache','.runtime','public_plot_preview'}
def files():
    return [p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] not in EXCLUDED and not any(x in {'.git','.venv','__pycache__','.pytest_cache','.ipynb_checkpoints'} for x in p.relative_to(ROOT).parts) and p.name!='private_inputs.json']
def validate():
    failures=[];counts={}
    for p in files():
        rel=p.relative_to(ROOT).as_posix();counts[p.suffix]=counts.get(p.suffix,0)+1
        if p.suffix in {'.pt','.pth','.npz','.zip','.csv'} and p.suffix!='.csv':failures.append(rel+': prohibited artifact type')
        if p.suffix=='.png':continue
        text=p.read_text(encoding='utf-8')
        if re.search(r'[\u4e00-\u9fff]',text):failures.append(rel+': untranslated Chinese text')
        # Build patterns from pieces so this checker does not flag itself.
        if ('/Us'+'ers/') in text:failures.append(rel+': personal absolute path')
        if re.search(r'gh'+'[pousr]_[A-Za-z0-9]{30,}|github'+'_pat_[A-Za-z0-9_]{30,}|hf'+'_[A-Za-z0-9]{25,}|BEGIN '+'(?:RSA |OPENSSH )?PRIVATE KEY',text):failures.append(rel+': possible secret')
        if p.suffix=='.py':ast.parse(text,filename=rel)
        if p.suffix=='.csv':
            header=next(csv.reader(text.splitlines()))
            prohibited={'example_id','annotator_id','annotation_id','example','wiktionary_definition','target_start','target_end','probability','prediction','term','sense_id'}
            if set(header)&prohibited:failures.append(rel+': per-example or lexical data columns')
        if p.suffix=='.ipynb':
            import nbformat
            n=nbformat.reads(text,as_version=4);nbformat.validate(n)
            for i,c in enumerate(n.cells):
                if c.cell_type=='code':
                    if c.outputs or c.execution_count is not None:failures.append(rel+': saved notebook output')
                    if re.search(r'^\s*PAYLOAD\s*=',c.source,re.M):failures.append(rel+': embedded source archive')
            if 'widgets' in n.metadata:failures.append(rel+': widget state')
    schedule=list(csv.DictReader((ROOT/'run_schedule.csv').open()))
    assert len(schedule)==len({r['run_id'] for r in schedule})==460
    assert sum(r['phase']=='single_seed' for r in schedule)==140
    assert sum(r['phase']=='core_repeats' for r in schedule)==320
    for p in (ROOT/'results/tables/core_repeats').glob('primary_tests_holm.csv'):
        tests=list(csv.DictReader(p.open()));assert len(tests)==10
    report={'status':'passed' if not failures else 'failed','files':len(files()),'file_types':counts,'failures':failures,'scope':'Public content, structure, syntax, notebook metadata and schedule. No training or private-data scoring.'}
    print(json.dumps(report,indent=2))
    if failures:raise SystemExit(1)
    return report
if __name__=='__main__':validate()
