from pathlib import Path
import hashlib
import json

RAW_SHA256 = '02d3e637bd13aac599d94263da36ce8953c09ca65c1fad1fb872f57f39efdcc4'
FOLD_SEED = 20261026
SEEDS = [13, 37, 73, 101, 149]
VARIANTS = ['bert_context', 'hatebert_context', 'full', 'no_gloss', 'concat', 'no_contrastive', 'frozen_context']
CORE_VARIANTS = ['bert_context', 'hatebert_context', 'full', 'no_gloss']
REVISIONS = {
    'bert': {'name': 'google-bert/bert-base-uncased', 'revision': '86b5e0934494bd15c9632b12f734a8a67f723594'},
    'hatebert': {'name': 'GroNLP/hateBERT', 'revision': 'ab7a2d40f6c973cb57e63f23bfa2b51d981d028c'},
}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def code_hash():
    folder = Path(__file__).parent
    return object_hash({p.name: sha256(p) for p in sorted(folder.glob('*.py'))})
