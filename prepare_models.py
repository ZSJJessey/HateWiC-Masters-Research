from pathlib import Path
import sys,os,gc
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from hatewic_formal.tokenization import create_tokenizers
from hatewic_formal.models import build_model
cache=ROOT/'public_model_cache';cache.mkdir(exist_ok=True)
create_tokenizers(cache_dir=str(cache))
model=build_model('full',seed=13,cache_dir=str(cache))
del model;gc.collect()
print('Pinned public models and tokenizers are cached locally.')
