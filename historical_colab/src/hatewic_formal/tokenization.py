"""Offset-checked, target-preserving model inputs. Source text is never rewritten."""
from __future__ import annotations
import torch
from .models import BERT_NAME, BERT_REVISION, HATEBERT_NAME, HATEBERT_REVISION


def create_tokenizers(cache_dir=None, local_files_only=False):
    from transformers import AutoTokenizer
    def load(name, revision):
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True,
                                                  cache_dir=cache_dir, local_files_only=local_files_only)
        if not tokenizer.is_fast:
            raise ValueError('A fast tokenizer with character offsets is required.')
        return tokenizer
    return {'bert_base': load(BERT_NAME, BERT_REVISION), 'hatebert': load(HATEBERT_NAME, HATEBERT_REVISION),
            'gloss': load(BERT_NAME, BERT_REVISION)}


def _flag(value):
    return value is True or str(value).lower() in ('true', '1')


def tokenize_context(text, start, end, tokenizer, max_length=128, substitute_u01c3=False):
    start, end = int(start), int(end)
    if not 0 <= start < end <= len(text):
        raise ValueError('Invalid target character span.')
    model_text = text.replace('\u01c3', '!') if substitute_u01c3 else text
    encoded = tokenizer(model_text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offsets, ids = encoded['offset_mapping'], encoded['input_ids']
    selected = [i for i, (a,b) in enumerate(offsets) if a < end and b > start]
    if not selected or selected != list(range(selected[0], selected[-1]+1)):
        raise ValueError('Target tokens are missing or noncontiguous.')
    if any(a < start or b > end for i in selected for a,b in [offsets[i]]):
        raise ValueError('A token crosses the confirmed target boundary.')
    covered = {j for i in selected for j in range(*offsets[i])}
    if any(not model_text[j].isspace() and j not in covered for j in range(start,end)):
        raise ValueError('Tokenizer offsets do not cover the complete target expression.')
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length-specials
    if budget < len(selected):
        raise ValueError('The full target exceeds the context window.')
    left = max(0, selected[0]-(budget-len(selected))//2)
    left = min(left, max(0,len(ids)-budget))
    right = min(len(ids),left+budget)
    window = ids[left:right]
    # Both pinned backbones use BERT's single-sequence CLS/SEP template.
    # Explicit construction avoids fast-tokenizer special-mask API differences.
    if specials != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError('The expected BERT CLS/SEP template is unavailable.')
    assembled = [tokenizer.cls_token_id] + window + [tokenizer.sep_token_id]
    target_mask = [0] + [int(i in selected) for i in range(left,right)] + [0]
    padding = max_length-len(assembled)
    if padding < 0 or tokenizer.padding_side != 'right':
        raise ValueError('Right padding and a valid context length are required.')
    return {'input_ids': assembled+[tokenizer.pad_token_id]*padding,
            'attention_mask': [1]*len(assembled)+[0]*padding,
            'target_mask': target_mask+[0]*padding}


def tokenize_gloss(gloss, tokenizer, max_length=64):
    if not isinstance(gloss,str) or not gloss.strip():
        raise ValueError('A nonempty linked gloss is required for gloss configurations.')
    encoded = tokenizer(gloss, truncation=True, max_length=max_length, padding='max_length',
                        return_special_tokens_mask=True)
    pool = [int(bool(a) and not bool(s)) for a,s in zip(encoded['attention_mask'],encoded['special_tokens_mask'])]
    if not any(pool):
        raise ValueError('The linked gloss contains no poolable tokens.')
    return {'gloss_input_ids': encoded['input_ids'], 'gloss_attention_mask': encoded['attention_mask'], 'gloss_pool_mask': pool}


class TokenizedExamples(torch.utils.data.Dataset):
    def __init__(self, records, example_ids):
        self.records, self.example_ids = records, list(example_ids)
    def __len__(self):
        return len(self.records)
    def __getitem__(self,index):
        return self.records[index]


def tokenize_dataframe(df, tokenizers, context_name='hatebert', context_max_length=128,
                       gloss_max_length=64, include_gloss=True):
    if df.example_id.duplicated().any():
        raise ValueError('Example IDs must be unique.')
    records = []
    for row in df.to_dict('records'):
        try:
            record = tokenize_context(str(row['example']), row['target_start'], row['target_end'],
                                      tokenizers[context_name],context_max_length,
                                      _flag(row.get('needs_U01C3_input_substitution', False)))
            if include_gloss:
                record.update(tokenize_gloss(row['wiktionary_definition'],tokenizers['gloss'],gloss_max_length))
            label = int(row['majority_label'])
            if label not in (0,1):
                raise ValueError('Binary labels must be 0 or 1.')
            records.append({**{k:torch.tensor(v,dtype=torch.long) for k,v in record.items()},
                            'labels':torch.tensor(label,dtype=torch.float32)})
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Tokenization failed for example ID {row['example_id']}: {exc}") from exc
    return TokenizedExamples(records,df.example_id.tolist())
