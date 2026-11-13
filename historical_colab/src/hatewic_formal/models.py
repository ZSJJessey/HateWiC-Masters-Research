"""Matched target-token classifiers for the preregistered HateWiC comparisons."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

BERT_NAME = 'google-bert/bert-base-uncased'
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
HATEBERT_NAME = 'GroNLP/hateBERT'
HATEBERT_REVISION = 'ab7a2d40f6c973cb57e63f23bfa2b51d981d028c'
VARIANTS = ('bert_context', 'hatebert_context', 'full', 'no_gloss', 'concat', 'no_contrastive', 'frozen_context')


def masked_mean(hidden, mask):
    mask = mask.to(dtype=hidden.dtype)
    counts = mask.sum(1, keepdim=True)
    if (counts <= 0).any():
        raise ValueError('Every example must contain at least one pooling token.')
    return (hidden * mask.unsqueeze(-1)).sum(1) / counts


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Mean over eligible anchors; positives are other examples with the same label."""
    if temperature <= 0:
        raise ValueError('Temperature must be positive.')
    labels = labels.reshape(-1)
    if len(features) != len(labels):
        raise ValueError('Feature and label batch sizes differ.')
    if len(features) < 2:
        return features.sum() * 0
    z = F.normalize(features, dim=-1)
    logits = z @ z.T / temperature
    nonself = ~torch.eye(len(z), device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & nonself
    eligible = positives.any(1)
    if not eligible.any():
        return features.sum() * 0
    denominator = torch.logsumexp(logits.masked_fill(~nonself, -torch.inf), dim=1)
    log_prob = logits - denominator[:, None]
    per_anchor = -(log_prob.masked_fill(~positives, 0).sum(1) / positives.sum(1).clamp_min(1))
    return per_anchor[eligible].mean()


class TargetSenseClassifier(nn.Module):
    def __init__(self, variant, context_encoder, gloss_encoder=None, hidden_size=256, dropout=0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.variant = variant
        self.context_name = 'bert_base' if variant == 'bert_context' else 'hatebert'
        self.uses_gloss = variant not in ('bert_context', 'hatebert_context', 'no_gloss')
        self.context_encoder = context_encoder
        self.gloss_encoder = gloss_encoder if self.uses_gloss else None
        d = context_encoder.config.hidden_size
        if self.uses_gloss and (gloss_encoder is None or gloss_encoder.config.hidden_size != d):
            raise ValueError('The independent gloss encoder must have the same hidden dimension.')
        multiplier = (2 if variant == 'concat' else 4) if self.uses_gloss else 1
        self.head = nn.Sequential(nn.Linear(d * multiplier, hidden_size), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        self.contrastive_weight = 0.0 if variant in ('bert_context', 'hatebert_context', 'no_contrastive') else 0.1
        self.temperature = 0.1
        self.spec = dict(variant=variant, context_model=BERT_NAME if self.context_name == 'bert_base' else HATEBERT_NAME,
                         context_revision=BERT_REVISION if self.context_name == 'bert_base' else HATEBERT_REVISION,
                         gloss_model=BERT_NAME if self.uses_gloss else None, gloss_revision=BERT_REVISION if self.uses_gloss else None,
                         encoder_hidden_size=d, head_hidden_size=hidden_size, dropout=dropout,
                         contrastive_weight=self.contrastive_weight, temperature=self.temperature)
        if variant == 'frozen_context':
            for parameter in self.context_encoder.parameters():
                parameter.requires_grad_(False)
            self.context_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.variant == 'frozen_context':
            self.context_encoder.eval()
        return self

    def forward(self, input_ids, attention_mask, target_mask, gloss_input_ids=None,
                gloss_attention_mask=None, gloss_pool_mask=None, labels=None):
        hidden = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ht = masked_mean(hidden, target_mask)
        if self.uses_gloss:
            if gloss_input_ids is None or gloss_pool_mask is None:
                raise ValueError('This configuration requires linked gloss tokens.')
            hd = masked_mean(self.gloss_encoder(input_ids=gloss_input_ids, attention_mask=gloss_attention_mask).last_hidden_state, gloss_pool_mask)
            joined = torch.cat([ht, hd], -1) if self.variant == 'concat' else torch.cat([ht, hd, (ht-hd).abs(), ht*hd], -1)
        else:
            joined = ht
        logits = self.head(joined).squeeze(-1)
        result = {'logits': logits, 'target_embeddings': ht}
        if labels is not None:
            bce = F.binary_cross_entropy_with_logits(logits, labels.float().reshape(-1))
            contrastive = supervised_contrastive_loss(ht, labels, self.temperature) if self.contrastive_weight else ht.sum()*0
            result.update(loss=bce+self.contrastive_weight*contrastive, bce_loss=bce, supcon_loss=contrastive)
        return result


def build_model(variant, context_encoder=None, gloss_encoder=None, cache_dir=None, local_files_only=False,
                hidden_size=256, dropout=0.1, seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    from transformers import AutoModel
    if context_encoder is None:
        name, revision = (BERT_NAME, BERT_REVISION) if variant == 'bert_context' else (HATEBERT_NAME, HATEBERT_REVISION)
        context_encoder = AutoModel.from_pretrained(name, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    if variant not in ('bert_context', 'hatebert_context', 'no_gloss') and gloss_encoder is None:
        gloss_encoder = AutoModel.from_pretrained(BERT_NAME, revision=BERT_REVISION, cache_dir=cache_dir, local_files_only=local_files_only)
    if gloss_encoder is context_encoder:
        raise ValueError('Context and gloss encoders must be independent modules.')
    return TargetSenseClassifier(variant, context_encoder, gloss_encoder, hidden_size, dropout)
