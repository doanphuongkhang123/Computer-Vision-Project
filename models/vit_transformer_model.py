import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCrossAttentionPooling(nn.Module):
    def __init__(
        self,
        dim,
        ratio=0.25,
        max_tokens=1024,
        heads=8,
        max_retrieved=None,
        prototype_dropout=0.0,
    ):
        super().__init__()
        if not 0.0 <= prototype_dropout < 1.0:
            raise ValueError("prototype_dropout must be in [0, 1)")

        self.ratio = ratio
        self.max_tokens = max_tokens
        self.max_retrieved = None if max_retrieved is None else int(max_retrieved)
        self.prototype_dropout = float(prototype_dropout)
        self.cluster_pool = nn.Parameter(torch.randn(max_tokens, dim))
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def _prototype_keep_mask(self, batch_size, retrieved_count, device):
        keep_mask = torch.ones(batch_size, self.max_tokens, dtype=torch.bool, device=device)
        if not self.training or self.prototype_dropout == 0.0:
            return keep_mask

        drop_count = int(self.max_tokens * self.prototype_dropout)
        drop_count = min(drop_count, self.max_tokens - retrieved_count)
        if drop_count > 0:
            drop_idx = torch.rand(batch_size, self.max_tokens, device=device).topk(drop_count, dim=-1).indices
            keep_mask.scatter_(1, drop_idx, False)
        return keep_mask

    def forward(self, x, attn_mask=None, return_attn=False):
        b, t, _ = x.shape
        if attn_mask is None:
            k = max(4, int(t * self.ratio))
        else:
            k = max(4, int((~attn_mask.bool()).sum(dim=1).float().mean() * self.ratio))
        if self.max_retrieved is not None:
            k = min(k, self.max_retrieved)
        k = max(1, min(k, self.max_tokens))

        sim = torch.einsum("btd,kd->btk", F.normalize(x, dim=-1), F.normalize(self.cluster_pool, dim=-1))
        prototype_keep_mask = self._prototype_keep_mask(b, k, x.device)
        sim = sim.masked_fill(~prototype_keep_mask[:, None, :], float("-inf"))
        token_idx = sim.topk(k, dim=-1).indices.reshape(b, -1)
        weights = torch.ones(b, t, k, dtype=x.dtype, device=x.device)
        if attn_mask is not None:
            weights = weights * (~attn_mask.bool()).unsqueeze(-1)

        freq = torch.zeros(b, self.max_tokens, device=x.device)
        freq.scatter_add_(1, token_idx, weights.reshape(b, -1).to(freq.dtype))
        freq = freq.masked_fill(~prototype_keep_mask, float("-inf"))
        queries = self.cluster_pool[freq.topk(k, dim=-1).indices]
        out, attn = self.cross_attn(queries, x, x, key_padding_mask=attn_mask)
        out = self.norm(out)
        return (out, attn) if return_attn else out


class VTransAdaptive(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.15,
        hidden_dim=1536,
        ratio=0.5,
        cls_ratio=0.8,
        max_retrieved=None,
        prototype_dropout=0.0,
    ):
        super().__init__()
        self.cls_ratio = cls_ratio
        self.token_pooler = AdaptiveCrossAttentionPooling(
            dim=hidden_dim,
            ratio=ratio,
            heads=8,
            max_retrieved=max_retrieved,
            prototype_dropout=prototype_dropout,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, dropout=dropout, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats, attn_mask, epoch=0, stride=1, return_attn=False):
        b, t, _ = feats.shape
        if attn_mask.dim() == 1:
            attn_mask = torch.arange(t, device=feats.device)[None, :] >= attn_mask[:, None]

        idxs = (torch.arange(0, t, stride, device=feats.device) + (epoch % stride)) % t
        feats = feats[:, idxs]
        attn_mask = attn_mask[:, idxs]

        if return_attn:
            clustered_feats, attn = self.token_pooler(feats, attn_mask, return_attn=True)
        else:
            clustered_feats = self.token_pooler(feats, attn_mask)
        cls = self.cls_token.expand(b, -1, -1)
        out = self.temporal_encoder(torch.cat([cls, clustered_feats], dim=1))
        out = out[:, 0] * self.cls_ratio + out[:, 1:].mean(dim=1) * (1 - self.cls_ratio)
        logits = self.fc(out)
        if return_attn:
            return logits, attn, idxs
        return logits
