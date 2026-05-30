import itertools
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _cfg_value(cfg, name, default):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        if input_dim == output_dim:
            self.net = nn.Identity()
        else:
            layers: List[nn.Module] = [nn.Linear(input_dim, output_dim), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int, n_branches: int = 1, dropout: float = 0.0):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(input_dim, attn_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(input_dim, attn_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.attention_c = nn.Linear(attn_dim, n_branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attention_a(x)
        b = self.attention_b(x)
        return self.attention_c(self.dropout(a * b)).transpose(0, 1)


class PlainAttention(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int, n_branches: int = 1, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, attn_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(attn_dim, n_branches))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(0, 1)


class BagMILBase(nn.Module):
    def _slice_valid_bags(
        self,
        feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
        stride: int = 1,
    ) -> List[torch.Tensor]:
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if attn_mask is None:
            attn_mask = torch.zeros(feats.shape[:2], dtype=torch.bool, device=feats.device)
        elif attn_mask.dim() == 1:
            lengths = attn_mask.to(feats.device)
            attn_mask = torch.arange(feats.shape[1], device=feats.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            attn_mask = attn_mask.to(feats.device).bool()

        if stride > 1:
            idxs = (torch.arange(0, feats.shape[1], stride, device=feats.device) + (int(epoch) % stride)) % feats.shape[1]
            feats = feats[:, idxs]
            attn_mask = attn_mask[:, idxs]

        bags = []
        for i in range(feats.shape[0]):
            valid = ~attn_mask[i]
            if not torch.any(valid):
                valid = torch.zeros_like(valid)
                valid[0] = True
            bags.append(feats[i, valid].float())
        return bags

    def forward_with_loss(
        self,
        feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        epoch: int = 0,
        stride: int = 1,
        loss_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self(feats, attn_mask, epoch, stride)
        criterion = loss_fn if loss_fn is not None else F.cross_entropy
        return logits, criterion(logits, labels)


class ABMILBaseline(BagMILBase):
    """AttentionDeepMIL-style gated attention over pre-extracted UNI2-h features."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.25,
        gated: bool = True,
    ):
        super().__init__()
        self.projector = FeatureProjector(input_dim, hidden_dim, dropout)
        attention_cls = GatedAttention if gated else PlainAttention
        self.attention = attention_cls(hidden_dim, attn_dim, n_branches=1, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.apply(_init_weights)

    def _forward_bag(self, bag: torch.Tensor) -> torch.Tensor:
        h = self.projector(bag)
        a = F.softmax(self.attention(h), dim=1)
        m = torch.mm(a, h)
        return self.classifier(m).squeeze(0)

    def forward(self, feats: torch.Tensor, attn_mask=None, epoch: int = 0, stride: int = 1) -> torch.Tensor:
        bags = self._slice_valid_bags(feats, attn_mask, epoch, stride)
        return torch.stack([self._forward_bag(bag) for bag in bags], dim=0)


class ACMILBaseline(BagMILBase):
    """ACMIL gated-attention variant adapted to UNI2-h feature bags."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 128,
        num_classes: int = 3,
        n_token: int = 5,
        dropout: float = 0.25,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0,
        sub_loss_weight: float = 1.0,
        diversity_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.projector = FeatureProjector(input_dim, hidden_dim, dropout)
        self.attention = GatedAttention(hidden_dim, attn_dim, n_branches=n_token, dropout=0.0)
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(n_token)])
        self.slide_classifier = nn.Linear(hidden_dim, num_classes)
        self.n_token = int(n_token)
        self.n_masked_patch = int(n_masked_patch)
        self.mask_drop = float(mask_drop)
        self.sub_loss_weight = float(sub_loss_weight)
        self.diversity_loss_weight = float(diversity_loss_weight)
        self.apply(_init_weights)

    def _mask_top_attention(self, a_raw: torch.Tensor) -> torch.Tensor:
        if self.n_masked_patch <= 0 or not self.training:
            return a_raw
        k, n = a_raw.shape
        n_masked = min(self.n_masked_patch, n)
        n_drop = int(n_masked * self.mask_drop)
        if n_drop <= 0:
            return a_raw

        _, indices = torch.topk(a_raw, n_masked, dim=-1)
        rand_selected = torch.argsort(torch.rand(indices.shape, device=a_raw.device), dim=-1)[:, :n_drop]
        masked_indices = indices.gather(1, rand_selected)
        keep_mask = torch.ones(k, n, device=a_raw.device, dtype=torch.bool)
        keep_mask.scatter_(1, masked_indices, False)
        return a_raw.masked_fill(~keep_mask, -1e9)

    def _forward_bag(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.projector(bag)
        a_raw = self._mask_top_attention(self.attention(h))
        a = F.softmax(a_raw, dim=1)
        token_feats = torch.mm(a, h)
        sub_logits = torch.stack([head(token_feats[i]) for i, head in enumerate(self.classifiers)], dim=0)
        bag_attention = a.mean(dim=0, keepdim=True)
        bag_feat = torch.mm(bag_attention, h)
        slide_logits = self.slide_classifier(bag_feat).squeeze(0)
        return slide_logits, sub_logits, a_raw

    def forward(self, feats: torch.Tensor, attn_mask=None, epoch: int = 0, stride: int = 1) -> torch.Tensor:
        bags = self._slice_valid_bags(feats, attn_mask, epoch, stride)
        return torch.stack([self._forward_bag(bag)[0] for bag in bags], dim=0)

    def forward_with_loss(
        self,
        feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        epoch: int = 0,
        stride: int = 1,
        loss_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        criterion = loss_fn if loss_fn is not None else F.cross_entropy
        bags = self._slice_valid_bags(feats, attn_mask, epoch, stride)
        slide_logits = []
        sub_logits = []
        sub_labels = []
        diversity_losses = []

        for bag, label in zip(bags, labels):
            slide_logit, sub_logit, a_raw = self._forward_bag(bag)
            slide_logits.append(slide_logit)
            if self.n_token > 1 and self.sub_loss_weight > 0:
                sub_logits.append(sub_logit)
                sub_labels.append(label.repeat(self.n_token))
            if self.n_token > 1 and self.diversity_loss_weight > 0:
                a = F.softmax(a_raw, dim=1)
                pairs = [
                    F.cosine_similarity(a[i].unsqueeze(0), a[j].unsqueeze(0), dim=1).mean()
                    for i, j in itertools.combinations(range(self.n_token), 2)
                ]
                if pairs:
                    diversity_losses.append(torch.stack(pairs).mean())

        logits = torch.stack(slide_logits, dim=0)
        total_loss = criterion(logits, labels)
        if sub_logits:
            total_loss = total_loss + self.sub_loss_weight * criterion(
                torch.cat(sub_logits, dim=0),
                torch.cat(sub_labels, dim=0),
            )
        if diversity_losses:
            total_loss = total_loss + self.diversity_loss_weight * torch.stack(diversity_losses).mean()
        return logits, total_loss


class CLAMBaseline(BagMILBase):
    """CLAM single-branch or multi-branch head for UNI2-h feature bags."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.25,
        gated: bool = True,
        variant: str = "mb",
        k_sample: int = 8,
        instance_eval: bool = True,
        subtyping: bool = True,
        bag_weight: float = 0.7,
    ):
        super().__init__()
        self.projector = FeatureProjector(input_dim, hidden_dim, dropout)
        attention_cls = GatedAttention if gated else PlainAttention
        self.variant = variant.lower()
        n_branches = num_classes if self.variant == "mb" else 1
        self.attention = attention_cls(hidden_dim, attn_dim, n_branches=n_branches, dropout=dropout)
        if self.variant == "mb":
            self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])
        else:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_classes)])
        self.num_classes = int(num_classes)
        self.k_sample = int(k_sample)
        self.instance_eval = bool(instance_eval)
        self.subtyping = bool(subtyping)
        self.bag_weight = float(bag_weight)
        self.instance_loss_fn = nn.CrossEntropyLoss()
        self.apply(_init_weights)

    @staticmethod
    def _targets(length: int, value: int, device: torch.device) -> torch.Tensor:
        return torch.full((length,), value, dtype=torch.long, device=device)

    def _instance_loss(self, attention: torch.Tensor, h: torch.Tensor, classifier: nn.Module, positive: bool) -> torch.Tensor:
        k = min(self.k_sample, h.shape[0])
        if k <= 0:
            return h.sum() * 0.0
        top_p_ids = torch.topk(attention, k=k, dim=0).indices
        top_p = h.index_select(0, top_p_ids)
        if not positive:
            targets = self._targets(k, 0, h.device)
            return self.instance_loss_fn(classifier(top_p), targets)

        top_n_ids = torch.topk(-attention, k=k, dim=0).indices
        top_n = h.index_select(0, top_n_ids)
        logits = classifier(torch.cat([top_p, top_n], dim=0))
        targets = torch.cat([self._targets(k, 1, h.device), self._targets(k, 0, h.device)], dim=0)
        return self.instance_loss_fn(logits, targets)

    def _forward_bag(self, bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.projector(bag)
        a_raw = self.attention(h)
        a = F.softmax(a_raw, dim=1)
        m = torch.mm(a, h)
        if self.variant == "mb":
            logits = torch.empty(self.num_classes, device=h.device, dtype=h.dtype)
            for c, classifier in enumerate(self.classifiers):
                logits[c] = classifier(m[c]).squeeze(0)
        else:
            logits = self.classifier(m).squeeze(0)
        return logits, a, h

    def forward(self, feats: torch.Tensor, attn_mask=None, epoch: int = 0, stride: int = 1) -> torch.Tensor:
        bags = self._slice_valid_bags(feats, attn_mask, epoch, stride)
        return torch.stack([self._forward_bag(bag)[0] for bag in bags], dim=0)

    def forward_with_loss(
        self,
        feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        epoch: int = 0,
        stride: int = 1,
        loss_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        criterion = loss_fn if loss_fn is not None else F.cross_entropy
        bags = self._slice_valid_bags(feats, attn_mask, epoch, stride)
        logits = []
        instance_losses = []
        for bag, label in zip(bags, labels):
            bag_logits, attention, h = self._forward_bag(bag)
            logits.append(bag_logits)
            if self.instance_eval:
                one_hot = F.one_hot(label, num_classes=self.num_classes)
                for c, classifier in enumerate(self.instance_classifiers):
                    if one_hot[c].item() == 1:
                        branch = c if self.variant == "mb" else 0
                        instance_losses.append(self._instance_loss(attention[branch], h, classifier, positive=True))
                    elif self.subtyping:
                        branch = c if self.variant == "mb" else 0
                        instance_losses.append(self._instance_loss(attention[branch], h, classifier, positive=False))

        logits_tensor = torch.stack(logits, dim=0)
        bag_loss = criterion(logits_tensor, labels)
        if not instance_losses:
            return logits_tensor, bag_loss
        instance_loss = torch.stack(instance_losses).mean()
        total_loss = self.bag_weight * bag_loss + (1.0 - self.bag_weight) * instance_loss
        return logits_tensor, total_loss


class AttriMILBaseline(BagMILBase):
    """AttriMIL class-specific attribute scoring over UNI2-h feature bags."""

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.projector = FeatureProjector(input_dim, hidden_dim, dropout)
        self.adaptor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.attention_nets = nn.ModuleList(
            [GatedAttention(hidden_dim, hidden_dim // 2, n_branches=1, dropout=dropout) for _ in range(num_classes)]
        )
        self.classifiers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_classes)])
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.num_classes = int(num_classes)
        self.apply(_init_weights)

    def _forward_bag(self, bag: torch.Tensor) -> torch.Tensor:
        h = self.projector(bag)
        h = h + self.adaptor(h)
        logits = []
        for attention_net, classifier, bias in zip(self.attention_nets, self.classifiers, self.bias):
            a_raw = attention_net(h)
            weights = F.softmax(a_raw, dim=1)
            instance_scores = classifier(h).transpose(0, 1)
            logits.append((weights * instance_scores).sum(dim=1) + bias)
        return torch.cat(logits, dim=0)

    def forward(self, feats: torch.Tensor, attn_mask=None, epoch: int = 0, stride: int = 1) -> torch.Tensor:
        bags = self._slice_valid_bags(feats, attn_mask, epoch, stride)
        return torch.stack([self._forward_bag(bag) for bag in bags], dim=0)


def build_mil_baseline(cfg) -> nn.Module:
    model_name = str(_cfg_value(cfg, "model", "")).lower()
    input_dim = int(_cfg_value(cfg, "hidden_size", 1536))
    hidden_dim = int(_cfg_value(cfg, "baseline_dim", 512))
    attn_dim = int(_cfg_value(cfg, "baseline_attn_dim", max(128, hidden_dim // 2)))
    num_classes = int(_cfg_value(cfg, "num_classes", 3))
    dropout = float(_cfg_value(cfg, "baseline_dropout", _cfg_value(cfg, "dropout", 0.25)))

    if model_name == "abmil":
        return ABMILBaseline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            num_classes=num_classes,
            dropout=dropout,
            gated=bool(_cfg_value(cfg, "abmil_gated", True)),
        )
    if model_name == "acmil":
        return ACMILBaseline(
            input_dim=input_dim,
            hidden_dim=int(_cfg_value(cfg, "acmil_dim", hidden_dim)),
            attn_dim=int(_cfg_value(cfg, "acmil_attn_dim", 128)),
            num_classes=num_classes,
            n_token=int(_cfg_value(cfg, "acmil_n_token", 5)),
            dropout=dropout,
            n_masked_patch=int(_cfg_value(cfg, "n_masked_patch", 0)),
            mask_drop=float(_cfg_value(cfg, "mask_drop", 0.0)),
            sub_loss_weight=float(_cfg_value(cfg, "acmil_sub_loss_weight", 1.0)),
            diversity_loss_weight=float(_cfg_value(cfg, "acmil_diversity_loss_weight", 1.0)),
        )
    if model_name == "clam":
        return CLAMBaseline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            num_classes=num_classes,
            dropout=dropout,
            gated=bool(_cfg_value(cfg, "clam_gated", True)),
            variant=str(_cfg_value(cfg, "clam_variant", "mb")),
            k_sample=int(_cfg_value(cfg, "clam_k_sample", 8)),
            instance_eval=bool(_cfg_value(cfg, "clam_instance_eval", True)),
            subtyping=bool(_cfg_value(cfg, "clam_subtyping", True)),
            bag_weight=float(_cfg_value(cfg, "clam_bag_weight", 0.7)),
        )
    if model_name == "attrimil":
        return AttriMILBaseline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unknown MIL baseline model: {model_name}")
