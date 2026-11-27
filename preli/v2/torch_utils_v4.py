import gc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

CACHE_DIR_HF = r"D:\Datathon\hf_cache"
LOG_COLS_V4 = (0, 1, 2, 16, 17, 18, 20)
LOG_DIVS_V4 = (6.0, 6.0, 3.0, 3.0, 3.0, 4.0, 4.0)


def scale_feats_v4(X: np.ndarray) -> np.ndarray:
    """Deterministic log-scale normalization of unbounded count features in 58-dim v4 array."""
    Xd = np.array(X, dtype=np.float32, copy=True)
    for col, div in zip(LOG_COLS_V4, LOG_DIVS_V4):
        Xd[:, col] = np.log1p(np.maximum(0.0, Xd[:, col])) / div
    return np.nan_to_num(Xd, nan=0.0, posinf=5.0, neginf=0.0)


class MultiTaskKDDataset(Dataset):
    """Dataset supporting hard labels and optional soft teacher probabilities for distillation."""

    def __init__(self, texts, feats, yl=None, yt=None, ya=None, soft_l=None, soft_t=None, soft_a=None, weights=None):
        self.texts = list(texts)
        self.feats = np.asarray(feats, dtype=np.float32)
        self.yl = None if yl is None else np.asarray(yl, dtype=np.int64)
        self.yt = None if yt is None else np.asarray(yt, dtype=np.float32)
        self.ya = None if ya is None else np.asarray(ya, dtype=np.int64)
        self.soft_l = None if soft_l is None else np.asarray(soft_l, dtype=np.float32)
        self.soft_t = None if soft_t is None else np.asarray(soft_t, dtype=np.float32)
        self.soft_a = None if soft_a is None else np.asarray(soft_a, dtype=np.float32)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = [self.texts[idx], self.feats[idx]]
        if self.yl is not None:
            item.extend([int(self.yl[idx]), self.yt[idx], int(self.ya[idx])])
        if self.soft_l is not None:
            item.extend([self.soft_l[idx], self.soft_t[idx], self.soft_a[idx]])
        if self.weights is not None:
            item.append(float(self.weights[idx]))
        return tuple(item)


class BucketBatchSampler:
    """Length-bucketed sampler minimizing padding waste and avoiding WDDM VRAM spikes."""

    def __init__(self, texts, bs, seed=42, jitter=16.0):
        self.lens = np.array([len(str(t)) for t in texts], dtype=np.float32)
        self.bs = int(bs)
        self.seed = int(seed)
        self.jitter = float(jitter)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        n = len(self.lens)
        order = np.argsort(self.lens + rng.uniform(0.0, self.jitter, size=n))
        batches = [order[i:i + self.bs].tolist() for i in range(0, n, self.bs)]
        rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return (len(self.lens) + self.bs - 1) // self.bs


def make_collate_fn(tok, max_len=160, has_labels=True, has_soft=False, has_weights=False):
    def collate(batch):
        texts = [b[0] for b in batch]
        feats = torch.from_numpy(np.stack([b[1] for b in batch])).float()
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out = [enc["input_ids"], enc["attention_mask"], feats]
        pos = 2
        if has_labels:
            out.append(torch.tensor([b[pos] for b in batch], dtype=torch.long))
            out.append(torch.from_numpy(np.stack([b[pos + 1] for b in batch])).float())
            out.append(torch.tensor([b[pos + 2] for b in batch], dtype=torch.long))
            pos += 3
        if has_soft:
            out.append(torch.from_numpy(np.stack([b[pos] for b in batch])).float())
            out.append(torch.from_numpy(np.stack([b[pos + 1] for b in batch])).float())
            out.append(torch.from_numpy(np.stack([b[pos + 2] for b in batch])).float())
            pos += 3
        if has_weights:
            out.append(torch.tensor([b[pos] for b in batch], dtype=torch.float32))
        return tuple(out)

    return collate


class HeadConditionedTransformer(nn.Module):
    """Multi-task transformer with multi-sample dropout and head-conditioned Task B/C."""

    def __init__(self, model_name, n_feats=58, pool_mode="mean", proj_dim=256, num_freeze_layers=0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR_HF, local_files_only=True, dtype=torch.float32)
        self.pool_mode = pool_mode
        h = self.backbone.config.hidden_size
        pool_dim = 2 * h if pool_mode == "cls_mean" else h

        if num_freeze_layers > 0:
            self._freeze_bottom(num_freeze_layers)

        self.feat_norm = nn.LayerNorm(n_feats)
        self.fuse = nn.Sequential(
            nn.Linear(pool_dim + n_feats, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 + 0.04 * i) for i in range(4)])
        self.head_label = nn.Linear(proj_dim, 6)
        self.head_tech = nn.Linear(proj_dim + 6, 6)
        self.head_action = nn.Linear(proj_dim + 6, 6)

    def _freeze_bottom(self, k):
        for name, p in self.backbone.named_parameters():
            if any(f"layer.{i}." in name for i in range(k)) or "embeddings" in name:
                p.requires_grad = False

    def _pool(self, last_hidden, attention_mask):
        if self.pool_mode == "cls":
            return last_hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        mean_p = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.pool_mode == "cls_mean":
            return torch.cat([last_hidden[:, 0, :], mean_p], dim=-1)
        return mean_p

    def forward(self, input_ids, attention_mask, feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self._pool(out, attention_mask)
        fn = self.feat_norm(feats)
        z = self.fuse(torch.cat([pooled, fn], dim=-1))

        logits_l = sum(self.head_label(d(z)) for d in self.dropouts) / len(self.dropouts)
        p_l_cond = F.softmax(logits_l.detach(), dim=-1)
        zc = torch.cat([z, p_l_cond], dim=-1)
        logits_t = sum(self.head_tech(d(zc)) for d in self.dropouts) / len(self.dropouts)
        logits_a = sum(self.head_action(d(zc)) for d in self.dropouts) / len(self.dropouts)
        return logits_l, logits_t, logits_a

    def param_groups(self, lr_backbone=2.5e-5, lr_head=1.5e-4, weight_decay=0.01):
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        bb_d, bb_nd, hd_d, hd_nd = [], [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_hd = not n.startswith("backbone.")
            is_nd = any(k in n for k in no_decay)
            if is_hd:
                (hd_nd if is_nd else hd_d).append(p)
            else:
                (bb_nd if is_nd else bb_d).append(p)
        return [
            {"params": bb_d, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": bb_nd, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": hd_d, "lr": lr_head, "weight_decay": weight_decay},
            {"params": hd_nd, "lr": lr_head, "weight_decay": 0.0},
        ]


def compute_multitask_loss(
    logits_l, logits_t, logits_a,
    yl, yt, ya,
    label_weights=None, act_weights=None, sample_w=None,
    soft_l=None, soft_t=None, soft_a=None, kd_alpha=0.45, temp=2.0,
):
    sw = sample_w if sample_w is not None else torch.ones_like(yl, dtype=torch.float32)
    sw = sw / sw.mean().clamp(min=1e-6)

    ce_l = F.cross_entropy(logits_l, yl, weight=label_weights, reduction="none", label_smoothing=0.02)
    loss_l = (ce_l * sw).mean()

    scam_mask = (yl >= 1) & (yl <= 4)
    if scam_mask.any():
        bce_t = F.binary_cross_entropy_with_logits(logits_t[scam_mask], yt[scam_mask], reduction="none").mean(dim=-1)
        loss_t = (bce_t * sw[scam_mask]).mean()
    else:
        loss_t = logits_t.sum() * 0.0

    act_mask = scam_mask & (ya >= 1) & (ya <= 5)
    if act_mask.any():
        # Mask out logit 0 or index on classes 1..5 cleanly
        ya_sub = ya[act_mask] - 1
        aw_sub = act_weights[1:] if act_weights is not None else None
        ce_a = F.cross_entropy(logits_a[act_mask, 1:], ya_sub, weight=aw_sub, reduction="none", label_smoothing=0.02)
        loss_a = (ce_a * sw[act_mask]).mean()
    else:
        loss_a = logits_a.sum() * 0.0

    loss_hard = 1.0 * loss_l + 0.55 * loss_t + 0.65 * loss_a

    if soft_l is not None and kd_alpha > 0:
        log_p_l = F.log_softmax(logits_l / temp, dim=-1)
        kd_l = F.kl_div(log_p_l, soft_l.clamp(min=1e-6), reduction="none").sum(dim=-1) * (temp * temp)
        loss_kd = (kd_l * sw).mean()
        if scam_mask.any():
            bce_kt = F.binary_cross_entropy_with_logits(logits_t[scam_mask], soft_t[scam_mask].clamp(1e-4, 1 - 1e-4), reduction="none").mean(dim=-1)
            loss_kd = loss_kd + 0.50 * (bce_kt * sw[scam_mask]).mean()
        if act_mask.any():
            log_p_a = F.log_softmax(logits_a[act_mask, 1:] / temp, dim=-1)
            p_soft_a = soft_a[act_mask, 1:].clamp(min=1e-6)
            p_soft_a = p_soft_a / p_soft_a.sum(dim=-1, keepdim=True)
            kd_a = F.kl_div(log_p_a, p_soft_a, reduction="none").sum(dim=-1) * (temp * temp)
            loss_kd = loss_kd + 0.55 * (kd_a * sw[act_mask]).mean()
        return (1.0 - kd_alpha) * loss_hard + kd_alpha * loss_kd
    return loss_hard


@torch.no_grad()
def predict_transformer(model, tok, texts, feats, bs=64, max_len=160, device="cuda"):
    model.eval()
    n = len(texts)
    # Sort by length for zero padding waste, then unsort back to original order
    lens = np.array([len(str(t)) for t in texts])
    order = np.argsort(lens)
    inv_order = np.argsort(order)

    ordered_texts = [texts[i] for i in order]
    ordered_feats = feats[order]

    p_l = np.zeros((n, 6), dtype=np.float32)
    p_t = np.zeros((n, 6), dtype=np.float32)
    p_a = np.zeros((n, 6), dtype=np.float32)

    for s in range(0, n, bs):
        e = min(n, s + bs)
        enc = tok(ordered_texts[s:e], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device, non_blocking=False)
        mask = enc["attention_mask"].to(device, non_blocking=False)
        f_t = torch.from_numpy(ordered_feats[s:e]).to(device, dtype=torch.float32)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            ll, lt, la = model(ids, mask, f_t)
            pl = F.softmax(ll.float(), dim=-1).cpu().numpy()
            pt = torch.sigmoid(lt.float()).cpu().numpy()
            pa_sub = F.softmax(la[:, 1:].float(), dim=-1).cpu().numpy()
        p_l[s:e] = pl
        p_t[s:e] = pt
        p_a[s:e, 1:] = pa_sub

    return p_l[inv_order], p_t[inv_order], p_a[inv_order]


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
