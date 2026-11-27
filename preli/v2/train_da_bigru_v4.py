import argparse
import gc
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
CKPT_DIR = CACHE_DIR / "ckpt_v4"
OUT_DIR = CACHE_DIR / "preds_v4"
sys.path.insert(0, str(V2_DIR))

from cv_ood import tech_str_to_vec, get_balanced_3folds
from normalizer_v4 import format_turn_aware
from torch_utils_v4 import CACHE_DIR_HF, scale_feats_v4, BucketBatchSampler, free_cuda


class BPEBiGRUNet(nn.Module):
    """Non-Transformer Sequence Model: Subword-Hashed BiGRU + Attention/Max Pool + 58 Domain Features."""

    def __init__(self, n_buckets=32000, emb_dim=128, hidden=160, n_feats=58, proj_dim=224):
        super().__init__()
        self.emb = nn.Embedding(n_buckets, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(0.15)
        self.gru = nn.GRU(emb_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.attn = nn.Linear(2 * hidden, 1)
        self.feat_norm = nn.LayerNorm(n_feats)
        self.fuse = nn.Sequential(
            nn.Linear(4 * hidden + n_feats, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.dropouts = nn.ModuleList([nn.Dropout(0.12 + 0.04 * i) for i in range(3)])
        self.head_label = nn.Linear(proj_dim, 6)
        self.head_tech = nn.Linear(proj_dim + 6, 6)
        self.head_action = nn.Linear(proj_dim + 6, 6)

    def forward(self, input_ids, attention_mask, feats):
        h_ids = input_ids % self.emb.num_embeddings
        h_ids = h_ids.masked_fill(attention_mask == 0, 0)
        x = self.emb_drop(self.emb(h_ids))
        out, _ = self.gru(x)
        mask_f = attention_mask.unsqueeze(-1).float()
        w_logits = self.attn(out).squeeze(-1).masked_fill(attention_mask == 0, -1e4)
        w = F.softmax(w_logits, dim=-1).unsqueeze(-1)
        attn_pool = (out * w).sum(dim=1)
        max_pool = (out.masked_fill(mask_f == 0, -1e4)).max(dim=1).values
        fn = self.feat_norm(feats)
        z = self.fuse(torch.cat([attn_pool, max_pool, fn], dim=-1))
        ll = sum(self.head_label(d(z)) for d in self.dropouts) / len(self.dropouts)
        zc = torch.cat([z, F.softmax(ll.detach(), dim=-1)], dim=-1)
        lt = sum(self.head_tech(d(zc)) for d in self.dropouts) / len(self.dropouts)
        la = sum(self.head_action(d(zc)) for d in self.dropouts) / len(self.dropouts)
        return ll, lt, la


def train_eval_bigru_fold(tr_texts, tr_feats, tr_yl, tr_yt, tr_ya, va_texts, va_feats, va_yl=None, sl=None, st=None, sa=None, sw=None, epochs=5, bs=96, seed=42):
    from torch_utils_v4 import MultiTaskKDDataset, make_collate_fn, compute_multitask_loss
    free_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = BPEBiGRUNet(n_feats=tr_feats.shape[1]).cuda()

    has_soft = sl is not None
    has_w = sw is not None
    ds = MultiTaskKDDataset(tr_texts, tr_feats, tr_yl, tr_yt, tr_ya, sl, st, sa, sw)
    sampler = BucketBatchSampler(tr_texts, bs=bs, seed=seed, jitter=16.0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(tok, max_len=160, has_labels=True, has_soft=has_soft, has_weights=has_w))

    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(loader) * epochs))

    cnt_l = np.bincount(tr_yl, minlength=6).astype(np.float32)
    w_l = torch.tensor(np.sqrt(cnt_l.sum() / (6.0 * np.maximum(cnt_l, 1.0))), dtype=torch.float32, device="cuda")
    w_l /= w_l.mean()
    scam_ya = tr_ya[(tr_ya >= 1) & (tr_ya <= 5)]
    w_a_arr = np.ones(6, dtype=np.float32)
    cnt_a = np.array([max(1.0, float((scam_ya == c).sum())) for c in range(1, 6)], dtype=np.float32)
    w_a_arr[1:] = np.clip(np.sqrt(cnt_a.sum() / (5.0 * cnt_a)), 0.7, 2.2)
    w_a = torch.tensor(w_a_arr, dtype=torch.float32, device="cuda")

    best_f1 = -1.0
    best_va = None
    best_sd = None
    for ep in range(epochs):
        model.train()
        for batch in loader:
            ids, mask, f_t = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            yl, yt, ya = batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
            pos = 6
            bsl = bst = bsa = bsw = None
            if has_soft:
                bsl, bst, bsa = batch[pos].cuda(), batch[pos + 1].cuda(), batch[pos + 2].cuda()
                pos += 3
            if has_w:
                bsw = batch[pos].cuda()
            opt.zero_grad(set_to_none=True)
            ll, lt, la = model(ids, mask, f_t)
            loss = compute_multitask_loss(ll, lt, la, yl, yt, ya, label_weights=w_l, act_weights=w_a, sample_w=bsw, soft_l=bsl, soft_t=bst, soft_a=bsa, kd_alpha=0.45 if has_soft else 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

        if va_texts is not None and va_yl is not None:
            pl, pt, pa = predict_bigru(model, tok, va_texts, va_feats, bs=256)
            mf1 = 100.0 * f1_score(va_yl, pl.argmax(1), average="macro", zero_division=0)
            if mf1 >= best_f1:
                best_f1 = mf1
                best_va = (pl, pt, pa)
                best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict({k: v.cuda() for k, v in best_sd.items()})
    return model, tok, best_va, best_f1


@torch.no_grad()
def predict_bigru(model, tok, texts, feats, bs=256, max_len=160):
    model.eval()
    dev = next(model.parameters()).device
    n = len(texts)
    order = np.argsort([len(str(t)) for t in texts])
    inv = np.argsort(order)
    ot = [texts[i] for i in order]
    of = feats[order]
    pl = np.zeros((n, 6), dtype=np.float32)
    pt = np.zeros((n, 6), dtype=np.float32)
    pa = np.zeros((n, 6), dtype=np.float32)
    for s in range(0, n, bs):
        e = min(n, s + bs)
        enc = tok(ot[s:e], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ll, lt, la = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev), torch.from_numpy(of[s:e]).to(dev))
        pl[s:e] = F.softmax(ll, dim=-1).cpu().numpy()
        pt[s:e] = torch.sigmoid(lt).cpu().numpy()
        pa[s:e, 1:] = F.softmax(la[:, 1:], dim=-1).cpu().numpy()
    return pl[inv], pt[inv], pa[inv]


def main():
    t0 = time.time()
    sdf = pd.read_csv(CACHE_DIR / "sample_v4.csv")
    sf = scale_feats_v4(np.load(CACHE_DIR / "sample_v4_feats.npy"))
    sy = pd.read_csv(CACHE_DIR / "synth_64k_v4.csv")
    syf = scale_feats_v4(np.load(CACHE_DIR / "synth_64k_v4_feats.npy"))

    s_tx = [format_turn_aware(str(x)) for x in sdf["clean_text"].fillna("")]
    sy_tx = [format_turn_aware(str(x)) for x in sy["clean_text"].fillna("")]
    s_yl = sdf["label"].values.astype(np.int64)
    s_yt = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]]).astype(np.float32)
    s_ya = sdf["action"].values.astype(np.int64)
    s_folds = get_balanced_3folds(sdf["scenario_family"].values)

    sy_yl = sy["label"].values.astype(np.int64)
    sy_yt = np.vstack([tech_str_to_vec(x) for x in sy["techniques"]]).astype(np.float32)
    sy_ya = sy["action"].values.astype(np.int64)
    sy_folds = get_balanced_3folds(sy["scenario_family"].values)

    # 1. Zero-Leakage Balanced 3-Fold Scenario-Disjoint BiGRU-v5 OOF
    o_s_l, o_s_t, o_s_a = np.zeros((len(sdf), 6), np.float32), np.zeros((len(sdf), 6), np.float32), np.zeros((len(sdf), 6), np.float32)
    o_sy_l, o_sy_t, o_sy_a = np.zeros((len(sy), 6), np.float32), np.zeros((len(sy), 6), np.float32), np.zeros((len(sy), 6), np.float32)
    for fold in (0, 1, 2):
        tr_sy, va_sy = np.where(sy_folds != fold)[0], np.where(sy_folds == fold)[0]
        tr_s, va_s = np.where(s_folds != fold)[0], np.where(s_folds == fold)[0]
        tr_tx = [sy_tx[i] for i in tr_sy] + [s_tx[i] for i in tr_s] * 2
        tr_f = np.vstack([syf[tr_sy], sf[tr_s], sf[tr_s]])
        va_tx = [s_tx[i] for i in va_s] + [sy_tx[i] for i in va_sy]
        va_f = np.vstack([sf[va_s], syf[va_sy]])
        m, tok, (pl, pt, pa), mf1 = train_eval_bigru_fold(
            tr_tx, tr_f,
            np.concatenate([sy_yl[tr_sy], s_yl[tr_s], s_yl[tr_s]]),
            np.vstack([sy_yt[tr_sy], s_yt[tr_s], s_yt[tr_s]]),
            np.concatenate([sy_ya[tr_sy], s_ya[tr_s], s_ya[tr_s]]),
            va_tx, va_f, va_yl=np.concatenate([s_yl[va_s], sy_yl[va_sy]]), epochs=4, bs=96, seed=70 + fold,
        )
        ns = len(va_s)
        o_s_l[va_s], o_s_t[va_s], o_s_a[va_s] = pl[:ns], pt[:ns], pa[:ns]
        o_sy_l[va_sy], o_sy_t[va_sy], o_sy_a[va_sy] = pl[ns:], pt[ns:], pa[ns:]
        print(f"BiGRU-v5 Scenario-Disjoint Fold {fold} OOD_F1={mf1:.2f} elapsed={time.time()-t0:.1f}s", flush=True)
        del m; free_cuda()

    np.savez_compressed(
        OUT_DIR / "oof_bigru_v4.npz",
        s_l=o_s_l, s_t=o_s_t, s_a=o_s_a,
        sy_l=o_sy_l, sy_t=o_sy_t, sy_a=o_sy_a,
    )

    # 2. Domain-Adapted BiGRU-v5 distilled from domain_pseudo_v5 + synth + sample -> predict 500k
    dp_p = CACHE_DIR / "domain_pseudo_v5.npz"
    dp = np.load(dp_p if dp_p.exists() else (CACHE_DIR / "domain_pseudo_v4.npz"), allow_pickle=True)
    from train_students_v4 import build_distill_corpus
    tr_tx, tr_f, tr_yl, tr_yt, tr_ya, tr_sl, tr_st, tr_sa, tr_w = build_distill_corpus(sdf, sf, sy, syf, dp, text_mode="plain")
    m_full, tok, _, _ = train_eval_bigru_fold(
        tr_tx, tr_f, tr_yl, tr_yt, tr_ya, None, None, None,
        sl=tr_sl, st=tr_st, sa=tr_sa, sw=tr_w, epochs=5, bs=112, seed=909,
    )
    torch.save({"state_dict": m_full.state_dict()}, CKPT_DIR / "bigru_v4_distill.pt")

    te_f = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")
    n_te = len(te_f)
    p_l, p_t, p_a = np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32)
    off = 0
    for df_c in pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=50000):
        m = len(df_c)
        txts = [format_turn_aware(str(x)) for x in df_c["clean_text"].fillna("")]
        pl, pt, pa = predict_bigru(m_full, tok, txts, scale_feats_v4(te_f[off:off + m]), bs=256)
        p_l[off:off + m], p_t[off:off + m], p_a[off:off + m] = pl, pt, pa
        off += m
    np.savez_compressed(OUT_DIR / "test_bigru_v4.npz", p_l=p_l, p_t=p_t, p_a=p_a)
    print(f"BiGRU-v4 500k prediction complete in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
