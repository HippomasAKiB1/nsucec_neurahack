import gc
import os
import re
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from cv_metrics import (
    tech_str_to_vec, vec_to_tech_str, eval_multitask,
    build_template_groups, corrupt_text_adversarial, decode_consistent
)
from normalizer import clean_for_model, extract_structural_features
from extra_features import extract_extra_feats_batch

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FEAT_NAMES = list(extract_structural_features("a", "a").keys())
MAX_WORD_LEN = 96
MAX_CHAR_LEN = 256
CHAR_BUCKETS = 2048
WORD_BUCKETS = 32768


def scale_feats(X, X_extra):
    Xd = X.copy().astype(np.float32)
    Xd[:, 0] = np.log1p(Xd[:, 0]) / 5.0
    Xd[:, 1] = Xd[:, 1] / 12.0
    Xd[:, 25] = np.log1p(Xd[:, 25]) / 3.0
    Xd[:, 26] = np.log1p(Xd[:, 26]) / 3.0
    return np.hstack([Xd, X_extra.astype(np.float32)])


def stable_hash_word(tok: str) -> int:
    h = 2166136261
    for ch in tok:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % (WORD_BUCKETS - 1)) + 1


def encode_texts_batch(texts):
    n = len(texts)
    w_ids = np.zeros((n, MAX_WORD_LEN), dtype=np.int64)
    c_ids = np.zeros((n, MAX_CHAR_LEN), dtype=np.int64)
    w_lens = np.ones(n, dtype=np.int64)
    for i, t in enumerate(texts):
        tl = str(t).lower()
        toks = re.findall(r"\|\|\||https?://\S+|[\u0980-\u09FF]+|[a-z0-9_']+|[^\s]", tl)[:MAX_WORD_LEN]
        w_lens[i] = max(1, len(toks))
        for j, tok in enumerate(toks):
            w_ids[i, j] = stable_hash_word(tok)
        for j, ch in enumerate(tl[:MAX_CHAR_LEN]):
            c_ids[i, j] = (ord(ch) % (CHAR_BUCKETS - 1)) + 1
    return w_ids, c_ids, w_lens


class HybridBiGRUCNN(nn.Module):
    """Non-Transformer dual Word-BiGRU+Attention + Multi-Scale Char-CNN + 27 structural features."""

    def __init__(self, n_feats=27):
        super().__init__()
        self.w_emb = nn.Embedding(WORD_BUCKETS, 96, padding_idx=0)
        self.c_emb = nn.Embedding(CHAR_BUCKETS, 48, padding_idx=0)
        self.w_drop = nn.Dropout(0.15)
        self.gru = nn.GRU(96, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15)
        self.attn_proj = nn.Linear(256, 1)

        self.conv3 = nn.Conv1d(48, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(48, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(48, 64, kernel_size=7, padding=3)

        # 256 (attn) + 256 (mean) + 192*2 (char max+avg) + n_feats
        dim_in = 256 + 256 + 192 * 2 + n_feats
        self.fuse = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.drops = nn.ModuleList([nn.Dropout(p) for p in (0.15, 0.25, 0.35)])
        self.head_label = nn.Linear(256, 6)
        self.head_tech = nn.Linear(256, 6)
        self.head_action = nn.Linear(256, 6)

    def forward(self, w_ids, c_ids, feats):
        w_mask = (w_ids > 0)
        we = self.w_drop(self.w_emb(w_ids))
        gru_out, _ = self.gru(we)
        scores = self.attn_proj(gru_out).squeeze(-1)
        scores = scores.masked_fill(~w_mask, -1e4)
        alpha = torch.softmax(scores, dim=-1).unsqueeze(-1)
        w_attn = (gru_out * alpha).sum(dim=1)
        mf = w_mask.unsqueeze(-1).type_as(gru_out)
        w_mean = (gru_out * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1e-9)

        ce = self.c_emb(c_ids).transpose(1, 2)
        c3 = torch.relu(self.conv3(ce))
        c5 = torch.relu(self.conv5(ce))
        c7 = torch.relu(self.conv7(ce))
        cc = torch.cat([c3, c5, c7], dim=1)
        c_max = cc.max(dim=-1).values
        c_avg = cc.mean(dim=-1)

        z = self.fuse(torch.cat([w_attn, w_mean, c_max, c_avg, feats], dim=-1))
        if self.training:
            ll = sum(self.head_label(d(z)) for d in self.drops) / len(self.drops)
            lt = sum(self.head_tech(d(z)) for d in self.drops) / len(self.drops)
            la = sum(self.head_action(d(z)) for d in self.drops) / len(self.drops)
            return ll, lt, la
        return self.head_label(z), self.head_tech(z), self.head_action(z)


class SeqDataset(Dataset):
    def __init__(self, texts, feats, yl=None, yt=None, ya=None):
        self.w_ids, self.c_ids, _ = encode_texts_batch(texts)
        self.feats = feats.astype(np.float32)
        self.yl = yl.astype(np.int64) if yl is not None else None
        self.yt = yt.astype(np.float32) if yt is not None else None
        self.ya = ya.astype(np.int64) if ya is not None else None

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        if self.yl is not None:
            return self.w_ids[idx], self.c_ids[idx], self.feats[idx], self.yl[idx], self.yt[idx], self.ya[idx]
        return self.w_ids[idx], self.c_ids[idx], self.feats[idx]


@torch.no_grad()
def predict_texts(mdl, texts, feats, bs=512):
    mdl.eval()
    n = len(texts)
    pl_list, pt_list, pa_list = [], [], []
    for i in range(0, n, bs):
        wi, ci, _ = encode_texts_batch(texts[i:i+bs])
        wi_t = torch.from_numpy(wi).to("cuda", non_blocking=True)
        ci_t = torch.from_numpy(ci).to("cuda", non_blocking=True)
        ft_t = torch.from_numpy(feats[i:i+bs]).to("cuda", non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ll, lt, la = mdl(wi_t, ci_t, ft_t)
        pl_list.append(torch.softmax(ll.float(), dim=-1).cpu().numpy())
        pt_list.append(torch.sigmoid(lt.float()).cpu().numpy())
        pa_list.append(torch.softmax(la.float(), dim=-1).cpu().numpy())
    return np.vstack(pl_list), np.vstack(pt_list), np.vstack(pa_list)


def train_loop(mdl, texts, feats, yl, yt, ya, epochs=6, bs=128, lr=1.5e-3):
    ds = SeqDataset(texts, feats, yl, yt, ya)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, len(dl) * epochs))
    scaler = torch.amp.GradScaler("cuda")
    ce_l = nn.CrossEntropyLoss(label_smoothing=0.03)
    ce_a = nn.CrossEntropyLoss(label_smoothing=0.03)
    bce_t = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        mdl.train()
        tot = 0.0
        t0 = time.time()
        for wi, ci, ft, byl, byt, bya in dl:
            wi, ci, ft = wi.to("cuda"), ci.to("cuda"), ft.to("cuda")
            byl, byt, bya = byl.to("cuda"), byt.to("cuda"), bya.to("cuda")
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ll, lt, la = mdl(wi, ci, ft)
                loss = 1.2 * ce_l(ll, byl) + 0.8 * bce_t(lt, byt) + 0.7 * ce_a(la, bya)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        print(f"  Epoch {ep+1}/{epochs} | loss={tot/len(dl):.4f} ({time.time()-t0:.1f}s)", flush=True)


def main():
    syn_df = pd.read_csv("preli/cache/synth_32k_v3.csv")
    syn_texts = syn_df["clean_text"].fillna("").astype(str).tolist()
    syn_feats = scale_feats(np.load("preli/cache/synth_32k_v3_feats.npy"), np.load("preli/cache/synth_32k_v3_extra.npy"))
    syn_yl = syn_df["label"].values
    syn_yt = np.vstack([tech_str_to_vec(x) for x in syn_df["techniques"]])
    syn_ya = syn_df["action"].values

    s_df = pd.read_csv("preli/cache/sample_clean.csv")
    s_texts = s_df["clean_text"].fillna("").astype(str).tolist()
    s_feats = scale_feats(np.load("preli/cache/sample_feats.npy"), np.load("preli/cache/sample_extra.npy"))
    s_yl = s_df["label"].values
    s_yt = np.vstack([tech_str_to_vec(x) for x in s_df["techniques"]])
    s_ya = s_df["action"].values
    groups = build_template_groups(s_texts)

    rng_adv = random.Random(42 + 99)
    adv_clean, adv_feats_raw, adv_p = [], [], []
    for i, raw in enumerate(s_df["text"].tolist()):
        for _ in range(5):
            cr = corrupt_text_adversarial(raw, rng_adv)
            cc = clean_for_model(cr)
            d = extract_structural_features(cr, cc)
            adv_clean.append(cc)
            adv_feats_raw.append([d[k] for k in FEAT_NAMES])
            adv_p.append(i)
    adv_feats = scale_feats(np.array(adv_feats_raw, dtype=np.float32), extract_extra_feats_batch(adv_clean))
    adv_p = np.array(adv_p)

    ckpt_a = "preli/cache/bicnn_stageA.pt"
    mdl = HybridBiGRUCNN(n_feats=s_feats.shape[1]).to("cuda")
    if not os.path.exists(ckpt_a):
        print("--- Stage A: Training HybridBiGRUCNN on 32k v3 synthetic ---", flush=True)
        train_loop(mdl, syn_texts, syn_feats, syn_yl, syn_yt, syn_ya, epochs=7, bs=128, lr=1.5e-3)
        torch.save(mdl.state_dict(), ckpt_a)
    else:
        mdl.load_state_dict(torch.load(ckpt_a, map_location="cuda"))

    pl0, pt0, pa0 = predict_texts(mdl, s_texts, s_feats)
    d0_l, d0_t, d0_a = decode_consistent(pl0, pt0, pa0)
    r0 = eval_multitask(s_yl, d0_l, s_yt, d0_t, s_ya, d0_a, pl0.max(1))
    print("[BiGRU+CharCNN Stage A Zero-Shot on sample.csv]:", {k: round(v, 2) for k, v in r0.items()}, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_l, oof_t, oof_a = np.zeros((86, 6), dtype=np.float32), np.zeros((86, 6), dtype=np.float32), np.zeros((86, 6), dtype=np.float32)
    adv_l, adv_t, adv_a = np.zeros((len(adv_clean), 6), dtype=np.float32), np.zeros((len(adv_clean), 6), dtype=np.float32), np.zeros((len(adv_clean), 6), dtype=np.float32)

    s_raw_list = s_df["text"].tolist()
    for fold, (tr, va) in enumerate(sgkf.split(s_feats, s_yl, groups)):
        mdl.load_state_dict(torch.load(ckpt_a, map_location="cuda"))
        rng_f = random.Random(SEED + 500 + fold)
        rep_idx = np.random.RandomState(SEED + fold).choice(len(syn_texts), size=800, replace=False)
        tr_clean_rep = np.repeat(tr, 4)
        tr_adv_idx = np.repeat(tr, 2)
        tr_adv_texts, tr_adv_feats = [], []
        for idx in tr_adv_idx:
            cr = corrupt_text_adversarial(s_raw_list[idx], rng_f)
            cc = clean_for_model(cr)
            d = extract_structural_features(cr, cc)
            tr_adv_texts.append(cc)
            tr_adv_feats.append([d[k] for k in FEAT_NAMES])
        tr_adv_feats = scale_feats(np.array(tr_adv_feats, dtype=np.float32), extract_extra_feats_batch(tr_adv_texts))

        f_texts = [s_texts[i] for i in tr_clean_rep] + tr_adv_texts + [syn_texts[i] for i in rep_idx]
        f_feats = np.vstack([s_feats[tr_clean_rep], tr_adv_feats, syn_feats[rep_idx]])
        f_yl = np.concatenate([s_yl[tr_clean_rep], s_yl[tr_adv_idx], syn_yl[rep_idx]])
        f_yt = np.vstack([s_yt[tr_clean_rep], s_yt[tr_adv_idx], syn_yt[rep_idx]])
        f_ya = np.concatenate([s_ya[tr_clean_rep], s_ya[tr_adv_idx], syn_ya[rep_idx]])
        print(f"--- Stage B Fold {fold} ({len(f_texts)} rows) ---", flush=True)
        train_loop(mdl, f_texts, f_feats, f_yl, f_yt, f_ya, epochs=4, bs=64, lr=5e-4)
        oof_l[va], oof_t[va], oof_a[va] = predict_texts(mdl, [s_texts[i] for i in va], s_feats[va])
        am = np.isin(adv_p, va)
        adv_l[am], adv_t[am], adv_a[am] = predict_texts(mdl, [adv_clean[i] for i in np.where(am)[0]], adv_feats[am])

    oof_pl, oof_pt, oof_pa = decode_consistent(oof_l, oof_t, oof_a)
    adv_pl, adv_pt, adv_pa = decode_consistent(adv_l, adv_t, adv_a)
    r_oof = eval_multitask(s_yl, oof_pl, s_yt, oof_pt, s_ya, oof_pa, oof_l.max(1))
    r_adv = eval_multitask(s_yl[adv_p], adv_pl, s_yt[adv_p], adv_pt, s_ya[adv_p], adv_pa, adv_l.max(1))
    print("[BiGRU+CharCNN Stage B] Group-CV :", {k: round(v, 2) for k, v in r_oof.items()}, flush=True)
    print("[BiGRU+CharCNN Stage B] Robust-CV:", {k: round(v, 2) for k, v in r_adv.items()}, flush=True)
    np.savez("preli/cache/oof_bicnn.npz", oof_l=oof_l, oof_t=oof_t, oof_a=oof_a, adv_l=adv_l, adv_t=adv_t, adv_a=adv_a)

    mdl.load_state_dict(torch.load(ckpt_a, map_location="cuda"))
    rng_full = random.Random(SEED + 999)
    rep_idx = np.random.RandomState(SEED + 999).choice(len(syn_texts), size=1000, replace=False)
    all_clean_rep = np.repeat(np.arange(len(s_texts)), 4)
    all_adv_idx = np.repeat(np.arange(len(s_texts)), 2)
    all_adv_texts, all_adv_feats = [], []
    for idx in all_adv_idx:
        cr = corrupt_text_adversarial(s_raw_list[idx], rng_full)
        cc = clean_for_model(cr)
        d = extract_structural_features(cr, cc)
        all_adv_texts.append(cc)
        all_adv_feats.append([d[k] for k in FEAT_NAMES])
    all_adv_feats = scale_feats(np.array(all_adv_feats, dtype=np.float32), extract_extra_feats_batch(all_adv_texts))

    f_texts = [s_texts[i] for i in all_clean_rep] + all_adv_texts + [syn_texts[i] for i in rep_idx]
    f_feats = np.vstack([s_feats[all_clean_rep], all_adv_feats, syn_feats[rep_idx]])
    f_yl = np.concatenate([s_yl[all_clean_rep], s_yl[all_adv_idx], syn_yl[rep_idx]])
    f_yt = np.vstack([s_yt[all_clean_rep], s_yt[all_adv_idx], syn_yt[rep_idx]])
    f_ya = np.concatenate([s_ya[all_clean_rep], s_ya[all_adv_idx], syn_ya[rep_idx]])
    print("--- Stage B Full Fit -> 500k Inference ---", flush=True)
    train_loop(mdl, f_texts, f_feats, f_yl, f_yt, f_ya, epochs=4, bs=64, lr=5e-4)
    torch.save(mdl.state_dict(), "preli/cache/bicnn_full.pt")
    del syn_df, syn_texts, syn_feats, f_texts, f_feats
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    te_feats = scale_feats(np.load("preli/cache/test_feats.npy"), np.load("preli/cache/test_extra.npy"))
    te_l = np.memmap("preli/cache/te_bicnn_l.dat", dtype="float32", mode="w+", shape=(500000, 6))
    te_t = np.memmap("preli/cache/te_bicnn_t.dat", dtype="float32", mode="w+", shape=(500000, 6))
    te_a = np.memmap("preli/cache/te_bicnn_a.dat", dtype="float32", mode="w+", shape=(500000, 6))
    offset = 0
    for chunk in pd.read_csv("preli/cache/test_clean.csv", chunksize=50000):
        n = len(chunk)
        cl = chunk["clean_text"].fillna("").astype(str).tolist()
        pl, pt, pa = predict_texts(mdl, cl, te_feats[offset:offset+n], bs=1024)
        te_l[offset:offset+n] = pl
        te_t[offset:offset+n] = pt
        te_a[offset:offset+n] = pa
        offset += n
        print(f"500k BiGRU+CharCNN inferred {offset}/500000 ({time.time()-t0:.1f}s)", flush=True)

    for fn, arr in [("te_bicnn_l.npy", te_l), ("te_bicnn_t.npy", te_t), ("te_bicnn_a.npy", te_a)]:
        arr.flush()
        np.save(f"preli/cache/{fn}", np.array(arr))
        if getattr(arr, "_mmap", None) is not None:
            arr._mmap.close()
    del te_l, te_t, te_a
    gc.collect()
    for fn in ["te_bicnn_l.dat", "te_bicnn_t.dat", "te_bicnn_a.dat"]:
        os.remove(f"preli/cache/{fn}")
    print(f"ALL DONE BiGRU+CharCNN 500k in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
