import gc
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold

from cv_metrics import (
    tech_str_to_vec, vec_to_tech_str, eval_multitask,
    build_template_groups, corrupt_text_adversarial, decode_consistent
)
from normalizer import clean_for_model, extract_structural_features
from extra_features import extract_extra_feats_batch

SEED = 103
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CACHE_DIR = r"D:\Datathon\hf_cache"
MODEL_NAME = "intfloat/multilingual-e5-small"
FEAT_NAMES = list(extract_structural_features("a", "a").keys())


def scale_feats(X, X_extra):
    Xd = X.copy().astype(np.float32)
    Xd[:, 0] = np.log1p(Xd[:, 0]) / 5.0
    Xd[:, 1] = Xd[:, 1] / 12.0
    Xd[:, 25] = np.log1p(Xd[:, 25]) / 3.0
    Xd[:, 26] = np.log1p(Xd[:, 26]) / 3.0
    return np.hstack([Xd, X_extra.astype(np.float32)])


def add_e5_prefix(texts):
    return ["passage: " + t for t in texts]


class MultiTaskTextDataset(Dataset):
    def __init__(self, texts, feats, yl=None, yt=None, ya=None):
        self.texts = texts
        self.feats = feats
        self.yl = yl
        self.yt = yt
        self.ya = ya

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.yl is not None:
            return self.texts[idx], self.feats[idx], self.yl[idx], self.yt[idx], self.ya[idx]
        return self.texts[idx], self.feats[idx]


class E5MultiTaskModel(nn.Module):
    """Dual CLS+Mean pooled intfloat/multilingual-e5-small + 27 structural features with multi-sample dropout."""

    def __init__(self, model_name, n_feats=27):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True)
        h = self.backbone.config.hidden_size
        self.fuse = nn.Sequential(
            nn.Linear(2 * h + n_feats, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.drops = nn.ModuleList([nn.Dropout(p) for p in (0.1, 0.2, 0.3)])
        self.head_label = nn.Linear(256, 6)
        self.head_tech = nn.Linear(256, 6)
        self.head_action = nn.Linear(256, 6)

    def forward(self, input_ids, attention_mask, feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_emb = out[:, 0]
        mask = attention_mask.unsqueeze(-1).type_as(out)
        mean_emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        z = self.fuse(torch.cat([cls_emb, mean_emb, feats], dim=-1))
        if self.training:
            ll = sum(self.head_label(d(z)) for d in self.drops) / len(self.drops)
            lt = sum(self.head_tech(d(z)) for d in self.drops) / len(self.drops)
            la = sum(self.head_action(d(z)) for d in self.drops) / len(self.drops)
            return ll, lt, la
        return self.head_label(z), self.head_tech(z), self.head_action(z)


def make_collate(tok, max_len=192, train=True):
    def collate(batch):
        texts = add_e5_prefix([b[0] for b in batch])
        feats = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        if train:
            yl = torch.tensor([b[2] for b in batch], dtype=torch.long)
            yt = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
            ya = torch.tensor([b[4] for b in batch], dtype=torch.long)
            return enc["input_ids"], enc["attention_mask"], feats, yl, yt, ya
        return enc["input_ids"], enc["attention_mask"], feats
    return collate


class BucketBatchSampler:
    """Group similar-length examples into batches to eliminate padding waste and WDDM VRAM spill."""

    def __init__(self, texts, bs, seed=103):
        self.lens = np.array([len(t) for t in texts])
        self.bs = bs
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        n = len(self.lens)
        noisy = self.lens + self.rng.uniform(0, 12, size=n)
        order = np.argsort(noisy)
        batches = [order[i:i + self.bs].tolist() for i in range(0, n, self.bs)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lens) + self.bs - 1) // self.bs


@torch.no_grad()
def predict_texts(mdl, tok, texts, feats, bs=128, max_len=160):
    mdl.eval()
    order = np.argsort([len(x) for x in texts])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    s_texts = [texts[i] for i in order]
    s_feats = feats[order]

    pl_list, pt_list, pa_list = [], [], []
    for i in range(0, len(s_texts), bs):
        batch_txt = add_e5_prefix(s_texts[i:i+bs])
        enc = tok(batch_txt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        ids = enc["input_ids"].to("cuda", non_blocking=True)
        msk = enc["attention_mask"].to("cuda", non_blocking=True)
        ft = torch.tensor(s_feats[i:i+bs], dtype=torch.float32, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ll, lt, la = mdl(ids, msk, ft)
        pl_list.append(torch.softmax(ll.float(), dim=-1).cpu().numpy())
        pt_list.append(torch.sigmoid(lt.float()).cpu().numpy())
        pa_list.append(torch.softmax(la.float(), dim=-1).cpu().numpy())
        if (i // bs) % 400 == 399:
            print(f"  Infer {i+bs}/{len(s_texts)}", flush=True)
    return np.vstack(pl_list)[inv], np.vstack(pt_list)[inv], np.vstack(pa_list)[inv]


def train_loop(mdl, tok, texts, feats, yl, yt, ya, epochs=2, bs=24, accum=2, lr=4e-5, max_len=160):
    ds = MultiTaskTextDataset(texts, feats, yl, yt, ya)
    sampler = BucketBatchSampler(texts, bs=bs, seed=SEED)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate(tok, max_len, True))
    total_opt_steps = max(1, (len(dl) + accum - 1) // accum) * epochs
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(10, total_opt_steps // 10), num_training_steps=total_opt_steps)
    scaler = torch.amp.GradScaler("cuda")
    ce_l = nn.CrossEntropyLoss(label_smoothing=0.03)
    ce_a = nn.CrossEntropyLoss(label_smoothing=0.03)
    bce_t = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        mdl.train()
        tot_loss = 0.0
        t0 = time.time()
        opt.zero_grad()
        for step, (ids, msk, ft, byl, byt, bya) in enumerate(dl):
            ids, msk, ft = ids.to("cuda"), msk.to("cuda"), ft.to("cuda")
            byl, byt, bya = byl.to("cuda"), byt.to("cuda"), bya.to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                ll, lt, la = mdl(ids, msk, ft)
                loss = (1.2 * ce_l(ll, byl) + 0.8 * bce_t(lt, byt) + 0.7 * ce_a(la, bya)) / accum
            scaler.scale(loss).backward()
            tot_loss += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    sched.step()
                opt.zero_grad()
            if (step + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  Ep {ep+1} step {step+1}/{len(dl)} | {((step+1)*bs)/max(el,1e-6):.1f} rows/s | elapsed={el:.1f}s", flush=True)
        print(f"  Epoch {ep+1}/{epochs} | loss={tot_loss/len(dl):.4f} ({time.time()-t0:.1f}s)", flush=True)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
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

    ckpt_a = "preli/cache/me5small_stageA.pt"
    mdl = E5MultiTaskModel(MODEL_NAME, n_feats=s_feats.shape[1]).to("cuda")
    if not os.path.exists(ckpt_a):
        print("--- Stage A: Training mE5-small on 20k v3 synthetic (2 ep, bucketed bs=24x2) ---", flush=True)
        idx_a = np.random.RandomState(SEED).choice(len(syn_texts), size=20000, replace=False)
        train_loop(mdl, tok, [syn_texts[i] for i in idx_a], syn_feats[idx_a], syn_yl[idx_a], syn_yt[idx_a], syn_ya[idx_a], epochs=2, bs=24, accum=2, lr=4e-5, max_len=160)
        torch.save(mdl.state_dict(), ckpt_a)
    else:
        mdl.load_state_dict(torch.load(ckpt_a, map_location="cuda"))

    pl0, pt0, pa0 = predict_texts(mdl, tok, s_texts, s_feats, bs=128, max_len=160)
    d0_l, d0_t, d0_a = decode_consistent(pl0, pt0, pa0)
    r0 = eval_multitask(s_yl, d0_l, s_yt, d0_t, s_ya, d0_a, pl0.max(1))
    print("[mE5-small Stage A Zero-Shot on sample.csv]:", {k: round(v, 2) for k, v in r0.items()}, flush=True)

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_l, oof_t, oof_a = np.zeros((86, 6), dtype=np.float32), np.zeros((86, 6), dtype=np.float32), np.zeros((86, 6), dtype=np.float32)
    adv_l, adv_t, adv_a = np.zeros((len(adv_clean), 6), dtype=np.float32), np.zeros((len(adv_clean), 6), dtype=np.float32), np.zeros((len(adv_clean), 6), dtype=np.float32)

    s_raw_list = s_df["text"].tolist()
    for fold, (tr, va) in enumerate(sgkf.split(s_feats, s_yl, groups)):
        mdl.load_state_dict(torch.load(ckpt_a, map_location="cuda"))
        rng_f = random.Random(SEED + 500 + fold)
        rep_idx = np.random.RandomState(SEED + fold).choice(len(syn_texts), size=1000, replace=False)
        tr_clean_rep = np.repeat(tr, 3)
        tr_adv_idx = np.repeat(tr, 1)
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
        train_loop(mdl, tok, f_texts, f_feats, f_yl, f_yt, f_ya, epochs=2, bs=24, accum=1, lr=1e-5, max_len=160)
        oof_l[va], oof_t[va], oof_a[va] = predict_texts(mdl, tok, [s_texts[i] for i in va], s_feats[va], bs=128, max_len=160)
        am = np.isin(adv_p, va)
        adv_l[am], adv_t[am], adv_a[am] = predict_texts(mdl, tok, [adv_clean[i] for i in np.where(am)[0]], adv_feats[am], bs=128, max_len=160)

    oof_pl, oof_pt, oof_pa = decode_consistent(oof_l, oof_t, oof_a)
    adv_pl, adv_pt, adv_pa = decode_consistent(adv_l, adv_t, adv_a)
    r_oof = eval_multitask(s_yl, oof_pl, s_yt, oof_pt, s_ya, oof_pa, oof_l.max(1))
    r_adv = eval_multitask(s_yl[adv_p], adv_pl, s_yt[adv_p], adv_pt, s_ya[adv_p], adv_pa, adv_l.max(1))
    print("[mE5-small Stage B] Group-CV :", {k: round(v, 2) for k, v in r_oof.items()}, flush=True)
    print("[mE5-small Stage B] Robust-CV:", {k: round(v, 2) for k, v in r_adv.items()}, flush=True)
    print("Errors of mE5-small Stage B Group-CV:", flush=True)
    for i in range(len(s_df)):
        if oof_pl[i] != s_yl[i] or oof_pa[i] != s_ya[i]:
            print(f"  id={s_df.id.iloc[i]} L:{s_yl[i]}->{oof_pl[i]}({oof_l[i].max():.2f}) A:{s_ya[i]}->{oof_pa[i]} | {s_df.text.iloc[i][:90]}", flush=True)
    np.savez("preli/cache/oof_me5small.npz", oof_l=oof_l, oof_t=oof_t, oof_a=oof_a, adv_l=adv_l, adv_t=adv_t, adv_a=adv_a)

    mdl.load_state_dict(torch.load(ckpt_a, map_location="cuda"))
    rng_full = random.Random(SEED + 999)
    rep_idx = np.random.RandomState(SEED + 999).choice(len(syn_texts), size=1200, replace=False)
    all_clean_rep = np.repeat(np.arange(len(s_texts)), 3)
    all_adv_idx = np.repeat(np.arange(len(s_texts)), 1)
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
    train_loop(mdl, tok, f_texts, f_feats, f_yl, f_yt, f_ya, epochs=2, bs=24, accum=1, lr=1e-5, max_len=160)
    torch.save(mdl.state_dict(), "preli/cache/me5small_full.pt")
    del syn_df, syn_texts, syn_feats, f_texts, f_feats
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    te_feats = scale_feats(np.load("preli/cache/test_feats.npy"), np.load("preli/cache/test_extra.npy"))
    te_l = np.memmap("preli/cache/te_me5_l.dat", dtype="float32", mode="w+", shape=(500000, 6))
    te_t = np.memmap("preli/cache/te_me5_t.dat", dtype="float32", mode="w+", shape=(500000, 6))
    te_a = np.memmap("preli/cache/te_me5_a.dat", dtype="float32", mode="w+", shape=(500000, 6))
    offset = 0
    for chunk in pd.read_csv("preli/cache/test_clean.csv", chunksize=50000):
        n = len(chunk)
        cl = chunk["clean_text"].fillna("").astype(str).tolist()
        pl, pt, pa = predict_texts(mdl, tok, cl, te_feats[offset:offset+n], bs=128, max_len=160)
        te_l[offset:offset+n] = pl
        te_t[offset:offset+n] = pt
        te_a[offset:offset+n] = pa
        offset += n
        print(f"500k mE5-small inferred {offset}/500000 ({time.time()-t0:.1f}s)", flush=True)

    for fn, arr in [("te_me5_l.npy", te_l), ("te_me5_t.npy", te_t), ("te_me5_a.npy", te_a)]:
        arr.flush()
        np.save(f"preli/cache/{fn}", np.array(arr))
        if getattr(arr, "_mmap", None) is not None:
            arr._mmap.close()
    del te_l, te_t, te_a
    gc.collect()
    for fn in ["te_me5_l.dat", "te_me5_t.dat", "te_me5_a.dat"]:
        os.remove(f"preli/cache/{fn}")
    print(f"ALL DONE mE5-small 500k in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
