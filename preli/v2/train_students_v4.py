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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
CKPT_DIR = CACHE_DIR / "ckpt_v4"
OUT_DIR = CACHE_DIR / "preds_v4"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V2_DIR))

from cv_ood import tech_str_to_vec, decode_consistent, score_section10_proxy, get_balanced_3folds
from normalizer_v4 import format_turn_aware, bn_to_en_roman_v4
from torch_utils_v4 import (
    CACHE_DIR_HF, scale_feats_v4, MultiTaskKDDataset, BucketBatchSampler,
    make_collate_fn, HeadConditionedTransformer, compute_multitask_loss,
    predict_transformer, free_cuda,
)


def compute_weights(yl, ya):
    cnt_l = np.bincount(yl, minlength=6).astype(np.float32)
    w_l = np.sqrt(cnt_l.sum() / (6.0 * np.maximum(cnt_l, 1.0)))
    w_l = w_l / w_l.mean()
    scam_ya = ya[(ya >= 1) & (ya <= 5)]
    cnt_a = np.ones(6, dtype=np.float32)
    for c in range(1, 6):
        cnt_a[c] = max(1.0, float((scam_ya == c).sum()))
    w_a = np.ones(6, dtype=np.float32)
    w_a[1:] = np.sqrt(cnt_a[1:].sum() / (5.0 * cnt_a[1:]))
    w_a[1:] = np.clip(w_a[1:] / w_a[1:].mean(), 0.7, 2.2)
    return torch.tensor(w_l, dtype=torch.float32, device="cuda"), torch.tensor(w_a, dtype=torch.float32, device="cuda")


def prep_text_list(clean_texts, mode="plain"):
    if mode == "e5":
        return ["passage: " + format_turn_aware(str(t)) for t in clean_texts]
    if mode == "bge":
        return ["Represent this sentence for searching relevant passages: " + bn_to_en_roman_v4(format_turn_aware(str(t))) for t in clean_texts]
    return [format_turn_aware(str(t)) for t in clean_texts]


def train_student_model(
    model_name, tag, tr_texts, tr_feats, tr_yl, tr_yt, tr_ya,
    tr_soft_l=None, tr_soft_t=None, tr_soft_a=None, tr_weights=None,
    va_texts=None, va_feats=None, va_yl=None, n_sample_va=43,
    pool_mode="mean", bs=32, max_len=152, epochs=2, lr_bb=3.5e-5, lr_hd=2.0e-4, seed=42, kd_alpha=0.45,
):
    free_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    ckpt_path = CKPT_DIR / f"{tag}.pt"
    oof_path = OUT_DIR / f"oof_{tag}.npz"
    if va_texts is not None and ckpt_path.exists() and oof_path.exists():
        z = np.load(oof_path)
        print(f"  [{tag}] Resuming cached {oof_path.name}!", flush=True)
        return (z["p_l"], z["p_t"], z["p_a"]), ckpt_path

    has_soft = tr_soft_l is not None
    has_w = tr_weights is not None
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = HeadConditionedTransformer(
        model_name, n_feats=tr_feats.shape[1], pool_mode=pool_mode, proj_dim=256, num_freeze_layers=0
    ).cuda()

    ds = MultiTaskKDDataset(
        tr_texts, tr_feats, tr_yl, tr_yt, tr_ya,
        soft_l=tr_soft_l, soft_t=tr_soft_t, soft_a=tr_soft_a, weights=tr_weights,
    )
    sampler = BucketBatchSampler(tr_texts, bs=bs, seed=seed, jitter=16.0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(tok, max_len=max_len, has_labels=True, has_soft=has_soft, has_weights=has_w))

    mon_texts = va_texts
    mon_feats = va_feats
    mon_yl = va_yl
    if va_texts is not None and len(va_texts) > 3200 + n_sample_va:
        rng_v = np.random.RandomState(seed)
        sub_tail = np.sort(n_sample_va + rng_v.choice(len(va_texts) - n_sample_va, size=3200, replace=False))
        mon_idx = np.concatenate([np.arange(n_sample_va), sub_tail])
        mon_texts = [va_texts[i] for i in mon_idx]
        mon_feats = va_feats[mon_idx]
        mon_yl = va_yl[mon_idx]

    total_steps = max(1, len(loader) * epochs)
    opt = torch.optim.AdamW(model.param_groups(lr_backbone=lr_bb, lr_head=lr_hd, weight_decay=0.01))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(20, int(0.08 * total_steps)), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    w_l, w_a = compute_weights(tr_yl, tr_ya)

    best_score = -1.0

    for ep in range(epochs):
        model.train()
        rloss = 0.0
        for batch in loader:
            ids = batch[0].cuda(non_blocking=False)
            mask = batch[1].cuda(non_blocking=False)
            f_t = batch[2].cuda(non_blocking=False)
            yl = batch[3].cuda(non_blocking=False)
            yt = batch[4].cuda(non_blocking=False)
            ya = batch[5].cuda(non_blocking=False)
            pos = 6
            sl = st = sa = sw = None
            if has_soft:
                sl = batch[pos].cuda(non_blocking=False)
                st = batch[pos + 1].cuda(non_blocking=False)
                sa = batch[pos + 2].cuda(non_blocking=False)
                pos += 3
            if has_w:
                sw = batch[pos].cuda(non_blocking=False)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                ll, lt, la = model(ids, mask, f_t)
                loss = compute_multitask_loss(
                    ll, lt, la, yl, yt, ya,
                    label_weights=w_l, act_weights=w_a, sample_w=sw,
                    soft_l=sl, soft_t=st, soft_a=sa, kd_alpha=kd_alpha,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            rloss += float(loss.item())

        if va_texts is not None:
            p_ml, _, _ = predict_transformer(model, tok, mon_texts, mon_feats, bs=max(bs * 2, 64), max_len=max_len, device="cuda")
            mf1 = 100.0 * f1_score(mon_yl, p_ml.argmax(axis=1), average="macro", zero_division=0)
            print(f"  [{tag}] ep={ep+1}/{epochs} loss={rloss/max(1,len(loader)):.4f} OOD_mon_F1={mf1:.2f} elapsed={time.time()-t0:.1f}s", flush=True)
            if mf1 >= best_score:
                best_score = mf1
                torch.save({"state_dict": model.state_dict(), "model_name": model_name, "pool_mode": pool_mode}, ckpt_path)
        else:
            print(f"  [{tag}] ep={ep+1}/{epochs} loss={rloss/max(1,len(loader)):.4f} elapsed={time.time()-t0:.1f}s", flush=True)
            torch.save({"state_dict": model.state_dict(), "model_name": model_name, "pool_mode": pool_mode}, ckpt_path)

    best_va = None
    if va_texts is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model = model.cuda().eval()
        p_l, p_t, p_a = predict_transformer(model, tok, va_texts, va_feats, bs=max(bs * 2, 64), max_len=max_len, device="cuda")
        np.savez_compressed(oof_path, p_l=p_l, p_t=p_t, p_a=p_a)
        best_va = (p_l, p_t, p_a)

    del model, opt, sched, scaler, loader
    free_cuda()
    return best_va, ckpt_path


def stream_predict_500k(ckpt_path, out_npz_path, text_mode="plain", bs=96, max_len=152):
    free_cuda()
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mname = ckpt["model_name"]
    tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = HeadConditionedTransformer(mname, n_feats=58, pool_mode=ckpt["pool_mode"], proj_dim=256)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda().eval()

    te_f = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")
    n_te = len(te_f)
    p_l = np.zeros((n_te, 6), dtype=np.float32)
    p_t = np.zeros((n_te, 6), dtype=np.float32)
    p_a = np.zeros((n_te, 6), dtype=np.float32)

    off = 0
    for df_c in pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=50000):
        m = len(df_c)
        txts = prep_text_list(df_c["clean_text"].fillna("").tolist(), mode=text_mode)
        f_c = scale_feats_v4(te_f[off:off + m])
        pl, pt, pa = predict_transformer(model, tok, txts, f_c, bs=bs, max_len=max_len, device="cuda")
        p_l[off:off + m] = pl
        p_t[off:off + m] = pt
        p_a[off:off + m] = pa
        off += m
        print(f"  [{out_npz_path.name}] {off}/{n_te} in {time.time()-t0:.1f}s", flush=True)

    np.savez_compressed(out_npz_path, p_l=p_l, p_t=p_t, p_a=p_a)
    del model, ckpt
    free_cuda()


def _stratified_synth_idx(yl, max_n, rng):
    if max_n is None or len(yl) <= max_n:
        return np.arange(len(yl), dtype=np.int64)
    parts = []
    for c in range(6):
        idx_c = np.where(yl == c)[0]
        if len(idx_c) == 0:
            continue
        take = min(len(idx_c), max(1, int(round(max_n * len(idx_c) / len(yl)))))
        parts.append(rng.choice(idx_c, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def _balanced_dp_idx(yl, conf, max_n, rng):
    if max_n is None or len(yl) <= max_n:
        return np.arange(len(yl), dtype=np.int64)
    cap_c = max(1, max_n // 5)
    parts = []
    for c in range(6):
        idx_c = np.where(yl == c)[0]
        if len(idx_c) <= cap_c:
            parts.append(idx_c)
        else:
            order = idx_c[np.argsort(-conf[idx_c])]
            n_top = int(cap_c * 0.85)
            n_rem = cap_c - n_top
            top_part = order[:n_top]
            rem_part = rng.choice(order[n_top:], size=n_rem, replace=False) if n_rem > 0 else np.empty(0, dtype=np.int64)
            parts.append(np.concatenate([top_part, rem_part]))
    kept = np.sort(np.concatenate(parts))
    if len(kept) > max_n:
        kept = np.sort(rng.choice(kept, size=max_n, replace=False))
    return kept


def build_distill_corpus(sdf, sf, sy, syf, dp, text_mode="plain", seed=42, max_synth=26000, max_dp=34000):
    """Build combined inductive distillation training set: balanced synth + sample*4 + domain pseudo."""
    rng = np.random.RandomState(seed)

    s_yl = sdf["label"].values.astype(np.int64)
    s_yt = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]]).astype(np.float32)
    s_ya = sdf["action"].values.astype(np.int64)

    sy_yl_all = sy["label"].values.astype(np.int64)
    sy_keep = _stratified_synth_idx(sy_yl_all, max_synth, rng)
    sy_sub = sy.iloc[sy_keep]
    syf_sub = syf[sy_keep]
    sy_yl = sy_yl_all[sy_keep]
    sy_yt = np.vstack([tech_str_to_vec(x) for x in sy_sub["techniques"]]).astype(np.float32)
    sy_ya = sy_sub["action"].values.astype(np.int64)

    dp_yl_all = dp["y_l"].astype(np.int64)
    dp_conf_all = dp["conf"].astype(np.float32)
    dp_keep = _balanced_dp_idx(dp_yl_all, dp_conf_all, max_dp, rng)

    s_clean = sdf["clean_text"].fillna("").tolist()
    sy_clean = sy_sub["clean_text"].fillna("").tolist()
    dp_clean_all = dp["clean_texts"].tolist()
    dp_clean = [dp_clean_all[i] for i in dp_keep]

    dp_feats = scale_feats_v4(np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")[dp["proto_idx"][dp_keep]])
    all_texts = prep_text_list(sy_clean, text_mode) + prep_text_list(s_clean, text_mode) * 4 + prep_text_list(dp_clean, text_mode)
    all_feats = np.vstack([syf_sub, sf, sf, sf, sf, dp_feats])

    all_yl = np.concatenate([sy_yl, s_yl, s_yl, s_yl, s_yl, dp_yl_all[dp_keep]])
    all_yt = np.vstack([sy_yt, s_yt, s_yt, s_yt, s_yt, dp["y_t"][dp_keep].astype(np.float32)])
    all_ya = np.concatenate([sy_ya, s_ya, s_ya, s_ya, s_ya, dp["y_a"][dp_keep].astype(np.int64)])

    # For hard-labeled source rows, construct sharpened 1-hot soft targets so KD loss is well-defined uniformly
    def onehot6(y):
        oh = np.full((len(y), 6), 0.005, dtype=np.float32)
        oh[np.arange(len(y)), y] = 0.975
        return oh

    n_src = len(sy_yl) + 4 * len(s_yl)
    src_yl = np.concatenate([sy_yl, s_yl, s_yl, s_yl, s_yl])
    src_yt = np.vstack([sy_yt, s_yt, s_yt, s_yt, s_yt])
    src_ya = np.concatenate([sy_ya, s_ya, s_ya, s_ya, s_ya])
    src_sl = onehot6(src_yl)
    src_st = src_yt * 0.94 + 0.03
    src_sa = onehot6(src_ya)

    all_sl = np.vstack([src_sl, dp["p_l"][dp_keep].astype(np.float32)])
    all_st = np.vstack([src_st, dp["p_t"][dp_keep].astype(np.float32)])
    all_sa = np.vstack([src_sa, dp["p_a"][dp_keep].astype(np.float32)])

    # Confidence weighting: source=1.0, pseudo scaled smoothly [0.35..1.15] by teacher consensus
    w_src = np.ones(n_src, dtype=np.float32)
    w_dp = np.clip((dp_conf_all[dp_keep] - 0.35) / 0.65, 0.35, 1.15).astype(np.float32)
    all_w = np.concatenate([w_src, w_dp])
    return all_texts, all_feats, all_yl, all_yt, all_ya, all_sl, all_st, all_sa, all_w


def run_selective_mdeberta_escalation(top_k=30000):
    """Inductive Teacher Escalation: run mDeBERTa-v3-base live on highest-uncertainty/disagreement rows."""
    p_m = np.load(OUT_DIR / "test_mminilm_v4.npz")
    p_e = np.load(OUT_DIR / "test_me5small_v4.npz")
    p_b = np.load(OUT_DIR / "test_bge_v4.npz")
    da_p = OUT_DIR / "test_da_tab_v4.npz"
    p_base = np.load(da_p if da_p.exists() else (OUT_DIR / "test_baseline_v4.npz"))

    pb_l = 0.32 * p_base["lin_l"] + 0.26 * p_base["lgb_l"] + 0.22 * p_base["xgb_l"] + 0.20 * p_base["cb_l"]
    blend_l = 0.34 * p_m["p_l"] + 0.30 * p_e["p_l"] + 0.20 * p_b["p_l"] + 0.16 * pb_l

    # Purely inductive uncertainty + cross-student disagreement score (works identically on any unseen dataset)
    ent = -np.sum(blend_l * np.log(np.clip(blend_l, 1e-6, 1.0)), axis=1)
    dis_l = (
        (p_m["p_l"].argmax(1) != p_e["p_l"].argmax(1)).astype(np.float32)
        + (p_m["p_l"].argmax(1) != p_b["p_l"].argmax(1)).astype(np.float32)
        + (p_m["p_l"].argmax(1) != pb_l.argmax(1)).astype(np.float32)
    )
    dis_a = (p_m["p_a"].argmax(1) != p_e["p_a"].argmax(1)).astype(np.float32) * np.isin(blend_l.argmax(1), [1, 2, 3, 4])
    score = ent + 0.65 * dis_l + 0.30 * dis_a + (1.0 - blend_l.max(1))

    esc_idx = np.sort(np.argsort(-score)[:top_k].astype(np.int32))
    esc_mask = np.zeros(len(score), dtype=bool)
    esc_mask[esc_idx] = True

    esc_clean = []
    off = 0
    for df_c in pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=50000):
        m = len(df_c)
        loc = esc_mask[off:off + m]
        if loc.any():
            esc_clean.extend(df_c["clean_text"].fillna("").values[loc].tolist())
        off += m
    esc_texts = [format_turn_aware(t) for t in esc_clean]
    esc_feats = scale_feats_v4(np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")[esc_idx])

    del p_m, p_e, p_b, p_base, pb_l, blend_l
    free_cuda()

    full_ckpt = CKPT_DIR / "mdeberta_full_distill.pt"
    if full_ckpt.exists():
        # Prefer full-scenario mDeBERTa (zero half-fold blind spot!)
        ck = torch.load(full_ckpt, map_location="cpu", weights_only=False)
        mname = ck["model_name"]
        tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True)
        mdl = HeadConditionedTransformer(mname, n_feats=58, pool_mode=ck["pool_mode"], proj_dim=256, num_freeze_layers=5)
        mdl.load_state_dict(ck["state_dict"], strict=True)
        mdl = mdl.cuda().eval()
        pl, pt, pa = predict_transformer(mdl, tok, esc_texts, esc_feats, bs=48, max_len=152)
        del mdl, ck
        free_cuda()
    else:
        from train_teachers_v4 import load_predict_ckpt
        l0, t0_, a0 = load_predict_ckpt(CKPT_DIR / "mdeberta_f0.pt", esc_texts, esc_feats, bs=48, max_len=152)
        l1, t1_, a1 = load_predict_ckpt(CKPT_DIR / "mdeberta_f1.pt", esc_texts, esc_feats, bs=48, max_len=152)
        pl, pt, pa = 0.5 * (l0 + l1), 0.5 * (t0_ + t1_), 0.5 * (a0 + a1)

    np.savez_compressed(
        OUT_DIR / "test_mdeberta_escalate_v4.npz",
        esc_idx=esc_idx, p_l=pl, p_t=pt, p_a=pa,
    )
    print(f"Saved selective inductive mDeBERTa escalation on top {len(esc_idx)} uncertain rows!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="all", help="all|ood|mminilm|me5small|bge|mdeberta_full|escalate")
    ap.add_argument("--esc-k", type=int, default=26000)
    ap.add_argument("--force", action="store_true", help="Re-run even if output cache already exists")
    args = ap.parse_args()

    sdf = pd.read_csv(CACHE_DIR / "sample_v4.csv")
    sf = scale_feats_v4(np.load(CACHE_DIR / "sample_v4_feats.npy"))
    sy = pd.read_csv(CACHE_DIR / "synth_64k_v4.csv")
    syf = scale_feats_v4(np.load(CACHE_DIR / "synth_64k_v4_feats.npy"))

    s_yl = sdf["label"].values.astype(np.int64)
    s_yt = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]]).astype(np.float32)
    s_ya = sdf["action"].values.astype(np.int64)
    s_folds = get_balanced_3folds(sdf["scenario_family"].values)

    sy_yl = sy["label"].values.astype(np.int64)
    sy_yt = np.vstack([tech_str_to_vec(x) for x in sy["techniques"]]).astype(np.float32)
    sy_ya = sy["action"].values.astype(np.int64)
    sy_folds = get_balanced_3folds(sy["scenario_family"].values)

    # 1. Zero-Leakage Balanced 3-Fold Scenario-Disjoint Student OOF (67% of every class in train!)
    oof_out = OUT_DIR / "oof_students_v4.npz"
    if args.only in ("all", "ood"):
        if oof_out.exists() and not args.force:
            print(f"[ood] {oof_out.name} already exists, skipping (pass --force to re-run).", flush=True)
        else:
            specs = [
                ("mminilm", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "plain", "mean", 48, 4.0e-5),
                ("me5small", "intfloat/multilingual-e5-small", "e5", "cls_mean", 36, 4.0e-5),
                ("bge", "BAAI/bge-small-en-v1.5", "bge", "cls_mean", 48, 4.0e-5),
            ]
            oof_dict = {}
            rng_sub = np.random.RandomState(42)
            for key, mname, tmode, pmode, bs, lrbb in specs:
                s_txts = prep_text_list(sdf["clean_text"].fillna(""), tmode)
                sy_txts = prep_text_list(sy["clean_text"].fillna(""), tmode)
                o_s_l = np.zeros((len(sdf), 6), dtype=np.float32)
                o_s_t = np.zeros((len(sdf), 6), dtype=np.float32)
                o_s_a = np.zeros((len(sdf), 6), dtype=np.float32)
                o_sy_l = np.zeros((len(sy), 6), dtype=np.float32)
                o_sy_t = np.zeros((len(sy), 6), dtype=np.float32)
                o_sy_a = np.zeros((len(sy), 6), dtype=np.float32)
                for fold in (0, 1, 2):
                    tr_sy = np.where(sy_folds != fold)[0]
                    if len(tr_sy) > 14000:
                        tr_sy = np.sort(rng_sub.choice(tr_sy, size=14000, replace=False))
                    va_sy = np.where(sy_folds == fold)[0]
                    tr_s = np.where(s_folds != fold)[0]
                    va_s = np.where(s_folds == fold)[0]
                    tr_tx = [sy_txts[i] for i in tr_sy] + [s_txts[i] for i in tr_s] * 3
                    tr_f = np.vstack([syf[tr_sy], sf[tr_s], sf[tr_s], sf[tr_s]])
                    va_tx = [s_txts[i] for i in va_s] + [sy_txts[i] for i in va_sy]
                    va_f = np.vstack([sf[va_s], syf[va_sy]])
                    (pl, pt, pa), _ = train_student_model(
                        mname, f"{key}_oof3_f{fold}",
                        tr_tx, tr_f,
                        np.concatenate([sy_yl[tr_sy], s_yl[tr_s], s_yl[tr_s], s_yl[tr_s]]),
                        np.vstack([sy_yt[tr_sy], s_yt[tr_s], s_yt[tr_s], s_yt[tr_s]]),
                        np.concatenate([sy_ya[tr_sy], s_ya[tr_s], s_ya[tr_s], s_ya[tr_s]]),
                        va_texts=va_tx, va_feats=va_f, va_yl=np.concatenate([s_yl[va_s], sy_yl[va_sy]]), n_sample_va=len(va_s),
                        pool_mode=pmode, bs=bs, max_len=144, epochs=2, lr_bb=lrbb, seed=300 + fold, kd_alpha=0.0,
                    )
                    ns = len(va_s)
                    o_s_l[va_s], o_s_t[va_s], o_s_a[va_s] = pl[:ns], pt[:ns], pa[:ns]
                    o_sy_l[va_sy], o_sy_t[va_sy], o_sy_a[va_sy] = pl[ns:], pt[ns:], pa[ns:]
                oof_dict[f"s_{key}_l"] = o_s_l; oof_dict[f"s_{key}_t"] = o_s_t; oof_dict[f"s_{key}_a"] = o_s_a
                oof_dict[f"sy_{key}_l"] = o_sy_l; oof_dict[f"sy_{key}_t"] = o_sy_t; oof_dict[f"sy_{key}_a"] = o_sy_a
            np.savez_compressed(oof_out, **oof_dict)

    # 2. Domain-Adapted Inductive Students (distilled from ultra-clean consensus domain_pseudo_v5)
    dp_path = CACHE_DIR / "domain_pseudo_v5.npz"
    if not dp_path.exists():
        dp_path = CACHE_DIR / "domain_pseudo_v4.npz"

    if args.only in ("all", "mminilm", "me5small", "bge"):
        dp = np.load(dp_path, allow_pickle=True)
        run_list = []
        if args.only in ("all", "mminilm"):
            run_list.append(("mminilm", "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", "plain", "mean", 32, 112, 15000, 25000, 2, 3.5e-5))
        if args.only in ("all", "me5small"):
            run_list.append(("me5small", "intfloat/multilingual-e5-small", "e5", "cls_mean", 24, 128, 11000, 23000, 1, 4.2e-5))
        if args.only in ("all", "bge"):
            run_list.append(("bge", "BAAI/bge-small-en-v1.5", "bge", "cls_mean", 32, 160, 11000, 23000, 1, 4.2e-5))

        for key, mname, tmode, pmode, bs_tr, bs_inf, m_syn, m_dp, ep_tr, lr_tr in run_list:
            pred_out = OUT_DIR / f"test_{key}_v4.npz"
            if pred_out.exists() and not args.force:
                print(f"[{key}] {pred_out.name} already exists, skipping (pass --force to re-run).", flush=True)
                continue
            ckpt = CKPT_DIR / f"{key}_distill.pt"
            if ckpt.exists() and not args.force:
                print(f"[{key}] Reusing existing checkpoint {ckpt.name} for 500k inference.", flush=True)
            else:
                tr_tx, tr_f, tr_yl, tr_yt, tr_ya, tr_sl, tr_st, tr_sa, tr_w = build_distill_corpus(
                    sdf, sf, sy, syf, dp, text_mode=tmode, max_synth=m_syn, max_dp=m_dp,
                )
                print(f"[{key}] Distill corpus rows={len(tr_tx)} ep={ep_tr}", flush=True)
                _, ckpt = train_student_model(
                    mname, f"{key}_distill", tr_tx, tr_f, tr_yl, tr_yt, tr_ya,
                    tr_soft_l=tr_sl, tr_soft_t=tr_st, tr_soft_a=tr_sa, tr_weights=tr_w,
                    pool_mode=pmode, bs=bs_tr, max_len=144, epochs=ep_tr, lr_bb=lr_tr, seed=500, kd_alpha=0.45,
                )
            stream_predict_500k(ckpt, pred_out, text_mode=tmode, bs=bs_inf, max_len=144)

    if args.only in ("all", "mdeberta_full"):
        md_ckpt = CKPT_DIR / "mdeberta_full_distill.pt"
        if md_ckpt.exists() and not args.force:
            print(f"[mdeberta_full] {md_ckpt.name} already exists, skipping.", flush=True)
        else:
            dp = np.load(dp_path, allow_pickle=True)
            # Capped cleanly at ~20,344 class-balanced rows (8k synth + 4*86 sample + 12k real dp) -> ~24m total!
            tr_tx, tr_f, tr_yl, tr_yt, tr_ya, tr_sl, tr_st, tr_sa, tr_w = build_distill_corpus(
                sdf, sf, sy, syf, dp, text_mode="plain", seed=2026, max_synth=8000, max_dp=12000,
            )
            print(f"[mdeberta_full] Balanced corpus rows={len(tr_tx)}", flush=True)
            tok_md = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", cache_dir=CACHE_DIR_HF, local_files_only=True)
            mdl_md = HeadConditionedTransformer("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", n_feats=58, pool_mode="mean", proj_dim=256, num_freeze_layers=5).cuda()
            ds_md = MultiTaskKDDataset(
                tr_tx, tr_f, tr_yl, tr_yt, tr_ya,
                soft_l=tr_sl, soft_t=tr_st, soft_a=tr_sa, weights=tr_w,
            )
            loader_md = DataLoader(ds_md, batch_sampler=BucketBatchSampler(tr_tx, bs=8, seed=777), collate_fn=make_collate_fn(tok_md, max_len=152, has_labels=True, has_soft=True, has_weights=True))
            opt_md = torch.optim.AdamW(mdl_md.param_groups(lr_backbone=2.5e-5, lr_head=1.8e-4))
            sched_md = get_cosine_schedule_with_warmup(opt_md, num_warmup_steps=80, num_training_steps=max(1, len(loader_md) // 4))
            sc_md = torch.amp.GradScaler("cuda")
            w_l, w_a = compute_weights(tr_yl, tr_ya)
            t_md = time.time()
            mdl_md.train(); opt_md.zero_grad(set_to_none=True)
            for st_idx, b in enumerate(loader_md):
                ids, msk, ft, yl, yt, ya, sl, st, sa, sw = [x.cuda() for x in b]
                with torch.amp.autocast("cuda"):
                    ll, lt, la = mdl_md(ids, msk, ft)
                    loss = compute_multitask_loss(ll, lt, la, yl, yt, ya, label_weights=w_l, act_weights=w_a, sample_w=sw, soft_l=sl, soft_t=st, soft_a=sa, kd_alpha=0.45) / 4.0
                sc_md.scale(loss).backward()
                if (st_idx + 1) % 4 == 0 or (st_idx + 1) == len(loader_md):
                    sc_md.unscale_(opt_md); torch.nn.utils.clip_grad_norm_(mdl_md.parameters(), 1.0)
                    sc_md.step(opt_md); sc_md.update(); sched_md.step(); opt_md.zero_grad(set_to_none=True)
                if (st_idx + 1) % 500 == 0:
                    print(f"  [mdeberta_full_distill] step={st_idx+1}/{len(loader_md)} elapsed={time.time()-t_md:.1f}s", flush=True)
            torch.save({"state_dict": mdl_md.state_dict(), "model_name": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", "pool_mode": "mean"}, md_ckpt)
            del mdl_md, opt_md, sc_md, loader_md; free_cuda()

    if args.only in ("all", "escalate"):
        esc_out = OUT_DIR / "test_mdeberta_escalate_v4.npz"
        if esc_out.exists() and not args.force:
            print(f"[escalate] {esc_out.name} already exists, skipping.", flush=True)
        else:
            run_selective_mdeberta_escalation(top_k=args.esc_k)


if __name__ == "__main__":
    main()
