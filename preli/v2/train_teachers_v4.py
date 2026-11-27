import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
CKPT_DIR = CACHE_DIR / "ckpt_v4"
OUT_DIR = CACHE_DIR / "preds_v4"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V2_DIR))

from cv_ood import (
    TECH_LIST, tech_str_to_vec, decode_consistent,
    score_section10_proxy, get_scenario_disjoint_folds,
)
from normalizer_v4 import format_turn_aware
from torch_utils_v4 import (
    CACHE_DIR_HF, scale_feats_v4, MultiTaskKDDataset, BucketBatchSampler,
    make_collate_fn, HeadConditionedTransformer, compute_multitask_loss,
    predict_transformer, free_cuda,
)


def compute_class_weights(yl, ya):
    # Balanced label weights tempered with sqrt to avoid destabilizing CE while maximizing Macro-F1
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


def train_one_teacher_fold(
    model_name, tag, tr_texts, tr_feats, tr_yl, tr_yt, tr_ya,
    va_texts, va_feats, va_yl, n_sample_va=43,
    pool_mode="mean", freeze_k=6, bs=16, grad_accum=2, max_len=160,
    epochs=2, lr_bb=2.5e-5, lr_hd=1.8e-4, seed=42, add_e5_prefix=False,
):
    free_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    ckpt_path = CKPT_DIR / f"{tag}.pt"
    oof_path = OUT_DIR / f"oof_{tag}.npz"
    if ckpt_path.exists() and oof_path.exists():
        z = np.load(oof_path)
        print(f"  [{tag}] Resuming from cached {oof_path.name}!", flush=True)
        return (z["p_l"], z["p_t"], z["p_a"]), ckpt_path

    tr_in = ["passage: " + t for t in tr_texts] if add_e5_prefix else tr_texts
    va_in = ["passage: " + t for t in va_texts] if add_e5_prefix else va_texts

    # Use a fixed stratified ~3200-row slice (always including ALL sample_va rows) during intermediate epochs
    rng_v = np.random.RandomState(seed)
    if len(va_in) > 3200 + n_sample_va:
        sub_tail = np.sort(n_sample_va + rng_v.choice(len(va_in) - n_sample_va, size=3200, replace=False))
        mon_idx = np.concatenate([np.arange(n_sample_va), sub_tail])
    else:
        mon_idx = np.arange(len(va_in))
    mon_texts = [va_in[i] for i in mon_idx]
    mon_feats = va_feats[mon_idx]
    mon_yl = va_yl[mon_idx]

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = HeadConditionedTransformer(
        model_name, n_feats=tr_feats.shape[1], pool_mode=pool_mode,
        proj_dim=256, num_freeze_layers=freeze_k,
    ).cuda()

    ds = MultiTaskKDDataset(tr_in, tr_feats, tr_yl, tr_yt, tr_ya)
    sampler = BucketBatchSampler(tr_in, bs=bs, seed=seed, jitter=16.0)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate_fn(tok, max_len=max_len, has_labels=True))

    total_steps = max(1, (len(loader) * epochs) // grad_accum)
    opt = torch.optim.AdamW(model.param_groups(lr_backbone=lr_bb, lr_head=lr_hd, weight_decay=0.01))
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=max(10, int(0.08 * total_steps)), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    w_l, w_a = compute_class_weights(tr_yl, tr_ya)

    best_score = -1.0

    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, (ids, mask, f_t, yl, yt, ya) in enumerate(loader):
            ids = ids.cuda(non_blocking=False)
            mask = mask.cuda(non_blocking=False)
            f_t = f_t.cuda(non_blocking=False)
            yl = yl.cuda(non_blocking=False)
            yt = yt.cuda(non_blocking=False)
            ya = ya.cuda(non_blocking=False)

            with torch.amp.autocast("cuda"):
                ll, lt, la = model(ids, mask, f_t)
                loss = compute_multitask_loss(ll, lt, la, yl, yt, ya, label_weights=w_l, act_weights=w_a) / grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if (step + 1) % 300 == 0:
                print(f"    [{tag}] ep={ep+1} step={step+1}/{len(loader)} loss={running_loss/(step+1):.4f} elapsed={time.time()-t0:.1f}s", flush=True)

        p_ml, _, _ = predict_transformer(model, tok, mon_texts, mon_feats, bs=max(bs * 2, 32), max_len=max_len, device="cuda")
        mf1 = 100.0 * f1_score(mon_yl, p_ml.argmax(axis=1), average="macro", zero_division=0)
        print(f"  [{tag}] ep={ep+1}/{epochs} loss={running_loss/max(1,len(loader)):.4f} OOD_mon_F1={mf1:.2f} elapsed={time.time()-t0:.1f}s", flush=True)
        if mf1 >= best_score:
            best_score = mf1
            torch.save({"state_dict": model.state_dict(), "model_name": model_name, "pool_mode": pool_mode, "freeze_k": freeze_k}, ckpt_path)

    # Load best checkpoint and evaluate full OOD fold once cleanly
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda().eval()
    p_l, p_t, p_a = predict_transformer(model, tok, va_in, va_feats, bs=max(bs * 2, 32), max_len=max_len, device="cuda")
    np.savez_compressed(oof_path, p_l=p_l, p_t=p_t, p_a=p_a)

    del model, opt, sched, scaler, loader, ckpt
    free_cuda()
    return (p_l, p_t, p_a), ckpt_path


def load_predict_ckpt(ckpt_path, texts, feats, bs=48, max_len=160, add_e5_prefix=False):
    free_cuda()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mname = ckpt["model_name"]
    tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = HeadConditionedTransformer(
        mname, n_feats=feats.shape[1], pool_mode=ckpt["pool_mode"], proj_dim=256, num_freeze_layers=ckpt["freeze_k"]
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda().eval()
    inp = ["passage: " + t for t in texts] if add_e5_prefix else texts
    pl, pt, pa = predict_transformer(model, tok, inp, feats, bs=bs, max_len=max_len, device="cuda")
    del model, ckpt
    free_cuda()
    return pl, pt, pa


def select_domain_adaptation_prototypes(te_rules, te_base, te_clusters, te_feats, total_n=52000, seed=42):
    """Select diverse high-value training prototypes across cluster hubs, uncertainty frontier, and long-tail."""
    rng = np.random.RandomState(seed)
    n_te = len(te_feats)
    p_base_l = 0.32 * te_base["lin_l"] + 0.26 * te_base["lgb_l"] + 0.22 * te_base["xgb_l"] + 0.20 * te_base["cb_l"]
    p_base_a = 0.35 * te_base["lin_a"] + 0.65 * te_base["gb_a"]
    pred_l = p_base_l.argmax(axis=1)
    pred_a = np.where(np.isin(pred_l, [0, 5]), 0, 1 + p_base_a[:, 1:].argmax(axis=1))
    r_l = te_rules["lbl"].astype(int)
    r_a = te_rules["act"].astype(int)

    selected = []
    used = np.zeros(n_te, dtype=bool)

    def _add(idxs):
        for idx in idxs:
            if not used[idx]:
                used[idx] = True
                selected.append(int(idx))

    # 1. All cluster representatives for clusters >= 2 up to 22,000 (covering bulk of structured families)
    uniq_first = te_clusters["first_idx"]
    counts = te_clusters["counts"]
    ge2 = uniq_first[counts >= 2][:22000]
    _add(ge2)

    # 2. Disagreement & low-confidence frontier between rules_v4 and TFIDF-GBDT + rare action=5/1/4
    disagree_l = (r_l != pred_l)
    disagree_a = (r_a != pred_a) & ((r_l >= 1) & (r_l <= 4) | (pred_l >= 1) & (pred_l <= 4))
    low_conf = p_base_l.max(axis=1) < 0.84
    rare_act = (r_a == 5) | (pred_a == 5) | ((pred_a == 1) & (pred_l != 1)) | ((r_a == 1) & (pred_l != 1))
    frontier_pool = np.where((disagree_l | disagree_a | low_conf | rare_act) & (~used))[0]
    if len(frontier_pool) > 19000:
        # Prioritize severe disagreement + rare action first
        score_f = (
            3.0 * disagree_l[frontier_pool].astype(np.float32)
            + 2.0 * rare_act[frontier_pool].astype(np.float32)
            + (1.0 - p_base_l.max(axis=1)[frontier_pool])
        )
        frontier_pool = frontier_pool[np.argsort(-score_f)[:19000]]
    _add(frontier_pool)

    # 3. Balanced stratified fill from remaining singletons across predicted classes 0..5 & multi-turn >=3
    rem = max(0, total_n - len(selected))
    per_cls = max(1, rem // 6)
    for c in range(6):
        pool_c = np.where((pred_l == c) & (~used))[0]
        if len(pool_c) > 0:
            take = min(len(pool_c), per_cls)
            _add(rng.choice(pool_c, size=take, replace=False))
    if len(selected) < total_n:
        pool_rem = np.where(~used)[0]
        if len(pool_rem) > 0:
            _add(rng.choice(pool_rem, size=min(len(pool_rem), total_n - len(selected)), replace=False))

    idx_arr = np.sort(np.array(selected[:total_n], dtype=np.int32))
    print(f"Selected {len(idx_arr)} domain-adaptation prototypes (clusters>=2={len(ge2)}, frontier={len(frontier_pool)})", flush=True)
    return idx_arr


def run_qwen_tiebreaker(texts, cand_local_idx):
    """Run Qwen2.5-1.5B-Instruct on the hardest teacher-disagreement prototypes."""
    if len(cand_local_idx) == 0:
        return {}, {}
    free_cuda()
    mname = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True, dtype=torch.float16).cuda().eval()

    sys_msg = (
        "Classify Bangladeshi SMS/dialogue into STRICT Priority Order:\n"
        "L1=MFS/OTP/PIN/NID/CVV/wrong-transfer ferot/card theft (beats prize/parcel/job if asks PIN/OTP/NID/CVV)\n"
        "L2=Fake Prize/Cashback/Lottery/Airdrop/Survey/Gift win\n"
        "L4=Delivery/Parcel/Customs fee or address link/call scam\n"
        "L3=Fake Job/Task/Like/Crypto/Investment/Deposit/Visa fee\n"
        "L5=Legitimate Promotional Discount/Sale/BOGO/App-Update/Dream-Flat Ad (NOT scam)\n"
        "L0=Legitimate Safe (OTP warning 'never share', TX alert, COD rider, HR offer letter, CV interview, scam warning 'someone messaged...scam', noise)\n"
        "Action C: 0 if L0 or L5; else 1=share PIN/OTP/NID/CVV/PII, 2=pay/send money/fee/ferot, 3=click link/survey, 4=install APK/join TG/WA group/tasks, 5=call/WhatsApp number.\n"
        "Output ONLY compact JSON: {\"L\": int, \"C\": int}"
    )

    q_lbl, q_act = {}, {}
    t0 = time.time()
    for step, li in enumerate(cand_local_idx):
        txt = str(texts[li])[:360]
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Text: {txt}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        ml = re.search(r'"?L"?\s*[:=]\s*([0-5])', ans)
        mc = re.search(r'"?C"?\s*[:=]\s*([0-5])', ans)
        if ml:
            lv = int(ml.group(1))
            cv = int(mc.group(1)) if mc else (0 if lv in (0, 5) else 2)
            if lv in (0, 5):
                cv = 0
            elif cv == 0:
                cv = {1: 1, 2: 3, 3: 4, 4: 3}[lv]
            q_lbl[int(li)] = lv
            q_act[int(li)] = cv
        if (step + 1) % 250 == 0:
            print(f"  Qwen judge: {step+1}/{len(cand_local_idx)} parsed={len(q_lbl)} elapsed={time.time()-t0:.1f}s", flush=True)

    del model
    free_cuda()
    return q_lbl, q_act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-fold", type=int, default=18000)
    ap.add_argument("--proto-n", type=int, default=48000)
    ap.add_argument("--qwen-max", type=int, default=0)
    ap.add_argument("--epochs-mdeb", type=int, default=2)
    ap.add_argument("--epochs-e5b", type=int, default=2)
    args = ap.parse_args()

    sdf = pd.read_csv(CACHE_DIR / "sample_v4.csv")
    sf = scale_feats_v4(np.load(CACHE_DIR / "sample_v4_feats.npy"))
    sy = pd.read_csv(CACHE_DIR / "synth_64k_v4.csv")
    syf = scale_feats_v4(np.load(CACHE_DIR / "synth_64k_v4_feats.npy"))

    s_txt = [format_turn_aware(str(x)) for x in sdf["clean_text"].fillna("")]
    sy_txt = [format_turn_aware(str(x)) for x in sy["clean_text"].fillna("")]
    s_yl = sdf["label"].values.astype(int)
    s_yt = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]])
    s_ya = sdf["action"].values.astype(int)
    s_folds = get_scenario_disjoint_folds(sdf["scenario_family"].values)

    sy_yl = sy["label"].values.astype(int)
    sy_yt = np.vstack([tech_str_to_vec(x) for x in sy["techniques"]])
    sy_ya = sy["action"].values.astype(int)
    sy_folds = get_scenario_disjoint_folds(sy["scenario_family"].values)

    print(f"Scenario-Disjoint folds: sample F0={(s_folds==0).sum()} F1={(s_folds==1).sum()} | synth F0={(sy_folds==0).sum()} F1={(sy_folds==1).sum()}", flush=True)

    oof_mdeb_s_l = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_mdeb_s_t = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_mdeb_s_a = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_e5b_s_l = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_e5b_s_t = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_e5b_s_a = np.zeros((len(sdf), 6), dtype=np.float32)

    oof_mdeb_sy_l = np.zeros((len(sy), 6), dtype=np.float32)
    oof_mdeb_sy_t = np.zeros((len(sy), 6), dtype=np.float32)
    oof_mdeb_sy_a = np.zeros((len(sy), 6), dtype=np.float32)
    oof_e5b_sy_l = np.zeros((len(sy), 6), dtype=np.float32)
    oof_e5b_sy_t = np.zeros((len(sy), 6), dtype=np.float32)
    oof_e5b_sy_a = np.zeros((len(sy), 6), dtype=np.float32)

    mdeb_ckpts, e5b_ckpts = [], []
    rng = np.random.RandomState(42)

    for fold in (0, 1):
        tr_sy_idx = np.where(sy_folds != fold)[0]
        va_sy_idx = np.where(sy_folds == fold)[0]
        tr_s_idx = np.where(s_folds != fold)[0]
        va_s_idx = np.where(s_folds == fold)[0]

        if len(tr_sy_idx) > args.max_per_fold:
            # Stratified cap preserving family diversity
            tr_sy_idx = np.sort(rng.choice(tr_sy_idx, size=args.max_per_fold, replace=False))

        tr_texts = [sy_txt[i] for i in tr_sy_idx] + [s_txt[i] for i in tr_s_idx] * 2
        tr_feats = np.vstack([syf[tr_sy_idx], sf[tr_s_idx], sf[tr_s_idx]])
        tr_yl = np.concatenate([sy_yl[tr_sy_idx], s_yl[tr_s_idx], s_yl[tr_s_idx]])
        tr_yt = np.vstack([sy_yt[tr_sy_idx], s_yt[tr_s_idx], s_yt[tr_s_idx]])
        tr_ya = np.concatenate([sy_ya[tr_sy_idx], s_ya[tr_s_idx], s_ya[tr_s_idx]])

        va_texts = [s_txt[i] for i in va_s_idx] + [sy_txt[i] for i in va_sy_idx]
        va_feats = np.vstack([sf[va_s_idx], syf[va_sy_idx]])
        va_yl = np.concatenate([s_yl[va_s_idx], sy_yl[va_sy_idx]])

        (pl_m, pt_m, pa_m), c_m = train_one_teacher_fold(
            "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", f"mdeberta_f{fold}",
            tr_texts, tr_feats, tr_yl, tr_yt, tr_ya, va_texts, va_feats, va_yl, n_sample_va=len(va_s_idx),
            pool_mode="mean", freeze_k=5, bs=8, grad_accum=4, max_len=152,
            epochs=args.epochs_mdeb, lr_bb=2.5e-5, lr_hd=1.8e-4, seed=100 + fold, add_e5_prefix=False,
        )
        ns = len(va_s_idx)
        oof_mdeb_s_l[va_s_idx], oof_mdeb_s_t[va_s_idx], oof_mdeb_s_a[va_s_idx] = pl_m[:ns], pt_m[:ns], pa_m[:ns]
        oof_mdeb_sy_l[va_sy_idx], oof_mdeb_sy_t[va_sy_idx], oof_mdeb_sy_a[va_sy_idx] = pl_m[ns:], pt_m[ns:], pa_m[ns:]
        mdeb_ckpts.append(c_m)

        (pl_e, pt_e, pa_e), c_e = train_one_teacher_fold(
            "intfloat/multilingual-e5-base", f"me5base_f{fold}",
            tr_texts, tr_feats, tr_yl, tr_yt, tr_ya, va_texts, va_feats, va_yl, n_sample_va=len(va_s_idx),
            pool_mode="cls_mean", freeze_k=4, bs=16, grad_accum=2, max_len=152,
            epochs=args.epochs_e5b, lr_bb=3.0e-5, lr_hd=2.0e-4, seed=200 + fold, add_e5_prefix=True,
        )
        oof_e5b_s_l[va_s_idx], oof_e5b_s_t[va_s_idx], oof_e5b_s_a[va_s_idx] = pl_e[:ns], pt_e[:ns], pa_e[:ns]
        oof_e5b_sy_l[va_sy_idx], oof_e5b_sy_t[va_sy_idx], oof_e5b_sy_a[va_sy_idx] = pl_e[ns:], pt_e[ns:], pa_e[ns:]
        e5b_ckpts.append(c_e)

    # Report true Zero-Leakage Scenario-Disjoint Teacher OOF composite scores
    p_mix_s_l = 0.60 * oof_mdeb_s_l + 0.40 * oof_e5b_s_l
    p_mix_s_t = 0.60 * oof_mdeb_s_t + 0.40 * oof_e5b_s_t
    p_mix_s_a = 0.60 * oof_mdeb_s_a + 0.40 * oof_e5b_s_a
    spl, spt, spa = decode_consistent(p_mix_s_l, p_mix_s_t, p_mix_s_a, tech_th=0.45)

    p_mix_sy_l = 0.60 * oof_mdeb_sy_l + 0.40 * oof_e5b_sy_l
    p_mix_sy_t = 0.60 * oof_mdeb_sy_t + 0.40 * oof_e5b_sy_t
    p_mix_sy_a = 0.60 * oof_mdeb_sy_a + 0.40 * oof_e5b_sy_a
    sypl, sypt, sypa = decode_consistent(p_mix_sy_l, p_mix_sy_t, p_mix_sy_a, tech_th=0.45)
    raw_syf = np.load(CACHE_DIR / "synth_64k_v4_feats.npy")
    multi_mask = raw_syf[:, 2] >= 3
    scores = score_section10_proxy(
        s_yl, spl, s_yt, spt, s_ya, spa, p_mix_s_l.max(axis=1),
        y_l_ood=sy_yl, p_l_ood=sypl, conf_ood=p_mix_sy_l.max(axis=1),
        y_l_adv=sy_yl, p_l_adv=sypl, conf_adv=p_mix_sy_l.max(axis=1),
        y_l_multi=sy_yl[multi_mask], p_l_multi=sypl[multi_mask],
    )
    print("\n=== ZERO-LEAKAGE SCENARIO-DISJOINT TEACHER ENSEMBLE OOF ===", scores, flush=True)

    np.savez_compressed(
        OUT_DIR / "oof_teachers_v4.npz",
        s_mdeb_l=oof_mdeb_s_l, s_mdeb_t=oof_mdeb_s_t, s_mdeb_a=oof_mdeb_s_a,
        s_e5b_l=oof_e5b_s_l, s_e5b_t=oof_e5b_s_t, s_e5b_a=oof_e5b_s_a,
        sy_mdeb_l=oof_mdeb_sy_l, sy_mdeb_t=oof_mdeb_sy_t, sy_mdeb_a=oof_mdeb_sy_a,
        sy_e5b_l=oof_e5b_sy_l, sy_e5b_t=oof_e5b_sy_t, sy_e5b_a=oof_e5b_sy_a,
    )

    # ---- STAGE 2: DOMAIN-ADAPTATION PROTOTYPE PSEUDO-LABELING (Rulebook Section 5) ----
    te_rules = np.load(CACHE_DIR / "test_v4_rule_pred.npz")
    te_base = np.load(OUT_DIR / "test_baseline_v4.npz")
    te_clusters = np.load(CACHE_DIR / "test_skel_clusters.npz")
    te_feats_mm = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")

    proto_idx = select_domain_adaptation_prototypes(te_rules, te_base, te_clusters, te_feats_mm, total_n=args.proto_n)
    proto_mask = np.zeros(len(te_feats_mm), dtype=bool)
    proto_mask[proto_idx] = True

    # Copy baseline subset on prototypes then free large 500k baseline arrays to guard 8.24 GB RAM
    pb_l = 0.32 * te_base["lin_l"][proto_idx] + 0.26 * te_base["lgb_l"][proto_idx] + 0.22 * te_base["xgb_l"][proto_idx] + 0.20 * te_base["cb_l"][proto_idx]
    pb_t = 0.35 * te_base["lin_t"][proto_idx] + 0.65 * te_base["gb_t"][proto_idx]
    pb_a = 0.35 * te_base["lin_a"][proto_idx] + 0.65 * te_base["gb_a"][proto_idx]
    r_arg = te_rules["lbl"][proto_idx].astype(int)
    del te_base, te_rules, te_clusters, oof_mdeb_sy_l, oof_mdeb_sy_t, oof_mdeb_sy_a, oof_e5b_sy_l, oof_e5b_sy_t, oof_e5b_sy_a, raw_syf, sy, syf, sy_txt
    gc.collect()

    clean_chunks, raw_chunks = [], []
    off = 0
    for df_c, df_r in zip(
        pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=50000),
        pd.read_csv(V2_DIR.parents[1] / "Main Preli Round/dataset/test.csv", usecols=["text"], chunksize=50000),
    ):
        m = len(df_c)
        loc = proto_mask[off:off + m]
        if loc.any():
            clean_chunks.extend(df_c["clean_text"].fillna("").values[loc].tolist())
            raw_chunks.extend(df_r["text"].fillna("").values[loc].tolist())
        off += m
    proto_turn_texts = [format_turn_aware(t) for t in clean_chunks]
    proto_feats = scale_feats_v4(te_feats_mm[proto_idx])

    print("Scoring domain-adaptation prototypes with mDeBERTa (2 folds) & mE5-base (2 folds)...", flush=True)
    pm_cache = OUT_DIR / "proto_mdeb_tmp.npz"
    if pm_cache.exists():
        zm = np.load(pm_cache)
        pm_l, pm_t, pm_a = zm["pm_l"], zm["pm_t"], zm["pm_a"]
    else:
        m0_l, m0_t, m0_a = load_predict_ckpt(mdeb_ckpts[0], proto_turn_texts, proto_feats, bs=48, max_len=160, add_e5_prefix=False)
        m1_l, m1_t, m1_a = load_predict_ckpt(mdeb_ckpts[1], proto_turn_texts, proto_feats, bs=48, max_len=160, add_e5_prefix=False)
        pm_l, pm_t, pm_a = 0.5 * (m0_l + m1_l), 0.5 * (m0_t + m1_t), 0.5 * (m0_a + m1_a)
        np.savez_compressed(pm_cache, pm_l=pm_l, pm_t=pm_t, pm_a=pm_a)
        del m0_l, m0_t, m0_a, m1_l, m1_t, m1_a
        gc.collect()

    pe_cache = OUT_DIR / "proto_e5b_tmp.npz"
    if pe_cache.exists():
        ze = np.load(pe_cache)
        pe_l, pe_t, pe_a = ze["pe_l"], ze["pe_t"], ze["pe_a"]
    else:
        e0_l, e0_t, e0_a = load_predict_ckpt(e5b_ckpts[0], proto_turn_texts, proto_feats, bs=64, max_len=160, add_e5_prefix=True)
        e1_l, e1_t, e1_a = load_predict_ckpt(e5b_ckpts[1], proto_turn_texts, proto_feats, bs=64, max_len=160, add_e5_prefix=True)
        pe_l, pe_t, pe_a = 0.5 * (e0_l + e1_l), 0.5 * (e0_t + e1_t), 0.5 * (e0_a + e1_a)
        np.savez_compressed(pe_cache, pe_l=pe_l, pe_t=pe_t, pe_a=pe_a)
        del e0_l, e0_t, e0_a, e1_l, e1_t, e1_a
        gc.collect()

    p_cons_l = 0.50 * pm_l + 0.34 * pe_l + 0.16 * pb_l
    p_cons_t = 0.50 * pm_t + 0.34 * pe_t + 0.16 * pb_t
    p_cons_a = 0.50 * pm_a + 0.34 * pe_a + 0.16 * pb_a

    if args.qwen_max > 0:
        m_arg = pm_l.argmax(axis=1)
        e_arg = pe_l.argmax(axis=1)
        disagree = (m_arg != e_arg) | (m_arg != r_arg) | (p_cons_l.max(axis=1) < 0.72)
        cand_q = np.where(disagree)[0]
        if len(cand_q) > args.qwen_max:
            cand_q = cand_q[np.argsort(p_cons_l.max(axis=1)[cand_q])[:args.qwen_max]]
        q_lbl_map, q_act_map = run_qwen_tiebreaker(raw_chunks, cand_q)
        for li, lv in q_lbl_map.items():
            q_vec_l = np.full(6, 0.01, dtype=np.float32)
            q_vec_l[lv] = 0.95
            p_cons_l[li] = 0.80 * p_cons_l[li] + 0.20 * q_vec_l
            p_cons_l[li] /= p_cons_l[li].sum()
            if lv in (1, 2, 3, 4):
                av = q_act_map[li]
                q_vec_a = np.full(6, 0.02, dtype=np.float32)
                q_vec_a[0] = 0.0
                q_vec_a[av] = 0.92
                p_cons_a[li, 1:] = 0.78 * p_cons_a[li, 1:] + 0.22 * q_vec_a[1:]
                p_cons_a[li, 1:] /= p_cons_a[li, 1:].sum()

    # Guard Priority 0/5 & non-empty Task B rules cleanly
    pseudo_l, pseudo_t_bin, pseudo_a = decode_consistent(p_cons_l, p_cons_t, p_cons_a, tech_th=0.45)
    pseudo_l = pseudo_l.astype(np.int64)
    pseudo_t_bin = pseudo_t_bin.astype(np.float32)
    pseudo_a = pseudo_a.astype(np.int64)
    conf_cons = p_cons_l.max(axis=1).astype(np.float32)

    np.savez_compressed(
        CACHE_DIR / "domain_pseudo_v4.npz",
        proto_idx=proto_idx,
        clean_texts=np.array(clean_chunks, dtype=object),
        turn_texts=np.array(proto_turn_texts, dtype=object),
        p_l=p_cons_l, p_t=p_cons_t, p_a=p_cons_a,
        pm_l=pm_l, pm_t=pm_t, pm_a=pm_a,
        pe_l=pe_l, pe_t=pe_t, pe_a=pe_a,
        y_l=pseudo_l, y_t=pseudo_t_bin, y_a=pseudo_a,
        conf=conf_cons,
    )
    print(f"Saved domain_pseudo_v4.npz ({len(proto_idx)} rows, mean_conf={conf_cons.mean():.4f}, dist={pd.Series(pseudo_l).value_counts().sort_index().to_dict()})", flush=True)


if __name__ == "__main__":
    main()
