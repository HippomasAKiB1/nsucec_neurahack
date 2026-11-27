"""Purely Inductive Section 7.3 End-to-End Inference Harness for ScamShield BD v4."""
import argparse
import gc
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack, csr_matrix

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
CKPT_DIR = CACHE_DIR / "ckpt_v4"
sys.path.insert(0, str(V2_DIR))

from cv_ood import vec_to_tech_str, decode_consistent
from normalizer_v4 import clean_for_model, format_turn_aware, bn_to_en_roman_v4
from rules_v4 import extract_rule_prediction
from torch_utils_v4 import scale_feats_v4, HeadConditionedTransformer, predict_transformer, free_cuda, CACHE_DIR_HF
from train_da_bigru_v4 import BPEBiGRUNet, predict_bigru
from train_meta_calibrate_v4 import onehot_from_int, build_uncertainty_features
from transformers import AutoTokenizer


def run_inductive_inference(input_csv: Path, output_csv: Path, esc_frac: float = 0.055):
    t0 = time.time()
    df = pd.read_csv(input_csv)
    raw_texts = df["text"].fillna("").astype(str).tolist()
    n = len(raw_texts)

    cleans = []
    feats = np.zeros((n, 58), dtype=np.float32)
    r_lbls = np.zeros(n, dtype=np.int64)
    r_tvecs = np.zeros((n, 6), dtype=np.float32)
    r_acts = np.zeros(n, dtype=np.int64)
    for i, raw in enumerate(raw_texts):
        cleans.append(clean_for_model(raw))
        l, tv, a, f = extract_rule_prediction(raw)
        r_lbls[i], r_tvecs[i], r_acts[i], feats[i] = l, tv, a, f

    turns = [format_turn_aware(c) for c in cleans]
    e5_texts = ["passage: " + t for t in turns]
    bge_texts = ["Represent this sentence for searching relevant passages: " + bn_to_en_roman_v4(t) for t in turns]
    sc_feats = scale_feats_v4(feats)

    # 1. Domain-Adapted Tabular Linear + LGBM/XGB/CatBoost
    tab = joblib.load(CKPT_DIR / "da_tab_v4.joblib")
    lin_l = np.zeros((n, 6), np.float32)
    lin_t = np.zeros((n, 6), np.float32)
    lin_a = np.zeros((n, 6), np.float32)
    gb_l = np.zeros((n, 6), np.float32)
    gb_t = np.zeros((n, 6), np.float32)
    gb_a = np.zeros((n, 6), np.float32)

    for s in range(0, n, 50000):
        e = min(n, s + 50000)
        X_tw = hstack([tab["vec_w"].transform(turns[s:e]), tab["vec_c"].transform(turns[s:e])]).tocsr()
        F_c = tab["scaler"].transform(feats[s:e]).astype(np.float32)
        Z_c = tab["svd"].transform(X_tw).astype(np.float32)
        X_lc = hstack([X_tw, csr_matrix(F_c * 0.25)]).tocsr()
        X_tc = np.hstack([F_c, Z_c]).astype(np.float32)

        fl, fts, fa = tab["lin"]
        lin_l[s:e] = fl.predict_proba(X_lc).astype(np.float32)
        lin_t[s:e] = np.column_stack([c.predict_proba(X_lc)[:, 1] for c in fts]).astype(np.float32)
        pa_sub = fa.predict_proba(X_lc).astype(np.float32)
        for ci, cls_id in enumerate(fa.classes_):
            lin_a[s:e, int(cls_id)] = pa_sub[:, ci]

        gm = tab["gb"]
        pl1 = gm["lgb_l"].predict_proba(X_tc).astype(np.float32)
        px1 = gm["xgb_l"].predict_proba(X_tc).astype(np.float32)
        if gm.get("cb_l") is not None:
            pcb1 = gm["cb_l"].predict_proba(X_tc).astype(np.float32)
            gb_l[s:e] = 0.40 * pl1 + 0.35 * px1 + 0.25 * pcb1
        else:
            gb_l[s:e] = 0.55 * pl1 + 0.45 * px1
        gb_t[s:e] = np.column_stack([c.predict_proba(X_tc)[:, 1] for c in gm["lgbs_t"]]).astype(np.float32)
        pa_g = gm["lgb_a"].predict_proba(X_tc).astype(np.float32)
        for ci, cls_id in enumerate(gm["lgb_a"].classes_):
            gb_a[s:e, int(cls_id) + 1] = pa_g[:, ci]
    del tab
    gc.collect()

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Transformer & BiGRU Students (streaming one model at a time for strict <2GB VRAM / <4GB RAM)
    def run_ckpt(ckpt_name, in_texts, bs=96, max_len=152):
        free_cuda()
        ck = torch.load(CKPT_DIR / ckpt_name, map_location="cpu", weights_only=False)
        mname = ck["model_name"]
        tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR_HF, local_files_only=True)
        mdl = HeadConditionedTransformer(mname, n_feats=58, pool_mode=ck["pool_mode"], proj_dim=256)
        mdl.load_state_dict(ck["state_dict"], strict=True)
        mdl = mdl.to(DEV).eval()
        pl, pt, pa = predict_transformer(mdl, tok, in_texts, sc_feats, bs=bs, max_len=max_len, device=DEV)
        del mdl, ck
        free_cuda()
        return pl, pt, pa

    m_l, m_t, m_a = run_ckpt("mminilm_distill.pt", turns, bs=96)
    e_l, e_t, e_a = run_ckpt("me5small_distill.pt", e5_texts, bs=96)
    b_l, b_t, b_a = run_ckpt("bge_distill.pt", bge_texts, bs=96)

    # BiGRU-v4
    free_cuda()
    tok_bg = AutoTokenizer.from_pretrained("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", cache_dir=CACHE_DIR_HF, local_files_only=True)
    bg = BPEBiGRUNet(n_feats=58).to(DEV).eval()
    bg.load_state_dict(torch.load(CKPT_DIR / "bigru_v4_distill.pt", map_location="cpu", weights_only=False)["state_dict"], strict=True)
    bg_l, bg_t, bg_a = predict_bigru(bg, tok_bg, turns, sc_feats, bs=256)
    del bg
    free_cuda()

    # 3. Rule priors & Meta Stacking
    r_pl = onehot_from_int(r_lbls, conf=0.84)
    r_pt = r_tvecs * 0.84 + 0.08
    r_pa = np.zeros((n, 6), np.float32)
    pos_a = r_acts > 0
    r_pa[pos_a, :] = 0.0375
    r_pa[np.where(pos_a)[0], r_acts[pos_a]] = 0.85

    meta = joblib.load(CKPT_DIR / "meta_calibrator_v4.joblib")
    L_stu = [m_l, e_l, b_l, bg_l, lin_l, gb_l, r_pl]
    T_stu = [m_t, e_t, b_t, bg_t, lin_t, gb_t, r_pt]
    A_stu = [m_a, e_a, b_a, bg_a, lin_a, gb_a, r_pa]

    P_L = sum(w * x for w, x in zip(meta["w_l_stu"], L_stu))
    P_T = sum(w * x for w, x in zip(meta.get("w_t_stu", meta["w_l_stu"]), T_stu))
    P_A = sum(w * x for w, x in zip(meta.get("w_a_stu", meta["w_l_stu"]), A_stu))

    # 4. Selective Inductive mDeBERTa-v3-base Escalation on top uncertain frontier
    k_esc = max(1, int(round(n * esc_frac)))
    ent = -np.sum(P_L * np.log(np.clip(P_L, 1e-6, 1.0)), axis=1)
    dis_l = (m_l.argmax(1) != e_l.argmax(1)).astype(np.float32) + (m_l.argmax(1) != b_l.argmax(1)).astype(np.float32) + (m_l.argmax(1) != gb_l.argmax(1)).astype(np.float32)
    score_esc = ent + 0.65 * dis_l + (1.0 - P_L.max(1))
    esc_idx = np.sort(np.argsort(-score_esc)[:k_esc])

    esc_tx = [turns[i] for i in esc_idx]
    esc_f = sc_feats[esc_idx]
    full_ck = CKPT_DIR / "mdeberta_full_distill.pt"
    if full_ck.exists():
        free_cuda()
        ck = torch.load(full_ck, map_location="cpu", weights_only=False)
        tok_d = AutoTokenizer.from_pretrained(ck["model_name"], cache_dir=CACHE_DIR_HF, local_files_only=True)
        mdl_d = HeadConditionedTransformer(ck["model_name"], n_feats=58, pool_mode=ck["pool_mode"], proj_dim=256, num_freeze_layers=5)
        mdl_d.load_state_dict(ck["state_dict"], strict=True)
        mdl_d = mdl_d.to(DEV).eval()
        esc_l, esc_t, esc_a = predict_transformer(mdl_d, tok_d, esc_tx, esc_f, bs=48, max_len=152, device=DEV)
        del mdl_d, ck
        free_cuda()
    else:
        for fold_tag, wt in [("mdeberta_f0.pt", 0.5), ("mdeberta_f1.pt", 0.5)]:
            free_cuda()
            ck = torch.load(CKPT_DIR / fold_tag, map_location="cpu", weights_only=False)
            tok_d = AutoTokenizer.from_pretrained(ck["model_name"], cache_dir=CACHE_DIR_HF, local_files_only=True)
            mdl_d = HeadConditionedTransformer(ck["model_name"], n_feats=58, pool_mode=ck["pool_mode"], proj_dim=256, num_freeze_layers=ck["freeze_k"])
            mdl_d.load_state_dict(ck["state_dict"], strict=True)
            mdl_d = mdl_d.to(DEV).eval()
            dl, dt, da = predict_transformer(mdl_d, tok_d, esc_tx, esc_f, bs=48, max_len=152, device=DEV)
            if fold_tag.endswith("f0.pt"):
                esc_l, esc_t, esc_a = wt * dl, wt * dt, wt * da
            else:
                esc_l += wt * dl; esc_t += wt * dt; esc_a += wt * da
            del mdl_d, ck
            free_cuda()

    w_esc = meta["w_l_esc"][-1]
    P_L[esc_idx] = (1.0 - w_esc) * P_L[esc_idx] + w_esc * esc_l
    P_T[esc_idx] = (1.0 - w_esc) * P_T[esc_idx] + w_esc * esc_t
    P_A[esc_idx] = (1.0 - w_esc) * P_A[esc_idx] + w_esc * esc_a

    X_cal = build_uncertainty_features(L_stu, feats, r_lbls)
    conf = np.clip(meta["cal_lr"].predict_proba(X_cal)[:, 1], 0.05, 0.996).astype(np.float32)

    final_l, final_t, final_a = decode_consistent(P_L * meta["best_alpha"], P_T, P_A * meta["best_beta"], tech_th=meta["tech_th"])
    out = pd.DataFrame({
        "id": df["id"].values,
        "label": final_l,
        "techniques": [vec_to_tech_str(final_t[i]) for i in range(n)],
        "action": final_a,
        "confidence": np.round(conf, 4),
    })
    out.to_csv(output_csv, index=False)
    print(f"Inductive prediction complete on {n} rows -> {output_csv} in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--esc-frac", type=float, default=0.055)
    args = ap.parse_args()
    run_inductive_inference(Path(args.input), Path(args.output), esc_frac=args.esc_frac)
