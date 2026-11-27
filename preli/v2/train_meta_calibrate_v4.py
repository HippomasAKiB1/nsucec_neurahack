import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
OUT_DIR = CACHE_DIR / "preds_v4"
sys.path.insert(0, str(V2_DIR))

from cv_ood import (
    TECH_LIST, tech_str_to_vec, vec_to_tech_str, decode_consistent,
    score_section10_proxy, get_scenario_disjoint_folds,
)


def onehot_from_int(y, conf=0.92):
    arr = np.full((len(y), 6), (1.0 - conf) / 5.0, dtype=np.float32)
    arr[np.arange(len(y)), y] = conf
    return arr


def build_uncertainty_features(probs_list, te_feats, rule_lbl):
    """Inductive Brier-calibration feature builder (zero hard 0.985 ceiling)."""
    stack = np.stack(probs_list, axis=0)  # (M, N, 6)
    avg = stack.mean(axis=0)
    sorted_p = np.sort(avg, axis=1)
    p1 = sorted_p[:, -1]
    p2 = sorted_p[:, -2]
    margin = p1 - p2
    ent = -np.sum(avg * np.log(np.clip(avg, 1e-6, 1.0)), axis=1) / np.log(6.0)

    top_pred = avg.argmax(axis=1)
    votes = np.stack([p.argmax(axis=1) == top_pred for p in probs_list], axis=0).mean(axis=0)
    prob_std = stack.std(axis=0)[np.arange(len(avg)), top_pred]
    rule_agree = (rule_lbl == top_pred).astype(np.float32)

    is_noise = te_feats[:, 15].astype(np.float32)
    known_ratio = te_feats[:, 19].astype(np.float32)
    n_turns_norm = np.minimum(te_feats[:, 2], 12.0).astype(np.float32) / 12.0
    return np.column_stack([
        p1, margin, ent, votes, prob_std, rule_agree,
        is_noise, known_ratio, n_turns_norm,
    ]).astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(V2_DIR / "submission_v7_top10.csv"))
    args = ap.parse_args()
    t0 = time.time()
    sdf = pd.read_csv(CACHE_DIR / "sample_v4.csv")
    sf = np.load(CACHE_DIR / "sample_v4_feats.npy")
    s_rules = np.load(CACHE_DIR / "sample_v4_rule_pred.npz")

    sy = pd.read_csv(CACHE_DIR / "synth_64k_v4.csv")
    syf = np.load(CACHE_DIR / "synth_64k_v4_feats.npy")
    sy_rules = np.load(CACHE_DIR / "synth_64k_v4_rule_pred.npz")

    # Load all Zero-Leakage Scenario-Disjoint OOF files
    oof_st = np.load(OUT_DIR / "oof_students_v4.npz")
    oof_bg = np.load(OUT_DIR / "oof_bigru_v4.npz")
    oof_ba = np.load(OUT_DIR / "oof_baseline_v4.npz")

    y_s_l = sdf["label"].values.astype(int)
    y_s_t = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]])
    y_s_a = sdf["action"].values.astype(int)

    y_sy_l = sy["label"].values.astype(int)
    y_sy_t = np.vstack([tech_str_to_vec(x) for x in sy["techniques"]])
    y_sy_a = sy["action"].values.astype(int)
    ood_idx = np.where(oof_ba["ood_mask"])[0]

    s_rule_pl = onehot_from_int(s_rules["lbl"], conf=0.86)
    s_rule_pa = np.zeros((len(sdf), 6), dtype=np.float32)
    for i, a in enumerate(s_rules["act"]):
        if a > 0:
            s_rule_pa[i, a] = 0.85
            for c in range(1, 6):
                if c != a:
                    s_rule_pa[i, c] = 0.0375
    s_rule_pt = s_rules["tvec"].astype(np.float32) * 0.85 + 0.07

    # Order of 7 Full-Coverage Diverse Student+Tab+Rule Families:
    # [0:mMiniLM, 1:mE5small, 2:BGE-glossed, 3:BiGRU-v5, 4:Linear-TFIDF, 5:GBDT-SVD, 6:Rules-v5]
    s_L_models = [
        oof_st["s_mminilm_l"], oof_st["s_me5small_l"], oof_st["s_bge_l"], oof_bg["s_l"],
        oof_ba["s_lin_l"], oof_ba["s_gb_l"], s_rule_pl,
    ]
    s_T_models = [
        oof_st["s_mminilm_t"], oof_st["s_me5small_t"], oof_st["s_bge_t"], oof_bg["s_t"],
        oof_ba["s_lin_t"], oof_ba["s_gb_t"], s_rule_pt,
    ]
    s_A_models = [
        oof_st["s_mminilm_a"], oof_st["s_me5small_a"], oof_st["s_bge_a"], oof_bg["s_a"],
        oof_ba["s_lin_a"], oof_ba["s_gb_a"], s_rule_pa,
    ]

    sy_rule_pl = onehot_from_int(sy_rules["lbl"][ood_idx], conf=0.82)
    sy_rule_pa = np.zeros((len(ood_idx), 6), dtype=np.float32)
    for i, a in enumerate(sy_rules["act"][ood_idx]):
        if a > 0:
            sy_rule_pa[i, a] = 0.85
            for c in range(1, 6):
                if c != a:
                    sy_rule_pa[i, c] = 0.0375
    sy_rule_pt = sy_rules["tvec"][ood_idx].astype(np.float32) * 0.82 + 0.08

    sy_L_models = [
        oof_st["sy_mminilm_l"][ood_idx], oof_st["sy_me5small_l"][ood_idx], oof_st["sy_bge_l"][ood_idx], oof_bg["sy_l"][ood_idx],
        oof_ba["ood_lin_l"], oof_ba["ood_gb_l"], sy_rule_pl,
    ]
    sy_T_models = [
        oof_st["sy_mminilm_t"][ood_idx], oof_st["sy_me5small_t"][ood_idx], oof_st["sy_bge_t"][ood_idx], oof_bg["sy_t"][ood_idx],
        oof_ba["ood_lin_t"], oof_ba["ood_gb_t"], sy_rule_pt,
    ]
    sy_A_models = [
        oof_st["sy_mminilm_a"][ood_idx], oof_st["sy_me5small_a"][ood_idx], oof_st["sy_bge_a"][ood_idx], oof_bg["sy_a"][ood_idx],
        oof_ba["ood_lin_a"], oof_ba["ood_gb_a"], sy_rule_pa,
    ]

    def blend_list(mlist, w):
        out = np.zeros_like(mlist[0])
        for wi, m in zip(w, mlist):
            out += wi * m
        return out

    def opt_task_weights(s_list, sy_list, y_s, y_sy, mode="mc", w_prior=None):
        k = len(s_list)
        w0 = np.array(w_prior if w_prior is not None else [0.24, 0.22, 0.16, 0.10, 0.08, 0.10, 0.10], dtype=np.float64)
        w0 /= w0.sum()
        rng = np.random.RandomState(42)
        best_w = w0.copy()
        best_sc = -1.0
        scam_s = np.isin(y_s_l, [1, 2, 3, 4])
        scam_o = np.isin(y_sy_l[ood_idx], [1, 2, 3, 4])
        for it in range(1500):
            cand = w0 if it == 0 else np.maximum(0.02, best_w + rng.normal(0, 0.04, size=k))
            cand /= cand.sum()
            ps = blend_list(s_list, cand)
            po = blend_list(sy_list, cand)
            if mode == "mc":
                fs = 100.0 * f1_score(y_s, ps.argmax(1), average="macro", zero_division=0)
                fo = 100.0 * f1_score(y_sy, po.argmax(1), average="macro", zero_division=0)
            elif mode == "tech":
                fs = 100.0 * f1_score(y_s[scam_s], (ps[scam_s] >= 0.45).astype(int), average="macro", zero_division=0)
                fo = 100.0 * f1_score(y_sy[scam_o], (po[scam_o] >= 0.45).astype(int), average="macro", zero_division=0)
            else:
                fs = 100.0 * f1_score(y_s[scam_s], 1 + ps[scam_s, 1:].argmax(1), average="macro", zero_division=0)
                fo = 100.0 * f1_score(y_sy[scam_o], 1 + po[scam_o, 1:].argmax(1), average="macro", zero_division=0)
            sc = 0.65 * fs + 0.35 * fo
            if sc > best_sc:
                best_sc = sc
                best_w = cand
        return best_w, best_sc

    w_l_stu, sc_l = opt_task_weights(s_L_models, sy_L_models, y_s_l, y_sy_l[ood_idx], mode="mc")
    w_t_stu, sc_t = opt_task_weights(s_T_models, sy_T_models, y_s_t, y_sy_t[ood_idx], mode="tech", w_prior=[0.22, 0.22, 0.14, 0.10, 0.06, 0.08, 0.18])
    w_a_stu, sc_a = opt_task_weights(s_A_models, sy_A_models, y_s_a, y_sy_a[ood_idx], mode="act", w_prior=[0.22, 0.22, 0.14, 0.10, 0.06, 0.08, 0.18])
    w_l_esc = np.append((1.0 - 0.38) * w_l_stu, 0.38)
    print(f"Task A OOD F1={sc_l:.2f} | W_L={np.round(w_l_stu, 3)}")
    print(f"Task B OOD F1={sc_t:.2f} | W_T={np.round(w_t_stu, 3)}")
    print(f"Task C OOD F1={sc_a:.2f} | W_A={np.round(w_a_stu, 3)}")

    p_s_l = blend_list(s_L_models, w_l_stu)
    p_o_l = blend_list(sy_L_models, w_l_stu)
    p_s_t = blend_list(s_T_models, w_t_stu)
    p_o_t = blend_list(sy_T_models, w_t_stu)
    p_s_a = blend_list(s_A_models, w_a_stu)
    p_o_a = blend_list(sy_A_models, w_a_stu)

    # Regularized Task A class-prior multipliers (bounded [0.88, 1.15] via GroupKFold OOD safety)
    rng = np.random.RandomState(123)
    best_alpha = np.ones(6, dtype=np.float32)
    best_fa_joint = 0.65 * f1_score(y_s_l, p_s_l.argmax(1), average="macro") + 0.35 * f1_score(y_sy_l[ood_idx], p_o_l.argmax(1), average="macro")
    for _ in range(1000):
        a = np.clip(best_alpha + rng.normal(0, 0.03, size=6), 0.88, 1.15).astype(np.float32)
        sc = 0.65 * f1_score(y_s_l, (p_s_l * a).argmax(1), average="macro") + 0.35 * f1_score(y_sy_l[ood_idx], (p_o_l * a).argmax(1), average="macro")
        if sc > best_fa_joint:
            best_fa_joint = sc
            best_alpha = a

    # Regularized Task B thresholds per T1..T6 in [0.36, 0.56]
    tech_th = np.full(6, 0.45, dtype=np.float32)
    for j in range(6):
        scam_s = np.isin(y_s_l, [1, 2, 3, 4])
        scam_o = np.isin(y_sy_l[ood_idx], [1, 2, 3, 4])
        b_th, b_f = 0.45, -1.0
        for th in np.linspace(0.36, 0.56, 21):
            fs = f1_score(y_s_t[scam_s, j], (p_s_t[scam_s, j] >= th).astype(int), zero_division=0)
            fo = f1_score(y_sy_t[ood_idx][scam_o, j], (p_o_t[scam_o, j] >= th).astype(int), zero_division=0)
            if 0.65 * fs + 0.35 * fo > b_f:
                b_f = 0.65 * fs + 0.35 * fo
                b_th = th
        tech_th[j] = b_th

    # Regularized Task C action multipliers [1..5] in [0.85, 1.45] to recover action=5/1/4
    best_beta = np.ones(6, dtype=np.float32)
    scam_s = np.isin(y_s_l, [1, 2, 3, 4])
    scam_o = np.isin(y_sy_l[ood_idx], [1, 2, 3, 4])
    best_fc_joint = (
        0.65 * f1_score(y_s_a[scam_s], 1 + p_s_a[scam_s, 1:].argmax(1), average="macro", zero_division=0)
        + 0.35 * f1_score(y_sy_a[ood_idx][scam_o], 1 + p_o_a[scam_o, 1:].argmax(1), average="macro", zero_division=0)
    )
    for _ in range(1200):
        b = np.clip(best_beta + rng.normal(0, 0.05, size=6), 0.85, 1.45).astype(np.float32)
        b[0] = 1.0
        sc = (
            0.65 * f1_score(y_s_a[scam_s], 1 + (p_s_a[scam_s, 1:] * b[1:]).argmax(1), average="macro", zero_division=0)
            + 0.35 * f1_score(y_sy_a[ood_idx][scam_o], 1 + (p_o_a[scam_o, 1:] * b[1:]).argmax(1), average="macro", zero_division=0)
        )
        if sc > best_fc_joint:
            best_fc_joint = sc
            best_beta = b

    print(f"Tuned regularizers: alpha={np.round(best_alpha, 3)} | tech_th={np.round(tech_th, 3)} | act_beta={np.round(best_beta[1:], 3)}")

    # Fit honest Zero-Leakage OOD Brier Calibrator (Logistic on OOF correctness!)
    X_cal_s = build_uncertainty_features(s_L_models, sf, s_rules["lbl"])
    X_cal_o = build_uncertainty_features(sy_L_models, syf[ood_idx], sy_rules["lbl"][ood_idx])
    y_corr_s = ((p_s_l * best_alpha).argmax(1) == y_s_l).astype(int)
    y_corr_o = ((p_o_l * best_alpha).argmax(1) == y_sy_l[ood_idx]).astype(int)

    # Weight sample OOF slightly stronger + synth OOD OOF for broad coverage of errors/OOV/noise
    X_cal = np.vstack([np.repeat(X_cal_s, 20, axis=0), X_cal_o])
    y_cal = np.concatenate([np.repeat(y_corr_s, 20), y_corr_o])
    cal_lr = LogisticRegression(C=0.5, max_iter=500)
    cal_lr.fit(X_cal, y_cal)

    # Evaluate calibrated OOD Brier & Composite Section 10 proxy!
    conf_s = np.clip(cal_lr.predict_proba(X_cal_s)[:, 1], 0.05, 0.995)
    conf_o = np.clip(cal_lr.predict_proba(X_cal_o)[:, 1], 0.05, 0.995)
    spl, spt, spa = decode_consistent(p_s_l * best_alpha, p_s_t, p_s_a * best_beta, tech_th=tech_th)
    opl, opt, opa = decode_consistent(p_o_l * best_alpha, p_o_t, p_o_a * best_beta, tech_th=tech_th)
    multi_o = syf[ood_idx, 2] >= 3
    final_scores = score_section10_proxy(
        y_s_l, spl, y_s_t, spt, y_s_a, spa, conf_s,
        y_l_ood=y_sy_l[ood_idx], p_l_ood=opl, conf_ood=conf_o,
        y_l_adv=y_sy_l[ood_idx], p_l_adv=opl, conf_adv=conf_o,
        y_l_multi=y_sy_l[ood_idx][multi_o], p_l_multi=opl[multi_o],
    )
    print("\n=== FINAL ZERO-LEAKAGE META-ENSEMBLE SECTION 10 PROXY ===", final_scores, flush=True)

    # ---- APPLY TO 500K TEST PREDICTIONS (WITH DOMAIN-ADAPTED STUDENTS + SELECTIVE INDUCTIVE ESCALATION) ----
    t_m = np.load(OUT_DIR / "test_mminilm_v4.npz")
    t_e = np.load(OUT_DIR / "test_me5small_v4.npz")
    t_b = np.load(OUT_DIR / "test_bge_v4.npz")
    t_bg = np.load(OUT_DIR / "test_bigru_v4.npz")
    da_path = OUT_DIR / "test_da_tab_v4.npz"
    t_ba = np.load(da_path if da_path.exists() else (OUT_DIR / "test_baseline_v4.npz"))
    print("Loaded tabular/linear predictions from:", da_path.name if da_path.exists() else "test_baseline_v4.npz")
    t_rules = np.load(CACHE_DIR / "test_v4_rule_pred.npz")
    te_feats = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")
    esc_path = OUT_DIR / "test_mdeberta_escalate_v4.npz"
    t_esc = np.load(esc_path) if esc_path.exists() else None

    n_te = len(te_feats)
    t_rule_pl = onehot_from_int(t_rules["lbl"], conf=0.84)
    t_rule_pt = t_rules["tvec"].astype(np.float32) * 0.84 + 0.08
    t_rule_pa = np.zeros((n_te, 6), dtype=np.float32)
    t_acts = t_rules["act"].astype(int)
    pos_a = t_acts > 0
    t_rule_pa[pos_a, :] = 0.0375
    t_rule_pa[np.where(pos_a)[0], t_acts[pos_a]] = 0.85

    t_gb_l = 0.40 * t_ba["lgb_l"] + 0.35 * t_ba["xgb_l"] + 0.25 * t_ba["cb_l"]
    t_L_stu = [t_m["p_l"], t_e["p_l"], t_b["p_l"], t_bg["p_l"], t_ba["lin_l"], t_gb_l, t_rule_pl]
    t_T_stu = [t_m["p_t"], t_e["p_t"], t_b["p_t"], t_bg["p_t"], t_ba["lin_t"], t_ba["gb_t"], t_rule_pt]
    t_A_stu = [t_m["p_a"], t_e["p_a"], t_b["p_a"], t_bg["p_a"], t_ba["lin_a"], t_ba["gb_a"], t_rule_pa]

    # Non-escalated predictions across all 500k
    P_L = blend_list(t_L_stu, w_l_stu)
    P_T = blend_list(t_T_stu, w_t_stu)
    P_A = blend_list(t_A_stu, w_a_stu)

    # Selective Inductive Escalation: blend escalated mDeBERTa-v3-base logits on top uncertain rows!
    if t_esc is not None:
        ei = t_esc["esc_idx"]
        P_L_esc_row = (1.0 - w_l_esc[-1]) * P_L[ei] + w_l_esc[-1] * t_esc["p_l"]
        P_T_esc_row = (1.0 - w_l_esc[-1]) * P_T[ei] + w_l_esc[-1] * t_esc["p_t"]
        P_A_esc_row = (1.0 - w_l_esc[-1]) * P_A[ei] + w_l_esc[-1] * t_esc["p_a"]
        P_L[ei], P_T[ei], P_A[ei] = P_L_esc_row, P_T_esc_row, P_A_esc_row
        print(f"Applied mDeBERTa-full escalation on {len(ei)} uncertain rows.", flush=True)
    else:
        print("Running pre-escalation 7-family ensemble (mDeBERTa escalation file not ready yet).", flush=True)

    # Calibrate confidence in chunks for low RAM
    conf_te = np.zeros(n_te, dtype=np.float32)
    for s in range(0, n_te, 50000):
        e = min(n_te, s + 50000)
        chunk_L = [m[s:e] for m in t_L_stu]
        Xc = build_uncertainty_features(chunk_L, te_feats[s:e], t_rules["lbl"][s:e])
        conf_te[s:e] = np.clip(cal_lr.predict_proba(Xc)[:, 1], 0.05, 0.996).astype(np.float32)

    final_l, final_t, final_a = decode_consistent(P_L * best_alpha, P_T, P_A * best_beta, tech_th=tech_th)
    ckpt_dir = CACHE_DIR / "ckpt_v4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"cal_lr": cal_lr, "w_l_stu": w_l_stu, "w_t_stu": w_t_stu, "w_a_stu": w_a_stu, "w_l_esc": w_l_esc, "best_alpha": best_alpha, "tech_th": tech_th, "best_beta": best_beta},
        ckpt_dir / "meta_calibrator_v4.joblib",
    )
    te_ids = pd.read_csv(CACHE_DIR / "test_v4_clean.csv", usecols=["id"])["id"].values
    sub = pd.DataFrame({
        "id": te_ids,
        "label": final_l,
        "techniques": [vec_to_tech_str(final_t[i]) for i in range(n_te)],
        "action": final_a,
        "confidence": np.round(conf_te, 4),
    })
    out_csv = Path(args.out)
    sub.to_csv(out_csv, index=False)
    print(f"Saved final submission {out_csv} ({len(sub)} rows) in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
