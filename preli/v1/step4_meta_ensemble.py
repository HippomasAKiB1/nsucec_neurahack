import os
import random
import numpy as np
import pandas as pd
from scipy.special import softmax, expit, logit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import lightgbm as lgb

from cv_metrics import (
    TECH_LIST, tech_str_to_vec, vec_to_tech_str,
    eval_multitask, build_template_groups, corrupt_text_adversarial, decode_consistent
)
from normalizer import clean_for_model, extract_structural_features

SEED = 42
CACHE_DIR = "preli/cache"
FEAT_NAMES = list(extract_structural_features("a", "a").keys())


def scale_feats(X):
    Xd = X.copy().astype(np.float32)
    Xd[:, 0] = np.log1p(Xd[:, 0]) / 5.0
    Xd[:, 1] = Xd[:, 1] / 12.0
    Xd[:, 25] = np.log1p(Xd[:, 25]) / 3.0
    Xd[:, 26] = np.log1p(Xd[:, 26]) / 3.0
    return Xd


def load_available_models():
    """Load OOF (clean + adv) and 500k memmapped test prediction arrays for every trained family."""
    models = {}
    v3_npz = os.path.join(CACHE_DIR, "oof_v3_trees.npz")
    if os.path.exists(v3_npz):
        v3 = np.load(v3_npz)
        for tag in ["lin3", "lgb3", "xgb3", "cat3"]:
            te_fns = (f"te_{tag}_l.npy", f"te_{tag}_t.npy", f"te_{tag}_a.npy")
            if all(os.path.exists(os.path.join(CACHE_DIR, t)) for t in te_fns):
                models[tag] = {
                    "oof": (v3[f"{tag}_l"], v3[f"{tag}_t"], v3[f"{tag}_a"]),
                    "adv": (v3[f"adv_{tag}_l"], v3[f"adv_{tag}_t"], v3[f"adv_{tag}_a"]),
                    "te": te_fns,
                }
    else:
        base_npz = os.path.join(CACHE_DIR, "oof_base.npz")
        if os.path.exists(base_npz):
            b = np.load(base_npz)
            models["lin"] = {
                "oof": (b["lin_l"], b["lin_t"], b["lin_a"]),
                "adv": (b["adv_lin_l"], b["adv_lin_t"], b["adv_lin_a"]),
                "adv_p": b["adv_parent_idx"],
                "te": ("te_lin_l.npy", "te_lin_t.npy", "te_lin_a.npy"),
            }
            models["lgb"] = {
                "oof": (b["lgb_l"], b["lgb_t"], b["lgb_a"]),
                "adv": (b["adv_lgb_l"], b["adv_lgb_t"], b["adv_lgb_a"]),
                "adv_p": b["adv_parent_idx"],
                "te": ("te_lgb_l.npy", "te_lgb_t.npy", "te_lgb_a.npy"),
            }

    candidates = [
        ("mminilm", "oof_mminilm.npz", ("te_mm_l.npy", "te_mm_t.npy", "te_mm_a.npy")),
        ("me5small", "oof_me5small.npz", ("te_me5_l.npy", "te_me5_t.npy", "te_me5_a.npy")),
        ("bicnn", "oof_bicnn.npz", ("te_bicnn_l.npy", "te_bicnn_t.npy", "te_bicnn_a.npy")),
        ("bge", "oof_bge.npz", ("te_bge_l.npy", "te_bge_t.npy", "te_bge_a.npy")),
    ]
    for name, npz_fn, te_fns in candidates:
        p = os.path.join(CACHE_DIR, npz_fn)
        if os.path.exists(p) and all(os.path.exists(os.path.join(CACHE_DIR, t)) for t in te_fns):
            z = np.load(p)
            models[name] = {
                "oof": (z["oof_l"], z["oof_t"], z["oof_a"]),
                "adv": (z["adv_l"], z["adv_t"], z["adv_a"]),
                "te": te_fns,
            }
    return models


def blend_probs(probs_list, weights, log_odds_mix=0.35):
    """Blend probability matrices via arithmetic + geometric (log-prob) Product-of-Experts pooling."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-12)
    eps = 1e-6
    arith = sum(wi * np.clip(p, eps, 1.0 - eps) for wi, p in zip(w, probs_list))
    log_p = sum(wi * np.log(np.clip(p, eps, 1.0 - eps)) for wi, p in zip(w, probs_list))
    geom = softmax(log_p, axis=1) if probs_list[0].shape[1] == 6 and abs(probs_list[0][0].sum() - 1.0) < 0.05 else expit(
        sum(wi * logit(np.clip(p, eps, 1.0 - eps)) for wi, p in zip(w, probs_list))
    )
    return (1.0 - log_odds_mix) * arith + log_odds_mix * geom


def blend_multilabel(probs_list, weights, log_odds_mix=0.40):
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w = w / max(w.sum(), 1e-12)
    eps = 1e-6
    arith = sum(wi * np.clip(p, eps, 1.0 - eps) for wi, p in zip(w, probs_list))
    geom = expit(sum(wi * logit(np.clip(p, eps, 1.0 - eps)) for wi, p in zip(w, probs_list)))
    return (1.0 - log_odds_mix) * arith + log_odds_mix * geom


def fit_simplex_weights_multiclass(oof_list, adv_list, y_clean, y_adv, mask_clean, mask_adv, min_w=0.0):
    """Optimize exact simplex weights via SLSQP on log-loss + mild L2 regularization."""
    m = len(oof_list)
    if m == 1:
        return np.array([1.0])
    P_tr = [np.vstack([oof_list[k][mask_clean], adv_list[k][mask_adv]]) for k in range(m)]
    y_tr = np.concatenate([y_clean[mask_clean], y_adv[mask_adv]])
    sw = np.concatenate([np.ones(mask_clean.sum()), np.full(mask_adv.sum(), 0.20)])
    sw /= sw.sum()
    idx = np.arange(len(y_tr))
    y_oh = np.eye(6, dtype=np.float64)[y_tr]
    eff_min = min(min_w, 0.5 / m)

    def obj(w):
        w = np.clip(w, 0.0, 1.0)
        w = w / max(w.sum(), 1e-12)
        p = blend_probs(P_tr, w)
        brier = np.sum(sw[:, None] * (p - y_oh) ** 2)
        nll = -np.sum(sw * np.log(np.clip(p[idx, y_tr], 1e-6, 1.0)))
        reg = 0.005 * np.sum(w ** 2)
        return brier + 0.15 * nll + reg

    # Multi-start SLSQP: uniform start + top-individual starts
    best_res = None
    inits = [np.full(m, 1.0 / m)]
    indiv_nll = [-np.sum(sw * np.log(np.clip(P_tr[k][idx, y_tr], 1e-6, 1.0))) for k in range(m)]
    best_k = int(np.argmin(indiv_nll))
    top_init = np.full(m, 0.02)
    top_init[best_k] = 1.0 - 0.02 * (m - 1)
    inits.append(top_init)
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bnds = [(eff_min, 1.0)] * m
    for x0 in inits:
        res = minimize(obj, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    w_opt = np.clip(best_res.x, eff_min, 1.0)
    return w_opt / w_opt.sum()


def fit_simplex_weights_multilabel(oof_list, adv_list, yt_clean, yt_adv, mask_clean, mask_adv, min_w=0.0):
    m = len(oof_list)
    if m == 1:
        return np.array([1.0])
    P_tr = [np.vstack([oof_list[k][mask_clean], adv_list[k][mask_adv]]) for k in range(m)]
    y_tr = np.vstack([yt_clean[mask_clean], yt_adv[mask_adv]])
    sw = np.concatenate([np.ones(mask_clean.sum()), np.full(mask_adv.sum(), 0.20)])[:, None]
    sw /= sw.sum()
    eff_min = min(min_w, 0.5 / m)

    def obj(w):
        w = np.clip(w, 0.0, 1.0)
        w = w / max(w.sum(), 1e-12)
        p = np.clip(blend_multilabel(P_tr, w), 1e-6, 1.0 - 1e-6)
        brier = np.sum(sw * (p - y_tr) ** 2)
        bce = -np.sum(sw * (y_tr * np.log(p) + (1.0 - y_tr) * np.log(1.0 - p)))
        reg = 0.005 * np.sum(w ** 2)
        return brier + 0.15 * bce + reg

    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bnds = [(eff_min, 1.0)] * m
    res = minimize(obj, np.full(m, 1.0 / m), method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    w_opt = np.clip(res.x, eff_min, 1.0)
    return w_opt / w_opt.sum()


def tune_tech_thresholds(p_tech, y_tech, pred_lbl):
    """Per-technique threshold search maximizing Task B Macro-F1 across T1..T6."""
    th = np.full(6, 0.50)
    is_scam = ~np.isin(pred_lbl, [0, 5])
    if is_scam.sum() < 5:
        return th
    for j in range(6):
        best_f, best_t = -1.0, 0.50
        for cand in np.linspace(0.30, 0.65, 15):
            pred_j = ((p_tech[:, j] >= cand) & is_scam).astype(int)
            f = f1_score(y_tech[:, j], pred_j, zero_division=0)
            if f > best_f + 1e-9 or (abs(f - best_f) < 1e-9 and abs(cand - 0.50) < abs(best_t - 0.50)):
                best_f, best_t = f, float(cand)
        th[j] = best_t
    return th


def extract_conf_features(p_lbl, agreement_vec):
    eps = 1e-6
    ps = np.sort(np.clip(p_lbl, eps, 1.0 - eps), axis=1)
    p1 = ps[:, -1]
    p2 = ps[:, -2]
    margin = p1 - p2
    entropy = -np.sum(np.clip(p_lbl, eps, 1.0) * np.log(np.clip(p_lbl, eps, 1.0)), axis=1)
    return np.column_stack([logit(np.clip(p1, 0.01, 0.99)), p1, margin, entropy, agreement_vec])


def compute_model_agreement(probs_list):
    if len(probs_list) <= 1:
        return np.ones(len(probs_list[0]), dtype=np.float32)
    ens_cls = sum(probs_list).argmax(axis=1)
    agree = np.mean([(p.argmax(axis=1) == ens_cls).astype(np.float32) for p in probs_list], axis=0)
    return agree


def main():
    s_df = pd.read_csv("preli/cache/sample_clean.csv")
    s_texts = s_df["clean_text"].fillna("").astype(str).tolist()
    s_feats = scale_feats(np.load("preli/cache/sample_feats.npy"))
    y_lbl = s_df["label"].values
    y_tech = np.vstack([tech_str_to_vec(x) for x in s_df["techniques"]])
    y_act = s_df["action"].values
    groups = build_template_groups(s_texts)

    rng_adv = random.Random(SEED + 99)
    adv_clean, adv_feats_raw, adv_p = [], [], []
    for i, raw in enumerate(s_df["text"].tolist()):
        for _ in range(5):
            cr = corrupt_text_adversarial(raw, rng_adv)
            cc = clean_for_model(cr)
            d = extract_structural_features(cr, cc)
            adv_clean.append(cc)
            adv_feats_raw.append([d[k] for k in FEAT_NAMES])
            adv_p.append(i)
    adv_feats = scale_feats(np.array(adv_feats_raw, dtype=np.float32))
    adv_p = np.array(adv_p)

    models = load_available_models()
    m_names = list(models.keys())
    print(f"Loaded {len(m_names)} model families: {m_names}", flush=True)
    assert len(m_names) >= 2, "Need at least 2 trained model families in preli/cache!"

    # Report each individual model's Group-CV & Robust-CV cleanly
    for name in m_names:
        ol, ot, oa = models[name]["oof"]
        al, at, aa = models[name]["adv"]
        pl, pt, pa = decode_consistent(ol, ot, oa)
        apl, apt, apa = decode_consistent(al, at, aa)
        r_c = eval_multitask(y_lbl, pl, y_tech, pt, y_act, pa, ol.max(1))
        r_a = eval_multitask(y_lbl[adv_p], apl, y_tech[adv_p], apt, y_act[adv_p], apa, al.max(1))
        print(f"  [{name:10s}] Group-CV={r_c['CompositeProxy']:.2f} (L={r_c['F_label']:.2f}, T={r_c['F_tech']:.2f}, A={r_c['F_action']:.2f}, B={r_c['Brier']:.4f}) | Robust-CV={r_a['CompositeProxy']:.2f}", flush=True)

    # Strict Nested 5-Fold StratifiedGroupKFold Evaluation:
    # Optimize Task A (label) & Task C (action) over models that generalize cleanly across Banglish+Bengali,
    # while optimizing Task B (techniques) across all 8 models (including Romanized/Glossed BGE which boosts T4->100%),
    # plus a controlled neural diversity blend (when bge+mminilm exist) verified on Group-CV and Robust-CV.
    sgkf = StratifiedGroupKFold(n_splits=5)
    core_names = [n for n in m_names if n != "bge"] if "bge" in m_names and len(m_names) >= 7 else m_names
    core_idx = [m_names.index(n) for n in core_names]

    oof_l_list = [models[n]["oof"][0] for n in core_names]
    oof_t_list = [models[n]["oof"][1] for n in m_names]
    oof_a_list = [models[n]["oof"][2] for n in core_names]
    adv_l_list = [models[n]["adv"][0] for n in core_names]
    adv_t_list = [models[n]["adv"][1] for n in m_names]
    adv_a_list = [models[n]["adv"][2] for n in core_names]

    nested_l = np.zeros((86, 6), dtype=np.float64)
    nested_t = np.zeros((86, 6), dtype=np.float64)
    nested_a = np.zeros((86, 6), dtype=np.float64)
    nested_adv_l = np.zeros((len(adv_p), 6), dtype=np.float64)
    nested_adv_t = np.zeros((len(adv_p), 6), dtype=np.float64)
    nested_adv_a = np.zeros((len(adv_p), 6), dtype=np.float64)
    nested_th = np.zeros((86, 6), dtype=np.float64)
    nested_adv_th = np.zeros((len(adv_p), 6), dtype=np.float64)
    nested_conf = np.zeros(86, dtype=np.float64)
    nested_adv_conf = np.zeros(len(adv_p), dtype=np.float64)

    fold_w_l, fold_w_t, fold_w_a, fold_ths = [], [], [], []
    all_oof_l = [models[n]["oof"][0] for n in m_names]
    all_adv_l = [models[n]["adv"][0] for n in m_names]
    agree_clean_all = compute_model_agreement(all_oof_l)
    agree_adv_all = compute_model_agreement(all_adv_l)

    has_div_pair = ("mminilm" in m_names) and ("bge" in m_names)
    idx_mm = m_names.index("mminilm") if "mminilm" in m_names else -1
    idx_bge = m_names.index("bge") if "bge" in m_names else -1

    for fold, (tr, va) in enumerate(sgkf.split(s_feats, y_lbl, groups)):
        m_c = np.zeros(86, dtype=bool)
        m_c[tr] = True
        m_a = np.isin(adv_p, tr)
        va_a = np.isin(adv_p, va)

        w_l = fit_simplex_weights_multiclass(oof_l_list, adv_l_list, y_lbl, y_lbl[adv_p], m_c, m_a)
        w_t = fit_simplex_weights_multilabel(oof_t_list, adv_t_list, y_tech, y_tech[adv_p], m_c, m_a)
        w_a = fit_simplex_weights_multiclass(oof_a_list, adv_a_list, y_act, y_act[adv_p], m_c, m_a)
        fold_w_l.append(w_l); fold_w_t.append(w_t); fold_w_a.append(w_a)

        tr_bl_l = blend_probs([np.vstack([oof_l_list[k][tr], adv_l_list[k][m_a]]) for k in range(len(core_names))], w_l)
        tr_bl_t = blend_multilabel([np.vstack([oof_t_list[k][tr], adv_t_list[k][m_a]]) for k in range(len(m_names))], w_t)
        tr_yt = np.vstack([y_tech[tr], y_tech[adv_p][m_a]])
        tr_yl = np.concatenate([y_lbl[tr], y_lbl[adv_p][m_a]])
        th = tune_tech_thresholds(tr_bl_t, tr_yt, tr_bl_l.argmax(1))
        fold_ths.append(th)

        va_l = blend_probs([p[va] for p in oof_l_list], w_l)
        va_adv_l = blend_probs([p[va_a] for p in adv_l_list], w_l)
        if has_div_pair:
            va_l = 0.94 * va_l + 0.03 * all_oof_l[idx_mm][va] + 0.03 * all_oof_l[idx_bge][va]
            va_adv_l = 0.94 * va_adv_l + 0.03 * all_adv_l[idx_mm][va_a] + 0.03 * all_adv_l[idx_bge][va_a]

        nested_l[va] = va_l
        nested_t[va] = blend_multilabel([p[va] for p in oof_t_list], w_t)
        nested_a[va] = blend_probs([p[va] for p in oof_a_list], w_a)
        nested_th[va] = th

        nested_adv_l[va_a] = va_adv_l
        nested_adv_t[va_a] = blend_multilabel([p[va_a] for p in adv_t_list], w_t)
        nested_adv_a[va_a] = blend_probs([p[va_a] for p in adv_a_list], w_a)
        nested_adv_th[va_a] = th

        # OOF Confidence calibrator (Platt logistic regression on max_p/logit/margin/entropy/agreement)
        X_cal_tr = extract_conf_features(tr_bl_l, np.concatenate([agree_clean_all[tr], agree_adv_all[m_a]]))
        y_cal_tr = (tr_bl_l.argmax(1) == tr_yl).astype(int)
        # Ensure both classes present cleanly
        cal = LogisticRegression(C=0.5)
        if len(np.unique(y_cal_tr)) == 2:
            cal.fit(X_cal_tr, y_cal_tr)
            nested_conf[va] = np.clip(cal.predict_proba(extract_conf_features(nested_l[va], agree_clean_all[va]))[:, 1], 0.05, 0.992)
            nested_adv_conf[va_a] = np.clip(cal.predict_proba(extract_conf_features(nested_adv_l[va_a], agree_adv_all[va_a]))[:, 1], 0.05, 0.992)
        else:
            nested_conf[va] = np.clip(nested_l[va].max(1), 0.05, 0.992)
            nested_adv_conf[va_a] = np.clip(nested_adv_l[va_a].max(1), 0.05, 0.992)

    # Compare Nested Ensemble (threshold-tuned vs 0.5, calibrated conf vs raw max_p)
    pl_50, pt_50, pa_50 = decode_consistent(nested_l, nested_t, nested_a, tech_th=0.50)
    pl_th = pl_50
    pt_th = (nested_t >= nested_th).astype(int)
    is_np = np.isin(pl_th, [0, 5])
    pt_th[is_np] = 0
    need = (~is_np) & (pt_th.sum(1) == 0)
    if need.any():
        pt_th[need, nested_t[need].argmax(1)] = 1

    r_raw_conf = eval_multitask(y_lbl, pl_50, y_tech, pt_50, y_act, pa_50, nested_l.max(1))
    r_th_cal = eval_multitask(y_lbl, pl_th, y_tech, pt_th, y_act, pa_50, nested_conf)

    apl_50, apt_50, apa_50 = decode_consistent(nested_adv_l, nested_adv_t, nested_adv_a, tech_th=0.50)
    apt_th = (nested_adv_t >= nested_adv_th).astype(int)
    is_np_a = np.isin(apl_50, [0, 5])
    apt_th[is_np_a] = 0
    need_a = (~is_np_a) & (apt_th.sum(1) == 0)
    if need_a.any():
        apt_th[need_a, nested_adv_t[need_a].argmax(1)] = 1

    r_adv_raw = eval_multitask(y_lbl[adv_p], apl_50, y_tech[adv_p], apt_50, y_act[adv_p], apa_50, nested_adv_l.max(1))
    r_adv_cal = eval_multitask(y_lbl[adv_p], apl_50, y_tech[adv_p], apt_th, y_act[adv_p], apa_50, nested_adv_conf)

    print("\n[Nested Ensemble th=0.5, raw conf] Group-CV :", {k: round(v, 3) for k, v in r_raw_conf.items()}, flush=True)
    print("[Nested Ensemble th=opt, platt  ] Group-CV :", {k: round(v, 3) for k, v in r_th_cal.items()}, flush=True)
    print("[Nested Ensemble th=0.5, raw conf] Robust-CV:", {k: round(v, 3) for k, v in r_adv_raw.items()}, flush=True)
    print("[Nested Ensemble th=opt, platt  ] Robust-CV:", {k: round(v, 3) for k, v in r_adv_cal.items()}, flush=True)

    # Decide threshold & calibrator conservatively: never fit Platt when too few (<15) OOF errors exist!
    n_cal_neg = int((pl_50 != y_lbl).sum() + (apl_50 != y_lbl[adv_p]).sum())
    use_opt_th = (r_th_cal["F_tech"] > r_raw_conf["F_tech"] + 0.2) and (r_adv_cal["F_tech"] >= r_adv_raw["F_tech"] - 0.1)
    use_platt = (n_cal_neg >= 15) and (r_th_cal["Brier"] < r_raw_conf["Brier"]) and (r_adv_cal["Brier"] <= r_adv_raw["Brier"] + 0.002)

    final_w_l_core = np.mean(fold_w_l, axis=0)
    final_w_t = np.mean(fold_w_t, axis=0)
    final_w_a_core = np.mean(fold_w_a, axis=0)
    final_th = np.mean(fold_ths, axis=0) if use_opt_th else np.full(6, 0.50)

    # Map core weights back to m_names cleanly
    final_w_l = np.zeros(len(m_names), dtype=np.float64)
    final_w_a = np.zeros(len(m_names), dtype=np.float64)
    for ci, mi in enumerate(core_idx):
        final_w_l[mi] = final_w_l_core[ci]
        final_w_a[mi] = final_w_a_core[ci]
    if has_div_pair:
        final_w_l = 0.94 * final_w_l
        final_w_l[idx_mm] += 0.03
        final_w_l[idx_bge] += 0.03

    print("\nFinal Model Weights:")
    for n, wl, wt, wa in zip(m_names, final_w_l, final_w_t, final_w_a):
        print(f"  {n:10s}: w_label={wl:.3f}, w_tech={wt:.3f}, w_action={wa:.3f}", flush=True)
    print(f"Use tuned tech thresholds={use_opt_th}: {np.round(final_th, 3)}", flush=True)
    print(f"Use Platt calibrator={use_platt} (OOF error count={n_cal_neg})", flush=True)

    print("\nRemaining Errors of Nested Ensemble on sample.csv Group-CV:", flush=True)
    pt_show = pt_th if use_opt_th else pt_50
    print("Ensemble Per-tech F1:", {TECH_LIST[j]: round(100 * f1_score(y_tech[:, j], pt_show[:, j]), 2) for j in range(6)}, flush=True)
    for i in range(len(s_df)):
        if pl_50[i] != y_lbl[i] or pa_50[i] != y_act[i]:
            print(f"  id={s_df.id.iloc[i]} L:{y_lbl[i]}->{pl_50[i]}({nested_l[i].max():.2f}) A:{y_act[i]}->{pa_50[i]} T:{s_df.techniques.iloc[i]}->{vec_to_tech_str(pt_show[i])} | {s_df.text.iloc[i][:90]}", flush=True)

    # Fit final confidence calibrator ONLY when enough errors prevent intercept saturation; otherwise use smooth agreement-moderated max probability
    X_cal_all = np.vstack([
        extract_conf_features(nested_l, agree_clean_all),
        extract_conf_features(nested_adv_l, agree_adv_all),
    ])
    y_cal_all = np.concatenate([(pl_50 == y_lbl).astype(int), (apl_50 == y_lbl[adv_p]).astype(int)])
    cal_final = LogisticRegression(C=0.5)
    if use_platt and len(np.unique(y_cal_all)) == 2:
        cal_final.fit(X_cal_all, y_cal_all)

    # Stream 500,000 test predictions chunkwise using mmap_mode="r" (< 150 MB RAM!)
    te_l_maps = [np.load(os.path.join(CACHE_DIR, models[n]["te"][0]), mmap_mode="r") for n in m_names]
    te_t_maps = [np.load(os.path.join(CACHE_DIR, models[n]["te"][1]), mmap_mode="r") for n in m_names]
    te_a_maps = [np.load(os.path.join(CACHE_DIR, models[n]["te"][2]), mmap_mode="r") for n in m_names]
    sub_ids = pd.read_csv(os.path.join(CACHE_DIR, "test_clean.csv"), usecols=["id"])["id"].values

    out_path = "preli/submission_final.csv"
    if os.path.exists(out_path):
        os.remove(out_path)
    for start in range(0, 500000, 50000):
        end = min(500000, start + 50000)
        cl_l = [np.array(m[start:end], dtype=np.float64) for m in te_l_maps]
        cl_t = [np.array(m[start:end], dtype=np.float64) for m in te_t_maps]
        cl_a = [np.array(m[start:end], dtype=np.float64) for m in te_a_maps]

        bl_l_core = blend_probs([cl_l[mi] for mi in core_idx], final_w_l_core)
        bl_l = (0.94 * bl_l_core + 0.03 * cl_l[idx_mm] + 0.03 * cl_l[idx_bge]) if has_div_pair else bl_l_core
        bl_t = blend_multilabel(cl_t, final_w_t)
        bl_a = blend_probs([cl_a[mi] for mi in core_idx], final_w_a_core)

        pred_l, pred_t, pred_a = decode_consistent(bl_l, bl_t, bl_a, tech_th=final_th)
        pred_t_str = [vec_to_tech_str(v) for v in pred_t]

        agree_c = compute_model_agreement(cl_l)
        if use_platt and len(np.unique(y_cal_all)) == 2:
            conf_c = np.clip(cal_final.predict_proba(extract_conf_features(bl_l, agree_c))[:, 1], 0.08, 0.988)
        else:
            # Well-calibrated ensemble confidence bounded safely for Brier stability
            conf_c = np.clip(0.88 * bl_l.max(1) + 0.12 * agree_c, 0.10, 0.985)

        df_c = pd.DataFrame({
            "id": sub_ids[start:end],
            "label": pred_l,
            "techniques": pred_t_str,
            "action": pred_a,
            "confidence": np.round(conf_c, 4),
        })
        df_c.to_csv(out_path, mode="a", header=(start == 0), index=False)

    print(f"\nSAVED {out_path} cleanly!", flush=True)


if __name__ == "__main__":
    main()
