import gc
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
try:
    import torch
    XGB_DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    XGB_DEV = "cpu"
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
OUT_DIR = CACHE_DIR / "preds_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V2_DIR))

from cv_ood import TECH_LIST, tech_str_to_vec, vec_to_tech_str, decode_consistent, score_section10_proxy
from normalizer_v4 import format_turn_aware

# Map each sample.csv coarse scenario_family to ALL matching synth_64k_v4 families so LOSO CV has ZERO leakage
LOSO_SYNTH_EXCLUSIONS = {
    "mfs_mistake": {"wrong_tx_pin", "wrong_tx_readout", "wrong_tx_return_money"},
    "mfs_cred_harvest": {"mfs_alert_card", "mfs_account_block", "mfs_limit_aml", "mfs_breb_western", "borrowed_wallet_pin"},
    "pri1_cross_cat": {"pri1_scholarship_pin", "pri1_parcel_pin", "pri1_prize_pin", "pri1_job_pin"},
    "prize_cashback_lakh": {"prize_link_bn", "prize_link_en", "prize_fee_bn", "prize_fee_en", "prize_call_wa_bn_en"},
    "prize_gift_voucher": {"prize_usdt_survey", "prize_mystery_box", "prize_router_allowance", "prize_apk_rewards", "prize_pii_reply"},
    "job_recruitment_fee": {"job_data_entry", "job_bn_hiring_reg", "job_package_ambassador", "job_call_visa_bn"},
    "job_crypto_invest": {"job_crypto_arb", "job_double_invest"},
    "job_task_earn": {"job_task_like", "job_volunteer_review"},
    "deliv_customs_duty": {"deliv_customs_duty"},
    "deliv_addr_phish": {"deliv_addr_url", "deliv_bn_url", "deliv_call_phone"},
    "deliv_fee_link": {"deliv_storage_charge"},
    "promo_sale_disc": {"promo_sale", "promo_app_bogo"},
    "promo_offer_ad": {"promo_realestate_edu", "promo_telco"},
    "legit_scam_warn": {"legit_scam_warn"},
    "legit_otp_warn": {"legit_otp"},
    "legit_tx_bill": {"legit_tx_bill", "legit_agent_cashout"},
    "legit_deliv_ok": {"legit_cod"},
    "legit_chat_other": {"legit_chat_faq", "legit_hr_doc", "legit_tuition", "legit_noise"},
}

UNSEEN_HOLDOUT_FAMS = {
    "pri1_scholarship_pin", "wrong_tx_readout", "borrowed_wallet_pin",
    "prize_usdt_survey", "prize_router_allowance", "prize_call_wa_bn_en",
    "job_bn_hiring_reg", "job_package_ambassador",
    "deliv_addr_url", "legit_hr_doc", "legit_cod", "promo_app_bogo",
}


def make_turn_texts(clean_series):
    return [format_turn_aware(str(x)) for x in clean_series]


def fit_predict_linear_stack(X_tr, y_l, y_t, y_a, X_vals):
    """Fit fast calibrated OVR Linear models for Task A, B, C and predict on list of sparse validation/test matrices."""
    clf_l = LogisticRegression(C=2.5, max_iter=180, tol=1e-3, class_weight="balanced", solver="lbfgs")
    clf_l.fit(X_tr, y_l)

    clfs_t = []
    scam_mask = np.isin(y_l, [1, 2, 3, 4])
    for j in range(6):
        c = LogisticRegression(C=2.0, max_iter=150, tol=1e-3, class_weight="balanced")
        c.fit(X_tr[scam_mask], y_t[scam_mask, j])
        clfs_t.append(c)

    act_mask = np.isin(y_l, [1, 2, 3, 4]) & (y_a > 0)
    clf_a = LogisticRegression(C=2.0, max_iter=180, tol=1e-3, class_weight="balanced", solver="lbfgs")
    clf_a.fit(X_tr[act_mask], y_a[act_mask])

    outs = []
    for Xv in X_vals:
        pl = clf_l.predict_proba(Xv).astype(np.float32)
        pt = np.column_stack([c.predict_proba(Xv)[:, 1] for c in clfs_t]).astype(np.float32)
        pa_sub = clf_a.predict_proba(Xv).astype(np.float32)
        pa = np.zeros((Xv.shape[0], 6), dtype=np.float32)
        for col_idx, cls_id in enumerate(clf_a.classes_):
            pa[:, int(cls_id)] = pa_sub[:, col_idx]
        outs.append((pl, pt, pa))
    return outs, (clf_l, clfs_t, clf_a)


def fit_gbdt_multitask(X_tr, y_l, y_t, y_a, X_va=None, y_va_l=None, seed=42, fast=True):
    n_tr_mc = 260 if fast else 380
    lgb_l = lgb.LGBMClassifier(
        n_estimators=n_tr_mc, learning_rate=0.06, num_leaves=47, subsample_for_bin=30000,
        colsample_bytree=0.75, subsample=0.85, class_weight="balanced",
        random_state=seed, n_jobs=8, verbose=-1,
    )
    if X_va is not None:
        lgb_l.fit(X_tr, y_l, eval_set=[(X_va, y_va_l)], callbacks=[lgb.early_stopping(35, verbose=False)])
    else:
        lgb_l.fit(X_tr, y_l)

    xgb_l = xgb.XGBClassifier(
        n_estimators=220 if fast else 320, learning_rate=0.07, max_depth=6, colsample_bytree=0.7, subsample=0.85,
        tree_method="hist", device=XGB_DEV, n_jobs=8, eval_metric="mlogloss", random_state=seed + 1,
    )
    xgb_l.fit(X_tr, y_l)

    cb_l = None
    if HAS_CB:
        cb_l = CatBoostClassifier(
            iterations=220 if fast else 360, learning_rate=0.08, depth=6,
            auto_class_weights="Balanced", thread_count=8, random_seed=seed + 2, verbose=0,
        )
        cb_l.fit(X_tr, y_l)

    scam_mask = np.isin(y_l, [1, 2, 3, 4])
    lgbs_t = []
    for j in range(6):
        c = lgb.LGBMClassifier(
            n_estimators=260, learning_rate=0.05, num_leaves=31,
            colsample_bytree=0.75, subsample=0.85, class_weight="balanced",
            random_state=seed + 10 + j, n_jobs=8, verbose=-1,
        )
        c.fit(X_tr[scam_mask], y_t[scam_mask, j])
        lgbs_t.append(c)

    act_mask = scam_mask & (y_a > 0)
    ya_zero = y_a[act_mask] - 1
    lgb_a = lgb.LGBMClassifier(
        n_estimators=350, learning_rate=0.05, num_leaves=31,
        colsample_bytree=0.75, subsample=0.85, class_weight="balanced",
        random_state=seed + 99, n_jobs=8, verbose=-1,
    )
    lgb_a.fit(X_tr[act_mask], ya_zero)
    return {"lgb_l": lgb_l, "xgb_l": xgb_l, "cb_l": cb_l, "lgbs_t": lgbs_t, "lgb_a": lgb_a}


def predict_gbdt(models, X):
    p_lgb = models["lgb_l"].predict_proba(X).astype(np.float32)
    p_xgb = models["xgb_l"].predict_proba(X).astype(np.float32)
    if models.get("cb_l") is not None:
        p_cb = models["cb_l"].predict_proba(X).astype(np.float32)
        pl = 0.40 * p_lgb + 0.35 * p_xgb + 0.25 * p_cb
    else:
        p_cb = p_lgb
        pl = 0.55 * p_lgb + 0.45 * p_xgb
    pt = np.column_stack([c.predict_proba(X)[:, 1] for c in models["lgbs_t"]]).astype(np.float32)
    pa_sub = models["lgb_a"].predict_proba(X).astype(np.float32)
    pa = np.zeros((X.shape[0], 6), dtype=np.float32)
    for ci, cls_id in enumerate(models["lgb_a"].classes_):
        pa[:, int(cls_id) + 1] = pa_sub[:, ci]
    return p_lgb, p_xgb, p_cb, pl, pt, pa


def main():
    t0 = time.time()
    sdf = pd.read_csv(CACHE_DIR / "sample_v4.csv")
    sf = np.load(CACHE_DIR / "sample_v4_feats.npy")
    sy = pd.read_csv(CACHE_DIR / "synth_64k_v4.csv")
    syf = np.load(CACHE_DIR / "synth_64k_v4_feats.npy")

    s_txt = make_turn_texts(sdf["clean_text"].fillna(""))
    sy_txt = make_turn_texts(sy["clean_text"].fillna(""))

    s_y_l = sdf["label"].values.astype(int)
    s_y_t = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]])
    s_y_a = sdf["action"].values.astype(int)
    s_fams = sdf["scenario_family"].values

    sy_y_l = sy["label"].values.astype(int)
    sy_y_t = np.vstack([tech_str_to_vec(x) for x in sy["techniques"]])
    sy_y_a = sy["action"].values.astype(int)
    sy_fams = sy["scenario_family"].values

    # Vectorize TF-IDF on synth + sample cleanly
    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=80000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b|[<>\[\]\|]+")
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, sublinear_tf=True, min_df=3)
    Xw_sy = vec_w.fit_transform(sy_txt)
    Xc_sy = vec_c.fit_transform(sy_txt)
    X_tfidf_sy = hstack([Xw_sy, Xc_sy]).tocsr()

    Xw_s = vec_w.transform(s_txt)
    Xc_s = vec_c.transform(s_txt)
    X_tfidf_s = hstack([Xw_s, Xc_s]).tocsr()

    svd = TruncatedSVD(n_components=128, random_state=42)
    Z_sy = svd.fit_transform(X_tfidf_sy).astype(np.float32)
    Z_s = svd.transform(X_tfidf_s).astype(np.float32)

    scaler = StandardScaler()
    F_sy = scaler.fit_transform(syf).astype(np.float32)
    F_s = scaler.transform(sf).astype(np.float32)

    X_lin_sy = hstack([X_tfidf_sy, csr_matrix(F_sy * 0.25)]).tocsr()
    X_lin_s = hstack([X_tfidf_s, csr_matrix(F_s * 0.25)]).tocsr()

    X_tab_sy = np.hstack([F_sy, Z_sy]).astype(np.float32)
    X_tab_s = np.hstack([F_s, Z_s]).astype(np.float32)

    # ---- PILLAR 1: ZERO-LEAKAGE LOSO ON SAMPLE.CSV ----
    gkf = StratifiedGroupKFold(n_splits=5)
    oof_lin_s_l = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_lin_s_t = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_lin_s_a = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_gb_s_l = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_gb_s_t = np.zeros((len(sdf), 6), dtype=np.float32)
    oof_gb_s_a = np.zeros((len(sdf), 6), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(sdf, s_y_l, groups=s_fams)):
        va_coarse = set(s_fams[va_idx])
        excluded_synth = set()
        for cf in va_coarse:
            excluded_synth.update(LOSO_SYNTH_EXCLUSIONS.get(cf, set()))
        keep_sy = ~np.isin(sy_fams, list(excluded_synth))

        X_tr_l = vstack([X_lin_sy[keep_sy], X_lin_s[tr_idx]]).tocsr()
        y_tr_l = np.concatenate([sy_y_l[keep_sy], s_y_l[tr_idx]])
        y_tr_t = np.vstack([sy_y_t[keep_sy], s_y_t[tr_idx]])
        y_tr_a = np.concatenate([sy_y_a[keep_sy], s_y_a[tr_idx]])

        [(pl, pt, pa)], _ = fit_predict_linear_stack(X_tr_l, y_tr_l, y_tr_t, y_tr_a, [X_lin_s[va_idx]])
        oof_lin_s_l[va_idx], oof_lin_s_t[va_idx], oof_lin_s_a[va_idx] = pl, pt, pa

        # Train GBDT excluding same leaked families
        X_tr_g = np.vstack([X_tab_sy[keep_sy], np.repeat(X_tab_s[tr_idx], 2, axis=0)])
        yg_l = np.concatenate([sy_y_l[keep_sy], np.repeat(s_y_l[tr_idx], 2)])
        yg_t = np.vstack([sy_y_t[keep_sy], np.repeat(s_y_t[tr_idx], 2, axis=0)])
        yg_a = np.concatenate([sy_y_a[keep_sy], np.repeat(s_y_a[tr_idx], 2)])
        gm = fit_gbdt_multitask(X_tr_g, yg_l, yg_t, yg_a, seed=42 + fold)
        _, _, _, pgl, pgt, pga = predict_gbdt(gm, X_tab_s[va_idx])
        oof_gb_s_l[va_idx], oof_gb_s_t[va_idx], oof_gb_s_a[va_idx] = pgl, pgt, pga
        print(f"LOSO fold {fold}: excluded {len(excluded_synth)} synth fams, kept {keep_sy.sum()} rows | acc_lin={(pl.argmax(1)==s_y_l[va_idx]).mean():.3f} acc_gb={(pgl.argmax(1)==s_y_l[va_idx]).mean():.3f}", flush=True)

    # ---- PILLAR 2: UNSEEN SCENARIO FAMILY HOLDOUT ON SYNTH_64K ----
    ood_mask = np.isin(sy_fams, list(UNSEEN_HOLDOUT_FAMS))
    id_mask = ~ood_mask
    [(pl_ood, pt_ood, pa_ood)], _ = fit_predict_linear_stack(
        X_lin_sy[id_mask], sy_y_l[id_mask], sy_y_t[id_mask], sy_y_a[id_mask], [X_lin_sy[ood_mask]]
    )
    gm_ood = fit_gbdt_multitask(X_tab_sy[id_mask], sy_y_l[id_mask], sy_y_t[id_mask], sy_y_a[id_mask], seed=777)
    _, _, _, pgl_ood, pgt_ood, pga_ood = predict_gbdt(gm_ood, X_tab_sy[ood_mask])

    p_mix_s_l = 0.40 * oof_lin_s_l + 0.60 * oof_gb_s_l
    p_mix_s_t = 0.40 * oof_lin_s_t + 0.60 * oof_gb_s_t
    p_mix_s_a = 0.40 * oof_lin_s_a + 0.60 * oof_gb_s_a
    s_pl, s_pt, s_pa = decode_consistent(p_mix_s_l, p_mix_s_t, p_mix_s_a, tech_th=0.45)

    p_mix_ood_l = 0.40 * pl_ood + 0.60 * pgl_ood
    p_mix_ood_t = 0.40 * pt_ood + 0.60 * pgt_ood
    p_mix_ood_a = 0.40 * pa_ood + 0.60 * pga_ood
    ood_pl, ood_pt, ood_pa = decode_consistent(p_mix_ood_l, p_mix_ood_t, p_mix_ood_a, tech_th=0.45)

    raw_s = sdf["text"].values
    n_turns_ood = syf[ood_mask, 2]
    multi_mask = n_turns_ood >= 3

    scores = score_section10_proxy(
        s_y_l, s_pl, s_y_t, s_pt, s_y_a, s_pa, p_mix_s_l.max(axis=1),
        y_l_ood=sy_y_l[ood_mask], p_l_ood=ood_pl, conf_ood=p_mix_ood_l.max(axis=1),
        y_l_adv=sy_y_l[ood_mask], p_l_adv=ood_pl, conf_adv=p_mix_ood_l.max(axis=1),
        y_l_multi=sy_y_l[ood_mask][multi_mask], p_l_multi=ood_pl[multi_mask],
    )
    print("\n=== ZERO-LEAKAGE OOD BENCHMARK (TF-IDF Linear + LGBM/XGB) ===", scores, flush=True)

    # Save OOD OOF predictions for stacking
    np.savez_compressed(
        OUT_DIR / "oof_baseline_v4.npz",
        s_lin_l=oof_lin_s_l, s_lin_t=oof_lin_s_t, s_lin_a=oof_lin_s_a,
        s_gb_l=oof_gb_s_l, s_gb_t=oof_gb_s_t, s_gb_a=oof_gb_s_a,
        ood_mask=ood_mask,
        ood_lin_l=pl_ood, ood_lin_t=pt_ood, ood_lin_a=pa_ood,
        ood_gb_l=pgl_ood, ood_gb_t=pgt_ood, ood_gb_a=pga_ood,
    )

    # ---- FIT FULL MODELS ON ALL 64K + SAMPLE AND STREAM 500K PREDICTIONS ----
    X_full_lin = vstack([X_lin_sy, X_lin_s]).tocsr()
    y_full_l = np.concatenate([sy_y_l, s_y_l])
    y_full_t = np.vstack([sy_y_t, s_y_t])
    y_full_a = np.concatenate([sy_y_a, s_y_a])
    _, (full_l, full_ts, full_a) = fit_predict_linear_stack(X_full_lin, y_full_l, y_full_t, y_full_a, [])

    X_full_tab = np.vstack([X_tab_sy, np.repeat(X_tab_s, 3, axis=0)])
    yg_full_l = np.concatenate([sy_y_l, np.repeat(s_y_l, 3)])
    yg_full_t = np.vstack([sy_y_t, np.repeat(s_y_t, 3, axis=0)])
    yg_full_a = np.concatenate([sy_y_a, np.repeat(s_y_a, 3)])
    full_gb = fit_gbdt_multitask(X_full_tab, yg_full_l, yg_full_t, yg_full_a, seed=999, fast=False)
    ckpt_dir = CACHE_DIR / "ckpt_v4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vec_w": vec_w, "vec_c": vec_c, "svd": svd, "scaler": scaler, "lin": (full_l, full_ts, full_a), "gb": full_gb}, ckpt_dir / "baseline_v4.joblib", compress=3)

    te_f = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")
    n_te = len(te_f)
    lin_l = np.zeros((n_te, 6), dtype=np.float32)
    lin_t = np.zeros((n_te, 6), dtype=np.float32)
    lin_a = np.zeros((n_te, 6), dtype=np.float32)
    lgb_l = np.zeros((n_te, 6), dtype=np.float32)
    xgb_l = np.zeros((n_te, 6), dtype=np.float32)
    cb_l = np.zeros((n_te, 6), dtype=np.float32)
    gb_t = np.zeros((n_te, 6), dtype=np.float32)
    gb_a = np.zeros((n_te, 6), dtype=np.float32)

    chunk = 50000
    offset = 0
    for df_c in pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=chunk):
        m = len(df_c)
        txts = make_turn_texts(df_c["clean_text"].fillna(""))
        X_tw = hstack([vec_w.transform(txts), vec_c.transform(txts)]).tocsr()
        F_c = scaler.transform(te_f[offset:offset + m]).astype(np.float32)
        Z_c = svd.transform(X_tw).astype(np.float32)
        X_lc = hstack([X_tw, csr_matrix(F_c * 0.25)]).tocsr()
        X_tc = np.hstack([F_c, Z_c]).astype(np.float32)

        lin_l[offset:offset + m] = full_l.predict_proba(X_lc).astype(np.float32)
        lin_t[offset:offset + m] = np.column_stack([c.predict_proba(X_lc)[:, 1] for c in full_ts]).astype(np.float32)
        pa_sub = full_a.predict_proba(X_lc).astype(np.float32)
        for ci, cls_id in enumerate(full_a.classes_):
            lin_a[offset:offset + m, int(cls_id)] = pa_sub[:, ci]

        pl1, px1, pcb1, _, pgt1, pga1 = predict_gbdt(full_gb, X_tc)
        lgb_l[offset:offset + m] = pl1
        xgb_l[offset:offset + m] = px1
        cb_l[offset:offset + m] = pcb1
        gb_t[offset:offset + m] = pgt1
        gb_a[offset:offset + m] = pga1
        offset += m
        print(f"  Predicted {offset}/{n_te} test rows...", flush=True)

    np.savez_compressed(
        OUT_DIR / "test_baseline_v4.npz",
        lin_l=lin_l, lin_t=lin_t, lin_a=lin_a,
        lgb_l=lgb_l, xgb_l=xgb_l, cb_l=cb_l, gb_t=gb_t, gb_a=gb_a,
    )

    # Blend inductive rules_v4 signals softly with Linear + GBDT for sharp Task B/C without rigid hard overrides
    t_ru = np.load(CACHE_DIR / "test_v4_rule_pred.npz")
    te_ids = pd.read_csv(CACHE_DIR / "test_v4_clean.csv", usecols=["id"])["id"].values
    r_pl = np.full((n_te, 6), 0.03, dtype=np.float32)
    r_pl[np.arange(n_te), t_ru["lbl"].astype(int)] = 0.85
    r_pt = t_ru["tvec"].astype(np.float32) * 0.84 + 0.08
    r_pa = np.zeros((n_te, 6), dtype=np.float32)
    ra_int = t_ru["act"].astype(int)
    pos_ra = ra_int > 0
    r_pa[pos_ra, 1:] = 0.0375
    r_pa[np.where(pos_ra)[0], ra_int[pos_ra]] = 0.85

    p_l = 0.28 * lin_l + 0.24 * lgb_l + 0.20 * xgb_l + 0.16 * cb_l + 0.12 * r_pl
    p_t = 0.28 * lin_t + 0.48 * gb_t + 0.24 * r_pt
    p_a = 0.28 * lin_a + 0.48 * gb_a + 0.24 * r_pa
    pl, pt, pa = decode_consistent(p_l, p_t, p_a, tech_th=0.45)
    maxp = p_l.max(axis=1)
    is_noise = te_f[:, 15] > 0.5
    conf = np.clip(0.85 * maxp + 0.12 * (1.0 - (te_f[:, 19] < 0.35).astype(np.float32)), 0.05, 0.992)
    conf = np.where(is_noise, np.minimum(conf, 0.90), conf)
    sub = pd.DataFrame({
        "id": te_ids,
        "label": pl,
        "techniques": [vec_to_tech_str(pt[i]) for i in range(n_te)],
        "action": pa,
        "confidence": np.round(conf, 4),
    })
    sub_path = V2_DIR / "sub_v6b_v5.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved {sub_path} in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
