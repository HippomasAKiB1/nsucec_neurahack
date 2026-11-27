import os
import time
import random
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from normalizer import clean_for_model, extract_structural_features
from extra_features import extract_extra_feats_batch
from cv_metrics import (
    TECH_LIST, tech_str_to_vec, vec_to_tech_str,
    eval_multitask, build_template_groups, corrupt_text_adversarial, decode_consistent
)

SEED = 777
np.random.seed(SEED)
random.seed(SEED)
FEAT_NAMES = list(extract_structural_features("a", "a").keys())


def scale_dense(X_27, X_4):
    Xd = X_27.copy().astype(np.float32)
    Xd[:, 0] = np.log1p(Xd[:, 0]) / 5.0
    Xd[:, 1] = Xd[:, 1] / 12.0
    Xd[:, 25] = np.log1p(Xd[:, 25]) / 3.0
    Xd[:, 26] = np.log1p(Xd[:, 26]) / 3.0
    return np.hstack([Xd, X_4.astype(np.float32)])


def main():
    s_df = pd.read_csv("preli/cache/sample_clean.csv")
    s_clean = s_df["clean_text"].fillna("").astype(str).tolist()
    s_dense = scale_dense(np.load("preli/cache/sample_feats.npy"), np.load("preli/cache/sample_extra.npy"))
    y_lbl = s_df["label"].values
    y_tech = np.vstack([tech_str_to_vec(x) for x in s_df["techniques"]])
    y_act = s_df["action"].values
    groups = build_template_groups(s_clean)

    rng_adv = random.Random(42 + 99)
    adv_clean, adv_27, adv_p = [], [], []
    for i, raw in enumerate(s_df["text"].tolist()):
        for _ in range(5):
            cr = corrupt_text_adversarial(raw, rng_adv)
            cc = clean_for_model(cr)
            d = extract_structural_features(cr, cc)
            adv_clean.append(cc)
            adv_27.append([d[k] for k in FEAT_NAMES])
            adv_p.append(i)
    adv_dense = scale_dense(np.array(adv_27, dtype=np.float32), extract_extra_feats_batch(adv_clean))
    adv_p = np.array(adv_p)

    syn_df = pd.read_csv("preli/cache/synth_32k_v3.csv")
    syn_clean = syn_df["clean_text"].fillna("").astype(str).tolist()
    syn_dense = scale_dense(np.load("preli/cache/synth_32k_v3_feats.npy"), np.load("preli/cache/synth_32k_v3_extra.npy"))
    y_syn_lbl = syn_df["label"].values
    y_syn_tech = np.vstack([tech_str_to_vec(x) for x in syn_df["techniques"]])
    y_syn_act = syn_df["action"].values

    vec = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 3), max_features=50000, sublinear_tf=True, min_df=2)),
        ("c", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=60000, sublinear_tf=True, min_df=3)),
    ])
    X_syn_tf = vec.fit_transform(syn_clean)
    X_s_tf = vec.transform(s_clean)
    X_adv_tf = vec.transform(adv_clean)

    X_syn_all = hstack([X_syn_tf, csr_matrix(syn_dense)], format="csr")
    X_s_all = hstack([X_s_tf, csr_matrix(s_dense)], format="csr")
    X_adv_all = hstack([X_adv_tf, csr_matrix(adv_dense)], format="csr")

    def fit_lin(X_tr, yl, yt, ya, sw=None):
        clf_l = LogisticRegression(C=3.0, max_iter=350, solver="lbfgs").fit(X_tr, yl, sample_weight=sw)
        clf_t = [LogisticRegression(C=2.5, max_iter=350, solver="lbfgs").fit(X_tr, yt[:, j], sample_weight=sw) for j in range(6)]
        clf_a = LogisticRegression(C=3.0, max_iter=350, solver="lbfgs").fit(X_tr, ya, sample_weight=sw)
        return clf_l, clf_t, clf_a

    def pred_lin(mdls, X_te):
        clf_l, clf_t, clf_a = mdls
        return (
            clf_l.predict_proba(X_te).astype(np.float32),
            np.column_stack([m.predict_proba(X_te)[:, 1] for m in clf_t]).astype(np.float32),
            clf_a.predict_proba(X_te).astype(np.float32),
        )

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_l3, oof_t3, oof_a3 = np.zeros((86, 6), dtype=np.float32), np.zeros((86, 6), dtype=np.float32), np.zeros((86, 6), dtype=np.float32)
    adv_l3, adv_t3, adv_a3 = np.zeros((len(adv_clean), 6), dtype=np.float32), np.zeros((len(adv_clean), 6), dtype=np.float32), np.zeros((len(adv_clean), 6), dtype=np.float32)

    for tr, va in sgkf.split(X_s_all, y_lbl, groups):
        X_tr = vstack([X_syn_all, X_s_all[tr]], format="csr")
        yl_tr = np.concatenate([y_syn_lbl, y_lbl[tr]])
        yt_tr = np.vstack([y_syn_tech, y_tech[tr]])
        ya_tr = np.concatenate([y_syn_act, y_act[tr]])
        sw = np.concatenate([np.ones(len(syn_df)), np.full(len(tr), 4.0)])
        mdls = fit_lin(X_tr, yl_tr, yt_tr, ya_tr, sw=sw)
        oof_l3[va], oof_t3[va], oof_a3[va] = pred_lin(mdls, X_s_all[va])
        am = np.isin(adv_p, va)
        adv_l3[am], adv_t3[am], adv_a3[am] = pred_lin(mdls, X_adv_all[am])

    pl3, pt3, pa3 = decode_consistent(oof_l3, oof_t3, oof_a3)
    apl3, apt3, apa3 = decode_consistent(adv_l3, adv_t3, adv_a3)
    r3 = eval_multitask(y_lbl, pl3, y_tech, pt3, y_act, pa3, oof_l3.max(1))
    ra3 = eval_multitask(y_lbl[adv_p], apl3, y_tech[adv_p], apt3, y_act[adv_p], apa3, adv_l3.max(1))
    print("[v3 Linear TFIDF+31Feats] Group-CV :", {k: round(v, 2) for k, v in r3.items()}, flush=True)
    print("[v3 Linear TFIDF+31Feats] Robust-CV:", {k: round(v, 2) for k, v in ra3.items()}, flush=True)
    from sklearn.metrics import f1_score
    print("  v3 Linear Per-tech F1:", {TECH_LIST[j]: round(100 * f1_score(y_tech[:, j], pt3[:, j]), 2) for j in range(6)}, flush=True)

    # 5-fold synthetic OOF for v3 GBDTs (LightGBM + CatBoost + XGBoost)
    syn_oof_l = np.zeros((len(syn_df), 6), dtype=np.float32)
    syn_oof_t = np.zeros((len(syn_df), 6), dtype=np.float32)
    syn_oof_a = np.zeros((len(syn_df), 6), dtype=np.float32)
    idx_all = np.arange(len(syn_df))
    for f_idx in range(5):
        va_m = (idx_all % 5 == f_idx)
        mdls = fit_lin(X_syn_all[~va_m], y_syn_lbl[~va_m], y_syn_tech[~va_m], y_syn_act[~va_m])
        syn_oof_l[va_m], syn_oof_t[va_m], syn_oof_a[va_m] = pred_lin(mdls, X_syn_all[va_m])

    # Full v3 Linear fit & 500k streaming
    X_full = vstack([X_syn_all, X_s_all], format="csr")
    yl_full = np.concatenate([y_syn_lbl, y_lbl])
    yt_full = np.vstack([y_syn_tech, y_tech])
    ya_full = np.concatenate([y_syn_act, y_act])
    sw_full = np.concatenate([np.ones(len(syn_df)), np.full(len(s_df), 4.0)])
    full_lin = fit_lin(X_full, yl_full, yt_full, ya_full, sw=sw_full)

    te_dense = scale_dense(np.load("preli/cache/test_feats.npy"), np.load("preli/cache/test_extra.npy"))
    te_l3 = np.zeros((500000, 6), dtype=np.float32)
    te_t3 = np.zeros((500000, 6), dtype=np.float32)
    te_a3 = np.zeros((500000, 6), dtype=np.float32)
    off = 0
    for chunk in pd.read_csv("preli/cache/test_clean.csv", chunksize=50000):
        n = len(chunk)
        Xc = hstack([vec.transform(chunk["clean_text"].fillna("").astype(str).tolist()), csr_matrix(te_dense[off:off+n])], format="csr")
        te_l3[off:off+n], te_t3[off:off+n], te_a3[off:off+n] = pred_lin(full_lin, Xc)
        off += n

    Xm_syn = np.hstack([syn_oof_l, syn_oof_t, syn_oof_a, syn_dense])
    Xm_s = np.hstack([oof_l3, oof_t3, oof_a3, s_dense])
    Xm_adv = np.hstack([adv_l3, adv_t3, adv_a3, adv_dense])
    Xm_te = np.hstack([te_l3, te_t3, te_a3, te_dense])

    def run_gbdt_family(name, make_mc, make_bin):
        ol = np.zeros((86, 6), dtype=np.float32)
        ot = np.zeros((86, 6), dtype=np.float32)
        oa = np.zeros((86, 6), dtype=np.float32)
        al = np.zeros((len(adv_clean), 6), dtype=np.float32)
        at = np.zeros((len(adv_clean), 6), dtype=np.float32)
        aa = np.zeros((len(adv_clean), 6), dtype=np.float32)
        for tr, va in sgkf.split(s_dense, y_lbl, groups):
            Xtr = np.vstack([Xm_syn, Xm_s[tr]])
            ylt = np.concatenate([y_syn_lbl, y_lbl[tr]])
            ytt = np.vstack([y_syn_tech, y_tech[tr]])
            yat = np.concatenate([y_syn_act, y_act[tr]])
            swt = np.concatenate([np.ones(len(syn_df)), np.full(len(tr), 4.0)])
            ml = make_mc().fit(Xtr, ylt, sample_weight=swt)
            ma = make_mc().fit(Xtr, yat, sample_weight=swt)
            mt = [make_bin().fit(Xtr, ytt[:, j], sample_weight=swt) for j in range(6)]
            ol[va], oa[va] = ml.predict_proba(Xm_s[va]), ma.predict_proba(Xm_s[va])
            ot[va] = np.column_stack([m.predict_proba(Xm_s[va])[:, 1] for m in mt])
            am = np.isin(adv_p, va)
            al[am], aa[am] = ml.predict_proba(Xm_adv[am]), ma.predict_proba(Xm_adv[am])
            at[am] = np.column_stack([m.predict_proba(Xm_adv[am])[:, 1] for m in mt])

        pl, pt, pa = decode_consistent(ol, ot, oa)
        apl, apt, apa = decode_consistent(al, at, aa)
        rc = eval_multitask(y_lbl, pl, y_tech, pt, y_act, pa, ol.max(1))
        ra = eval_multitask(y_lbl[adv_p], apl, y_tech[adv_p], apt, y_act[adv_p], apa, al.max(1))
        print(f"[v3 {name}] Group-CV :", {k: round(v, 2) for k, v in rc.items()}, flush=True)
        print(f"[v3 {name}] Robust-CV:", {k: round(v, 2) for k, v in ra.items()}, flush=True)
        print(f"  {name} Per-tech F1:", {TECH_LIST[j]: round(100 * f1_score(y_tech[:, j], pt[:, j]), 2) for j in range(6)}, flush=True)

        Xfull = np.vstack([Xm_syn, Xm_s])
        mlf = make_mc().fit(Xfull, yl_full, sample_weight=sw_full)
        maf = make_mc().fit(Xfull, ya_full, sample_weight=sw_full)
        mtf = [make_bin().fit(Xfull, yt_full[:, j], sample_weight=sw_full) for j in range(6)]
        tl = mlf.predict_proba(Xm_te).astype(np.float32)
        ta = maf.predict_proba(Xm_te).astype(np.float32)
        tt = np.column_stack([m.predict_proba(Xm_te)[:, 1] for m in mtf]).astype(np.float32)
        return (ol, ot, oa, al, at, aa), (tl, tt, ta)

    lgb_oof, lgb_te = run_gbdt_family(
        "LightGBM",
        lambda: lgb.LGBMClassifier(objective="multiclass", num_class=6, n_estimators=180, learning_rate=0.06, num_leaves=23, min_child_samples=25, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=8),
        lambda: lgb.LGBMClassifier(objective="binary", n_estimators=180, learning_rate=0.06, num_leaves=23, min_child_samples=25, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=8),
    )
    xgb_oof, xgb_te = run_gbdt_family(
        "XGBoost",
        lambda: xgb.XGBClassifier(objective="multi:softprob", num_class=6, n_estimators=180, learning_rate=0.06, max_depth=4, min_child_weight=10, colsample_bytree=0.8, subsample=0.8, tree_method="hist", random_state=SEED, n_jobs=8, verbosity=0),
        lambda: xgb.XGBClassifier(objective="binary:logistic", n_estimators=180, learning_rate=0.06, max_depth=4, min_child_weight=10, colsample_bytree=0.8, subsample=0.8, tree_method="hist", random_state=SEED, n_jobs=8, verbosity=0),
    )
    cat_oof, cat_te = run_gbdt_family(
        "CatBoost",
        lambda: CatBoostClassifier(loss_function="MultiClass", iterations=350, learning_rate=0.08, depth=4, l2_leaf_reg=5.0, random_seed=SEED, thread_count=8, verbose=0),
        lambda: CatBoostClassifier(loss_function="Logloss", iterations=350, learning_rate=0.08, depth=4, l2_leaf_reg=5.0, random_seed=SEED, thread_count=8, verbose=0),
    )

    np.savez(
        "preli/cache/oof_v3_trees.npz",
        lin3_l=oof_l3, lin3_t=oof_t3, lin3_a=oof_a3,
        adv_lin3_l=adv_l3, adv_lin3_t=adv_t3, adv_lin3_a=adv_a3,
        lgb3_l=lgb_oof[0], lgb3_t=lgb_oof[1], lgb3_a=lgb_oof[2],
        adv_lgb3_l=lgb_oof[3], adv_lgb3_t=lgb_oof[4], adv_lgb3_a=lgb_oof[5],
        xgb3_l=xgb_oof[0], xgb3_t=xgb_oof[1], xgb3_a=xgb_oof[2],
        adv_xgb3_l=xgb_oof[3], adv_xgb3_t=xgb_oof[4], adv_xgb3_a=xgb_oof[5],
        cat3_l=cat_oof[0], cat3_t=cat_oof[1], cat3_a=cat_oof[2],
        adv_cat3_l=cat_oof[3], adv_cat3_t=cat_oof[4], adv_cat3_a=cat_oof[5],
        adv_parent_idx=adv_p,
    )
    for tag, arrs in [("lin3", (te_l3, te_t3, te_a3)), ("lgb3", lgb_te), ("xgb3", xgb_te), ("cat3", cat_te)]:
        np.save(f"preli/cache/te_{tag}_l.npy", arrs[0])
        np.save(f"preli/cache/te_{tag}_t.npy", arrs[1])
        np.save(f"preli/cache/te_{tag}_a.npy", arrs[2])
    print("ALL v3 LINEAR + LGBM + XGB + CAT DONE!", flush=True)


if __name__ == "__main__":
    main()
