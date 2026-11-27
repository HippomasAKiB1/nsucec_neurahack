import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
CKPT_DIR = CACHE_DIR / "ckpt_v4"
OUT_DIR = CACHE_DIR / "preds_v4"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V2_DIR))

from cv_ood import tech_str_to_vec
from normalizer_v4 import format_turn_aware
from train_tfidf_gbdt_v4 import fit_predict_linear_stack, fit_gbdt_multitask, predict_gbdt


def main():
    t0 = time.time()
    sdf = pd.read_csv(CACHE_DIR / "sample_v4.csv")
    sf = np.load(CACHE_DIR / "sample_v4_feats.npy")
    sy = pd.read_csv(CACHE_DIR / "synth_64k_v4.csv")
    syf = np.load(CACHE_DIR / "synth_64k_v4_feats.npy")
    dp_p = CACHE_DIR / "domain_pseudo_v5.npz"
    dp = np.load(dp_p if dp_p.exists() else (CACHE_DIR / "domain_pseudo_v4.npz"), allow_pickle=True)
    te_f = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")
    dp_f = np.array(te_f[dp["proto_idx"]], dtype=np.float32)

    s_tx = [format_turn_aware(str(x)) for x in sdf["clean_text"].fillna("")]
    sy_tx = [format_turn_aware(str(x)) for x in sy["clean_text"].fillna("")]
    dp_tx = [format_turn_aware(str(x)) for x in dp["clean_texts"].tolist()]

    # Filter high-consensus domain prototypes cleanly for crisp tree/linear targets
    keep_dp = dp["conf"] >= 0.62
    dp_tx_f = [dp_tx[i] for i in range(len(dp_tx)) if keep_dp[i]]
    dp_f_f = dp_f[keep_dp]

    all_tx = sy_tx + s_tx * 2 + dp_tx_f
    all_f = np.vstack([syf, sf, sf, dp_f_f]).astype(np.float32)

    s_yl = sdf["label"].values.astype(int)
    s_yt = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]])
    s_ya = sdf["action"].values.astype(int)
    sy_yl = sy["label"].values.astype(int)
    sy_yt = np.vstack([tech_str_to_vec(x) for x in sy["techniques"]])
    sy_ya = sy["action"].values.astype(int)

    all_yl = np.concatenate([sy_yl, s_yl, s_yl, dp["y_l"][keep_dp].astype(int)])
    all_yt = np.vstack([sy_yt, s_yt, s_yt, dp["y_t"][keep_dp].astype(int)])
    all_ya = np.concatenate([sy_ya, s_ya, s_ya, dp["y_a"][keep_dp].astype(int)])

    print(f"Fitting Domain-Adapted Linear + LGBM/XGB/CatBoost on {len(all_tx)} rows...", flush=True)
    vec_w = TfidfVectorizer(ngram_range=(1, 2), max_features=85000, sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w+\b|[<>\[\]\|]+")
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=55000, sublinear_tf=True, min_df=3)
    X_tw = hstack([vec_w.fit_transform(all_tx), vec_c.fit_transform(all_tx)]).tocsr()

    svd = TruncatedSVD(n_components=128, random_state=42)
    Z_tr = svd.fit_transform(X_tw).astype(np.float32)
    scaler = StandardScaler()
    F_tr = scaler.fit_transform(all_f).astype(np.float32)

    X_lin = hstack([X_tw, csr_matrix(F_tr * 0.25)]).tocsr()
    X_tab = np.hstack([F_tr, Z_tr]).astype(np.float32)

    _, (full_l, full_ts, full_a) = fit_predict_linear_stack(X_lin, all_yl, all_yt, all_ya, [])
    full_gb = fit_gbdt_multitask(X_tab, all_yl, all_yt, all_ya, seed=2026)

    joblib.dump(
        {"vec_w": vec_w, "vec_c": vec_c, "svd": svd, "scaler": scaler, "lin": (full_l, full_ts, full_a), "gb": full_gb},
        CKPT_DIR / "da_tab_v4.joblib",
        compress=3,
    )

    n_te = len(te_f)
    lin_l, lin_t, lin_a = np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32)
    lgb_l, xgb_l, cb_l = np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32)
    gb_t, gb_a = np.zeros((n_te, 6), np.float32), np.zeros((n_te, 6), np.float32)

    off = 0
    for df_c in pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=50000):
        m = len(df_c)
        txts = [format_turn_aware(str(x)) for x in df_c["clean_text"].fillna("")]
        X_c = hstack([vec_w.transform(txts), vec_c.transform(txts)]).tocsr()
        F_c = scaler.transform(te_f[off:off + m]).astype(np.float32)
        Z_c = svd.transform(X_c).astype(np.float32)
        X_lc = hstack([X_c, csr_matrix(F_c * 0.25)]).tocsr()
        X_tc = np.hstack([F_c, Z_c]).astype(np.float32)

        lin_l[off:off + m] = full_l.predict_proba(X_lc).astype(np.float32)
        lin_t[off:off + m] = np.column_stack([c.predict_proba(X_lc)[:, 1] for c in full_ts]).astype(np.float32)
        pa_sub = full_a.predict_proba(X_lc).astype(np.float32)
        for ci, cls_id in enumerate(full_a.classes_):
            lin_a[off:off + m, int(cls_id)] = pa_sub[:, ci]

        pl1, px1, pcb1, _, pgt1, pga1 = predict_gbdt(full_gb, X_tc)
        lgb_l[off:off + m], xgb_l[off:off + m], cb_l[off:off + m] = pl1, px1, pcb1
        gb_t[off:off + m], gb_a[off:off + m] = pgt1, pga1
        off += m

    np.savez_compressed(
        OUT_DIR / "test_da_tab_v4.npz",
        lin_l=lin_l, lin_t=lin_t, lin_a=lin_a,
        lgb_l=lgb_l, xgb_l=xgb_l, cb_l=cb_l, gb_t=gb_t, gb_a=gb_a,
    )
    print(f"Saved test_da_tab_v4.npz in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
