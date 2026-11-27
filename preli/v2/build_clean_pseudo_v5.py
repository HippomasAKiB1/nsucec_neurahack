import sys
from pathlib import Path
import numpy as np
import pandas as pd

V2_DIR = Path(__file__).resolve().parent
CACHE_DIR = V2_DIR / "cache"
OUT_DIR = CACHE_DIR / "preds_v4"
sys.path.insert(0, str(V2_DIR))

from cv_ood import decode_consistent


def main():
    dp = np.load(CACHE_DIR / "domain_pseudo_v4.npz", allow_pickle=True)
    ru = np.load(CACHE_DIR / "test_v4_rule_pred.npz")
    ba = np.load(OUT_DIR / "test_baseline_v4.npz")

    idx = dp["proto_idx"]
    # Align refreshed URL-masked test_v4_clean texts on the 45k prototypes
    mask = np.zeros(len(ru["lbl"]), dtype=bool)
    mask[idx] = True
    clean_refreshed = []
    off = 0
    for df_c in pd.read_csv(CACHE_DIR / "test_v4_clean.csv", chunksize=50000):
        m = len(df_c)
        loc = mask[off:off + m]
        if loc.any():
            clean_refreshed.extend(df_c["clean_text"].fillna("").values[loc].tolist())
        off += m

    rl = ru["lbl"][idx].astype(int)
    rt = ru["tvec"][idx].astype(np.float32)
    ra = ru["act"][idx].astype(int)

    pb_l = 0.28 * ba["lin_l"][idx] + 0.24 * ba["lgb_l"][idx] + 0.20 * ba["xgb_l"][idx] + 0.16 * ba["cb_l"][idx] + 0.12 * (
        np.eye(6, dtype=np.float32)[rl] * 0.85 + 0.03
    )
    pb_l /= pb_l.sum(axis=1, keepdims=True)
    pb_t = 0.28 * ba["lin_t"][idx] + 0.48 * ba["gb_t"][idx] + 0.24 * (rt * 0.84 + 0.08)
    pb_a = 0.28 * ba["lin_a"][idx] + 0.48 * ba["gb_a"][idx]
    ra_oh = np.zeros((len(idx), 6), dtype=np.float32)
    pos_a = ra > 0
    ra_oh[pos_a, 1:] = 0.0375
    ra_oh[np.where(pos_a)[0], ra[pos_a]] = 0.85
    pb_a = pb_a + 0.24 * ra_oh

    pm_l, pm_t, pm_a = dp["pm_l"], dp["pm_t"], dp["pm_a"]
    pe_l, pe_t, pe_a = dp["pe_l"], dp["pe_t"], dp["pe_a"]

    # Consensus filtering: protect against half-fold teacher blind spots and secrecy decoys!
    b_arg = pb_l.argmax(axis=1)
    m_arg = pm_l.argmax(axis=1)
    e_arg = pe_l.argmax(axis=1)

    p_cons_l = 0.24 * pm_l + 0.26 * pe_l + 0.50 * pb_l
    p_cons_t = 0.24 * pm_t + 0.26 * pe_t + 0.50 * pb_t
    p_cons_a = 0.24 * pm_a + 0.26 * pe_a + 0.50 * pb_a

    # Keep rows with high-trust consensus across rules_v5 / URL-masked Baseline_v5 / Teachers
    cond1 = (rl == b_arg) & (pb_l.max(axis=1) >= 0.72) & ((m_arg == b_arg) | (e_arg == b_arg) | (pb_l.max(axis=1) >= 0.86))
    cond2 = (m_arg == e_arg) & (e_arg == b_arg) & (p_cons_l.max(axis=1) >= 0.78)
    keep = cond1 | cond2

    k_idx = idx[keep]
    k_clean = [clean_refreshed[i] for i in np.where(keep)[0]]
    pl, pt, pa = decode_consistent(p_cons_l[keep], p_cons_t[keep], p_cons_a[keep], tech_th=0.45)

    np.savez_compressed(
        CACHE_DIR / "domain_pseudo_v5.npz",
        proto_idx=k_idx,
        clean_texts=np.array(k_clean, dtype=object),
        p_l=p_cons_l[keep].astype(np.float32),
        p_t=p_cons_t[keep].astype(np.float32),
        p_a=p_cons_a[keep].astype(np.float32),
        y_l=pl.astype(np.int64),
        y_t=pt.astype(np.float32),
        y_a=pa.astype(np.int64),
        conf=p_cons_l[keep].max(axis=1).astype(np.float32),
    )
    print(
        f"Filtered ultra-clean domain_pseudo_v5.npz: kept {keep.sum()}/{len(idx)} ({100*keep.mean():.1f}%) "
        f"| mean_conf={p_cons_l[keep].max(axis=1).mean():.4f} "
        f"| dist={pd.Series(pl).value_counts().sort_index().to_dict()} "
        f"| act_dist={pd.Series(pa).value_counts().sort_index().to_dict()}"
    )


if __name__ == "__main__":
    main()
