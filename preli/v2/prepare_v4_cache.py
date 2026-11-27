import hashlib
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalizer_v4 import clean_for_model, format_turn_aware
from rules_v4 import extract_rule_prediction
from synth_generator_v4 import generate_dataset_v4
from cv_ood import assign_sample_scenario_family

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _canonical_skel(clean_t: str) -> str:
    s = re.sub(r"https?://\S+", " URL ", clean_t.lower())
    s = re.sub(r"(?:\+?880[\s-]?)?01\d[\dxX]{5,9}", " PHN ", s)
    s = re.sub(r"\d+", " N ", s)
    s = re.sub(r"[^a-z\u0980-\u09FF]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()[:120]


def _worker_chunk(texts):
    cleans, feats, lbls, tvecs, acts, skel_hashes = [], [], [], [], [], []
    for raw in texts:
        c = clean_for_model(raw)
        l, tv, a, f = extract_rule_prediction(raw)
        sk = hashlib.blake2b(_canonical_skel(c).encode("utf-8"), digest_size=8).digest()
        cleans.append(c)
        feats.append(f)
        lbls.append(l)
        tvecs.append(tv)
        acts.append(a)
        skel_hashes.append(np.frombuffer(sk, dtype=np.uint64)[0])
    return (
        cleans,
        np.vstack(feats).astype(np.float32),
        np.array(lbls, dtype=np.uint8),
        np.vstack(tvecs).astype(np.uint8),
        np.array(acts, dtype=np.uint8),
        np.array(skel_hashes, dtype=np.uint64),
    )


def parallel_featurize(texts, chunk_size=5000, workers=10):
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    all_c, all_f, all_l, all_t, all_a, all_s = [], [], [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c, f, l, tv, a, s in ex.map(_worker_chunk, chunks):
            all_c.extend(c)
            all_f.append(f)
            all_l.append(l)
            all_t.append(tv)
            all_a.append(a)
            all_s.append(s)
    return (
        all_c,
        np.vstack(all_f),
        np.concatenate(all_l),
        np.vstack(all_t),
        np.concatenate(all_a),
        np.concatenate(all_s),
    )


def main():
    t0 = time.time()
    # 1. Sample.csv (86 rows)
    sdf = pd.read_csv(ROOT / "Main Preli Round/dataset/sample.csv")
    sc, sf, sl, st, sa, ss = parallel_featurize(sdf["text"].tolist(), chunk_size=30, workers=2)
    sdf["clean_text"] = sc
    sdf["scenario_family"] = [assign_sample_scenario_family(sc[i], int(sdf["label"].iloc[i])) for i in range(len(sdf))]
    sdf.to_csv(CACHE_DIR / "sample_v4.csv", index=False)
    np.save(CACHE_DIR / "sample_v4_feats.npy", sf)
    np.savez_compressed(CACHE_DIR / "sample_v4_rule_pred.npz", lbl=sl, tvec=st, act=sa, skel=ss)
    print(f"Cached sample_v4 ({len(sdf)} rows, {sf.shape[1]} feats) in {time.time()-t0:.1f}s")

    # 2. Synth 72k v5
    t1 = time.time()
    sy = generate_dataset_v4(n_samples=72000, seed=4242, corrupt_prob=0.68)
    syc, syf, syl, syt, sya, sysk = parallel_featurize(sy["text"].tolist(), chunk_size=4000, workers=10)
    sy["clean_text"] = syc
    sy.to_csv(CACHE_DIR / "synth_64k_v4.csv", index=False)
    np.save(CACHE_DIR / "synth_64k_v4_feats.npy", syf)
    np.savez_compressed(CACHE_DIR / "synth_64k_v4_rule_pred.npz", lbl=syl, tvec=syt, act=sya, skel=sysk)
    print(f"Cached synth_64k_v4 ({len(sy)} rows, rule agree={(syl==sy['label'].values).mean():.4f}) in {time.time()-t1:.1f}s")

    # 3. Test 500k
    t2 = time.time()
    te = pd.read_csv(ROOT / "Main Preli Round/dataset/test.csv")
    tec, tef, tel, tet, tea, tesk = parallel_featurize(te["text"].fillna("").astype(str).tolist(), chunk_size=10000, workers=10)
    pd.DataFrame({"id": te["id"].values, "clean_text": tec}).to_csv(CACHE_DIR / "test_v4_clean.csv", index=False)
    np.save(CACHE_DIR / "test_v4_feats.npy", tef)
    np.savez_compressed(CACHE_DIR / "test_v4_rule_pred.npz", lbl=tel, tvec=tet, act=tea, skel=tesk)

    # Build canonical cluster statistics for domain-adaptation prototype selection (training-only!)
    vals, first_idx, counts = np.unique(tesk, return_index=True, return_counts=True)
    order = np.argsort(-counts)
    vals_sorted = vals[order]
    counts_sorted = counts[order]
    first_sorted = first_idx[order]
    np.savez_compressed(
        CACHE_DIR / "test_skel_clusters.npz",
        uniq_hash=vals_sorted,
        first_idx=first_sorted,
        counts=counts_sorted,
    )
    n_ge10 = int((counts >= 10).sum())
    cov_ge10 = int(counts[counts >= 10].sum())
    print(
        f"Cached test_v4 (500k rows) in {time.time()-t2:.1f}s | "
        f"unique_skels={len(vals)}, clusters>=10: {n_ge10} covering {cov_ge10} ({100*cov_ge10/len(te):.1f}%)"
    )


if __name__ == "__main__":
    main()
