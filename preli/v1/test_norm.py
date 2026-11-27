import time
import re
import pandas as pd
from normalizer import clean_for_model, extract_structural_features

s = pd.read_csv("Main Preli Round/dataset/sample.csv")
print("=== SAMPLE NORMALIZATION CHECK ===")
for idx in [21, 26, 27, 32, 34, 37, 52, 57, 80, 81, 84]:
    r = s.iloc[idx]
    cl = clean_for_model(r.text)
    ft = extract_structural_features(r.text, cl)
    print(f"[{r.id} lbl={r.label} act={r.action}] {r.text[:60]}  ==>  {cl[:65]} | warn={int(ft['has_scam_warning'])} noise={int(ft['is_noise_only'])}")

t0 = time.time()
df = pd.read_csv("Main Preli Round/dataset/test.csv", nrows=50000)
uniq_clean = set()
noise_cnt = 0
warn_cnt = 0
for raw in df["text"].fillna("").astype(str):
    cl = clean_for_model(raw)
    ft = extract_structural_features(raw, cl)
    noise_cnt += int(ft["is_noise_only"])
    warn_cnt += int(ft["has_scam_warning"])
    sk = re.sub(r"\d+", "<N>", cl.lower())
    uniq_clean.add(sk)

dt = time.time() - t0
print(f"\n50k clean+featurize in {dt:.1f}s (est 500k = {dt*10:.1f}s) | uniq_clean_50k={len(uniq_clean)} | noise_only={noise_cnt} | scam_warn={warn_cnt}")
