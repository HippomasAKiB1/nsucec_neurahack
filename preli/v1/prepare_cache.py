import os
import time
import numpy as np
import pandas as pd
from normalizer import clean_for_model, extract_structural_features

os.makedirs("preli/cache", exist_ok=True)
FEAT_NAMES = list(extract_structural_features("a", "a").keys())
print(f"Using {len(FEAT_NAMES)} structural features: {FEAT_NAMES}", flush=True)

def process_df_rows(texts_raw):
    clean_texts = []
    feats = np.zeros((len(texts_raw), len(FEAT_NAMES)), dtype=np.float32)
    for i, raw in enumerate(texts_raw):
        cl = clean_for_model(raw)
        clean_texts.append(cl)
        d = extract_structural_features(raw, cl)
        feats[i] = [d[k] for k in FEAT_NAMES]
    return clean_texts, feats

# 1. Sample.csv
s = pd.read_csv("Main Preli Round/dataset/sample.csv")
s_cl, s_ft = process_df_rows(s["text"].fillna("").astype(str).tolist())
s["clean_text"] = s_cl
s.to_csv("preli/cache/sample_clean.csv", index=False)
np.save("preli/cache/sample_feats.npy", s_ft)
print(f"Saved sample_clean.csv ({len(s)}) and sample_feats.npy {s_ft.shape}", flush=True)

# 2. Test.csv (500k rows streamed in chunks -> low RAM!)
t0 = time.time()
out_csv = "preli/cache/test_clean.csv"
all_feats = np.memmap("preli/cache/test_feats.dat", dtype="float32", mode="w+", shape=(500000, len(FEAT_NAMES)))
first = True
offset = 0
for chunk_idx, chunk in enumerate(pd.read_csv("Main Preli Round/dataset/test.csv", chunksize=50000)):
    raws = chunk["text"].fillna("").astype(str).tolist()
    cl_list, ft = process_df_rows(raws)
    n = len(chunk)
    all_feats[offset:offset+n] = ft
    offset += n
    pd.DataFrame({"id": chunk["id"].values, "clean_text": cl_list}).to_csv(
        out_csv, mode="w" if first else "a", header=first, index=False
    )
    first = False
    print(f"  Processed {offset}/500000 rows ({time.time()-t0:.1f}s)", flush=True)

all_feats.flush()
del all_feats
# Convert memmap to compact .npy safely in slices
mm = np.memmap("preli/cache/test_feats.dat", dtype="float32", mode="r", shape=(500000, len(FEAT_NAMES)))
np.save("preli/cache/test_feats.npy", np.array(mm))
del mm
os.remove("preli/cache/test_feats.dat")
print(f"Done caching 500k clean texts and features in {time.time()-t0:.1f}s!", flush=True)
