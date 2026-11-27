import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

CACHE_DIR = r"D:\Datathon\hf_cache"
df = pd.read_csv("Main Preli Round/dataset/test.csv", nrows=4096)
texts = df["text"].fillna("").astype(str).tolist()
# Sort by length for minimal padding overhead
order = np.argsort([len(x) for x in texts])
sorted_texts = [texts[i] for i in order]

for mname, max_len, bs in [
    ("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", 128, 128),
    ("intfloat/multilingual-e5-base", 128, 128),
]:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR, local_files_only=True)
    mdl = AutoModel.from_pretrained(mname, cache_dir=CACHE_DIR, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    # warmup
    with torch.no_grad():
        enc = tok(sorted_texts[:64], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
        _ = mdl(**enc)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        for i in range(0, len(sorted_texts), bs):
            enc = tok(sorted_texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
            _ = mdl(**enc)
    torch.cuda.synchronize()
    dt = time.time() - t1
    rps = len(sorted_texts) / dt
    est_500k_min = (500000 / rps) / 60.0
    print(f"{mname:42s} | {rps:6.1f} rows/s | est 500k time = {est_500k_min:5.1f} min | maxVRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    del mdl
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
