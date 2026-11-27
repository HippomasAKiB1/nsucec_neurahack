import gc
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

CACHE_DIR = r"D:\Datathon\hf_cache"
df = pd.read_csv("Main Preli Round/dataset/test.csv", nrows=4096)
texts = df["text"].fillna("").astype(str).tolist()
order = np.argsort([len(x) for x in texts])
sorted_texts = [texts[i] for i in order]

for mname, max_len, bs in [
    ("MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli", 112, 256),
    ("intfloat/multilingual-e5-small", 112, 256),
    ("BAAI/bge-small-en-v1.5", 112, 256),
    ("intfloat/multilingual-e5-base", 112, 128),
]:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mname, cache_dir=CACHE_DIR)
    mdl = AutoModel.from_pretrained(mname, cache_dir=CACHE_DIR, dtype=torch.float16).to("cuda").eval()
    with torch.no_grad():
        enc = tok(sorted_texts[:128], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
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
    est_min = (500000 / rps) / 60.0
    print(f"{mname:48s} | {rps:6.1f} rows/s | 500k={est_min:5.1f} min | VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
