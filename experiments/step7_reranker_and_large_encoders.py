import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from text_clean import canonical_key

CACHE_DIR = r"D:\Datathon\hf_cache"
s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

s_en = s_df["en_norm"].fillna("").tolist()
t_en = t_df["en_norm"].fillna("").tolist()
s_den = s_df["denoised"].fillna("").tolist()
t_den = t_df["denoised"].fillna("").tolist()
s_gls = s_df["glossed"].fillna("").tolist()
t_gls = t_df["glossed"].fillna("").tolist()

t_keys = [canonical_key(x) for x in t_en]
uniq_keys, key_rep = [], {}
for i, k in enumerate(t_keys):
    if k not in key_rep:
        key_rep[k] = i
        uniq_keys.append(k)
u_en = [t_en[key_rep[k]] for k in uniq_keys]
u_den = [t_den[key_rep[k]] for k in uniq_keys]
print(f"Sample={len(s_en)}, Test={len(t_en)}, Canonical Test={len(u_en)}", flush=True)

@torch.no_grad()
def encode_model(model_name, key, texts_s, texts_t, pooling="cls", prefix="", trust_remote_code=False):
    t0 = time.time()
    tk = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=trust_remote_code)
    md = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR, trust_remote_code=trust_remote_code, dtype=torch.float16).to("cuda").eval()
    all_t = [prefix + x for x in (texts_s + texts_t)]
    embs = []
    for i in range(0, len(all_t), 32):
        enc = tk(all_t[i:i+32], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        h = md(**enc).last_hidden_state
        if pooling == "cls":
            e = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).type_as(h)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        e = torch.nn.functional.normalize(e.float(), p=2, dim=-1)
        embs.append(e.cpu().numpy())
    del md; gc.collect(); torch.cuda.empty_cache()
    E = np.vstack(embs)
    np.savez(f"experiments/cache/{key}.npz", s=E[:60], t=E[60:])
    print(f"Encoded {key} in {time.time()-t0:.1f}s", flush=True)
    return E[:60], E[60:]

# 1. Re-encode cleanly normalized texts across BGE-M3, E5-Base, Nomic-v1.5, plus BGE-Large-EN-v1.5
s_bge_en, t_bge_en = encode_model("BAAI/bge-m3", "v2_bge_m3_en", s_en, t_en, pooling="cls")
s_bge_gl, t_bge_gl = encode_model("BAAI/bge-m3", "v2_bge_m3_gls", s_gls, t_gls, pooling="cls")
s_e5_en, t_e5_en   = encode_model("intfloat/multilingual-e5-base", "v2_e5_base_en", s_en, t_en, pooling="mean", prefix="query: ")
s_nom_en, t_nom_en = encode_model("nomic-ai/nomic-embed-text-v1.5", "v2_nomic_en", s_en, t_en, pooling="mean", prefix="classification: ", trust_remote_code=True)
s_bge_lg, t_bge_lg = encode_model("BAAI/bge-large-en-v1.5", "v2_bge_large_en", s_en, t_en, pooling="cls", prefix="Represent this sentence for searching relevant passages: ")

for name, X in [("BGE-M3(en)", s_bge_en), ("BGE-M3(gls)", s_bge_gl), ("E5-Base(en)", s_e5_en), ("Nomic(en)", s_nom_en), ("BGE-Large-EN", s_bge_lg)]:
    for cname, fn in [("Ridge(0.5)", lambda: RidgeClassifier(0.5)), ("SVC(1.5)", lambda: LinearSVC(C=1.5))]:
        preds = np.zeros_like(y)
        for tr, va in LeaveOneOut().split(X):
            m = fn(); m.fit(X[tr], y[tr]); preds[va] = m.predict(X[va])
        print(f"{name:14s} | {cname:10s} | LOO={accuracy_score(y, preds):.4f}/{f1_score(y, preds, average='macro'):.4f}", flush=True)

# 2. BAAI/bge-reranker-v2-m3 Cross-Encoder Pairwise Kernel (60x60 LOO + 183x60 test-to-support)
t0 = time.time()
rr_name = "BAAI/bge-reranker-v2-m3"
rr_tok = AutoTokenizer.from_pretrained(rr_name, cache_dir=CACHE_DIR)
rr_mdl = AutoModelForSequenceClassification.from_pretrained(rr_name, cache_dir=CACHE_DIR, dtype=torch.float16).to("cuda").eval()
print(f"Loaded {rr_name} in {time.time()-t0:.1f}s", flush=True)

@torch.no_grad()
def cross_sim_matrix(queries, supports, bs=64):
    pa, pb = [], []
    for q in queries:
        for sup in supports:
            pa.append(q)
            pb.append(sup)
    scores = []
    for i in range(0, len(pa), bs):
        enc = rr_tok(pa[i:i+bs], pb[i:i+bs], padding=True, truncation=True, max_length=72, return_tensors="pt").to("cuda")
        lg = rr_mdl(**enc).logits.squeeze(-1).float()
        scores.append(lg.cpu().numpy())
    return np.concatenate(scores).reshape(len(queries), len(supports))

t1 = time.time()
S_mat = cross_sim_matrix(s_en, s_en, bs=64)
U_mat = cross_sim_matrix(u_en, s_en, bs=64)
print(f"Cross-Encoder matrices ({S_mat.shape}, {U_mat.shape}) computed in {time.time()-t1:.1f}s", flush=True)
del rr_mdl; gc.collect(); torch.cuda.empty_cache()

# Evaluate LOO Kernel kNN across temperatures T
for temp in [0.5, 1.0, 1.5, 2.0, 3.0]:
    oof_p = np.zeros((60, 6))
    for i in range(60):
        # Leave out self!
        mask = np.ones(60, dtype=bool)
        mask[i] = False
        w = softmax(S_mat[i, mask] / temp)
        for c in range(6):
            oof_p[i, c] = w[y[mask] == c].sum()
    print(f"BGE-Reranker-v2-M3 Kernel (T={temp:.1f}) | LOO={accuracy_score(y, oof_p.argmax(1)):.4f}/{f1_score(y, oof_p.argmax(1), average='macro'):.4f}", flush=True)

# Map U_mat back to all 300 test rows
U_full = np.array([U_mat[uniq_keys.index(k)] for k in t_keys])
np.savez("experiments/cache/v2_reranker_mats.npz", S_mat=S_mat, U_mat=U_mat, U_full=U_full, uniq_keys=np.array(uniq_keys))
