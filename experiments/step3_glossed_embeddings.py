import gc
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from transformers import AutoTokenizer, AutoModel

from text_clean import strip_wrappers, build_swap_fixer, gloss_banglish

SEED = 42
s = pd.read_csv("mock/dataset/sample.csv")
t = pd.read_csv("mock/dataset/test.csv")
y = s["label"].values

all_clean_vocab = [strip_wrappers(x) for x in s["text"]] + [strip_wrappers(x) for x in t["text"]]
fix_swaps = build_swap_fixer(all_clean_vocab)

s_den = [fix_swaps(strip_wrappers(x)) for x in s["text"]]
t_den = [fix_swaps(strip_wrappers(x)) for x in t["text"]]
s_gls = [gloss_banglish(x) for x in s_den]
t_gls = [gloss_banglish(x) for x in t_den]
s_dual = [f"{a} | {b}" if a.lower() != b.lower() else a for a, b in zip(s_den, s_gls)]
t_dual = [f"{a} | {b}" if a.lower() != b.lower() else a for a, b in zip(t_den, t_gls)]

print("Sample Banglish -> Glossed examples:")
for i in [0, 7, 11, 25, 31, 48, 50, 55, 56, 58]:
    print(f"  [{y[i]}] {s_den[i]}  ==>  {s_gls[i]}")

@torch.no_grad()
def encode_save(model_name, key, pooling="cls", prefix=""):
    path = f"experiments/cache/{key}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["s_gls"], d["t_gls"], d["s_dual"], d["t_dual"]
    t0 = time.time()
    tk = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    md = AutoModel.from_pretrained(model_name, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    all_t = [prefix + x for x in (s_gls + t_gls + s_dual + t_dual)]
    out_embs = []
    for i in range(0, len(all_t), 32):
        enc = tk(all_t[i:i+32], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        h = md(**enc).last_hidden_state
        if pooling == "cls":
            e = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).type_as(h)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        e = torch.nn.functional.normalize(e.float(), p=2, dim=-1)
        out_embs.append(e.cpu().numpy())
    del md; gc.collect(); torch.cuda.empty_cache()
    E = np.vstack(out_embs)
    sg, tg, sd, td = E[:60], E[60:360], E[360:420], E[420:]
    np.savez(path, s_gls=sg, t_gls=tg, s_dual=sd, t_dual=td)
    print(f"Cached {key} in {time.time()-t0:.1f}s")
    return sg, tg, sd, td

sg_bge, tg_bge, sd_bge, td_bge = encode_save("BAAI/bge-m3", "bge_m3_gls", pooling="cls", prefix="")
sg_e5, tg_e5, sd_e5, td_e5 = encode_save("intfloat/multilingual-e5-base", "e5_base_gls", pooling="mean", prefix="query: ")

raw_bge = np.load("experiments/cache/bge_m3.npz")["s"]
raw_e5 = np.load("experiments/cache/e5_base.npz")["s"]

def eval_loo(X, name):
    for cname, fn in [
        ("LogReg(C=10)", lambda: LogisticRegression(C=10.0, max_iter=1000)),
        ("Ridge(a=0.5)", lambda: RidgeClassifier(alpha=0.5)),
        ("LinearSVC(C=1.5)", lambda: LinearSVC(C=1.5)),
        ("SVC-RBF(C=3)", lambda: SVC(C=3.0, kernel="rbf")),
    ]:
        preds = np.zeros_like(y)
        for tr, va in LeaveOneOut().split(X):
            m = fn(); m.fit(X[tr], y[tr]); preds[va] = m.predict(X[va])
        print(f"{name:20s} | {cname:16s} | LOO acc={accuracy_score(y, preds):.4f} f1={f1_score(y, preds, average='macro'):.4f}")

def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True)

eval_loo(raw_bge, "BGE raw")
eval_loo(sg_bge, "BGE glossed")
eval_loo(sd_bge, "BGE dual")
eval_loo(l2(np.hstack([raw_bge, sg_bge])), "BGE raw+glossed")
eval_loo(sg_e5, "E5 glossed")
eval_loo(l2(np.hstack([sg_bge, sg_e5])), "BGE+E5 glossed")
eval_loo(l2(np.hstack([raw_bge, sg_bge, raw_e5, sg_e5])), "BGE+E5 raw+gls")
