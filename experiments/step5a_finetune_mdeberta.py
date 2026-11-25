import gc
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from text_clean import canonical_key

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CACHE_DIR = r"D:\Datathon\hf_cache"
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

s_texts = [f"{a} — {b}" if a.lower() != b.lower() else b for a, b in zip(s_df["denoised"], s_df["en_norm"])]
t_texts = [f"{a} — {b}" if a.lower() != b.lower() else b for a, b in zip(t_df["denoised"], t_df["en_norm"])]

# Deduplicate test texts by canonical key so inference runs 2x faster
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
uniq_keys, key_rep = [], {}
for i, k in enumerate(t_keys):
    if k not in key_rep:
        key_rep[k] = i
        uniq_keys.append(k)
u_texts = [t_texts[key_rep[k]] for k in uniq_keys]

tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)

@torch.no_grad()
def predict_logits(mdl, texts):
    mdl.eval()
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], padding=True, truncation=True, max_length=72, return_tensors="pt").to("cuda")
        lg = mdl(**enc).logits.float()
        out.append(lg.cpu().numpy())
    return np.vstack(out)

def train_fold(tr_idx, epochs=14, lr=6e-5):
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
        num_labels=6, ignore_mismatched_sizes=True
    ).to("cuda")
    # Freeze embeddings + bottom 8 encoder layers (< 1.2 GB VRAM!)
    for name, p in mdl.named_parameters():
        if "embeddings" in name or any(f"layer.{i}." in name for i in range(8)):
            p.requires_grad = False
    trainable = [p for p in mdl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.02)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    tr_list = list(tr_idx)
    for ep in range(epochs):
        mdl.train()
        random.shuffle(tr_list)
        for i in range(0, len(tr_list), 8):
            batch_ids = tr_list[i:i+8]
            bx = [s_texts[j] if random.random() < 0.7 else s_df["en_norm"].iloc[j] for j in batch_ids]
            by = torch.tensor([y[j] for j in batch_ids], dtype=torch.long, device="cuda")
            enc = tok(bx, padding=True, truncation=True, max_length=72, return_tensors="pt").to("cuda")
            opt.zero_grad()
            loss = loss_fn(mdl(**enc).logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
    return mdl

t0 = time.time()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
oof_logits = np.zeros((60, 6))
u_probs = np.zeros((len(u_texts), 6))

for fold, (tr, va) in enumerate(skf.split(np.zeros(60), y)):
    ft0 = time.time()
    mdl = train_fold(tr, epochs=16, lr=6e-5)
    va_lg = predict_logits(mdl, [s_texts[i] for i in va])
    u_lg = predict_logits(mdl, u_texts)
    oof_logits[va] = va_lg
    u_probs += softmax(u_lg, axis=1) / 10.0
    print(f"Fold {fold}: val_acc={accuracy_score(y[va], va_lg.argmax(1)):.4f} ({time.time()-ft0:.1f}s) maxVRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    del mdl; gc.collect(); torch.cuda.empty_cache()

acc = accuracy_score(y, oof_logits.argmax(1))
f1 = f1_score(y, oof_logits.argmax(1), average="macro")
print(f"\n>>> Fine-Tuned mDeBERTa-v3 10-Fold OOF: Acc={acc:.4f}, Macro-F1={f1:.4f} (total {time.time()-t0:.1f}s)")
for i in range(60):
    if oof_logits.argmax(1)[i] != y[i]:
        print(f"  [ERR] id={s_df.id.iloc[i]:2d} true={y[i]} pred={oof_logits.argmax(1)[i]} | {s_df.raw.iloc[i]}")

k2p = {k: u_probs[j] for j, k in enumerate(uniq_keys)}
t_probs = np.array([k2p[k] for k in t_keys])
np.savez("experiments/cache/mdeberta_ft6_oof.npz", oof_logits=oof_logits, test_probs=t_probs)
