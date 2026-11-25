import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

os.makedirs("experiments/cache", exist_ok=True)
s = pd.read_csv("mock/dataset/sample.csv")
y = s["label"].values
texts = s["text"].tolist()

# 1. Batched Zero-Shot NLI with mMiniLM
t0 = time.time()
nli_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
tok = AutoTokenizer.from_pretrained(nli_name, local_files_only=True)
mdl = AutoModelForSequenceClassification.from_pretrained(nli_name, local_files_only=True, dtype=torch.float16).to("cuda").eval()

label_hyps = [
    "This message is about bus delay, late arrival, traffic jam, long waiting time, breakdown, or schedule.",
    "This message is about bus fare, ticket price, overcharging, student discount, change, or payment issue.",
    "This message is about passenger safety, reckless driving, speeding, accident, harassment, pickpocketing, or crime.",
    "This message is about dirt, garbage, cleanliness, bad smell, vomit, toilets, or pests like cockroaches.",
    "This message is about rude staff behavior, poor service, AC off, overcrowding, skipping stops, or women's seat.",
    "This message is a general question, route inquiry, lost item, app or website feedback, praise, missing sign, or form test.",
]
entail_id = [k for k, v in mdl.config.id2label.items() if "entail" in v.lower()][0]
contra_id = [k for k, v in mdl.config.id2label.items() if "contra" in v.lower()][0]

pairs_a = []
pairs_b = []
for txt in texts:
    for h in label_hyps:
        pairs_a.append(txt)
        pairs_b.append(h)

scores_all = []
bs = 64
with torch.no_grad():
    for i in range(0, len(pairs_a), bs):
        enc = tok(pairs_a[i:i+bs], pairs_b[i:i+bs], padding=True, truncation=True, max_length=160, return_tensors="pt").to("cuda")
        logits = mdl(**enc).logits.float()
        sc = logits[:, entail_id] - logits[:, contra_id]
        scores_all.append(sc.cpu().numpy())
scores = np.concatenate(scores_all).reshape(len(texts), 6)
preds_nli = scores.argmax(1)
print(f"mMiniLM NLI clean acc={accuracy_score(y, preds_nli):.4f}, macro_f1={f1_score(y, preds_nli, average='macro'):.4f} ({time.time()-t0:.1f}s)")
del mdl; gc.collect(); torch.cuda.empty_cache()

# 2. Compute & cache BGE-M3 embeddings for sample + test
from text_clean import strip_wrappers, build_swap_fixer
t_df = pd.read_csv("mock/dataset/test.csv")
all_clean_vocab = [strip_wrappers(x) for x in texts] + [strip_wrappers(x) for x in t_df["text"]]
fix_swaps = build_swap_fixer(all_clean_vocab)
t_denoised = [fix_swaps(strip_wrappers(x)) for x in t_df["text"]]

@torch.no_grad()
def encode_save(model_name, key, pooling="cls", prefix=""):
    path = f"experiments/cache/{key}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["s"], d["t_raw"], d["t_den"]
    t1 = time.time()
    tk = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    md = AutoModel.from_pretrained(model_name, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    all_t = [prefix + x for x in (texts + t_df["text"].tolist() + t_denoised)]
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
    Es, Etr, Etd = E[:60], E[60:360], E[360:]
    np.savez(path, s=Es, t_raw=Etr, t_den=Etd)
    print(f"Cached {key} in {time.time()-t1:.1f}s")
    return Es, Etr, Etd

Es_bge, _, Etd_bge = encode_save("BAAI/bge-m3", "bge_m3", pooling="cls", prefix="")
Es_e5b, _, Etd_e5b = encode_save("intfloat/multilingual-e5-base", "e5_base", pooling="mean", prefix="query: ")

# Inspect LOO errors of BGE-M3 + LinearSVC
loo = LeaveOneOut()
oof = np.zeros_like(y)
for tr, va in loo.split(Es_bge):
    clf = LinearSVC(C=2.0)
    clf.fit(Es_bge[tr], y[tr])
    oof[va] = clf.predict(Es_bge[va])

print(f"\nBGE-M3 LOO acc={accuracy_score(y, oof):.4f}, f1={f1_score(y, oof, average='macro'):.4f}")
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y, oof))
print("\nMisclassified sample examples (BGE-M3 LOO):")
for i in range(60):
    if oof[i] != y[i]:
        print(f"  id={s.id.iloc[i]:2d} true={y[i]} pred={oof[i]} | {texts[i]}")
print("\nMisclassified sample examples (mMiniLM ZeroShot NLI):")
for i in range(60):
    if preds_nli[i] != y[i]:
        print(f"  id={s.id.iloc[i]:2d} true={y[i]} pred={preds_nli[i]} | {texts[i]}")
