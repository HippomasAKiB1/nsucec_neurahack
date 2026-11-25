import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

SEED = 42
os.makedirs("experiments/cache", exist_ok=True)

s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

s_den = s_df["denoised"].fillna("").tolist()
t_den = t_df["denoised"].fillna("").tolist()
s_gls = s_df["glossed"].fillna("").tolist()
t_gls = t_df["glossed"].fillna("").tolist()
s_en = s_df["en_norm"].fillna("").tolist()
t_en = t_df["en_norm"].fillna("").tolist()
s_dual = [f"{a} | {b}" if a.lower() != b.lower() else a for a, b in zip(s_gls, s_en)]
t_dual = [f"{a} | {b}" if a.lower() != b.lower() else a for a, b in zip(t_gls, t_en)]

def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

def eval_all(X, name, clf_list=None):
    if clf_list is None:
        clf_list = [
            ("LogReg(C=10)", lambda: LogisticRegression(C=10.0, max_iter=1000)),
            ("Ridge(a=0.5)", lambda: RidgeClassifier(alpha=0.5)),
            ("LinearSVC(C=1.5)", lambda: LinearSVC(C=1.5)),
        ]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for cname, fn in clf_list:
        oof10 = np.zeros_like(y)
        for tr, va in skf.split(X, y):
            m = fn(); m.fit(X[tr], y[tr]); oof10[va] = m.predict(X[va])
        oofloo = np.zeros_like(y)
        for tr, va in LeaveOneOut().split(X):
            m = fn(); m.fit(X[tr], y[tr]); oofloo[va] = m.predict(X[va])
        print(f"{name:22s} | {cname:16s} | 10F: {accuracy_score(y, oof10):.4f}/{f1_score(y, oof10, average='macro'):.4f} | LOO: {accuracy_score(y, oofloo):.4f}/{f1_score(y, oofloo, average='macro'):.4f}")

# 1. TF-IDF on en_norm
vec = FeatureUnion([
    ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("c", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True)),
])
X_tf = vec.fit_transform(s_en + t_en)
eval_all(X_tf[:60], "TFIDF(en_norm)", [
    ("LogReg(C=5)", lambda: LogisticRegression(C=5.0, max_iter=1000)),
    ("LinearSVC(C=1)", lambda: LinearSVC(C=1.0)),
    ("MNB(a=0.3)", lambda: MultinomialNB(alpha=0.3)),
])

@torch.no_grad()
def encode_save(model_name, key, pooling="cls", prefix="", trust_remote_code=False, local_only=False, cache_dir=r"D:\Datathon\hf_cache"):
    path = f"experiments/cache/{key}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["s_en"], d["t_en"], d["s_dual"], d["t_dual"]
    t0 = time.time()
    tk = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only, trust_remote_code=trust_remote_code, cache_dir=cache_dir)
    md = AutoModel.from_pretrained(model_name, local_files_only=local_only, trust_remote_code=trust_remote_code, cache_dir=cache_dir, dtype=torch.float16).to("cuda").eval()
    all_t = [prefix + x for x in (s_en + t_en + s_dual + t_dual)]
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
    se, te, sd, td = E[:60], E[60:360], E[360:420], E[420:]
    np.savez(path, s_en=se, t_en=te, s_dual=sd, t_dual=td)
    print(f"Cached {key} in {time.time()-t0:.1f}s", flush=True)
    return se, te, sd, td

se_bge, te_bge, sd_bge, td_bge = encode_save("BAAI/bge-m3", "bge_m3_en", pooling="cls", prefix="")
se_e5, te_e5, sd_e5, td_e5 = encode_save("intfloat/multilingual-e5-base", "e5_base_en", pooling="mean", prefix="query: ")
se_nom, te_nom, sd_nom, td_nom = encode_save("nomic-ai/nomic-embed-text-v1.5", "nomic_en", pooling="mean", prefix="classification: ", trust_remote_code=True)

eval_all(se_bge, "BGE-M3(en_norm)")
eval_all(sd_bge, "BGE-M3(dual)")
eval_all(se_e5, "E5-Base(en_norm)")
eval_all(se_nom, "Nomic-v1.5(en_norm)")
eval_all(l2(np.hstack([se_bge, se_e5, se_nom])), "BGE+E5+Nomic(en)")
eval_all(l2(np.hstack([sd_bge, sd_e5, se_nom])), "BGE_dual+E5_d+Nomic")

# Inspect remaining LOO errors of best combination
best_X = l2(np.hstack([se_bge, se_e5, se_nom]))
oof = np.zeros_like(y)
for tr, va in LeaveOneOut().split(best_X):
    m = RidgeClassifier(alpha=0.5).fit(best_X[tr], y[tr])
    oof[va] = m.predict(best_X[va])
print(f"\nErrors of BGE+E5+Nomic(en) Ridge(a=0.5) [LOO acc={accuracy_score(y, oof):.4f}]:")
for i in range(60):
    if oof[i] != y[i]:
        print(f"  id={s_df.id.iloc[i]:2d} true={y[i]} pred={oof[i]} | {s_df.raw.iloc[i]} ==> {s_en[i]}")

# 4. Zero-shot NLI with MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (stored on D:\Datathon\hf_cache)
nli_cache = r"D:\Datathon\hf_cache"
nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
t0 = time.time()
tok = AutoTokenizer.from_pretrained(nli_name, cache_dir=nli_cache)
mdl = AutoModelForSequenceClassification.from_pretrained(nli_name, cache_dir=nli_cache, dtype=torch.float16).to("cuda").eval()
ent_id = [k for k, v in mdl.config.id2label.items() if "entail" in v.lower()][0]
con_id = [k for k, v in mdl.config.id2label.items() if "contra" in v.lower()][0]

# Multi-prompt hypotheses (3 prompt sets x 6 classes)
HYP_SETS = [
    [
        "This passenger feedback is about bus delay, late arrival, traffic jam, long waiting time, breakdown, or schedule failure.",
        "This passenger feedback is about bus fare, ticket price, overcharging, student discount refusal, missing change, or payment deduction error.",
        "This passenger feedback is about safety or security danger, reckless driving, speeding, accident, harassment, pickpocketing, or snatching.",
        "This passenger feedback is about dirtiness, garbage, bad smell, vomit, filthy toilets, stains, or pests like cockroaches or lizards.",
        "This passenger feedback is about rude staff, poor service quality, broken air conditioning, overcrowding, skipping stops, or men taking women's reserved seats.",
        "This passenger feedback is a route or timetable question, lost item inquiry, app or website issue or praise, missing stop sign, new route request, or form test.",
    ],
    [
        "Category: bus late, delayed, traffic, waiting time, missed bus, or broken down.",
        "Category: ticket fare, overcharge, student half fare, cash change, or bkash/card payment issue.",
        "Category: unsafe travel, dangerous driving, speeding, crash, harassment, drunk driver, or theft.",
        "Category: unhygienic, dirty seats, trash, spit, stink, vomit, cockroach, or unclean bus/stop.",
        "Category: staff misbehavior, conductor/helper/driver attitude, AC off, crowded standing, stop skipping, or reserved seat occupied.",
        "Category: general inquiry, lost bag/wallet, suggestion, route request, praise/compliment, website/app feedback, or test message.",
    ],
]

@torch.no_grad()
def score_nli(texts, hyps):
    pa, pb = [], []
    for x in texts:
        for h in hyps:
            pa.append(x)
            pb.append(h)
    sc = []
    for i in range(0, len(pa), 64):
        enc = tok(pa[i:i+64], pb[i:i+64], padding=True, truncation=True, max_length=160, return_tensors="pt").to("cuda")
        logits = mdl(**enc).logits.float()
        sc.append((logits[:, ent_id] - logits[:, con_id]).cpu().numpy())
    return np.concatenate(sc).reshape(len(texts), 6)

all_scores_s = np.zeros((60, 6))
all_scores_t = np.zeros((len(t_en), 6))
for view_name, (vs, vt) in [("glossed", (s_gls, t_gls)), ("en_norm", (s_en, t_en))]:
    for p_idx, hyps in enumerate(HYP_SETS):
        sc_s = score_nli(vs, hyps)
        sc_t = score_nli(vt, hyps)
        mu = np.vstack([sc_s, sc_t]).mean(axis=0, keepdims=True)
        sd = np.vstack([sc_s, sc_t]).std(axis=0, keepdims=True).clip(min=1e-6)
        sc_s_cal = (sc_s - mu) / sd
        sc_t_cal = (sc_t - mu) / sd
        print(f"mDeBERTa ZS [{view_name:7s}] P{p_idx} raw={accuracy_score(y, sc_s.argmax(1)):.4f} z-cal={accuracy_score(y, sc_s_cal.argmax(1)):.4f}/{f1_score(y, sc_s_cal.argmax(1), average='macro'):.4f}")
        all_scores_s += sc_s_cal / (len(HYP_SETS) * 2)
        all_scores_t += sc_t_cal / (len(HYP_SETS) * 2)

np.savez("experiments/cache/mdeberta_zs.npz", s_scores=all_scores_s, t_scores=all_scores_t)
print(f"mDeBERTa ZeroShot Ensemble z-cal | acc={accuracy_score(y, all_scores_s.argmax(1)):.4f} f1={f1_score(y, all_scores_s.argmax(1), average='macro'):.4f} ({time.time()-t0:.1f}s)")
for i in range(60):
    if all_scores_s.argmax(1)[i] != y[i]:
        print(f"  [NLI err] id={s_df.id.iloc[i]:2d} true={y[i]} pred={all_scores_s.argmax(1)[i]} | {s_en[i]}")
del mdl; gc.collect(); torch.cuda.empty_cache()
