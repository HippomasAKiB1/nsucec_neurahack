import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from transformers import AutoTokenizer, AutoModel

from text_clean import canonical_key, gloss_banglish, gloss_all_en
from synthetic_aug import get_synthetic_df

SEED = 42
CACHE_DIR = r"D:\Datathon\hf_cache"

s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
sub_ref = pd.read_csv("mock/dataset/sample_submission.csv")
syn_df = get_synthetic_df()

# Strict Rule-5 check: zero test strings copied into synthetic training data
assert len(set(syn_df["text"].str.lower()) & set(t_df["raw"].str.lower())) == 0
assert len(set(syn_df["text"].str.lower()) & set(t_df["denoised"].str.lower())) == 0

y_s = s_df["label"].values
y_syn = syn_df["label"].values
syn_en = [gloss_all_en(x) for x in syn_df["text"]]
syn_gls = [gloss_banglish(x) for x in syn_df["text"]]

def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

@torch.no_grad()
def encode_syn(model_name, cache_key, texts, pooling="cls", prefix="", trust_remote_code=False):
    path = f"experiments/cache/{cache_key}.npz"
    if os.path.exists(path):
        return np.load(path)["syn"]
    t0 = time.time()
    tk = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True, trust_remote_code=trust_remote_code)
    md = AutoModel.from_pretrained(model_name, cache_dir=CACHE_DIR, local_files_only=True, trust_remote_code=trust_remote_code, dtype=torch.float16).to("cuda").eval()
    inp_t = [prefix + x for x in texts]
    embs = []
    for i in range(0, len(inp_t), 36):
        enc = tk(inp_t[i:i+36], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        h = md(**enc).last_hidden_state
        if pooling == "cls":
            e = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).type_as(h)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        embs.append(torch.nn.functional.normalize(e.float(), p=2, dim=-1).cpu().numpy())
    del md; gc.collect(); torch.cuda.empty_cache()
    E = np.vstack(embs)
    np.savez(path, syn=E)
    print(f"Encoded synthetic {cache_key} ({len(E)}) in {time.time()-t0:.1f}s", flush=True)
    return E

syn_bge_en = encode_syn("BAAI/bge-m3", "v2_syn_bge_en", syn_en, pooling="cls")
syn_bge_gl = encode_syn("BAAI/bge-m3", "v2_syn_bge_gl", syn_gls, pooling="cls")
syn_e5_en  = encode_syn("intfloat/multilingual-e5-base", "v2_syn_e5_en", syn_en, pooling="mean", prefix="query: ")
syn_nom_en = encode_syn("nomic-ai/nomic-embed-text-v1.5", "v2_syn_nom_en", syn_en, pooling="mean", prefix="classification: ", trust_remote_code=True)
syn_bge_lg = encode_syn("BAAI/bge-large-en-v1.5", "v2_syn_bge_lg", syn_en, pooling="cls", prefix="Represent this sentence for searching relevant passages: ")

bge_en = np.load("experiments/cache/v2_bge_m3_en.npz")
bge_gl = np.load("experiments/cache/v2_bge_m3_gls.npz")
e5_en  = np.load("experiments/cache/v2_e5_base_en.npz")
nom_en = np.load("experiments/cache/v2_nomic_en.npz")
bge_lg = np.load("experiments/cache/v2_bge_large_en.npz")
if os.path.exists("experiments/cache/v2_qwen15b_mp.npz"):
    qw = np.load("experiments/cache/v2_qwen15b_mp.npz")
    q_opts = [
        ("RAW",      qw["oof_logits"],   qw["t_logits"]),
        ("BAL-CAL",  qw["oof_cal"],      qw["t_cal"]),
        ("NULL-CAL", qw["oof_null_cal"], qw["t_null_cal"]),
    ]
else:
    qw = np.load("experiments/cache/qwen15b_kv.npz")
    q_opts = [
        ("QWEN15B-RAW",     qw["s_logits"], qw["t_logits"]),
        ("QWEN15B-BAL-CAL", qw["s_logits"] - qw["s_logits"].mean(0, keepdims=True), qw["t_logits"] - qw["s_logits"].mean(0, keepdims=True)),
    ]

Xs = {
    "bge_en": (bge_en["s"], syn_bge_en, bge_en["t"]),
    "bge_gl": (bge_gl["s"], syn_bge_gl, bge_gl["t"]),
    "e5_en":  (e5_en["s"],  syn_e5_en,  e5_en["t"]),
    "nom_en": (nom_en["s"], syn_nom_en, nom_en["t"]),
    "bge_lg": (bge_lg["s"], syn_bge_lg, bge_lg["t"]),
    "all_v2": (
        l2(np.hstack([bge_en["s"], bge_gl["s"], e5_en["s"], nom_en["s"], bge_lg["s"]])),
        l2(np.hstack([syn_bge_en, syn_bge_gl, syn_e5_en, syn_nom_en, syn_bge_lg])),
        l2(np.hstack([bge_en["t"], bge_gl["t"], e5_en["t"], nom_en["t"], bge_lg["t"]])),
    ),
}

vec = FeatureUnion([
    ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("c", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True)),
])
tf_all = vec.fit_transform(s_df["en_norm"].fillna("").tolist() + syn_en + t_df["en_norm"].fillna("").tolist())
Xs["tfidf_en"] = (tf_all[:60].toarray(), tf_all[60:60+len(syn_en)].toarray(), tf_all[60+len(syn_en):].toarray())

def fit_predict_probs(X_tr, y_tr, X_te, mtype, C, sw=None):
    if mtype == "ridge":
        m = RidgeClassifier(alpha=C)
        m.fit(X_tr, y_tr, sample_weight=sw)
        return softmax(m.decision_function(X_te) * 3.0, axis=1)
    elif mtype == "logreg":
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(X_tr, y_tr, sample_weight=sw)
        return m.predict_proba(X_te)
    elif mtype == "svc":
        m = LinearSVC(C=C)
        m.fit(X_tr, y_tr, sample_weight=sw)
        return softmax(m.decision_function(X_te) * 2.5, axis=1)
    elif mtype == "rbf":
        m = SVC(C=C, kernel="rbf")
        m.fit(X_tr, y_tr, sample_weight=sw)
        return softmax(m.decision_function(X_te) * 2.5, axis=1)

emb_cfg = [
    ("bge_en",   "ridge",  0.5, 1.2),
    ("bge_gl",   "ridge",  0.5, 1.2),
    ("e5_en",    "svc",    1.5, 1.0),
    ("nom_en",   "ridge",  0.5, 1.2),
    ("bge_lg",   "ridge",  0.5, 1.5),
    ("all_v2",   "ridge",  0.5, 2.0),
    ("all_v2",   "rbf",    3.0, 1.2),
    ("tfidf_en", "logreg", 8.0, 0.5),
]
w_sum = sum(w for _, _, _, w in emb_cfg)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
loo = LeaveOneOut()

# Compare sample-only vs sample + synthetic on STRICT held-out sample.csv OOF CV!
for syn_weight in [0.0, 0.6, 0.85, 1.0]:
    p_oof = np.zeros((60, 6))
    for feat, mtype, cval, w in emb_cfg:
        Xs_tr, Xsyn_tr, _ = Xs[feat]
        ploo = np.zeros((60, 6))
        for tr, va in loo.split(Xs_tr):
            if syn_weight > 0:
                X_comb = np.vstack([Xs_tr[tr], Xsyn_tr])
                y_comb = np.concatenate([y_s[tr], y_syn])
                sw = np.concatenate([np.ones(len(tr)), np.full(len(y_syn), syn_weight)])
                ploo[va] = fit_predict_probs(X_comb, y_comb, Xs_tr[va], mtype, cval, sw=sw)
            else:
                ploo[va] = fit_predict_probs(Xs_tr[tr], y_s[tr], Xs_tr[va], mtype, cval)
        p_oof += (w / w_sum) * ploo
    print(f"Emb Stack (syn_w={syn_weight:.2f}) LOO: Acc={accuracy_score(y_s, p_oof.argmax(1)):.4f} Macro-F1={f1_score(y_s, p_oof.argmax(1), average='macro'):.4f}")

# Use best syn_weight on LOO
best_sw = max([0.0, 0.6, 0.85, 1.0], key=lambda sw_val: (
    lambda p: (accuracy_score(y_s, p.argmax(1)), f1_score(y_s, p.argmax(1), average="macro"))
)(sum((w/w_sum)*np.vstack([
    fit_predict_probs(
        np.vstack([Xs[f][0][tr], Xs[f][1]]) if sw_val > 0 else Xs[f][0][tr],
        np.concatenate([y_s[tr], y_syn]) if sw_val > 0 else y_s[tr],
        Xs[f][0][va], mt, cv,
        sw=np.concatenate([np.ones(len(tr)), np.full(len(y_syn), sw_val)]) if sw_val > 0 else None
    ) for tr, va in loo.split(Xs[f][0])
]) for f, mt, cv, w in emb_cfg)))

print(f"\nBest syn_weight on LOO = {best_sw}")
p_emb_oof = np.zeros((60, 6))
p_emb_te  = np.zeros((len(t_df), 6))
for f, mt, cv, w in emb_cfg:
    Xs_tr, Xsyn_tr, X_te = Xs[f]
    ploo = np.zeros((60, 6))
    for tr, va in loo.split(Xs_tr):
        if best_sw > 0:
            X_c = np.vstack([Xs_tr[tr], Xsyn_tr])
            y_c = np.concatenate([y_s[tr], y_syn])
            sw_c = np.concatenate([np.ones(len(tr)), np.full(len(y_syn), best_sw)])
            ploo[va] = fit_predict_probs(X_c, y_c, Xs_tr[va], mt, cv, sw=sw_c)
        else:
            ploo[va] = fit_predict_probs(Xs_tr[tr], y_s[tr], Xs_tr[va], mt, cv)
    p_emb_oof += (w / w_sum) * ploo
    if best_sw > 0:
        X_full = np.vstack([Xs_tr, Xsyn_tr])
        y_full = np.concatenate([y_s, y_syn])
        sw_full = np.concatenate([np.ones(60), np.full(len(y_syn), best_sw)])
        p_emb_te += (w / w_sum) * fit_predict_probs(X_full, y_full, X_te, mt, cv, sw=sw_full)
    else:
        p_emb_te += (w / w_sum) * fit_predict_probs(Xs_tr, y_s, X_te, mt, cv)

# Select best Qwen variant by OOF Macro-F1
best_q = max(q_opts, key=lambda x: (f1_score(y_s, x[1].argmax(1), average="macro"), accuracy_score(y_s, x[1].argmax(1))))
print(f"Selected Qwen variant: {best_q[0]} | 5F-OOF Acc={accuracy_score(y_s, best_q[1].argmax(1)):.4f} F1={f1_score(y_s, best_q[1].argmax(1), average='macro'):.4f}")
p_qw_s = softmax(best_q[1], axis=1)
p_qw_t = softmax(best_q[2], axis=1)

# Stage-1 Blend Optimization (strictly maximizing Macro-F1 on OOF!)
best_wq, best_s1_sc = 0.5, (-1, -1)
for wq in [0.25, 0.35, 0.45, 0.50, 0.55, 0.65]:
    ps = wq * p_qw_s + (1.0 - wq) * p_emb_oof
    sc = (f1_score(y_s, ps.argmax(1), average="macro"), accuracy_score(y_s, ps.argmax(1)))
    print(f"  Blend w_qwen={wq:.2f} | OOF F1={sc[0]:.4f} Acc={sc[1]:.4f}")
    if sc > best_s1_sc:
        best_s1_sc, best_wq = sc, wq
print(f">>> Best Stage-1 w_qwen={best_wq:.2f}: OOF F1={best_s1_sc[0]:.4f} Acc={best_s1_sc[1]:.4f}")

# Canonical cluster aggregation
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
key_to_idxs = {}
for i, k in enumerate(t_keys):
    key_to_idxs.setdefault(k, []).append(i)

def cluster_avg(p):
    out = p.copy()
    for idxs in key_to_idxs.values():
        out[idxs] = p[idxs].mean(axis=0, keepdims=True)
    return out

p_s1_te = cluster_avg(best_wq * p_qw_t + (1.0 - best_wq) * p_emb_te)

# Stage-2 OOF-verified Transductive Self-Training on top of Sample + Synthetic
oof_s2_emb = np.zeros((60, 6))
for tr, va in skf.split(np.zeros(60), y_s):
    p_emb_tr_te = np.zeros((len(t_df), 6))
    for f, mt, cv, w in emb_cfg:
        Xs_tr, Xsyn_tr, X_te = Xs[f]
        if best_sw > 0:
            X_c = np.vstack([Xs_tr[tr], Xsyn_tr])
            y_c = np.concatenate([y_s[tr], y_syn])
            sw_c = np.concatenate([np.ones(len(tr)), np.full(len(y_syn), best_sw)])
            p_emb_tr_te += (w / w_sum) * fit_predict_probs(X_c, y_c, X_te, mt, cv, sw=sw_c)
        else:
            p_emb_tr_te += (w / w_sum) * fit_predict_probs(Xs_tr[tr], y_s[tr], X_te, mt, cv)
    p_fold_te = cluster_avg(best_wq * p_qw_t + (1.0 - best_wq) * p_emb_tr_te)

    p_idxs, p_ys = [], []
    pred_c, pred_cf = p_fold_te.argmax(1), p_fold_te.max(1)
    for c in range(6):
        cands = [i for i in range(len(t_df)) if pred_c[i] == c and pred_cf[i] >= 0.72]
        cands.sort(key=lambda i: -pred_cf[i])
        for i in cands[:35]:
            p_idxs.append(i); p_ys.append(c)

    for f, mt, cv, w in emb_cfg:
        Xs_tr, Xsyn_tr, X_te = Xs[f]
        parts_X = [Xs_tr[tr]] + ([Xsyn_tr] if best_sw > 0 else []) + [X_te[p_idxs]]
        parts_y = [y_s[tr]] + ([y_syn] if best_sw > 0 else []) + [np.array(p_ys)]
        parts_sw = [np.ones(len(tr))] + ([np.full(len(y_syn), best_sw)] if best_sw > 0 else []) + [np.full(len(p_ys), 0.7)]
        oof_s2_emb[va] += (w / w_sum) * fit_predict_probs(np.vstack(parts_X), np.concatenate(parts_y), Xs_tr[va], mt, cv, sw=np.concatenate(parts_sw))

p_s2_oof = best_wq * p_qw_s + (1.0 - best_wq) * oof_s2_emb
s2_sc = (f1_score(y_s, p_s2_oof.argmax(1), average="macro"), accuracy_score(y_s, p_s2_oof.argmax(1)))
print(f">>> Stage-2 Emb+Synth+SelfTrain OOF : F1={f1_score(y_s, oof_s2_emb.argmax(1), average='macro'):.4f} Acc={accuracy_score(y_s, oof_s2_emb.argmax(1)):.4f}")
print(f">>> Stage-2 Full Ensemble OOF       : F1={s2_sc[0]:.4f} Acc={s2_sc[1]:.4f}")
for i in range(60):
    if p_s2_oof.argmax(1)[i] != y_s[i]:
        print(f"  [S2 ERR] id={s_df.id.iloc[i]:2d} true={y_s[i]} pred={p_s2_oof.argmax(1)[i]} | {s_df.raw.iloc[i]}")

# Full Stage-2 test prediction
p_idxs_f, p_ys_f = [], []
pred_c, pred_cf = p_s1_te.argmax(1), p_s1_te.max(1)
for c in range(6):
    cands = [i for i in range(len(t_df)) if pred_c[i] == c and pred_cf[i] >= 0.72]
    cands.sort(key=lambda i: -pred_cf[i])
    for i in cands[:35]:
        p_idxs_f.append(i); p_ys_f.append(c)

p_s2_emb_te = np.zeros((len(t_df), 6))
for f, mt, cv, w in emb_cfg:
    Xs_tr, Xsyn_tr, X_te = Xs[f]
    parts_X = [Xs_tr] + ([Xsyn_tr] if best_sw > 0 else []) + [X_te[p_idxs_f]]
    parts_y = [y_s] + ([y_syn] if best_sw > 0 else []) + [np.array(p_ys_f)]
    parts_sw = [np.ones(60)] + ([np.full(len(y_syn), best_sw)] if best_sw > 0 else []) + [np.full(len(p_ys_f), 0.7)]
    p_s2_emb_te += (w / w_sum) * fit_predict_probs(np.vstack(parts_X), np.concatenate(parts_y), X_te, mt, cv, sw=np.concatenate(parts_sw))

use_s2 = s2_sc >= best_s1_sc
p_final = cluster_avg(best_wq * p_qw_t + (1.0 - best_wq) * p_s2_emb_te) if use_s2 else p_s1_te
final_preds = p_final.argmax(1).astype(int)

print("\nUsed Stage-2?", use_s2, "| Pseudo-labeled count:", len(p_idxs_f))
print("Final Test Class Distribution:", pd.Series(final_preds).value_counts().sort_index().to_dict())
print("Qwen vs Emb+Synth-S2 Agreement:", (p_qw_t.argmax(1) == p_s2_emb_te.argmax(1)).sum(), "/ 300")

# Compare changes vs previous submission_final.csv and inspect any remaining borderline clusters
prev = pd.read_csv("experiments/submission_final.csv")
changed = (prev["label"].values != final_preds)
print(f"Changed rows vs previous submission_final.csv: {changed.sum()} / 300")
seen = set()
for i in np.where(changed)[0]:
    k = t_df["en_norm"].iloc[i]
    if k in seen: continue
    seen.add(k)
    print(f"  [FLIP] id={t_df.id.iloc[i]} old={prev.label.iloc[i]} -> new={final_preds[i]} (conf={p_final[i].max():.2f}) | {t_df.denoised.iloc[i]} ==> {k}")

seen = set()
print("\nRemaining borderline clusters (conf < 0.70):")
for i, k in enumerate(t_keys):
    if k in seen: continue
    seen.add(k)
    if p_final[i].max() < 0.70:
        top2 = np.argsort(-p_final[i])[:2]
        print(f"  conf={p_final[i].max():.2f} (x{len(key_to_idxs[k])}) [{top2[0]}:{p_final[i,top2[0]]:.2f}, {top2[1]}:{p_final[i,top2[1]]:.2f}] | qw={p_qw_t[i].argmax()} emb={p_s2_emb_te[i].argmax()} | {t_df.denoised.iloc[i]}")

sub = pd.DataFrame({"id": t_df["id"].astype(int), "label": final_preds})
assert len(sub) == len(sub_ref) == 300
assert (sub["id"].values == sub_ref["id"].values).all()
assert sub["label"].isin([0, 1, 2, 3, 4, 5]).all()
sub.to_csv("submission.csv", index=False)
sub.to_csv("experiments/submission_v2.csv", index=False)
print("SAVED & VERIFIED upgraded submission.csv (v2)!")
