import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.special import softmax

from text_clean import canonical_key

s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

bge_gls = np.load("experiments/cache/bge_m3_gls.npz")
bge_en = np.load("experiments/cache/bge_m3_en.npz")
e5_en = np.load("experiments/cache/e5_base_en.npz")
nom_en = np.load("experiments/cache/nomic_en.npz")
mdeb = np.load("experiments/cache/mdeberta_zs.npz")

Xs = {
    "bge_gls": (bge_gls["s_gls"], bge_gls["t_gls"]),
    "bge_en": (bge_en["s_en"], bge_en["t_en"]),
    "e5_en": (e5_en["s_en"], e5_en["t_en"]),
    "nom_en": (nom_en["s_en"], nom_en["t_en"]),
    "all_mix": (l2(np.hstack([bge_gls["s_gls"], bge_en["s_en"], e5_en["s_en"], nom_en["s_en"]])),
                l2(np.hstack([bge_gls["t_gls"], bge_en["t_en"], e5_en["t_en"], nom_en["t_en"]]))),
}
vec = FeatureUnion([
    ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("c", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True)),
])
tf_all = vec.fit_transform(s_df["en_norm"].fillna("").tolist() + t_df["en_norm"].fillna("").tolist())
Xs["tfidf_en"] = (tf_all[:60].toarray(), tf_all[60:].toarray())

def fit_predict_probs(X_tr, y_tr, X_te, mtype, C):
    if mtype == "ridge":
        m = RidgeClassifier(alpha=C).fit(X_tr, y_tr)
        return softmax(m.decision_function(X_te) * 3.0, axis=1)
    elif mtype == "logreg":
        return LogisticRegression(C=C, max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)
    elif mtype == "svc":
        m = LinearSVC(C=C).fit(X_tr, y_tr)
        return softmax(m.decision_function(X_te) * 2.5, axis=1)

cfg = [
    ("bge_gls", "ridge", 0.5, 1.5),
    ("bge_en", "ridge", 0.5, 1.0),
    ("e5_en", "svc", 1.5, 1.2),
    ("nom_en", "ridge", 0.5, 1.2),
    ("all_mix", "ridge", 0.5, 1.5),
    ("tfidf_en", "logreg", 8.0, 0.4),
]

# LOO evaluation
p_loo = np.zeros((60, 6))
for feat, mtype, cval, w in cfg:
    Xtr, _ = Xs[feat]
    p_m = np.zeros((60, 6))
    for tr, va in LeaveOneOut().split(Xtr):
        p_m[va] = fit_predict_probs(Xtr[tr], y[tr], Xtr[va], mtype, cval)
    p_loo += w * p_m
p_loo += 0.5 * softmax(mdeb["s_scores"] * 1.5, axis=1)
p_loo /= p_loo.sum(axis=1, keepdims=True)
print(f"LOO Ensemble Acc={accuracy_score(y, p_loo.argmax(1)):.4f} F1={f1_score(y, p_loo.argmax(1), average='macro'):.4f}")
for i in range(60):
    if p_loo.argmax(1)[i] != y[i]:
        print(f"  [LOO ERR] id={s_df.id.iloc[i]:2d} true={y[i]} pred={p_loo.argmax(1)[i]} conf={p_loo[i].max():.2f} | {s_df.raw.iloc[i]}")

# Full fit on sample -> predict on test
p_t_models = {}
p_t_ens = np.zeros((len(t_df), 6))
for feat, mtype, cval, w in cfg:
    Xtr, Xte = Xs[feat]
    pm = fit_predict_probs(Xtr, y, Xte, mtype, cval)
    p_t_models[feat] = pm
    p_t_ens += w * pm
p_t_ens += 0.5 * softmax(mdeb["t_scores"] * 1.5, axis=1)
p_t_ens /= p_t_ens.sum(axis=1, keepdims=True)

# Cluster averaging by canonical key
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
key_to_idxs = {}
for i, k in enumerate(t_keys):
    key_to_idxs.setdefault(k, []).append(i)

p_t_clust = p_t_ens.copy()
flips_by_clust = 0
for k, idxs in key_to_idxs.items():
    avg = p_t_ens[idxs].mean(axis=0, keepdims=True)
    for idx in idxs:
        if p_t_ens[idx].argmax() != avg[0].argmax():
            flips_by_clust += 1
        p_t_clust[idx] = avg[0]

preds_raw = p_t_ens.argmax(1)
preds_clust = p_t_clust.argmax(1)
print(f"\nFlips fixed by canonical cluster averaging: {flips_by_clust}")
print("Test predicted class distribution (300 rows):", pd.Series(preds_clust).value_counts().sort_index().to_dict())
print("Confidence quantiles on test:", np.quantile(p_t_clust.max(1), [0.0, 0.05, 0.10, 0.25, 0.50, 0.90]))

# Unanimous agreement across 4 main models (bge_gls, bge_en, e5_en, nom_en)
main_preds = np.stack([p_t_models[k].argmax(1) for k in ["bge_gls", "bge_en", "e5_en", "nom_en"]], axis=1)
unanimous = (main_preds == main_preds[:, [0]]).all(axis=1).sum()
print(f"Unanimous agreement across 4 diverse embedding families: {unanimous}/300 ({unanimous/300:.1%})")

# Print ONLY low-confidence or split canonical clusters (for feature/model diagnosis, NO manual labeling!)
print("\nLowest confidence canonical clusters (conf < 0.55):")
seen = set()
rows_low = []
for i, k in enumerate(t_keys):
    if k in seen: continue
    seen.add(k)
    c = p_t_clust[i].max()
    if c < 0.55:
        top2 = np.argsort(-p_t_clust[i])[:2]
        rows_low.append((c, len(key_to_idxs[k]), top2[0], p_t_clust[i, top2[0]], top2[1], p_t_clust[i, top2[1]], t_df.en_norm.iloc[i]))
for c, cnt, c1, p1, c2, p2, txt in sorted(rows_low):
    print(f"  conf={c:.2f} (x{cnt}) top2=[{c1}:{p1:.2f}, {c2}:{p2:.2f}] | {txt}")
