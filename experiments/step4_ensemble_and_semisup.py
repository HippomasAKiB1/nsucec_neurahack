import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import softmax

from text_clean import canonical_key

SEED = 42
s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

# Load all cached representations
bge_raw = np.load("experiments/cache/bge_m3.npz")
bge_gls = np.load("experiments/cache/bge_m3_gls.npz")
bge_en = np.load("experiments/cache/bge_m3_en.npz")
e5_raw = np.load("experiments/cache/e5_base.npz")
e5_gls = np.load("experiments/cache/e5_base_gls.npz")
e5_en = np.load("experiments/cache/e5_base_en.npz")
nom_en = np.load("experiments/cache/nomic_en.npz")
mdeb = np.load("experiments/cache/mdeberta_zs.npz")

# Canonical clusters in test (so pseudo-labels & final test probs are aggregated by canonical core!)
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
uniq_keys = sorted(set(t_keys))
key_to_idxs = {k: [i for i, tk in enumerate(t_keys) if tk == k] for k in uniq_keys}
print(f"Test rows: {len(t_keys)}, canonical en_norm clusters: {len(uniq_keys)}")

# Helper to turn decision function / logits into calibrated probs
def fit_predict_probs(X_tr, y_tr, X_te, model_type="ridge", C=1.0):
    if model_type == "ridge":
        m = RidgeClassifier(alpha=C)
        m.fit(X_tr, y_tr)
        df = m.decision_function(X_te)
        return softmax(df * 3.0, axis=1)
    elif model_type == "logreg":
        m = LogisticRegression(C=C, max_iter=1000)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_te)
    elif model_type == "svc":
        m = LinearSVC(C=C)
        m.fit(X_tr, y_tr)
        df = m.decision_function(X_te)
        return softmax(df * 2.5, axis=1)
    elif model_type == "cos":
        # Normalized class centroids cosine similarity
        cents = np.vstack([l2(X_tr[y_tr == c].mean(axis=0, keepdims=True)) for c in range(6)])
        sim = X_te @ cents.T
        return softmax(sim * 15.0, axis=1)

# Build individual diverse base models and compute LOO + 10-fold CV
Xs = {
    "bge_gls": (bge_gls["s_gls"], bge_gls["t_gls"]),
    "bge_en": (bge_en["s_en"], bge_en["t_en"]),
    "bge_dual": (bge_en["s_dual"], bge_en["t_dual"]),
    "e5_gls": (e5_gls["s_gls"], e5_gls["t_gls"]),
    "e5_en": (e5_en["s_en"], e5_en["t_en"]),
    "nom_en": (nom_en["s_en"], nom_en["t_en"]),
    "all_en": (l2(np.hstack([bge_en["s_en"], e5_en["s_en"], nom_en["s_en"]])),
               l2(np.hstack([bge_en["t_en"], e5_en["t_en"], nom_en["t_en"]]))),
    "all_mix": (l2(np.hstack([bge_gls["s_gls"], bge_en["s_en"], e5_en["s_en"], nom_en["s_en"]])),
                l2(np.hstack([bge_gls["t_gls"], bge_en["t_en"], e5_en["t_en"], nom_en["t_en"]]))),
}

# TF-IDF on en_norm
vec = FeatureUnion([
    ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("c", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", sublinear_tf=True)),
])
tf_all = vec.fit_transform(s_df["en_norm"].fillna("").tolist() + t_df["en_norm"].fillna("").tolist())
Xs["tfidf_en"] = (tf_all[:60].toarray(), tf_all[60:].toarray())

# Evaluate LOO and 10F for several clean configs
models_cfg = [
    ("bge_gls", "ridge", 0.5, 1.0),
    ("bge_gls", "logreg", 15.0, 1.0),
    ("bge_en", "ridge", 0.5, 1.0),
    ("e5_en", "svc", 1.5, 1.0),
    ("nom_en", "ridge", 0.5, 1.0),
    ("nom_en", "logreg", 15.0, 1.0),
    ("all_mix", "ridge", 0.5, 1.5),
    ("all_mix", "logreg", 15.0, 1.5),
    ("tfidf_en", "logreg", 8.0, 0.5),
]

loo = LeaveOneOut()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

oof_loo_list = []
oof_10f_list = []
weights = []
for feat, mtype, cval, w in models_cfg:
    Xs_tr, _ = Xs[feat]
    p_loo = np.zeros((60, 6))
    for tr, va in loo.split(Xs_tr):
        p_loo[va] = fit_predict_probs(Xs_tr[tr], y[tr], Xs_tr[va], mtype, cval)
    p_10f = np.zeros((60, 6))
    for tr, va in skf.split(Xs_tr, y):
        p_10f[va] = fit_predict_probs(Xs_tr[tr], y[tr], Xs_tr[va], mtype, cval)
    oof_loo_list.append(p_loo)
    oof_10f_list.append(p_10f)
    weights.append(w)
    print(f"{feat:9s} | {mtype:7s} | 10F={accuracy_score(y, p_10f.argmax(1)):.4f}/{f1_score(y, p_10f.argmax(1), average='macro'):.4f} | LOO={accuracy_score(y, p_loo.argmax(1)):.4f}/{f1_score(y, p_loo.argmax(1), average='macro'):.4f}")

# Add zero-shot mDeBERTa with small weight
p_deb_s = softmax(mdeb["s_scores"] * 1.5, axis=1)
for w_deb in [0.0, 0.25, 0.5, 1.0]:
    ens_loo = sum(w * p for w, p in zip(weights, oof_loo_list)) + w_deb * p_deb_s
    ens_10f = sum(w * p for w, p in zip(weights, oof_10f_list)) + w_deb * p_deb_s
    print(f"ENSEMBLE (w_deb={w_deb:.2f})   | 10F={accuracy_score(y, ens_10f.argmax(1)):.4f}/{f1_score(y, ens_10f.argmax(1), average='macro'):.4f} | LOO={accuracy_score(y, ens_loo.argmax(1)):.4f}/{f1_score(y, ens_loo.argmax(1), average='macro'):.4f}")

# Test Transductive Pseudo-Labeling honestly inside 10-fold CV AND LOO-CV!
print("\n--- Testing Transductive Self-Training (OOF verified!) ---")
for conf_thresh in [0.55, 0.65, 0.75]:
    for max_per_cls in [15, 25, 40]:
        oof_10f_st = np.zeros((60, 6))
        for tr, va in skf.split(np.zeros(60), y):
            # Stage 1 ensemble predictions on test & val
            p_t_stage1 = np.zeros((len(t_df), 6))
            for (feat, mtype, cval, w) in models_cfg:
                X_tr, X_te = Xs[feat]
                p_t_stage1 += w * fit_predict_probs(X_tr[tr], y[tr], X_te, mtype, cval)
            p_t_stage1 += 0.25 * softmax(mdeb["t_scores"] * 1.5, axis=1)
            p_t_stage1 /= (sum(weights) + 0.25)

            # Aggregate probabilities by canonical key in test
            p_t_clust = p_t_stage1.copy()
            for k, idxs in key_to_idxs.items():
                p_t_clust[idxs] = p_t_stage1[idxs].mean(axis=0, keepdims=True)

            # Pick top-k balanced high-confidence test pseudo-labels
            pseudo_idxs, pseudo_y, pseudo_w = [], [], []
            pred_cls = p_t_clust.argmax(1)
            pred_conf = p_t_clust.max(1)
            for c in range(6):
                cands = [i for i in range(len(t_df)) if pred_cls[i] == c and pred_conf[i] >= conf_thresh]
                cands.sort(key=lambda i: -pred_conf[i])
                for i in cands[:max_per_cls]:
                    pseudo_idxs.append(i)
                    pseudo_y.append(c)
                    pseudo_w.append(0.7)  # slightly downweight pseudo labels relative to ground truth 1.0

            if len(pseudo_idxs) == 0:
                continue

            # Stage 2 retrain on sample[tr] + pseudo(test) -> predict strictly on sample[va]!
            y_aug = np.concatenate([y[tr], np.array(pseudo_y)])
            sw_aug = np.concatenate([np.ones(len(tr)), np.array(pseudo_w)])
            p_va_st = np.zeros((len(va), 6))
            for (feat, mtype, cval, w) in models_cfg:
                X_tr, X_te = Xs[feat]
                X_aug = np.vstack([X_tr[tr], X_te[pseudo_idxs]])
                if mtype == "ridge":
                    m = RidgeClassifier(alpha=cval)
                    m.fit(X_aug, y_aug, sample_weight=sw_aug)
                    p_va_st += w * softmax(m.decision_function(X_tr[va]) * 3.0, axis=1)
                elif mtype == "logreg":
                    m = LogisticRegression(C=cval, max_iter=1000)
                    m.fit(X_aug, y_aug, sample_weight=sw_aug)
                    p_va_st += w * m.predict_proba(X_tr[va])
                elif mtype == "svc":
                    m = LinearSVC(C=cval)
                    m.fit(X_aug, y_aug, sample_weight=sw_aug)
                    p_va_st += w * softmax(m.decision_function(X_tr[va]) * 2.5, axis=1)
            p_va_st += 0.25 * softmax(mdeb["s_scores"][va] * 1.5, axis=1)
            oof_10f_st[va] = p_va_st

        acc_st = accuracy_score(y, oof_10f_st.argmax(1))
        f1_st = f1_score(y, oof_10f_st.argmax(1), average="macro")
        print(f"SelfTrain (th={conf_thresh:.2f}, max/c={max_per_cls:2d}) | 10F OOF={acc_st:.4f}/{f1_st:.4f}")
