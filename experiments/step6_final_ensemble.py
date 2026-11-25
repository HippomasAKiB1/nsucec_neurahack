import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.special import softmax

from text_clean import canonical_key

SEED = 42
s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
sub_sample = pd.read_csv("mock/dataset/sample_submission.csv")
y = s_df["label"].values

def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

bge_gls = np.load("experiments/cache/bge_m3_gls.npz")
bge_en = np.load("experiments/cache/bge_m3_en.npz")
e5_en = np.load("experiments/cache/e5_base_en.npz")
nom_en = np.load("experiments/cache/nomic_en.npz")
mdeb = np.load("experiments/cache/mdeberta_zs.npz")
qwen = np.load("experiments/cache/qwen15b_kv.npz")

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

emb_cfg = [
    ("bge_gls", "ridge", 0.5, 1.5),
    ("bge_en", "ridge", 0.5, 1.0),
    ("e5_en", "svc", 1.5, 1.2),
    ("nom_en", "ridge", 0.5, 1.2),
    ("all_mix", "ridge", 0.5, 1.5),
    ("tfidf_en", "logreg", 8.0, 0.4),
]

# 1. Stage-1 Embedding Ensemble OOF (10-fold & LOO)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
loo = LeaveOneOut()
p_emb_10f = np.zeros((60, 6))
p_emb_loo = np.zeros((60, 6))
p_emb_te = np.zeros((len(t_df), 6))
w_sum = sum(w for _, _, _, w in emb_cfg)

for feat, mtype, cval, w in emb_cfg:
    Xtr, Xte = Xs[feat]
    p10 = np.zeros((60, 6))
    for tr, va in skf.split(Xtr, y):
        p10[va] = fit_predict_probs(Xtr[tr], y[tr], Xtr[va], mtype, cval)
    ploo = np.zeros((60, 6))
    for tr, va in loo.split(Xtr):
        ploo[va] = fit_predict_probs(Xtr[tr], y[tr], Xtr[va], mtype, cval)
    p_emb_10f += (w / w_sum) * p10
    p_emb_loo += (w / w_sum) * ploo
    p_emb_te += (w / w_sum) * fit_predict_probs(Xtr, y, Xte, mtype, cval)

p_deb_s = softmax(mdeb["s_scores"] * 1.5, axis=1)
p_deb_t = softmax(mdeb["t_scores"] * 1.5, axis=1)

# Qwen calibrated probabilities (temperature=1.0 on digit logits)
p_qwen_s = softmax(qwen["s_logits"], axis=1)
p_qwen_t = softmax(qwen["t_logits"], axis=1)

print(f"Embedding Ensemble LOO : Acc={accuracy_score(y, p_emb_loo.argmax(1)):.4f} F1={f1_score(y, p_emb_loo.argmax(1), average='macro'):.4f}")
print(f"mDeBERTa Zero-Shot     : Acc={accuracy_score(y, p_deb_s.argmax(1)):.4f} F1={f1_score(y, p_deb_s.argmax(1), average='macro'):.4f}")
print(f"Qwen2.5-1.5B OOF       : Acc={accuracy_score(y, p_qwen_s.argmax(1)):.4f} F1={f1_score(y, p_qwen_s.argmax(1), average='macro'):.4f}")

# Tune blend weight on OOF (coarse grid over w_qwen vs w_emb vs w_deb)
best_acc, best_f1, best_w = -1, -1, None
for wq in [0.4, 0.5, 0.6, 0.7, 0.8]:
    for wd in [0.0, 0.05, 0.10]:
        we = 1.0 - wq - wd
        if we < 0: continue
        ps = wq * p_qwen_s + we * p_emb_loo + wd * p_deb_s
        acc = accuracy_score(y, ps.argmax(1))
        f1 = f1_score(y, ps.argmax(1), average="macro")
        if (acc, f1) > (best_acc, best_f1):
            best_acc, best_f1, best_w = acc, f1, (wq, we, wd)
        print(f"  Blend w_qwen={wq:.2f} w_emb={we:.2f} w_deb={wd:.2f} | OOF Acc={acc:.4f} F1={f1:.4f}")

wq, we, wd = best_w
print(f"\n>>> Best Stage-1 Blend weights (qwen, emb, deb) = {best_w} -> OOF Acc={best_acc:.4f} F1={best_f1:.4f}")

# 2. Canonical Cluster Aggregation on test
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
key_to_idxs = {}
for i, k in enumerate(t_keys):
    key_to_idxs.setdefault(k, []).append(i)

p_stage1_te = wq * p_qwen_t + we * p_emb_te + wd * p_deb_t
p_clust_te = p_stage1_te.copy()
for k, idxs in key_to_idxs.items():
    p_clust_te[idxs] = p_stage1_te[idxs].mean(axis=0, keepdims=True)

# 3. Stage-2 Transductive Semi-Supervised Self-Training (OOF verified!)
# Check if retraining embeddings on sample + confident Stage-1 test clusters improves OOF CV
oof_st_emb = np.zeros((60, 6))
for tr, va in skf.split(np.zeros(60), y):
    # Stage-1 emb on tr -> te
    p_emb_tr_te = np.zeros((len(t_df), 6))
    for feat, mtype, cval, w in emb_cfg:
        Xtr, Xte = Xs[feat]
        p_emb_tr_te += (w / w_sum) * fit_predict_probs(Xtr[tr], y[tr], Xte, mtype, cval)
    p_tr_te = wq * p_qwen_t + we * p_emb_tr_te + wd * p_deb_t
    p_tr_clust = p_tr_te.copy()
    for k, idxs in key_to_idxs.items():
        p_tr_clust[idxs] = p_tr_te[idxs].mean(axis=0, keepdims=True)

    # Pick high-confidence pseudo-labels (conf >= 0.72, balanced up to 35/class)
    p_idxs, p_ys = [], []
    pred_c = p_tr_clust.argmax(1)
    pred_cf = p_tr_clust.max(1)
    for c in range(6):
        cands = [i for i in range(len(t_df)) if pred_c[i] == c and pred_cf[i] >= 0.72]
        cands.sort(key=lambda i: -pred_cf[i])
        for i in cands[:35]:
            p_idxs.append(i)
            p_ys.append(c)
    y_aug = np.concatenate([y[tr], np.array(p_ys)])
    sw_aug = np.concatenate([np.ones(len(tr)), np.full(len(p_ys), 0.7)])
    for feat, mtype, cval, w in emb_cfg:
        Xtr, Xte = Xs[feat]
        X_aug = np.vstack([Xtr[tr], Xte[p_idxs]])
        oof_st_emb[va] += (w / w_sum) * fit_predict_probs(X_aug, y_aug, Xtr[va], mtype, cval, sw=sw_aug)

p_st_oof = wq * p_qwen_s + we * oof_st_emb + wd * p_deb_s
print(f"Stage-2 Self-Trained Emb alone OOF : Acc={accuracy_score(y, oof_st_emb.argmax(1)):.4f} F1={f1_score(y, oof_st_emb.argmax(1), average='macro'):.4f}")
print(f"Stage-2 Full Ensemble OOF          : Acc={accuracy_score(y, p_st_oof.argmax(1)):.4f} F1={f1_score(y, p_st_oof.argmax(1), average='macro'):.4f}")

# Full Stage-2 test prediction (using all 60 sample + high-conf Stage-1 pseudo-labels)
p_idxs_full, p_ys_full = [], []
pred_c = p_clust_te.argmax(1)
pred_cf = p_clust_te.max(1)
for c in range(6):
    cands = [i for i in range(len(t_df)) if pred_c[i] == c and pred_cf[i] >= 0.72]
    cands.sort(key=lambda i: -pred_cf[i])
    for i in cands[:35]:
        p_idxs_full.append(i)
        p_ys_full.append(c)
print(f"Selected {len(p_idxs_full)} pseudo-labeled test rows across classes: {pd.Series(p_ys_full).value_counts().sort_index().to_dict()}")

y_aug_f = np.concatenate([y, np.array(p_ys_full)])
sw_aug_f = np.concatenate([np.ones(60), np.full(len(p_ys_full), 0.7)])
p_st_emb_te = np.zeros((len(t_df), 6))
for feat, mtype, cval, w in emb_cfg:
    Xtr, Xte = Xs[feat]
    X_aug = np.vstack([Xtr, Xte[p_idxs_full]])
    p_st_emb_te += (w / w_sum) * fit_predict_probs(X_aug, y_aug_f, Xte, mtype, cval, sw=sw_aug_f)

# Choose Stage-2 only if OOF >= Stage-1 OOF (otherwise stick with Stage-1!)
use_st = (accuracy_score(y, p_st_oof.argmax(1)), f1_score(y, p_st_oof.argmax(1), average="macro")) >= (best_acc, best_f1)
print("Using Stage-2 Self-Trained Ensemble?", use_st)
p_final_raw = (wq * p_qwen_t + we * p_st_emb_te + wd * p_deb_t) if use_st else p_stage1_te
p_final = p_final_raw.copy()
for k, idxs in key_to_idxs.items():
    p_final[idxs] = p_final_raw[idxs].mean(axis=0, keepdims=True)

final_preds = p_final.argmax(1).astype(int)
print("Final Test Class Distribution:", pd.Series(final_preds).value_counts().sort_index().to_dict())
print("Qwen vs Emb-Stage2 Agreement on Test:", (p_qwen_t.argmax(1) == p_st_emb_te.argmax(1)).sum(), "/ 300")

# Write submission.csv and verify strictly against sample_submission.csv
sub = pd.DataFrame({"id": t_df["id"].astype(int), "label": final_preds})
assert len(sub) == len(sub_sample) == 300
assert list(sub.columns) == list(sub_sample.columns) == ["id", "label"]
assert (sub["id"].values == sub_sample["id"].values).all()
assert sub["label"].isin([0, 1, 2, 3, 4, 5]).all()
assert sub.isnull().sum().sum() == 0

sub.to_csv("submission.csv", index=False)
sub.to_csv("experiments/submission_final.csv", index=False)
print("SAVED & VERIFIED submission.csv (300 rows)!")
