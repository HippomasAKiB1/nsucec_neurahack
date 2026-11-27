import os
import time
import random
import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack, csr_matrix
from scipy.special import softmax, expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import IsotonicRegression
import lightgbm as lgb

from normalizer import clean_for_model, extract_structural_features
from cv_metrics import (
    TECH_LIST, tech_str_to_vec, vec_to_tech_str,
    eval_multitask, build_template_groups, corrupt_text_adversarial, decode_consistent
)
from synth_generator import generate_dataset

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# 1. Load sample.csv and build template-grouped folds + adversarial robustness validation set
s_df = pd.read_csv("preli/cache/sample_clean.csv")
s_feats = np.load("preli/cache/sample_feats.npy")
FEAT_NAMES = list(extract_structural_features("a", "a").keys())

y_lbl = s_df["label"].values
y_tech = np.vstack([tech_str_to_vec(x) for x in s_df["techniques"]])
y_act = s_df["action"].values
groups = build_template_groups(s_df["clean_text"].fillna("").tolist())

# Generate 5 adversarial corruptions per sample row for Robustness CV (430 rows)
rng_adv = random.Random(SEED + 99)
adv_raw = []
adv_clean = []
adv_feats = []
adv_parent_idx = []
for i, raw in enumerate(s_df["text"].tolist()):
    for _ in range(5):
        c_raw = corrupt_text_adversarial(raw, rng_adv)
        c_cl = clean_for_model(c_raw)
        d = extract_structural_features(c_raw, c_cl)
        adv_raw.append(c_raw)
        adv_clean.append(c_cl)
        adv_feats.append([d[k] for k in FEAT_NAMES])
        adv_parent_idx.append(i)
adv_feats = np.array(adv_feats, dtype=np.float32)
adv_parent_idx = np.array(adv_parent_idx)

# 2. Load or generate Rule-14 synthetic training dataset (30,000 rows)
t0 = time.time()
if os.path.exists("preli/cache/synth_30k.csv") and os.path.exists("preli/cache/synth_30k_feats.npy"):
    syn_df = pd.read_csv("preli/cache/synth_30k.csv")
    syn_clean = syn_df["clean_text"].fillna("").astype(str).tolist()
    syn_feats = np.load("preli/cache/synth_30k_feats.npy")
    print(f"Loaded cached {len(syn_df)} synthetic examples in {time.time()-t0:.1f}s", flush=True)
else:
    syn_df = generate_dataset(n_samples=30000, seed=SEED, corrupt_prob=0.65)
    syn_clean = [clean_for_model(x) for x in syn_df["text"]]
    syn_feats = np.array([[extract_structural_features(r, c)[k] for k in FEAT_NAMES] for r, c in zip(syn_df["text"], syn_clean)], dtype=np.float32)
    syn_out = syn_df.copy()
    syn_out["clean_text"] = syn_clean
    syn_out.to_csv("preli/cache/synth_30k.csv", index=False)
    np.save("preli/cache/synth_30k_feats.npy", syn_feats)
    print(f"Generated & normalized {len(syn_df)} synthetic examples in {time.time()-t0:.1f}s", flush=True)

y_syn_lbl = syn_df["label"].values
y_syn_tech = np.vstack([tech_str_to_vec(x) for x in syn_df["techniques"]])
y_syn_act = syn_df["action"].values

# 3. Fit TF-IDF on synthetic + sample clean texts (memory-safe, capped vocab)
vec = FeatureUnion([
    ("w", TfidfVectorizer(ngram_range=(1, 3), max_features=45000, sublinear_tf=True, min_df=2)),
    ("c", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", max_features=55000, sublinear_tf=True, min_df=3)),
])
X_syn_tf = vec.fit_transform(syn_clean)
X_s_tf = vec.transform(s_df["clean_text"].fillna("").tolist())
X_adv_tf = vec.transform(adv_clean)

def combine_sparse_dense(X_tf, X_dense):
    # Scale numeric length/counts gently
    Xd = X_dense.copy()
    Xd[:, 0] = np.log1p(Xd[:, 0]) / 5.0
    Xd[:, 1] = Xd[:, 1] / 12.0
    Xd[:, 25] = np.log1p(Xd[:, 25]) / 3.0
    Xd[:, 26] = np.log1p(Xd[:, 26]) / 3.0
    return hstack([X_tf, csr_matrix(Xd)], format="csr")

X_syn_all = combine_sparse_dense(X_syn_tf, syn_feats)
X_s_all = combine_sparse_dense(X_s_tf, s_feats)
X_adv_all = combine_sparse_dense(X_adv_tf, adv_feats)

# 4. Evaluate Baseline A (sample-only) vs Baseline B (sample + 30k synthetic) on StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5)

def fit_linear_multitask(X_tr, yl, yt, ya, sw=None):
    clf_l = LogisticRegression(C=3.0, max_iter=300, solver="lbfgs")
    clf_l.fit(X_tr, yl, sample_weight=sw)
    clf_t = [LogisticRegression(C=3.0, max_iter=300).fit(X_tr, yt[:, j], sample_weight=sw) for j in range(6)]
    clf_a = LogisticRegression(C=3.0, max_iter=300).fit(X_tr, ya, sample_weight=sw)
    return clf_l, clf_t, clf_a

def pred_linear_multitask(models, X_te):
    clf_l, clf_t, clf_a = models
    pl = clf_l.predict_proba(X_te)
    pt = np.column_stack([m.predict_proba(X_te)[:, 1] for m in clf_t])
    pa = clf_a.predict_proba(X_te)
    return pl, pt, pa

# Baseline A: Sample-only
oof_l_a, oof_t_a, oof_a_a = np.zeros((86, 6)), np.zeros((86, 6)), np.zeros((86, 6))
for tr, va in sgkf.split(X_s_all, y_lbl, groups):
    mdls = fit_linear_multitask(X_s_all[tr], y_lbl[tr], y_tech[tr], y_act[tr])
    oof_l_a[va], oof_t_a[va], oof_a_a[va] = pred_linear_multitask(mdls, X_s_all[va])
res_a_pl, res_a_pt, res_a_pa = decode_consistent(oof_l_a, oof_t_a, oof_a_a)
res_a = eval_multitask(y_lbl, res_a_pl, y_tech, res_a_pt, y_act, res_a_pa, oof_l_a.max(1))
print("\n[Baseline A: Sample-only TFIDF+Feats] Group-CV:", {k: round(v, 2) for k, v in res_a.items()})

# Baseline B: Sample[tr] + 30k Synthetic
oof_l_b, oof_t_b, oof_a_b = np.zeros((86, 6)), np.zeros((86, 6)), np.zeros((86, 6))
adv_l_b, adv_t_b, adv_a_b = np.zeros((len(adv_clean), 6)), np.zeros((len(adv_clean), 6)), np.zeros((len(adv_clean), 6))
for fold, (tr, va) in enumerate(sgkf.split(X_s_all, y_lbl, groups)):
    X_tr = vstack([X_syn_all, X_s_all[tr]], format="csr")
    yl_tr = np.concatenate([y_syn_lbl, y_lbl[tr]])
    yt_tr = np.vstack([y_syn_tech, y_tech[tr]])
    ya_tr = np.concatenate([y_syn_act, y_act[tr]])
    # Upweight genuine sample rows slightly vs synthetic
    sw = np.concatenate([np.ones(len(syn_df)), np.full(len(tr), 4.0)])
    mdls = fit_linear_multitask(X_tr, yl_tr, yt_tr, ya_tr, sw=sw)
    oof_l_b[va], oof_t_b[va], oof_a_b[va] = pred_linear_multitask(mdls, X_s_all[va])
    adv_mask = np.isin(adv_parent_idx, va)
    adv_l_b[adv_mask], adv_t_b[adv_mask], adv_a_b[adv_mask] = pred_linear_multitask(mdls, X_adv_all[adv_mask])

b_pl, b_pt, b_pa = decode_consistent(oof_l_b, oof_t_b, oof_a_b)
b_adv_pl, b_adv_pt, b_adv_pa = decode_consistent(adv_l_b, adv_t_b, adv_a_b)
res_b = eval_multitask(y_lbl, b_pl, y_tech, b_pt, y_act, b_pa, oof_l_b.max(1))
res_b_adv = eval_multitask(
    y_lbl[adv_parent_idx], b_adv_pl,
    y_tech[adv_parent_idx], b_adv_pt,
    y_act[adv_parent_idx], b_adv_pa, adv_l_b.max(1)
)
print("[Baseline B: +30k v2 Synth TFIDF+Feats] Group-CV   :", {k: round(v, 2) for k, v in res_b.items()})
print("[Baseline B: +30k v2 Synth TFIDF+Feats] Robust-CV  :", {k: round(v, 2) for k, v in res_b_adv.items()})

# Print any remaining errors on sample.csv Group-CV
print("\nErrors of Baseline B on sample.csv Group-CV:")
for i in range(len(s_df)):
    if b_pl[i] != y_lbl[i] or b_pa[i] != y_act[i]:
        print(f"  id={s_df.id.iloc[i]} L:{y_lbl[i]}->{b_pl[i]}({oof_l_b[i].max():.2f}) A:{y_act[i]}->{b_pa[i]} T:{s_df.techniques.iloc[i]}->{vec_to_tech_str(b_pt[i])} | {s_df.text.iloc[i][:95]}")

# 5. Train full TF-IDF+Feats + LightGBM Stack and save OOF + 500k test probs!
# Compute 5-fold OOF on synthetic for GBDT meta features
syn_oof_l = np.zeros((len(syn_df), 6), dtype=np.float32)
syn_oof_t = np.zeros((len(syn_df), 6), dtype=np.float32)
syn_oof_a = np.zeros((len(syn_df), 6), dtype=np.float32)
idx_all = np.arange(len(syn_df))
for f_idx in range(5):
    va_m = (idx_all % 5 == f_idx)
    tr_m = ~va_m
    mdls = fit_linear_multitask(X_syn_all[tr_m], y_syn_lbl[tr_m], y_syn_tech[tr_m], y_syn_act[tr_m])
    syn_oof_l[va_m], syn_oof_t[va_m], syn_oof_a[va_m] = pred_linear_multitask(mdls, X_syn_all[va_m])

# Fit full linear models on synth + all 86 sample
X_full_tr = vstack([X_syn_all, X_s_all], format="csr")
yl_full = np.concatenate([y_syn_lbl, y_lbl])
yt_full = np.vstack([y_syn_tech, y_tech])
ya_full = np.concatenate([y_syn_act, y_act])
sw_full = np.concatenate([np.ones(len(syn_df)), np.full(len(s_df), 4.0)])
full_lin_mdls = fit_linear_multitask(X_full_tr, yl_full, yt_full, ya_full, sw=sw_full)

# Save linear OOF & test predictions (streamed in chunks over 500k!)
t1 = time.time()
te_feats = np.load("preli/cache/test_feats.npy")
te_l_lin = np.zeros((500000, 6), dtype=np.float32)
te_t_lin = np.zeros((500000, 6), dtype=np.float32)
te_a_lin = np.zeros((500000, 6), dtype=np.float32)
offset = 0
for chunk in pd.read_csv("preli/cache/test_clean.csv", chunksize=50000):
    n = len(chunk)
    X_tf_c = vec.transform(chunk["clean_text"].fillna("").tolist())
    X_c = combine_sparse_dense(X_tf_c, te_feats[offset:offset+n])
    te_l_lin[offset:offset+n], te_t_lin[offset:offset+n], te_a_lin[offset:offset+n] = pred_linear_multitask(full_lin_mdls, X_c)
    offset += n
print(f"Predicted 500k Linear+TFIDF in {time.time()-t1:.1f}s", flush=True)

# 6. Train LightGBM on [linear probs (18) + structural features (27)] = 45 dense features!
X_meta_syn = np.hstack([syn_oof_l, syn_oof_t, syn_oof_a, syn_feats])
X_meta_s = np.hstack([oof_l_b, oof_t_b, oof_a_b, s_feats])
X_meta_adv = np.hstack([adv_l_b, adv_t_b, adv_a_b, adv_feats])
X_meta_te = np.hstack([te_l_lin, te_t_lin, te_a_lin, te_feats])

# Validate LightGBM on sample Group-CV (using syn_oof + sample[tr] -> sample[va])
lgb_oof_l, lgb_oof_t, lgb_oof_a = np.zeros((86, 6)), np.zeros((86, 6)), np.zeros((86, 6))
lgb_adv_l, lgb_adv_t, lgb_adv_a = np.zeros((len(adv_clean), 6)), np.zeros((len(adv_clean), 6)), np.zeros((len(adv_clean), 6))

lgb_params_mc = dict(n_estimators=180, learning_rate=0.06, num_leaves=23, min_child_samples=25, subsample_for_bin=50000, colsample_bytree=0.80, subsample=0.80, subsample_freq=1, random_state=SEED, verbose=-1, n_jobs=8)
for tr, va in sgkf.split(X_s_all, y_lbl, groups):
    Xm_tr = np.vstack([X_meta_syn, X_meta_s[tr]])
    yl_tr = np.concatenate([y_syn_lbl, y_lbl[tr]])
    yt_tr = np.vstack([y_syn_tech, y_tech[tr]])
    ya_tr = np.concatenate([y_syn_act, y_act[tr]])
    sw_tr = np.concatenate([np.ones(len(syn_df)), np.full(len(tr), 4.0)])

    m_l = lgb.LGBMClassifier(objective="multiclass", num_class=6, **lgb_params_mc).fit(Xm_tr, yl_tr, sample_weight=sw_tr)
    m_a = lgb.LGBMClassifier(objective="multiclass", num_class=6, **lgb_params_mc).fit(Xm_tr, ya_tr, sample_weight=sw_tr)
    m_t = [lgb.LGBMClassifier(objective="binary", **lgb_params_mc).fit(Xm_tr, yt_tr[:, j], sample_weight=sw_tr) for j in range(6)]

    lgb_oof_l[va] = m_l.predict_proba(X_meta_s[va])
    lgb_oof_a[va] = m_a.predict_proba(X_meta_s[va])
    lgb_oof_t[va] = np.column_stack([m.predict_proba(X_meta_s[va])[:, 1] for m in m_t])

    am = np.isin(adv_parent_idx, va)
    lgb_adv_l[am] = m_l.predict_proba(X_meta_adv[am])
    lgb_adv_a[am] = m_a.predict_proba(X_meta_adv[am])
    lgb_adv_t[am] = np.column_stack([m.predict_proba(X_meta_adv[am])[:, 1] for m in m_t])

lgb_pl, lgb_pt, lgb_pa = decode_consistent(lgb_oof_l, lgb_oof_t, lgb_oof_a)
lgb_apl, lgb_apt, lgb_apa = decode_consistent(lgb_adv_l, lgb_adv_t, lgb_adv_a)
res_lgb = eval_multitask(y_lbl, lgb_pl, y_tech, lgb_pt, y_act, lgb_pa, lgb_oof_l.max(1))
res_lgb_adv = eval_multitask(y_lbl[adv_parent_idx], lgb_apl, y_tech[adv_parent_idx], lgb_apt, y_act[adv_parent_idx], lgb_apa, lgb_adv_l.max(1))
print("[Stack B: LightGBM over TFIDF+Feats] Group-CV  :", {k: round(v, 2) for k, v in res_lgb.items()})
print("[Stack B: LightGBM over TFIDF+Feats] Robust-CV :", {k: round(v, 2) for k, v in res_lgb_adv.items()})

# Fit full LightGBM on synth + all 86 sample -> predict 500k
Xm_full = np.vstack([X_meta_syn, X_meta_s])
m_l_f = lgb.LGBMClassifier(objective="multiclass", num_class=6, **lgb_params_mc).fit(Xm_full, yl_full, sample_weight=sw_full)
m_a_f = lgb.LGBMClassifier(objective="multiclass", num_class=6, **lgb_params_mc).fit(Xm_full, ya_full, sample_weight=sw_full)
m_t_f = [lgb.LGBMClassifier(objective="binary", **lgb_params_mc).fit(Xm_full, yt_full[:, j], sample_weight=sw_full) for j in range(6)]

te_l_lgb = m_l_f.predict_proba(X_meta_te).astype(np.float32)
te_a_lgb = m_a_f.predict_proba(X_meta_te).astype(np.float32)
te_t_lgb = np.column_stack([m.predict_proba(X_meta_te)[:, 1] for m in m_t_f]).astype(np.float32)

np.save("preli/cache/te_lin_l.npy", te_l_lin)
np.save("preli/cache/te_lin_t.npy", te_t_lin)
np.save("preli/cache/te_lin_a.npy", te_a_lin)
np.save("preli/cache/te_lgb_l.npy", te_l_lgb)
np.save("preli/cache/te_lgb_t.npy", te_t_lgb)
np.save("preli/cache/te_lgb_a.npy", te_a_lgb)
np.savez("preli/cache/oof_base.npz",
         lin_l=oof_l_b, lin_t=oof_t_b, lin_a=oof_a_b,
         lgb_l=lgb_oof_l, lgb_t=lgb_oof_t, lgb_a=lgb_oof_a,
         adv_lin_l=adv_l_b, adv_lin_t=adv_t_b, adv_lin_a=adv_a_b,
         adv_lgb_l=lgb_adv_l, adv_lgb_t=lgb_adv_t, adv_lgb_a=lgb_adv_a,
         adv_parent_idx=adv_parent_idx)

# Build baseline verified submission_v1.csv with consistent priority/technique rules & calibrated confidence
ens_l = 0.45 * te_l_lin + 0.55 * te_l_lgb
ens_t = 0.45 * te_t_lin + 0.55 * te_t_lgb
ens_a = 0.45 * te_a_lin + 0.55 * te_a_lgb

# Calibrate confidence on OOF (simple temperature/linear shrinkage calibrated against Brier score)
oof_ens_l = 0.45 * oof_l_b + 0.55 * lgb_oof_l
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_ens_l.max(1), (oof_ens_l.argmax(1) == y_lbl).astype(float))
conf_te = np.clip(iso.predict(ens_l.max(1)), 0.05, 0.995)

pred_lbl, pred_tech_vec, pred_act = decode_consistent(ens_l, ens_t, ens_a)
pred_tech_str = [vec_to_tech_str(v) for v in pred_tech_vec]

sub_ids = pd.read_csv("preli/cache/test_clean.csv", usecols=["id"])["id"].values
sub = pd.DataFrame({
    "id": sub_ids,
    "label": pred_lbl,
    "techniques": pred_tech_str,
    "action": pred_act,
    "confidence": np.round(conf_te, 4),
})
sub.to_csv("preli/submission_v1.csv", index=False)
print("\nSAVED preli/submission_v1.csv! Counts:", sub["label"].value_counts().sort_index().to_dict())
print("Action counts:", sub["action"].value_counts().sort_index().to_dict())
