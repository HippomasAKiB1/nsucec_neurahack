import gc
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from text_clean import strip_wrappers, build_swap_fixer, canonical_key

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PREFIX_NOISE = [
    "Every day, ", "Sotti bolte, ", "Honestly, ", "Seriously, ",
    "Again, ", "Complaint: ", "Today ", "Hello, ", "Hi, ",
    "Bhai, ", "Ajke ", "আবারও, ", "দেখুন, ", "ভাই, ", "আজ "
]
SUFFIX_NOISE = [
    " এর সমাধান চাই।", " খুবই বিরক্তিকর।", " দয়া করে দেখুন।", " ধন্যবাদ।",
    " solve korun", " please fix this.", " please look into it.",
    " please dekhen", " please check", " plz fix", " not ok.",
    " very frustrating.", " eta ki thik?", " bhison kharap",
    " #dhakatransport", " dhonnobad", " thanks.", " ugh", " !!"
]

def inject_noise(text: str, rng: random.Random) -> str:
    out = text
    # Random swap on one latin word len >= 4 with p=0.35
    if rng.random() < 0.35:
        words = out.split()
        latin_idxs = [i for i, w in enumerate(words) if w.isascii() and w.isalpha() and len(w) >= 4]
        if latin_idxs:
            idx = rng.choice(latin_idxs)
            w = list(words[idx])
            pos = rng.randint(0, len(w) - 2)
            w[pos], w[pos + 1] = w[pos + 1], w[pos]
            words[idx] = "".join(w)
            out = " ".join(words)
    if rng.random() < 0.65:
        out = rng.choice(PREFIX_NOISE) + out
    if rng.random() < 0.70:
        out = out + rng.choice(SUFFIX_NOISE)
    r = rng.random()
    if r < 0.10:
        out = out.upper()
    elif r < 0.30:
        out = out.lower()
    return out

def eval_metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

@torch.no_grad()
def encode_texts(model_name: str, texts: list[str], pooling: str = "mean", prefix: str = "", batch_size: int = 32) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_name, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    inp_texts = [prefix + t for t in texts]
    embs = []
    for i in range(0, len(inp_texts), batch_size):
        batch = tok(inp_texts[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        out = mdl(**batch).last_hidden_state
        if pooling == "cls":
            emb = out[:, 0]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).type_as(out)
            emb = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
        embs.append(emb.cpu().numpy())
    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    return np.vstack(embs)

def run_cv_embeddings(X_clean, y, X_noisy_list, y_noisy_list, clf_fn, name):
    # 1. Stratified 10-fold on clean
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    oof_clean = np.zeros_like(y)
    for tr, va in skf.split(X_clean, y):
        clf = clf_fn()
        clf.fit(X_clean[tr], y[tr])
        oof_clean[va] = clf.predict(X_clean[va])
    acc10, f1_10 = eval_metrics(y, oof_clean)

    # 2. LOO-CV on clean
    loo = LeaveOneOut()
    oof_loo = np.zeros_like(y)
    for tr, va in loo.split(X_clean):
        clf = clf_fn()
        clf.fit(X_clean[tr], y[tr])
        oof_loo[va] = clf.predict(X_clean[va])
    accloo, f1_loo = eval_metrics(y, oof_loo)

    # 3. 10-fold CV evaluated on noisy OOF
    oof_noisy_preds, oof_noisy_true = [], []
    for tr, va in skf.split(X_clean, y):
        clf = clf_fn()
        clf.fit(X_clean[tr], y[tr])
        # each val idx has noisy copies
        for idx in va:
            preds = clf.predict(X_noisy_list[idx])
            oof_noisy_preds.extend(preds)
            oof_noisy_true.extend([y[idx]] * len(preds))
    acc_n, f1_n = eval_metrics(oof_noisy_true, oof_noisy_preds)
    print(f"{name:38s} | 10F: {acc10:.4f}/{f1_10:.4f} | LOO: {accloo:.4f}/{f1_loo:.4f} | NoisyOOF: {acc_n:.4f}/{f1_n:.4f}")
    return oof_loo

def main():
    s = pd.read_csv("mock/dataset/sample.csv")
    t = pd.read_csv("mock/dataset/test.csv")
    y = s["label"].values

    # Generate 5 synthetic noisy variants per sample row (for OOF robustness testing)
    rng = random.Random(SEED)
    noisy_raw_rows = []
    for _, r in s.iterrows():
        noisy_raw_rows.append([inject_noise(r["text"], rng) for _ in range(5)])

    # Denoiser fitted using only sample clean + test (unsupervised vocab frequency)
    all_clean_for_vocab = [strip_wrappers(x) for x in s["text"]] + [strip_wrappers(x) for x in t["text"]]
    fix_swaps = build_swap_fixer(all_clean_for_vocab)

    s_clean = s["text"].tolist()
    noisy_raw_flat = [x for sub in noisy_raw_rows for x in sub]
    noisy_denoised_flat = [fix_swaps(strip_wrappers(x)) for x in noisy_raw_flat]

    # TF-IDF Baseline (raw noisy vs denoised noisy)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    loo = LeaveOneOut()
    vec = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, analyzer="word")),
        ("c", TfidfVectorizer(ngram_range=(3, 5), sublinear_tf=True, analyzer="char_wb")),
    ])
    X_all_tfidf = vec.fit_transform(s_clean + noisy_raw_flat + noisy_denoised_flat)
    X_s_tf = X_all_tfidf[:60]
    X_nr_tf = X_all_tfidf[60:60+300]
    X_nd_tf = X_all_tfidf[60+300:]
    X_nr_list = [X_nr_tf[i*5:(i+1)*5] for i in range(60)]
    X_nd_list = [X_nd_tf[i*5:(i+1)*5] for i in range(60)]
    run_cv_embeddings(X_s_tf, y, X_nr_list, None, lambda: LogisticRegression(C=5.0, max_iter=1000), "TFIDF + LogReg (raw noisy)")
    run_cv_embeddings(X_s_tf, y, X_nd_list, None, lambda: LogisticRegression(C=5.0, max_iter=1000), "TFIDF + LogReg (denoised)")

    # Embedding models
    for mname, pool, pfx, short in [
        ("intfloat/multilingual-e5-small", "mean", "query: ", "mE5-small"),
        ("intfloat/multilingual-e5-base", "mean", "query: ", "mE5-base"),
        ("BAAI/bge-m3", "cls", "", "BGE-M3"),
    ]:
        embs = encode_texts(mname, s_clean + noisy_raw_flat + noisy_denoised_flat, pooling=pool, prefix=pfx)
        E_s = embs[:60]
        E_nr = [embs[60 + i*5 : 60 + (i+1)*5] for i in range(60)]
        E_nd = [embs[360 + i*5 : 360 + (i+1)*5] for i in range(60)]
        run_cv_embeddings(E_s, y, E_nr, None, lambda: LogisticRegression(C=10.0, max_iter=1000), f"{short} + LogReg(C=10) [raw]")
        run_cv_embeddings(E_s, y, E_nd, None, lambda: LogisticRegression(C=10.0, max_iter=1000), f"{short} + LogReg(C=10) [denoised]")
        run_cv_embeddings(E_s, y, E_nd, None, lambda: LinearSVC(C=2.0), f"{short} + LinearSVC(C=2) [denoised]")

    # Zero-shot NLI baseline (MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli)
    nli_name = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
    tok = AutoTokenizer.from_pretrained(nli_name, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(nli_name, local_files_only=True, dtype=torch.float16).to("cuda").eval()
    label_hyps = [
        "This message is about bus delay, waiting time, traffic jam, missed bus, breakdown, or schedule.",
        "This message is about bus fare, ticket price, overcharging, student discount, change, or payment failure.",
        "This message is about passenger safety, reckless driving, speeding, accident, harassment, pickpocketing, or crime.",
        "This message is about bus or terminal cleanliness, dirt, garbage, bad smell, vomit, toilets, or pests.",
        "This message is about staff rudeness, poor service, air conditioning off, overcrowding, skipping stops, or women's seat.",
        "This message is a route inquiry, lost item question, app or website feedback, praise, suggestion, missing sign, or form test.",
    ]
    entail_id = [k for k, v in mdl.config.id2label.items() if "entail" in v.lower()][0]

    @torch.no_grad()
    def predict_nli(texts):
        probs_all = []
        for txt in texts:
            enc = tok([txt]*6, label_hyps, padding=True, truncation=True, max_length=160, return_tensors="pt").to("cuda")
            logits = mdl(**enc).logits.float()
            # compare entailment vs contradiction or softmax entailment logits across classes
            if logits.shape[1] == 3:
                contra_id = [k for k, v in mdl.config.id2label.items() if "contra" in v.lower()][0]
                scores = logits[:, entail_id] - logits[:, contra_id]
            else:
                scores = logits[:, entail_id]
            probs_all.append(torch.softmax(scores, dim=0).cpu().numpy())
        return np.array(probs_all)

    p_clean = predict_nli(s_clean)
    p_nraw = predict_nli(noisy_raw_flat)
    p_nden = predict_nli(noisy_denoised_flat)
    acc_c, f1_c = eval_metrics(y, p_clean.argmax(1))
    y_n = np.repeat(y, 5)
    acc_nr, f1_nr = eval_metrics(y_n, p_nraw.argmax(1))
    acc_nd, f1_nd = eval_metrics(y_n, p_nden.argmax(1))
    print(f"{'ZeroShot mMiniLM-NLI [clean/raw/den]':38s} | Clean: {acc_c:.4f}/{f1_c:.4f} | NoisyRaw: {acc_nr:.4f}/{f1_nr:.4f} | NoisyDen: {acc_nd:.4f}/{f1_nd:.4f}")
    del mdl; gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
