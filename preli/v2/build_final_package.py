"""Assemble self-contained, executable Section 9.2 notebook.ipynb, execute it, and pack NeuraHack_Preli_Turtlers.zip."""
import ast
import hashlib
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(r"d:\Datathon\nsucec_neurahackl")
V2 = ROOT / "preli" / "v2"
PKG = ROOT / "final_package_turtlers"
PKG.mkdir(parents=True, exist_ok=True)


def extract_clean_body(
    py_path: Path,
    main_rename: str | None = None,
    drop_prefixes=(),
    extra_replacements=(),
) -> str:
    txt = py_path.read_text(encoding="utf-8")
    lines = txt.splitlines()
    out = []
    skip_block = False
    in_local_import_paren = False

    local_mods = {
        "v2.cv_ood", "cv_ood", "v2.normalizer_v4", "normalizer_v4",
        "v2.rules_v4", "rules_v4", "v2.synth_generator_v4", "synth_generator_v4",
        "torch_utils_v4", "train_tfidf_gbdt_v4", "train_da_bigru_v4",
        "train_teachers_v4", "train_students_v4", "train_meta_calibrate_v4",
        "build_v8_hotfix",
    }

    for i, ln in enumerate(lines):
        s = ln.strip()
        if in_local_import_paren:
            if ")" in s:
                in_local_import_paren = False
            continue
        if s.startswith("if __name__ == "):
            break
        if any(s.startswith(p) for p in drop_prefixes):
            continue
        # Drop boilerplate module path setup and duplicate optional GBDT imports already guarded in Cell 1
        if s in ("import lightgbm as lgb", "import xgboost as xgb"):
            continue
        if any(s.startswith(k) for k in ("ROOT =", "V2_DIR =", "CACHE_DIR =", "CKPT_DIR =", "OUT_DIR =", "sys.path.insert", "CACHE_DIR.mkdir", "CKPT_DIR.mkdir", "OUT_DIR.mkdir", "CACHE_DIR_HF =")):
            continue
        # Drop try/except blocks wrapping local v2 imports
        if s in ("try:", "except ImportError:") and i + 1 < len(lines) and ("from v2." in lines[i + 1] or "sys.path.insert" in lines[i + 1] or "from normalizer_v4" in lines[i + 1]):
            skip_block = True
            continue
        if skip_block:
            if s.startswith(("from ", "import ", "sys.path", "except ")):
                continue
            if ln.startswith(("    ", "\t")):
                continue
            skip_block = False

        # Drop single or multiline imports of local modules already merged into notebook
        if s.startswith("from ") and any(s.startswith(f"from {m} import") for m in local_mods):
            if "(" in s and ")" not in s:
                in_local_import_paren = True
            continue

        if main_rename and re.match(r"^def main\s*\(\):", ln):
            ln = f"def {main_rename}():"

        ln = ln.replace("local_files_only=True", "local_files_only=HF_LOCAL_ONLY")
        if "ap.parse_args()" in ln and "args = ap.parse_args()" in ln and "out_csv_default" not in str(extra_replacements):
            ln = ln.replace("ap.parse_args()", "ap.parse_args([])")

        for old, new in extra_replacements:
            ln = ln.replace(old, new)
        out.append(ln)

    body = "\n".join(out).strip()
    return body


def main():
    # 1. Verify & copy submission.csv cleanly
    src_sub = V2 / "submission_v9_final.csv"
    dst_sub = PKG / "submission.csv"
    shutil.copyfile(src_sub, dst_sub)

    with open(src_sub, "rb") as f:
        h1 = hashlib.sha256(f.read()).hexdigest()
    with open(dst_sub, "rb") as f:
        h2 = hashlib.sha256(f.read()).hexdigest()
    assert h1 == h2, "submission.csv SHA-256 mismatch!"

    nb = nbf.v4.new_notebook()
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(src):
        ast.parse(src)
        cells.append(nbf.v4.new_code_cell(src.strip()))

    md("""# NSUCEC NeuraHack 2026 Preliminary Round — ScamShield BD

**Team:** `Turtlers` — Akib Hasan Pyil, K M Mubin, Abdullah Omar Faruk  
**Final Selected Kaggle Submission:** `submission_v9_final.csv` (**Public Score: `0.93661`**; second selection: `submission_v8_hotfix.csv`, **`0.93552`**)  

This notebook is a **100% self-contained, reproducible pipeline** implementing adversarial multilingual normalization, guide-priority rules (`1 > 2 > 4 > 3 > 5 > 0`), programmatic scenario generation (`72,000` conversations across `48` scenario families), scenario-disjoint CV, base teacher consensus domain adaptation (`mDeBERTa-v3-base` + `mE5-base`), diverse student distillation (`mMiniLM`, `mE5-small`, glossed `BGE-small`, `BiGRU-v5`, `DA-Tab LGBM/XGB/CatBoost`), selective `mDeBERTa-full` escalation, zero-leakage meta-calibration, and final `v8/v9` obfuscation refinement.

> **Execution Mode:** By default, `RUN_FULL_TRAINING = False` executes full EDA, normalization demonstrations, rule verification on `sample.csv`, cached OOD model evaluation, and strict schema validation on `submission.csv` (`~30s`). Set `RUN_FULL_TRAINING = True` to re-run end-to-end training & inference from scratch.""")

    md("""## 1. Environment Configuration, Declarations & Reproducibility Seeds (Sections 14–15)

### Mandatory External Resource & Compliance Declarations
- **Datasets**: Only the competition dataset (`sample.csv`, `test.csv`, `sample_submission.csv`) is used. No external labeled datasets are imported.
- **Programmatic Augmentation (Section 14)**: Synthetic data generator `generate_dataset_v4` (`seed=4242`) creates `72,000` labeled scenarios across `48` scenario families. Unlabeled `test.csv` text is hashed into `285,810` canonical structural skeletons; `45,000` cluster prototypes are scored by 2-fold scenario-disjoint teachers and consensus-filtered (`39,618` rows, `mean_conf=0.8201`) for domain adaptation. At inference time (`predict_inductive_v4`), predictions run **strictly inductively** per conversation without test-cluster lookup.
- **Open-Weight Pretrained Models (Hugging Face, MIT License)**:
  1. `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
  2. `intfloat/multilingual-e5-base`
  3. `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`
  4. `intfloat/multilingual-e5-small`
  5. `BAAI/bge-small-en-v1.5`
- **External Hosted APIs**: None (`0` API calls; `.env` is intentionally empty).
- **Random Seeds**: `4242` (synth), `42`/`123`/`2026` (SVD/GBDT/meta), `101+f` & `201+f` (teachers), `70+f`/`909` (BiGRU), `300+f`/`500`/`777` (students & full mDeBERTa).""")

    code("""import os
import re
import gc
import sys
import math
import time
import json
import random
import hashlib
import warnings
import unicodedata
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import hstack, vstack, csr_matrix
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, brier_score_loss, confusion_matrix
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from catboost import CatBoostClassifier
    HAS_CB = True
except ImportError:
    HAS_CB = False
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_cosine_schedule_with_warmup

# Reproducibility toggle: False runs EDA, rule/OOD audits & submission validation cleanly in <1 min
RUN_FULL_TRAINING = False

def _find_data_dir():
    for cand in [
        Path("."),
        Path("./dataset"),
        Path("./Main Preli Round/dataset"),
        Path("../Main Preli Round/dataset"),
        Path("/kaggle/input/nsucec-neurahack-2026"),
        Path("/kaggle/input/scamshield-bd"),
    ]:
        if (cand / "sample.csv").exists() and (cand / "test.csv").exists():
            return cand.resolve()
    raise FileNotFoundError("Could not locate sample.csv and test.csv!")

DATA_DIR = _find_data_dir()
WORK_DIR = Path.cwd()
ROOT = DATA_DIR.parent.parent if DATA_DIR.name == "dataset" and DATA_DIR.parent.name == "Main Preli Round" else WORK_DIR

# When inspecting existing workspace verification caches without overwriting package folder:
EXISTING_V2_CACHE = ROOT / "preli" / "v2" / "cache"
if (WORK_DIR / "cache").exists() or not EXISTING_V2_CACHE.exists():
    CACHE_DIR = WORK_DIR / "cache"
else:
    CACHE_DIR = EXISTING_V2_CACHE

CKPT_DIR = CACHE_DIR / "ckpt_v4"
OUT_DIR = CACHE_DIR / "preds_v4"
V2_DIR = WORK_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR_HF = os.environ.get("HF_HOME", r"D:\\Datathon\\hf_cache" if Path(r"D:\\Datathon\\hf_cache").exists() else str(WORK_DIR / "hf_cache"))
HF_LOCAL_ONLY = os.environ.get("HF_LOCAL_ONLY", "0") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
XGB_DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | Device: {DEV} | HF_LOCAL_ONLY={HF_LOCAL_ONLY}")
print(f"Transformers: {__import__('transformers').__version__} | LightGBM: {lgb.__version__ if lgb else 'optional(CPU-env)'} | XGBoost: {xgb.__version__ if xgb else 'optional(CPU-env)'}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"CACHE_DIR: {CACHE_DIR}")
""")

    md("""## 2. Exploratory Data Analysis (Executed)

We inspect both `sample.csv` (`86` labeled records) and `test.csv` (`500,000` unlabeled records) across class distributions, multi-turn `|||` conversation density, Bangla script vs Banglish/English usage, and adversarial corruption vectors.""")

    code("""sample_df = pd.read_csv(DATA_DIR / "sample.csv")
test_raw_df = pd.read_csv(DATA_DIR / "test.csv", usecols=["id", "text"])
raw_tx = test_raw_df["text"].fillna("").astype(str)

print(f"sample.csv: {len(sample_df)} rows | label counts: {sample_df['label'].value_counts().sort_index().to_dict()}")
print(f"sample multi-turn (|||): {int(sample_df['text'].str.contains(r'\\|\\|\\|').sum())} / {len(sample_df)}")
print(f"test.csv:   {len(test_raw_df):,} rows")
print(f"  Multi-turn (|||):        {int(raw_tx.str.contains(r'\\|\\|\\|').sum()):,} ({100*raw_tx.str.contains(r'\\|\\|\\|').mean():.2f}%)")
print(f"  Bangla script (U+0980):  {int(raw_tx.str.contains(r'[\\u0980-\\u09FF]').sum()):,} ({100*raw_tx.str.contains(r'[\\u0980-\\u09FF]').mean():.2f}%)")
print(f"  Zero-width obfuscation:  {int(raw_tx.str.contains(r'[\\u200b-\\u200f\\u2060\\ufeff\\ufe0f\\u00ad]').sum()):,} ({100*raw_tx.str.contains(r'[\\u200b-\\u200f\\u2060\\ufeff\\ufe0f\\u00ad]').mean():.2f}%)")
print(f"  Cyrillic/Greek homoglyph:{int(raw_tx.str.contains(r'[\\u0400-\\u04ff\\u0370-\\u03ff]').sum()):,} ({100*raw_tx.str.contains(r'[\\u0400-\\u04ff\\u0370-\\u03ff]').mean():.2f}%)")
lens = raw_tx.str.len()
print(f"  Char lengths -> min={int(lens.min())}, median={int(lens.median())}, p95={int(lens.quantile(0.95))}, max={int(lens.max())}")
""")

    md("""## 3. Multilingual Normalization, Domain Masking & Feature Extraction (`normalizer_v4`)

Cleans zero-width joiners, homoglyphs, noise tokens, character floods, leetspeak, un-defangs URLs (`hxxps://`, `[.]`), preserves semantic emoji tokens (`<EMOJI_GIFT>`), prevents URL domain poisoning (`mask_urls_for_text_rules`), transliterates/glosses Bangla (`bn_to_en_roman_v4`), formats turn structure (`format_turn_aware`), and extracts `24` domain-general structural features (`extract_base_struct_v4`).""")

    code(extract_clean_body(V2 / "normalizer_v4.py"))

    code("""# Live adversarial normalization demonstration (raw obfuscated -> normalized clean -> rule/BGE gloss)
demo_msgs = [
    "mnbvcx Your bKash PIIIN will expire today Rpely witttth your currentP1N ad a new PIN to 💰 contiune service",
    "866970 is y\\u200bournagad lоgіncode don07 share it w\\u200bith anyoe even mnbvcx na\\u200bgad 5taff",
    "khala ami,pohne hariyeche 💰 Tomr nubmer e je cooodeaschhe oa bOLo",
    "আপনার পার্সেল আটকে আছে। ডেলিভারি চার্জ ১৫০ টাকা পরিশোধ করতে hxxps://tap-track[.]site/7be4 এ ক্লিক করুন",
]
for i, msg in enumerate(demo_msgs, 1):
    c = clean_for_model(msg)
    m, u = mask_urls_for_text_rules(c)
    print(f"[{i}] CLEAN: {c}")
    if m != c:
        print(f"    MASKD: {m}")
""")

    md("""## 4. Scenario-Disjoint Validation Splits & Section 10 Metric (`cv_ood`)

Implements consistent post-decoding (`action=0` & `NONE` iff `label in {0, 5}`), coarse scenario-family assignment, balanced 3-fold scenario-disjoint partitioning (`get_balanced_3folds` retaining 67% of every class per fold), and `score_section10_proxy`.""")

    code(extract_clean_body(V2 / "cv_ood.py"))

    md("""## 5. Guide-Aligned Rule Teacher & 58-Dim Feature Vector (`rules_v4` + Live Verification)

Implements strict Problem Statement Section 4 priority ordering (`1 > 2 > 4 > 3 > 5 > 0`), multi-clause negation-aware credential harvest checks, hard-legitimate guards (safe OTP warnings, COD cash notices, balance receipts, HR interview letters), and validates untuned accuracy directly on `sample.csv` (`86` labeled records).""")

    code(extract_clean_body(V2 / "rules_v4.py"))

    code("""# Live validation of Guide-aligned rules_v4 on labeled sample.csv (86 rows)
r_pl, r_pt, r_pa = [], [], []
for _, row in sample_df.iterrows():
    l, tv, a, _ = extract_rule_prediction(row["text"])
    r_pl.append(l); r_pt.append(tv); r_pa.append(a)
r_pl, r_pt, r_pa = np.array(r_pl), np.vstack(r_pt), np.array(r_pa)
y_st = np.vstack([tech_str_to_vec(x) for x in sample_df["techniques"]])
res_r4 = score_section10_proxy(
    sample_df["label"].values, r_pl, y_st, r_pt, sample_df["action"].values, r_pa,
    np.where(r_pl == sample_df["label"].values, 0.95, 0.25),
)
print("rules_v4 on sample.csv:", res_r4)
""")

    md("""## 6. Programmatic Synthetic Generator (`72,000` Rows Across `48` Families) & Caching (`synth_generator_v4`, `prepare_v4_cache`)

Generates balanced multilingual/code-mixed single & multi-turn conversations with realistic adversarial noise/homoglyph corruption (`corrupt_prob=0.68`, `seed=4242`) and hashes `test.csv` into structural clusters (`test_skel_clusters.npz`).""")

    code(extract_clean_body(V2 / "synth_generator_v4.py"))
    prep_src = extract_clean_body(
        V2 / "prepare_v4_cache.py",
        main_rename="run_prepare_cache",
        extra_replacements=[
            ('ROOT / "Main Preli Round/dataset/sample.csv"', 'DATA_DIR / "sample.csv"'),
            ('ROOT / "Main Preli Round/dataset/test.csv"', 'DATA_DIR / "test.csv"'),
        ],
    )
    prep_src = prep_src.replace(
        """    with ProcessPoolExecutor(max_workers=workers) as ex:
        for c, f, l, tv, a, s in ex.map(_worker_chunk, chunks):
            all_c.extend(c)
            all_f.append(f)
            all_l.append(l)
            all_t.append(tv)
            all_a.append(a)
            all_s.append(s)""",
        """    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for c, f, l, tv, a, s in ex.map(_worker_chunk, chunks):
                all_c.extend(c); all_f.append(f); all_l.append(l); all_t.append(tv); all_a.append(a); all_s.append(s)
    except Exception:
        for ch in chunks:
            c, f, l, tv, a, s = _worker_chunk(ch)
            all_c.extend(c); all_f.append(f); all_l.append(l); all_t.append(tv); all_a.append(a); all_s.append(s)""",
    )
    code(prep_src)

    code("""# Quick programmatic verification of synthetic generator diversity (small deterministic sample)
sy_preview = generate_dataset_v4(n_samples=600, seed=4242, corrupt_prob=0.68)
print(f"Generated preview rows={len(sy_preview)} | distinct scenario families={sy_preview['scenario_family'].nunique()}")
print("Preview label distribution:", sy_preview["label"].value_counts().sort_index().to_dict())
print("3-fold scenario-disjoint split counts:", pd.Series(get_balanced_3folds(sy_preview["scenario_family"].values)).value_counts().sort_index().to_dict())
""")

    md("""## 7. Multi-Task Neural & Tabular Pipeline (`torch_utils_v4`, `TFIDF+GBDT`, `BiGRU-v5`, `Base Teachers`, `Domain Consensus`, `Students`, `DA-Tab`)

- **`HeadConditionedTransformer`**: Pretrained backbone (`mDeBERTa-v3-base`, `mE5-base`, `mMiniLM`, `mE5-small`, `BGE-small`) concatenated with LayerNorm-scaled `58`-dim structural/rule features, 4x multi-sample dropout, and detached Task-A probabilities conditioning Task B (`techniques`) and Task C (`action`).
- **`BPEBiGRUNet`**: 32k-bucket hashed subword 2-layer BiGRU + Attention/Max pooling for diverse sequence inductive bias.
- **`fit_gbdt_multitask`**: Multi-class `LightGBM + XGBoost + CatBoost` for Task A + per-technique/action LightGBM heads.""")

    code(extract_clean_body(V2 / "torch_utils_v4.py"))
    code(extract_clean_body(
        V2 / "train_tfidf_gbdt_v4.py",
        main_rename="run_baseline_v5",
        extra_replacements=[
            ('ROOT / "Main Preli Round/dataset/sample_submission.csv"', 'DATA_DIR / "sample_submission.csv"'),
            ('V2_DIR / "sub_v6b_v5.csv"', 'WORK_DIR / "sub_v6b_v5.csv"'),
        ],
    ))
    teach_src = extract_clean_body(
        V2 / "train_teachers_v4.py",
        main_rename="run_teachers_v4",
        extra_replacements=[
            ('device="cuda"', 'device=DEV'),
            (".cuda()", ".to(DEV)"),
            ("from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup", "from transformers import AutoTokenizer, get_cosine_schedule_with_warmup"),
        ],
    )
    # Remove unused experimental Qwen helper block (never invoked; --qwen-max was 0 in final pipeline)
    teach_src = re.sub(r"def run_qwen_tiebreaker\(.*?\n\s*return q_lbl, q_act\n*", "", teach_src, flags=re.S)
    teach_src = re.sub(r"    if args\.qwen_max > 0:.*?p_cons_a\[li, 1:\] /= p_cons_a\[li, 1:\]\.sum\(\)\n*", "", teach_src, flags=re.S)
    code(teach_src)
    code(extract_clean_body(V2 / "build_clean_pseudo_v5.py", main_rename="run_build_clean_pseudo_v5"))
    code(extract_clean_body(
        V2 / "train_students_v4.py",
        main_rename="run_students_v4",
        extra_replacements=[
            ('device="cuda"', 'device=DEV'),
            ('device = "cuda"', 'device = DEV'),
            (".cuda()", ".to(DEV)"),
            ("def compute_weights(", "def compute_student_weights("),
            ("compute_weights(", "compute_student_weights("),
        ],
    ))
    code(extract_clean_body(
        V2 / "train_da_bigru_v4.py",
        main_rename="run_da_bigru_v4",
        extra_replacements=[
            ("from train_students_v4 import build_distill_corpus", ""),
            ('device="cuda"', 'device=DEV'),
            (".cuda()", ".to(DEV)"),
        ],
    ))
    code(extract_clean_body(V2 / "train_da_tab_v4.py", main_rename="run_da_tab_v4"))

    md("""## 8. Zero-Leakage Meta-Calibration, `v8/v9` Adversarial Refinement & Purely Inductive Inference

1. **`run_meta_calibrate_v4`**: Optimizes separate Task-A (`W_L`), Task-B (`W_T`), Task-C (`W_A`) blend weights across all 7 full-coverage families, integrates selective 26k `mDeBERTa-full` escalation, fits a Logistic uncertainty/Brier calibrator on scenario-disjoint OOF folds, and exports `submission_v7_top10.csv`.
2. **`clean_v8` / `clean_v9` + `run_v8_hotfix` + `run_v9_final`**: Repairs newly discovered adversarial failure modes in `test.csv` (removing `"please do not reply"` from `NEG_WARN_RE`, separating `"clearance charge"` prize/job scams from parcel scams, capturing `"select hoyechen"`/`"AnyDesk"`/`"card number + CVV"`/concatenated OTP tokens, removing extreme sample-86 `alpha`/`T1` distortion, re-scoring cleaned rows with `mMiniLM`, and smoothing Brier confidence in `[0.25, 0.985]`).
3. **`run_inductive_inference`**: Strictly inductive raw-text end-to-end inference pipeline satisfying Section 7.3 technical verification.""")

    code(extract_clean_body(
        V2 / "train_meta_calibrate_v4.py",
        main_rename="run_meta_calibrate_v4",
        extra_replacements=[
            ("ap = argparse.ArgumentParser()", ""),
            ("ap.add_argument", "# ap.add_argument"),
            ("args = ap.parse_args()", "out_csv_default = V2_DIR / 'submission_v7_top10.csv'"),
            ("Path(args.out)", "out_csv_default"),
        ],
    ))
    code(extract_clean_body(
        V2 / "build_v8_hotfix.py",
        main_rename="run_v8_hotfix",
        extra_replacements=[
            ('ROOT / "Main Preli Round" / "dataset" / "sample.csv"', 'DATA_DIR / "sample.csv"'),
            ('ROOT / "Main Preli Round" / "dataset" / "test.csv"', 'DATA_DIR / "test.csv"'),
            (".cuda()", ".to(DEV)"),
            ("bs=112, max_len=144)", "bs=112, max_len=144, device=DEV)"),
        ],
    ))
    code(extract_clean_body(
        V2 / "build_v9_final.py",
        main_rename="run_v9_final",
        extra_replacements=[
            ('ROOT / "Main Preli Round" / "dataset" / "test.csv"', 'DATA_DIR / "test.csv"'),
            ('V2_DIR / "submission_v9_final.csv"', 'WORK_DIR / "submission.csv"'),
        ],
    ))
    code("""# Live V9 adversarial repair comparison (raw -> clean_for_model v4 -> final clean_v9)
adv_examples = [
    "mnbvcx Your bKash PIIIN will expire today Rpely witttth your currentP1N ad a new PIN to contiune service",
    "866970 is y\\u200bournagad lоgіncode don07 share it w\\u200bith anyoe even mnbvcx na\\u200bgad 5taff",
    "khala ami,pohne hariyeche 💰 Tomr nubmer e je cooodeaschhe oa bOLo",
    "You have been selected for a a laptop reward. Pay Tk 3,500 clearance charge to +880 015XXXXXX86.",
]
for idx_ex, ex in enumerate(adv_examples, 1):
    print(f"[{idx_ex}] v4: {clean_for_model(ex)}")
    print(f"    v9: {clean_v9(ex)}")
""")
    code(extract_clean_body(
        V2 / "predict_inductive_v4.py",
        main_rename=None,
        extra_replacements=[
            ("DEV = ", "_dev_unused = "),
        ],
    ))

    code("""# Master end-to-end execution guard (set RUN_FULL_TRAINING=True to reproduce everything from scratch)
if RUN_FULL_TRAINING:
    run_prepare_cache()
    run_baseline_v5()
    run_teachers_v4()
    run_build_clean_pseudo_v5()
    run_da_bigru_v4()
    run_da_tab_v4()
    run_students_v4()
    run_meta_calibrate_v4()
    run_v8_hotfix()
    run_v9_final()
else:
    print("RUN_FULL_TRAINING=False -> Using verified checkpoints/predictions for audit & validation cells below.")
""")

    md("""## 9. Validation Benchmarks, Error Analysis & Final Verification (Executed)

### 9.1 Zero-Leakage Scenario-Disjoint Validation Summary
Evaluates saved OOD predictions (`oof_*.npz`) if local caches are present, and demonstrates why multi-task Neural models complement TF-IDF/GBDT cleanly (`BiGRU-v5` & Transformers gain `+6` to `+18` F1 on Task B `techniques` and Task C `action`, while `rules` + GBDT stabilize Task A primary classes).""")

    code("""oof_ba_p = OUT_DIR / "oof_baseline_v4.npz"
oof_bg_p = OUT_DIR / "oof_bigru_v4.npz"
oof_st_p = OUT_DIR / "oof_students_v4.npz"
s_cache_p = CACHE_DIR / "sample_v4.csv"

if all(p.exists() for p in [oof_ba_p, oof_bg_p, oof_st_p, s_cache_p]):
    sdf_v4 = pd.read_csv(s_cache_p)
    ys_l = sdf_v4["label"].values.astype(int)
    ys_t = np.vstack([tech_str_to_vec(x) for x in sdf_v4["techniques"]])
    ys_a = sdf_v4["action"].values.astype(int)
    ba = np.load(oof_ba_p); bg = np.load(oof_bg_p); st = np.load(oof_st_p)
    rows_eval = []
    for nm, pl, pt, pa in [
        ("TF-IDF Linear (OOD)", ba["s_lin_l"], ba["s_lin_t"], ba["s_lin_a"]),
        ("SVD+GBDT Stack (OOD)", ba["s_gb_l"], ba["s_gb_t"], ba["s_gb_a"]),
        ("BiGRU-v5 (3-Fold OOD)", bg["s_l"], bg["s_t"], bg["s_a"]),
        ("mMiniLM-v5 (3-Fold OOD)", st["s_mminilm_l"], st["s_mminilm_t"], st["s_mminilm_a"]),
        ("mE5-small-v5 (3-Fold OOD)", st["s_me5small_l"], st["s_me5small_t"], st["s_me5small_a"]),
        ("BGE-small-v5 (3-Fold OOD)", st["s_bge_l"], st["s_bge_t"], st["s_bge_a"]),
    ]:
        dl, dt, da = decode_consistent(pl, pt, pa)
        sc = score_section10_proxy(ys_l, dl, ys_t, dt, ys_a, da, pl.max(1))
        rows_eval.append({"Model": nm, **sc})
    print(pd.DataFrame(rows_eval)[["Model", "Composite", "F_A_primary", "F_B_tech", "F_C_action", "Brier", "Acc_primary"]].to_string(index=False))

    # Optional inductive end-to-end check on sample.csv if all model weights & GBDT libs are present locally
    if lgb is not None and xgb is not None and (CKPT_DIR / "meta_calibrator_v4.joblib").exists() and (CKPT_DIR / "mdeberta_full_distill.pt").exists():
        ind_tmp = WORK_DIR / "_tmp_inductive_sample.csv"
        run_inductive_inference(DATA_DIR / "sample.csv", ind_tmp)
        chk = pd.read_csv(ind_tmp)
        ind_sc = score_section10_proxy(
            sample_df["label"].values, chk["label"].values,
            np.vstack([tech_str_to_vec(x) for x in sample_df["techniques"]]),
            np.vstack([tech_str_to_vec(x) for x in chk["techniques"]]),
            sample_df["action"].values, chk["action"].values, chk["confidence"].values,
        )
        print("\\nPurely Inductive End-to-End Check on sample.csv (Section 7.3):", ind_sc)
        if ind_tmp.exists():
            ind_tmp.unlink()
else:
    print("Cached OOD arrays not found in standalone directory — skipped cached OOD table.")
""")

    md("""### 9.2 Error Analysis, Adversarial Normalization & Limitations

1. **Class Confusion & Rule Priority Audit**: Early rules misclassified *"selected for a laptop/Smart TV reward + clearance charge"* and *"work from home job + clearance charge"* as Class 4 (Parcel) because `clearance charge` sat inside `PARCEL_RE`; restricting `PARCEL_RE` to true parcel/courier/customs context restored Classes 2 and 3. Similarly, `"please do not reply"` sat inside `NEG_WARN_RE`, allowing scam footers to mask deceptive requests (`idx=7050`).
2. **Adversarial & Multilingual Robustness**: `test.csv` systematically combines zero-width joiners (`54,310` rows), Cyrillic/Greek confusables (`42,538` rows), random noise tokens (`asdkjh`, `wkrjfejejrn`, `mnbvcx`, `zzqwpo`), leetspeak (`b4nk`, `don07`, `s3cur1ty`), character repeats (`taaaakadaly`), and token merging (`sharethe`, `coodeaschhe`, `cvvdin`, `dibenna`). Adding `clean_v8`/`clean_v9` and re-scoring repaired rows with `mMiniLM` lifted Kaggle Public LB from `0.93177 -> 0.93552 -> 0.93661`.
3. **Hard Legitimate Separation**: Genuine OTP warnings (`"866970 is your nagad login code don07 share it with anyone"` or `"Standard Chrtered: PIN, OTP ba password kauke dibenna, b4nk er lokeo na"`), COD delivery notices, transaction alerts (`TrxID`), and friend scam-warning threads (`"someone from ... messaged me ... eta scam"`) contain high-overlap scam vocabulary and are protected through explicit negation-window scoping and contextual features.
4. **Justification & Honest Limitations**: Under a strict 4 GB VRAM single-GPU constraint, 2-fold teachers with aggressive domain holdout initially experienced scenario blind spots (`~74.9` OOD), motivating balanced 3-fold OOD splits (`67%` of all classes retained per fold), consensus prototype filtering (`39,618 / 45,000`), compact distilled Transformer students, and selective `26k` `mDeBERTa` escalation. Future improvements would expand synthetic multi-turn/adversarial coverage and distill `mMiniLM`/`mE5-small` directly on `clean_v9` text.""")

    md("""### 9.3 Final Submission Verification (`submission.csv`, Executed)""")

    ver_body = extract_clean_body(
        ROOT / "preli" / "verify_submission.py",
        extra_replacements=[
            ('sample_sub_path: str = "Main Preli Round/dataset/sample_submission.csv"', 'sample_sub_path: str = str(DATA_DIR / "sample_submission.csv")'),
        ],
    )
    code(ver_body + """


sub_target = WORK_DIR / "submission.csv"
verify_submission(str(sub_target), str(DATA_DIR / "sample_submission.csv"))
raw_tx_check = test_raw_df["text"].fillna("").astype(str)
sub_df_check = pd.read_csv(sub_target)
print("\\nFinal submission SHA-256:", hashlib.sha256(sub_target.read_bytes()).hexdigest())
for name, pat in [
    ("Bangla script", r"[\\u0980-\\u09FF]"),
    ("Multi-turn |||", r"\\|\\|\\|"),
    ("Zero-width", r"[\\u200b-\\u200f\\u2060\\ufeff\\ufe0f\\u00ad]"),
    ("Cyrillic/Greek", r"[\\u0400-\\u04ff\\u0370-\\u03ff]"),
]:
    m = raw_tx_check.str.contains(pat, regex=True).values
    print(f"  Subset {name:15s} (n={int(m.sum()):6d}) labels: {np.bincount(sub_df_check['label'].values[m], minlength=6).tolist()}")
""")

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }

    nb_path = PKG / "notebook.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {nb_path} with {len(cells)} cells. Executing end-to-end with nbconvert...")

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(nb_path, encoding="utf-8") as f:
        nb_loaded = nbf.read(f, as_version=4)
    ep.preprocess(nb_loaded, {"metadata": {"path": str(PKG)}})
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb_loaded, f)
    print("Notebook execution succeeded cleanly with all outputs rendered!")

    # Ensure strictly empty 0-byte .env file (no API keys used)
    (PKG / ".env").write_bytes(b"")

    # Pack exact 4 required files into NeuraHack_Preli_Turtlers.zip
    zip_path = ROOT / "NeuraHack_Preli_Turtlers.zip"
    req_files = ["submission.csv", "notebook.ipynb", "README.md", ".env"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in req_files:
            p = PKG / fn
            assert p.exists(), f"Missing required package file: {p}"
            zf.write(p, arcname=fn)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(zf.namelist())
        assert names == sorted(req_files), f"ZIP contents mismatch: {names}"
        for info in zf.infolist():
            print(f"  ZIP entry: {info.filename:18s} size={info.file_size:10d} compressed={info.compress_size:10d}")
    print(f"Created verified {zip_path} ({zip_path.stat().st_size:,} bytes)!")


if __name__ == "__main__":
    main()
