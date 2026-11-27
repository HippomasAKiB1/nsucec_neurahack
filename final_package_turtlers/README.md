# NSUCEC NeuraHack 2026 Preliminary Round — ScamShield BD

**Team Name:** `Turtlers`  
**Members:** Akib Hasan Pyil, K M Mubin, Abdullah Omar Faruk  
**Final Selected Submission (`submission.csv`):** `submission_v9_final.csv` — **Kaggle Public Score: `0.93661`** (`submission_v8_hotfix.csv` at **`0.93552`**)  
**SHA-256 (`submission.csv`):** `1c9206daeecbd8a19004d1bbeadaa04ca86945e5fd5c6bfb358987c5a87b0989`

---

## 1. Approach Summary

Our solution treats **ScamShield BD** as a tightly coupled **adversarial multilingual NLP + policy-priority multi-task problem** across English, Bangla (`\u0980-\u09FF`), Banglish, and multi-turn `|||` conversations (up to 12 exchanges), jointly predicting:
- **Task A (`label`)**: 6-class priority classification (`1 > 2 > 4 > 3 > 5 > 0`)
- **Task B (`techniques`)**: 6-bit multi-label tactic tagging (`T1..T6`, or `NONE` when `label in {0, 5}`)
- **Task C (`action`)**: 6-class most consequential victim action (`0..5`, `0` iff `label in {0, 5}`)
- **Task D (`confidence`)**: Brier-calibrated prediction confidence in `(0, 1]`

### Key Architectural Pillars
1. **Adversarial Normalization & Turn-Aware Representation (`normalizer_v4` + `clean_v8/v9`)**:
   - Strips zero-width obfuscation characters (`\u200b-\u200f`, `\u2060`, `\ufeff`, `\ufe0f`, `\u00ad` — present in `54,310` test rows) and Cyrillic/Greek homoglyphs (`42,538` test rows).
   - Removes inserted adversarial noise strings (`wkrjfejejrn`, `asdkjh`, `zzqwpo`, `xkcdpl`, `mnbvcx`, `qzxvbn`, etc.), repairs leetspeak (`b4nk`, `don07`, `s3cur1ty`, `r3del1v3ry`), splits concatenated words (`sharethe`, `coodeaschhe`, `cradnumber`, `cvvdin`), collapses character floods (`3+` repeats), un-defangs `hxxp`/`[.]` URLs, and masks untrusted raw domains (`<URL_SUSP>/<URL>`) to avoid domain-token shortcuts.
   - Generates a Bengali-to-Banglish/English semantic gloss (`bn_to_en_roman_v4`) for English-centric representations (`BGE-small`) and extracts `58` domain-engineered structural and rule features (`24` structural + `34` policy indicators).
2. **Programmatic Synthetic Scenario Generation & Zero-Leakage Validation (`synth_generator_v4` + `cv_ood`)**:
   - Because `sample.csv` contains only `86` labeled rows, we programmatically generated **`72,000` diverse synthetic conversations across `48` distinct scenario families** (English/Bangla/Banglish, multi-turn narrative shifts, hard legitimate notices, adversarial corruptions).
   - All model selection, meta-stacking, and calibration were validated strictly using **Scenario-Disjoint Balanced 3-Fold CV (`get_balanced_3folds`)** and **Leave-One-Scenario-Out (`LOSO`)** exclusion so zero template skeletons overlap between train and validation.
3. **Diverse 7-Family Student Stack + Consensus Domain Adaptation + Selective Escalation**:
   - **Tabular / Lexical Families**: Word (1–2g) + Char (3–5g) TF-IDF `LogisticRegression` and SVD-reduced Multi-Task `LightGBM + XGBoost + CatBoost` (both synthetic/sample OOD baseline and domain-adapted `da_tab_v4`).
   - **Non-Transformer Sequence Family**: Subword-hashed 2-layer Bidirectional GRU with Attention+Max pooling and head-conditioned Task B/C heads (`BPEBiGRUNet`, `BiGRU-v5`).
   - **Knowledge Distillation from Base Teachers (`mDeBERTa-v3-base` + `mE5-base`)**: 2-fold scenario-disjoint base teachers scored `45,000` canonical cluster prototypes in `test.csv` text; consensus filtering (`build_clean_pseudo_v5`) retained **`39,618 / 45,000` (`88.0%`, mean consensus confidence `0.8201`)** ultra-clean domain prototypes.
   - **Three Diverse Transformer Students (`HeadConditionedTransformer`)**: `mMiniLMv2-L6-mnli-xnli`, `multilingual-e5-small`, and glossed `bge-small-en-v1.5`, each projecting `[pooled_text; LayerNorm(feats_58)]` into shared Task A logits that feed detached probabilities into Task B and Task C heads.
   - **Selective Inductive `mDeBERTa-v3-base` Escalation**: Full-scenario distilled `mDeBERTa-v3-base` (`freeze=5`) runs selectively on the top `26,000` highest-entropy/cross-model-disagreement rows (`predict_inductive_v4.py` verifies strictly inductive Section 7.3 inference from raw text at test time).
4. **Zero-Leakage Meta-Calibration + `v8/v9` Adversarial Priority Refinement**:
   - Separate per-task blend weights (`W_L`, `W_T`, `W_A`) and consensus confidence calibration, followed by `v8/v9` surgical fixes for sample-86 `alpha`/`T1` overfitting, `clearance charge` parcel-vs-prize/job priority (`1 > 2 > 4 > 3 > 5 > 0`), `"please do not reply"` footer false negatives, `"select hoyechen"` scams, `AnyDesk`/concatenated credential traps, and re-scoring repaired adversarial rows with `mMiniLM`.

---

## 2. Progression & Validation Results

| Version / Stage | Validation / Zero-Leakage OOD Score | Public LB Score |
|---|---|---|
| Previous multi-model baseline (`submission_final.csv`) | `—` | `0.85118` |
| URL-masked TF-IDF + GBDT + `rules_v4` (`sub_v6_fast.csv`) | `—` | `0.88933` |
| Audit v5 TF-IDF + LGBM/XGB/CatBoost + `rules_v5` (`sub_v6b_v5.csv`) | `Composite 91.08` (`F_A 97.17`, `F_B 76.30`, `F_C 77.87`, `Brier 0.03497`) | `0.93011` |
| Individual OOD Baselines (`sample_v4`): `BiGRU_v5` / `mMiniLM` / `mE5small` / `BGE` | `94.456` / `91.995` / `89.743` / `87.609` Composite | `—` |
| 8-Family Meta-Stack + `mDeBERTa` Escalation (`submission_v7_top10.csv`) | `Composite 98.299` (`F_A 100.0`, `F_B 91.86`, `F_C 100.0`, `F_D 98.66`, `Brier 0.01022`) | `0.93177` |
| + `v8` Adversarial Normalizer, Priority & `T1`/Brier Fix (`submission_v8_hotfix.csv`) | Verified on `sample.csv` (`rules_v8 Composite 98.225`) | `0.93552` |
| + `v9` Zero-FP Priority & Concatenated Credential Patch (`submission_v9_final.csv`) | Verified inductive pipeline (`predict_inductive_v4` on `sample.csv`: `Composite 99.393`, `F_A 100.0`) | **`0.93661`** |

---

## 3. Section 14 & Section 15 Declarations (Models, Resources & Seeds)

### External Pretrained Open-Weight Models (Hugging Face — Zero Hosted APIs Used)
| Model Identifier | Provider / License | Role in Pipeline |
|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Moritz Laurer (Hugging Face, MIT) | 2-fold domain teachers, full-scenario `mDeBERTa_full`, and selective `26k` uncertainty escalation |
| `intfloat/multilingual-e5-base` | Microsoft / intfloat (Hugging Face, MIT) | 2-fold domain teacher for consensus prototype scoring |
| `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` | Moritz Laurer (Hugging Face, MIT) | Distilled Student 1, OOD folds, adversarial re-scoring, & shared BPE tokenizer for `BiGRU-v5` |
| `intfloat/multilingual-e5-small` | Microsoft / intfloat (Hugging Face, MIT) | Distilled Student 2 (multilingual E5 `passage:` prefix) |
| `BAAI/bge-small-en-v1.5` | BAAI (Hugging Face, MIT) | Distilled Student 3 (English representation over `bn_to_en_roman_v4` glossed inputs) |

### Datasets, Synthetic Augmentation & APIs
- **Organizer Files Only**: `sample.csv` (`86` rows), `test.csv` (`500,000` rows), `sample_submission.csv` (`500,000` rows). No external labeled datasets used.
- **Synthetic Data Generation**: `generate_dataset_v4` programmatically generates `72,000` balanced conversations across `48` scenario families (`seed=4242`).
- **Programmatic Unlabeled Domain Adaptation (Section 14)**: Unlabeled `test.csv` text is hashed into `285,810` canonical structural skeletons; `45,000` representative prototypes are pseudo-labeled programmatically by teacher consensus and filtered to `39,618` high-agreement training prototypes (`build_clean_pseudo_v5`). Inference works purely inductively from raw text (`predict_inductive_v4`).
- **Hosted / Commercial APIs**: **None**. `.env` is intentionally empty.

### Software Environment & Exact Tested Versions
- **Python (GPU/Transformers)**: Python `3.13.9` (`torch 2.11.0+cu128`, `transformers 5.16.1`, `xgboost 3.2.0`, `scikit-learn 1.7.2`, `numpy 2.3.5`, `pandas 2.3.3`) on NVIDIA GTX 1650 (4 GB VRAM).
- **Python (GBDT / Stacking / CPU verification)**: Python `3.14` (`lightgbm 4.6.0`, `xgboost 3.2.0`, `catboost 1.2.10`, `scikit-learn 1.9.0`, `scipy 1.17.1`, `joblib 1.5.3`, `numpy 2.4.6`, `pandas 2.3.3`).
- **Notebook execution**: Install unified dependencies via:
  ```bash
  pip install torch transformers lightgbm xgboost catboost scikit-learn scipy joblib numpy pandas jupyter nbconvert
  ```

### Random Seeds
- Synthetic generation (`generate_dataset_v4`): `4242`
- TruncatedSVD / Linear / Meta optimization: `42`, `123`, `2026`
- Base Teachers (`train_teachers_v4`): `101 + fold` (`mDeBERTa`), `201 + fold` (`mE5-base`)
- BiGRU (`train_da_bigru_v4`): `70 + fold`, `909`
- Students (`train_students_v4`): `300 + fold` (3-fold OOD), `500` (distilled students), `777` & `2026` (`mDeBERTa_full`)

---

## 4. Instructions to Reproduce

1. Place competition files (`sample.csv`, `test.csv`, `sample_submission.csv`) in `./dataset/` or `./Main Preli Round/dataset/` (the notebook auto-detects common paths).
2. Keep `RUN_FULL_TRAINING = False` to run fast verification, EDA, rule evaluation (`sample.csv`), and full `submission.csv` schema & distribution checks (`~30s`).
3. Set `RUN_FULL_TRAINING = True` (GPU recommended; requires Hugging Face model downloads for the 5 MIT-licensed backbones above) and execute `notebook.ipynb` top-to-bottom to regenerate caches, synthetic data (`72,000` rows), teachers, domain consensus (`39,618` rows), students, `BiGRU`, `DA_Tab`, meta-stacking, `v8/v9` adversarial hotfixes, and `submission.csv`.
