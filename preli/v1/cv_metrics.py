import re
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

TECH_LIST = ["T1", "T2", "T3", "T4", "T5", "T6"]

def decode_consistent(p_lbl, p_tech, p_act, tech_th=0.5):
    """Enforce exact Section 2/4 structural consistency across Task A, B, and C."""
    pred_l = p_lbl.argmax(axis=1).astype(int)
    pred_a = np.where(np.isin(pred_l, [0, 5]), 0, 1 + p_act[:, 1:].argmax(axis=1)).astype(int)
    if np.ndim(tech_th) == 0:
        pred_t = (p_tech >= tech_th).astype(int)
    else:
        pred_t = (p_tech >= np.asarray(tech_th)[None, :]).astype(int)
    is_np = np.isin(pred_l, [0, 5])
    pred_t[is_np] = 0
    need_one = (~is_np) & (pred_t.sum(axis=1) == 0)
    if need_one.any():
        pred_t[need_one, p_tech[need_one].argmax(axis=1)] = 1
    return pred_l, pred_t, pred_a

def tech_str_to_vec(t_str: str) -> np.ndarray:
    v = np.zeros(6, dtype=np.int32)
    s = str(t_str).strip()
    if s and s != "NONE":
        for code in s.split("|"):
            if code in TECH_LIST:
                v[TECH_LIST.index(code)] = 1
    return v

def vec_to_tech_str(v: np.ndarray) -> str:
    codes = [TECH_LIST[i] for i in range(6) if v[i] > 0]
    return "|".join(codes) if codes else "NONE"

def eval_multitask(y_lbl_true, y_lbl_pred, y_tech_true, y_tech_pred, y_act_true, y_act_pred, conf_pred=None):
    fa = 100.0 * f1_score(y_lbl_true, y_lbl_pred, average="macro", zero_division=0)
    # Multi-label Macro F1 across T1..T6 (plus NONE indicator or standard 6-label macro)
    fb_6 = 100.0 * f1_score(y_tech_true, y_tech_pred, average="macro", zero_division=0)
    fc = 100.0 * f1_score(y_act_true, y_act_pred, average="macro", zero_division=0)
    acc_a = 100.0 * accuracy_score(y_lbl_true, y_lbl_pred)
    if conf_pred is not None:
        is_corr = (y_lbl_true == y_lbl_pred).astype(np.float64)
        brier = float(np.mean((conf_pred - is_corr) ** 2))
        c_score = 100.0 * max(0.0, 1.0 - brier)
    else:
        brier, c_score = 0.25, 75.0
    # Proxy composite (weighting Task A, B, C, Confidence proportionally: 40+15+10+5 -> scaled, or full proxy)
    proxy_comp = 0.70 * fa + 0.15 * fb_6 + 0.10 * fc + 0.05 * c_score
    return {
        "F_label": fa,
        "Acc_label": acc_a,
        "F_tech": fb_6,
        "F_action": fc,
        "C_conf": c_score,
        "Brier": brier,
        "CompositeProxy": proxy_comp,
    }

def build_template_groups(texts):
    """Group near-duplicate templates in sample.csv so CV never leaks templates across folds."""
    groups = []
    seen_skeletons = {}
    for i, t in enumerate(texts):
        # Aggressive skeleton: first 25 normalized alphanum chars or key template tokens
        cl = re.sub(r"[^a-z\u0980-\u09FF]+", " ", t.lower()).strip()
        # Normalize known duplicate pairs in sample.csv
        cl = re.sub(r"\d+", "N", cl)
        words = cl.split()[:4]
        key = " ".join(words)
        if "parcel atke" in cl or "পার্সেল আটকে" in cl:
            key = "PARCEL_ATKE_TEMPLATE"
        elif "ghor bose" in cl or "ঘরে বসে" in cl:
            key = "GHOR_BOSE_TEMPLATE"
        elif "otp is" in cl and "never share" in cl or "do not share" in cl:
            key = "OTP_NEVER_SHARE_TEMPLATE"
        elif "cash-out" in cl or "ক্যাশ আউট" in cl:
            key = "CASHOUT_SUCCESS_TEMPLATE"
        if key not in seen_skeletons:
            seen_skeletons[key] = len(seen_skeletons)
        groups.append(seen_skeletons[key])
    return np.array(groups)

def corrupt_text_adversarial(text: str, rng: random.Random) -> str:
    """Apply realistic ScamShield adversarial distortions for robustness validation & augmentation."""
    NOISE_WORDS = ["zzqwpo", "wkrjfejejrn", "mnbvcx", "qzxvbn", "xkcdpl", "asdkjh"]
    EMOJIS = ["💰", "🎁", "🙏", "⚠️", "📲", "🔥", "😊"]
    CYR = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "i": "\u0456", "c": "\u0441", "p": "\u0440", "y": "\u0443", "x": "\u0445"}
    LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

    out = text
    # Defang URLs
    if rng.random() < 0.5:
        out = re.sub(r"https?://", lambda m: "hxxps://" if "s" in m.group(0).lower() else "hxxp://", out, flags=re.I)
        out = out.replace(".", "[.]") if "http" in out.lower() and "[.]" not in out else out

    words = out.split(" ")
    new_words = []
    for w in words:
        if w == "|||" or "http" in w.lower() or "hxxp" in w.lower():
            new_words.append(w)
            continue
        # Occasional random noise token injection
        if rng.random() < 0.08:
            new_words.append(rng.choice(NOISE_WORDS))
        # Character-level distortions on Latin tokens
        if w.isascii() and len(w) >= 3 and rng.random() < 0.30:
            chars = list(w)
            op = rng.random()
            if op < 0.25:
                # Zero-width insertion
                pos = rng.randint(1, len(chars) - 1)
                chars.insert(pos, rng.choice(["\u200b", "\u200c"]))
            elif op < 0.50:
                # Cyrillic homoglyph
                for j, c in enumerate(chars):
                    if c.lower() in CYR and rng.random() < 0.35:
                        chars[j] = CYR[c.lower()]
            elif op < 0.70:
                # Leetspeak
                for j, c in enumerate(chars):
                    if c.lower() in LEET and rng.random() < 0.30:
                        chars[j] = LEET[c.lower()]
            elif op < 0.85:
                # Repeat a letter 3-4 times
                j = rng.randint(0, len(chars) - 1)
                if chars[j].isalpha():
                    chars[j] = chars[j] * rng.randint(3, 4)
            else:
                # Adjacent swap or vowel drop
                if len(chars) >= 4:
                    j = rng.randint(0, len(chars) - 2)
                    chars[j], chars[j + 1] = chars[j + 1], chars[j]
            w = "".join(chars)
        elif re.search(r"[\u0980-\u09FF]", w) and rng.random() < 0.15 and len(w) >= 2:
            # ZWNJ inside Bengali word
            pos = rng.randint(1, len(w) - 1)
            w = w[:pos] + "\u200c" + w[pos:]
        new_words.append(w)
        if rng.random() < 0.05:
            new_words.append(rng.choice(EMOJIS))

    res = " ".join(new_words)
    # Occasional merged space
    if rng.random() < 0.20:
        res = re.sub(r" ([a-z]{3,5}) ", lambda m: m.group(1) + " ", res, count=1)
    # Random casing
    r = rng.random()
    if r < 0.08:
        res = res.upper()
    elif r < 0.22:
        res = res.lower()
    return res
