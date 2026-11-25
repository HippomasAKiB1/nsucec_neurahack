import copy
import gc
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_clean import canonical_key

CACHE_DIR = r"D:\Datathon\hf_cache"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

LABELS_DESC = """Classify Dhaka public bus passenger feedback into exactly ONE primary category (0, 1, 2, 3, 4, or 5) per the official guidelines:
0 (Delay): Late buses, traffic-related delays, long waiting times, breakdowns causing delay, or missed schedules.
1 (Fare): Ticket prices, overcharging, fare disputes/arguments, refusing change, student half-fare refusal, no ticket given, or bkash/card payment-related issues.
2 (Safety): Harassment of passengers, reckless/fast/drunk/phone driving, racing, collisions/accidents, falling off moving bus, open doors, overloaded roof/no emergency exit, unsafe behaviour, or pickpocket/snatching security concerns.
3 (Cleanliness): Dirty vehicles/seats/windows/handles, waste/garbage, spit, vomit, unpleasant/rotten smell, filthy toilets, cockroaches/lizards/pests, or hygiene-related complaints.
4 (Service): Staff behaviour (rude/unhelpful/shouting/ignoring/disrespecting), overcrowding (packed/standing/no room to board), AC off/suffocating, skipping/refusing stops, men sitting in women's reserved seats, poor service quality, or general negative service experience.
5 (Other): Feedback that does not clearly belong to categories 0-4, including route/timing/contact inquiries, lost & found items, app/website issues, positive praise/compliments/thanks, missing stop signs, suggestions (new route/live tracking/ramps), or form tests."""

def fmt_ex(raw_txt, en_txt):
    r, e = str(raw_txt).strip(), str(en_txt).strip()
    return f"{r} (English: {e})" if r.lower() != e.lower() else e

def make_interleaved_order(indices, seed):
    rng = random.Random(seed)
    by_cls = {c: [i for i in indices if y[i] == c] for c in range(6)}
    for c in range(6):
        rng.shuffle(by_cls[c])
    out = []
    max_len = max(len(v) for v in by_cls.values())
    cls_order = list(range(6))
    for k in range(max_len):
        rng.shuffle(cls_order)
        for c in cls_order:
            if k < len(by_cls[c]):
                out.append(by_cls[c][k])
    return out

def build_prefix_prompt(ex_indices):
    lines = [f"Message: {fmt_ex(s_df['denoised'].iloc[idx], s_df['en_norm'].iloc[idx])} -> {y[idx]}" for idx in ex_indices]
    return (
        "<|im_start|>system\nYou are an expert Dhaka public bus feedback classifier.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{LABELS_DESC}\n\nExamples:\n" + "\n".join(lines) + "\n\nClassify each message below. Reply ONLY with the single digit (0, 1, 2, 3, 4, or 5).\n"
    )

def build_suffix(raw_txt, en_txt):
    return f"Message: {fmt_ex(raw_txt, en_txt)} ->" + "<|im_end|>\n<|im_start|>assistant\n"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True, dtype=torch.float16, device_map="auto"
).eval()
digit_ids = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(6)]
print(f"Loaded {MODEL_NAME} in {time.time()-t0:.1f}s on {mdl.device}", flush=True)

@torch.no_grad()
def score_with_prefix_kv(ex_indices, target_pairs):
    pfx_str = build_prefix_prompt(ex_indices)
    pfx_ids = tok(pfx_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(mdl.device)
    pfx_kv = mdl(input_ids=pfx_ids, use_cache=True).past_key_values
    pfx_len = pfx_ids.shape[1]

    logits_out = []
    for r_txt, e_txt in target_pairs:
        sfx_ids = tok(build_suffix(r_txt, e_txt), return_tensors="pt", add_special_tokens=False)["input_ids"].to(mdl.device)
        kv_copy = copy.deepcopy(pfx_kv)
        pos_ids = torch.arange(pfx_len, pfx_len + sfx_ids.shape[1], device=mdl.device).unsqueeze(0)
        out = mdl(input_ids=sfx_ids, past_key_values=kv_copy, position_ids=pos_ids, use_cache=False)
        logits_out.append(out.logits[0, -1, digit_ids].float().cpu().numpy())
    return np.array(logits_out)

# 1. 5-Fold Stratified OOF with 2 interleaved permutations per fold (48 train -> 12 val per fold!)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
s_pairs = [(s_df["denoised"].iloc[i], s_df["en_norm"].iloc[i]) for i in range(60)]
oof_logits = np.zeros((60, 6))
t1 = time.time()
for fold, (tr, va) in enumerate(skf.split(np.zeros(60), y)):
    va_pairs = [s_pairs[i] for i in va]
    ord_tr = make_interleaved_order(tr, 101 + fold)
    lg_acc = score_with_prefix_kv(ord_tr, va_pairs)
    oof_logits[va] = lg_acc
    print(f"5-Fold [{fold+1}/5] val_acc={accuracy_score(y[va], lg_acc.argmax(1)):.4f} (elapsed {time.time()-t1:.1f}s)", flush=True)

# Compute null-prompt calibration baseline as well as balanced OOF mean
null_logits = np.zeros(6)
for p_seed in [11, 22]:
    ord_all = make_interleaved_order(list(range(60)), p_seed)
    null_logits += score_with_prefix_kv(ord_all, [("N/A", "N/A")])[0] / 2.0

oof_mean = oof_logits.mean(axis=0, keepdims=True)
oof_cal = oof_logits - oof_mean
oof_null_cal = oof_logits - null_logits[None, :]
print(f"\nQwen2.5-1.5B (v2 norm, 5-Fold) RAW     : Acc={accuracy_score(y, oof_logits.argmax(1)):.4f} F1={f1_score(y, oof_logits.argmax(1), average='macro'):.4f}", flush=True)
print(f"Qwen2.5-1.5B (v2 norm, 5-Fold) BAL-CAL : Acc={accuracy_score(y, oof_cal.argmax(1)):.4f} F1={f1_score(y, oof_cal.argmax(1), average='macro'):.4f}", flush=True)
print(f"Qwen2.5-1.5B (v2 norm, 5-Fold) NULL-CAL: Acc={accuracy_score(y, oof_null_cal.argmax(1)):.4f} F1={f1_score(y, oof_null_cal.argmax(1), average='macro'):.4f}", flush=True)
for i in range(60):
    if oof_cal.argmax(1)[i] != y[i]:
        print(f"  [ERR] id={s_df.id.iloc[i]:2d} true={y[i]} pred={oof_cal.argmax(1)[i]} | {s_df.raw.iloc[i]} ==> {s_df.en_norm.iloc[i]}", flush=True)

# 2. Test Inference: 2 interleaved permutations of all 60 examples on canonical test clusters
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
uniq_keys, key_rep = [], {}
for i, k in enumerate(t_keys):
    if k not in key_rep:
        key_rep[k] = i
        uniq_keys.append(k)
u_pairs = [(t_df["denoised"].iloc[key_rep[k]], t_df["en_norm"].iloc[key_rep[k]]) for k in uniq_keys]

u_logits = np.zeros((len(u_pairs), 6))
t2 = time.time()
for p_seed in [11, 22]:
    ord_all = make_interleaved_order(list(range(60)), p_seed)
    u_logits += score_with_prefix_kv(ord_all, u_pairs) / 2.0
    print(f"Test perm seed={p_seed} done in {time.time()-t2:.1f}s", flush=True)

k2l = {k: u_logits[j] for j, k in enumerate(uniq_keys)}
t_logits = np.array([k2l[k] for k in t_keys])
t_cal = t_logits - oof_mean
t_null_cal = t_logits - null_logits[None, :]
print("Test MultiPerm BAL-CAL class counts:", pd.Series(t_cal.argmax(1)).value_counts().sort_index().to_dict(), flush=True)
np.savez("experiments/cache/v2_qwen15b_mp.npz", oof_logits=oof_logits, oof_cal=oof_cal, oof_null_cal=oof_null_cal, t_logits=t_logits, t_cal=t_cal, t_null_cal=t_null_cal)
del mdl; gc.collect(); torch.cuda.empty_cache()
