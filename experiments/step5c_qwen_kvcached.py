import copy
import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_clean import canonical_key

CACHE_DIR = r"D:\Datathon\hf_cache"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

s_df = pd.read_csv("experiments/cache/sample_norm.csv")
t_df = pd.read_csv("experiments/cache/test_norm.csv")
y = s_df["label"].values

LABELS_DESC = """Classify Dhaka public bus passenger feedback into exactly ONE label ID (0, 1, 2, 3, 4, or 5):
0: Delay, Late Arrival, Traffic Jam, Long Waiting Time, Missed Bus, Bus Breakdown, Schedule Failure (e.g. bus late, stuck in jam, long wait, never follows schedule).
1: Fare, Overcharging, Ticketing & Payment (e.g. extra fare, high ticket price, student half-fare refused, change not returned, paid without ticket, bkash/card deducted twice).
2: Safety, Security & Dangerous Driving (e.g. reckless/fast driving, racing, drunk driver, accident/collision, doors open while moving, falling off, unsafe at night, harassment of women, pickpocket/snatching).
3: Cleanliness, Hygiene & Pests (e.g. dirty seats/floor/windows, dust, stains, garbage, spit, vomit, bad smell, filthy terminal toilets, cockroaches/lizards).
4: Staff Conduct & Service Quality/Comfort (e.g. rude/unhelpful conductor/helper/staff, AC off/hot, overcrowded/packed/standing, skipping or refusing stops, men sitting in women's reserved seats).
5: General Inquiry, Suggestions, Lost & Found, App/Website, Praise & Signage (e.g. asking bus timing/route/office number, lost bag/wallet, requesting new route/live tracking/ramps, positive praise/thanks, missing stop name sign, testing form)."""

def fmt_ex(raw_txt, en_txt):
    return f"{raw_txt} (English: {en_txt})" if str(raw_txt).lower() != str(en_txt).lower() else str(raw_txt)

# Build a clean shared 30-shot prefix (5 examples per class) for test inference,
# and also support fast KV-cached LOO-CV (using the 30 exemplars NOT in the current fold, or 2 disjoint halves!)
def build_prefix_prompt(ex_indices):
    lines = []
    for idx in ex_indices:
        lines.append(f"Message: {fmt_ex(s_df['denoised'].iloc[idx], s_df['en_norm'].iloc[idx])} -> {y[idx]}")
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
    MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True,
    dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
).eval()
print(f"Loaded {MODEL_NAME} in {time.time()-t0:.1f}s (device: {mdl.device})", flush=True)

digit_ids = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(6)]

@torch.no_grad()
def eval_with_prefix_kv(ex_indices, target_pairs, label="run"):
    t1 = time.time()
    pfx_str = build_prefix_prompt(ex_indices)
    pfx_ids = tok(pfx_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(mdl.device)
    pfx_out = mdl(input_ids=pfx_ids, use_cache=True)
    pfx_kv = pfx_out.past_key_values
    pfx_len = pfx_ids.shape[1]
    print(f"[{label}] Prefix len={pfx_len} cached in {time.time()-t1:.1f}s", flush=True)

    logits_out = []
    t2 = time.time()
    for i, (r_txt, e_txt) in enumerate(target_pairs):
        sfx_str = build_suffix(r_txt, e_txt)
        sfx_ids = tok(sfx_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(mdl.device)
        kv_copy = copy.deepcopy(pfx_kv)
        pos_ids = torch.arange(pfx_len, pfx_len + sfx_ids.shape[1], device=mdl.device).unsqueeze(0)
        out = mdl(input_ids=sfx_ids, past_key_values=kv_copy, position_ids=pos_ids, use_cache=False)
        lg = out.logits[0, -1, digit_ids].float().cpu().numpy()
        logits_out.append(lg)
        if (i + 1) % 30 == 0 or (i + 1) == len(target_pairs):
            print(f"[{label}] {i+1}/{len(target_pairs)} done in {time.time()-t2:.1f}s", flush=True)
    return np.array(logits_out)

# Fast 2-fold disjoint 30-shot CV on all 60 sample rows (30 train -> 30 val, twice = 60 OOF!)
# Every row is scored with 5 exemplars per class strictly disjoint from its fold!
fold0_idx, fold1_idx = [], []
for c in range(6):
    c_idxs = [i for i in range(60) if y[i] == c]
    fold0_idx.extend(c_idxs[:5])
    fold1_idx.extend(c_idxs[5:])

s_pairs = [(s_df["denoised"].iloc[i], s_df["en_norm"].iloc[i]) for i in range(60)]
lg_f1 = eval_with_prefix_kv(fold0_idx, [s_pairs[i] for i in fold1_idx], label="CV-Fold1")
lg_f0 = eval_with_prefix_kv(fold1_idx, [s_pairs[i] for i in fold0_idx], label="CV-Fold0")

s_logits = np.zeros((60, 6))
s_logits[fold1_idx] = lg_f1
s_logits[fold0_idx] = lg_f0
s_probs = softmax(s_logits, axis=1)
print(f"\n>>> Qwen2.5-1.5B 2-Fold (30-shot) OOF: Acc={accuracy_score(y, s_probs.argmax(1)):.4f}, F1={f1_score(y, s_probs.argmax(1), average='macro'):.4f}", flush=True)
for i in range(60):
    if s_probs.argmax(1)[i] != y[i]:
        print(f"  [ERR] id={s_df.id.iloc[i]:2d} true={y[i]} pred={s_probs.argmax(1)[i]} p={s_probs[i].max():.2f} | {s_df.raw.iloc[i]}", flush=True)

# Now use ALL 60 sample examples (interleaved by class) as prefix for test canonical clusters!
all_60_interleaved = []
per_cls = {c: [i for i in range(60) if y[i] == c] for c in range(6)}
for k in range(10):
    for c in range(6):
        all_60_interleaved.append(per_cls[c][k])

t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
uniq_keys, key_rep = [], {}
for i, k in enumerate(t_keys):
    if k not in key_rep:
        key_rep[k] = i
        uniq_keys.append(k)

u_pairs = [(t_df["denoised"].iloc[key_rep[k]], t_df["en_norm"].iloc[key_rep[k]]) for k in uniq_keys]
u_logits = eval_with_prefix_kv(all_60_interleaved, u_pairs, label="Test-60Shot")
u_probs = softmax(u_logits, axis=1)
k2p = {k: u_probs[j] for j, k in enumerate(uniq_keys)}
k2l = {k: u_logits[j] for j, k in enumerate(uniq_keys)}
t_probs = np.array([k2p[k] for k in t_keys])
t_logits = np.array([k2l[k] for k in t_keys])

print("Qwen2.5-1.5B 60-shot Test Class Counts:", pd.Series(t_probs.argmax(1)).value_counts().sort_index().to_dict(), flush=True)
np.savez("experiments/cache/qwen15b_kv.npz", s_logits=s_logits, s_probs=s_probs, t_logits=t_logits, t_probs=t_probs)
del mdl; gc.collect(); torch.cuda.empty_cache()
