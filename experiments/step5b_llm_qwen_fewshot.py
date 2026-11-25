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
# 3.08 GB fp16 fits in ~3.45 GB free VRAM with device_map="auto" (only single-token forward pass!)
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

# Pick 3 diverse exemplars per class (18 total) excluding any target row during LOO
def build_prompt(target_raw: str, target_en: str, exclude_idx: int = -1) -> str:
    ex_lines = []
    for c in range(6):
        idxs = [i for i in range(60) if y[i] == c and i != exclude_idx]
        # pick first 4 diverse examples per class (24 examples total)
        for i in idxs[:4]:
            r_txt = s_df["denoised"].iloc[i]
            e_txt = s_df["en_norm"].iloc[i]
            if r_txt.lower() != e_txt.lower():
                ex_lines.append(f"Message: {r_txt} (English: {e_txt}) -> {c}")
            else:
                ex_lines.append(f"Message: {r_txt} -> {c}")
    tgt_str = f"{target_raw} (English: {target_en})" if target_raw.lower() != target_en.lower() else target_raw
    return (
        f"{LABELS_DESC}\n\nExamples:\n"
        + "\n".join(ex_lines)
        + f"\n\nClassify the following message. Reply ONLY with the single digit (0, 1, 2, 3, 4, or 5).\nMessage: {tgt_str} ->"
    )

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, cache_dir=CACHE_DIR, dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
).eval()
print(f"Loaded {MODEL_NAME} in {time.time()-t0:.1f}s", flush=True)

digit_ids = [tok.encode(str(d), add_special_tokens=False)[-1] for d in range(6)]
# Also check space-prefixed digit tokens just in case (" 0" vs "0")
digit_sp_ids = [tok.encode(" " + str(d), add_special_tokens=False)[-1] for d in range(6)]
print("digit_ids:", digit_ids, "digit_sp_ids:", digit_sp_ids, flush=True)

@torch.no_grad()
def score_prompts(prompts):
    logits_out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(txt, return_tensors="pt").to(mdl.device)
        out = mdl(**inp).logits[0, -1].float()
        # Take max/logsumexp between bare digit and space-prefixed digit
        l_bare = out[digit_ids].cpu().numpy()
        l_sp = out[digit_sp_ids].cpu().numpy()
        logits_out.append(np.logaddexp(l_bare, l_sp))
    return np.array(logits_out)

# 1. LOO-CV on sample.csv (60 items)
t1 = time.time()
s_prompts = [build_prompt(s_df["denoised"].iloc[i], s_df["en_norm"].iloc[i], exclude_idx=i) for i in range(60)]
s_logits = score_prompts(s_prompts)
s_probs = softmax(s_logits, axis=1)
acc = accuracy_score(y, s_probs.argmax(1))
f1 = f1_score(y, s_probs.argmax(1), average="macro")
print(f"Qwen2.5-1.5B Few-Shot LOO: Acc={acc:.4f}, Macro-F1={f1:.4f} ({time.time()-t1:.1f}s)", flush=True)
for i in range(60):
    if s_probs.argmax(1)[i] != y[i]:
        print(f"  [ERR] id={s_df.id.iloc[i]:2d} true={y[i]} pred={s_probs.argmax(1)[i]} p={s_probs[i].max():.2f} | {s_df.raw.iloc[i]}")

# 2. Score unique canonical test items (only ~162 items, mapped back to all 300 rows!)
t_keys = [canonical_key(x) for x in t_df["en_norm"].fillna("")]
uniq_keys = []
seen = set()
key_rep_idx = {}
for i, k in enumerate(t_keys):
    if k not in seen:
        seen.add(k)
        uniq_keys.append(k)
        key_rep_idx[k] = i

t2 = time.time()
u_prompts = [build_prompt(t_df["denoised"].iloc[key_rep_idx[k]], t_df["en_norm"].iloc[key_rep_idx[k]], exclude_idx=-1) for k in uniq_keys]
u_logits = score_prompts(u_prompts)
u_probs = softmax(u_logits, axis=1)
key_to_prob = {k: u_probs[j] for j, k in enumerate(uniq_keys)}
key_to_logit = {k: u_logits[j] for j, k in enumerate(uniq_keys)}
t_probs = np.array([key_to_prob[k] for k in t_keys])
t_logits = np.array([key_to_logit[k] for k in t_keys])
print(f"Scored {len(uniq_keys)} unique test clusters in {time.time()-t2:.1f}s", flush=True)
print("Qwen2.5-1.5B test class counts:", pd.Series(t_probs.argmax(1)).value_counts().sort_index().to_dict())

np.savez("experiments/cache/qwen15b_fs.npz", s_logits=s_logits, s_probs=s_probs, t_logits=t_logits, t_probs=t_probs)
del mdl; gc.collect(); torch.cuda.empty_cache()
