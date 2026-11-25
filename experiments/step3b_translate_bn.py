import gc
import os
os.environ["HF_HOME"] = r"D:\Datathon\hf_cache"
import re
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from text_clean import strip_wrappers, build_swap_fixer, gloss_banglish

s = pd.read_csv("mock/dataset/sample.csv")
t = pd.read_csv("mock/dataset/test.csv")
all_clean_vocab = [strip_wrappers(x) for x in s["text"]] + [strip_wrappers(x) for x in t["text"]]
fix_swaps = build_swap_fixer(all_clean_vocab)

s_gls = [gloss_banglish(fix_swaps(strip_wrappers(x))) for x in s["text"]]
t_gls = [gloss_banglish(fix_swaps(strip_wrappers(x))) for x in t["text"]]

# Identify rows that still contain Bengali script
all_gls = s_gls + t_gls
bn_indices = [i for i, x in enumerate(all_gls) if re.search(r"[\u0980-\u09FF]", x)]
print(f"Rows with Bengali script to translate: {len(bn_indices)} / {len(all_gls)}")

t0 = time.time()
mname = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(mname, src_lang="ben_Beng")
mdl = AutoModelForSeq2SeqLM.from_pretrained(mname, dtype=torch.float16).to("cuda").eval()
eng_id = tok.convert_tokens_to_ids("eng_Latn")

bn_texts = [all_gls[i] for i in bn_indices]
en_trans = []
for i in range(0, len(bn_texts), 16):
    inp = tok(bn_texts[i:i+16], padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = mdl.generate(**inp, forced_bos_token_id=eng_id, max_new_tokens=64, num_beams=2)
    en_trans.extend(tok.batch_decode(gen, skip_special_tokens=True))

del mdl; gc.collect(); torch.cuda.empty_cache()
print(f"Translated {len(bn_texts)} texts in {time.time()-t0:.1f}s")

all_en = list(all_gls)
for idx, tr in zip(bn_indices, en_trans):
    all_en[idx] = tr.strip()

print("\nSample Bengali/Mixed -> English translations:")
for i in range(60):
    if i in bn_indices:
        print(f"  [{s.label.iloc[i]}] {s_gls[i]}  ==>  {all_en[i]}")

os.makedirs("experiments/cache", exist_ok=True)
pd.DataFrame({"id": s["id"], "label": s["label"], "raw": s["text"], "denoised": [fix_swaps(strip_wrappers(x)) for x in s["text"]], "glossed": s_gls, "en_norm": all_en[:60]}).to_csv("experiments/cache/sample_norm.csv", index=False)
pd.DataFrame({"id": t["id"], "raw": t["text"], "denoised": [fix_swaps(strip_wrappers(x)) for x in t["text"]], "glossed": t_gls, "en_norm": all_en[60:]}).to_csv("experiments/cache/test_norm.csv", index=False)
print("Saved experiments/cache/sample_norm.csv and test_norm.csv")
