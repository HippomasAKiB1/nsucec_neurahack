import re
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from normalizer import clean_for_model, extract_structural_features
from synth_generator import generate_dataset

FEAT_NAMES = list(extract_structural_features("a", "a").keys())

# High-precision general domain regexes for Task B (T1/T5) and Task C (action=2 vs 3)
ENTITY_IMPERSONATION_RE = re.compile(
    r"\b(?:bkash|nagad|rocket|upay|surecash|mcash|tap|ok wallet|pathao pay|"
    r"dbbl|dutch[- ]?bangla|islami bank|brac bank|city bank|prime bank|pubali|jamuna|standard chartered|ucb|ebl|trust bank|"
    r"sundarban|paribahan|pathao|redx|steadfast|paperfly|karatoa|ecourier|dhl|fedex|janani|sheba|"
    r"grameenphone|gp|robi|banglalink|airtel|teletalk|skitto|"
    r"daraz|rokomari|ajkerdeal|star tech|aarong|shwapno|chaldal|pickaboo|"
    r"bdjobs|unilever|beximco|walton|square|pran|apex|"
    r"bangladesh bank|nbr|nid|btrc|customs|passport|pm education|vfs|govt|government|ministry|"
    r"desco|nesco|wasa|titas|btcl|dpdc|breb|"
    r"sim|agent|helpline|support|staff|officer|representative|customer care|service team)\b"
    r"|বিকাশ|নগদ|রকেট|উপায়|গ্রামীণফোন|রবি|বাংলালিংক|দারাজ|পাঠাও|সুন্দরবন|কাস্টমস|সরকার|ব্যাংক|সিম|এজেন্ট|হেল্পলাইন|কর্মকর্তা",
    flags=re.IGNORECASE,
)

STRONG_URGENCY_RE = re.compile(
    r"\b(?:within \d+|24 ?h|12 ?h|24 ghonta|expir(?:e|es|ed)|will be block|band hobe|bondho|suspend|"
    r"last warning|immediately|taratari|hurry|bipode|job jabe|avoid cancellation|"
    r"will be returned|ekhoni|today only|guaranteed|100% guaranteed|\d+% (?:weekly|daily) profit|"
    r"offer expires|claim (?:now|within))\b"
    r"|এখনই|নইলে|আজই|বন্ধ হয়ে|বাতিল|তাড়াতাড়ি|সতর্ক|জরুরি|গ্যারান্টি|নিশ্চিত",
    flags=re.IGNORECASE,
)

PAY_FEE_RE = re.compile(
    r"\b(?:pay\s+(?:tk|bdt|rs|\$)|pay\s+\d+|deposit\s+(?:tk|bdt)?\s*\d+|\d+\s*(?:taka|tk)\s*(?:deposit|fee|charge|duty|ferot)|"
    r"processing fee|delivery (?:fee|charge)|redelivery charge|verification fee|registration fee|"
    r"duty/storage|customs (?:fee|duty|today)|send (?:money|to bkash)|invest\s+(?:tk|bdt)|"
    r"ferot (?:din|den)|fee dite)\b"
    r"|প্রসেসিং ফি|ডেলিভারি চার্জ|ভেরিফিকেশন ফি|ডিপোজিট|টাকা দিন|টাকা পাঠান|টাকা ফেরত|পরিশোধ|শুল্ক",
    flags=re.IGNORECASE,
)

PROMO_SALE_RE = re.compile(
    r"\b(?:11\.11|flash sale|eid offer|winter sale|end of season|meg[a]? offer|1 ta kinle|"
    r"vouchers? up to|up to \d+% off|\d+% discount|\d+% cashback on your|code:\s*[a-z0-9]+|shortto projojjo|t&c apply|"
    r"book your dream flat|admission going on|combo:)\b"
    r"|ছাড়|অফার|শর্ত প্রযোজ্য|কিনুন",
    flags=re.IGNORECASE,
)


def extract_extra_feats_batch(clean_texts):
    out = np.zeros((len(clean_texts), 4), dtype=np.float32)
    for i, t in enumerate(clean_texts):
        s = str(t)
        out[i, 0] = 1.0 if ENTITY_IMPERSONATION_RE.search(s) else 0.0
        out[i, 1] = 1.0 if STRONG_URGENCY_RE.search(s) else 0.0
        out[i, 2] = 1.0 if PAY_FEE_RE.search(s) else 0.0
        out[i, 3] = 1.0 if PROMO_SALE_RE.search(s) else 0.0
    return out


def _process_chunk(rows):
    res_clean = []
    res_feats = []
    for raw in rows:
        c = clean_for_model(raw)
        d = extract_structural_features(raw, c)
        res_clean.append(c)
        res_feats.append([d[k] for k in FEAT_NAMES])
    return res_clean, np.array(res_feats, dtype=np.float32)


def regen_synth_v3(n_samples=32000, seed=420):
    t0 = time.time()
    df = generate_dataset(n_samples=n_samples, seed=seed, corrupt_prob=0.65)
    texts = df["text"].tolist()
    chunk_size = 2000
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    all_clean, all_feats = [], []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for cc, ff in ex.map(_process_chunk, chunks):
            all_clean.extend(cc)
            all_feats.append(ff)
    feats = np.vstack(all_feats)
    df["clean_text"] = all_clean
    df.to_csv("preli/cache/synth_32k_v3.csv", index=False)
    np.save("preli/cache/synth_32k_v3_feats.npy", feats)
    np.save("preli/cache/synth_32k_v3_extra.npy", extract_extra_feats_batch(all_clean))
    print(f"Generated & parallel-normalized 32k v3 in {time.time()-t0:.1f}s", flush=True)


def build_extra_caches():
    s_df = pd.read_csv("preli/cache/sample_clean.csv")
    s_ex = extract_extra_feats_batch(s_df["clean_text"].fillna("").astype(str).tolist())
    np.save("preli/cache/sample_extra.npy", s_ex)

    te_list = []
    for chunk in pd.read_csv("preli/cache/test_clean.csv", chunksize=100000):
        te_list.append(extract_extra_feats_batch(chunk["clean_text"].fillna("").astype(str).tolist()))
    np.save("preli/cache/test_extra.npy", np.vstack(te_list))
    print("Saved sample_extra.npy and test_extra.npy!", flush=True)


if __name__ == "__main__":
    build_extra_caches()
    regen_synth_v3()
