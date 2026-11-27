import re
import time
import unicodedata
from collections import Counter
import pandas as pd

t0 = time.time()
s_df = pd.read_csv("Main Preli Round/dataset/sample.csv")

# Scan all 500k rows in chunks for non-ASCII, non-Bengali characters (homoglyphs/zero-width/emojis/etc.)
weird_chars = Counter()
speaker_prefixes = Counter()
noise_candidates = Counter()
url_domains = Counter()
turn_counts = Counter()

# Basic fast normalization to measure unique skeleton count across 500k rows
skeletons = set()
n_rows = 0

for chunk in pd.read_csv("Main Preli Round/dataset/test.csv", chunksize=50000):
    n_rows += len(chunk)
    texts = chunk["text"].fillna("").astype(str).tolist()
    for txt in texts:
        turns = txt.split("|||")
        turn_counts[len(turns)] += 1
        for trn in turns:
            m = re.match(r"^\s*(\[?[A-Za-z0-9_ ]{1,20}\]?:)\s*", trn)
            if m:
                speaker_prefixes[m.group(1).lower()] += 1
        for ch in txt:
            o = ord(ch)
            # Flag anything outside basic ASCII printable and Bengali block (0980-09FF)
            if not ((32 <= o <= 126) or (0x0980 <= o <= 0x09FF)):
                weird_chars[ch] += 1
        # Collect potential URL domains
        for u in re.findall(r"(?:https?|hxxps?)://([^\s/]+)", txt, flags=re.I):
            url_domains[u.lower()] += 1
        # Coarse skeleton: lowercase, mask digits/urls, strip punctuation
        sk = re.sub(r"\d+", "<N>", txt.lower())
        sk = re.sub(r"(?:https?|hxxps?)://\S+", "<U>", sk)
        skeletons.add(sk)

print(f"Scanned {n_rows} rows in {time.time()-t0:.1f}s")
print(f"Coarse digit/url-masked unique skeletons: {len(skeletons)}")
print("Turn count distribution:", dict(sorted(turn_counts.items())))
print("\nTop 40 non-(ASCII/Bengali) characters:")
for ch, cnt in weird_chars.most_common(40):
    print(f"  U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN'):35s} ({ch!r}): {cnt}")

print("\nTop 25 speaker prefixes:")
for k, v in speaker_prefixes.most_common(25):
    print(f"  {k!r}: {v}")

print("\nTop 25 URL domains:")
for k, v in url_domains.most_common(25):
    print(f"  {k}: {v}")
