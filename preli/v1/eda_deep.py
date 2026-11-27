import re
import time
import unicodedata
from collections import Counter
import pandas as pd

t0 = time.time()
weird_chars = Counter()
url_tlds = Counter()
url_full = Counter()
short_msgs = Counter()
noise_words = Counter()
empty_or_tiny = 0

# Check sample-like patterns: URLs, suspicious tokens, all non-ASCII/Bengali
for chunk in pd.read_csv("Main Preli Round/dataset/test.csv", chunksize=50000):
    for txt in chunk["text"].fillna("").astype(str):
        for ch in txt:
            o = ord(ch)
            if not ((32 <= o <= 126) or (0x0980 <= o <= 0x09FF)):
                weird_chars[ch] += 1
        for u in re.findall(r"(?:https?|hxxps?)://([^\s/,;\"']+)", txt, flags=re.I):
            url_full[u.lower()] += 1
            parts = u.lower().replace("[.]", ".").split(".")
            if len(parts) >= 2:
                url_tlds[parts[-1]] += 1
        if len(txt) <= 25:
            empty_or_tiny += 1
            if len(short_msgs) < 50:
                short_msgs[txt] += 1

print(f"Done in {time.time()-t0:.1f}s | short<=25 count={empty_or_tiny}")
print("ALL non-(ASCII/Bengali) characters (top 80):")
for ch, cnt in weird_chars.most_common(80):
    print(f"  U+{ord(ch):04X} {unicodedata.name(ch, 'UNKNOWN'):42s} ({ch!r}): {cnt}")

print("\nURL TLDs:", url_tlds.most_common(25))
print("Top 40 URL domains:", url_full.most_common(40))
print("\nSample short<=25 messages:")
for k, v in list(short_msgs.items())[:30]:
    print(f"  {k!r} (x{v})")
