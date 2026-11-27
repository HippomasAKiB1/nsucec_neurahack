import re
from collections import Counter
import pandas as pd

ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\ufe0f]")
CONF = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0456": "i",
    "\u0441": "c", "\u0440": "p", "\u0443": "y", "\u0445": "x",
    "\u0410": "A", "\u0415": "E", "\u041e": "O", "\u0406": "I",
    "\u0421": "C", "\u0420": "P", "\u0423": "Y", "\u0425": "X",
    "\u0964": ".",
})

# Sample slices across the entire 500k file to detect any domain-shift / novel-scenario blocks
slices = [0, 100000, 200000, 300000, 400000, 450000, 490000]
paths = Counter()
bengali_samples = []

for chunk_idx, chunk in enumerate(pd.read_csv("Main Preli Round/dataset/test.csv", chunksize=10000)):
    start_row = chunk_idx * 10000
    if start_row in slices:
        print(f"\n=== SLICE starting at row {start_row} (id={chunk['id'].iloc[0]}) ===")
        texts = [ZW_RE.sub("", str(x)).translate(CONF) for x in chunk["text"].fillna("")]
        avg_len = sum(len(x) for x in texts) / len(texts)
        mt_pct = sum("|||" in x for x in texts) / len(texts)
        bn_pct = sum(bool(re.search(r"[\u0980-\u09FF]", x)) for x in texts) / len(texts)
        print(f"  avg_len={avg_len:.1f}, multi_turn={mt_pct:.1%}, has_bn={bn_pct:.1%}")
        for j in range(0, 20, 4):
            print(f"  [{chunk['id'].iloc[j]}] {texts[j][:220]}")
    if chunk_idx < 5:
        for raw in chunk["text"].fillna("").astype(str):
            d = ZW_RE.sub("", raw).translate(CONF)
            for p in re.findall(r"(?:https?|hxxps?)://[^\s/]+(/[^\s\"']*)", d, flags=re.I):
                paths[p.lower().split("?")[0][:30]] += 1
            if re.search(r"[\u0980-\u09FF]", d) and len(bengali_samples) < 20:
                bengali_samples.append(d)

print("\nTop URL paths:", paths.most_common(25))
print("\nSample Bengali/mixed rows:")
for b in bengali_samples[:15]:
    print(f"  {b[:180]}")
