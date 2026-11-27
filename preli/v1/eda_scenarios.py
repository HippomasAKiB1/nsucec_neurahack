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

df = pd.read_csv("Main Preli Round/dataset/test.csv", nrows=50000)
texts = [ZW_RE.sub("", str(x)).translate(CONF) for x in df["text"].fillna("")]

# 1. Inspect embedded gibberish noise words (high frequency, low vowel ratio or 3+ consecutive consonants)
tok_counts = Counter()
for x in texts:
    for w in re.findall(r"[A-Za-z]{4,}", x.lower()):
        tok_counts[w] += 1

print("Likely embedded noise/gibberish tokens (count >= 100, odd consonant clusters):")
for w, c in tok_counts.most_common(600):
    if c >= 100 and (re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", w) or w in {"mnbvcx", "xkcdpl", "zzqwpo", "asdkjh", "qzxvbn"}):
        print(f"  {w}({c})")

# 2. Sample diverse rows across turn counts and buckets (purely structural inspection)
print("\nSample diverse 1-turn rows (spread across file):")
for i in range(0, 20000, 1000):
    if "|||" not in texts[i]:
        print(f"  [{df.id.iloc[i]}] {texts[i]}")

print("\nSample diverse multi-turn rows (6+ turns):")
shown = 0
for i in range(len(texts)):
    if texts[i].count("|||") >= 5:
        print(f"  [{df.id.iloc[i]}] {texts[i][:320]}")
        shown += 1
        if shown >= 15:
            break
