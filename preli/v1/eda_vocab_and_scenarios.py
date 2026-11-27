import re
import time
from collections import Counter
import pandas as pd

# De-obfuscation table from Task 474
ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\ufe0f]")
CONFUSABLES = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0456": "i",
    "\u0441": "c", "\u0440": "p", "\u0443": "y", "\u0445": "x",
    "\u0410": "A", "\u0415": "E", "\u041e": "O", "\u0406": "I",
    "\u0421": "C", "\u0420": "P", "\u0423": "Y", "\u0425": "X",
    "\u0964": ".",
})

def quick_deobf(txt: str) -> str:
    txt = ZW_RE.sub("", txt).translate(CONFUSABLES)
    return txt

t0 = time.time()
latin_words = Counter()
bn_words = Counter()
multi_turn_examples = []
long_examples = []
unique_deobf = set()

for idx_chunk, chunk in enumerate(pd.read_csv("Main Preli Round/dataset/test.csv", chunksize=50000)):
    for row_id, raw in zip(chunk["id"], chunk["text"].fillna("").astype(str)):
        d = quick_deobf(raw)
        sk = re.sub(r"\d+", "<N>", d.lower())
        sk = re.sub(r"(?:https?|hxxps?)://\S+", "<U>", sk)
        unique_deobf.add(sk)
        if idx_chunk == 0:
            for w in re.findall(r"[A-Za-z]+", d.lower()):
                latin_words[w] += 1
            for w in re.findall(r"[\u0980-\u09FF]+", d):
                bn_words[w] += 1
            if "|||" in d and len(multi_turn_examples) < 25:
                multi_turn_examples.append((row_id, d))
            if len(d) > 350 and len(long_examples) < 10:
                long_examples.append((row_id, d))

print(f"500k deobf skeleton count: {len(unique_deobf)} ({time.time()-t0:.1f}s)")
print("\nTop 150 Latin words (first 50k rows):")
print(", ".join(f"{w}({c})" for w, c in latin_words.most_common(150)))
print("\nTop 120 Bengali words (first 50k rows):")
print(", ".join(f"{w}({c})" for w, c in bn_words.most_common(120)))
print("\nSample multi-turn conversations (first 15):")
for rid, txt in multi_turn_examples[:15]:
    print(f"  [{rid}] {txt}")
