import numpy as np
import pandas as pd

s = pd.read_csv("experiments/cache/sample_norm.csv")
t = pd.read_csv("experiments/cache/test_norm.csv")
sub = pd.read_csv("submission.csv")
q = np.load("experiments/cache/qwen15b_kv.npz")

print("submission.csv shape:", sub.shape)
print("Class counts:", sub["label"].value_counts().sort_index().to_dict())

qp = q["t_probs"].argmax(1)
qconf = q["t_probs"].max(1)
diffs = qp != sub["label"].values
print(f"\nQwen(60-shot) vs Final Submission diffs: {diffs.sum()} / 300")
seen = set()
for i in np.where(diffs)[0]:
    k = t["en_norm"].iloc[i]
    if k in seen:
        continue
    seen.add(k)
    print(f"  id={t.id.iloc[i]} qwen={qp[i]}({qconf[i]:.2f}) final={sub.label.iloc[i]} | {t.denoised.iloc[i]} ==> {k}")
