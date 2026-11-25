import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

s = pd.read_csv("mock/dataset/sample.csv")
y = s["label"].values
bge = np.load("experiments/cache/bge_m3.npz")["s"]
e5b = np.load("experiments/cache/e5_base.npz")["s"]
both = np.hstack([bge, e5b])
both /= np.linalg.norm(both, axis=1, keepdims=True)

for name, X in [("BGE-M3", bge), ("E5-Base", e5b), ("BGE+E5", both)]:
    for cname, fn in [
        ("Centroid", lambda: NearestCentroid()),
        ("KNN(3,cos)", lambda: KNeighborsClassifier(3, metric="cosine", weights="distance")),
        ("KNN(5,cos)", lambda: KNeighborsClassifier(5, metric="cosine", weights="distance")),
        ("LogReg(C=5)", lambda: LogisticRegression(C=5.0, max_iter=1000)),
        ("LogReg(C=20)", lambda: LogisticRegression(C=20.0, max_iter=1000)),
        ("Ridge(a=0.5)", lambda: RidgeClassifier(alpha=0.5)),
        ("LinearSVC(C=1)", lambda: LinearSVC(C=1.0)),
        ("SVC-RBF(C=3)", lambda: SVC(C=3.0, kernel="rbf")),
    ]:
        preds = np.zeros_like(y)
        for tr, va in LeaveOneOut().split(X):
            m = fn()
            m.fit(X[tr], y[tr])
            preds[va] = m.predict(X[va])
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        print(f"{name:8s} | {cname:14s} | LOO acc={acc:.4f} f1={f1:.4f}")
