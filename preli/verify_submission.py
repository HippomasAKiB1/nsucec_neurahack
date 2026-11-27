import sys
import numpy as np
import pandas as pd


def verify_submission(sub_path: str, sample_sub_path: str = "Main Preli Round/dataset/sample_submission.csv"):
    print(f"Verifying {sub_path} against {sample_sub_path} ...")
    ref_ids = pd.read_csv(sample_sub_path, usecols=["id"])["id"].values
    ref_cols = list(pd.read_csv(sample_sub_path, nrows=1).columns)

    df = pd.read_csv(sub_path, keep_default_na=False)
    assert list(df.columns) == ref_cols, f"Column mismatch: {list(df.columns)} != {ref_cols}"
    assert len(df) == len(ref_ids), f"Row count mismatch: {len(df)} != {len(ref_ids)}"
    assert np.array_equal(df["id"].values, ref_ids), "ID alignment mismatch!"

    # Check zero blanks / NaNs
    for col in df.columns:
        s = df[col].astype(str)
        assert not (s == "").any(), f"Blank values in {col}"
        assert not (s.str.lower() == "nan").any(), f"NaN string in {col}"

    # Validate label in 0..5
    lbl = pd.to_numeric(df["label"], errors="raise").values
    assert np.isin(lbl, [0, 1, 2, 3, 4, 5]).all(), "Invalid label outside 0..5!"

    # Validate action in 0..5 and consistency with label
    act = pd.to_numeric(df["action"], errors="raise").values
    assert np.isin(act, [0, 1, 2, 3, 4, 5]).all(), "Invalid action outside 0..5!"
    is_np = np.isin(lbl, [0, 5])
    assert (act[is_np] == 0).all(), f"Found {(act[is_np] != 0).sum()} rows with label in {{0,5}} but action != 0!"
    assert (act[~is_np] != 0).all(), f"Found {(act[~is_np] == 0).sum()} scam rows (1..4) with action == 0!"

    # Validate techniques syntax and consistency
    valid_codes = {"T1", "T2", "T3", "T4", "T5", "T6"}
    tech = df["techniques"].astype(str).values
    assert (tech[is_np] == "NONE").all(), f"Found {(tech[is_np] != 'NONE').sum()} rows with label in {{0,5}} but techniques != NONE!"
    assert (tech[~is_np] != "NONE").all(), f"Found {(tech[~is_np] == 'NONE').sum()} scam rows (1..4) with techniques == NONE!"
    bad_syntax = 0
    for s in set(tech):
        if s == "NONE":
            continue
        parts = s.split("|")
        if len(parts) != len(set(parts)) or not all(p in valid_codes for p in parts) or parts != sorted(parts):
            bad_syntax += 1
    assert bad_syntax == 0, f"Found {bad_syntax} invalid technique strings!"

    # Validate confidence in (0, 1]
    conf = pd.to_numeric(df["confidence"], errors="raise").values
    assert np.isfinite(conf).all(), "Non-finite confidence!"
    assert ((conf >= 0.0) & (conf <= 1.0)).all(), "Confidence outside [0, 1]!"

    print("ALL ASSERTIONS PASSED SUCCESSFULLY!")
    print("Label distribution (%):")
    vc_l = df["label"].value_counts().sort_index()
    for k, v in vc_l.items():
        print(f"  label={k}: {v:7d} ({100*v/len(df):.2f}%)")
    print("Action distribution (%):")
    vc_a = df["action"].value_counts().sort_index()
    for k, v in vc_a.items():
        print(f"  action={k}: {v:7d} ({100*v/len(df):.2f}%)")
    print(f"Top 10 techniques: {df['techniques'].value_counts().head(10).to_dict()}")
    print(f"Confidence stats: min={conf.min():.4f}, p25={np.percentile(conf,25):.4f}, p50={np.percentile(conf,50):.4f}, mean={conf.mean():.4f}, p75={np.percentile(conf,75):.4f}, max={conf.max():.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "preli/submission_v1.csv"
    verify_submission(path)
