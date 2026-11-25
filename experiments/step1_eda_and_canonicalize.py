import re
import difflib
from collections import Counter
import pandas as pd

# Exact synthetic wrapper prefixes observed in test.csv (plus 1-transposition variants handled automatically)
RAW_PREFIXES = [
    "every day,", "every day",
    "sotti bolte,", "sotti bolte",
    "honestly,", "honestly",
    "seriously,", "seriously",
    "again,", "again",
    "complaint:",
    "today",
    "hello,", "hello",
    "hi,", "hi",
    "bhai,", "bhai",
    "ajke",
    "আবারও,", "আবারও",
    "দেখুন,", "দেখুন",
    "ভাই,", "ভাই",
    "আজ",
]

RAW_SUFFIXES = [
    "এর সমাধান চাই।", "এর সমাধান চাই",
    "খুবই বিরক্তিকর।", "খুবই বিরক্তিকর",
    "দয়া করে দেখুন।", "দয়া করে দেখুন",
    "ধন্যবাদ।", "ধন্যবাদ",
    "please look into it.", "please look into it",
    "please fix this.", "please fix this",
    "very frustrating.", "very frustrating",
    "please dekhen",
    "please check",
    "solve korun",
    "eta ki thik?", "eta ki thik",
    "bhison kharap",
    "#dhakatransport",
    "not ok.", "not ok",
    "plz fix",
    "dhonnobad",
    "thanks.", "thanks",
    "ugh",
    "!!",
]

def adjacent_swaps(word: str):
    out = {word}
    for i in range(len(word) - 1):
        w = list(word)
        w[i], w[i + 1] = w[i + 1], w[i]
        out.add("".join(w))
    return out

def match_phrase_fuzzy_start(text: str, phrase: str) -> str | None:
    # Match prefix allowing comma/colon and up to 1 adjacent transposition per Latin word
    t_strip = text.lstrip()
    words_p = phrase.split()
    # Split t_strip keeping separators
    m = re.match(r"^(" + r"\s+".join([r"[^\s]+"] * len(words_p)) + r")(?:\s+(.*))?$", t_strip)
    if not m:
        return None
    head, rest = m.group(1), m.group(2) or ""
    head_words = head.split()
    if len(head_words) != len(words_p):
        return None
    for hw, pw in zip(head_words, words_p):
        hw_l = hw.lower()
        pw_l = pw.lower()
        if hw_l == pw_l:
            continue
        # Strip trailing punctuation on last token only if pw doesn't end with it
        if pw_l.isascii() and hw_l.rstrip(",:") == pw_l.rstrip(",:") and hw_l[-1:] in (",", ":"):
            continue
        if hw_l not in adjacent_swaps(pw_l):
            return None
    return rest.strip()

def match_phrase_fuzzy_end(text: str, phrase: str) -> str | None:
    t_strip = text.rstrip()
    words_p = phrase.split()
    m = re.match(r"^(.*?)\s+(" + r"\s+".join([r"[^\s]+"] * len(words_p)) + r")$", t_strip)
    if not m:
        return None
    body, tail = m.group(1), m.group(2)
    tail_words = tail.split()
    if len(tail_words) != len(words_p):
        return None
    for tw, pw in zip(tail_words, words_p):
        tw_l = tw.lower()
        pw_l = pw.lower()
        if tw_l == pw_l:
            continue
        if tw_l.rstrip(".!?।") == pw_l.rstrip(".!?।") and pw_l[-1:] in (".", "!", "?", "।", ""):
            continue
        if tw_l.rstrip(".!?।") in adjacent_swaps(pw_l.rstrip(".!?।")):
            continue
        return None
    return body.strip()

def strip_wrappers(text: str) -> str:
    y = " ".join(text.strip().split())
    # Only strip if remaining text has >= 3 words or >= 12 chars (safety guard)
    changed = True
    while changed:
        changed = False
        for p in RAW_PREFIXES:
            res = match_phrase_fuzzy_start(y, p)
            if res and len(res) >= 10:
                y = res
                changed = True
                break
        for sf in RAW_SUFFIXES:
            res = match_phrase_fuzzy_end(y, sf)
            if res and len(res) >= 10:
                y = res
                changed = True
                break
    return y.strip(" ,")

def main():
    s = pd.read_csv("mock/dataset/sample.csv")
    t = pd.read_csv("mock/dataset/test.csv")

    s["clean"] = s["text"].apply(strip_wrappers)
    t["clean"] = t["text"].apply(strip_wrappers)

    # Check if stripping changed anything harmful in sample.csv
    changed_s = s[s["clean"] != s["text"]]
    print(f"Sample rows modified by strip_wrappers ({len(changed_s)}):")
    for _, r in changed_s.iterrows():
        print(f"  [{r.label}] {r.text!r} -> {r.clean!r}")

    # Build clean word vocabulary from sample + multi-occurrence clean words in test
    vocab = Counter()
    for txt in list(s["clean"]) + list(t["clean"]):
        for w in re.findall(r"[A-Za-z]+", txt.lower()):
            vocab[w] += 1

    # Fix single-occurrence words that are 1 adjacent swap from a frequent (>=2) word
    def fix_swaps(txt: str) -> str:
        def repl(m):
            w = m.group(0)
            wl = w.lower()
            if vocab[wl] >= 2:
                return w
            cands = [cand for cand in adjacent_swaps(wl) if cand != wl and vocab[cand] >= 2]
            if len(cands) == 1:
                return cands[0]
            if len(cands) > 1:
                return max(cands, key=lambda c: vocab[c])
            return w
        return re.sub(r"[A-Za-z]+", repl, txt)

    t["canon"] = t["clean"].apply(fix_swaps).str.lower().str.strip(" .!?")
    s["canon"] = s["clean"].apply(fix_swaps).str.lower().str.strip(" .!?")

    print(f"\nTest raw lower unique: {t['text'].str.lower().nunique()}")
    print(f"Test stripped clean lower unique: {t['clean'].str.lower().nunique()}")
    print(f"Test canon unique: {t['canon'].nunique()}")

    # Print all unique canonical test strings and their counts
    vc = t["canon"].value_counts()
    print(f"\nAll {len(vc)} unique canonical test cores:")
    for i, (k, c) in enumerate(vc.items(), 1):
        print(f"{i:3d} (x{c}): {k}")

if __name__ == "__main__":
    main()
