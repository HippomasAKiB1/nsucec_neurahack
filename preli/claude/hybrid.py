import re, numpy as np, pandas as pd, sys
sys.path.insert(0,'/home/claude')
from rules import norm, SENS, ASK, NEGW, MISTAKE, RETURN
from skel import skel
SCAMWARN=r'(eta scam|this is a scam|scam bhai|block kor|ignore korsi|স্ক্যাম|sabdhan|beware)'
OTP_DELIVERY=r'(\d{4,8}.{0,40}(code|otp)|(code|otp).{0,30}\d{4,8})'
def tags(raw):
    t=norm(raw)
    warn=bool(re.search(NEGW,t)); scamw=bool(re.search(SCAMWARN,t))
    req=False
    for m in re.finditer(r'[^.!?।|]+',t):
        seg=m.group(0)
        if re.search(SENS,seg) and re.search(ASK,seg) and not re.search(NEGW,seg): req=True
    mistake=bool(re.search(MISTAKE,t)) and bool(re.search(RETURN,t)) and bool(re.search(r'(tk|taka|টাকা|\d{2,})',t))
    A=(req or mistake) and not scamw
    # hard legit: genuine code delivery with explicit warning, COD notice, balance/credit notices without any ask
    L=False
    if not A and not scamw:
        if warn and re.search(OTP_DELIVERY,t): L=True
        elif re.search(r'(cod tk [\d,]+ ready rakhun|আজ ডেলিভারি হবে|ajke deliver hobe\.?$)',t) and not req: L=True
        elif re.search(r'(account e tk [\d,]+ add hoyeche|current balance tk|ব্যালেন্স)',t) and not req and not re.search(r'(link|hxxp|click|dial|pin)',t): L=True
    return A,L,scamw
