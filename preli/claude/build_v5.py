"""v5 = existing 7-model ensemble + guide-derived priority-1 / hard-legit overrides (+ optional template-cluster smoothing).
USE_CLUSTER=False  -> fully inductive (no use of unlabeled test text beyond per-row inference). Safe under the rulebook.
USE_CLUSTER=True   -> transductive smoothing; ONLY use if the final rulebook explicitly permits unlabeled-test-text use."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0,'/home/claude')
from rules import norm, NEGW, MISTAKE, RETURN
from hybrid import tags
from skel import skel
TE='/home/claude/info/dataset/test.csv'; SUB='/home/claude/preli/submission_final.csv'
t=pd.read_csv(TE); f=pd.read_csv(SUB)
assert (t.id.values==f.id.values).all()
tg=t.text.map(tags); A=np.array([x[0] for x in tg]); L=np.array([x[1] for x in tg])
lab=f.label.values.copy(); act=f.action.values.copy(); tech=f.techniques.values.astype(object).copy(); conf=f.confidence.values.copy()
ml_lab=lab.copy()
def tech_for_A(raw):
    n=norm(raw); T=[]
    secret=bool(re.search(r'(pin|otp|password|nid|cvv|card|code|ওটিপি|পিন|কোড|এনআইডি)',n))
    ret=bool(re.search(MISTAKE,n)) 
    imp=bool(re.search(r'(helpline|officer|agent|support|fraud control|security|customs|courier|parcel|bank|dear (customer|student|citizen)|^\w+ ?\w*:|i am from|priyo|প্রিয়|[a-z]+ alert)',n))
    urg=bool(re.search(r'(hour|minute|ghont|today|now|expire|block|band|disconnect|asap|ekhoni|immediately|ঘণ্ট|বন্ধ|এখনই|আজ)',n))
    if imp and not ret: T.append('T1')
    if secret and not (ret and not re.search(r'(pin|otp|password|nid|cvv|ওটিপি|পিন)',n)): T.append('T2')
    if ret and not secret: T.append('T3')
    if urg or ret: T.append('T5')
    if re.search(r'(won|jit|prize|gift|bonus|reward|জিত)',n): T.append('T4')
    return '|'.join(sorted(set(T), key=lambda x:int(x[1]))) or 'T2'
idx=np.where(A)[0]
for i in idx:
    raw=t.text.iat[i]; n=norm(raw)
    lab[i]=1
    only_return = bool(re.search(MISTAKE,n)) and not re.search(r'(pin|otp|password|nid|cvv|ওটিপি|পিন)',n)
    act[i]=2 if only_return else 1
    tech[i]=tech_for_A(raw)
idx=np.where(L & ~A)[0]
lab[idx]=0; act[idx]=0; tech[idx]='NONE'
# consistency: legit/promo carry no technique and action 0
m=np.isin(lab,[0,5]); act[m]=0; tech[m]='NONE'
changed=(lab!=ml_lab)
# confidence: overrides that fixed a disagreement get 0.93, others keep ML conf but cap
conf=np.where(A|(L&~A), np.where(changed,0.93,np.minimum(conf,0.97)), conf)
out_a=pd.DataFrame({'id':t.id,'label':lab,'techniques':tech,'action':act,'confidence':np.clip(conf,0.05,0.985).round(4)})
out_a.to_csv('/home/claude/submission_v5a_rules_only.csv',index=False)
print('v5a changed labels',changed.sum(),f'({changed.mean():.2%})'); print(pd.Series(lab).value_counts(normalize=True).sort_index().round(4).to_dict())
# ---------- optional transductive template smoothing ----------
NOISE=re.compile(r'^[^aeiou\u0980-\u09ff]{5,}$|^(zzqwpo|wkrjfejejrn|mnbvcx|qzxvbn|xkcdpl|asdkjh)$')
def key(raw):
    s=skel(norm(raw)); toks=[w for w in re.sub(r'[^a-z\u0980-\u09ffA-Z ]',' ',s).split() if not NOISE.match(w)]
    return ' '.join(toks)
t['k']=t.text.map(key); t['lab']=lab; t['conf']=conf; t['ov']=A|L
g=t.groupby('k')
size=g.lab.transform('size')
w=pd.get_dummies(t.lab).mul(t.conf,axis=0); 
share=w.groupby(t.k).transform('sum'); share=share.div(share.sum(axis=1),axis=0)
maj=share.values.argmax(1); majp=share.values.max(1)
sm=(size>=8)&(majp>=0.75)&(maj!=t.lab.values)&(~t.ov.values)
lab2=np.where(sm,maj,t.lab.values)
print('clusters(size>=8)',(g.size()>=8).sum(),' smoothed rows',sm.sum())
act2=act.copy(); tech2=tech.copy()
# for flipped rows copy modal action/techniques from the cluster's majority-label rows
dfm=pd.DataFrame({'k':t.k,'lab':t.lab,'act':act,'tech':tech})
mode_act=dfm.groupby(['k','lab']).act.agg(lambda x:x.mode().iat[0]); mode_tech=dfm.groupby(['k','lab']).tech.agg(lambda x:x.mode().iat[0])
for i in np.where(sm)[0]:
    kk=(t.k.iat[i],maj[i]); act2[i]=mode_act.get(kk,act[i]); tech2[i]=mode_tech.get(kk,tech[i])
m=np.isin(lab2,[0,5]); act2[m]=0; tech2[m]='NONE'
conf2=np.where(sm,np.minimum(conf,0.8),conf)
pd.DataFrame({'id':t.id,'label':lab2,'techniques':tech2,'action':act2,'confidence':np.clip(conf2,0.05,0.985).round(4)}).to_csv('/home/claude/submission_v5b_rules_plus_cluster.csv',index=False)
