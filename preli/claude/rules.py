"""Guide-derived rule teacher for ScamShield BD (priority 1>2>4>3>5>0). Built ONLY from the problem statement + sample.csv."""
import re, unicodedata
from skel import skel
ZW=re.compile(r'[\u200b-\u200f\u2060\ufeff]')
HOMO=str.maketrans({'а':'a','е':'e','о':'o','р':'p','с':'c','х':'x','у':'y','і':'i','ѕ':'s','ԁ':'d','ӏ':'l','ο':'o','ι':'i','α':'a'})
def norm(s):
    s=unicodedata.normalize('NFKC',s); s=ZW.sub('',s).translate(HOMO).lower()
    s=re.sub(r'(.)\1{2,}',r'\1',s)          # PIIIN -> pin
    s=s.replace('p1n','pin').replace('0tp','otp').replace('fr33','free').replace('a1rdrop','airdrop')
    s=re.sub(r'\b[a-z]{6,}\b(?<!ashe)',lambda m:m.group(0),s)
    return s
SENS=r'(pin|otp|password|passcode|cvv|card (number|details)|nid|verification code|login code|security code|\bcode\b|6[- ]?digit|ওটিপি|পিন|পাসওয়ার্ড|এনআইডি|কোড|কার্ড|code ta|code ta)'
ASK=r'(send|share|give|tell|reply|provide|enter|submit|forward|confirm(?!ation)\b|pathan|pathaben|din\b|dao\b|bolen|bolo\b|bolun|dibi|deben|জানান|পাঠান|দিন|বলুন|বলো|দেবেন|দাও|পাঠাও|ta bolen|ta bolo|ta din)'
NEGW=r'(only with the rider|trip pin|do not share|don.t share|never share|not share|never ask|never call|will never|kauke (share )?(korben|bolben|dien)|kauke dien na|কাউকে (শেয়ার )?(দেবেন|করবেন|বলবেন) না|bolben na|dibi na|dien na|ask korbe na|চাইবে না|শেয়ার করবেন না|কখনো)'
MISTAKE=r'(accident|by mistake|vul kore|bhul kore|vul e|bhul e|bhul hoye|vul hoye|ভুল করে|ভুলে|wrong number|mistaken|ভুল নম্বর)'
RETURN=r'(send (it )?back|return|ferot|ফেরত|refund me|pathiye din|pathan)'
def feats(raw):
    t=norm(raw); parts=[p for p in re.split(r'\|\|\|',t)]
    return t,parts
def classify(raw):
    t,parts=feats(raw)
    warn=bool(re.search(NEGW,t))
    scamwarn=bool(re.search(r'(eta scam|this is a scam|it.?s a scam|scam bhai|block kor|ignore korsi|scam hote pare|স্ক্যাম|সাবধান|sabdhan|beware|alert:? (fraud|scam))',t))
    sens=bool(re.search(SENS,t)); ask=bool(re.search(ASK,t))
    # does an imperative request for a secret exist?  "<verb> ... <secret>" or "<secret> ... <verb>" within ~40 chars, not negated
    req_secret=False
    for m in re.finditer(r'[^.!?।|]+',t):
        seg=m.group(0)
        if re.search(SENS,seg) and re.search(ASK,seg) and not re.search(NEGW,seg):
            req_secret=True
    mistake=bool(re.search(MISTAKE,t)) and bool(re.search(RETURN,t)) and bool(re.search(r'(tk|taka|টাকা|\d{2,})',t))
    prize=bool(re.search(r'(you won|won |winner|jit(echen|echhen|echhe|echo)|jitar|lucky draw|lottery|giveaway|airdrop|mystery box|spin and win|gift|জিতেছ|লাকি ড্র|ক্যাশব্যাক জিত|অভিনন্দন|congratulation|select(ed|)|selected|hoyechen|you.ve been chosen|free (iphone|usdt|data|gift)|reward|bonus|বোনাস|উপহার|ঈদ বোনাস)',t))
    claim=bool(re.search(r'(claim|click|link|fee|charge|pay|processing|survey|connect (your )?wallet|whatsapp|call|যোগাযোগ|ফি|চার্জ|ক্লিক|লিংক|url|hxxp|download|install|pathan|share|reply|dial)',t))
    parcel=bool(re.search(r'(parcel|package|courier|customs|delivery|deliver|কুরিয়ার|পার্সেল|ডেলিভারি|shipment|booking|consignment|tracking)',t))
    paylink=bool(re.search(r'(fee|charge|duty|storage|pay |payment|confirm (new )?address|address|update address|link|hxxp|https?|click|ঠিকানা|চার্জ|ফি|পরিশোধ|লিংকে|ক্লিক|url|call korun|call us|ekhoni call|verify)',t)) and not re.search(r'(cod |cash on delivery|tk [\d,]+ ready rakhun|ready rakhun|has been shipped|out for delivery|arrived at|collect from|will arrive|ajke deliver hobe\.?$|আজ ডেলিভারি হবে|ship hoyeche)',t)
    job=bool(re.search(r'(earn|income|আয়|work.from.home|part.time|data entry|recruit|registration fee|invest|double your|crypto|usdt|btc|signal|task|like (and )?(youtube|videos?|page)|watch and like|commission|per day|weekly profit|visa fee|abroad job|abroad|ঘরে বসে|লোক নিচ্ছে|নিচ্ছে|job|hiring|vacancy|guarantee|freelanc|withdraw|profit|allowance arrears|রেজিস্ট্রেশন ফি|ডাবল)',t))
    jobfee=bool(re.search(r'(fee|deposit|security|invest|registration|join|whatsapp|telegram|task|commission|guarantee|profit|রেজিস্ট্রেশন|ফি|যোগ|income|earn|আয়)',t))
    promo=bool(re.search(r'(offer|sale|discount|% ?(off|discount|cashback)|%|emi|chhar|ছাড়|অফার|সেল|buy|order now|admission|combo|free .*get|bonus|বোনাস|pack|gb|dial \*|code [a-z0-9]{3,}|download|app|voucher|installment|apartment|book your|কিনুন|চাই)',t))
    legit_ctx=bool(re.search(r'(successful|সফল|received|peyechen|credited|debited|balance|trxid|ট্রানজেকশন|ব্যালেন্স|bill|বিল|due|valid|বৈধ|meyad|never share|do not share|your order|will arrive|out for delivery|interview|appointment|meeting|class|exam|reminder|routine|welcome to|rider|library|hub|collect)',t))
    # ---- priority 1
    if req_secret and not (warn and not re.search(r'(pathan|send|share|bolen|din|dao)\b.*(otp|pin|code)',t) and legit_ctx and not re.search(r'(helpline|officer|agent|support|fraud control|verif)',t)):
        # FAQ / security-notice guard
        if not (warn and legit_ctx and not re.search(r'(reply|helpline|officer|agent|fraud control)',t)):
            return 1
    if mistake and not scamwarn: return 1
    if scamwarn and not req_secret: return 0
    # ---- priority 2
    if prize and claim and not legit_ctx_strong(t): return 2
    # ---- priority 4
    if parcel and paylink: return 4
    # ---- priority 3
    if job and jobfee and not re.search(r'(interview|bring cv|reply yes to confirm)',t): return 3
    # ---- priority 5
    if promo and not legit_ctx: return 5
    return 0
def legit_ctx_strong(t):
    return bool(re.search(r'(you earned tk [\d,]+ cashback on your bill|has been credited|successful)',t))
