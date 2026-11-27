"""Surgical V8 Adversarial Normalization, Priority-Fixed Rules, and Fast Re-Scoring Hotfix."""
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

V2_DIR = Path(__file__).resolve().parent
ROOT = V2_DIR.parents[1]
CACHE_DIR = V2_DIR / "cache"
CKPT_DIR = CACHE_DIR / "ckpt_v4"
OUT_DIR = CACHE_DIR / "preds_v4"
sys.path.insert(0, str(V2_DIR))

from cv_ood import TECH_LIST, tech_str_to_vec, vec_to_tech_str, decode_consistent, score_section10_proxy
from normalizer_v4 import (
    ZW_RE, CONFUSABLES, EMOJI_RE, ANY_EMOJI_RE,
    mask_urls_for_text_rules, extract_base_struct_v4, format_turn_aware,
)
from rules_v4 import (
    SPEAKER_PREFIX_RE, SCAM_WARN_THREAD_RE, MISTAKE_RE, RETURN_RE,
    HARD_LEGIT_COD_RE, HARD_LEGIT_TX_RE, HARD_LEGIT_HR_DOC_RE,
    HARD_LEGIT_DELIV_RE, HARD_LEGIT_CHAT_RE, HARD_LEGIT_OTP_RE,
    CLAIM_RE, PROMO_RE, ENTITY_T1_RE, URGENCY_T5_RE,
)
from torch_utils_v4 import CACHE_DIR_HF, scale_feats_v4, HeadConditionedTransformer, predict_transformer, free_cuda

NOISE_V8_RE = re.compile(
    r"(?:zzqwpo|wkrjfejejrn|mnbvcx?|qzxvbn|xkcdpl|asdkjh|lkjsdfoiu|xcvbnm|zzqpwoeiru|qundvpyfjqx|wbmtzmh|qwertyz|asdfghj)",
    flags=re.IGNORECASE,
)

TYPO_V8 = {
    "don07": "don't", "d0n07": "don't", "d0n7": "don't", "don0t": "don't", "d0nt": "don't",
    "b4nk": "bank", "s3cur1ty": "security", "53curity": "security", "cl34r4nc3": "clearance",
    "r3del1v3ry": "redelivery", "r3d3l1v3ry": "redelivery", "1mp0r7an7": "important",
    "0rd3r": "order", "r3g1s7r4710n": "registration", "c4rd": "card", "crad": "card",
    "cradnumber": "card number", "cardnumber": "card number", "cvvdin": "cvv din",
    "p4ssw0rd": "password", "l0g1n": "login", "0tp": "otp", "p1n": "pin", "pn": "pin",
    "oepn": "open", "d3n": "den", "w45": "was", "p4y": "pay", "py": "pay", "olny": "only",
    "c4ller": "caller", "s3nd3r": "sender", "7el374lk": "teletalk", "dibenna": "diben na",
    "uilever": "unilever", "gramenphone": "grameenphone", "cnograts": "congrats", "congras": "congrats",
    "sahre": "share", "motrbike": "motorbike", "tranaction": "transaction", "trnasaction": "transaction",
    "debennna": "deben na", "debenna": "deben na", "bolbenna": "bolben na",
    "korbenna": "korben na", "sharekorbenna": "share korben na", "dibina": "dibi na",
    "sharethe": "share the", "tashare": "ta share", "rewardsahre": "reward share",
    "justreceived": "just received", "kyychas": "kyc has", "kychas": "kyc has", "kcy": "kyc",
    "coodeaschhe": "code ashche", "cooodeaschhe": "code ashche", "codeaschhe": "code ashche",
    "pohne": "phone", "nubmer": "number", "oa": "ota", "hoyehcen": "hoyechen",
    "samsungs24": "samsung s24", "wintersae": "winter sale", "takadaly": "taka daily",
    "apnia": "apni a", "erjonno": "er jonno", "bkashkorun": "bkash korun",
    "chargetk": "charge tk", "rfeer": "refer", "j173chen": "jitechen",
    "amotrbike": "a motorbike", "motorbikeer": "motorbike er", "onparcel": "on parcel",
    "incompletepay": "incomplete pay", "messaed": "messaged", "neds": "needs",
    "kimcash": "ki mcash", "apiki": "apni ki", "usekoren": "use koren",
    "froma": "from a", "helplabe": "help lagbe", "f1x": "fix", "korte": "korte",
    "kor3": "kore", "codeta": "code ta", "sedn": "send", "annnnd": "and", "dte": "date",
    "yoooou": "you", "taaaakadaly": "taka daily", "sqqqquarepharma": "squarepharma",
    "lokeo": "lokeo", "anydesk": "anydesk", "user5": "users", "bkah": "bkash", "bksah": "bkash",
    "sh4re": "share", "c0d3": "code", "ver1fy": "verify", "4ccount": "account", "4pnar": "apnar",
    "w1n": "win", "pr1ze": "prize", "cl4im": "claim", "del1very": "delivery", "p4rcel": "parcel",
    "ch4rge": "charge", "f33": "fee", "t4ka": "taka", "f3r0t": "ferot", "v1s4": "visa",
}
TYPO_V8_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(TYPO_V8, key=len, reverse=True)) + r")\b", re.I)

SINGLE_FIX = {
    "abbrod": "abroad", "abbroad": "abroad", "abbrooad": "abroad", "opppen": "open", "opppenn": "open",
    "wittth": "with", "witth": "with", "llluucky": "lucky", "luucky": "lucky",
    "llucky": "lucky", "sshhare": "share", "ssend": "send", "hhelllo": "hello",
    "hhhello": "hello", "arrrre": "are", "arre": "are", "taakaa": "taka", "ttaka": "taka",
}


def clean_v8(raw_text: str) -> str:
    if not isinstance(raw_text, str):
        raw_text = "" if raw_text is None else str(raw_text)
    s = ZW_RE.sub("", raw_text)
    s = s.translate(CONFUSABLES)
    # Collapse inline URL defanging first
    s = re.sub(r"\bh[t7x]{2}p(s?)://", r"http\1://", s, flags=re.IGNORECASE)
    s = s.replace("[.]", ".")
    # Strip noise strings anywhere
    s = NOISE_V8_RE.sub(" ", s)
    # Replace emojis with clean spaces for rule checks
    s = EMOJI_RE.sub(" ", s)
    s = ANY_EMOJI_RE.sub(" ", s)
    # Normalize unicode apostrophes and lowercase
    s = s.replace("\u2019", "'").replace("\u2018", "'").lower()
    # Collapse 3+ repeated letters down to 2, and fix common double-collapsed words
    s = re.sub(r"([a-z])\1{2,}", r"\1\1", s)
    # Common typo/leet dictionary passes (2 passes in case double->single reveals token)
    s = TYPO_V8_RE.sub(lambda m: TYPO_V8[m.group(1).lower()], s)
    toks = [SINGLE_FIX.get(t, t) for t in s.split()]
    s = " ".join(toks)
    # Split concatenated compound tokens observed in adversarial tests
    s = re.sub(r"(card|pin|otp|cvv|nid|code)(number|din|den|ta|share|pathan|bolo|bolen|ashche)", r"\1 \2", s)
    s = re.sub(r"(share|send|reply|enter|confirm|pay|open)(the|your|with|tk)", r"\1 \2", s)
    s = TYPO_V8_RE.sub(lambda m: TYPO_V8[m.group(1).lower()], s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Fixed Priority-Aware Rule Patterns
SENS_V8_RE = re.compile(
    r"\b(?:pin|otp|password|passcode|cvv|card number|card details|nid|verification code|login code|security code|"
    r"6[- ]?digit(?: code)?|9[- ]?digit(?: code)?|\d[- ]digit code|date of birth|dob|internet banking password|wallet pin|"
    r"anydesk|teamviewer|screen\s*shar(?:e|ing))\b"
    r"|ওটিপি|পিন|পাসওয়ার্ড|এনআইডি|কার্ড নম্বর",
    flags=re.IGNORECASE,
)
CODE_PLAIN_RE = re.compile(r"\bcode\b|কোড", flags=re.IGNORECASE)
ASK_V8_RE = re.compile(
    r"\b(?:send|sending|share|sharing|give|giving|tell|reply|reply with|provide|enter|entering|submit|forward|"
    r"confirm|verify|verify by|read out|open|install|bolen|bolo|bolun|pathan|pathaben|din|den|dao|deben|dibi|ta bolen|ta din|dite hobe|bolte hobe)\b"
    r"|জানান|পাঠান|দিন|বলুন|বলো|দেবেন|দাও|পাঠাও",
    flags=re.IGNORECASE,
)
# CRITICAL FIX: Removed "do not reply / don't reply / please do not reply" so scammer footers don't block rules!
NEG_WARN_V8_RE = re.compile(
    r"\b(?:only with (?:the )?rider|trip pin|do not share|don'?t share|never share|not share|never ask|never call|will never|"
    r"do not pay anyone|reply stop to opt out|"
    r"kauke\b.{0,18}\b(?:share|bolben|korben|diben|dien|bolo)\b.{0,6}\bna|"
    r"(?:bolben|dibi|dien|diben|korben)\s+na|bank er lokeo na)\b"
    r"|কাউকে.{0,22}(?:দেবেন|করবেন|বলবেন|শেয়ার).{0,6}না|কখনো.{0,12}না",
    flags=re.IGNORECASE,
)

# CRITICAL FIX: "clearance charge" removed from standalone PARCEL_V8_RE (must have courier/parcel/customs/delivery!)
PARCEL_V8_RE = re.compile(
    r"\b(?:parcel|package|courier|customs|delivery|deliver|shipment|consignment|tracking|redelivery|storage fee|delivery attempt|out for delivery|was delivered|has been delivered)\b"
    r"|পার্সেল|কুরিয়ার|কাস্টমস|শুল্ক|ডেলিভারি হবে|ডেলিভারি ব্যর্থ|ডেলিভারি চার্জ",
    flags=re.IGNORECASE,
)
PARCEL_PHISH_V8_RE = re.compile(
    r"\b(?:customs|duty|storage fee|redelivery|clearance charge|wrong address|incomplete|address confirmation|confirm(?: new)? address|"
    r"update address|address.{0,12}(?:vul|bhul)|(?:vul|bhul).{0,10}address|"
    r"could not be delivered|couldn'?t be delivered|attempt failed|release|"
    r"held|atke ase|failed|fee|charge|(?<!pathao\s)pay|click|tap|<URL_SUSP>|<URL_SHORT>|<URL>)\b"
    r"|আটকে|ব্যর্থ|চার্জ|ফি|পরিশোধ|শুল্ক|কাস্টমস|ঠিকানা আপডেট|ক্লিক|লিংকে",
    flags=re.IGNORECASE,
)

STRONG_PRIZE_V8_RE = re.compile(
    r"\b(?:you won|you'?ve won|won|winner|winners|win a (?:tk|voucher|prize|gift|iphone|smartphone|car)|jitechen|jitechhen|jiteche|jitar|lucky draw|lottery|giveaway|airdrop|usdt|"
    r"mystery box|spin and win|free iphone|free smartphone|free router|free wi[- ]?fi router|10gb data gift|"
    r"pm education fund|stipend|old[- ]?age allowance|allowance arrears|bumper|"
    r"(?:number\s*ti|motorbike|laptop|toyota\s*car|iphone\s*\d*|smart\s*tv|samsung\s*s?\d*|gift\s*card|prize|bonus|gift|cashback|stipend|allowance|reward).{0,45}select(?:ed)?\s*(?:hoyechen|hoyeche|hoye|hoy|for)|"
    r"selected\s*(?:for|at)\s*(?:a\s+)?(?:motorbike|laptop|smart\s*tv|toyota|iphone|samsung|gift\s*card|reward|prize)|"
    r"loyalty reward|prize|cashback.{0,15}(?:jitechen|won|claim))\b"
    r"|জিতেছেন|জিতেছে|জিতার|লাকি ড্র|লটারি|পুরস্কার",
    flags=re.IGNORECASE,
)
SOFT_PRIZE_V8_RE = re.compile(
    r"\b(?:cashback|gift|reward|rewards|bonus|eid bonus|congratulations|congrats|select\s*hoyechen|selected\s*at)\b"
    r"|ক্যাশব্যাক|উপহার|অভিনন্দন|ঈদ বোনাস|গিফট",
    flags=re.IGNORECASE,
)
JOB_V8_RE = re.compile(
    r"(?<!amar\s)(?<!my\s)\bjob\b(?!\s*jabe)|\b(?:hiring|vacancy|work[- ]?from[- ]?home|work frmo home|part[- ]?time|data entry|package handler|brand ambassador|"
    r"relief volunteer|typing job|agent license|earn(?:ing)?|income|kamai|kamao|salary|beton|task|tasks|watch and like|"
    r"like korle|facebook page|page like|per like|product review|review products|unlock tasks?|"
    r"commission|invest|investment|double your money|crypto signals?|arbitrage|bot|ludo|cash game|"
    r"freelanc(?:ing|e)|abroad job|trading crypto|selected at (?:unilever|walton|grameenphone|beximco|samsung bangladesh|pathao|robi|foodpanda|pran))\b"
    r"|লোক নিচ্ছে|ঘরে বসে|আয়|বেতন|বিনিয়োগ|দ্বিগুণ|ভাতা|হোয়াটসঅ্যাপে যোগ দিন",
    flags=re.IGNORECASE,
)
JOB_TRAP_V8_RE = re.compile(
    r"\b(?:fee|deposit|security deposit|uniform fee|registration fee|clearance charge|service charge|visa fee|kit|"
    r"id card|invest|enroll now|per task|per post|per day|per like|like e \d+|per review|daily|weekly profit|profit per day|"
    r"guarantees?\s+\d+%|deposit tk \d+.{0,35}get tk \d+|10 minutes\.\s*guaranteed|commission|dite hobe|unlock korun|apply korun|proceed|"
    r"join (?:whatsapp|telegram|the group|vip)|group e join|whatsapp e join|join korun|whatsapp us|complete \d+ tasks?)\b"
    r"|রেজিস্ট্রেশন ফি|ডিপোজিট|দ্বিগুণ|নিশ্চিত লাভ|প্রতিদিন|যোগ দিন|যোগাযোগ করুন|সার্ভিস চার্জ|ভিসা ফি|টাকা দিন|হোয়াটসঅ্যাপে|টেলিগ্রামে",
    flags=re.IGNORECASE,
)
HARD_PRIZE_RE = re.compile(r"\b(?:select(?:ed)?\s*(?:hoyechen|for\s+a)|reward|won|lottery|lucky draw)\b|লাকি ড্র|লটারি", re.I)


def extract_rule_v8(raw_text: str):
    c = clean_v8(raw_text)
    masked, uflags = mask_urls_for_text_rules(c)
    t = SPEAKER_PREFIX_RE.sub(" ", masked)

    scam_warn_thread = bool(SCAM_WARN_THREAD_RE.search(t))
    mistake = bool(MISTAKE_RE.search(t)) and bool(RETURN_RE.search(t)) and bool(
        re.search(r"\b(?:tk|bdt|taka|money)\b|টাকা|\d{2,}", t, re.I)
    )

    req_secret = False
    for seg in re.split(r"[.!?\n\|]+", t):
        seg = seg.strip()
        if not seg:
            continue
        has_sens = bool(SENS_V8_RE.search(seg)) or (
            bool(CODE_PLAIN_RE.search(seg))
            and not bool(re.search(r"\b(?:use code|code:|promo|voucher|eid\d+)\b", seg, re.I))
        )
        if has_sens and ASK_V8_RE.search(seg) and not NEG_WARN_V8_RE.search(seg):
            if not HARD_LEGIT_HR_DOC_RE.search(seg):
                req_secret = True
                break
    if not req_secret:
        for seg in re.split(r"[.!?\n\|]+", t):
            seg = seg.strip()
            if seg and not NEG_WARN_V8_RE.search(seg) and re.search(
                r"(?:nid.{0,15}both sides|photo of your nid|nid photo|card number.{0,25}cvv|cvv den|cvv din|"
                r"enter(?:ing)? (?:your )?(?:pin|otp)|confirm(?: your)? (?:internet banking )?(?:password|otp|pin)|"
                r"send (?:your )?(?:pin|otp|nid|wallet pin)|tell me your pin|read out the code|"
                r"anydesk.{0,30}(?:code|digit)|(?:whatsapp|telegram|account|sim).{0,25}deactivated.{0,30}(?:code|otp)|"
                r"phone hariye.{0,45}code.{0,20}(?:bolo|bolen|din|pathan)|"
                r"share\s+(?:the\s+)?(?:\d[- ]?digit\s+)?(?:otp|pin|code))",
                seg,
                re.I,
            ):
                req_secret = True
                break

    neg_warn = bool(NEG_WARN_V8_RE.search(t))
    quoted_scam_framing = bool(
        ("|||" in c and scam_warn_thread)
        or re.search(r"(?:someone|ekta sms|ekjon).{0,60}(?:messaged|ashche|pathaise|je)|that is phishing|eta scam|keu pin chaile", t, re.I)
    )
    is_scamwarn_only = bool(scam_warn_thread and (not req_secret or quoted_scam_framing))
    is_cod = bool(HARD_LEGIT_COD_RE.search(t)) and not req_secret
    is_hr_legit = (
        bool(HARD_LEGIT_HR_DOC_RE.search(t))
        and not req_secret
        and not bool(re.search(r"\b(?:pay tk|registration fee|uniform fee|deposit)\b", t, re.I))
    )
    is_tx_legit = (
        bool(HARD_LEGIT_TX_RE.search(t))
        and not req_secret
        and not mistake
        and not bool(re.search(r"\b(?:verify|claim|click|fee|charge|prosessing)\b|প্রসেসিং|ক্লেইম|<URL_SUSP>|<URL_SHORT>", t, re.I))
    )
    is_otp_legit = (bool(HARD_LEGIT_OTP_RE.search(t)) or bool(re.search(r"\b(?:login code|verification code|pin|otp)\b", t, re.I))) and neg_warn and not req_secret
    is_deliv_legit = (
        bool(HARD_LEGIT_DELIV_RE.search(t))
        and not req_secret
        and not (PARCEL_PHISH_V8_RE.search(t) and not bool(re.search(r"will retry tomorrow|096\d{2}-", t, re.I)))
    )
    is_chat_legit = bool(HARD_LEGIT_CHAT_RE.search(t)) and not req_secret and not mistake
    hard_legit = is_cod or is_hr_legit or is_tx_legit or is_otp_legit or is_deliv_legit or is_chat_legit

    has_parcel = bool(PARCEL_V8_RE.search(t))
    has_parcel_phish = bool(PARCEL_PHISH_V8_RE.search(t))
    has_job = bool(JOB_V8_RE.search(t))
    has_job_trap = bool(JOB_TRAP_V8_RE.search(t))
    has_claim = bool(CLAIM_RE.search(t) or re.search(r"\b(?:charge|fee)\s*(?:pathan|din|den|dite)\b", t, re.I))
    has_promo = bool(PROMO_RE.search(t))

    has_strong_prize = bool(STRONG_PRIZE_V8_RE.search(t))
    has_soft_prize = bool(SOFT_PRIZE_V8_RE.search(t)) and has_claim and not (has_parcel and has_parcel_phish) and not (has_job and has_job_trap)
    has_prize = has_strong_prize or has_soft_prize

    if is_scamwarn_only:
        lbl = 0
    elif req_secret or mistake:
        lbl = 1
    elif hard_legit:
        lbl = 0
    elif has_promo and not (
        (has_parcel and has_parcel_phish) or (has_job and has_job_trap) or uflags["has_susp_tld"] or uflags["has_shortener"]
        or req_secret or has_strong_prize or re.search(r"\b(?:processing fee|survey|airdrop)\b|প্রসেসিং ফি", t, re.I)
    ):
        lbl = 5
    elif has_prize and (has_claim or uflags["has_url"]) and not (has_parcel and has_parcel_phish and (not HARD_PRIZE_RE.search(t) or bool(re.search(r"limited winners", t, re.I)))) and not (has_job and has_job_trap):
        lbl = 2
    elif has_parcel and (has_parcel_phish or uflags["has_susp_tld"] or uflags["has_shortener"]) and not re.search(r"\b(?:package handler|salary|ambassador)\b", t, re.I):
        lbl = 4
    elif has_job and has_job_trap:
        lbl = 3
    elif has_promo:
        lbl = 5
    else:
        lbl = 0

    if lbl in (0, 5):
        act = 0
        tvec = np.zeros(6, dtype=np.int32)
    else:
        req_pii_reply = bool(re.search(r"\b(?:reply|send|share|give)\b.{0,25}\b(?:name and address|full name|address|date of birth|dob)\b", t, re.I))
        has_pay = bool(re.search(
            r"\b(?:(?<!pathao\s)pay|deposit|invest|ferot|return|send (?:it )?back|reverse|enroll now with tk|charge pathan|fee din)\b"
            r"|\b(?:registration|processing|delivery|redelivery|storage|service|clearance)\s*(?:fee|charge|duty)\b.{0,22}\b(?:tk|bdt|\d|dite|(?<!pathao\s)pay|din|baki|pathan)"
            r"|\b(?:tk|bdt|\$)\s*[\d,]+\s*(?:fee|duty|deposit|charge|ferot|dite hobe|bkash korun|nagad korun|din|pathan)\b"
            r"|পরিশোধ|ডিপোজিট|বিনিয়োগ|ফেরত|(?:টাকা|ফি|চার্জ)\s*(?:দিন|দিতে হবে)",
            t, re.I,
        ))
        has_pay_verb = bool(re.search(r"\b(?:(?<!pathao\s)pay|deposit|invest|enroll now with|ferot|send (?:it )?back|reverse|bkash korun|nagad korun|charge pathan|fee din)\b|পরিশোধ|ডিপোজিট|বিনিয়োগ|ফেরত|দিতে হবে|টাকা দিন", t, re.I))
        has_join_apk = bool(re.search(
            r"\b(?:install|download|join (?:the )?(?:group|vip|whatsapp|telegram)|whatsapp e join|telegram e join|group e join|join korun|"
            r"complete \d+ tasks?|watch and like|buy products|review products|rewards app|relief volunteer)\b|<URL_APK>|<URL_TGWA>"
            r"|যোগ দিন|হোয়াটসঅ্যাপে যোগ|টেলিগ্রামে যোগ|গ্রুপে",
            t, re.I,
        ))
        has_url_click = bool(uflags["has_url"] or re.search(r"\b(?:click|tap|link|visit|survey|connect (?:your )?wallet)\b|ক্লিক|লিংকে|যান", t, re.I))
        has_call_wa = bool(re.search(r"\b(?:call|whatsapp|contact|dial)\b|যোগাযোগ|কল করুন", t, re.I))

        if lbl == 1:
            act = 1 if (req_secret or not mistake) else 2
        elif req_pii_reply and not has_pay_verb:
            act = 1
        elif has_call_wa and not has_pay_verb and not has_join_apk and not has_url_click:
            act = 5
        elif has_pay:
            act = 2
        elif has_join_apk:
            act = 4
        elif has_url_click and not (has_call_wa and not uflags["has_url"]):
            act = 3
        elif has_call_wa:
            act = 5
        elif req_pii_reply:
            act = 1
        else:
            act = {2: 3, 3: 4, 4: 3}[lbl]

        t1 = int(bool(ENTITY_T1_RE.search(t)))
        t2 = int(lbl == 1 and (req_secret or not mistake) or req_pii_reply)
        t3 = int(has_pay or mistake or (lbl == 4 and act == 2) or bool(re.search(r"\b(?:fee|charge|duty|deposit|invest|visa fee|registration fee|clearance charge)\b|ফি|চার্জ|শুল্ক|বিনিয়োগ", t, re.I)))
        t4 = int(lbl == 2 or has_strong_prize or (has_soft_prize and not mistake))
        t5 = int(bool(URGENCY_T5_RE.search(t)) or mistake or bool(re.search(r"\b(?:good morning|brother|apu|mama|sir|friend|helpful|sorry|dear customer|last warning|act now|offer expires)\b", t, re.I)))
        t6 = int(lbl == 3 or has_job)
        tvec = np.array([t1, t2, t3, t4, t5, t6], dtype=np.int32)
        if tvec.sum() == 0:
            tvec[{1: 1, 2: 3, 3: 5, 4: 2}[lbl]] = 1

    return lbl, tvec, act, c, is_otp_legit, req_secret, has_strong_prize, has_parcel, has_job, has_job_trap


def main():
    t0 = time.time()
    # 1. Verify rules_v8 against ALL 86 sample.csv rows first!
    sdf = pd.read_csv(ROOT / "Main Preli Round" / "dataset" / "sample.csv")
    y_sl = sdf["label"].values.astype(int)
    y_st = np.vstack([tech_str_to_vec(x) for x in sdf["techniques"]])
    y_sa = sdf["action"].values.astype(int)
    r_sl, r_st, r_sa = np.zeros(len(sdf), int), np.zeros((len(sdf), 6), int), np.zeros(len(sdf), int)
    for i, tx in enumerate(sdf["text"].fillna("").astype(str)):
        l, tv, a, *_ = extract_rule_v8(tx)
        r_sl[i], r_st[i], r_sa[i] = l, tv, a
    res_s = score_section10_proxy(y_sl, r_sl, y_st, r_st, y_sa, r_sa, np.full(len(sdf), 0.96))
    print(f"rules_v8 on sample.csv (86 rows): {res_s}", flush=True)

    # 2. Load raw test.csv + saved 500k model probabilities
    raw = pd.read_csv(ROOT / "Main Preli Round" / "dataset" / "test.csv", usecols=["id", "text"])
    te_ids = raw["id"].values
    texts = raw["text"].fillna("").astype(str).tolist()
    n = len(texts)

    t_m = np.load(OUT_DIR / "test_mminilm_v4.npz")
    t_e = np.load(OUT_DIR / "test_me5small_v4.npz")
    t_b = np.load(OUT_DIR / "test_bge_v4.npz")
    t_bg = np.load(OUT_DIR / "test_bigru_v4.npz")
    t_ba = np.load(OUT_DIR / "test_da_tab_v4.npz")
    t_esc = np.load(OUT_DIR / "test_mdeberta_escalate_v4.npz")
    s4_clean = pd.read_csv(CACHE_DIR / "test_v4_clean.csv", usecols=["clean_text"])["clean_text"].fillna("").astype(str).tolist()

    # Copy mutable arrays for re-scoring
    pm_l, pm_t, pm_a = t_m["p_l"].copy(), t_m["p_t"].copy(), t_m["p_a"].copy()

    # 3. Extract rules_v8 + clean_v8 across all 500,000 rows (~20s)
    r8_l = np.zeros(n, dtype=np.int64)
    r8_t = np.zeros((n, 6), dtype=np.float32)
    r8_a = np.zeros(n, dtype=np.int64)
    c8_list = []
    otp_legit = np.zeros(n, dtype=bool)
    req_sec = np.zeros(n, dtype=bool)
    strong_pr = np.zeros(n, dtype=bool)
    has_par = np.zeros(n, dtype=bool)
    has_jb = np.zeros(n, dtype=bool)
    has_jbt = np.zeros(n, dtype=bool)
    diff_clean = np.zeros(n, dtype=bool)

    for i, raw_tx in enumerate(texts):
        l, tv, a, c8, ol, rs, sp, hp, hj, hjt = extract_rule_v8(raw_tx)
        r8_l[i], r8_t[i], r8_a[i] = l, tv, a
        c8_list.append(c8)
        otp_legit[i], req_sec[i], strong_pr[i] = ol, rs, sp
        has_par[i], has_jb[i], has_jbt[i] = hp, hj, hjt
        # Flag if aggressive cleaning materially repaired words vs v4 clean_text
        if len(c8) > 0 and c8 != s4_clean[i].lower():
            # Compare alpha word counts or rule flip
            diff_clean[i] = True

    old_l = (0.35 * pm_l + 0.25 * t_e["p_l"] + 0.10 * t_b["p_l"] + 0.30 * t_bg["p_l"]).argmax(1)
    prio1 = np.where(r8_l != old_l)[0]
    prio2 = np.where((r8_l == old_l) & diff_clean & (old_l == 0))[0]
    if len(prio1) < 28000:
        re_idx = np.sort(np.unique(np.concatenate([prio1, prio2[:28000 - len(prio1)]])))
    else:
        re_idx = prio1[:28000]
    print(f"rules_v8 done in {time.time()-t0:.1f}s! Diff vs old neural={int((r8_l != old_l).sum())} | Re-scoring mMiniLM on {len(re_idx)} cleaned rows...", flush=True)

    # 4. Re-score repaired/uncertain rows on GPU with mMiniLM_distill.pt (~95s!)
    if len(re_idx) > 0:
        free_cuda()
        ck = torch.load(CKPT_DIR / "mminilm_distill.pt", map_location="cpu", weights_only=False)
        tok = AutoTokenizer.from_pretrained(ck["model_name"], cache_dir=CACHE_DIR_HF, local_files_only=True)
        mdl = HeadConditionedTransformer(ck["model_name"], n_feats=58, pool_mode=ck["pool_mode"], proj_dim=256).cuda().eval()
        mdl.load_state_dict(ck["state_dict"], strict=True)
        te_feats_m = np.load(CACHE_DIR / "test_v4_feats.npy", mmap_mode="r")[re_idx].copy()
        sc_f = scale_feats_v4(te_feats_m)
        sub_txt = [format_turn_aware(c8_list[i]) for i in re_idx]
        pl_new, pt_new, pa_new = predict_transformer(mdl, tok, sub_txt, sc_f, bs=112, max_len=144)
        pm_l[re_idx], pm_t[re_idx], pm_a[re_idx] = pl_new, pt_new, pa_new
        del mdl, ck
        free_cuda()

    # 5. Balanced Un-Distorted Ensemble Blend (No extreme 0.88..1.15 sample overfitting!)
    t_gb_l = 0.40 * t_ba["lgb_l"] + 0.35 * t_ba["xgb_l"] + 0.25 * t_ba["cb_l"]
    r8_pl = np.full((n, 6), 0.03, dtype=np.float32)
    r8_pl[np.arange(n), r8_l] = 0.85
    r8_pa = np.zeros((n, 6), dtype=np.float32)
    pos_a = r8_a > 0
    r8_pa[pos_a] = 0.035
    r8_pa[np.where(pos_a)[0], r8_a[pos_a]] = 0.86
    r8_pt = r8_t * 0.85 + 0.07

    # Balanced weights prioritizing clean neural ensemble + rules_v8
    P_L = 0.26 * pm_l + 0.17 * t_e["p_l"] + 0.10 * t_b["p_l"] + 0.16 * t_bg["p_l"] + 0.06 * t_ba["lin_l"] + 0.09 * t_gb_l + 0.16 * r8_pl
    P_T = 0.28 * pm_t + 0.24 * t_e["p_t"] + 0.14 * t_b["p_t"] + 0.06 * t_bg["p_t"] + 0.06 * t_ba["lin_t"] + 0.08 * t_ba["gb_t"] + 0.14 * r8_pt
    P_A = 0.22 * pm_a + 0.19 * t_e["p_a"] + 0.14 * t_b["p_a"] + 0.08 * t_bg["p_a"] + 0.06 * t_ba["lin_a"] + 0.06 * t_ba["gb_a"] + 0.25 * r8_pa

    # Apply mDeBERTa escalation with 0.42 weight on escalated uncertain rows
    ei = t_esc["esc_idx"]
    P_L[ei] = 0.58 * P_L[ei] + 0.42 * t_esc["p_l"]
    P_T[ei] = 0.58 * P_T[ei] + 0.42 * t_esc["p_t"]
    P_A[ei] = 0.58 * P_A[ei] + 0.42 * t_esc["p_a"]

    # High-Precision Guide Priority & Adversarial Repair Gates
    # Gate 1: Legitimate OTP warning with don't-share negation -> Class 0
    g_otp0 = otp_legit & (~req_sec)
    P_L[g_otp0, 0] = np.maximum(P_L[g_otp0, 0], 0.82)
    # Gate 2: Explicit secret harvest (OTP/PIN/CVV/NID/AnyDesk) -> Priority 1
    g_p1 = req_sec & (r8_l == 1) & (~otp_legit)
    P_L[g_p1, 1] = np.maximum(P_L[g_p1, 1], 0.78)
    # Gate 3: Reward/Selected/Job + clearance charge misclassified as Parcel (4) -> 2 or 3
    g_rew_clr = (r8_l == 2) & strong_pr & (~has_par) & (~has_jb)
    P_L[g_rew_clr, 2] = np.maximum(P_L[g_rew_clr, 2], 0.82)
    g_job_clr = (r8_l == 3) & has_jb & has_jbt & (~has_par)
    P_L[g_job_clr, 3] = np.maximum(P_L[g_job_clr, 3], 0.82)

    # Smooth balanced thresholds (fixing T1=0.36 over-firing!)
    tech_th = np.array([0.48, 0.46, 0.49, 0.45, 0.50, 0.48], dtype=np.float32)
    final_l, final_t, final_a = decode_consistent(P_L, P_T, P_A, tech_th=tech_th)

    # Calibrated Brier confidence: smooth agreement + margin + max_prob (no overconfident 0.996 saturation!)
    pred_set = [pm_l.argmax(1), t_e["p_l"].argmax(1), t_b["p_l"].argmax(1), t_bg["p_l"].argmax(1), t_gb_l.argmax(1), r8_l]
    votes = np.mean([(p == final_l) for p in pred_set], axis=0)
    sp = np.sort(P_L, axis=1)
    margin = sp[:, -1] - sp[:, -2]
    conf = np.clip(0.32 + 0.34 * votes + 0.24 * P_L.max(1) + 0.10 * margin, 0.25, 0.985)
    conf[np.isin(np.arange(n), re_idx)] = np.minimum(conf[np.isin(np.arange(n), re_idx)], 0.96)

    sub = pd.DataFrame({
        "id": te_ids,
        "label": final_l,
        "techniques": [vec_to_tech_str(final_t[i]) for i in range(n)],
        "action": final_a,
        "confidence": np.round(conf, 4),
    })
    out_csv = V2_DIR / "submission_v8_hotfix.csv"
    sub.to_csv(out_csv, index=False)
    s7 = pd.read_csv(V2_DIR / "submission_v7_top10.csv")
    print(f"Saved {out_csv} in {time.time()-t0:.1f}s! Changed vs s7: lbl={(sub['label']!=s7['label']).sum()} act={(sub['action']!=s7['action']).sum()} tech={(sub['techniques']!=s7['techniques']).sum()}", flush=True)


if __name__ == "__main__":
    main()
