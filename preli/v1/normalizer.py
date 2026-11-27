import re
from typing import Dict, List, Tuple

ZW_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\ufe0f]")
CONFUSABLES = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0456": "i",
    "\u0441": "c", "\u0440": "p", "\u0443": "y", "\u0445": "x",
    "\u0410": "A", "\u0415": "E", "\u041e": "O", "\u0406": "I",
    "\u0421": "C", "\u0420": "P", "\u0423": "Y", "\u0425": "X",
    "\u0964": ".",
})

EMOJI_MAP = {
    "\U0001f4b0": " <money> ",
    "\U0001f381": " <gift> ",
    "\U0001f64f": " <pray> ",
    "\u26a0": " <warn> ",
    "\U0001f4f2": " <mobile> ",
    "\U0001f525": " <fire> ",
    "\U0001f60a": " <smile> ",
}
EMOJI_RE = re.compile("|".join(re.escape(k) for k in EMOJI_MAP))

NOISE_TOKENS_RE = re.compile(
    r"\b(?:zzqwpo|wkrjfejejrn|mnbvcx|qzxvbn|xkcdpl|asdkjh|lkjsdfoiu|xcvbnm|zzqpwoeiru)\b",
    flags=re.IGNORECASE,
)

# Common words where double-letter collapse of 3+ repetitions should keep 2 letters
KEEP_DOUBLE_WORDS = {
    "free", "fee", "offer", "book", "good", "soon", "need", "keep", "see", "call",
    "bill", "will", "all", "full", "sell", "tell", "well", "pass", "less", "boss",
    "hello", "address", "account", "apply", "app", "support", "access", "confirm",
    "commission", "winning", "winner", "innocent", "sorry", "shipping", "hurry",
    "arriving", "arrived", "received", "success", "successful", "install", "installment",
    "whatsapp", "telegram", "message", "press", "correct", "collect", "off", "staff",
    "poor", "room", "too", "look", "door", "cool", "took", "week", "weekly", "speed",
    "guaranteed", "deen", "allah", "assalamu", "walaikum", "korben", "taka", "bolen",
}

LEET_TRANS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})

# Known clean core words to repair leetspeak / minor 1-char swaps / collapsed spaces
CORE_LEXICON = {
    "bkash", "nagad", "rocket", "upay", "surecash", "mcash", "tap", "ok", "wallet",
    "pathao", "daraz", "sundarban", "steadfast", "redx", "paperfly", "ecourier",
    "fedex", "dhl", "paribahan", "karatoa", "sheba", "shwapno", "aarong", "robi",
    "grameenphone", "airtel", "banglalink", "teletalk", "skitto", "unilever", "walton",
    "beximco", "jamuna", "bank", "islami", "prime", "brac", "city", "dbbl", "ucb",
    "pin", "otp", "cvv", "nid", "kyc", "card", "password", "account", "verify",
    "verification", "login", "blocked", "suspend", "suspended", "expired", "expire",
    "flagged", "limit", "upgrade", "helpline", "security", "support", "agent", "staff",
    "refund", "mistake", "wrong", "vul", "bhul", "ferot", "return", "cancel", "bolen",
    "pathan", "din", "share", "send", "enter", "reply", "click", "call", "dial",
    "winner", "won", "win", "prize", "lottery", "lucky", "draw", "cashback", "bonus",
    "gift", "reward", "smartphone", "iphone", "car", "tv", "claim", "congratulations",
    "jitechen", "jiteche", "free", "spin", "parcel", "package", "courier", "delivery",
    "customs", "duty", "fee", "charge", "storage", "address", "incomplete", "failed",
    "atke", "ase", "held", "confirm", "update", "track", "tracking", "job", "part",
    "time", "task", "tasks", "earn", "income", "salary", "beton", "deposit", "invest",
    "investment", "crypto", "profit", "double", "guaranteed", "whatsapp", "telegram",
    "group", "join", "apply", "register", "registration", "offer", "sale", "discount",
    "chhar", "code", "voucher", "eid", "winter", "flash", "recharge", "gb", "bill",
    "due", "paid", "successful", "balance", "trxid", "cash", "out", "received",
    "brother", "sir", "madam", "apu", "mama", "bhai", "friend", "hello", "good",
    "morning", "number", "from", "help", "lagbe", "kotha", "chilo", "busy", "ekhon",
    "amar", "apnar", "apni", "koren", "korun", "korben", "kauke", "na", "never",
    "anyone", "scam", "phishing", "alert", "warning", "last", "hurry", "immediately",
    "taratari", "within", "hours", "minutes", "today", "only", "limited", "nearest",
    "outlet", "use", "via", "with", "your", "you", "new", "exciting", "features",
    "cooperation", "thank", "subject", "sender", "caller", "kal", "tay", "dekha", "hobe",
    "happy", "birthday", "dinner", "tonight", "treat", "ielts", "course", "admission",
    "cholche", "exam", "routine", "bazar", "school", "college", "class", "meeting",
    "interview", "library", "reserved", "available", "collect", "appointment", "dentist",
    "reschedule", "book", "flat", "apartment", "dream", "winter", "hair", "oil", "pizza",
    "combo", "drink", "days", "abroad", "malaysia", "visa", "salary", "position", "office",
    "nid", "copy", "bring", "confirm", "welcome", "install", "order", "valid", "minute",
    "minutes", "invalid", "block", "report", "someone", "messaged", "agent", "cash",
}

EXACT_TYPO_REPAIRS = {
    "bksah": "bkash", "bksh": "bkash", "bkah": "bkash", "baksh": "bkash", "bkashop": "bkash otp",
    "srecash": "surecash", "upyause": "upay use", "upayuse": "upay use",
    "chrtered": "chartered", "recahrge": "recharge", "pge": "page",
    "agnt": "agent", "dal": "dial", "dil": "dial", "jitechne": "jitechen",
    "daivd": "david", "firend": "friend", "fooooor": "for", "yu": "you",
    "yor": "your", "yoru": "your", "cad": "card", "adn": "and", "ad": "and",
    "olny": "only", "hsbeen": "has been", "reutrnit": "return it",
    "helplabe": "help lagbe", "hepl": "help", "rpely": "reply", "contiune": "continue",
    "hep": "help", "etka": "ekta", "ekon": "ekhon", "bsuy": "busy",
    "tyadeha": "tay dekha", "dekhahobe": "dekha hobe", "nearestoutlet": "nearest outlet",
    "withexciting": "with exciting", "yourcooperation": "your cooperation",
    "numberfrom": "number from", "whichfriend": "which friend", "kihelp": "ki help",
    "releasekorte": "release korte", "forour": "for our", "assecurity": "as security",
    "bhulnumber": "bhul number", "deha": "dekha", "ist": "is", "iste": "is the",
    "si,rapni": "sir apni", "brothe,etka": "brother ekta",
    "13l75": "ielts", "ieltsstuy": "ielts study", "brthday": "birthday",
    "dinnertonight": "dinner tonight", "currentpin": "current pin",
    "admissioncholche": "admission cholche", "coursee": "course",
}

SUSPICIOUS_TLDS = {
    "top", "xyz", "site", "online", "vip", "link", "click", "info", "co", "cc", "icu", "live"
}
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "rb.gy", "cutt.ly", "shorturl.at"}
OFFICIAL_BRAND_DOMAINS = {
    "bkash.com", "bkash.com.bd", "bkash.org.bd", "bkash.net",
    "nagad.com", "nagad.com.bd", "nagad.net", "nagad.org.bd",
    "rocket.com.bd", "rocket.org.bd", "upay.com.bd", "surecash.com", "surecash.com.bd",
    "tap.org.bd", "okwallet.com.bd", "pathaopay.org.bd",
    "daraz.com.bd", "robi.com", "robi.com.bd", "robi.net", "robi.org.bd",
    "grameenphone.com", "grameenphone.com.bd", "grameenphone.net", "grameenphone.org.bd",
    "gp.com", "gp.com.bd", "gp.net", "gp.org.bd",
    "airtel.com", "airtel.com.bd", "airtel.net", "airtel.org.bd",
    "banglalink.com", "banglalink.com.bd", "banglalink.net", "banglalink.org.bd",
    "teletalk.com", "teletalk.com.bd", "teletalk.net", "teletalk.org.bd",
    "skitto.com", "skitto.com.bd", "skitto.net", "skitto.org.bd",
    "payment.breb.org.bd",
}

def deobfuscate_basic(text: str) -> str:
    t = ZW_RE.sub("", str(text)).translate(CONFUSABLES)
    t = EMOJI_RE.sub(lambda m: EMOJI_MAP[m.group(0)], t)
    # Fix defanged URLs ([.] or [])
    t = re.sub(r"\[\s*\.\s*\]|\[\s*\]", ".", t)
    t = re.sub(r"\bhxxp(s?)://", r"http\1://", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()

def _repair_token(tok: str) -> str:
    tl = tok.lower()
    if tl in EXACT_TYPO_REPAIRS:
        return EXACT_TYPO_REPAIRS[tl]
    # Collapse 3+ repeated characters
    if re.search(r"([a-zA-Z])\1{2,}", tl):
        c2 = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", tl)
        c1 = re.sub(r"([a-zA-Z])\1{2,}", r"\1", tl)
        tl = c2 if c2 in KEEP_DOUBLE_WORDS else c1
        if tl in EXACT_TYPO_REPAIRS:
            return EXACT_TYPO_REPAIRS[tl]
    # Repair leetspeak if mixed with lowercase/known pattern (avoid TrxIDs like X5DRERAFMR or D6FMURDEP)
    if re.search(r"[a-z]", tok) and re.search(r"[013457]", tok):
        leet_cand = tl.translate(LEET_TRANS)
        if leet_cand in CORE_LEXICON or re.search(r"^[a-z]+$", leet_cand):
            # Only apply if it looks like a real word, not code like save208 / deal50
            if not re.match(r"^(?:save|deal|eid|winter|welcome)\d+$", tl):
                tl = leet_cand
                if tl in EXACT_TYPO_REPAIRS:
                    return EXACT_TYPO_REPAIRS[tl]
    return tl

def clean_for_model(text: str) -> str:
    t = deobfuscate_basic(text)
    t = NOISE_TOKENS_RE.sub(" ", t)
    # Split glued known tokens like "currentpin" -> "current pin"
    t = re.sub(r"\b(current)(pin|otp|password)\b", r"\1 \2", t, flags=re.IGNORECASE)
    # Split accidentally glued speaker/word comma patterns like "Si,rapni" or "Bak:Book"
    t = re.sub(r"([A-Za-z]),([A-Za-z])", r"\1 \2", t)
    # Repair individual ASCII tokens while preserving Bengali script, URLs, and ||| delimiters
    parts = re.split(r"(\|\|\||https?://\S+|\s+)", t)
    out = []
    for p in parts:
        if not p or p.isspace():
            continue
        if p == "|||" or p.startswith("http"):
            out.append(p)
        elif re.fullmatch(r"[A-Za-z0-9']+", p):
            out.append(_repair_token(p))
        else:
            # Token with trailing/leading punctuation
            m = re.match(r"^([^A-Za-z0-9']*)([A-Za-z0-9']+)([^A-Za-z0-9']*)$", p)
            if m:
                out.append(m.group(1) + _repair_token(m.group(2)) + m.group(3))
            else:
                out.append(p)
    return re.sub(r"\s+", " ", " ".join(out)).strip()

def extract_structural_features(raw_text: str, clean_text: str) -> Dict[str, float]:
    raw_s = str(raw_text)
    cl_low = clean_text.lower()
    urls = re.findall(r"https?://([^\s/]+)(/[^\s\"']*)?", cl_low)
    has_url = int(len(urls) > 0 or bool(re.search(r"\b(?:bit\.ly|tinyurl\.com|t\.me)/\S+", cl_low)))
    has_hxxp = int(bool(re.search(r"hxxps?://", raw_s, flags=re.I)))
    has_susp_tld = 0
    has_official_dom = 0
    has_apk = int(".apk" in cl_low)
    has_tg_wa_link = int(bool(re.search(r"t\.me/|wa\.me/", cl_low)))
    has_url_phish_path = 0
    for dom, path in urls:
        tld = dom.split(".")[-1] if "." in dom else ""
        if tld in SUSPICIOUS_TLDS:
            has_susp_tld = 1
        if dom in OFFICIAL_BRAND_DOMAINS:
            has_official_dom = 1
        if path and any(k in path for k in ("/pay", "/verify", "/claim", "/login", "/kyc")):
            has_url_phish_path = 1

    # Detect gibberish / noise-only text (requires zero known words AND zero BN AND gibberish consonant runs or all-random)
    alpha_tokens = re.findall(r"[a-z]{3,}", cl_low)
    has_bn = int(bool(re.search(r"[\u0980-\u09FF]", clean_text)))
    known_hits = sum(1 for w in alpha_tokens if w in CORE_LEXICON)
    has_gibberish_run = bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", cl_low))
    is_noise_only = int(
        len(cl_low.strip()) == 0
        or (
            (not has_bn) and (known_hits == 0) and (not has_url)
            and not re.search(r"\d{3,}", cl_low)
            and (len(alpha_tokens) <= 4 or has_gibberish_run)
        )
    )

    # Scam-warning conversation or legitimate OTP warning (must NOT actively ask to send/bolen/pathan/reply PIN/OTP/CVV!)
    active_harvest = bool(re.search(
        r"(?:card number.*cvv|reply with.*(?:pin|otp)|enter your (?:pin|otp)|"
        r"share (?:the )?(?:\d[- ]digit )?(?:pin|otp)|(?:pin|otp|code|nid).{0,25}(?:bolen|pathan|din|send|share)|"
        r"(?:seta|ta) bolen)",
        cl_low,
    ))
    quoted_scam_warn = bool(re.search(
        r"(?:someone|ekta sms|ekjon).*?(?:messaged|ashche|pathaise).*?\|\|\|.*?(?:scam|phishing|block|never share|kauke.*(?:dibi|bolben))",
        cl_low,
    ))
    plain_otp_warn = bool(re.search(
        r"(?:never share|do not share|don'?t share|kauke (?:bolben|share korben|dibi) na|eta scam)",
        cl_low,
    ))
    has_scam_warning = int(quoted_scam_warn or (plain_otp_warn and not active_harvest))
    # Multi-turn info
    turns = clean_text.split("|||")
    n_turns = len(turns)
    zw_count = len(ZW_RE.findall(raw_s))
    cyr_count = sum(1 for c in raw_s if "\u0400" <= c <= "\u04ff")

    return {
        "n_chars": float(len(raw_s)),
        "n_turns": float(n_turns),
        "has_bn": float(has_bn),
        "has_url": float(has_url),
        "has_hxxp": float(has_hxxp),
        "has_susp_tld": float(has_susp_tld),
        "has_official_dom": float(has_official_dom),
        "has_apk": float(has_apk),
        "has_tg_wa_link": float(has_tg_wa_link),
        "has_url_phish_path": float(has_url_phish_path),
        "has_phone": float(bool(re.search(r"(?:\+?880\s*)?01\d[\dxX]{5,9}|\b\d{5}-\w+|\b\*\d{3}(?:\*\d+)?#", cl_low))),
        "has_short_code": float(bool(re.search(r"\b(?:16247|16216|16263|16167)\b|\*\d{3}#", cl_low))),
        "has_money": float(bool(re.search(r"\b(?:tk|bdt|taka)\b|টাকা|\$", cl_low))),
        "has_cred_kw": float(bool(re.search(r"\b(?:pin|otp|cvv|nid|password|card number|expir)\b|পিন|ওটিপি|এনআইডি", cl_low))),
        "has_prize_kw": float(bool(re.search(r"\b(?:won|win|winner|prize|lottery|lucky draw|cashback|reward|gift|jitechen|jiteche|spin)\b|জিতেছেন|জিতেছে|লটারি|লাকি|ক্যাশব্যাক|অভিনন্দন", cl_low))),
        "has_job_kw": float(bool(re.search(r"\b(?:part[- ]?time|task|tasks|earn|income|salary|beton|deposit|invest|crypto|arbitrage|freelanc|data entry|ghor bose|kaj)\b|আয়|বিনিয়োগ|দ্বিগুণ|ঘরে বসে|হোয়াটসঅ্যাপে", cl_low))),
        "has_parcel_kw": float(bool(re.search(r"\b(?:parcel|package|courier|customs|duty|redelivery|storage fee|delivery|steadfast|sundarban|paperfly|paribahan|redx|karatoa|ecourier|dhl|fedex)\b|পার্সেল|ডেলিভারি|কাস্টমস|শুল্ক", cl_low))),
        "has_promo_kw": float(bool(re.search(r"\b(?:discount|sale|eid offer|winter sale|voucher|off|recharge|1 ta kinle|shortto projojjo|limited time|buy \d+|get \d+)\b|ছাড়|রিচার্জে|অফার", cl_low))),
        "has_urgency_kw": float(bool(re.search(r"\b(?:urgent|last warning|within \d+|blocked|suspend|expired|immediately|taratari|hurry|bipode|amar job jabe|ekhoni|today only)\b|এখনই|নইলে|বন্ধ|সতর্ক", cl_low))),
        "has_link_click_kw": float(bool(re.search(r"\b(?:click|tap|visit|open)\b|ক্লিক|যান", cl_low))),
        "has_join_install_kw": float(bool(re.search(r"\b(?:join|install|download|apk|whatsapp|telegram)\b|যোগ দিন|হোয়াটসঅ্যাপে", cl_low))),
        "has_call_reply_kw": float(bool(re.search(r"\b(?:call|whatsapp korun|reply|contact)\b|কল করুন|যোগাযোগ", cl_low))),
        "has_noise_tok": float(bool(NOISE_TOKENS_RE.search(raw_s))),
        "is_noise_only": float(is_noise_only),
        "has_scam_warning": float(has_scam_warning),
        "zw_count": float(zw_count),
        "cyr_count": float(cyr_count),
    }
