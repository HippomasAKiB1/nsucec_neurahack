import re
import unicodedata
from typing import Dict, List, Tuple

ZW_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\ufe0f\u00ad]")
CONFUSABLES = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0456": "i",
    "\u0441": "c", "\u0440": "p", "\u0443": "y", "\u0445": "x",
    "\u0455": "s", "\u0501": "d", "\u04cf": "l",
    "\u0410": "A", "\u0415": "E", "\u041e": "O", "\u0406": "I",
    "\u0421": "C", "\u0420": "P", "\u0423": "Y", "\u0425": "X",
    "\u03bf": "o", "\u03b9": "i", "\u03b1": "a",
    "\u0964": ".",
})

EMOJI_MAP = {
    "\U0001f4b0": " <EMOJI_MONEY> ",
    "\U0001f381": " <EMOJI_GIFT> ",
    "\U0001f64f": " <EMOJI_PRAY> ",
    "\u26a0": " <EMOJI_WARN> ",
    "\U0001f4f2": " <EMOJI_MOBILE> ",
    "\U0001f525": " <EMOJI_FIRE> ",
    "\U0001f60a": " <EMOJI_SMILE> ",
}
EMOJI_RE = re.compile("|".join(re.escape(k) for k in EMOJI_MAP))
ANY_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]+")

NOISE_TOKENS_RE = re.compile(
    r"\b(?:zzqwpo|wkrjfejejrn|mnbvcx|qzxvbn|xkcdpl|asdkjh|lkjsdfoiu|xcvbnm|zzqpwoeiru|qundvpyfjqx|wbmtzmh|qwertyz|asdfghj)\b",
    flags=re.IGNORECASE,
)

KEEP_DOUBLE_WORDS = {
    "free", "fee", "offer", "book", "good", "soon", "need", "keep", "see", "call",
    "bill", "will", "all", "full", "sell", "tell", "well", "pass", "less", "boss",
    "hello", "address", "account", "apply", "app", "support", "access", "confirm",
    "commission", "winning", "winner", "innocent", "sorry", "shipping", "hurry",
    "arriving", "arrived", "received", "success", "successful", "install", "installment",
    "whatsapp", "telegram", "message", "press", "correct", "collect", "off", "staff",
    "poor", "room", "too", "look", "door", "cool", "took", "week", "weekly", "speed",
    "guaranteed", "deen", "allah", "assalamu", "walaikum", "korben", "taka", "bolen",
    "connect", "appreciate", "allow", "allowance", "arrears", "attach", "attached",
    "officer", "approved", "official", "borrow", "tomorrow", "messenger", "shopping",
    "annual", "add", "missed", "error", "reverse", "review", "ambassador", "committee",
}

LEET_TRANS = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})

CORE_LEXICON = {
    "bkash", "nagad", "rocket", "upay", "surecash", "mcash", "tap", "ok", "wallet",
    "pathao", "daraz", "sundarban", "steadfast", "redx", "paperfly", "ecourier",
    "fedex", "dhl", "paribahan", "karatoa", "sheba", "shwapno", "aarong", "robi",
    "grameenphone", "airtel", "banglalink", "teletalk", "skitto", "unilever", "walton",
    "beximco", "jamuna", "bank", "islami", "prime", "brac", "city", "dbbl", "ucb",
    "ebl", "pubali", "sonali", "desco", "nesco", "wasa", "titas", "btcl", "dpdc", "breb",
    "agora", "yellow", "foodpanda", "chaldal", "pickaboo", "rokomari", "ajkerdeal",
    "pin", "otp", "cvv", "nid", "kyc", "card", "password", "passcode", "account",
    "verify", "verification", "login", "blocked", "suspend", "suspended", "expired",
    "expire", "expires", "flagged", "limit", "upgrade", "helpline", "security",
    "support", "agent", "staff", "officer", "department", "fraud", "control",
    "refund", "mistake", "wrong", "reverse", "reimbursement", "vul", "bhul", "ferot",
    "return", "cancel", "cancellation", "bolen", "bolo", "bolun", "pathan", "pathaben",
    "din", "dao", "deben", "share", "send", "enter", "reply", "click", "call", "dial",
    "read", "out", "winner", "won", "win", "prize", "lottery", "lucky", "draw",
    "cashback", "bonus", "gift", "reward", "rewards", "smartphone", "iphone", "car",
    "tv", "claim", "congratulations", "congrats", "jitechen", "jiteche", "free",
    "spin", "airdrop", "usdt", "survey", "mystery", "box", "parcel", "package",
    "courier", "delivery", "deliver", "delivered", "customs", "duty", "fee", "charge",
    "clearance", "storage", "address", "incomplete", "failed", "atke", "ase", "held",
    "confirm", "update", "track", "tracking", "cod", "cash", "ready", "rakhun",
    "rider", "shipped", "hub", "collect", "job", "part", "time", "task", "tasks",
    "earn", "earning", "income", "salary", "beton", "deposit", "invest", "investment",
    "crypto", "profit", "double", "guaranteed", "whatsapp", "telegram", "group",
    "join", "apply", "register", "registration", "offer", "sale", "discount", "chhar",
    "code", "voucher", "eid", "winter", "flash", "recharge", "flexiload", "gb",
    "bill", "due", "paid", "successful", "balance", "trxid", "credited", "debited",
    "prepaid", "token", "meter", "out", "received", "add", "hoyeche", "brother",
    "sir", "madam", "apu", "mama", "bhai", "chacha", "khala", "friend", "hello",
    "good", "morning", "number", "from", "help", "lagbe", "kotha", "chilo", "busy",
    "ekhon", "amar", "apnar", "apni", "koren", "korun", "korben", "kauke", "na",
    "never", "anyone", "scam", "phishing", "alert", "warning", "sabdhan", "last",
    "hurry", "immediately", "taratari", "within", "hours", "minutes", "today",
    "only", "limited", "nearest", "outlet", "use", "via", "with", "your", "you",
    "new", "exciting", "features", "cooperation", "thank", "subject", "sender",
    "caller", "kal", "tay", "dekha", "hobe", "happy", "birthday", "dinner",
    "tonight", "treat", "ielts", "course", "admission", "cholche", "exam",
    "routine", "bazar", "school", "college", "class", "meeting", "interview",
    "library", "reserved", "available", "appointment", "dentist", "reschedule",
    "book", "flat", "apartment", "dream", "hair", "oil", "pizza", "combo",
    "drink", "days", "abroad", "malaysia", "visa", "position", "office", "copy",
    "bring", "welcome", "install", "order", "valid", "minute", "minutes", "invalid",
    "block", "report", "someone", "messaged", "scholarship", "allowance", "arrears",
    "citizen", "student", "approved", "uniform", "kit", "ambassador", "handler",
    "tuition", "portal", "spring", "bank", "booth", "letter", "attached", "hr",
    "sign", "joining", "date", "renew", "pack", "ota", "ashche", "ashe", "je",
    "seta", "theke", "detect", "borrow", "trouble", "deactivated", "sim", "poto",
    "photo", "both", "sides", "review", "proceed", "receive", "transfer", "kindly",
    "kamai", "unlock", "arbitrage", "bot", "app", "update", "exciting", "features",
    "weekend", "fiber", "router", "birthday", "tuition", "booth", "apartment",
    "installment", "delivered", "release", "helpline", "limited", "winners",
}

EXACT_TYPO_REPAIRS = {
    "bksah": "bkash", "bksh": "bkash", "bkah": "bkash", "baksh": "bkash",
    "bkashop": "bkash otp", "srecash": "surecash", "upyause": "upay use",
    "upayuse": "upay use", "chrtered": "chartered", "recahrge": "recharge",
    "pge": "page", "agnt": "agent", "dal": "dial", "dil": "dial",
    "jitechne": "jitechen", "daivd": "david", "firend": "friend",
    "fooooor": "for", "yu": "you", "yor": "your", "yoru": "your",
    "cad": "card", "adn": "and", "ad": "and", "olny": "only",
    "hsbeen": "has been", "reutrnit": "return it", "helplabe": "help lagbe",
    "hepl": "help", "rpely": "reply", "contiune": "continue", "hep": "help",
    "etka": "ekta", "ekon": "ekhon", "bsuy": "busy", "tyadeha": "tay dekha",
    "dekhahobe": "dekha hobe", "nearestoutlet": "nearest outlet",
    "withexciting": "with exciting", "yourcooperation": "your cooperation",
    "numberfrom": "number from", "whichfriend": "which friend",
    "kihelp": "ki help", "releasekorte": "release korte", "forour": "for our",
    "assecurity": "as security", "bhulnumber": "bhul number", "deha": "dekha",
    "ist": "is", "iste": "is the", "si,rapni": "sir apni",
    "brothe,etka": "brother ekta", "brothe,ekta": "brother ekta",
    "13l75": "ielts", "ieltsstuy": "ielts study",
    "brthday": "birthday", "dinnertonight": "dinner tonight",
    "currentpin": "current pin", "admissioncholche": "admission cholche",
    "coursee": "course", "shre": "share", "sharekroun": "share korun",
    "korbn": "korben", "dibenna": "diben na", "kuke": "kauke",
    "kuake": "kauke", "neevr": "never", "prcel": "parcel", "obth": "both",
    "addrss": "address", "poto": "photo", "mnths": "months", "thnks": "thanks",
    "jecoooode": "je code", "jecooode": "je code", "jecode": "je code",
    "ashcheota": "ashche ota", "thatis": "that is", "nevershare": "never share",
    "yourpin": "your pin", "pinta": "pin ta", "detecthoyeceh": "detect hoyeche",
    "applykorun": "apply korun", "andcvv": "and cvv", "passwordat": "password at",
    "airt3l": "airtel", "4ir73l": "airtel", "5k1tto": "skitto",
    "4dm15s10n": "admission", "c4shb4ck": "cashback",
    "securitydepartmenttheke": "security department theke",
    "accountwillbe": "account will be", "willbe": "will be",
    "sharekorben": "share korben", "bolbenna": "bolben na",
    "hobecod": "hobe cod", "apnaraccount": "apnar account",
    "takinle": "ta kinle", "tafree": "ta free", "shorttoprojojjo": "shortto projojjo",
    "releaseit": "release it", "unlockkorun": "unlock korun", "sendingnid": "sending nid",
    "potoand": "photo and", "couldnt": "could not", "deliverd": "delivered",
    "ktakotha": "ekta kotha", "kothachilo": "kotha chilo", "kiupay": "ki upay",
    "usekoren": "use koren", "seeender": "sender", "sennnder": "sender",
    "goodmoring": "good morning", "frmo": "from", "taks": "tasks",
    "newapp": "new app", "updte": "update", "bnak": "bank", "toapply": "to apply",
    "helpla": "help lagbe", "bhulnumber-e": "bhul number", "feedin": "fee din", "cashout": "cash out",
}

SUSPICIOUS_TLDS = {"top", "xyz", "site", "online", "vip", "link", "click", "info", "co", "cc", "icu", "live"}
SHORTENER_DOMAINS = {"bit.ly", "tinyurl.com", "t.co", "rb.gy", "cutt.ly", "shorturl.at"}
OFFICIAL_BRAND_DOMAINS = {
    "bkash.com", "bkash.com.bd", "bkash.org.bd", "bkash.net",
    "nagad.com", "nagad.com.bd", "nagad.net", "nagad.org.bd",
    "rocket.com.bd", "rocket.org.bd", "upay.com.bd", "surecash.com", "surecash.com.bd", "surecash.org.bd",
    "tap.com", "tap.org.bd", "okwallet.com.bd", "pathaopay.org.bd", "pathaopay.com.bd",
    "pathaofood.com.bd", "daraz.com.bd", "robi.com", "robi.com.bd", "robi.net", "robi.org.bd",
    "grameenphone.com", "grameenphone.com.bd", "grameenphone.net", "grameenphone.org.bd",
    "gp.com", "gp.com.bd", "gp.net", "gp.org.bd", "airtel.com", "airtel.com.bd", "airtel.net", "airtel.org.bd",
    "banglalink.com", "banglalink.com.bd", "banglalink.net", "banglalink.org.bd",
    "teletalk.com", "teletalk.com.bd", "teletalk.net", "teletalk.org.bd",
    "skitto.com", "skitto.com.bd", "skitto.net", "skitto.org.bd",
    "payment.breb.org.bd", "ecourier.com.bd", "citybank.org.bd", "ebl.net",
    "bracbank.net", "shebadelivery.net", "agora.net", "yellow.org.bd", "foodpanda.org.bd",
    "mcash.net",
}

BN_SPLIT_RE = re.compile(
    r"(ওটিপি|পিন|কোড|টাকা|লিংকে|পার্সেল|ক্লিক|যোগাযোগ)(দিন|করুন|নিন|পাঠান|হবে|আটকে|দিয়ে|করতে)"
)
PHONE_RE = re.compile(r"(?:\+?880[\s-]?)?01\d[\dxX]{6,9}\b")


def deobfuscate_basic(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = ZW_RE.sub("", t).translate(CONFUSABLES)
    t = EMOJI_RE.sub(lambda m: EMOJI_MAP[m.group(0)], t)
    t = ANY_EMOJI_RE.sub(" ", t)
    t = re.sub(r"\[\s*\.\s*\]|\[\s*\]", ".", t)
    t = re.sub(r"\bh[t7x]{2}p(s|5)?://", lambda m: "https://" if m.group(1) else "http://", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def _split_glued_ascii(tok: str) -> str:
    if len(tok) < 7 or tok in CORE_LEXICON or not tok.isalpha():
        return tok
    # Greedy 2- or 3-way split on CORE_LEXICON
    n = len(tok)
    dp = [None] * (n + 1)
    dp[0] = []
    for i in range(n):
        if dp[i] is None:
            continue
        for j in range(i + 2, n + 1):
            sub = tok[i:j]
            if sub in CORE_LEXICON:
                cand = dp[i] + [sub]
                if dp[j] is None or len(cand) < len(dp[j]):
                    dp[j] = cand
    if dp[n] is not None and 2 <= len(dp[n]) <= 4:
        return " ".join(dp[n])
    return tok


def _repair_token(tok: str) -> str:
    tl = tok.lower()
    if tl in EXACT_TYPO_REPAIRS:
        return EXACT_TYPO_REPAIRS[tl]
    if PHONE_RE.fullmatch(tok):
        return tok
    if re.search(r"([a-z])\1{2,}", tl):
        c2 = re.sub(r"([a-z])\1{2,}", r"\1\1", tl)
        c1 = re.sub(r"([a-z])\1{2,}", r"\1", tl)
        tl = c2 if c2 in KEEP_DOUBLE_WORDS else c1
        if tl in EXACT_TYPO_REPAIRS:
            return EXACT_TYPO_REPAIRS[tl]
    if re.search(r"[a-z]", tok) and re.search(r"[013457]", tok):
        # Preserve promo/trx codes like SAVE20, EID10, SP74153, FLASH1575
        if not re.match(r"^(?:save|deal|eid|winter|welcome|flash|sp|cn|pf|sc|ec|rx|sf)\d+$", tl) and not re.match(r"^[a-z]{1,3}\d{4,}$", tl):
            leet_cand = tl.translate(LEET_TRANS)
            if leet_cand in CORE_LEXICON or (re.fullmatch(r"[a-z]{3,}", leet_cand) and _split_glued_ascii(leet_cand) != leet_cand):
                tl = leet_cand
                if tl in EXACT_TYPO_REPAIRS:
                    return EXACT_TYPO_REPAIRS[tl]
    return _split_glued_ascii(tl)


def clean_for_model(text: str) -> str:
    t = deobfuscate_basic(text)
    t = NOISE_TOKENS_RE.sub(" ", t)
    # Fix Bengali glued post-positions/verbs like "ওটিপিদিন" -> "ওটিপি দিন"
    t = BN_SPLIT_RE.sub(r"\1 \2", t)
    # Split colon/comma/slash glued alphanumeric tokens outside URLs & |||
    parts = re.split(r"(\|\|\||https?://\S+|\s+)", t)
    out = []
    for p in parts:
        if not p or p.isspace():
            continue
        if p == "|||" or p.startswith("http"):
            # Repair domain-level leetspeak like 5k1tto.com.bd
            if p.startswith("http"):
                m = re.match(r"^(https?://)([^/\s]+)(.*)$", p, flags=re.I)
                if m:
                    dom = m.group(2).lower().replace("5k1tto", "skitto").replace("4ir73l", "airtel").replace("c0m", "com")
                    p = m.group(1) + dom + m.group(3)
            out.append(p)
        elif PHONE_RE.fullmatch(p):
            out.append(p)
        else:
            # Handle hyphen/colon-joined pieces gently unless USSD or short code
            if re.match(r"^\*\d+(?:\*\d+)?#$", p):
                out.append(p)
            else:
                # Split punctuation attached like Part-time, Robi:Special, bhulnumber-e
                subparts = re.split(r"([:,/()]+)", p)
                rebuilt = []
                for sp in subparts:
                    if not sp or re.fullmatch(r"[:,/()]+", sp):
                        if sp:
                            rebuilt.append(sp)
                    elif re.fullmatch(r"[A-Za-z0-9']+", sp):
                        rebuilt.append(_repair_token(sp))
                    else:
                        m = re.match(r"^([^A-Za-z0-9']*)([A-Za-z0-9']+)([^A-Za-z0-9']*)$", sp)
                        if m:
                            rebuilt.append(m.group(1) + _repair_token(m.group(2)) + m.group(3))
                        else:
                            rebuilt.append(sp)
                joined = "".join(rebuilt)
                # Drop unknown random 7+ consonant-heavy gibberish tokens
                if re.fullmatch(r"[bcdfghjklmnpqrstvwxz]{6,}", joined):
                    continue
                out.append(joined)
    res = re.sub(r"\s+", " ", " ".join(out)).strip()
    return res if res else "<NOISE>"


def mask_urls_for_text_rules(clean_text: str) -> Tuple[str, Dict[str, int]]:
    urls = re.findall(r"https?://([^\s/]+)(/[^\s\"']*)?", clean_text.lower())
    flags = {"has_url": int(len(urls) > 0), "has_susp_tld": 0, "has_official_dom": 0, "has_shortener": 0, "has_apk": int(".apk" in clean_text.lower()), "has_tg_wa": int(bool(re.search(r"t\.me/|wa\.me/", clean_text.lower())))}
    def _repl(m):
        dom = m.group(1).lower()
        path = (m.group(2) or "").lower()
        tld = dom.split(".")[-1] if "." in dom else ""
        if ".apk" in path:
            flags["has_apk"] = 1
            return " <URL_APK> "
        if "t.me" in dom or "wa.me" in dom:
            flags["has_tg_wa"] = 1
            return " <URL_TGWA> "
        if dom in SHORTENER_DOMAINS:
            flags["has_shortener"] = 1
            return " <URL_SHORT> "
        if dom in OFFICIAL_BRAND_DOMAINS or dom.endswith((".edu.bd", ".ac.bd", ".gov.bd", ".edu", ".gov")):
            flags["has_official_dom"] = 1
            return " <URL_OFFICIAL> "
        if tld in SUSPICIOUS_TLDS:
            flags["has_susp_tld"] = 1
            return " <URL_SUSP> "
        return " <URL> "
    masked = re.sub(r"https?://([^\s/]+)(/[^\s\"']*)?", _repl, clean_text)
    # Also mask scheme-less shorteners/tg/wa links
    masked = re.sub(r"\b(?:bit\.ly|tinyurl\.com|t\.co|rb\.gy|cutt\.ly|shorturl\.at)/\S+", lambda m: _set_flag_short(flags), masked)
    masked = re.sub(r"\b(?:t\.me|wa\.me)/\S+", lambda m: _set_flag_tgwa(flags), masked)
    return masked, flags


def _set_flag_short(flags):
    flags["has_url"] = 1
    flags["has_shortener"] = 1
    return " <URL_SHORT> "


def _set_flag_tgwa(flags):
    flags["has_url"] = 1
    flags["has_tg_wa"] = 1
    return " <URL_TGWA> "


def extract_base_struct_v4(raw_text: str, clean_text: str) -> list:
    """24-dim structural, obfuscation, OOV-proxy, and turn-structure features."""
    raw_s = str(raw_text)
    cl_low = clean_text.lower()
    masked, uflags = mask_urls_for_text_rules(clean_text)
    turns = clean_text.split("|||")
    zw_count = len(ZW_RE.findall(raw_s))
    cyr_count = sum(1 for c in raw_s if "\u0400" <= c <= "\u04ff")
    alpha_tokens = re.findall(r"[a-z]{3,}", cl_low)
    has_bn = int(bool(re.search(r"[\u0980-\u09FF]", clean_text)))
    known_hits = sum(1 for w in alpha_tokens if w in CORE_LEXICON)
    known_ratio = float(known_hits / max(1, len(alpha_tokens)))
    has_phone = int(bool(re.search(r"(?:\+?880\s*)?01\d[\dxX]{5,9}|\b\d{5}-[\dxX]+|\b\*\d{3}(?:\*\d+)?#", cl_low)))
    has_short_code = int(bool(re.search(r"\b(?:16247|16216|16263|16167)\b|\*\d{3}#", cl_low)))
    has_money = int(bool(re.search(r"\b(?:tk|bdt|taka)\b|টাকা|\$", cl_low)))
    has_hxxp = int(bool(re.search(r"h[t7x]{2}p", raw_s, flags=re.I)))
    has_noise_tok = int(bool(NOISE_TOKENS_RE.search(raw_s)))
    is_noise_only = int(clean_text == "<NOISE>" or ((not has_bn) and known_hits == 0 and uflags["has_url"] == 0 and not re.search(r"\d{3,}", cl_low) and len(alpha_tokens) <= 4))
    has_innocent_wrap = int(bool(re.search(r"hope this message finds you well|dear valued customer|regular communication|thank you for your cooperation", cl_low)))
    return [
        float(len(raw_s)), float(len(clean_text)), float(len(turns)), float(has_bn),
        float(uflags["has_url"]), float(has_hxxp), float(uflags["has_susp_tld"]),
        float(uflags["has_official_dom"]), float(uflags["has_shortener"]), float(uflags["has_apk"]),
        float(uflags["has_tg_wa"]), float(has_phone), float(has_short_code), float(has_money),
        float(has_noise_tok), float(is_noise_only), float(zw_count), float(cyr_count),
        float(known_hits), float(known_ratio), float(len(alpha_tokens)), float(has_innocent_wrap),
        float("|CTX|" in format_turn_aware(clean_text)), float("[" in raw_s and "]" in raw_s),
    ]


def format_turn_aware(clean_text: str, mask_urls: bool = True) -> str:
    """Mask untrusted URLs (killing spoofed brand/gift domains) & put key/late turns upfront."""
    t_in = mask_urls_for_text_rules(clean_text)[0] if mask_urls else clean_text
    t_in = re.sub(r"\be(\d+%)", r"e \1", t_in)
    if "|||" not in t_in or len(t_in) <= 180:
        return t_in
    turns = [x.strip() for x in t_in.split("|||") if x.strip()]
    if len(turns) <= 2:
        return t_in
    sig_re = re.compile(r"pin|otp|cvv|nid|password|code|fee|pay|deposit|won|prize|gift|parcel|customs|link|<URL|job|income|পিন|ওটিপি|টাকা|ফি|জিতে|পার্সেল|চাকরি|যোগ|scam|phishing|never share|card number|ludo", re.I)
    key_turns = [t for t in turns if sig_re.search(t)]
    tail_turns = turns[-2:]
    picked = []
    for t in key_turns + tail_turns:
        if t not in picked:
            picked.append(t)
    head = " ||| ".join(picked[-3:])
    return f"{head} |CTX| {t_in}"


BN_GLOSS = {
    "বিকাশ": "bkash", "নগদ": "nagad", "রকেট": "rocket", "উপায়": "upay",
    "পিন": "pin", "ওটিপি": "otp", "কোড": "code", "এনআইডি": "nid",
    "কার্ড": "card", "পাসওয়ার্ড": "password", "টাকা": "taka money",
    "হাজার": "thousand", "লাখ": "lakh", "জিতেছেন": "won prize",
    "জিতেছে": "won prize", "জিতার": "win prize", "লটারি": "lottery",
    "লাকি": "lucky", "ড্র": "draw", "ক্যাশব্যাক": "cashback",
    "বোনাস": "bonus", "পুরস্কার": "prize reward", "উপহার": "gift",
    "অভিনন্দন": "congratulations", "পার্সেল": "parcel", "ডেলিভারি": "delivery",
    "কাস্টমস": "customs", "শুল্ক": "duty fee", "আটকে": "held stuck",
    "ঠিকানা": "address", "ব্যর্থ": "failed", "আপডেট": "update",
    "লিংকে": "link", "লিংক": "link", "ক্লিক": "click", "যান": "visit",
    "পাঠান": "send", "বলুন": "tell", "দিন": "give", "করুন": "do",
    "নিন": "take", "দাও": "give", "দেবেন": "give", "জানান": "inform",
    "অ্যাকাউন্ট": "account", "একাউন্ট": "account", "বন্ধ": "closed blocked",
    "ব্লক": "block", "সতর্ক": "warning alert", "ভুয়া": "fake",
    "প্রতারণা": "scam fraud", "স্ক্যাম": "scam", "সাবধান": "beware",
    "আয়": "earn income", "বিনিয়োগ": "invest", "দ্বিগুণ": "double",
    "নিশ্চিত": "guaranteed", "লাভ": "profit", "চাকরি": "job",
    "কাজ": "work task", "বেতন": "salary", "লোক": "staff hiring",
    "নিচ্ছে": "hiring", "রেজিস্ট্রেশন": "registration", "ঘরে": "home",
    "বসে": "sitting", "হোয়াটসঅ্যাপে": "whatsapp", "টেলিগ্রামে": "telegram",
    "যোগ": "join", "যোগাযোগ": "contact call", "কল": "call",
    "ছাড়": "discount off", "অফার": "offer", "বিল": "bill",
    "পেমেন্টে": "payment", "রিচার্জে": "recharge", "পান": "get",
    "ডায়াল": "dial", "ক্যাশ": "cash", "আউট": "out", "সফল": "successful",
    "ব্যালেন্স": "balance", "জমা": "credited deposit", "পেয়েছেন": "received",
    "ট্রানজেকশন": "transaction", "গ্রাহক": "customer", "প্রিয়": "dear",
    "এখনই": "immediately now", "আজই": "today", "আজ": "today",
    "নইলে": "otherwise", "ফি": "fee", "চার্জ": "charge", "প্রসেসিং": "processing",
    "পরিশোধ": "pay", "ভুল": "mistake wrong", "ভুলে": "by mistake",
    "ফেরত": "return", "দয়া": "please", "করে": "by", "কাউকে": "anyone",
    "শেয়ার": "share", "কখনো": "never", "বৈধ": "valid", "মিনিট": "minutes",
    "ডিসেম্বর": "december", "নভেম্বর": "november", "ভাতা": "allowance",
    "বয়স্ক": "old age", "নম্বরটি": "number", "নম্বরে": "number",
}
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BN_CHAR = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t", "ং": "ng", "ঃ": "h", "ঁ": "n", "্": "",
    "৳": "Tk ",
}
BN_CONSONANTS = set("কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়")


def bn_to_en_roman_v4(text: str) -> str:
    s = str(text).translate(BN_DIGITS)
    if not re.search(r"[\u0980-\u09FF৳]", s):
        return s

    def repl_word(m):
        w = m.group(0)
        gloss = BN_GLOSS.get(w)
        out = []
        n = len(w)
        for i, ch in enumerate(w):
            out.append(BN_CHAR.get(ch, ch))
            if ch in BN_CONSONANTS and i + 1 < n and w[i + 1] in BN_CONSONANTS:
                out.append("a")
        rw = "".join(out)
        if gloss and gloss != rw:
            return f"{rw} ({gloss})"
        return rw

    return re.sub(r"[\u0980-\u09FF]+|৳", lambda m: "Tk " if m.group(0) == "৳" else repl_word(m), s)
