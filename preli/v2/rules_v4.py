import os
import re
import sys
from pathlib import Path
import numpy as np

try:
    from v2.normalizer_v4 import (
        clean_for_model,
        mask_urls_for_text_rules,
        extract_base_struct_v4,
        NOISE_TOKENS_RE,
    )
except ImportError:
    try:
        from normalizer_v4 import (
            clean_for_model,
            mask_urls_for_text_rules,
            extract_base_struct_v4,
            NOISE_TOKENS_RE,
        )
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from normalizer_v4 import (
            clean_for_model,
            mask_urls_for_text_rules,
            extract_base_struct_v4,
            NOISE_TOKENS_RE,
        )

# Strictly word-boundary-safe general linguistic & policy patterns (Problem Statement Sections 2-4)
SENS_RE = re.compile(
    r"\b(?:pin|otp|password|passcode|cvv|card number|card details|nid|verification code|login code|security code|"
    r"6[- ]?digit(?: code)?|\d[- ]digit code|date of birth|dob|internet banking password|wallet pin)\b"
    r"|ওটিপি|পিন|পাসওয়ার্ড|এনআইডি|কার্ড নম্বর",
    flags=re.IGNORECASE,
)
CODE_PLAIN_RE = re.compile(r"\bcode\b|কোড", flags=re.IGNORECASE)

ASK_SECRET_RE = re.compile(
    r"\b(?:send|sending|share|sharing|give|giving|tell|reply|reply with|provide|enter|entering|submit|forward|"
    r"confirm|verify|verify by|read out|bolen|bolo|bolun|pathan|pathaben|din|den|dao|deben|dibi|ta bolen|ta din|dite hobe|bolte hobe)\b"
    r"|জানান|পাঠান|দিন|বলুন|বলো|দেবেন|দাও|পাঠাও",
    flags=re.IGNORECASE,
)
SPEAKER_PREFIX_RE = re.compile(
    r"\[(?:sender|you|caller|agent)\]\s*:|(?:^|\|\|\|\s*)(?:sender|caller|you|s|r|a|b)\s*:",
    flags=re.IGNORECASE,
)

NEG_WARN_RE = re.compile(
    r"\b(?:only with (?:the )?rider|trip pin|do not share|don'?t share|never share|not share|never ask|never call|will never|"
    r"do not reply|don'?t reply|please do not reply|do not pay anyone|reply stop to opt out|"
    r"kauke\b.{0,18}\b(?:share|bolben|korben|diben|dien|bolo)\b.{0,6}\bna|"
    r"(?:bolben|dibi|dien|diben|korben)\s+na)\b"
    r"|কাউকে.{0,22}(?:দেবেন|করবেন|বলবেন|শেয়ার).{0,6}না|কখনো.{0,12}না",
    flags=re.IGNORECASE,
)

SCAM_WARN_THREAD_RE = re.compile(
    r"(?:someone|ekta sms|ekjon).{0,60}(?:messaged|ashche|pathaise).*?\|\|\|"
    r"|\b(?:eta scam|this is a scam|it'?s a scam|that is phishing|phishing|scam bhai|block and report|ignore korsi|scam hote pare|keu pin chaile)\b"
    r"|স্ক্যাম|প্রতারণা",
    flags=re.IGNORECASE,
)

MISTAKE_RE = re.compile(
    r"\b(?:accident(?:ally)?|by mistake|wrong transfer|wrong transaction|transfer.{0,20}wrong|sent.{0,20}wrong|"
    r"vul kore|bhul kore|vul e|bhul e|bhul hoye|vul hoye|wrong number|bhul number|vul number|mistaken|bhul transaction|vul transaction)\b"
    r"|ভুল করে|ভুলে|ভুল নম্বর",
    flags=re.IGNORECASE,
)
RETURN_RE = re.compile(
    r"\b(?:send (?:it )?back|return|ferot|refund|reverse it|pathiye din|flexiload|kindly return)\b"
    r"|ফেরত|পাঠিয়ে দিন",
    flags=re.IGNORECASE,
)

STRONG_PRIZE_RE = re.compile(
    r"\b(?:you won|you'?ve won|won|winner|winners|win a (?:tk|voucher|prize|gift|iphone|smartphone|car)|jitechen|jitechhen|jiteche|jitar|lucky draw|lottery|giveaway|airdrop|usdt|"
    r"mystery box|spin and win|free iphone|free smartphone|free router|free wi[- ]?fi router|10gb data gift|"
    r"pm education fund|stipend|old[- ]?age allowance|allowance arrears|bumper|"
    r"(?:number\s*ti|toyota\s*car|iphone\s*\d*|smart\s*tv|prize|bonus|gift|cashback|stipend|allowance).{0,35}select(?:ed)?\s*(?:hoyechen|hoyeche|for)|"
    r"loyalty reward|prize|cashback.{0,15}(?:jitechen|won|claim))\b"
    r"|জিতেছেন|জিতেছে|জিতার|লাকি ড্র|লটারি|পুরস্কার",
    flags=re.IGNORECASE,
)
SOFT_PRIZE_RE = re.compile(
    r"\b(?:cashback|gift|reward|rewards|bonus|eid bonus|congratulations|congrats)\b"
    r"|ক্যাশব্যাক|উপহার|অভিনন্দন|ঈদ বোনাস|গিফট",
    flags=re.IGNORECASE,
)
CLAIM_RE = re.compile(
    r"\b(?:claim|click|tap|link|fee|charge|(?<!pathao\s)pay|processing|survey|connect (?:your )?wallet|whatsapp|telegram|call|dial|install|download|reply|address|group|<URL_?\w*>|answer \d+ questions|submit your bkash)\b"
    r"|ক্লেইম|ক্লিক|লিংকে|লিংক|ফি|চার্জ|যোগাযোগ|যান|ইনস্টল|পাঠান|নিচের|দিন",
    flags=re.IGNORECASE,
)

PARCEL_RE = re.compile(
    r"\b(?:parcel|package|courier|customs|shipment|consignment|tracking|redelivery|storage fee|clearance charge|delivery attempt|out for delivery|was delivered|has been delivered)\b"
    r"|পার্সেল|কুরিয়ার|কাস্টমস|শুল্ক|ডেলিভারি হবে|ডেলিভারি ব্যর্থ|ডেলিভারি চার্জ",
    flags=re.IGNORECASE,
)
PARCEL_PHISH_CUE_RE = re.compile(
    r"\b(?:customs|duty|storage fee|redelivery|clearance charge|wrong address|incomplete|address confirmation|confirm(?: new)? address|"
    r"update address|address.{0,12}(?:vul|bhul)|(?:vul|bhul).{0,10}address|"
    r"could not be delivered|couldn'?t be delivered|attempt failed|release|"
    r"held|atke ase|failed|fee|charge|(?<!pathao\s)pay|click|tap|<URL_SUSP>|<URL_SHORT>|<URL>)\b"
    r"|আটকে|ব্যর্থ|চার্জ|ফি|পরিশোধ|শুল্ক|কাস্টমস|ঠিকানা আপডেট|ক্লিক|লিংকে",
    flags=re.IGNORECASE,
)

JOB_RE = re.compile(
    r"(?<!amar\s)(?<!my\s)\bjob\b(?!\s*jabe)|\b(?:hiring|vacancy|work[- ]?from[- ]?home|work frmo home|part[- ]?time|data entry|package handler|brand ambassador|"
    r"relief volunteer|typing job|agent license|earn(?:ing)?|income|kamai|kamao|salary|beton|task|tasks|watch and like|"
    r"like korle|facebook page|page like|per like|product review|review products|unlock tasks?|"
    r"commission|invest|investment|double your money|crypto signals?|arbitrage|bot|ludo|cash game|"
    r"freelanc(?:ing|e)|abroad job|trading crypto)\b"
    r"|লোক নিচ্ছে|ঘরে বসে|আয়|বেতন|বিনিয়োগ|দ্বিগুণ|ভাতা|হোয়াটসঅ্যাপে যোগ দিন",
    flags=re.IGNORECASE,
)
JOB_TRAP_RE = re.compile(
    r"\b(?:fee|deposit|security deposit|uniform fee|registration fee|clearance charge|service charge|visa fee|kit|"
    r"id card|invest|enroll now|per task|per post|per day|per like|like e \d+|per review|daily|weekly profit|profit per day|"
    r"guarantees?\s+\d+%|deposit tk \d+.{0,35}get tk \d+|10 minutes\.\s*guaranteed|commission|dite hobe|unlock korun|apply korun|"
    r"join (?:whatsapp|telegram|the group|vip)|group e join|whatsapp e join|join korun|whatsapp us|complete \d+ tasks?)\b"
    r"|রেজিস্ট্রেশন ফি|ডিপোজিট|দ্বিগুণ|নিশ্চিত লাভ|প্রতিদিন|যোগ দিন|যোগাযোগ করুন|সার্ভিস চার্জ|ভিসা ফি|টাকা দিন|হোয়াটসঅ্যাপে|টেলিগ্রামে",
    flags=re.IGNORECASE,
)

PROMO_RE = re.compile(
    r"\b(?:discount|sale|eid offer|winter sale|flash sale|11\.11|end of season|mega offer|1 ta kinle 1 ta free|"
    r"buy \d+ get \d+ free|\d+% (?:off|discount|cashback)|(?:tk|bdt|৳|\$)\s*\d+\s*off|off your first order|"
    r"0% emi|emi|chhar|voucher|vouchers up to|use code|code:\s*[a-z0-9]+|promo|shortto projojjo|t&c apply|"
    r"book your dream (?:flat|apartment|home)|easy installment|admission (?:going on|cholche)|learn english|combo|refer a friend|"
    r"special offer|\d+\s*gb\s*(?:for|at)\s*tk|"
    r"recharge tk \d+ and get|\d+ ?gb (?:extra|bonus)|new app update|exciting features|nearest outlet|limited time only)\b"
    r"|ছাড়|অফার|শর্ত প্রযোজ্য|রিচার্জে পান|বিল পেমেন্টে|আজই কিনুন|কিনুন",
    flags=re.IGNORECASE,
)

# High-precision Hard Legitimate patterns
HARD_LEGIT_COD_RE = re.compile(
    r"\b(?:cod tk [\d,]+ ready rakhun|keep tk [\d,]+ ready for cod|cash on delivery|collected in cash by our rider|"
    r"no online payment is needed|pay when it arrives)\b",
    flags=re.IGNORECASE,
)
HARD_LEGIT_OTP_RE = re.compile(
    r"(?:\b\d{4,8}\b.{0,45}\b(?:otp|code)\b|\b(?:otp|verification code|login code|confirmation code)\b.{0,35}\b\d{4,8}\b"
    r"|ওটিপি কোড.{0,20}\d{4,8})",
    flags=re.IGNORECASE,
)
HARD_LEGIT_TX_RE = re.compile(
    r"\b(?:cash[- ]?out.{0,25}successful|add money.{0,25}successful|mobile recharge.{0,25}successful|"
    r"payment.{0,20}successful|prepaid recharge.{0,20}successful|credited to a/?c|debited from a/?c|avl bal|"
    r"you have received tk|apni tk [\d,]+ peyechen|account e tk [\d,]+ add hoyeche|"
    r"has been credited|earned tk [\d,]+ cashback on your bill payment|current balance tk|"
    r"pathano hoyeche|trxid|bill.{0,25}(?:due|last date|shesh tarikh)|"
    r"(?:is this the .{0,20}agent.{0,40}cash\s*out)|send money to agent number|received,\s*here is your cash)\b"
    r"|ক্যাশ আউট সফল|পেয়েছেন|ট্রানজেকশন আইডি|অ্যাকাউন্টে.*জমা হয়েছে|ব্যালেন্স.*টাকা",
    flags=re.IGNORECASE,
)
HARD_LEGIT_HR_DOC_RE = re.compile(
    r"\b(?:offer letter attached|joining date:|application.{0,25}has been received|shortlisted candidates|"
    r"do not pay anyone for this process|interview schedule|bring cv and nid|cv ar nid copy niye ashben|"
    r"pay tuition via bank booth|official portal only)\b",
    flags=re.IGNORECASE,
)
HARD_LEGIT_DELIV_RE = re.compile(
    r"\b(?:out for delivery today|has been shipped|ship hoyeche|was delivered|has been delivered|arrived at .{0,20}hub|"
    r"collect from branch|will arrive|will retry tomorrow.{0,30}(?:call|reschedule)|096\d{2}-.{0,20}to reschedule)\b"
    r"|আজ ডেলিভারি হবে|আজকে ডেলিভারি হবে|ডেলিভারি হবে",
    flags=re.IGNORECASE,
)
HARD_LEGIT_CHAT_RE = re.compile(
    r"\b(?:forgot.{0,18}pin|how can i reset|choose\s*reset\s*pin|did you get my .{0,20}transfer|"
    r"dekh to ashche kina|bazar korish|p2p|dekha hobe|happy birthday|dinner tonight|"
    r"library: the book|dentist'?s office|meeting moved to|class kal|exam er routine|"
    r"study group|pack expires on)\b",
    flags=re.IGNORECASE,
)

# Entity Impersonation (strict T1: pretending to BE an entity/officer/helpline, not casual mention)
ENTITY_T1_RE = re.compile(
    r"(?:^(?:bkash|nagad|rocket|upay|surecash|mcash|tap|ok wallet|pathao|daraz|bdjobs|"
    r"sundarban|sa paribahan|paribahan|redx|steadfast|paperfly|ecourier|dhl|fedex|"
    r"gp|grameenphone|robi|banglalink|airtel|teletalk|skitto|jamuna|dbbl|brac|islami|city|prime)\b|"
    r"\b(?:courier notice|customs hold|security team|security department|fraud control|customer care|refund team|"
    r"this is (?:dhl|fedex|sundarban|pathao|redx|steadfast|bkash|nagad|rocket|upay)|"
    r"from (?:gp|grameenphone|robi|banglalink|airtel|daraz)|your (?:bkash|nagad|rocket|upay|dbbl|city bank|islami bank)|"
    r"helpline|officer|representative|unilever bangladesh|beximco|bdjobs|bangladesh bank|nbr|btrc|"
    r"desco|nesco|wasa|titas|breb|dpdc|western union|pm education|govt|government|"
    r"ami .{0,15}agent|dear (?:nagad|bkash|rocket|upay|gp|robi) (?:user|customer)|grameenphone lucky draw|"
    r"robi eid lucky draw|bkash anniversary|urgent bkash|urgent rocket|sim off)\b)"
    r"|বিকাশ|নগদ|রকেট|উপায়|গ্রামীণফোন|রবি|বাংলালিংক|দারাজ|পাঠাও|সুন্দরবন|সরকার|হেল্পলাইন|কর্মকর্তা",
    flags=re.IGNORECASE,
)

URGENCY_T5_RE = re.compile(
    r"\b(?:within \d+|24 ?h|12 ?h|1 hour|24 ghonta|expir(?:e|es|ed)|will be block|will be returned|will be deactivated|"
    r"band hobe|bondho|off hoye|suspend|last warning|final notice|flagged|immediately|taratari|hurry|"
    r"bacht|bachte|bipode|trouble|job jabe|avoid cancellation|ekhoni|today only|asap|na hole|claim within)\b"
    r"|এখনই|নইলে|আজই|বন্ধ হয়ে|বাতিল|তাড়াতাড়ি|সতর্ক|জরুরি|বিপদে|ঘণ্ট|মিনিট",
    flags=re.IGNORECASE,
)


def extract_rule_prediction(raw_text: str):
    """Inductive Guide-aligned classifier & 58-dim feature vector (24 base + 34 rule feats)."""
    clean = clean_for_model(raw_text)
    masked, uflags = mask_urls_for_text_rules(clean)
    # Strip speaker prefixes and decorative EMOJI tokens before semantic rule checks
    t = SPEAKER_PREFIX_RE.sub(" ", masked)
    t = re.sub(r"<EMOJI_\w+>", " ", t)

    scam_warn_thread = bool(SCAM_WARN_THREAD_RE.search(t))
    mistake = bool(MISTAKE_RE.search(t)) and bool(RETURN_RE.search(t)) and bool(
        re.search(r"\b(?:tk|bdt|taka|money)\b|টাকা|\d{2,}", t, re.I)
    )

    # Segment-level active credential harvest check
    req_secret = False
    for seg in re.split(r"[.!?\n\|]+", t):
        seg = seg.strip()
        if not seg:
            continue
        has_sens = bool(SENS_RE.search(seg)) or (
            bool(CODE_PLAIN_RE.search(seg))
            and not bool(re.search(r"\b(?:use code|code:|promo|voucher)\b", seg, re.I))
        )
        if has_sens and ASK_SECRET_RE.search(seg) and not NEG_WARN_RE.search(seg):
            if not HARD_LEGIT_HR_DOC_RE.search(seg):
                req_secret = True
                break
    if not req_secret:
        # Cross-clause explicit harvest (guarded against legit OTP/scam warnings without imperative ask)
        for seg in re.split(r"[.!?\n\|]+", t):
            seg = seg.strip()
            if seg and not NEG_WARN_RE.search(seg) and re.search(
                r"(?:nid.{0,15}both sides|photo of your nid|nid photo|card number.{0,20}cvv|cvv den|"
                r"enter(?:ing)? (?:your )?(?:pin|otp)|confirm(?: your)? (?:internet banking )?(?:password|otp|pin)|"
                r"send (?:your )?(?:pin|otp|nid|wallet pin)|tell me your pin|read out the code)",
                seg,
                re.I,
            ):
                req_secret = True
                break

    # Safe OTP warning vs Scam warning thread (never override unquoted active credential harvest)
    neg_warn = bool(NEG_WARN_RE.search(t))
    quoted_scam_framing = bool(
        ("|||" in clean and scam_warn_thread)
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
    is_otp_legit = bool(HARD_LEGIT_OTP_RE.search(t)) and neg_warn and not req_secret
    is_deliv_legit = (
        bool(HARD_LEGIT_DELIV_RE.search(t))
        and not req_secret
        and not (
            PARCEL_PHISH_CUE_RE.search(t)
            and not bool(re.search(r"will retry tomorrow|096\d{2}-", t, re.I))
        )
    )
    is_chat_legit = bool(HARD_LEGIT_CHAT_RE.search(t)) and not req_secret and not mistake
    hard_legit = is_cod or is_hr_legit or is_tx_legit or is_otp_legit or is_deliv_legit or is_chat_legit or (clean == "<NOISE>")

    has_parcel = bool(PARCEL_RE.search(t))
    has_parcel_phish = bool(PARCEL_PHISH_CUE_RE.search(t))
    has_job = bool(JOB_RE.search(t))
    has_job_trap = bool(JOB_TRAP_RE.search(t))
    has_claim = bool(CLAIM_RE.search(t))
    has_promo = bool(PROMO_RE.search(t))

    # Strict Priority Prize vs Parcel/Job conflict resolution
    has_strong_prize = bool(STRONG_PRIZE_RE.search(t))
    has_soft_prize = bool(SOFT_PRIZE_RE.search(t)) and has_claim and not (has_parcel and has_parcel_phish) and not (has_job and has_job_trap)
    has_prize = has_strong_prize or has_soft_prize

    # Strict Section 4 Priority Order: 1 > 2 > 4 > 3 > 5 > 0
    if is_scamwarn_only:
        lbl = 0
    elif req_secret or mistake:
        lbl = 1
    elif hard_legit:
        lbl = 0
    elif has_promo and not (
        (has_parcel and has_parcel_phish)
        or (has_job and has_job_trap)
        or uflags["has_susp_tld"]
        or uflags["has_shortener"]
        or req_secret
        or has_strong_prize
        or re.search(r"\b(?:processing fee|survey|airdrop)\b|প্রসেসিং ফি", t, re.I)
    ):
        lbl = 5
    elif has_prize and (has_claim or uflags["has_url"]) and not (has_parcel and has_parcel_phish and (not re.search(r"\b(?:won|lottery|lucky draw)\b|লাকি ড্র|লটারি", t, re.I) or bool(re.search(r"limited winners", t, re.I)))) and not (has_job and has_job_trap):
        lbl = 2
    elif has_parcel and (has_parcel_phish or uflags["has_susp_tld"] or uflags["has_shortener"]) and not re.search(r"\b(?:package handler|salary|ambassador)\b", t, re.I):
        lbl = 4
    elif has_job and has_job_trap:
        lbl = 3
    elif has_promo:
        lbl = 5
    else:
        lbl = 0

    # Derive Rule-aligned Task C Action (1 > 2 > 4 > 3 > 5) and Task B Techniques
    if lbl in (0, 5):
        act = 0
        tvec = np.zeros(6, dtype=np.int32)
    else:
        req_pii_reply = bool(re.search(r"\b(?:reply|send|share|give)\b.{0,25}\b(?:name and address|full name|address|date of birth|dob)\b", t, re.I))
        has_pay = bool(re.search(
            r"\b(?:(?<!pathao\s)pay|deposit|invest|ferot|return|send (?:it )?back|reverse|enroll now with tk)\b"
            r"|\b(?:registration|processing|delivery|redelivery|storage|service|clearance)\s*(?:fee|charge|duty)\b.{0,22}\b(?:tk|bdt|\d|dite|(?<!pathao\s)pay|din|baki)"
            r"|\b(?:customs|storage|clearance)\s*duty\b.{0,25}\b(?:tk|bdt|\d|bkash)"
            r"|\b(?:tk|bdt|\$)\s*[\d,]+\s*(?:fee|duty|deposit|charge|ferot|dite hobe|bkash korun|nagad korun)\b"
            r"|পরিশোধ|ডিপোজিট|বিনিয়োগ|ফেরত|(?:টাকা|ফি|চার্জ)\s*(?:দিন|দিতে হবে)",
            t,
            re.I,
        ))
        # Distinguish "visa fee Tk 80,000 ... Call 01..." (CTA=Call) from "Pay Tk 3,500 fee" (CTA=Pay)
        has_pay_verb = bool(re.search(r"\b(?:(?<!pathao\s)pay|deposit|invest|enroll now with|ferot|send (?:it )?back|reverse|bkash korun|nagad korun)\b|পরিশোধ|ডিপোজিট|বিনিয়োগ|ফেরত|দিতে হবে|টাকা দিন", t, re.I))
        has_join_apk = bool(re.search(
            r"\b(?:install|download|join (?:the )?(?:group|vip|whatsapp|telegram)|whatsapp e join|telegram e join|group e join|join korun|"
            r"complete \d+ tasks?|watch and like|buy products|review products|rewards app|relief volunteer)\b|<URL_APK>|<URL_TGWA>"
            r"|যোগ দিন|হোয়াটসঅ্যাপে যোগ|টেলিগ্রামে যোগ|গ্রুপে",
            t,
            re.I,
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
        else:
            act = {2: 3, 3: 4, 4: 3}[lbl]

        tvec = np.zeros(6, dtype=np.int32)
        if ENTITY_T1_RE.search(t) and not (lbl == 3 and not re.search(r"\b(?:license|ambassador|typing job|data entry|officer|hr|unilever|bdjobs|beximco|walton|rokomari|pickaboo)\b", t, re.I)):
            tvec[0] = 1
        if lbl == 1 and (req_secret or not mistake):
            tvec[1] = 1
        if (act == 2) or (lbl != 1 and has_pay) or (lbl == 3 and re.search(r"\b(?:invest|profit|salary|fee|deposit)\b", t, re.I)):
            tvec[2] = 1
        if lbl == 2 or has_prize:
            tvec[3] = 1
        if URGENCY_T5_RE.search(t) or mistake:
            tvec[4] = 1
        if lbl == 3 or has_job:
            tvec[5] = 1
        if tvec.sum() == 0:
            tvec[{1: 1, 2: 3, 3: 5, 4: 2}[lbl]] = 1

    rule_feats = [
        float(lbl == 0), float(lbl == 1), float(lbl == 2), float(lbl == 3), float(lbl == 4), float(lbl == 5),
        float(req_secret), float(mistake), float(scam_warn_thread), float(hard_legit),
        float(is_cod), float(is_hr_legit), float(is_tx_legit), float(is_otp_legit),
        float(has_prize), float(has_claim), float(has_parcel and has_parcel_phish),
        float(has_job and has_job_trap), float(has_promo),
        float(tvec[0]), float(tvec[1]), float(tvec[2]), float(tvec[3]), float(tvec[4]), float(tvec[5]),
        float(act == 1), float(act == 2), float(act == 3), float(act == 4), float(act == 5),
        float(is_deliv_legit), float(is_chat_legit), float(neg_warn), float(lbl != 0 and lbl != 5),
    ]
    feats = np.array(extract_base_struct_v4(raw_text, clean) + rule_feats, dtype=np.float32)
    return lbl, tvec, act, feats


if __name__ == "__main__":
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cv_ood import tech_str_to_vec, vec_to_tech_str, score_section10_proxy

    df = pd.read_csv("Main Preli Round/dataset/sample.csv")
    pl, pt, pa = [], [], []
    for _, r in df.iterrows():
        l, tv, a, f = extract_rule_prediction(r["text"])
        pl.append(l); pt.append(tv); pa.append(a)
    pl, pt, pa = np.array(pl), np.vstack(pt), np.array(pa)
    yt = np.vstack([tech_str_to_vec(x) for x in df["techniques"]])
    res = score_section10_proxy(
        df["label"].values, pl, yt, pt, df["action"].values, pa,
        np.where(pl == df["label"].values, 0.95, 0.25),
    )
    print(f"feats dim = {f.shape[0]} | rules_v4 on sample.csv (zero tuning):", res)
    for i, r in df.iterrows():
        gt = vec_to_tech_str(yt[i])
        pt_s = vec_to_tech_str(pt[i])
        if pl[i] != r["label"] or pa[i] != r["action"] or pt_s != gt:
            print(f"  MISS id={r['id']} L:{r['label']}->{pl[i]} A:{r['action']}->{pa[i]} T:{gt}->{pt_s} | {r['text'][:90]}")
