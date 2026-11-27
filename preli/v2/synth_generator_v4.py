import random
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from v2.cv_ood import TECH_LIST, vec_to_tech_str
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cv_ood import TECH_LIST, vec_to_tech_str

MFS_BRANDS = ["bKash", "Nagad", "Rocket", "Upay", "SureCash", "mCash", "Tap", "OK Wallet", "Pathao Pay"]
BANKS = ["DBBL", "Dutch-Bangla Bank", "Islami Bank", "BRAC Bank", "City Bank", "Prime Bank", "Pubali Bank", "Jamuna Bank", "Standard Chartered", "UCB", "EBL", "Trust Bank"]
COURIERS = ["Sundarban Courier", "SA Paribahan", "Pathao Courier", "RedX", "Steadfast", "Paperfly", "Karatoa", "eCourier", "DHL", "FedEx", "Janani Courier", "Sheba Delivery"]
TELECOMS = ["GP", "Grameenphone", "Robi", "Banglalink", "Airtel", "Teletalk", "Skitto"]
ECOMMS = ["Daraz", "Rokomari", "Ajkerdeal", "Star Tech", "Aarong", "Shwapno", "Chaldal", "Pickaboo"]
EMPLOYERS = ["Unilever Bangladesh", "Beximco", "Walton", "Square Group", "PRAN-RFL", "Apex", "Robi Axiata", "bKash Limited", "Daraz"]
GOV_AUTHORITIES = ["Bangladesh Bank", "NBR Tax Office", "NID Wing", "BTRC", "Customs House", "Passport Office", "PM Education Fund", "Social Welfare Office"]
UTILITIES = ["DESCO", "NESCO", "WASA", "Titas Gas", "BTCL", "DPDC", "BREB"]
SCAM_TLDS = ["top", "xyz", "site", "online", "vip", "link", "click", "info", "co", "cc", "icu", "live"]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy"]

SPEAKER_STYLES = [
    ("S: ", "R: "),
    ("A: ", "B: "),
    ("[Sender]: ", "[You]: "),
    ("[sender]: ", "[you]: "),
    ("Caller: ", "You: "),
    ("caller: ", "you: "),
    ("", ""),
]

INNOCENT_WRAPPERS = [
    ("We hope this message finds you well. ", ""),
    ("Subject: Account & service update. Dear Valued Customer, ", " Thanks, Customer Service"),
    ("As part of our regular communication, please read carefully. ", " Thank you for your cooperation."),
    ("Subject: Message from {brand}. Dear Customer, ", " Reply STOP to opt out."),
]


def _rand_phone(rng):
    op = rng.choice(["013", "014", "015", "016", "017", "018", "019"])
    p = f"{op}XXXXXX{rng.randint(10,99)}"
    return f"+880 {p}" if rng.random() < 0.35 else p


def _rand_code(rng):
    return "".join(rng.choices("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ", k=rng.randint(6, 10)))


def _rand_scam_url(rng, brand=None):
    if rng.random() < 0.16:
        return f"https://{rng.choice(SHORTENERS)}/{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))}"
    b = (brand or rng.choice(MFS_BRANDS + COURIERS + TELECOMS + ECOMMS)).lower().replace(" ", "")
    tag = rng.choice(["verify", "kyc", "claim", "track", "fee", "help", "support", "login", "secure", "reward", "bonus"])
    num = str(rng.randint(1, 99)) if rng.random() < 0.5 else ""
    tld = rng.choice(SCAM_TLDS)
    path = rng.choice(["/pay", "/verify", "/claim", "/login", "/" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5)), ""])
    return f"https://{b}-{tag}{num}.{tld}{path}"


def _rand_official_url(rng, brand):
    b = brand.lower().replace(" ", "").replace("-", "")
    tld = rng.choice(["com", "com.bd", "org.bd", "net"])
    path = rng.choice(["/track", "/offers", "/pay", "/app", "/careers", ""])
    return f"https://{b}.{tld}{path}"


def wrap_multiturn_v4(payload_msgs, rng: random.Random, allow_group=False, min_intro=1, max_intro=4):
    if allow_group and rng.random() < 0.25:
        names = ["James", "Nusrat", "Rahim", "Michael", "Karim", _rand_phone(rng)]
        n1, n2, n3 = rng.sample(names, 3)
        turns = [
            f"[Group] {n1}: {rng.choice(['anyone has the slides?', 'hi everyone', 'good evening', 'did you check the email?'])}",
            f"[Group] {n2}: {rng.choice(['i will share tonight', 'check the thread', 'just saw it', 'wait a second'])}",
        ]
        for m in payload_msgs:
            turns.append(f"[Group] {n3}: {m}")
        return " ||| ".join(turns)

    s_pfx, r_pfx = rng.choice(SPEAKER_STYLES)
    turns = []
    if rng.random() < 0.65:
        intros = [
            ("Hello", "Hi, who is this?"),
            ("Good morning sir", "Good morning, who's this?"),
            ("Brother, ekta help lagbe", "ki help?"),
            ("Hi, I got your number from a friend", "Which friend?"),
            ("Assalamu alaikum", "Walaikum assalam, ke bolchen?"),
            ("Apu, ekta kotha chilo", "bolen"),
            (f"Sir, apni ki {rng.choice(MFS_BRANDS)} use koren?", "ha, keno?"),
            ("Hello dear, found your number on Facebook", "who are you?"),
        ]
        for q, a in rng.sample(intros, rng.randint(min_intro, min(max_intro, len(intros)))):
            turns.append(f"{s_pfx}{q}")
            turns.append(f"{r_pfx}{a}")
    for i, m in enumerate(payload_msgs):
        turns.append(f"{s_pfx}{m}")
        if i < len(payload_msgs) - 1 and rng.random() < 0.70:
            turns.append(f"{r_pfx}{rng.choice(['ki?', 'kivabe?', 'kon company?', 'kothay?', 'thik ase, bolen', 'ok', 'ami to kichu bujhtesi na'])}")
    if rng.random() < 0.30:
        turns.append(f"{r_pfx}{rng.choice(['ami ekhon busy', 'really?', 'sure', 'ok let me check', 'seat kom!'])}")
    return " ||| ".join(turns)


def corrupt_text_v4(text: str, rng: random.Random) -> str:
    NOISE_WORDS = ["zzqwpo", "wkrjfejejrn", "mnbvcx", "qzxvbn", "xkcdpl", "asdkjh", "qundvpyfjqx", "wbmtzmh", "lkjsdfoiu"]
    EMOJIS = ["💰", "🎁", "🙏", "⚠️", "📲", "🔥", "😊"]
    CYR = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "i": "\u0456", "c": "\u0441", "p": "\u0440", "y": "\u0443", "x": "\u0445"}
    LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

    out = text
    r_url = rng.random()
    if r_url < 0.45 and "http" in out.lower():
        if r_url < 0.25:
            out = re.sub(r"https?://", lambda m: "hxxps://" if "s" in m.group(0).lower() else "hxxp://", out, flags=re.I)
            out = out.replace(".", "[.]")
        else:
            out = re.sub(r"https?://", lambda m: "h77p5://" if "s" in m.group(0).lower() else "h77p://", out, flags=re.I)
            out = out.replace(".", rng.choice(["[.]", "[]"]))

    words = out.split(" ")
    new_words = []
    for w in words:
        if w == "|||" or "http" in w.lower() or "hxxp" in w.lower() or "h77p" in w.lower():
            new_words.append(w)
            continue
        if rng.random() < 0.07:
            new_words.append(rng.choice(NOISE_WORDS))
        if w.isascii() and len(w) >= 3 and rng.random() < 0.28:
            chars = list(w)
            op = rng.random()
            if op < 0.24:
                pos = rng.randint(1, len(chars) - 1)
                chars.insert(pos, rng.choice(["\u200b", "\u200c"]))
            elif op < 0.48:
                for j, c in enumerate(chars):
                    if c.lower() in CYR and rng.random() < 0.35:
                        chars[j] = CYR[c.lower()]
            elif op < 0.68:
                for j, c in enumerate(chars):
                    if c.lower() in LEET and rng.random() < 0.30:
                        chars[j] = LEET[c.lower()]
            elif op < 0.85:
                j = rng.randint(0, len(chars) - 1)
                if chars[j].isalpha():
                    chars[j] = chars[j] * rng.randint(3, 4)
            elif len(chars) >= 4:
                j = rng.randint(0, len(chars) - 2)
                chars[j], chars[j + 1] = chars[j + 1], chars[j]
            w = "".join(chars)
        elif re.search(r"[\u0980-\u09FF]", w) and rng.random() < 0.14 and len(w) >= 2:
            pos = rng.randint(1, len(w) - 1)
            w = w[:pos] + "\u200c" + w[pos:]
        new_words.append(w)
        if rng.random() < 0.05:
            new_words.append(rng.choice(EMOJIS))

    res = " ".join(new_words)
    if rng.random() < 0.22:
        res = re.sub(r" ([a-z]{2,5}) ", lambda m: m.group(1) + "", res, count=1)
    rc = rng.random()
    if rc < 0.07:
        res = res.upper()
    elif rc < 0.22:
        res = res.lower()
    return res


def _wrap_innocent(txt, brand, rng, allow_rate=0.14):
    if "|||" in txt or rng.random() >= allow_rate:
        return txt
    pfx, sfx = rng.choice(INNOCENT_WRAPPERS)
    return f"{pfx.format(brand=brand)}{txt}{sfx}"


def gen_one_v4(rng: random.Random):
    """Generate one diverse sample with (text, label, tech, action, scenario_family)."""
    r = rng.random()
    if r < 0.31:
        lbl = 0
    elif r < 0.49:
        lbl = 1
    elif r < 0.63:
        lbl = 2
    elif r < 0.77:
        lbl = 3
    elif r < 0.89:
        lbl = 4
    else:
        lbl = 5

    mfs = rng.choice(MFS_BRANDS)
    bank = rng.choice(BANKS)
    cour = rng.choice(COURIERS)
    tel = rng.choice(TELECOMS)
    ecom = rng.choice(ECOMMS)
    emp = rng.choice(EMPLOYERS)
    util = rng.choice(UTILITIES)
    amt = rng.choice([100, 250, 450, 500, 1200, 1800, 2500, 4500, 10000, 25000])
    bal = amt + rng.randint(50, 50000)
    otp = rng.randint(100000, 999999)

    if lbl == 0:
        sub = rng.random()
        if sub < 0.18:
            fam = "legit_otp"
            txt = rng.choice([
                f"{mfs} OTP: {otp}. Kauke share korben na. {rng.randint(2,5)} minute valid.",
                f"Your {mfs} verification code is {otp}. {mfs} will never call you to ask for it.",
                f"{otp} is your {bank} login code. Do not share it with anyone, even {bank} staff.",
                f"{otp} ashe apnar {mfs} login code. Eta kauke bolben na. Meyad {rng.randint(2,5)} minute.",
                f"{bank} OTP {otp} for Tk {amt:,} payment to {ecom}. Never share this OTP with anyone.",
                f"আপনার {mfs} ওটিপি কোড {otp}। এটি কখনো কারো সাথে শেয়ার করবেন না।",
                f"Security alert: new device logged in to your {mfs} account. If this was you, ignore. We never ask for your PIN or OTP.",
            ])
        elif sub < 0.44:
            if rng.random() < 0.45:
                fam = "legit_agent_cashout"
                s_p, r_p = rng.choice(SPEAKER_STYLES[:5])
                txt = (
                    f"{s_p}Hello, is this the {mfs} agent? ||| {r_p}yes sir ||| "
                    f"{s_p}I want cash out {amt:,} ||| {r_p}ok send money to agent number {_rand_phone(rng)} ||| "
                    f"{s_p}sent ||| {r_p}received, here is your cash"
                )
            else:
                fam = "legit_tx_bill"
                txt = rng.choice([
                    f"{mfs}: Cash-out Tk {amt:,} successful. Fee Tk 0.00. Balance Tk {bal:,}. TrxID {_rand_code(rng)}.",
                    f"{mfs}: You have received Tk {amt:,} from {_rand_phone(rng)}. Balance Tk {bal:,}. TrxID {_rand_code(rng)}.",
                    f"{mfs}: Apni Tk {amt:,} peyechen {_rand_phone(rng)} theke. Current balance Tk {bal:,}.",
                    f"{mfs}: Account e Tk {amt:,} add hoyeche {_rand_phone(rng)} theke. TrxID {_rand_code(rng)}.",
                    f"{mfs}: আপনার {amt:,} টাকা ক্যাশ আউট সফল হয়েছে। বর্তমান ব্যালেন্স {bal:,} টাকা। ট্রানজেকশন আইডি {_rand_code(rng)}।",
                    f"{bank}: BDT {amt:,} debited from A/C XXXX{rng.randint(1000,9999)} at {ecom}. Avl bal BDT {bal:,}.",
                    f"{util}: Your bill for November is Tk {amt:,}. Last date {rng.randint(1,28)} Dec. Pay via app or {_rand_official_url(rng, util)}.",
                    f"{util}: Prepaid recharge Tk {amt:,} successful. Token: {rng.randint(1000,9999)}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}.",
                    f"Earned Tk {rng.choice([15, 25, 50])} cashback on your {util} bill payment via {mfs}. TrxID {_rand_code(rng)}.",
                    f"Your {ecom} refund of Tk {amt:,} has been credited to your {mfs} account.",
                    f"Dear customer, your {tel} {rng.choice([5,10,20])}GB pack expires on {rng.randint(1,28)} Nov. Dial *123*1# to renew.",
                ])
        elif sub < 0.60:
            fam = rng.choice(["legit_hr_doc", "legit_cod", "legit_tuition"])
            pid = f"{rng.choice(['SC','PF','RX','DZ','SP'])}{rng.randint(10000,99999)}"
            if fam == "legit_cod":
                txt = rng.choice([
                    f"{cour}: Your parcel {pid} is out for delivery today. COD Tk {amt:,} ready rakhun.",
                    f"{ecom}: Order #{pid} out for delivery. Keep Tk {amt:,} ready for COD. No online payment is needed.",
                    f"{cour}: Delivery today between 2-6 PM. Payment will be collected in cash by our rider.",
                    f"R: Is my order #{pid} here? ||| S: Rider called, will deliver at {rng.randint(3,8)} PM ||| R: ok I'll pay cash on delivery",
                    f"{cour}: Parcel {pid} arrived at Dhaka hub. Collect from branch with this SMS and a photo ID.",
                    f"{cour}: আপনার পার্সেল {pid} আজ ডেলিভারি হবে।",
                    f"Pathao: Your rider is arriving in {rng.randint(2,10)} min. Share your trip PIN only with the rider.",
                    f"{cour}: Delivery attempt unsuccessful. We will retry tomorrow. Call 09611-XXXXXX to reschedule.",
                ])
            elif fam == "legit_hr_doc":
                txt = rng.choice([
                    f"Offer letter attached. Joining date: 1 Jan. Salary BDT {rng.choice([25000,30000,45000,60000,80000]):,}. Please sign and return. - HR, {emp}",
                    f"Dear Applicant, your application for Engineer at {emp} has been received. Shortlisted candidates will be contacted via official email. Do not pay anyone for this process.",
                    f"Interview schedule: Tuesday 10 AM, {emp} office. Bring CV and NID copy. Reply YES to confirm. - HR, {emp}",
                    f"{emp} HR: Apnar interview kal 11 tay. CV ar NID copy niye ashben.",
                ])
            else:
                txt = f"NSU Accounts: Pay tuition for Spring via bank booth or official portal only ({_rand_official_url(rng, 'nsu')}). Do not pay any third party."
        elif sub < 0.72:
            fam = "legit_scam_warn"
            scam_q = rng.choice([
                "your SIM registration is incomplete. Share the 6-digit OTP you just received",
                "apnar bkash PIN din account block hoye jabe",
                "dear winner, you won a smart tv. pay tk 500 delivery charge",
                "apnar card flagged hoyeche, PIN ar CVV pathan",
            ])
            if rng.random() < 0.65:
                s_p, r_p = rng.choice(SPEAKER_STYLES[:5])
                warn = rng.choice([
                    "eta scam bhai, kauke PIN ba OTP dibi na, block kor",
                    "that is phishing. never share your PIN or OTP. block and report.",
                    "scam eta, kono taka ba code dibi na",
                    "ignore korsi, eto obvious scam",
                ])
                txt = f"{s_p}Someone from '{mfs} helpline' messaged me: '{scam_q}' ||| {r_p}{warn} ||| {s_p}thanks, blocking"
            else:
                txt = f"Sabdhan! Ekta SMS ashche je '{scam_q}'. Eta scam. Keu PIN ba OTP chaile dibi na."
        elif sub < 0.94:
            fam = "legit_chat_faq"
            s_p, r_p = rng.choice(SPEAKER_STYLES[:5])
            txt = rng.choice([
                f"{s_p}I forgot my {mfs} PIN. How can I reset it? ||| {r_p}Dial *247# and choose Reset PIN. Never share your PIN with anyone, including our staff.",
                f"Did you get my {mfs} transfer of Tk {amt:,}?",
                f"Ami ekhon {mfs.lower()} e {amt} taka pathaisi, dekh to ashche kina.",
                f"Kal {rng.randint(8,11)} tay dekha hobe, ok? ||| thik ase.",
                f"A: bhai, {amt} taka ferot dibi kobe? ||| B: sunday dibo, {mfs.lower()} e pathaya dibo ||| A: ok thik ase",
                f"Mama, ajke raate bari asbo. Bazar theke kichu lagbe?",
                f"NSU Library: The book you reserved is available. Please collect within 2 days.",
                f"Boss: Meeting moved to 3 PM. Please join the Google Meet link.",
                f"Happy birthday! Dinner tonight, my treat!",
            ])
        else:
            fam = "legit_noise"
            txt = " ".join("".join(rng.choices("bcdfghjklmnpqrstvwxzqaioeu", k=rng.randint(5, 13))) for _ in range(rng.randint(1, 3)))
            return txt, 0, "NONE", 0, fam
        txt = _wrap_innocent(txt, mfs, rng, allow_rate=0.18)
        return txt, 0, "NONE", 0, fam

    elif lbl == 1:
        # Priority #1 MFS / Credential Harvesting (no decoy leakage!)
        url = _rand_scam_url(rng, mfs)
        sub = rng.random()
        if sub < 0.42:
            fam = rng.choice(["mfs_alert_card", "mfs_account_block", "mfs_limit_aml", "mfs_breb_western"])
            t_set = {"T1", "T2"}
            if fam == "mfs_alert_card":
                t_set.add("T5")
                txt = rng.choice([
                    f"{mfs} Alert: your card ending {rng.randint(1000,9999)} has been flagged. Send your NID number and date of birth.",
                    f"{mfs} Alert: your card ending {rng.randint(1000,9999)} has been flagged. Share the 6-digit OTP you just received. Otherwise account will be suspended.",
                    f"{mfs} Helpline: Apnar account e unusual login detect hoyeche. Block theke bachte 6 digit code ta ei number e reply korun.",
                    f"Dear {mfs} customer, an unusual login was detected from another device. Enter your PIN at {url}",
                    f"{mfs} refund team: send card number, expiry and CVV to receive your Tk {amt:,} refund.",
                ])
            elif fam == "mfs_limit_aml":
                t_set.add("T5")
                txt = rng.choice([
                    f"{mfs} Security Team: your wallet limit upgrade has been approved. Confirm your password and OTP here: {url}",
                    f"{bank}: Your account is flagged under anti-money laundering review. Confirm your internet banking password at {url}",
                    f"SABDHAN! {mfs.upper()}: APNAR WALLET LIMIT BARANO HOBE. CARD NUMBER, EXPIRY AR CVV DEN. KAUKE BOLBEN NA.",
                    f"Dear {mfs} user, KYC update na korle account 24 ghontar moddhe band hobe. Update korte NID number o PIN pathan.",
                ])
            elif fam == "mfs_breb_western":
                t_set.add("T5")
                txt = rng.choice([
                    f"Western Union: Tk {amt:,} remittance from your brother abroad is ready. Verify by sending NID photo and {mfs} wallet PIN.",
                    f"BREB Meter notice: Your prepaid meter balance is critically low. Enter OTP at {url} to avoid disconnection.",
                    f"প্রিয় গ্রাহক, আপনার {mfs} অ্যাকাউন্ট বন্ধ হয়ে যাবে। এখনই আপনার পিন নম্বর ও ওটিপি পাঠান।",
                ])
            else:
                if rng.random() < 0.7:
                    t_set.add("T5")
                txt = rng.choice([
                    f"Apnar {mfs} account verify korte NID er front back photo ar OTP pathan, na hole SIM off hoye jabe.",
                    f"Your {mfs} PIN will expire today. Reply with your current PIN and a new PIN to continue service.",
                ])
            if rng.random() < 0.35:
                txt = wrap_multiturn_v4([txt], rng, allow_group=True)
            if rng.random() < 0.22:
                txt = txt + " Do not tell anyone about this."
            txt = _wrap_innocent(txt, mfs, rng, allow_rate=0.14)
            return txt, 1, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), 1, fam

        elif sub < 0.74:
            # Wrong-transfer (PIN theft act=1 vs ferot money act=2) + borrowed wallet
            if rng.random() < 0.55:
                fam = rng.choice(["wrong_tx_pin", "wrong_tx_readout", "borrowed_wallet_pin"])
                t_set = {"T2", "T5"}
                if fam == "wrong_tx_readout":
                    t_set.add("T1")
                    txt = rng.choice([
                        f"URGENT {mfs}: a wrong transaction of Tk {amt:,} was made from your account. Read out the code that was just sent to you.",
                        f"S: {mfs} agent bolchi, vul kore apnar number e Tk {amt:,} chole gese ||| S: account verify korte apnar PIN ar OTP ta bolen.",
                    ])
                elif fam == "borrowed_wallet_pin":
                    txt = rng.choice([
                        f"Mama ami bipode poreche, {mfs.lower()} e taka nai. Tomar PIN ta dao ami ektu send kore nei.",
                        f"Brother amar wallet limit ses, tomar {mfs} number ar PIN ta ektu diba please? Amar khub emergency.",
                    ])
                else:
                    t_set.add("T1")
                    txt = f"Ami {mfs} agent, bhul number-e Tk {amt} pathaisi. Ferot din, PIN ta bolen."
                if rng.random() < 0.35:
                    txt = wrap_multiturn_v4([txt], rng, allow_group=True)
                txt = _wrap_innocent(txt, mfs, rng, allow_rate=0.15)
                return txt, 1, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), 1, fam
            else:
                fam = "wrong_tx_return_money"
                txt = rng.choice([
                    f"Bhai ekta bhul hoye geche. Amar {amt} taka vul kore apnar number e chole gese. Please ferot den {_rand_phone(rng)} e. Ami onek bipode.",
                    f"Hi, I accidentally sent Tk {amt:,} to your number. Please send it back to {_rand_phone(rng)}.",
                    f"ভাই ভুল করে আপনার নম্বরে {amt:,} টাকা ফ্লেক্সিলোড হয়ে গেছে, দয়া করে {_rand_phone(rng)} নম্বরে ফেরত পাঠান।",
                    f"Mistaken transfer of Tk {amt:,} sent to your account. Kindly reverse it or return via {mfs} to {_rand_phone(rng)} immediately.",
                ])
                if rng.random() < 0.35:
                    txt = wrap_multiturn_v4([txt], rng, allow_group=True)
                txt = _wrap_innocent(txt, mfs, rng, allow_rate=0.22)  # Explicit fix for INNOCENT_WRAPPERS + wrong tx!
                return txt, 1, "T3|T5", 2, fam

        else:
            # Priority-1 cross-category traps (Scholarship/Prize/Parcel/Job asking PIN/OTP/CVV!)
            fam = rng.choice(["pri1_scholarship_pin", "pri1_parcel_pin", "pri1_prize_pin", "pri1_job_pin"])
            if fam == "pri1_scholarship_pin":
                t_set = {"T1", "T2", "T5"}
                txt = f"HSC Scholarship board: Apnar Tk {amt:,} scholarship approved. Dispurse korte {mfs} PIN o OTP ekhoni pathan."
            elif fam == "pri1_parcel_pin":
                t_set = {"T1", "T2"}
                txt = rng.choice([
                    f"A friend abroad sent you a gift worth 1800 pounds held at customs. To proceed send your {mfs} PIN and NID photo.",
                    f"{cour}: Apnar parcel customs e atke ase. Release korte PIN o OTP pathan.",
                ])
            elif fam == "pri1_prize_pin":
                t_set = {"T1", "T2", "T4"}
                txt = rng.choice([
                    f"Congratulations! Apni {mfs} draw te Tk {amt:,} jitechen. Claim korte apnar OTP share korun.",
                    f"{mfs} bonus: apni Tk {amt:,} bonus peyechen. Wallet e add korte *167# dial kore PIN din.",
                    f"BN National Prize draw: Apni 1 lakh taka jitechen. Claim korte apnar {mfs} OTP bolun.",
                ])
            else:
                t_set = {"T2", "T5", "T6"}
                txt = f"Ghor bose {amt} taka daily income! Register korte apnar {mfs} PIN pathan, seat ses!"
            txt = _wrap_innocent(txt, mfs, rng, allow_rate=0.14)
            return txt, 1, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), 1, fam

    elif lbl == 2:
        brand = rng.choice(MFS_BRANDS + TELECOMS + ECOMMS + GOV_AUTHORITIES)
        fee = rng.choice([50, 99, 150, 250, 300, 500])
        prize = rng.choice([f"Tk {amt:,} cashback", f"Tk {amt:,} prize", "iPhone 15", "Smart TV", "Toyota car", "free smartphone", "10GB data gift", "Free Wi-Fi router"])
        url = _rand_scam_url(rng, brand)
        act = rng.choices([3, 2, 5, 4, 1], weights=[0.32, 0.28, 0.20, 0.12, 0.08])[0]
        t_set = {"T4"}
        if act == 3:
            fam = rng.choice(["prize_link_en", "prize_link_bn", "prize_usdt_survey", "prize_mystery_box"])
            if fam == "prize_usdt_survey":
                txt = rng.choice([
                    f"USDT Airdrop: Claim 500 USDT reward! Connect your wallet here: {url}",
                    f"2-min customer survey: Complete to claim Tk {amt:,} gift voucher! Click {url}",
                ])
            elif fam == "prize_mystery_box":
                txt = f"{ecom} anniversary gift: won a Mystery Box worth Tk {amt:,}! Claim at {url}"
                t_set.add("T1")
            elif fam == "prize_link_bn":
                txt = f"অভিনন্দন! আপনি {amt:,} টাকা ক্যাশব্যাক জিতেছেন। পেতে নিচের লিংকে ক্লিক করুন। {url if rng.random()<0.6 else ''}"
                t_set.add("T5")
            else:
                txt = rng.choice([
                    f"Free {prize} from {brand} for you! Tap {url} now, offer expires in 1 hour.",
                    f"Apni {amt:,} taka jitechen. Claim korte link e click korun. {url}",
                ])
                t_set.update({"T1", "T5"})
        elif act == 2:
            fam = rng.choice(["prize_fee_en", "prize_router_allowance", "prize_fee_bn"])
            t_set.add("T3")
            if fam == "prize_router_allowance":
                txt = rng.choice([
                    f"Free Wi-Fi router for all {tel} fiber users! Pay only Tk {fee} delivery fee via {_rand_phone(rng)}.",
                    f"Dear citizen, you are eligible for old-age allowance arrears of Tk {amt:,}. Pay Tk {fee} service charge to release the amount.",
                    f"PM Education Fund: your child is selected for a Tk {amt:,} stipend. Submit your bKash number and pay a Tk {fee} verification fee.",
                    f"Apni {prize} er jonno select hoyechen. Pete Tk {fee} processing fee din. Taratari korun.",
                ])
                t_set.add("T1")
            elif fam == "prize_fee_bn":
                txt = f"অভিনন্দন! আপনি {prize} জিতেছেন। পেতে প্রসেসিং ফি {fee} টাকা দিন {_rand_phone(rng)}।"
            else:
                txt = f"You won {prize} from {brand} lucky draw! Pay Tk {fee} processing fee to claim via {_rand_phone(rng)}."
                t_set.add("T1")
        elif act == 5:
            # Explicit high coverage of Bangladeshi Prize Action=5 CTA ("যোগাযোগ করুন" / "WhatsApp korun" / "Call")!
            fam = "prize_call_wa_bn_en"
            t_set.add("T1")
            txt = rng.choice([
                f"আপনার নম্বরটি {brand} লাকি ড্রতে নির্বাচিত হয়েছে। পুরস্কার পেতে এই নম্বরে যোগাযোগ করুন {_rand_phone(rng)}।",
                f"অভিনন্দন! আপনি {amt:,} টাকা বাম্পার পুরস্কার জিতেছেন। সংগ্রহ করতে এখনই কল করুন {_rand_phone(rng)}।",
                f"Congratulations! Apnar number ti {brand} Eid lucky draw te select hoyeche. {prize} jitar jonno {_rand_phone(rng)} e WhatsApp korun.",
                f"Lucky draw winner! {brand} Bumper reward Tk {amt:,} jitechen! Call {_rand_phone(rng)} now to claim.",
            ])
        elif act == 4:
            fam = "prize_apk_rewards"
            t_set.add("T1")
            txt = rng.choice([
                f"{brand} anniversary: spin and win Tk {amt:,}! Download the spin app: {url}/app.apk",
                f"Install the {mfs} Rewards app and get Tk {amt:,} instantly: {url}",
            ])
        else:
            fam = "prize_pii_reply"
            txt = f"Lucky draw winner! Reply with your full name and address to receive a {prize}."
        if rng.random() < 0.28:
            txt = wrap_multiturn_v4([txt], rng, allow_group=True)
        txt = _wrap_innocent(txt, brand, rng, allow_rate=0.14)
        return txt, 2, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), act, fam

    elif lbl == 3:
        pay = rng.choice([500, 1500, 2000, 3000, 3500, 40000])
        dep = rng.choice([60, 200, 450, 700, 2000, 3500, 5000, 80000])
        t_set = {"T6"}
        act = rng.choices([4, 2, 5], weights=[0.42, 0.44, 0.14])[0]
        if act == 4:
            fam = rng.choice(["job_task_like", "job_crypto_arb", "job_volunteer_review"])
            if fam == "job_crypto_arb":
                t_set.add("T5")
                txt = rng.choice([
                    f"Crypto signals group - 300% weekly profit guaranteed! Join VIP: https://t.me/vip_crypto_bd",
                    f"AI arbitrage bot guarantees 25% profit per day. Join the group: {_rand_scam_url(rng, 'arb')}",
                    f"I earn 2,500 USD weekly trading crypto, I can teach you! Join Telegram: https://t.me/fx_bd_team",
                ])
            elif fam == "job_volunteer_review":
                txt = rng.choice([
                    f"{ecom} product review job: earn Tk {pay:,}/day! Join our WhatsApp group to start tasks.",
                    f"Relief volunteer recruitment: Tk {pay:,}/day allowance! Download volunteer form app: {_rand_scam_url(rng, 'relief')}/app.apk",
                ])
            else:
                txt = rng.choice([
                    f"Part-time job: like YouTube videos, earn Tk {pay} per task. Join Telegram: https://t.me/earn_bd_tasks",
                    f"Facebook page like korle protek like e 50 taka. Group e join korun.",
                    f"ঘরে বসে প্রতিদিন {pay} টাকা আয় করুন। আজই হোয়াটসঅ্যাপে যোগ দিন।",
                    f"Ghor bose din e {pay} taka income. WhatsApp e join korun.",
                    f"Task: buy products on {ecom} via our link, we refund + 20% commission. Complete 5 tasks, get Tk {pay:,}.",
                ])
        elif act == 2:
            fam = rng.choice(["job_bn_hiring_reg", "job_package_ambassador", "job_data_entry", "job_double_invest"])
            t_set.add("T3")
            if fam == "job_bn_hiring_reg":
                t_set.update({"T1", "T5"})
                txt = rng.choice([
                    f"{emp} এ লোক নিচ্ছে! বেতন {pay:,} টাকা। রেজিস্ট্রেশন ফি {dep} টাকা দিন {_rand_phone(rng)}।",
                    f"{emp} e lok niche! Beton {pay:,} taka. Registration fee {dep} taka dite hobe.",
                    f"{ecom} product review diye {pay:,} taka kamai korun! {dep} taka deposit diye task unlock korun. Aajkei apply korun.",
                ])
            elif fam == "job_package_ambassador":
                t_set.add("T1")
                txt = rng.choice([
                    f"{emp} hiring Package Handler: salary Tk {pay:,}/month. Pay Tk {dep} uniform fee to confirm.",
                    f"Brand Ambassador opening at {ecom}! Earn Tk {pay:,}/month. Order your starter kit deposit Tk {dep}.",
                ])
            elif fam == "job_double_invest":
                t_set.add("T5")
                txt = rng.choice([
                    f"Double your money in 7 days! Invest Tk {dep:,}, get Tk {dep*2:,} guaranteed. bKash {_rand_phone(rng)}",
                    f"Ludo cash game: deposit Tk {dep}, get Tk {dep*3} in 10 minutes. Guaranteed.",
                    f"৭ দিনে টাকা দ্বিগুণ! {dep:,} টাকা বিনিয়োগ করলে {dep*2:,} টাকা নিশ্চিত লাভ। বিকাশ করুন {_rand_phone(rng)}।",
                ])
            else:
                t_set.add("T1")
                txt = rng.choice([
                    f"Congratulations, you are selected for Data Entry Officer at {emp}. Pay Tk {dep:,} registration fee to confirm.",
                    f"Freelancing course: earn $500/month from home guaranteed. Enroll now with Tk {dep:,}.",
                    f"Hi, I saw your CV on Bdjobs. Work-from-home 40K/month ||| S: Deposit Tk {dep:,} as security first.",
                ])
        else:
            fam = "job_call_visa_bn"
            t_set.update({"T3", "T5"})
            txt = rng.choice([
                f"Abroad job in Malaysia, visa fee Tk {dep:,}. 100% guaranteed. Call {_rand_phone(rng)}",
                f"সরকারি/বিদেশি চাকরির সুযোগ! বেতন {pay:,} টাকা, বিস্তারিত জানতে কল করুন {_rand_phone(rng)}।",
            ])
        if rng.random() < 0.30 and "|||" not in txt:
            txt = wrap_multiturn_v4([txt], rng, allow_group=True)
        txt = _wrap_innocent(txt, emp, rng, allow_rate=0.14)
        return txt, 3, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), act, fam

    elif lbl == 4:
        pid = f"{rng.choice(['SP','CN','PF','SC','SF'])}{rng.randint(10000,99999)}"
        fee = rng.choice([45, 99, 120, 250, 1850, 2200])
        url = _rand_scam_url(rng, cour)
        t_set = set()
        act = rng.choices([2, 3, 5], weights=[0.46, 0.40, 0.14])[0]
        if act == 2:
            fam = rng.choice(["deliv_customs_duty", "deliv_storage_charge"])
            t_set.add("T3")
            if fam == "deliv_customs_duty":
                t_set.add("T1")
                txt = rng.choice([
                    f"{cour}: Parcel {pid} held at customs. Pay Tk {fee} duty/storage fee within 24h via bKash {_rand_phone(rng)} or {url}",
                    f"Customs hold: your parcel from {rng.choice(['China','UK','Malaysia','Dubai'])} needs Tk {fee:,} duty. Pay via bKash to release.",
                    f"Hello, this is {cour}. You have a package from Malaysia ||| R: I didn't order anything ||| Pay Tk {fee:,} customs today or it will be returned.",
                ])
                if "24h" in txt or "today" in txt:
                    t_set.add("T5")
            else:
                txt = rng.choice([
                    f"Parcel atke ase. Delivery fee dite link e jan. {url}",
                    f"আপনার পার্সেল আটকে আছে। ডেলিভারি চার্জ {fee} টাকা পরিশোধ করতে লিংকে ক্লিক করুন। {url}",
                    f"{cour}: Parcel {pid} couldnt be delivered. Pay Tk {fee} fee to release it: {url}",
                    f"Apnar package e customs duty baki. Customs duty Tk {fee:,} bkash korun {_rand_phone(rng)} e.",
                    f"Your package could not be delivered. Pay Tk {fee:,} clearance charge to {_rand_phone(rng)}. Hurry, limited winners only.",
                ])
                if cour in txt:
                    t_set.add("T1")
        elif act == 3:
            fam = rng.choice(["deliv_addr_url", "deliv_bn_url"])
            if fam == "deliv_bn_url":
                t_set.add("T1")
                txt = f"{cour}: ডেলিভারি ব্যর্থ হয়েছে। ঠিকানা আপডেট করতে {url} এ যান।"
            else:
                t_set.update({"T1", "T5"})
                txt = rng.choice([
                    f"Courier notice: package #{pid} requires address confirmation. Click {url} within 24h.",
                    f"{cour}: Your parcel could not be delivered. Confirm address at {url}",
                    f"Delivery attempt failed. Update address within 12 hours or parcel will be returned. {url}",
                ])
        else:
            fam = "deliv_call_phone"
            t_set.update({"T1", "T5"})
            txt = rng.choice([
                f"{cour}: Apnar parcel {pid} er address vul. Call korun {_rand_phone(rng)} ekhoni.",
                f"{cour}: আপনার পার্সেল ঠিকানা ভুল, কল করুন {_rand_phone(rng)}।",
            ])
        if rng.random() < 0.28 and "|||" not in txt:
            txt = wrap_multiturn_v4([txt], rng, allow_group=True)
        txt = _wrap_innocent(txt, cour, rng, allow_rate=0.14)
        return txt, 4, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), act, fam

    else:
        fam = rng.choice(["promo_sale", "promo_app_bogo", "promo_realestate_edu", "promo_telco"])
        brand = rng.choice(TELECOMS + ECOMMS + BANKS + MFS_BRANDS + ["Pizza Hut", "Star Tech", "Bashundhara Housing"])
        disc = rng.choice([10, 15, 20, 30, 40, 50])
        code = rng.choice(["EID10", "SAVE20", "WINTER30", "DEAL50", "WELCOME50"])
        if fam == "promo_sale":
            txt = rng.choice([
                f"{brand}: Eid offer! Sob item e {disc}% discount. Code: {code}. Shortto projojjo.",
                f"{brand} 11.11 sale starts tonight 12 AM. Vouchers up to Tk {amt:,}.",
                f"Get {disc}% cashback on your next top-up with code {code}.",
                f"Install our app and get Tk 50 off your first order. Code: WELCOME50",
                f"ঈদ উপলক্ষে সব পণ্যে {disc}% ছাড়! আজই কিনুন।",
                f"{brand}: End of season sale up to {disc}% off. Visit your nearest outlet. T&C apply.",
            ])
        elif fam == "promo_app_bogo":
            txt = rng.choice([
                f"{brand}: Buy 1 get 1 free this weekend! Visit your nearest outlet.",
                f"{brand}: 1 ta kinle 1 ta free! Code: {code}. Shortto projojjo.",
                f"{brand}: New app update with exciting features. Open the app to apply. Limited time only.",
                f"Pizza Hut combo: 2 large pizzas + drink for Tk 1,299 today only!",
            ])
        elif fam == "promo_realestate_edu":
            txt = rng.choice([
                f"Book your dream apartment in Bashundhara, easy installment. Call 09611-XXXXXX",
                f"Learn English in 30 days! Admission going on. Call now.",
            ])
        else:
            txt = rng.choice([
                f"{tel} Mega Offer! Buy 1GB get 1GB free. Dial *121*3#",
                f"{tel}: Special offer! 10GB for Tk 199, 7 days. Dial *123*199#",
            ])
        if rng.random() < 0.25 and "|||" not in txt:
            txt = wrap_multiturn_v4([txt], rng, allow_group=True)
        txt = _wrap_innocent(txt, brand, rng, allow_rate=0.14)
        return txt, 5, "NONE", 0, fam


def generate_dataset_v4(n_samples=60000, seed=4242, corrupt_prob=0.68):
    rng = random.Random(seed)
    rows = []
    for idx in range(n_samples):
        txt, lbl, tech, act, fam = gen_one_v4(rng)
        if rng.random() < corrupt_prob and fam != "legit_noise":
            txt = corrupt_text_v4(txt, rng)
        rows.append((1000000 + idx, txt, lbl, tech, act, fam))
    return pd.DataFrame(rows, columns=["id", "text", "label", "techniques", "action", "scenario_family"])


if __name__ == "__main__":
    df = generate_dataset_v4(2000, seed=4242)
    print("Label %:", df["label"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("Action %:", df["action"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("Unique families:", df["scenario_family"].nunique())
    print("Sample 5:", df[["label", "techniques", "action", "scenario_family", "text"]].head(5).to_string())
