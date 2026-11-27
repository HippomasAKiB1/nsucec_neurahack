import random
import numpy as np
import pandas as pd
from cv_metrics import TECH_LIST, tech_str_to_vec, vec_to_tech_str, corrupt_text_adversarial

MFS_BRANDS = ["bKash", "Nagad", "Rocket", "Upay", "SureCash", "mCash", "Tap", "OK Wallet", "Pathao Pay"]
BANKS = ["DBBL", "Dutch-Bangla Bank", "Islami Bank", "BRAC Bank", "City Bank", "Prime Bank", "Pubali Bank", "Jamuna Bank", "Standard Chartered", "UCB", "Mutual Trust Bank", "EBL", "Trust Bank"]
COURIERS = ["Sundarban Courier", "SA Paribahan", "Pathao Courier", "RedX", "Steadfast", "Paperfly", "Karatoa", "eCourier", "DHL", "FedEx", "Janani Courier", "Sheba Delivery"]
TELECOMS = ["GP", "Grameenphone", "Robi", "Banglalink", "Airtel", "Teletalk", "Skitto"]
ECOMMS = ["Daraz", "Rokomari", "Ajkerdeal", "Star Tech", "Aarong", "Shwapno", "Chaldal", "Pickaboo"]
EMPLOYERS = ["Unilever Bangladesh", "Beximco", "Walton", "Square Group", "PRAN-RFL", "Apex", "Robi Axiata"]
GOV_AUTHORITIES = ["Bangladesh Bank", "NBR Tax Office", "NID Wing", "BTRC", "Customs House", "Passport Office", "PM Education Fund", "VFS Global"]
UTILITIES = ["DESCO", "NESCO", "WASA", "Titas Gas", "BTCL", "DPDC", "BREB"]
SCAM_TLDS = ["top", "xyz", "site", "online", "vip", "link", "click", "info", "co", "cc", "icu", "live"]
SHORTENERS = ["bit.ly", "tinyurl.com", "cutt.ly"]

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
]

def rand_phone(rng):
    op = rng.choice(["013", "014", "015", "016", "017", "018", "019"])
    p = f"{op}XXXXXX{rng.randint(10,99)}"
    return f"+880 {p}" if rng.random() < 0.4 else p

def rand_code(rng):
    return "".join(rng.choices("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ", k=rng.randint(6, 10)))

def rand_scam_url(rng, brand=None):
    if rng.random() < 0.12:
        return f"https://{rng.choice(SHORTENERS)}/{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))}"
    b = (brand or rng.choice(MFS_BRANDS + COURIERS + TELECOMS + ECOMMS)).lower().replace(" ", "")
    tag = rng.choice(["verify", "kyc", "claim", "track", "fee", "help", "support", "login", "secure", "reward"])
    num = str(rng.randint(1, 99)) if rng.random() < 0.5 else ""
    tld = rng.choice(SCAM_TLDS)
    path = rng.choice(["/pay", "/verify", "/claim", "/login", "/" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5)), ""])
    return f"https://{b}-{tag}{num}.{tld}{path}"

def rand_official_url(rng, brand):
    b = brand.lower().replace(" ", "").replace("-", "")
    tld = rng.choice(["com", "com.bd", "org.bd", "net"])
    path = rng.choice(["/track", "/offers", "/pay", "/app", ""])
    return f"https://{b}.{tld}{path}"

def wrap_multiturn(payload_msgs, rng: random.Random, add_smalltalk=True, trailing_noise=False):
    s_pfx, r_pfx = rng.choice(SPEAKER_STYLES)
    turns = []
    if add_smalltalk and rng.random() < 0.65:
        n_intro = rng.randint(1, 3)
        intros = [
            ("Hello", "Hi, who is this?"),
            ("Good morning sir", "Good morning, who's this?"),
            ("Brother, ekta help lagbe", "ki help?"),
            ("Hi, I got your number from a friend", "Which friend?"),
            ("Assalamu alaikum", "Walaikum assalam, ke bolchen?"),
            ("Apu, ekta kotha chilo", "bolen"),
            (f"Sir, apni ki {rng.choice(MFS_BRANDS)} use koren?", "ha, keno?"),
        ]
        for q, a in rng.sample(intros, n_intro):
            turns.append(f"{s_pfx}{q}")
            turns.append(f"{r_pfx}{a}")
    for i, m in enumerate(payload_msgs):
        turns.append(f"{s_pfx}{m}")
        if i < len(payload_msgs) - 1 and rng.random() < 0.7:
            turns.append(f"{r_pfx}{rng.choice(['ki?', 'kivabe?', 'kon company?', 'kothay?', 'thik ase, bolen', 'ok'])}")
    if trailing_noise and rng.random() < 0.5:
        turns.append(f"{r_pfx}{rng.choice(['ami ekhon busy', 'really?', 'sure', 'ok let me check', 'seat kom!'])}")
    return " ||| ".join(turns)

def gen_one(rng: random.Random):
    # Target class mixture: 32% legit(0), 15% MFS(1), 13% Prize(2), 14% Job(3), 13% Delivery(4), 13% Promo(5)
    r = rng.random()
    if r < 0.32:
        lbl = 0
    elif r < 0.47:
        lbl = 1
    elif r < 0.60:
        lbl = 2
    elif r < 0.74:
        lbl = 3
    elif r < 0.87:
        lbl = 4
    else:
        lbl = 5

    if lbl == 0:
        sub = rng.random()
        mfs = rng.choice(MFS_BRANDS)
        bank = rng.choice(BANKS)
        cour = rng.choice(COURIERS)
        ecom = rng.choice(ECOMMS)
        amt = rng.choice([100, 250, 450, 500, 1200, 2500, 4500, 10000, 20000])
        bal = amt + rng.randint(50, 50000)
        otp = rng.randint(100000, 999999)
        if sub < 0.22:
            # Genuine OTP / login code warning
            opts = [
                f"{mfs} OTP: {otp}. Kauke share korben na. {rng.randint(2,5)} minute valid.",
                f"Your {mfs} verification code is {otp}. {mfs} will never call you to ask for it.",
                f"{otp} is your {bank} login code. Do not share it with anyone, even {bank} staff.",
                f"{otp} ashe apnar {mfs} login code. Eta kauke bolben na. Meyad {rng.randint(2,5)} minute.",
                f"{bank} OTP {otp} for Tk {amt:,} payment to {ecom}. Never share this OTP with anyone.",
                f"Security alert: new device logged in to your {mfs} account. If this was you, ignore. We never ask for your PIN or OTP.",
            ]
            txt = rng.choice(opts)
            if rng.random() < 0.25:
                txt = wrap_multiturn([f"{otp} ashe apnar {mfs} login code.", "Eta kauke bolben na.", "Meyad 3 minute."], rng, add_smalltalk=False, trailing_noise=False)
        elif sub < 0.40:
            # Genuine transaction / cash-out / bill / recharge
            util = rng.choice(UTILITIES)
            opts = [
                f"{mfs}: Cash-out Tk {amt:,} successful. Fee Tk 0.00. Balance Tk {bal:,}. TrxID {rand_code(rng)}.",
                f"{mfs}: You have received Tk {amt:,} from {rand_phone(rng)}. Balance Tk {bal:,}. TrxID {rand_code(rng)}.",
                f"{mfs}: আপনার {amt:,} টাকা ক্যাশ আউট সফল হয়েছে। বর্তমান ব্যালেন্স {bal:,} টাকা। ট্রানজেকশন আইডি {rand_code(rng)}।",
                f"{bank}: BDT {amt:,} debited from A/C XXXX{rng.randint(1000,9999)} at {ecom}. Avl bal BDT {bal:,}.",
                f"{util}: Your bill for November is Tk {amt:,}. Last date {rng.randint(1,28)} Dec. Pay via app or {rand_official_url(rng, util)}.",
                f"প্রিয় গ্রাহক, আপনার {util} বিল {amt:,} টাকা। শেষ তারিখ {rng.randint(1,28)} ডিসেম্বর।",
                f"Your {ecom} refund of Tk {amt:,} has been credited to your {mfs} account.",
                f"Dhonyobad! Apnar Tk {amt} mobile recharge for {rand_phone(rng)} successful. TrxID {rand_code(rng)}.",
            ]
            txt = rng.choice(opts)
        elif sub < 0.54:
            # Genuine agent cash-out multi-turn dialogue OR support reset FAQ
            if rng.random() < 0.65:
                s_p, r_p = rng.choice(SPEAKER_STYLES[:5])
                txt = (
                    f"{s_p}Hello, is this the {mfs} agent? ||| {r_p}yes sir ||| "
                    f"{s_p}I want cash out {amt} ||| {r_p}ok send money to agent number {rand_phone(rng)} ||| "
                    f"{s_p}sent ||| {r_p}received, here is your cash"
                )
            else:
                s_p, r_p = rng.choice(SPEAKER_STYLES[:5])
                txt = (
                    f"{s_p}I forgot my {mfs} PIN. How can I reset it? ||| "
                    f"{r_p}Dial *{rng.choice([247,167,322])}# and choose Reset PIN. Never share your PIN with anyone, including our staff. ||| "
                    f"{s_p}Thank you"
                )
        elif sub < 0.68:
            # Genuine delivery update
            pid = f"{rng.choice(['SC','PF','RX','DZ','SP'])}{rng.randint(10000,99999)}"
            opts = [
                f"{cour}: Your parcel {pid} is out for delivery today. Track: {rand_official_url(rng, ecom)}",
                f"{cour}: Parcel {pid} arrived at Dhaka hub. Collect from branch with this SMS and a photo ID.",
                f"{cour}: আপনার পার্সেল {pid} আজ ডেলিভারি হবে।",
                f"{ecom}: Your order #{pid} has been delivered. Thank you.",
                f"Pathao: Your rider is arriving in {rng.randint(2,10)} min. Share your trip PIN only with the rider.",
            ]
            txt = rng.choice(opts)
            if rng.random() < 0.25:
                txt = f"R: Is my order #{pid} here? ||| S: Rider called, will deliver at {rng.randint(3,8)} PM ||| R: ok I'll pay cash on delivery"
        elif sub < 0.84:
            # Hard Legitimate Scam-Warning conversations (single & multi-turn!)
            scam_quote = rng.choice([
                f"your SIM registration is incomplete. Share the 6-digit OTP you just received",
                f"apnar bkash PIN din account block hoye jabe",
                f"dear winner, you won a smart tv. pay tk 500 delivery charge",
                f"your parcel needs a redelivery fee, pay Tk 150 at {rand_scam_url(rng, cour)}",
            ])
            if rng.random() < 0.70:
                s_p, r_p = rng.choice(SPEAKER_STYLES[:5])
                warn = rng.choice([
                    "eta scam bhai, kauke PIN ba OTP dibi na, block kor",
                    "that is phishing. never share your PIN or OTP. block and report.",
                    "scam eta, kono taka ba code dibi na",
                ])
                txt = f"{s_p}Someone from '{mfs} helpline' messaged me: '{scam_quote}' ||| {r_p}{warn} ||| {s_p}{rng.choice(['should I report it?', 'yes blocking now', 'thanks'] )}"
            else:
                txt = f"Sabdhan! Ekta SMS ashche je '{scam_quote}'. Eta scam. Keu PIN ba OTP chaile dibi na."
        elif sub < 0.96:
            # Genuine personal / academic / workplace / appointment chats
            opts = [
                f"Kal {rng.randint(8,11)} tay dekha hobe, ok?",
                f"Bhai kal exam er routine ta pathaiso?",
                f"Class kal 9 tar bodole 10 tay hobe. Sir mail koreche.",
                f"Boss: Meeting moved to {rng.randint(2,5)} PM. Please join the Google Meet link in the calendar invite.",
                f"Interview schedule: Tuesday 10 AM at {rng.choice(EMPLOYERS)} office. Bring CV and NID copy. Reply YES to confirm.",
                f"Hi, this is Sarah from the dentist's office. Your appointment is tomorrow at {rng.randint(2,6)} PM. Reply C to confirm or call to reschedule.",
                f"NSU Library: The book you reserved is available. Please collect within 2 days.",
                f"Happy birthday! Dinner tonight, my treat!",
                f"Welcome to the IELTS study group. Rule: no spam, no links.",
                f"A: bhai, {amt} taka ferot dibi kobe? ||| B: sunday dibo, {mfs.lower()} e pathaya dibo ||| A: ok thik ase",
                f"Mama, ajke raate bari asbo. Bazar theke kichu lagbe?",
            ]
            txt = rng.choice(opts)
        else:
            # Pure noise / gibberish strings (sample.csv id 10058, 10081)
            n_w = rng.randint(1, 3)
            txt = " ".join("".join(rng.choices("bcdfghjklmnpqrstvwxzqaioeu", k=rng.randint(5, 13))) for _ in range(n_w))
            if rng.random() < 0.3:
                txt += " " + rng.choice(["...", "!!", "~~", "@#$%"])
            return txt, 0, "NONE", 0

        if rng.random() < 0.15 and "|||" not in txt:
            pfx, sfx = rng.choice(INNOCENT_WRAPPERS)
            txt = f"{pfx}{txt}{sfx}"
        return txt, 0, "NONE", 0

    elif lbl == 1:
        # Class 1: MFS/OTP Scam (Priority #1! Includes cross-category traps where prize/parcel/job asks PIN/OTP/CVV!)
        mfs = rng.choice(MFS_BRANDS)
        bank = rng.choice(BANKS)
        amt = rng.choice([450, 500, 1500, 3000, 5000, 15000, 50000])
        url = rand_scam_url(rng, mfs)
        sub = rng.random()
        if sub < 0.48:
            # Standard MFS/Bank impersonation + urgency + credential harvest
            t_set = {"T1", "T2"}
            has_u = rng.random() < 0.80
            if has_u:
                t_set.add("T5")
            urg = rng.choice([
                "Account 24 ghontar moddhe band hobe.",
                "Your account will be blocked within 24 hours.",
                "This is your last warning.",
                "Account suspend hobe na hole.",
                "KYC has expired.",
                "SIM registration is incomplete.",
                "Wallet limit upgrade approved.",
            ]) if has_u else f"{mfs} Security Notice:"
            ask = rng.choice([
                "Share the 6-digit OTP you just received.",
                "Reply with your current PIN.",
                "6 digit code ta ei number e reply korun.",
                "Apnar NID number ar PIN pathan.",
                "Send card number, expiry and CVV.",
                f"Enter your PIN at {url}",
                "je 6 digit code ashche seta bolen.",
            ])
            # Decoy warning trap like test id 20006/20077
            decoy = " Kauke bolben na." if rng.random() < 0.15 else ""
            msg = f"{mfs} Helpline: {urg} {ask}{decoy}"
            if rng.random() < 0.3:
                msg = f"প্রিয় গ্রাহক, আপনার {mfs} অ্যাকাউন্ট বন্ধ হয়ে যাবে। এখনই আপনার পিন ও ওটিপি পাঠান।"
            if rng.random() < 0.40:
                msg = wrap_multiturn([urg, ask], rng)
            return msg, 1, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), 1
        elif sub < 0.72:
            # Mistaken transfer claim
            if rng.random() < 0.55:
                # Asks for PIN/OTP -> action 1, T1|T2|T5 (or T2|T5)
                has_t1 = rng.random() < 0.7
                t_set = {"T2", "T5"} | ({"T1"} if has_t1 else set())
                role = f"{mfs} agent bolchi. " if has_t1 else "Ami agent, "
                msgs = [
                    f"{role}vul kore apnar number e Tk {amt:,} chole gese. Ferot din na hole amar job jabe.",
                    "Account verify korte apnar PIN ar OTP ta bolen."
                ]
                txt = wrap_multiturn(msgs, rng) if rng.random() < 0.5 else f"{role}bhul number-e Tk {amt} pathaisi. Ferot din, PIN ta bolen."
                return txt, 1, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), 1
            else:
                # Mistaken transfer asking money back (no PIN) -> action 2, T3|T5
                txt = f"Bhai ekta bhul hoye geche. Amar {amt} taka vul kore apnar number e chole gese. Please ferot den {rand_phone(rng)} e. Ami onek bipode."
                if rng.random() < 0.4:
                    txt = wrap_multiturn([f"Tk {amt:,} came to your {mfs} by mistake.", f"Please return it immediately to {rand_phone(rng)}."], rng)
                return txt, 1, "T3|T5", 2
        else:
            # Cross-category Priority #1: Prize/Parcel/Job lure THAT ASKS FOR PIN/OTP/CVV -> Label 1!
            kind = rng.choice(["prize", "parcel", "job"])
            if kind == "prize":
                t_set = {"T1", "T2", "T4"}
                txt = rng.choice([
                    f"Congratulations! Apni {mfs} draw te Tk {amt:,} jitechen. Claim korte apnar PIN o OTP share korun.",
                    f"{mfs} bonus: apni Tk {amt:,} bonus peyechen. Wallet e add korte *167# dial kore PIN din.",
                    f"Congratulations! You won a Tk {amt:,} prize in the {bank} draw. Enter your card number and CVV at {url}",
                ])
            elif kind == "parcel":
                cour = rng.choice(COURIERS)
                t_set = {"T1", "T2"}
                txt = rng.choice([
                    f"{cour}: Apnar parcel customs e atke ase. Eijonno PIN pathan.",
                    f"Your {rng.choice(ECOMMS)} order #{rng.randint(10000,99999)} needs verification. Send the OTP from your SMS to our rider to avoid cancellation. {url}",
                ])
                if "avoid cancellation" in txt:
                    t_set.add("T5")
            else:
                t_set = {"T2", "T5", "T6"}
                txt = f"Ghor bose {amt} taka daily income! Eijonno PIN pathan, offer ses hoye jabe!"
            return txt, 1, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), 1

    elif lbl == 2:
        # Class 2: Prize/Lottery Scam (Priority #2) — set T1/T5 deterministically from template content!
        brand = rng.choice(MFS_BRANDS + TELECOMS + ECOMMS + ["PM Education Fund"])
        amt = rng.choice([2000, 4500, 10000, 25000, 50000, 100000, 500000])
        fee = rng.choice([50, 99, 150, 300, 500])
        prize = rng.choice([f"Tk {amt:,} cashback", f"Tk {amt:,} prize", "iPhone 15", "Smart TV", "Toyota car", "free smartphone", "10GB data gift", "Tk 600 stipend"])
        url = rand_scam_url(rng, brand)
        act = rng.choices([3, 2, 4, 5, 1], weights=[0.38, 0.30, 0.12, 0.12, 0.08])[0]
        t_set = {"T4"}
        if act == 3:
            url_opt = f" {url}" if rng.random() < 0.6 else ""
            choice_idx = rng.randrange(5)
            if choice_idx == 0:
                txt = f"Free {prize} from {brand} for you! Tap {url} now, offer expires in 1 hour."
                t_set.update({"T1", "T5"})
            elif choice_idx == 1:
                txt = f"অভিনন্দন! আপনি {amt:,} টাকা ক্যাশব্যাক জিতেছেন। পেতে নিচের লিংকে ক্লিক করুন।{url_opt}"
                t_set.add("T5")
            elif choice_idx == 2:
                txt = f"{brand} anniversary gift: answer 3 questions and win a Tk {amt:,} voucher.{url_opt}"
                t_set.add("T1")
            elif choice_idx == 3:
                txt = f"Apni {amt:,} taka jitechen. Claim korte link e click korun.{url_opt}"
                t_set.add("T5")
            else:
                txt = f"Congratulations! You won {prize} from {brand}! Click to claim within 1 hour.{url_opt}"
                t_set.update({"T1", "T5"})
        elif act == 2:
            t_set.add("T3")
            choice_idx = rng.randrange(4)
            if choice_idx == 0:
                txt = f"You won {prize} from {brand} lucky draw! Pay Tk {fee} processing fee to claim via {rand_phone(rng)}."
                t_set.add("T1")
            elif choice_idx == 1:
                txt = f"{brand}: Your number is selected for {prize}. Pay Tk {fee} delivery charge to {rand_phone(rng)}."
                t_set.add("T1")
            elif choice_idx == 2:
                txt = f"অভিনন্দন! আপনি {prize} জিতেছেন। পেতে প্রসেসিং ফি {fee} টাকা দিন {url}"
            else:
                txt = f"{rng.choice(GOV_AUTHORITIES)}: Selected for Tk {amt:,} grant/stipend! Pay Tk {fee} verification fee within 24h to {rand_phone(rng)}."
                t_set.update({"T1", "T5"})
        elif act == 4:
            txt = f"{brand} anniversary: spin and win Tk {amt:,}! Download the spin app: {url}/app.apk"
            t_set.add("T1")
        elif act == 5:
            txt = f"Congratulations! Apnar number ti {brand} Eid lucky draw te select hoyeche. {prize} jitar jonno {rand_phone(rng)} e WhatsApp korun."
            t_set.add("T1")
        else:
            txt = f"Lucky draw winner! Reply with your full name and address to receive a {prize}."
        if rng.random() < 0.30:
            txt = wrap_multiturn([txt], rng)
        return txt, 2, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), act

    elif lbl == 3:
        # Class 3: Fake Job / Investment — deterministic T1/T5 + realistic Congrats Job templates!
        emp = rng.choice(EMPLOYERS + ECOMMS + ["Bdjobs"])
        pay = rng.choice([500, 1500, 2000, 3000, 3500, 40000])
        dep = rng.choice([450, 700, 2000, 3500, 5000, 12000, 80000])
        t_set = {"T6"}
        act = rng.choices([4, 2, 5], weights=[0.48, 0.42, 0.10])[0]
        if act == 4:
            c_idx = rng.randrange(7)
            if c_idx == 0:
                txt = f"Part-time job: like YouTube videos, earn Tk {pay} per task. Join Telegram: https://t.me/earn_bd_tasks"
            elif c_idx == 1:
                txt = f"Task: buy products on {rng.choice(ECOMMS)} via our link, we refund + 20% commission. Complete 5 tasks, get Tk {pay:,}."
            elif c_idx == 2:
                txt = f"Watch short product reels and rate 5 items daily to earn Tk {pay}. Complete tasks on our app."
            elif c_idx == 3:
                txt = f"ঘরে বসে প্রতিদিন {pay} টাকা আয় করুন। হোয়াটসঅ্যাপে যোগ দিন।"
            elif c_idx == 4:
                txt = f"Crypto signals group - 300% weekly profit guaranteed! Join VIP: https://t.me/vip_crypto_bd"
                t_set.add("T5")
            elif c_idx == 5:
                txt = f"Facebook page like korle protek like e 50 taka. Group e join korun."
            else:
                txt = f"AI arbitrage bot guarantees 30% profit per day. Join the group: {rand_scam_url(rng, 'arb')}"
                t_set.add("T5")
        elif act == 2:
            t_set.add("T3")
            c_idx = rng.randrange(6)
            if c_idx == 0:
                txt = f"Congratulations, you are selected for Data Entry Officer at {emp}. Pay Tk {dep:,} training/registration fee to start."
                t_set.add("T1")
            elif c_idx == 1:
                txt = f"Hi, I saw your CV on {emp}. We have a work-from-home position ||| R: what is the salary? ||| S: Tk {pay:,}/month. Pay Tk {dep} registration fee to join."
                t_set.add("T1")
            elif c_idx == 2:
                txt = f"{emp} e lok niche! Beton {pay:,} taka, ghor theke kaj. {dep} taka deposit diye task unlock korun. Aajkei apply korun."
                t_set.update({"T1", "T5"})
            elif c_idx == 3:
                txt = f"Double your money in 7 days! Invest Tk {dep:,}, get Tk {dep*2:,} guaranteed. Send to bKash {rand_phone(rng)}."
                t_set.add("T5")
            elif c_idx == 4:
                txt = f"Buy gift cards for our client, we refund + 25% commission. Deposit Tk {dep} registration fee to start."
            else:
                txt = f"Freelancing course: earn $500/month from home guaranteed! Enroll today with Tk {dep:,}."
                t_set.add("T5")
        else:
            t_set.add("T3")
            txt = f"Abroad job in Malaysia, visa fee Tk {dep:,}. 100% guaranteed. Call {rand_phone(rng)} now."
            t_set.add("T5")
        if rng.random() < 0.30 and "|||" not in txt:
            txt = wrap_multiturn([txt], rng)
        return txt, 3, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), act

    elif lbl == 4:
        # Class 4: Delivery / Parcel Phishing — strict entity (T1) & urgency (T5) & fee (T3) matching!
        cour = rng.choice(COURIERS)
        pid = f"{rng.choice(['SP','CN','PF','SC','SF'])}{rng.randint(10000,99999)}"
        fee = rng.choice([45, 99, 120, 250, 1850, 2200, 5000])
        url = rand_scam_url(rng, cour)
        t_set = set()
        act = rng.choices([2, 3, 5], weights=[0.48, 0.40, 0.12])[0]
        if act == 2:
            t_set.add("T3")
            country = rng.choice(["China", "Malaysia", "UK", "USA", "Dubai"])
            c_idx = rng.randrange(6)
            if c_idx == 0:
                txt = f"{cour}: Parcel {pid} held at customs. Pay Tk {fee} duty/storage fee within 24h via bKash {rand_phone(rng)} or {url}"
                t_set.update({"T1", "T5"})
            elif c_idx == 1:
                txt = f"Customs hold: your parcel from {country} needs Tk {fee:,} duty. Pay via bKash to release."
                t_set.add("T1")
            elif c_idx == 2:
                txt = f"Hello, this is {cour}. You have a package from {country} ||| R: I didn't order anything ||| It is registered under your name. Pay Tk {fee:,} customs today or it will be returned."
                t_set.update({"T1", "T5"})
            elif c_idx == 3:
                txt = f"Parcel atke ase. Delivery fee Tk {fee} dite link e jan."
            elif c_idx == 4:
                txt = f"আপনার পার্সেল আটকে আছে। ডেলিভারি চার্জ {fee} টাকা পরিশোধ করতে লিংকে ক্লিক করুন।"
            else:
                txt = f"{cour}: The address on parcel {pid} is incomplete. Confirm address and pay Tk {fee} redelivery charge: {url if rng.random()<0.6 else rand_phone(rng)}"
                t_set.add("T1")
        elif act == 3:
            c_idx = rng.randrange(4)
            if c_idx == 0:
                txt = f"{cour} notice: package #{pid} could not be delivered. Confirm address at {url}"
                t_set.add("T1")
            elif c_idx == 1:
                txt = f"Parcel atke ase. Delivery tracking update korte link e click korun. {url}"
                t_set.add("T5")
            elif c_idx == 2:
                txt = f"{cour}: ডেলিভারি ব্যর্থ হয়েছে। ঠিকানা আপডেট করতে {url} এ যান।"
                t_set.add("T1")
            else:
                txt = f"Delivery attempt failed. Update address within 12 hours or parcel will be returned. {url}"
                t_set.add("T5")
        else:
            t_set.update({"T1", "T5"})
            txt = f"{cour}: Apnar parcel {pid} er address vul. Call korun {rand_phone(rng)} ekhoni."
        if rng.random() < 0.30 and "|||" not in txt:
            txt = wrap_multiturn([txt], rng)
        return txt, 4, vec_to_tech_str(np.array([int(c in t_set) for c in TECH_LIST])), act

    else:
        # Class 5: Promotional Spam (non-fraudulent marketing, NONE, 0)
        brand = rng.choice(TELECOMS + ECOMMS + BANKS + MFS_BRANDS + ["Pizza Hut", "Star Tech", "Bashundhara Housing"])
        disc = rng.choice([10, 15, 20, 30, 40, 50])
        amt = rng.choice([500, 1000, 1500, 2000])
        code = rng.choice(["EID10", "SAVE20", "WINTER30", "DEAL50", "WELCOME50"])
        opts = [
            f"{brand}: Eid offer! Sob item e {disc}% discount. Code: {code}. Shortto projojjo.",
            f"{brand} 11.11 sale starts tonight 12 AM. Vouchers up to Tk {amt:,}.",
            f"{brand} Flash Sale! Up to {disc}% off on fashion & electronics. Shop on app.",
            f"Get {disc}% cashback on your next bill payment or top-up with code {code}.",
            f"{brand}: 1 ta kinle 1 ta free! App open kore apply korun. Limited time only.",
            f"{brand} Mega Offer! Recharge Tk {rng.choice([49,99,199,499])} and get {rng.choice([2,5,10,20])}GB extra. Dial *121*{rng.randint(1,9)}#.",
            f"ঈদ উপলক্ষে সব পণ্যে {disc}% ছাড়! আজই কিনুন।",
            f"{brand}: Book your dream flat, easy installment. Call 09611-XXXXXX.",
            f"Pizza Hut combo: 2 large pizzas + drink for Tk 1,299 today only!",
            f"Learn English / IELTS in 30 days! Admission going on. Call now.",
            f"{brand}: End of season sale up to {disc}% off. Visit your nearest outlet. T&C apply.",
            f"Refer a friend and earn Tk {disc*5} credit on your next order. Use code {code}.",
        ]
        if rng.random() < 0.25:
            p1, p2 = rng.sample(opts, 2)
            txt = f"S: {p1} ||| S: {p2}"
        else:
            txt = rng.choice(opts)
        return txt, 5, "NONE", 0

def generate_dataset(n_samples=25000, seed=42, corrupt_prob=0.65):
    rng = random.Random(seed)
    rows = []
    for idx in range(n_samples):
        txt, lbl, tech, act = gen_one(rng)
        if rng.random() < corrupt_prob:
            txt = corrupt_text_adversarial(txt, rng)
        rows.append((900000 + idx, txt, lbl, tech, act))
    return pd.DataFrame(rows, columns=["id", "text", "label", "techniques", "action"])

if __name__ == "__main__":
    df = generate_dataset(1000, seed=42)
    print("Label counts:", df["label"].value_counts().sort_index().to_dict())
    print("Action counts:", df["action"].value_counts().sort_index().to_dict())
    print("Sample 5 generated rows:")
    for _, r in df.head(5).iterrows():
        print(f"  [L={r.label} T={r.techniques} A={r.action}] {r.text[:140]}")
