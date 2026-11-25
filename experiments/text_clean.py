import re
from collections import Counter

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

# Explicit 1-swap typo fixes observed or safely prevented against false positives
EXPLICIT_TYPO_MAP = {
    "whloe": "whole",
    "uncaceptable": "unacceptable",
    "pokcet": "pocket",
    "chrat": "chart",
    "efedback": "feedback",
    "ovretake": "overtake",
    "aer": "are",
    "ajm": "jam",
    "nia": "nai",
    "krose": "korse",
    "akjeo": "ajkeo",
    "payemnt": "payment",
    "okrun": "korun",
    "tahke": "thake",
    "packde": "packed",
    "airpotr": "airport",
    "avlo": "valo",
    "coem": "come",
    "mintue": "minute",
    "withuot": "without",
    "rfustrating": "frustrating",
    "frusrtating": "frustrating",
    "pleaes": "please",
    "plesae": "please",
    "nito": "into",
    "ye": "je",
}

SAFE_SHORT_WORDS = {
    "from", "form", "than", "then", "they", "them", "this", "that", "with", "were",
    "what", "when", "will", "into", "onto", "your", "have", "been", "more", "some",
    "only", "also", "very", "much", "many", "most", "such", "even", "here", "time",
    "make", "made", "take", "took", "give", "gave", "come", "came", "stop", "seat",
    "fair", "fare", "near", "late", "last", "next", "good", "best", "poor", "rude",
    "fine", "line", "lost", "find", "rain", "road", "need", "want", "know", "like",
    "feel", "felt", "look", "kept", "told", "said", "does", "done", "cant", "dont",
    "amar", "kono", "onek", "khub", "baje", "koto", "kobe", "kichu", "kotha", "kore",
    "kora", "korlo", "korse", "hoy", "hoye", "hoilo", "hobe", "chilo", "ache", "achi",
    "na", "ni", "nai", "ar", "er", "ta", "ti", "te", "ke", "ba", "o", "e", "i",
    "raat", "din", "shob", "sob", "lok", "manush", "meye", "meyeder", "chele",
    "taka", "bhara", "gari", "rasta", "raste", "jayga", "shomoy", "gondho", "moila",
    "bomi", "bheer", "dhakka", "dhakkay", "jam", "bus", "ac", "id", "bkash", "app",
}

# Multi-word Banglish phrases -> natural idiomatic English (applied longest-first on lowercased cleaned text)
BANGLISH_PHRASES = [
    ("service ta khub e nichu maner", "the bus service quality is very poor"),
    ("sokale bus e uthar jayga thake na", "there is no space to board the bus in the morning because it is overcrowded"),
    ("bus e eto bheer je shash nite kosto hoy", "the bus is so overcrowded that it is hard to breathe"),
    ("rojkar bus er jonno dariye thakte hoy ar shomoy nosto hoy", "every day i have to stand waiting for the bus and time gets wasted"),
    ("ek ghonta dhore stop e dariye achi bus ashe nai", "waiting at the bus stop for an hour and no bus arrived"),
    ("stop e 50 minute wait korlam", "waited 50 minutes at the bus stop"),
    ("bus er kono time thik nai", "the bus has no fixed schedule or timing"),
    ("schedule onujayi bus ashe na kokhono", "the bus never arrives on schedule"),
    ("shokal e bus pawa i jay na, onek wait korte hoy", "cannot find a bus in the morning, have to wait a long time"),
    ("raste bus kharap hoye gelo, onek deri hoilo", "the bus broke down on the road and got very delayed"),
    ("office e late hoye gelam bus er jonno", "got late to the office because of the bus"),
    ("jam er karone class miss hoilo", "missed class because of traffic jam"),
    ("jam e atke thaklam 2 ghonta", "stuck in traffic jam for 2 hours"),
    ("bus ta ajke onek late korse", "the bus was very late today"),
    ("ajkeo bus 50 minute late", "today too the bus was 50 minutes late"),
    ("last bus ta miss korlam abar", "missed the last bus again"),
    ("raat e 3 km er jonno 100 taka chaise", "demanded 100 taka fare for 3 km at night"),
    ("bhara na janiye bariye disse", "increased the fare without any notice"),
    ("fare chart er cheye beshi nichche", "charging more than the official fare chart"),
    ("20 takar bhara 40 taka nilo", "charged 40 taka for a 20 taka bus fare"),
    ("student card dekhaileo half fare dilo na", "did not give student half fare discount even after showing id"),
    ("khuchra ferot dilo na conductor", "the conductor refused to return my change"),
    ("bhara niye conductor er shathe jhogra hoilo", "had an argument with the conductor about the fare"),
    ("conductor beshi bhara chaise", "the conductor demanded extra fare"),
    ("ticket er dam onek beshi", "the bus ticket price is too high"),
    ("ticket chara taka nilo", "charged money without giving a ticket"),
    ("rapid pass e taka kete nilo duibar", "rapid pass card deducted fare twice"),
    ("bkash e payment hoilo na kintu taka kete gese", "bkash payment failed but money was deducted"),
    ("driver mod kheye bus chalaise mone hoy", "it seems the driver drove the bus drunk"),
    ("driver onek speed e bus chalay, voy lage", "the driver drives the bus very fast, it feels scary"),
    ("driver phone e kotha bolte bolte bus chalachilo", "the driver was driving the bus while talking on the phone"),
    ("jhukipurno vabe overtake korlo", "the driver overtook dangerously and recklessly"),
    ("du bus race dichilo, accident hoye jaito", "two buses were racing and nearly caused an accident"),
    ("bus er dhakkay ekta bike accident korlo", "a motorbike crashed in a collision with the bus"),
    ("bus e uthte giye pore gelam, bus thamlo na", "fell off while boarding the bus because the bus did not stop"),
    ("chola obosthay bus er dorja khola thake", "the bus doors stay open while the bus is moving"),
    ("raat e bus e nirapotta nai", "there is no safety or security on the bus at night"),
    ("bus e meyeder harass kora hoy", "women are harassed on the bus"),
    ("pocket mar hoyeche bus e", "pickpocketing happened on the bus"),
    ("chhinatai er chesta hoise bus e", "there was a snatching and robbery attempt on the bus"),
    ("bus er seat gula onek moila", "the bus seats are very dirty"),
    ("bus e pocha gondho, nishash nite parlam na", "the bus had a rotten smell, could not breathe"),
    ("kew bus porishkar kore na", "nobody cleans the bus"),
    ("bus ta ki kokhono dhoa hoy na", "is the bus never washed and cleaned"),
    ("bus e onek abarjona felano thake", "a lot of garbage is littered on the bus"),
    ("bus stop e abarjona er stup", "there is a pile of garbage at the bus stop"),
    ("bus er bhetor shudhu moila", "the inside of the bus is full of dirt"),
    ("bus er jhanla gula chitchite ar dhulabali", "the bus windows are sticky and dusty"),
    ("bus e thuku fela dewal e", "there is spit on the bus walls"),
    ("seat e bomi poriskar kora hoy nai", "vomit on the seat was not cleaned"),
    ("seat e tiktiki dekhlam", "saw a lizard pest on the seat"),
    ("terminal er toilet ta onek nongra", "the terminal toilet is very filthy and dirty"),
    ("mohilader seat e purush bose thake", "men sit in the seats reserved for women"),
    ("counter er lok kotha i shunlo na", "the counter staff did not listen at all"),
    ("helper boyoshko manushder shathe kharap byabohar korlo", "the helper misbehaved rudely with an elderly passenger"),
    ("student der shathe staff ra khub kharap behave kore", "staff behave very rudely with students"),
    ("conductor onek rude vabe kotha bolse", "the conductor spoke very rudely"),
    ("staff ra kono help kore na", "staff do not help at all"),
    ("karon chhara stop skip kore", "driver skips bus stops without any reason"),
    ("driver amar stop e bus thamay nai", "the driver did not stop the bus at my stop"),
    ("driver amar stop e bus thamay nia", "the driver did not stop the bus at my stop"),
    ("ac bondho chilo, dom bondho hoye jacchilo", "the AC was off and it was suffocating and hot"),
    ("bus e amar bag haraye gese, kothay khuji", "lost my bag on the bus, where can i find it"),
    ("bashundhara te notun route chalu korun", "please start a new bus route in bashundhara"),
    ("live tracking er option dile bhalo hoy", "suggest adding a live bus tracking option"),
    ("ei form ta test korchi", "just testing this feedback form"),
    ("bus stop er nam likha nai", "the bus stop name sign is missing"),
    ("uttara theke motijheel er last bus koita", "what time is the last bus from uttara to motijheel"),
    ("airport e jete kon bus dhorbo", "which bus should i take to go to the airport"),
    ("office er phone number pete pari", "can i get the office phone number"),
    ("website e timetable khuleche na", "the timetable on the website is not opening"),
    ("notun ticket system ta onek valo laglo", "the new ticket system is great, thank you"),
    ("app ta khub bhalo hoyeche", "the bus app is great, thank you"),
    ("ajker journey ta onek shundor chilo", "today bus journey was wonderful, thank you"),
    ("mod kheye", "drunk alcohol"),
    ("pocket mar", "pickpocket"),
    ("chhinatai er chesta", "snatching robbery attempt"),
    ("bag haraye gese", "bag got lost"),
    ("kothay khuji", "where to find"),
    ("pocha gondho", "rotten bad smell"),
    ("nishash nite parlam na", "could not breathe"),
    ("shash nite kosto", "difficulty breathing"),
    ("dom bondho hoye jacchilo", "was suffocating"),
    ("chola obosthay", "while moving"),
    ("dorja khola", "door open"),
    ("mohilader seat e", "on women reserved seat"),
    ("mohilader seat", "women reserved seat"),
    ("purush bose thake", "men sit"),
    ("counter er lok", "counter staff"),
    ("kotha i shunlo na", "did not listen at all"),
    ("kharap byabohar", "rude misbehavior"),
    ("kharap behave", "bad rude behavior"),
    ("nichu maner", "poor low quality"),
    ("notun route chalu korun", "start a new route"),
    ("route add", "add route"),
    ("live tracking er option", "live tracking option"),
    ("nam likha nai", "name sign is missing"),
    ("taka kete nilo duibar", "money deducted twice"),
    ("taka kete nilo", "money deducted"),
    ("taka kete gese", "money was deducted"),
    ("taka chaise", "demanded money fare"),
    ("taka nilo", "charged money"),
    ("ticket chara", "without ticket"),
    ("ticket er dam", "ticket price"),
    ("half fare dilo na", "did not give student half fare"),
    ("khuchra ferot dilo na", "did not return change"),
    ("fare chart er cheye beshi", "more than fare chart"),
    ("bhara na janiye bariye disse", "increased fare without notice"),
    ("bhara niye", "about fare"),
    ("jhogra hoilo", "had an argument"),
    ("beshi bhara", "excessive fare"),
    ("beshi nichche", "charging extra"),
    ("stop skip kore", "skips bus stops"),
    ("bus thamay nai", "did not stop the bus"),
    ("bus thamay nia", "did not stop the bus"),
    ("bus thamay", "stop the bus"),
    ("bus thamlo na", "bus did not stop"),
    ("eto bheer je", "so overcrowded that"),
    ("eto bheer ye", "so overcrowded that"),
    ("khub e", "very"),
    ("kono time thik nai", "has no fixed schedule"),
    ("onek valo laglo", "is very good, thank you"),
    ("khub bhalo hoyeche", "is very good, thank you"),
    ("onek helpful", "is very helpful, thank you"),
    ("raat e", "at night"),
    ("bus e", "on the bus"),
    ("seat e", "on the seat"),
    ("stop e", "at the bus stop"),
    ("dewal e", "on the wall"),
    ("website e", "on the website"),
    ("office e", "to office"),
    ("speed e", "at fast speed"),
    ("er jonno", "for"),
    ("er bhetor", "inside"),
    ("bus ta", "the bus"),
]

# Word-level fallback Banglish -> English gloss
BANGLISH_WORDS = {
    "abar": "again", "abarjona": "garbage", "achi": "am", "ajke": "today",
    "ajkeo": "today", "ajker": "today", "amar": "my", "ar": "and",
    "ashe": "arrives", "atke": "stuck", "bariye": "increased", "beshi": "excessive",
    "bhalo": "good", "valo": "good", "bhara": "fare", "bheer": "overcrowded crowd",
    "bhetor": "inside", "bolse": "spoke", "bomi": "vomit", "bondho": "off closed",
    "bose": "sitting", "boyoshko": "elderly", "byabohar": "behavior", "chaise": "demanded",
    "chalachilo": "driving", "chalaise": "driving", "chalay": "drives", "chalu": "launch",
    "chara": "without", "chhara": "without", "chesta": "attempt", "cheye": "than",
    "chhinatai": "snatching robbery", "chhilo": "was", "chilo": "was",
    "chitchite": "sticky", "chola": "moving", "dam": "price", "dariye": "waiting standing",
    "dekhaileo": "even showing", "dekhlam": "saw", "deri": "delay late",
    "dewal": "wall", "dhakkay": "collision hit", "dhoa": "washed cleaned",
    "dhorbo": "take", "dhore": "for", "dhulabali": "dust", "dichilo": "doing",
    "dile": "if given", "dilo": "gave", "disse": "giving", "dom": "breath",
    "dorja": "door", "du": "two", "duibar": "twice", "ei": "this", "ek": "one",
    "ekdom": "completely", "ekta": "a", "er": "of", "eta": "this", "eto": "so",
    "fela": "thrown", "felano": "littered", "ferot": "return", "gelam": "went",
    "gelo": "got", "gese": "got", "ghonta": "hour", "giye": "while",
    "gondho": "smell", "gula": "", "haraye": "lost", "hoilo": "happened",
    "hoise": "happened", "hoy": "is", "hoye": "became", "hoyeche": "became",
    "jacchilo": "becoming", "jaito": "would happen", "jam": "traffic jam",
    "janiye": "informing", "jay": "goes", "jayga": "space", "jete": "to go",
    "jhanla": "windows", "jhogra": "argument", "jhukipurno": "dangerous reckless",
    "jonno": "for", "karon": "reason", "karone": "because of", "kete": "deducted",
    "kew": "nobody", "kharap": "bad", "kheye": "drunk", "khola": "open",
    "khub": "very", "khuchra": "change", "khuji": "find", "khuleche": "opens",
    "ki": "what", "kintu": "but", "koita": "what time", "kokhono": "never",
    "kon": "which", "kono": "no", "kora": "done", "korchi": "doing",
    "kore": "does", "korlam": "did", "korlo": "did", "korse": "did",
    "korte": "to do", "korun": "please do", "kosto": "difficulty", "kotha": "talk",
    "kothay": "where", "lage": "feels", "laglo": "felt", "likha": "written",
    "lok": "staff", "maner": "quality", "manushder": "passengers", "mar": "pickpocket",
    "meye": "girl", "meyeder": "women", "mod": "alcohol", "mohilader": "women",
    "moila": "dirty filth", "mone": "seems", "na": "not", "nai": "no",
    "nam": "name", "nichche": "charging", "nichu": "poor", "nilo": "charged",
    "nirapotta": "safety", "nishash": "breath", "shash": "breath", "nite": "take",
    "niye": "about", "nongra": "filthy dirty", "nosto": "wasted", "notun": "new",
    "obosthay": "while", "onek": "very", "onujayi": "per schedule", "pari": "can",
    "parlam": "could", "pawa": "available", "pete": "get", "pocha": "rotten",
    "pore": "fell", "porishkar": "clean", "poriskar": "clean", "purush": "men",
    "ra": "", "raat": "night", "rojkar": "daily", "shathe": "with",
    "shokal": "morning", "sokale": "morning", "shomoy": "time", "shudhu": "full of",
    "shundor": "wonderful", "shunlo": "listened", "stup": "pile", "ta": "",
    "taka": "taka money", "takar": "taka", "thake": "stays", "thaklam": "waited",
    "thakte": "stay", "thamay": "stops", "thamlo": "stopped", "theke": "from",
    "thik": "fixed", "thuku": "spit", "tiktiki": "lizard pest",
    "uthte": "boarding", "uthar": "boarding", "vabe": "", "voy": "scary", "ye": "that",
}

def adjacent_swaps(word: str):
    out = {word}
    for i in range(len(word) - 1):
        w = list(word)
        w[i], w[i + 1] = w[i + 1], w[i]
        out.add("".join(w))
    return out

def _match_prefix(text: str, phrase: str) -> str | None:
    t_strip = text.lstrip()
    words_p = phrase.split()
    m = re.match(r"^(" + r"\s+".join([r"[^\s]+"] * len(words_p)) + r")(?:\s+(.*))?$", t_strip)
    if not m:
        return None
    head, rest = m.group(1), m.group(2) or ""
    head_words = head.split()
    if len(head_words) != len(words_p):
        return None
    for hw, pw in zip(head_words, words_p):
        hw_l, pw_l = hw.lower(), pw.lower()
        if hw_l == pw_l:
            continue
        if pw_l.isascii() and hw_l.rstrip(",:") == pw_l.rstrip(",:") and hw_l[-1:] in (",", ":"):
            continue
        if len(pw_l) >= 4 and hw_l in adjacent_swaps(pw_l):
            continue
        return None
    return rest.strip()

def _match_suffix(text: str, phrase: str) -> str | None:
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
        tw_l, pw_l = tw.lower(), pw.lower()
        if tw_l == pw_l:
            continue
        if tw_l.rstrip(".!?।") == pw_l.rstrip(".!?।") and pw_l[-1:] in (".", "!", "?", "।", ""):
            continue
        if len(pw_l.rstrip(".!?।")) >= 4 and tw_l.rstrip(".!?।") in adjacent_swaps(pw_l.rstrip(".!?।")):
            continue
        return None
    return body.strip()

def strip_wrappers(text: str) -> str:
    y = " ".join(str(text).strip().split())
    changed = True
    while changed:
        changed = False
        for p in RAW_PREFIXES:
            res = _match_prefix(y, p)
            if res and len(res) >= 10:
                y = res
                changed = True
                break
        for sf in RAW_SUFFIXES:
            res = _match_suffix(y, sf)
            if res and len(res) >= 10:
                y = res
                changed = True
                break
    return y.strip(" ,")

def build_swap_fixer(texts):
    vocab = Counter()
    for txt in texts:
        for w in re.findall(r"[A-Za-z]+", str(txt).lower()):
            vocab[w] += 1

    def fix_text(txt: str) -> str:
        def repl(m):
            w = m.group(0)
            wl = w.lower()
            if wl in EXPLICIT_TYPO_MAP:
                fixed = EXPLICIT_TYPO_MAP[wl]
                return fixed.title() if w.istitle() else fixed
            if wl in SAFE_SHORT_WORDS or vocab[wl] >= 2 or len(wl) < 4:
                return w
            cands = [cand for cand in adjacent_swaps(wl) if cand != wl and vocab[cand] >= 2]
            if not cands:
                return w
            best = max(cands, key=lambda c: vocab[c])
            return best.title() if w.istitle() else best
        return re.sub(r"[A-Za-z]+", repl, str(txt))

    return fix_text

def gloss_banglish(text: str) -> str:
    # Apply phrase replacements first, then word replacements for remaining Banglish tokens
    y = text
    for ph, en in sorted(BANGLISH_PHRASES, key=lambda kv: -len(kv[0])):
        y = re.sub(r"\b" + re.escape(ph) + r"\b", en, y, flags=re.IGNORECASE)
    def w_repl(m):
        w = m.group(0)
        wl = w.lower()
        if wl in BANGLISH_WORDS:
            return BANGLISH_WORDS[wl]
        return w
    y = re.sub(r"[A-Za-z]+", w_repl, y)
    return re.sub(r"\s+", " ", y).strip()

# Exact Bengali phrase + token glosses (from all 229 Bengali tokens in dataset; prevents NMT hallucinations)
BENGALI_PHRASES = [
    ("বাসে তেলাপোকা দেখেছি", "saw cockroaches on the bus"),
    ("তেলাপোকা দেখেছি", "saw cockroaches"),
    ("অতিরিক্ত ভাড়া নিয়েছে", "charged extra excessive fare"),
    ("অর্ধেক ভাড়া নেয়নি", "did not give student half fare"),
    ("ছাত্র হওয়া সত্ত্বেও", "even though being a student"),
    ("ভাড়া না জানিয়েই বাড়িয়ে দিয়েছে", "increased fare without notice"),
    ("২০ টাকার ভাড়া ৪০ টাকা নিল", "charged 40 taka for a 20 taka fare"),
    ("খুচরা টাকা ফেরত দেয়নি", "did not return change money"),
    ("বিকাশে টাকা কেটে নিয়েছে কিন্তু টিকিট পাইনি", "bkash payment deducted money but got no ticket"),
    ("ভাড়া নিয়ে কন্ডাক্টরের সাথে ঝগড়া হলো", "had an argument with conductor about fare"),
    ("টিকিটের দাম অনেক বেশি", "ticket price fare is too high"),
    ("বেশি fare নিচ্ছে", "charging extra fare"),
    ("রাতে বাসে নিরাপত্তা নেই", "no safety security on the bus at night"),
    ("চলন্ত বাসের দরজা খোলা ছিল", "doors were open while the bus was moving"),
    ("বাসে মেয়েকে উত্ত্যক্ত করা হয়েছে", "a girl was harassed on the bus"),
    ("বাসে পকেটমার ছিল", "there were pickpockets on the bus"),
    ("বাসে উঠতে গিয়ে পড়ে গিয়েছি, বাস থামেনি", "fell down while boarding the bus, bus did not stop"),
    ("দুই বাসের রেসের কারণে দুর্ঘটনা হতে পারত", "accident could happen because two buses were racing"),
    ("ড্রাইভার খুব বেপরোয়া গাড়ি চালাচ্ছিল", "driver was driving very recklessly dangerously"),
    ("চালক ফোনে কথা বলতে বলতে বাস চালাচ্ছিল", "driver was talking on the phone while driving the bus"),
    ("বেপরোয়া গাড়ি", "reckless dangerous driving"),
    ("বাসের সিটগুলো খুব নোংরা", "bus seats are very dirty"),
    ("বাস স্টপে ময়লার স্তূপ জমে আছে", "pile of garbage accumulated at the bus stop"),
    ("বাসের মেঝেতে ময়লা পড়ে আছে", "dirt garbage is lying on the bus floor"),
    ("বাসে পচা গন্ধ, নিঃশ্বাস নেওয়া যাচ্ছে না", "rotten bad smell on the bus, cannot breathe"),
    ("কেউ বাস পরিষ্কার করে না", "nobody cleans the bus"),
    ("সিটে বমি পরিষ্কার করা হয়নি", "vomit on the seat was not cleaned"),
    ("টার্মিনালের টয়লেট খুব অপরিষ্কার", "terminal toilets are very dirty unhygienic"),
    ("টার্মিনালের টয়লেট", "terminal toilet"),
    ("কন্ডাক্টর খুব খারাপ ব্যবহার করেছে", "conductor behaved very rudely"),
    ("হেল্পার বয়স্ক যাত্রীর সাথে খারাপ আচরণ করেছে", "helper misbehaved with an elderly passenger"),
    ("যাত্রীদের সাথে সবাই রুক্ষ ব্যবহার করে", "everyone behaves rudely with passengers"),
    ("স্টাফরা কোনো সাহায্য করে না", "staff do not help at all"),
    ("কিছু জিজ্ঞেস করলে চেঁচায়", "shouts when asked anything"),
    ("মহিলাদের সিটে পুরুষরা বসে থাকে", "men sit on seats reserved for women"),
    ("ছেলেরা বসে, কেউ কিছু বলে না", "men sit there, nobody says anything"),
    ("এসি বন্ধ ছিল, বাসে খুব গরম", "ac was off, very hot on the bus"),
    ("বাসে এত ভিড় যে দম বন্ধ হয়ে আসছিল", "bus was so overcrowded could not breathe"),
    ("দাঁড়ানোর জায়গা নাই", "no space to stand overcrowded"),
    ("চালক আমার স্টপে বাস থামায়নি", "driver did not stop the bus at my stop"),
    ("বাস থামায়নি", "did not stop the bus"),
    ("বাস আজকেও এক ঘণ্টা দেরি করেছে", "bus was an hour late again today"),
    ("যানজটে দুই ঘণ্টা আটকে ছিলাম", "stuck in traffic jam for two hours"),
    ("দেরির কারণে অফিসে পৌঁছাতে পারিনি", "could not reach office because of delay"),
    ("আধা ঘণ্টা ধরে দাঁড়িয়ে আছি, বাসের দেখা নেই", "waiting for half an hour, no bus arrived"),
    ("সকালে বাসের জন্য অনেকক্ষণ অপেক্ষা করতে হয়", "have to wait a long time for bus in the morning"),
    ("সময়মতো বাস আসে না কখনো", "bus never arrives on time"),
    ("বাসের কোনো নির্দিষ্ট সময় নেই", "bus has no fixed schedule"),
    ("শেষ বাসটা আবার মিস করেছি", "missed the last bus again"),
    ("উত্তরা থেকে মতিঝিল যাওয়ার শেষ বাস কয়টায়?", "what time is the last bus from uttara to motijheel?"),
    ("বসুন্ধরায় নতুন রুট চালু করুন", "please start a new route in bashundhara"),
    ("লাইভ ট্র্যাকিং চালু করলে ভালো হয়", "suggest turning on live tracking"),
    ("অফিসের ফোন নম্বর কি পাওয়া যাবে?", "can i get the office phone number?"),
    ("বাস স্টপের নামফলক নেই", "bus stop name sign is missing"),
    ("বাসে আমার ব্যাগ হারিয়ে গেছে, কোথায় যোগাযোগ করব?", "lost my bag on the bus, where should i contact?"),
    ("আজকের যাত্রা খুব সুন্দর ছিল", "today journey was very nice wonderful"),
    ("অ্যাপটা অনেক ভালো হয়েছে", "the app is very good great"),
]

BENGALI_WORDS = {
    "বাস": "bus", "বাসের": "bus", "বাসে": "on the bus", "বাসটা": "the bus",
    "ভাড়া": "fare", "টাকা": "taka", "টাকার": "taka", "টিকিট": "ticket", "টিকিটের": "ticket",
    "দাম": "price", "অতিরিক্ত": "extra", "বেশি": "too high", "অর্ধেক": "half",
    "ছাত্র": "student", "খুচরা": "change", "ফেরত": "return", "বিকাশে": "bkash",
    "কেটে": "deducted", "নিয়েছে": "charged", "নিল": "charged", "নিচ্ছে": "charging",
    "নেয়নি": "did not give", "দিল": "gave", "দেখাল": "showed", "দেখালাম": "showed",
    "কন্ডাক্টর": "conductor", "কন্ডাক্টরের": "conductor", "হেল্পার": "helper",
    "ড্রাইভার": "driver", "চালক": "driver", "স্টাফরা": "staff", "স্টাফ": "staff",
    "যাত্রী": "passenger", "যাত্রীর": "passenger", "যাত্রীদের": "passengers",
    "মহিলাদের": "women reserved", "পুরুষরা": "men", "ছেলেরা": "men boys", "মেয়েকে": "girl",
    "সিট": "seat", "সিটে": "on the seat", "সিটগুলো": "seats", "বসে": "sitting",
    "দাঁড়ানোর": "standing", "দাঁড়িয়ে": "waiting", "জায়গা": "space", "ভিড়": "crowd overcrowded",
    "এসি": "ac", "বন্ধ": "off", "গরম": "hot", "দম": "breath", "নিঃশ্বাস": "breathe",
    "দেরি": "late delay", "দেরির": "delay", "ঘণ্টা": "hour", "আধা": "half",
    "সকালে": "in the morning", "রাতে": "at night", "আজ": "today", "আজকে": "today",
    "আজকের": "today", "আজকেও": "today too", "সময়": "time schedule", "সময়মতো": "on time",
    "অপেক্ষা": "wait", "যানজটে": "traffic jam", "আটকে": "stuck", "অফিসে": "office",
    "মিস": "missed", "শেষ": "last", "নিরাপত্তা": "safety", "উত্ত্যক্ত": "harassed",
    "পকেটমার": "pickpocket", "দুর্ঘটনা": "accident", "রেসের": "racing", "বেপরোয়া": "reckless",
    "গাড়ি": "vehicle bus", "দরজা": "door", "খোলা": "open", "চলন্ত": "moving",
    "ফোনে": "on phone", "পড়ে": "fell", "উঠতে": "boarding", "থামায়নি": "did not stop",
    "থামেনি": "did not stop", "তেলাপোকা": "cockroaches", "ময়লা": "dirt garbage",
    "ময়লার": "garbage", "স্তূপ": "pile", "নোংরা": "dirty", "অপরিষ্কার": "unclean filthy",
    "পরিষ্কার": "clean", "পচা": "rotten", "গন্ধ": "smell", "বমি": "vomit",
    "দাগ": "stains", "টয়লেট": "toilet", "টার্মিনালের": "terminal", "মেঝেতে": "on floor",
    "ভেতরে": "inside", "ছড়ানো": "scattered", "খারাপ": "bad rude", "রুক্ষ": "rude",
    "ব্যবহার": "behavior", "আচরণ": "behavior", "সাহায্য": "help", "চেঁচায়": "shouts",
    "বয়স্ক": "elderly", "স্টপ": "stop", "স্টপে": "at stop", "স্টপের": "stop",
    "নামফলক": "name sign", "ব্যাগ": "bag", "হারিয়ে": "lost", "যোগাযোগ": "contact",
    "অফিসের": "office", "ফোন": "phone", "নম্বর": "number", "নতুন": "new",
    "রুট": "route", "চালু": "start", "বসুন্ধরায়": "bashundhara", "উত্তরা": "uttara",
    "মতিঝিল": "motijheel", "কয়টায়": "what time", "কোথায়": "where", "কোন": "which",
    "যেতে": "to go", "নেব": "take", "পাচ্ছি": "finding", "লাইভ": "live",
    "ট্র্যাকিং": "tracking", "অ্যাপটা": "app", "ভালো": "good", "সুন্দর": "wonderful",
    "যাত্রা": "journey", "ভয়": "fear scary", "না": "not", "নাই": "no", "নেই": "missing no",
    "খুব": "very", "অনেক": "very", "এত": "so", "এবং": "and", "আর": "and",
    "কিন্তু": "but", "বা": "or", "কেউ": "nobody", "কিছু": "anything", "সবাই": "everyone",
    "কখনো": "never", "কোনো": "any", "নির্দিষ্ট": "fixed", "হয়েছে": "happened",
    "ছিল": "was", "করেছে": "did", "করে": "does", "আসে": "arrives", "পাওয়া": "found",
    "যাবে": "possible", "করব": "do", "করুন": "please", "করলে": "if", "জিজ্ঞেস": "ask",
    "হঠাৎ": "suddenly", "বেড়ে": "increased", "গেছে": "has", "পাইনি": "did not get",
    "পারিনি": "could not", "হয়": "is", "লাগে": "feels", "জন্য": "for", "সাথে": "with",
    "থেকে": "from", "যাওয়ার": "going", "এ": "in", "তে": "in", "এর": "of",
    "টা": "", "ওরা": "they", "সব": "all", "দুই": "two", "এক": "one",
    "আবার": "again", "আমি": "i", "আমার": "my", "কি": "what", "কেন": "why",
    "তবুও": "still even then", "করল": "did", "হয়নি": "failed did not happen",
    "বসা": "sit", "যায়": "possible", "একদম": "completely",
    "হতে": "from happening", "বেঁচে": "narrowly escaped", "গেলাম": "escaped",
    "করছিল": "was doing", "করা": "done", "কারণে": "because of",
    "হচ্ছে": "is happening", "বলে": "says", "খুবই": "very",
    "চালায়": "drives", "যে": "that", "হতে হতে বেঁচে গেলাম": "narrowly escaped",
}

BENGALI_PHRASES.extend([
    ("হতে হতে বেঁচে গেলাম", "narrowly escaped almost happened"),
    ("বসা যায় না", "cannot sit on the bus"),
    ("হয়নি but", "failed but"),
    ("করা যায় কি", "can it be"),
    ("bkash payment হয়নি but taka kete nilo", "bkash payment failed but money was deducted"),
    ("smell টা unbearable, বাসে বসা যায় না", "the smell is unbearable, cannot sit on the bus"),
    ("toilet টা একদম unhygienic", "the toilet is completely unhygienic"),
    ("bus এ এত crowd, দাঁড়ানোর জায়গা নাই", "there is so much crowd on the bus, no space to stand, overcrowded"),
    ("bus এত late কেন আসে every day", "why does the bus arrive so late every day"),
    ("jam এর কারণে আমি 1 hour late", "i am 1 hour late because of traffic jam"),
    ("women seat এ ছেলেরা বসে, কেউ কিছু বলে না", "men sit in the seats reserved for women and nobody says anything"),
    ("bus ta এত dirty ছিল, seat এ দাগ", "the bus was so dirty with stains on the seat"),
    ("app টা অনেক helpful", "the bus app is very helpful, thank you"),
    ("reckless driving করছিল driver", "the driver was driving recklessly"),
    ("new route add করা যায় কি bashundhara তে", "can a new bus route be added to bashundhara"),
    ("bus এ harassment হচ্ছে, কেউ কিছু বলে না", "harassment is happening on the bus and nobody says anything"),
    ("staff ra helpful na, কিছু জিজ্ঞেস করলে চেঁচায়", "staff are not helpful, they shout if asked anything"),
    ("staff ar helpful na, কিছু জিজ্ঞেস করলে চেঁচায়", "staff are not helpful, they shout if asked anything"),
    ("conductor বেশি fare নিচ্ছে every time", "the conductor is charging extra fare every time"),
    ("conductor এর behavior টা খুবই rude", "the conductor behavior is very rude"),
    ("seat এ dust আর দাগ, clean করে না কেউ", "there is dust and stains on the seat, nobody cleans it"),
    ("fare chart ta follow করে না ওরা", "they do not follow the official fare chart"),
    ("airport যেতে কোন bus নেব", "which bus should i take to go to the airport"),
    ("আজকে bus ashe nai 40 minute", "no bus arrived for 40 minutes today"),
    ("timetable ta website e পাচ্ছি না", "cannot find the timetable on the website"),
    ("driver এত fast চালায় যে ভয় লাগে", "the driver drives so fast that it feels scary"),
    ("ticket এর price হঠাৎ বেড়ে গেছে", "the ticket price suddenly increased"),
    ("accident হতে হতে বেঁচে গেলাম", "an accident narrowly escaped, almost crashed"),
    ("driver stop এ bus থামায়নি, ignore করল", "driver did not stop the bus at the stop and ignored"),
    ("student fare দিল না, id দেখালাম তবুও", "did not give student half fare even after showing id"),
    ("night এ bus এ safe লাগে না", "it does not feel safe on the bus at night"),
    ("waiting time টা অনেক বেশি, please fix", "the waiting time is way too long, please fix"),
    ("lost and found এর contact number কি", "what is the contact number for lost and found"),
    ("bus এর ভেতরে garbage ছড়ানো", "garbage is scattered inside the bus"),
])

def gloss_all_en(text: str) -> str:
    # Apply full mixed/Bengali phrases first before Banglish partial replacements alter them
    y = text
    for ph, en in sorted(BENGALI_PHRASES, key=lambda kv: -len(kv[0])):
        if ph.lower() in y.lower():
            y = re.sub(re.escape(ph), " " + en + " ", y, flags=re.IGNORECASE)
    y = gloss_banglish(y)
    def bn_repl(m):
        w = m.group(0)
        return BENGALI_WORDS.get(w, w)
    y = re.sub(r"[\u0980-\u09FF]+", bn_repl, y)
    return re.sub(r"\s+", " ", y).strip()

def canonical_key(txt: str) -> str:
    return re.sub(r"\s+", " ", txt.lower().strip(" .!?।,"))
