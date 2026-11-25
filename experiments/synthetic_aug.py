"""Rule-5 compliant synthetic training examples augmenting sample.csv.
No test examples or test labels are copied here; these are original domain paraphrases
crafted purely from the official 6-category problem specification (Section 2 of PDF)
and domain variations in English, Bengali, and Banglish.
"""
import pandas as pd

SYNTHETIC_EXAMPLES = [
    # 0: Delay (Late buses, traffic-related delays, long waiting times, missed schedules)
    ("Bus ta ajke 45 minute late kore ashlo, office miss holo", 0),
    ("Still standing at the bus counter for an hour with no bus in sight", 0),
    ("আজ সকালেও বাস সময়মতো আসেনি, এক ঘণ্টা অপেক্ষা করতে হলো", 0),
    ("Heavy traffic delayed our trip to Dhanmondi by two hours", 0),
    ("The morning bus schedule is completely unreliable on weekdays", 0),
    ("Bus rasta e noshto hoye gese, sobai deri te pouchailam", 0),
    ("No bus came for 40 minutes during evening rush hour", 0),
    ("রাস্তায় জ্যামের কারণে সময়মতো ক্লাসে যেতে পারিনি", 0),
    ("Why is there never a bus on time according to the timetable?", 0),
    ("Everyday waiting time at the stop exceeds half an hour", 0),
    ("প্রতিদিন সকালে বাস পেতে অনেক সময় নষ্ট হয়", 0),
    ("Missed the connecting coach because the earlier bus was delayed", 0),

    # 1: Fare (Ticket prices, overcharging, fare disputes, payment-related issues)
    ("Conductor refused to give back my 30 taka change after taking a 50 taka note", 1),
    ("They charge different ticket rates for the exact same distance every morning", 1),
    ("কন্ডাক্টর ছাত্র আইডি দেখার পরও হাফ ভাড়া নেয়নি", 1),
    ("Paid via mobile wallet twice because the first transaction errored out", 1),
    ("Ticket er dam achanok double kore dise kono notice charai", 1),
    ("Charged full adult fare to a university student with valid ID card", 1),
    ("অতিরিক্ত ভাড়া চাওয়ায় যাত্রীর সাথে কন্ডাক্টরের কথা কাটাকাটি হলো", 1),
    ("My travel pass card got debited two times at the gate", 1),
    ("ভাড়ার তালিকা না মেনে ইচ্ছেমতো টাকা নিচ্ছে", 1),
    ("Collected cash for the trip without issuing any printed ticket", 1),
    ("Ticket price has become unaffordable for daily commuters", 1),
    ("খুচরা না থাকার অজুহাতে বাকি টাকা ফেরত দিল না", 1),

    # 2: Safety (Harassment, reckless driving, accidents, unsafe behaviour, security concerns)
    ("Passengers were sitting on the bus roof and emergency exits were blocked", 2),
    ("Driver appeared intoxicated and was swerving across lanes", 2),
    ("বাসে মহিলা যাত্রীকে উত্ত্যক্ত করার চেষ্টা করা হয়েছে", 2),
    ("Two city buses competed at dangerous speeds and almost hit a pedestrian", 2),
    ("A commuter slipped and fell down while boarding because the driver accelerated early", 2),
    ("Bus er moddhe mobile churi r pickpocket onek bere gese", 2),
    ("চালক মোবাইল ফোনে কথা বলতে বলতে ঝুঁকিপূর্ণভাবে ওভারটেক করছিল", 2),
    ("Travelling alone after 10 PM feels completely unsafe due to poor security", 2),
    ("Someone attempted to snatch a passenger's bag near the open window", 2),
    ("বাসের ধাক্কায় রাস্তার পাশে একটি মোটরসাইকেল দুর্ঘটনার কবলে পড়েছে", 2),
    ("Driving at terrifying speed with doors wide open in heavy traffic", 2),
    ("রাতে বাসে কোনো নিরাপত্তা ব্যবস্থা নেই, ছিনতাইয়ের ভয় থাকে", 2),

    # 3: Cleanliness (Dirty vehicles, waste, unpleasant smell, hygiene-related complaints)
    ("There was spit on the side panels and sticky dirt on the handrails", 3),
    ("Saw small lizards and bugs crawling under the dirty seats", 3),
    ("বাসের ভেতরে আবর্জনা আর দুর্গন্ধে বসা যাচ্ছিল না", 3),
    ("The washroom at the main bus terminal is completely unhygienic and foul-smelling", 3),
    ("Bus er floor e pani chuiye porche ar kada bhore ache", 3),
    ("Garbage bins at the station are overflowing onto the platform", 3),
    ("সিটের ওপর বমির দাগ পরিষ্কার না করেই যাত্রী তোলা হয়েছে", 3),
    ("Dusty windows and stained seat covers make the cabin look filthy", 3),
    ("বাস কখনো ঠিকমতো ধোয়া হয় না, ভেতরে ভ্যাপসা ও পচা গন্ধ", 3),
    ("Sweat and rotting trash odor inside the closed bus made me nauseous", 3),
    ("টার্মিনালের টয়লেট ব্যবহারের অযোগ্য, চারদিকে ময়লা", 3),
    ("Empty food packets and plastic bottles scattered across the aisle", 3),

    # 4: Service (Staff behaviour, overcrowding, poor service quality, general passenger experience)
    ("Overall service standard is terrible and management ignores complaints", 4),
    ("In the morning the bus is so packed there is zero room even to step inside", 4),
    ("বাসে প্রচণ্ড ভিড় থাকায় ভেতরে দাঁড়ানোর মতো কোনো জায়গা ছিল না", 4),
    ("Commuters are squeezed tightly like sardines every weekday morning", 4),
    ("Staff shouted rudely when an elderly passenger asked which stop was next", 4),
    ("Bus staff treat students disrespectfully and refuse to answer questions", 4),
    ("মহিলাদের জন্য সংরক্ষিত সিটে পুরুষ যাত্রীরা বসে আছে, কেউ বারণ করছে না", 4),
    ("The bus was too crowded to breathe comfortably and the AC was turned off", 4),
    ("হেল্পার আমার সিগন্যাল দেখেও স্টপেজে বাস না থামিয়ে চলে গেল", 4),
    ("Ticket counter personnel completely ignored everyone waiting in line", 4),
    ("Driver refused to pull over at the designated bus bay", 4),
    ("এসি কাজ না করায় গরমে যাত্রীদের ভীষণ কষ্ট হচ্ছিল", 4),

    # 5: Other (Feedback not clearly belonging to categories 0-4: inquiries, lost & found, app/web, praise, signage, suggestions, testing)
    ("Please add wheelchair accessible ramps for disabled commuters", 5),
    ("Could you provide the helpline phone number of the transport office?", 5),
    ("আমার মানিব্যাগ বাসে পড়ে গেছে, কোথায় খুঁজতে হবে জানাবেন?", 5),
    ("Which bus number goes directly toward the airport terminal?", 5),
    ("The updated mobile ticketing app is very convenient and easy to use, thanks", 5),
    ("Had a very pleasant ride today, the staff was polite and helpful", 5),
    ("বাস স্টপেজে রুট ও স্টেশনের নাম লেখা কোনো সাইনবোর্ড নেই", 5),
    ("Suggest introducing GPS live tracking on the commuter portal", 5),
    ("Please consider launching a direct shuttle route to Banani and Uttara", 5),
    ("অনলাইন পোর্টালে বাসের সময়সূচি পেজটি লোড হচ্ছে না", 5),
    ("Checking whether this online feedback submission form works properly", 5),
    ("When does the final night coach depart from Motijheel toward Uttara?", 5),
]

def get_synthetic_df():
    return pd.DataFrame(SYNTHETIC_EXAMPLES, columns=["text", "label"])

if __name__ == "__main__":
    df = get_synthetic_df()
    print("Synthetic counts:", df["label"].value_counts().sort_index().to_dict(), "Total:", len(df))
