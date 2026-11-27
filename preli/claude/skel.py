import pandas as pd, numpy as np, re, unicodedata
BR="bkash|nagad|rocket|upay|surecash|mcash|tap|ok wallet|pathao pay|pathao courier|pathao|dbbl|dutch-bangla bank|islami bank|brac bank|city bank|prime bank|pubali bank|jamuna bank|standard chartered|ucb|mutual trust bank|ebl|trust bank|sundarban courier|sa paribahan|redx|steadfast|paperfly|karatoa|ecourier|dhl|fedex|janani courier|sheba delivery|grameenphone|gp|robi axiata|robi|banglalink|airtel|teletalk|skitto|daraz|rokomari|ajkerdeal|star tech|aarong|shwapno|chaldal|pickaboo|unilever bangladesh|beximco|walton|square group|pran-rfl|apex|bangladesh bank|nbr tax office|nid wing|btrc|customs house|passport office|pm education fund|vfs global|desco|nesco|wasa|titas gas|btcl|dpdc|breb"
def skel(s):
    s=unicodedata.normalize('NFKC',s).lower()
    s=re.sub(r'[\u200b-\u200f\u2060\ufeff]','',s)
    s=re.sub(r'https?://\S+|hxxps?://\S+','URL',s)
    s=re.sub(r'(\+?880 ?)?01\d[x\d]{8,9}','PH',s)
    s=re.sub(r'[0-9০-৯][0-9০-৯,\.]*','0',s)
    s=re.sub(r'\b('+BR+r')\b','BR',s)
    s=re.sub(r'\b[a-z]*\d[a-z\d]*\b','ID',s)
    s=re.sub(r'\s+',' ',s).strip()
    return s
