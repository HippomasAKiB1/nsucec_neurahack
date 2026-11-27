import re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

TECH_LIST = ["T1", "T2", "T3", "T4", "T5", "T6"]


def tech_str_to_vec(t_str: str) -> np.ndarray:
    v = np.zeros(6, dtype=np.int32)
    s = str(t_str).strip()
    if s and s != "NONE":
        for code in s.split("|"):
            if code in TECH_LIST:
                v[TECH_LIST.index(code)] = 1
    return v


def vec_to_tech_str(v: np.ndarray) -> str:
    codes = [TECH_LIST[i] for i in range(6) if v[i] > 0]
    return "|".join(codes) if codes else "NONE"


def decode_consistent(p_lbl, p_tech, p_act, tech_th=0.5):
    pred_l = p_lbl.argmax(axis=1).astype(int)
    pred_a = np.where(np.isin(pred_l, [0, 5]), 0, 1 + p_act[:, 1:].argmax(axis=1)).astype(int)
    if np.ndim(tech_th) == 0:
        pred_t = (p_tech >= tech_th).astype(int)
    else:
        pred_t = (p_tech >= np.asarray(tech_th)[None, :]).astype(int)
    is_np = np.isin(pred_l, [0, 5])
    pred_t[is_np] = 0
    need_one = (~is_np) & (pred_t.sum(axis=1) == 0)
    if need_one.any():
        pred_t[need_one, p_tech[need_one].argmax(axis=1)] = 1
    return pred_l, pred_t, pred_a


def assign_sample_scenario_family(clean_text: str, label: int) -> str:
    """Coarse scenario-family clustering on sample.csv for leak-free Leave-One-Scenario-Out CV."""
    t = clean_text.lower()
    if label == 1:
        if any(k in t for k in ("vul", "bhul", "mistake", "wrong", "ferot")):
            return "mfs_mistake"
        if any(k in t for k in ("jitechen", "won", "bonus", "prize", "parcel", "atke", "customs", "ghor bose")):
            return "pri1_cross_cat"
        return "mfs_cred_harvest"
    if label == 2:
        if "cashback" in t or "ক্যাশব্যাক" in t or "jitechen" in t or "জিতেছেন" in t:
            return "prize_cashback_lakh"
        return "prize_gift_voucher"
    if label == 3:
        if any(k in t for k in ("data entry", "unilever", "bdjobs", "beximco", "abroad", "malaysia")):
            return "job_recruitment_fee"
        if any(k in t for k in ("crypto", "invest", "double", "arbitrage")):
            return "job_crypto_invest"
        return "job_task_earn"
    if label == 4:
        if any(k in t for k in ("customs", "duty", "china", "malaysia")):
            return "deliv_customs_duty"
        if any(k in t for k in ("failed", "address", "delivery attempt")):
            return "deliv_addr_phish"
        return "deliv_fee_link"
    if label == 5:
        if any(k in t for k in ("sale", "discount", "off", "চাড়", "ছাড়", "11.11", "eid")):
            return "promo_sale_disc"
        return "promo_offer_ad"
    # label == 0 sub-families
    if any(k in t for k in ("scam", "phishing", "sabdhan", "block")):
        return "legit_scam_warn"
    if any(k in t for k in ("otp", "login code", "verification code")):
        return "legit_otp_warn"
    if any(k in t for k in ("cash out", "cash-out", "received", "debited", "credited", "bill", "recharge", "বিল", "ক্যাশ আউট")):
        return "legit_tx_bill"
    if any(k in t for k in ("parcel", "order", "delivered", "rider", "পার্সেল", "ডেলিভারি")):
        return "legit_deliv_ok"
    return "legit_chat_other"


# Scenario-Disjoint 2-Fold Assignment (matching sample.csv & synth_64k_v4 families share zero scenario leakage across folds!)
FOLD_1_FAMILIES = {
    "legit_tx_bill", "legit_hr_doc", "legit_tuition", "legit_chat_faq", "legit_chat_other",
    "mfs_account_block", "mfs_limit_aml", "mfs_cred_harvest", "wrong_tx_readout",
    "borrowed_wallet_pin", "pri1_parcel_pin", "pri1_prize_pin", "pri1_cross_cat",
    "prize_link_bn", "prize_mystery_box", "prize_fee_bn", "prize_call_wa_bn_en", "prize_pii_reply", "prize_cashback_lakh",
    "job_crypto_arb", "job_package_ambassador", "job_double_invest", "job_data_entry", "job_crypto_invest",
    "deliv_storage_charge", "deliv_addr_url", "deliv_call_phone", "deliv_addr_phish",
    "promo_app_bogo", "promo_telco", "promo_offer_ad",
}


def get_scenario_disjoint_folds(families) -> np.ndarray:
    """Return binary fold IDs (0 or 1) with zero scenario overlap between fold 0 and fold 1."""
    return np.array([1 if str(f) in FOLD_1_FAMILIES else 0 for f in families], dtype=np.int32)


BALANCED_3FOLD_MAP = {
    # Fold 0
    "legit_otp": 0, "legit_otp_warn": 0, "legit_chat_other": 0, "legit_noise": 0,
    "mfs_alert_card": 0, "wrong_tx_pin": 0, "pri1_scholarship_pin": 0, "mfs_mistake": 0,
    "prize_link_en": 0, "prize_router_allowance": 0, "prize_apk_rewards": 0, "prize_cashback_lakh": 0,
    "job_task_like": 0, "job_bn_hiring_reg": 0, "job_double_invest": 0, "job_recruitment_fee": 0,
    "deliv_customs_duty": 0, "deliv_addr_phish": 0, "promo_sale": 0, "promo_sale_disc": 0,
    # Fold 1
    "legit_tx_bill": 1, "legit_cod": 1, "legit_deliv_ok": 1, "legit_chat_faq": 1,
    "mfs_account_block": 1, "wrong_tx_return_money": 1, "pri1_parcel_pin": 1, "pri1_prize_pin": 1, "pri1_cross_cat": 1,
    "prize_link_bn": 1, "prize_fee_en": 1, "prize_call_wa_bn_en": 1,
    "job_crypto_arb": 1, "job_package_ambassador": 1, "job_call_visa_bn": 1, "job_task_earn": 1,
    "deliv_storage_charge": 1, "deliv_bn_url": 1, "deliv_fee_link": 1, "promo_app_bogo": 1, "promo_realestate_edu": 1,
    # Fold 2
    "legit_agent_cashout": 2, "legit_hr_doc": 2, "legit_tuition": 2, "legit_scam_warn": 2,
    "mfs_limit_aml": 2, "mfs_breb_western": 2, "wrong_tx_readout": 2, "borrowed_wallet_pin": 2, "pri1_job_pin": 2, "mfs_cred_harvest": 2,
    "prize_usdt_survey": 2, "prize_mystery_box": 2, "prize_fee_bn": 2, "prize_pii_reply": 2, "prize_gift_voucher": 2,
    "job_volunteer_review": 2, "job_data_entry": 2, "job_crypto_invest": 2,
    "deliv_addr_url": 2, "deliv_call_phone": 2, "promo_telco": 2, "promo_offer_ad": 2,
}


def get_balanced_3folds(families) -> np.ndarray:
    """Balanced 3-fold scenario-disjoint partition (every fold keeps 67% of every class!)."""
    return np.array([BALANCED_3FOLD_MAP.get(str(f), hash(str(f)) % 3) for f in families], dtype=np.int32)


def score_section10_proxy(
    y_l_primary, p_l_primary, y_t_primary, p_t_primary, y_a_primary, p_a_primary, conf_primary,
    y_l_ood=None, p_l_ood=None, y_t_ood=None, p_t_ood=None, y_a_ood=None, p_a_ood=None, conf_ood=None,
    y_l_adv=None, p_l_adv=None, conf_adv=None,
    y_l_multi=None, p_l_multi=None,
):
    """Exact Section 10 composite metric across Primary, OOD Domain-Shift/Novel, Adversarial, and Calibration."""
    fa = 100.0 * f1_score(y_l_primary, p_l_primary, average="macro", zero_division=0)
    fb = 100.0 * f1_score(y_t_primary, p_t_primary, average="macro", zero_division=0)
    fc = 100.0 * f1_score(y_a_primary, p_a_primary, average="macro", zero_division=0)

    fd = 100.0 * f1_score(y_l_ood, p_l_ood, average="macro", zero_division=0) if y_l_ood is not None else fa
    fe = 100.0 * f1_score(y_l_adv, p_l_adv, average="macro", zero_division=0) if y_l_adv is not None else fa
    ff = 100.0 * f1_score(y_l_multi, p_l_multi, average="macro", zero_division=0) if y_l_multi is not None else fd

    if conf_ood is not None and y_l_ood is not None:
        all_corr = np.concatenate([(y_l_primary == p_l_primary), (y_l_ood == p_l_ood)]).astype(np.float64)
        all_conf = np.concatenate([conf_primary, conf_ood]).astype(np.float64)
    else:
        all_corr = (y_l_primary == p_l_primary).astype(np.float64)
        all_conf = np.asarray(conf_primary, dtype=np.float64)
    brier = float(np.mean((all_conf - all_corr) ** 2))
    c_score = 100.0 * max(0.0, 1.0 - brier)
    comp = 0.40 * fa + 0.15 * fb + 0.10 * fc + 0.10 * fd + 0.10 * fe + 0.05 * c_score + 0.10 * ff
    return {
        "Composite": round(comp, 3),
        "F_A_primary": round(fa, 2),
        "F_B_tech": round(fb, 2),
        "F_C_action": round(fc, 2),
        "F_D_ood": round(fd, 2),
        "F_E_adv": round(fe, 2),
        "F_F_multi_novel": round(ff, 2),
        "C_conf": round(c_score, 2),
        "Brier": round(brier, 5),
        "Acc_primary": round(100.0 * accuracy_score(y_l_primary, p_l_primary), 2),
    }
