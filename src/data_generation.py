"""
Synthetic Epsilon-like Data Generator
======================================
Generates a dataset that mimics the structure of a real Epsilon prospect file:
  - 1,800 features grouped into demographics, financial, behavioral,
    geographic, psychographic, and noise blocks
  - 6 named categorical feature columns (stored as pd.Categorical) covering
    geography, life-stage, occupation, risk tier, and channel preference
  - Realistic NaN injection in selected numeric AND categorical columns
    (MCAR / MAR patterns, 5-18% missing rates)
  - 4 treatment arms: Control, $100, $400, $500 offer
  - remail as an additional binary prospect-level covariate
  - stipulation as a 5-level categorical prospect-level covariate
  - Heterogeneous treatment effects driven by estimated_income
  - Outcomes: opening_balance (continuous) and on_book_month9 (binary)

Usage:
    from src.data_generation import generate_epsilon_data, save_data
    df = generate_epsilon_data()
    save_data(df)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Feature-block spec  (total = 1800 numeric columns)
# ---------------------------------------------------------------------------
FEATURE_BLOCKS = [
    # (block_name, n_features)
    ('demo',         200),   # Age, income, education, HH composition
    ('financial',    300),   # Credit scores, debt ratios, banking products
    ('behavioral',   400),   # Purchase propensities, channel preferences
    ('geographic',   200),   # ZIP-level demographics, regional indicators
    ('psychographic',400),   # Modeled scores, life stages, wealth segments
    ('noise',        300),   # Near-constant & highly correlated noise
]

assert sum(n for _, n in FEATURE_BLOCKS) == config.N_FEATURES, \
    f"Block spec must sum to {config.N_FEATURES}"

# ---------------------------------------------------------------------------
# US states mapped to regions
# ---------------------------------------------------------------------------
_STATES_BY_REGION = {
    'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'Midwest':   ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO',
                  'NE', 'ND', 'SD'],
    'South':     ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL',
                  'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
    'West':      ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK',
                  'CA', 'HI', 'OR', 'WA'],
}
_ALL_STATES = [s for states in _STATES_BY_REGION.values() for s in states]
_STATE_TO_REGION = {s: r for r, states in _STATES_BY_REGION.items()
                    for s in states}


def _generate_named_features(n: int, rng: np.random.Generator) -> dict:
    """
    Generate the named key feature columns with realistic distributions.
    Returns a dict {col_name: np.ndarray}.
    """
    age                       = rng.normal(45, 14, n).clip(18, 85).astype(int)
    estimated_income          = np.exp(rng.normal(10.8, 0.6, n)).clip(15_000, 500_000)
    education_level           = rng.integers(1, 6, n)
    household_size            = rng.integers(1, 7, n)
    marital_status            = rng.integers(0, 3, n)
    number_of_children        = rng.integers(0, 6, n)
    homeowner_flag            = rng.binomial(1, 0.62, n)
    years_at_current_address  = rng.exponential(7, n).clip(0, 40).astype(int)
    employment_status         = rng.integers(0, 4, n)
    occupation_code           = rng.integers(1, 20, n)

    income_z                  = ((estimated_income - estimated_income.mean()) /
                                  estimated_income.std())
    credit_score_modeled      = (700 + 60 * income_z +
                                 rng.normal(0, 40, n)).clip(300, 850).astype(int)
    estimated_net_worth       = (estimated_income * rng.uniform(1, 5, n) +
                                 rng.normal(0, 20_000, n)).clip(0, 2_000_000)
    liquid_assets_estimated   = (estimated_net_worth * rng.uniform(0.05, 0.4, n) +
                                 rng.normal(0, 2_000, n)).clip(0, None)
    liquid_asset_indicator    = (liquid_assets_estimated > 5_000).astype(int)
    total_debt_estimated      = (estimated_income * rng.uniform(0.5, 3, n) +
                                 rng.normal(0, 10_000, n)).clip(0, None)
    mortgage_balance_estimated = (homeowner_flag * estimated_income *
                                   rng.uniform(2, 5, n)).clip(0, None)
    auto_loan_indicator        = rng.binomial(1, 0.48, n)
    number_of_credit_cards     = rng.integers(0, 8, n)
    revolving_utilization      = rng.beta(2, 5, n).clip(0, 1)
    number_of_bank_accounts    = rng.integers(0, 5, n)
    savings_account_indicator  = rng.binomial(1, 0.55, n)
    investment_account_indicator = rng.binomial(1, 0.35, n)
    retirement_account_indicator = rng.binomial(1, 0.45, n)
    checking_account_balance_est = (estimated_income / 12 *
                                     rng.uniform(0.5, 2, n)).clip(0, None)
    debt_to_income_ratio       = (total_debt_estimated /
                                   estimated_income.clip(1, None)).clip(0, 10)

    direct_mail_responsiveness = rng.beta(1.5, 3, n)
    online_banking_activity    = rng.integers(0, 30, n)
    mobile_banking_activity    = rng.integers(0, 50, n)
    credit_card_spend_monthly  = (estimated_income / 12 *
                                   rng.uniform(0.02, 0.20, n)).clip(0, None)
    debit_card_spend_monthly   = (estimated_income / 12 *
                                   rng.uniform(0.05, 0.30, n)).clip(0, None)
    atm_usage_frequency        = rng.integers(0, 10, n)
    branch_visit_frequency     = rng.integers(0, 4, n)
    online_purchase_propensity = rng.beta(2, 2, n)

    zip_median_income          = (estimated_income * rng.uniform(0.8, 1.2, n) +
                                   rng.normal(0, 5_000, n)).clip(20_000, None)
    zip_home_value_index       = (homeowner_flag * rng.normal(300_000, 100_000, n) +
                                   (1 - homeowner_flag) *
                                   rng.normal(200_000, 80_000, n)).clip(50_000, None)
    life_stage_score           = rng.normal(50, 15, n).clip(0, 100)
    wealth_segment_score       = (100 * (estimated_net_worth /
                                          estimated_net_worth.max())).clip(0, 100)
    financial_stress_index     = (100 * debt_to_income_ratio /
                                   debt_to_income_ratio.clip(1).max()).clip(0, 100)
    investment_propensity_score = (50 * investment_account_indicator +
                                    rng.normal(0, 10, n)).clip(0, 100)

    return {
        'age':                          age,
        'estimated_income':             estimated_income,
        'education_level':              education_level,
        'household_size':               household_size,
        'marital_status':               marital_status,
        'number_of_children':           number_of_children,
        'homeowner_flag':               homeowner_flag,
        'years_at_current_address':     years_at_current_address,
        'employment_status':            employment_status,
        'occupation_code':              occupation_code,
        'credit_score_modeled':         credit_score_modeled,
        'estimated_net_worth':          estimated_net_worth,
        'liquid_assets_estimated':      liquid_assets_estimated,
        'liquid_asset_indicator':       liquid_asset_indicator,
        'total_debt_estimated':         total_debt_estimated,
        'mortgage_balance_estimated':   mortgage_balance_estimated,
        'auto_loan_indicator':          auto_loan_indicator,
        'number_of_credit_cards':       number_of_credit_cards,
        'revolving_utilization':         revolving_utilization,
        'number_of_bank_accounts':      number_of_bank_accounts,
        'savings_account_indicator':    savings_account_indicator,
        'investment_account_indicator': investment_account_indicator,
        'retirement_account_indicator': retirement_account_indicator,
        'checking_account_balance_est': checking_account_balance_est,
        'debt_to_income_ratio':         debt_to_income_ratio,
        'direct_mail_responsiveness':   direct_mail_responsiveness,
        'online_banking_activity':      online_banking_activity,
        'mobile_banking_activity':      mobile_banking_activity,
        'credit_card_spend_monthly':    credit_card_spend_monthly,
        'debit_card_spend_monthly':     debit_card_spend_monthly,
        'atm_usage_frequency':          atm_usage_frequency,
        'branch_visit_frequency':       branch_visit_frequency,
        'online_purchase_propensity':   online_purchase_propensity,
        'zip_median_income':            zip_median_income,
        'zip_home_value_index':         zip_home_value_index,
        'life_stage_score':             life_stage_score,
        'wealth_segment_score':         wealth_segment_score,
        'financial_stress_index':       financial_stress_index,
        'investment_propensity_score':  investment_propensity_score,
    }


def _vectorized_choice(categories: list, prob_matrix: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised per-row categorical sampling.

    Parameters
    ----------
    categories   : list of K category strings
    prob_matrix  : (n, K) float array — each row is a probability distribution
                   (rows need not sum to exactly 1; they are normalised here)
    rng          : numpy Generator

    Returns numpy array of shape (n,) with category strings.
    """
    prob_matrix = np.clip(prob_matrix, 0.0, None)
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    cum = np.cumsum(prob_matrix, axis=1)         # (n, K)
    u   = rng.random(len(prob_matrix))[:, None]  # (n, 1)
    idx = (u > cum).sum(axis=1)                  # (n,) — index of chosen category
    idx = np.clip(idx, 0, len(categories) - 1)
    return np.array(categories)[idx]


def _generate_categorical_features(named: dict, n: int,
                                    rng: np.random.Generator) -> dict:
    """
    Generate 6 realistic categorical feature columns (fully vectorised).

    Columns
    -------
    state                : US state abbreviation (50 levels, 6% NaN)
    region               : Geographic region  (4 levels, derived from state)
    occupation_category  : 8 broad occupational groups (9% NaN)
    risk_tier            : A / B / C / D  derived from credit score (5% NaN)
    channel_preference   : Mail / Online / Branch / Mobile (11% NaN)
    life_stage_category  : 5 life-stage segments (5% NaN)
    """
    credit_scores = named['credit_score_modeled']
    ages          = named['age'].astype(float)
    income        = named['estimated_income']

    def _inject_nan(arr, rate):
        arr = arr.astype(object)
        idx = rng.choice(n, size=int(rate * n), replace=False)
        arr[idx] = None
        return arr

    def _to_cat(arr, categories):
        return pd.Categorical(arr, categories=categories)

    # -- state / region --------------------------------------------------
    state_arr = rng.choice(_ALL_STATES, size=n, replace=True).astype(object)
    state_arr = _inject_nan(state_arr, 0.06)
    region_arr = np.array(
        [_STATE_TO_REGION.get(s, None) if s is not None else None
         for s in state_arr],
        dtype=object,
    )

    # -- occupation_category  (9% NaN, vectorised) -----------------------
    occ_cats = ['Professional', 'Trade', 'Service', 'Retail',
                 'Finance', 'Healthcare', 'Education', 'Other']
    income_z = (income - income.mean()) / income.std()
    iz       = np.clip(income_z, -2, 2)
    # Build (n, 8) probability matrix
    w_prof  = (0.10 + 0.05 * iz).reshape(-1, 1)
    w_trade = (0.15 - 0.03 * iz).reshape(-1, 1)
    w_fin   = (0.08 + 0.04 * iz).reshape(-1, 1)
    w_rest  = np.full((n, 1), 0.13)
    w_base  = np.tile([0.0, 0.0, 0.13, 0.12, 0.0, 0.10, 0.08, 0.0], (n, 1))
    w_base[:, 0:1] = w_prof
    w_base[:, 1:2] = w_trade
    w_base[:, 4:5] = w_fin
    w_base[:, 7:8] = np.maximum(1.0 - w_prof - w_trade - w_fin - 0.43, 0.01)
    occ_arr = _inject_nan(_vectorized_choice(occ_cats, w_base, rng), 0.09)

    # -- risk_tier  (A/B/C/D derived from credit score, 5% NaN) ----------
    tier_arr = np.where(credit_scores >= 750, 'A',
               np.where(credit_scores >= 680, 'B',
               np.where(credit_scores >= 620, 'C', 'D'))).astype(object)
    tier_arr = _inject_nan(tier_arr, 0.05)

    # -- channel_preference  (11% NaN, vectorised) -----------------------
    ch_cats = ['Mail', 'Online', 'Branch', 'Mobile']
    age_z   = (ages - ages.mean()) / ages.std()
    az      = np.clip(age_z, -2, 2)
    ch_probs = np.column_stack([
        np.clip(0.20 + 0.08 * az, 0.05, None),   # Mail
        np.clip(0.30 - 0.06 * az, 0.05, None),   # Online
        np.clip(0.25 + 0.04 * az, 0.05, None),   # Branch
        np.clip(0.25 - 0.06 * az, 0.05, None),   # Mobile
    ])
    ch_arr = _inject_nan(_vectorized_choice(ch_cats, ch_probs, rng), 0.11)

    # -- life_stage_category  (5% NaN, vectorised) -----------------------
    ls_cats = ['Young Single', 'Young Family', 'Mature Family',
               'Empty Nester', 'Retiree']
    # Probability matrix driven by age bands
    ls_probs = np.zeros((n, 5))
    ls_probs[:, 0] = np.where(ages < 30, 0.70,
                     np.where(ages < 40, 0.20, 0.00))            # Young Single
    ls_probs[:, 1] = np.where(ages < 30, 0.30,
                     np.where(ages < 40, 0.50, 0.00))            # Young Family
    ls_probs[:, 2] = np.where(ages < 40, 0.00,
                     np.where(ages < 55, 0.60, 0.00))            # Mature Family
    ls_probs[:, 3] = np.where(ages < 40, 0.00,
                     np.where(ages < 55, 0.40,
                     np.where(ages < 65, 0.70, 0.05)))           # Empty Nester
    ls_probs[:, 4] = np.where(ages < 55, 0.00,
                     np.where(ages < 65, 0.30, 0.95))            # Retiree
    ls_probs = np.clip(ls_probs, 0.01, None)
    ls_arr = _inject_nan(_vectorized_choice(ls_cats, ls_probs, rng), 0.05)

    return {
        'state':               _to_cat(state_arr,  categories=_ALL_STATES),
        'region':              _to_cat(region_arr, categories=list(_STATES_BY_REGION.keys())),
        'occupation_category': _to_cat(occ_arr,    categories=occ_cats),
        'risk_tier':           _to_cat(tier_arr,   categories=['A', 'B', 'C', 'D']),
        'channel_preference':  _to_cat(ch_arr,     categories=ch_cats),
        'life_stage_category': _to_cat(ls_arr,     categories=ls_cats),
    }


# ---------------------------------------------------------------------------
# Target columns that must NEVER contain NaN
# ---------------------------------------------------------------------------
_TARGET_COLS = ('opening_balance', 'on_book_month9', 'treatment', 'offer')


def _inject_numeric_nan(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject NaN into numeric predictor columns with realistic rates.

    Two tiers of injection:
    1. Named domain columns — tailored missing-data rates that mirror
       real Epsilon data patterns (mortgage data gaps, partial bank
       reporting, etc.).
    2. Generic padded columns (demo_XXXX, fin_XXXX, …) — a random 10%
       sample of these gets a random MCAR rate (3–15%) applied, so the
       downstream pipeline must handle sparse NaN throughout the feature
       matrix, not just in a handful of named columns.

    Target columns (opening_balance, on_book_month9, treatment, offer)
    are explicitly excluded and will NEVER receive NaN.

    Parameters
    ----------
    df  : pd.DataFrame  (numeric + categorical block columns)
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame  with NaN injected into predictor columns only.
    """
    n = len(df)
    df = df.copy()

    # ── Tier 1: named domain columns ────────────────────────────────
    named_nan_spec = {
        'mortgage_balance_estimated':   0.18,
        'checking_account_balance_est': 0.10,
        'estimated_net_worth':          0.08,
        'credit_card_spend_monthly':    0.12,
        'liquid_assets_estimated':      0.07,
        'total_debt_estimated':         0.05,
        'debit_card_spend_monthly':     0.06,
        'revolving_utilization':        0.05,
    }
    for col, rate in named_nan_spec.items():
        if col in df.columns and col not in _TARGET_COLS:
            nan_idx = rng.choice(n, size=int(rate * n), replace=False)
            df.loc[nan_idx, col] = np.nan

    # ── Tier 2: generic padded columns (block_XXXX) ──────────────────
    # Identify all generic numeric padded columns (prefixed block names)
    pad_prefixes = ('demo_', 'fin_', 'beh_', 'geo_', 'psy_', 'noise_')
    pad_cols = [
        c for c in df.columns
        if any(c.startswith(p) for p in pad_prefixes)
        and c not in _TARGET_COLS
        and df[c].dtype.kind in ('f', 'i', 'u')
    ]

    # Randomly select ~10% of padded columns to receive NaN
    n_pad_nan = max(1, int(0.10 * len(pad_cols)))
    chosen_pad = rng.choice(pad_cols, size=n_pad_nan, replace=False)

    for col in chosen_pad:
        rate = rng.uniform(0.03, 0.15)          # random 3–15% MCAR rate
        nan_idx = rng.choice(n, size=int(rate * n), replace=False)
        df.loc[nan_idx, col] = np.nan

    return df


def _pad_to_block_size(named: dict, block: str, target_n: int,
                        rng: np.random.Generator, n_samples: int) -> dict:
    """
    Add generic numeric columns to a feature block until it has ``target_n``
    features.  Some include near-constant columns and correlated duplicates.
    """
    n_existing = len(named)
    n_pad      = target_n - n_existing
    result     = dict(named)

    for i in range(n_pad):
        col = f'{block}_{i:04d}'
        r   = rng.random()
        if r < 0.10:
            result[col] = rng.binomial(1, 0.005, n_samples).astype(float)
        elif r < 0.20:
            existing_keys = [k for k in result if k != col]
            if existing_keys:
                src_col = rng.choice(existing_keys)
                result[col] = result[src_col] * rng.uniform(0.95, 1.05)
            else:
                result[col] = rng.normal(0, 50, n_samples)
        elif r < 0.35:
            p = rng.uniform(0.05, 0.95)
            result[col] = rng.binomial(1, p, n_samples).astype(float)
        elif r < 0.55:
            mu, sigma = rng.normal(0, 100), rng.exponential(30)
            result[col] = rng.normal(mu, sigma, n_samples)
        elif r < 0.70:
            result[col] = np.exp(rng.normal(5, 1, n_samples))
        else:
            lo, hi = 0, int(rng.integers(3, 15))
            result[col] = rng.integers(lo, hi + 1, n_samples).astype(float)

    return result


def generate_epsilon_data() -> pd.DataFrame:
    """
    Generate a synthetic Epsilon-like prospect dataset.

    Returns
    -------
    pd.DataFrame with columns:
        <1800 numeric feature columns>
        state, region, occupation_category, risk_tier,
        channel_preference, life_stage_category  : pd.Categorical (with NaN)
        remail      : int  (0/1)
        stipulation : pd.Categorical  (5 levels: none/basic/standard/enhanced/premium)
        treatment      : int  (0=Control, 1=$100, 2=$400, 3=$500, ...)
        treatment_name : str
        offer          : int  (dollar amount)
        opening_balance: float
        on_book_month9 : int  (binary)
    """
    print("\n" + "=" * 60)
    print("DATA GENERATION  (Synthetic Epsilon-like Dataset)")
    print("=" * 60)

    n   = config.N_SAMPLES
    rng = np.random.default_rng(config.RANDOM_SEED)

    # ── Numeric feature blocks ───────────────────────────────────
    print(f"\n  Generating {n:,} samples x {config.N_FEATURES} numeric features ...")

    named_feats = _generate_named_features(n, rng)

    block_cols = {}

    demo_named = {k: v for k, v in named_feats.items()
                  if k in ['age', 'estimated_income', 'education_level',
                            'household_size', 'marital_status',
                            'number_of_children', 'homeowner_flag',
                            'years_at_current_address', 'employment_status',
                            'occupation_code']}
    block_cols.update(_pad_to_block_size(demo_named, 'demo', 200, rng, n))

    fin_named = {k: v for k, v in named_feats.items()
                 if k in ['credit_score_modeled', 'estimated_net_worth',
                           'liquid_assets_estimated', 'liquid_asset_indicator',
                           'total_debt_estimated', 'mortgage_balance_estimated',
                           'auto_loan_indicator', 'number_of_credit_cards',
                           'revolving_utilization', 'number_of_bank_accounts',
                           'savings_account_indicator',
                           'investment_account_indicator',
                           'retirement_account_indicator',
                           'checking_account_balance_est',
                           'debt_to_income_ratio']}
    block_cols.update(_pad_to_block_size(fin_named, 'fin', 300, rng, n))

    beh_named = {k: v for k, v in named_feats.items()
                 if k in ['direct_mail_responsiveness', 'online_banking_activity',
                           'mobile_banking_activity', 'credit_card_spend_monthly',
                           'debit_card_spend_monthly', 'atm_usage_frequency',
                           'branch_visit_frequency', 'online_purchase_propensity']}
    block_cols.update(_pad_to_block_size(beh_named, 'beh', 400, rng, n))

    geo_named = {k: v for k, v in named_feats.items()
                 if k in ['zip_median_income', 'zip_home_value_index']}
    block_cols.update(_pad_to_block_size(geo_named, 'geo', 200, rng, n))

    psy_named = {k: v for k, v in named_feats.items()
                 if k in ['life_stage_score', 'wealth_segment_score',
                           'financial_stress_index', 'investment_propensity_score']}
    block_cols.update(_pad_to_block_size(psy_named, 'psy', 400, rng, n))

    block_cols.update(_pad_to_block_size({}, 'noise', 300, rng, n))

    # ── Treatment assignment ────────────────────────────────────
    arm_ids       = list(config.TREATMENT_COMPONENTS.keys())
    probs         = config.TREATMENT_PROBS
    treatment_arr = rng.choice(arm_ids, size=n, p=probs)
    offer_arr     = np.array([config.TREATMENT_COMPONENTS[t]
                               for t in treatment_arr])

    # ── remail & stipulation ────────────────────────────────────
    treated_mask = offer_arr > 0
    remail       = np.zeros(n, dtype=int)
    remail[treated_mask] = rng.binomial(1, 0.50, treated_mask.sum())

    # stipulation: 5-level categorical, assigned independently of arm
    # (same offer can get different stipulation levels)
    stip_levels   = config.STIPULATION_LEVELS          # ['none','basic',...]
    stip_arr      = rng.choice(stip_levels, size=n)    # uniform, all prospects
    stipulation   = pd.Categorical(stip_arr, categories=stip_levels, ordered=True)

    # ── Outcome generation ───────────────────────────────────────
    income   = named_feats['estimated_income']
    inc_z    = (income - income.mean()) / income.std()
    stress   = named_feats['financial_stress_index']
    stress_z = (stress - stress.mean()) / stress.std()
    util     = named_feats['revolving_utilization']
    util_z   = (util - util.mean()) / util.std()
    dm_resp  = named_feats['direct_mail_responsiveness']
    dm_z     = (dm_resp - dm_resp.mean()) / dm_resp.std()
    homeown  = named_feats['homeowner_flag'].astype(float)
    nw       = named_feats['estimated_net_worth']
    nw_z     = (nw - nw.mean()) / nw.std()
    invest   = named_feats['investment_account_indicator'].astype(float)

    base_balance = np.exp(rng.normal(8.56, 0.40, n)).clip(500, 50_000)

    offer_arms = sorted(
        [(arm_id, offer)
         for arm_id, offer in config.TREATMENT_COMPONENTS.items()
         if arm_id != 0],
        key=lambda x: x[1],
    )
    n_offer_arms = len(offer_arms)
    max_offer    = offer_arms[-1][1] if offer_arms else 1

    cate = np.zeros(n)
    for rank, (arm_id, offer) in enumerate(offer_arms):
        frac      = rank / max(n_offer_arms - 1, 1)
        base_lift = offer * 0.86
        low_mod   = (1 - frac) * (-80 * inc_z + 60 * stress_z + 50 * util_z)
        high_mod  = frac       * (120 * inc_z + 100 * nw_z    + 80 * invest)
        mid_bell  = 1 - abs(2 * frac - 1)
        mid_mod   = mid_bell * (60 * dm_z + 80 * homeown - 30 * np.abs(inc_z))
        arm_effect = base_lift + low_mod + mid_mod + high_mod
        cate += np.where(treatment_arr == arm_id, arm_effect, 0.0)

    cate += remail * cate * config.REMAIL_EFFECT_BOOST
    stip_boost = np.array([config.STIPULATION_EFFECT_BY_LEVEL[s] for s in stip_arr])
    cate += stip_boost * cate
    opening_balance = (base_balance + cate + rng.normal(0, 300, n)).clip(0, None)

    credit_z  = (named_feats['credit_score_modeled'] - 650) / 80
    balance_z = ((opening_balance - opening_balance.mean()) /
                  opening_balance.std())
    age_arr   = named_feats['age'].astype(float)
    age_z     = (age_arr - age_arr.mean()) / age_arr.std()
    dti       = named_feats['debt_to_income_ratio']
    dti_z     = (dti - dti.mean()) / dti.std()
    chk       = named_feats['checking_account_balance_est']
    chk_z     = (chk - chk.mean()) / (chk.std() + 1e-9)

    highest_arm_id   = offer_arms[-1][0] if offer_arms else None
    income_x_top_arm = (inc_z * (treatment_arr == highest_arm_id).astype(float)
                         if highest_arm_id is not None else np.zeros(n))

    arm_retention_adj = {0: 0.0}
    for arm_id, offer in config.TREATMENT_COMPONENTS.items():
        if arm_id != 0:
            arm_retention_adj[arm_id] = -0.80 * (offer / max_offer)

    log_odds = (
        np.log(0.682 / 0.318)
        + 0.50 * credit_z
        + 0.25 * balance_z
        + np.array([arm_retention_adj[t] for t in treatment_arr])
        + 0.15 * age_z
        - 0.20 * dti_z
        - 0.18 * util_z
        + 0.12 * dm_z
        + 0.10 * homeown
        - 0.15 * stress_z
        + 0.08 * chk_z
        - 0.12 * income_x_top_arm
        + 0.05 * remail
        + stip_boost * 0.30   # higher stipulation level → slight retention lift
        + rng.normal(0, 0.15, n)
    )
    retention_prob = 1 / (1 + np.exp(-log_odds))
    on_book_month9 = rng.binomial(1, retention_prob)

    # ── Assemble DataFrame ──────────────────────────────────────
    df = pd.DataFrame(block_cols)

    # Inject NaN into selected numeric columns
    df = _inject_numeric_nan(df, rng)

    # Add remail (binary numeric) and stipulation (5-level categorical)
    df['remail']      = remail
    df['stipulation'] = stipulation

    # Add categorical feature columns
    print("\n  Adding 6 categorical feature columns (with realistic NaN rates) ...")
    cat_feats = _generate_categorical_features(named_feats, n, rng)
    for col, series in cat_feats.items():
        df[col] = series

    # Outcome / meta columns
    df['treatment']       = treatment_arr
    df['treatment_name']  = [config.TREATMENTS[t] for t in treatment_arr]
    df['offer']           = offer_arr
    df['opening_balance'] = opening_balance.round(2)
    df['on_book_month9']  = on_book_month9

    # Row identifier — inserted as the first column so it is never
    # confused with a feature. Simple 0-based integer range; stable
    # because the DataFrame is never shuffled during generation.
    df.insert(0, config.PROSPECT_ID_COL, np.arange(n, dtype=np.int64))

    # Summary
    n_numeric = sum(1 for c in df.columns
                    if df[c].dtype.kind in ('f', 'i', 'u')
                    and c not in ('treatment', 'offer', 'on_book_month9', 'remail'))
    n_cat = sum(1 for c in df.columns
                if hasattr(df[c], 'cat') or df[c].dtype == object)

    print(f"\n  Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Numeric feature columns   : {config.N_FEATURES} (+ 1 binary: remail)")
    print(f"  Categorical feature columns: {len(cat_feats) + 1} (incl. stipulation)")

    print("\n  Categorical feature summary:")
    for col in cat_feats:
        n_missing = df[col].isna().sum()
        n_unique  = df[col].nunique()
        print(f"    {col:<25} {n_unique} levels, "
              f"{n_missing:>5} NaN ({100*n_missing/n:.1f}%)")

    print("\n  Numeric NaN summary (selected columns):")
    nan_cols = ['mortgage_balance_estimated', 'checking_account_balance_est',
                'estimated_net_worth', 'credit_card_spend_monthly',
                'liquid_assets_estimated']
    for col in nan_cols:
        if col in df.columns:
            nn = df[col].isna().sum()
            print(f"    {col:<35} {nn:>5} NaN ({100*nn/n:.1f}%)")

    print(f"\n  Treatment distribution:")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        cnt = (df['treatment'] == arm_id).sum()
        pct = 100 * cnt / n
        off = config.TREATMENT_COMPONENTS[arm_id]
        print(f"    Arm {arm_id}: {config.TREATMENTS[arm_id]:<12}  "
              f"offer=${off:<5}  {cnt:>5,} ({pct:.1f}%)")

    mean_bal = df.groupby('treatment')['opening_balance'].mean()
    print("\n  Mean opening balance by arm:")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        print(f"    Arm {arm_id} ({config.TREATMENTS[arm_id]:<8}): "
              f"${mean_bal[arm_id]:,.2f}")

    ret_by_arm = df.groupby('treatment')['on_book_month9'].mean()
    print("\n  Retention rate by arm:")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        print(f"    Arm {arm_id} ({config.TREATMENTS[arm_id]:<8}): "
              f"{ret_by_arm[arm_id]:.1%}")

    # ── Sanity check: target columns must never have NaN ─────────
    for tgt in ('opening_balance', 'on_book_month9'):
        assert df[tgt].isna().sum() == 0, (
            f"Target column '{tgt}' contains NaN — check outcome generation."
        )
    print("  ✓ Target columns (opening_balance, on_book_month9) are NaN-free.")

    print("=" * 60 + "\n")
    return df


def save_data(df: pd.DataFrame, data_dir: str = None) -> str:
    """
    Save the generated DataFrame to CSV.

    Parameters
    ----------
    df       : pd.DataFrame  Output of generate_epsilon_data().
    data_dir : str           Target directory (default: config.DATA_DIR).

    Returns  str  path where file was saved.
    """
    data_dir = data_dir or config.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'epsilon_synthetic.csv')
    df.to_csv(path, index=False)
    print(f"  Data saved to: {path}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")
    return path


if __name__ == "__main__":
    df = generate_epsilon_data()
    save_data(df)
