"""
Synthetic Epsilon-like Data Generator
======================================
Generates a dataset that mimics the structure of a real Epsilon prospect file:
  - 1,800 features grouped into demographics, financial, behavioral,
    geographic, psychographic, and noise blocks
  - 4 treatment arms: Control, $100, $400, $500 offer
  - remail and stipulation as additional prospect-level covariates
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


# ──────────────────────────────────────────────────────────────────────────────
# Feature-block spec  (total = 1800)
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Named key covariates (must survive feature selection)
# ──────────────────────────────────────────────────────────────────────────────
KEY_FEATURE_NAMES = [
    # Demographics
    'age',
    'estimated_income',
    'education_level',
    'household_size',
    'marital_status',
    'number_of_children',
    'homeowner_flag',
    'years_at_current_address',
    'employment_status',
    'occupation_code',

    # Financial
    'credit_score_modeled',
    'estimated_net_worth',
    'liquid_assets_estimated',
    'liquid_asset_indicator',
    'total_debt_estimated',
    'mortgage_balance_estimated',
    'auto_loan_indicator',
    'number_of_credit_cards',
    'revolving_utilization',
    'number_of_bank_accounts',
    'savings_account_indicator',
    'investment_account_indicator',
    'retirement_account_indicator',
    'checking_account_balance_est',
    'debt_to_income_ratio',

    # Behavioral
    'direct_mail_responsiveness',
    'online_banking_activity',
    'mobile_banking_activity',
    'credit_card_spend_monthly',
    'debit_card_spend_monthly',
    'atm_usage_frequency',
    'branch_visit_frequency',
    'online_purchase_propensity',

    # Geographic / Psychographic
    'zip_median_income',
    'zip_home_value_index',
    'life_stage_score',
    'wealth_segment_score',
    'financial_stress_index',
    'investment_propensity_score',

    # Interaction-driving features
    'remail',
    'stipulation',
]


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

    # Credit & wealth — correlated with income
    income_z                  = (estimated_income - estimated_income.mean()) / \
                                 estimated_income.std()
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

    # Behavioural
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

    # Geographic / Psychographic
    zip_median_income          = (estimated_income * rng.uniform(0.8, 1.2, n) +
                                   rng.normal(0, 5_000, n)).clip(20_000, None)
    zip_home_value_index       = (homeowner_flag * rng.normal(300_000, 100_000, n)
                                   + (1 - homeowner_flag) * rng.normal(200_000, 80_000, n)
                                   ).clip(50_000, None)
    life_stage_score           = rng.normal(50, 15, n).clip(0, 100)
    wealth_segment_score       = (100 * (estimated_net_worth /
                                          estimated_net_worth.max())).clip(0, 100)
    financial_stress_index     = (100 * debt_to_income_ratio /
                                   debt_to_income_ratio.clip(1).max()).clip(0, 100)
    investment_propensity_score= (50 * investment_account_indicator +
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


def _pad_to_block_size(named: dict, block: str, target_n: int,
                        rng: np.random.Generator, n_samples: int) -> dict:
    """
    Add generic numeric columns to a feature block until it has ``target_n`` features.
    Some include near-constant columns and correlated duplicates (noise).
    """
    n_existing = len(named)
    n_pad      = target_n - n_existing
    result = dict(named)

    for i in range(n_pad):
        col = f'{block}_{i:04d}'
        # Inject variety of distributions
        r = rng.random()
        if r < 0.10:
            # Near-constant (noise): most values 0, occasional 1
            result[col] = rng.binomial(1, 0.005, n_samples).astype(float)
        elif r < 0.20:
            # Perfectly correlated duplicate of a random existing column
            # (only if named is non-empty; otherwise fall through to normal)
            existing_keys = [k for k in result if k != col]
            if existing_keys:
                src_col = rng.choice(existing_keys)
                result[col] = result[src_col] * rng.uniform(0.95, 1.05)
            else:
                result[col] = rng.normal(0, 50, n_samples)
        elif r < 0.35:
            # Bernoulli
            p = rng.uniform(0.05, 0.95)
            result[col] = rng.binomial(1, p, n_samples).astype(float)
        elif r < 0.55:
            # Normal
            mu, sigma = rng.normal(0, 100), rng.exponential(30)
            result[col] = rng.normal(mu, sigma, n_samples)
        elif r < 0.70:
            # LogNormal
            result[col] = np.exp(rng.normal(5, 1, n_samples))
        else:
            # Uniform integer ordinal
            lo, hi = 0, int(rng.integers(3, 15))
            result[col] = rng.integers(lo, hi + 1, n_samples).astype(float)

    return result


def generate_epsilon_data() -> pd.DataFrame:
    """
    Generate a synthetic Epsilon-like prospect dataset.

    Returns
    -------
    pd.DataFrame with columns:
        <1800 feature columns>
        treatment      : int  (0=Control, 1=$100, 2=$400, 3=$500)
        treatment_name : str
        offer          : int  (dollar amount)
        stipulation    : int  (0/1)
        remail         : int  (0/1)
        opening_balance: float
        on_book_month9 : int  (binary)
    """
    print("\n" + "="*60)
    print("DATA GENERATION  (Synthetic Epsilon-like Dataset)")
    print("="*60)

    n   = config.N_SAMPLES
    rng = np.random.default_rng(config.RANDOM_SEED)

    # ── Feature blocks ────────────────────────────────────────────
    print(f"\n  Generating {n:,} samples × {config.N_FEATURES} features ...")

    named_feats = _generate_named_features(n, rng)

    block_cols = {}

    # Demo block — inject named demo features first, pad the rest
    demo_named = {k: v for k, v in named_feats.items()
                  if k in ['age', 'estimated_income', 'education_level',
                            'household_size', 'marital_status', 'number_of_children',
                            'homeowner_flag', 'years_at_current_address',
                            'employment_status', 'occupation_code']}
    block_cols.update(_pad_to_block_size(demo_named, 'demo', 200, rng, n))

    # Financial block
    fin_named = {k: v for k, v in named_feats.items()
                 if k in ['credit_score_modeled', 'estimated_net_worth',
                           'liquid_assets_estimated', 'liquid_asset_indicator',
                           'total_debt_estimated', 'mortgage_balance_estimated',
                           'auto_loan_indicator', 'number_of_credit_cards',
                           'revolving_utilization', 'number_of_bank_accounts',
                           'savings_account_indicator', 'investment_account_indicator',
                           'retirement_account_indicator', 'checking_account_balance_est',
                           'debt_to_income_ratio']}
    fin_padded = _pad_to_block_size(fin_named, 'fin', 300, rng, n)
    block_cols.update(fin_padded)

    # Behavioral block
    beh_named = {k: v for k, v in named_feats.items()
                 if k in ['direct_mail_responsiveness', 'online_banking_activity',
                           'mobile_banking_activity', 'credit_card_spend_monthly',
                           'debit_card_spend_monthly', 'atm_usage_frequency',
                           'branch_visit_frequency', 'online_purchase_propensity']}
    beh_padded = _pad_to_block_size(beh_named, 'beh', 400, rng, n)
    block_cols.update(beh_padded)

    # Geographic block
    geo_named = {k: v for k, v in named_feats.items()
                 if k in ['zip_median_income', 'zip_home_value_index']}
    geo_padded = _pad_to_block_size(geo_named, 'geo', 200, rng, n)
    block_cols.update(geo_padded)

    # Psychographic block
    psy_named = {k: v for k, v in named_feats.items()
                 if k in ['life_stage_score', 'wealth_segment_score',
                           'financial_stress_index', 'investment_propensity_score']}
    psy_padded = _pad_to_block_size(psy_named, 'psy', 400, rng, n)
    block_cols.update(psy_padded)

    # Noise block (near-constant or highly correlated features)
    noise_padded = _pad_to_block_size({}, 'noise', 300, rng, n)
    block_cols.update(noise_padded)

    # ── Treatment assignment  ─────────────────────────────────────
    arm_ids  = list(config.TREATMENT_COMPONENTS.keys())
    probs    = config.TREATMENT_PROBS
    treatment_arr = rng.choice(arm_ids, size=n, p=probs)
    offer_arr     = np.array([config.TREATMENT_COMPONENTS[t] for t in treatment_arr])

    # ── remail & stipulation  (only among treated prospects) ──────
    treated_mask  = offer_arr > 0
    stipulation   = np.zeros(n, dtype=int)
    remail        = np.zeros(n, dtype=int)
    stipulation[treated_mask] = rng.binomial(1, 0.50, treated_mask.sum())
    remail[treated_mask]      = rng.binomial(1, 0.50, treated_mask.sum())

    # ── Outcome generation ────────────────────────────────────────
    # Opening balance — heterogeneous CATE driven by multiple features
    # so the optimizer will distribute offers across all three arms.
    #
    #   Arm 1 ($100) sweet-spot: low income + high financial stress
    #                             + high revolving utilisation
    #   Arm 2 ($400) sweet-spot: mid income + direct-mail responsive
    #                             + homeowner
    #   Arm 3 ($500) sweet-spot: high income + high net-worth
    #                             + holds investment account
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

    # Base balance: log-normal, mean ~$5,200
    base_balance = np.exp(rng.normal(8.56, 0.40, n)).clip(500, 50_000)

    # Arm-level CATE: each arm favours a distinct customer segment
    arm1_effect = (
          70                   # base lift
        - 80  * inc_z          # low-income → bigger response
        + 60  * stress_z       # financially stressed → needs relief
        + 50  * util_z         # high revolving util → responds to $100
    )

    arm2_effect = (
          280                  # base lift
        + 60  * dm_z           # direct-mail responsive → mid offer
        + 80  * homeown        # homeowners value mid offer
        - 30  * np.abs(inc_z)  # penalise extremes (mid-income sweet-spot)
    )

    arm3_effect = (
          430                  # base lift
        + 120 * inc_z          # high income → premium offer
        + 100 * nw_z           # high net-worth → high-value product
        + 80  * invest         # investment account holders respond to $500
    )

    cate = np.zeros(n)
    for arm_id, arm_eff in [(1, arm1_effect), (2, arm2_effect), (3, arm3_effect)]:
        cate += np.where(treatment_arr == arm_id, arm_eff, 0.0)

    # Boost from remail and stipulation
    cate += remail      * cate * config.REMAIL_EFFECT_BOOST
    cate += stipulation * cate * config.STIPULATION_EFFECT_BOOST

    opening_balance = (base_balance + cate + rng.normal(0, 300, n)).clip(0, None)

    # on_book_month9 — driven by many covariates for a richer attrition model
    credit_z   = (named_feats['credit_score_modeled'] - 650) / 80
    balance_z  = (opening_balance - opening_balance.mean()) / opening_balance.std()

    age_arr    = named_feats['age'].astype(float)
    age_z      = (age_arr - age_arr.mean()) / age_arr.std()

    dti        = named_feats['debt_to_income_ratio']
    dti_z      = (dti - dti.mean()) / dti.std()

    # util_z already computed above (revolving_utilization)
    # dm_z already computed above (direct_mail_responsiveness)
    # homeown already computed above (homeowner_flag)
    # stress_z already computed above (financial_stress_index)

    chk        = named_feats['checking_account_balance_est']
    chk_z      = (chk - chk.mean()) / (chk.std() + 1e-9)

    # High-income customers receiving $500 offer have strongest negative effect
    # (they are more selective and won't tolerate stipulation conditions)
    income_x_arm3 = inc_z * (treatment_arr == 3).astype(float)

    BASE_RETENTION = {0: 0.682, 1: 0.615, 2: 0.589, 3: 0.521}
    log_odds = (
        np.log(0.682 / 0.318)           # intercept (control mean ~68%)
        + 0.50 * credit_z               # better credit → stays longer
        + 0.25 * balance_z              # higher balance → more committed
        + np.array([{0: 0.0, 1: -0.18, 2: -0.55, 3: -0.80}[t] for t in treatment_arr])
                                        # offer incentives → higher churn
        + 0.15 * age_z                  # older → more stable/loyal
        - 0.20 * dti_z                  # high DTI → financial strain → churn
        - 0.18 * util_z                 # high revolving util → less stable
        + 0.12 * dm_z                   # direct-mail responsive → engages more
        + 0.10 * homeown                # homeowners → more stable
        - 0.15 * stress_z               # financial stress → churn risk
        + 0.08 * chk_z                  # higher checking balance → more engaged
        - 0.12 * income_x_arm3          # high-income × $500 offer → more selective
        + 0.05 * remail
        + 0.03 * stipulation
        + rng.normal(0, 0.15, n)        # reduced noise (richer model needs less)
    )
    retention_prob  = 1 / (1 + np.exp(-log_odds))
    on_book_month9  = rng.binomial(1, retention_prob)

    # ── Assemble DataFrame ────────────────────────────────────────
    # Features first (so they're the first 1800 columns)
    df = pd.DataFrame(block_cols)

    # Add remail and stipulation as feature columns (they survive feature selection
    # as predictors and are used by the model)
    df['remail']      = remail
    df['stipulation'] = stipulation

    # Outcome / meta columns (excluded from X in pipeline.py)
    df['treatment']      = treatment_arr
    df['treatment_name'] = [config.TREATMENTS[t] for t in treatment_arr]
    df['offer']          = offer_arr
    df['opening_balance'] = opening_balance.round(2)
    df['on_book_month9'] = on_book_month9

    print(f"\n  ✓ Generated {df.shape[0]:,} samples × {df.shape[1]} columns")
    print(f"  Feature columns : {df.shape[1] - 6}")

    print(f"\n  Treatment distribution:")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        cnt  = (df['treatment'] == arm_id).sum()
        pct  = 100 * cnt / n
        off  = config.TREATMENT_COMPONENTS[arm_id]
        print(f"    Arm {arm_id}: {config.TREATMENTS[arm_id]:<12}  "
              f"offer=${off:<5}  {cnt:>5,} ({pct:.1f}%)")

    mean_bal_by_arm = df.groupby('treatment')['opening_balance'].mean()
    print(f"\n  Mean opening balance by arm:")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        print(f"    Arm {arm_id} ({config.TREATMENTS[arm_id]:<8}): "
              f"${mean_bal_by_arm[arm_id]:,.2f}")

    ret_by_arm = df.groupby('treatment')['on_book_month9'].mean()
    print(f"\n  Retention rate by arm:")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        print(f"    Arm {arm_id} ({config.TREATMENTS[arm_id]:<8}): "
              f"{ret_by_arm[arm_id]:.1%}")

    print("="*60 + "\n")
    return df


def save_data(df: pd.DataFrame, data_dir: str = None) -> str:
    """
    Save the generated DataFrame to CSV.

    Parameters
    ----------
    df       : pd.DataFrame  Output of generate_epsilon_data().
    data_dir : str           Target directory (default: config.DATA_DIR).

    Returns  str — path where file was saved.
    """
    data_dir = data_dir or config.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'epsilon_synthetic.csv')
    df.to_csv(path, index=False)
    print(f"  ✓ Data saved to: {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return path


if __name__ == "__main__":
    df = generate_epsilon_data()
    save_data(df)
