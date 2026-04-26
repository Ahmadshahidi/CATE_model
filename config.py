# Configuration for Incremental Campaign Uplift Modeling Project
# Treatment Design: offer amounts only (configurable)
# remail is treated as a predictor / covariate and toggled in optimization scenarios.
# stipulation is a 5-level categorical predictor; cost is zero and it is not toggled.

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Data Generation Parameters
N_SAMPLES = 20000        # ~5,000 samples per treatment arm
N_FEATURES = 1800
RANDOM_SEED = 42

# ===========================================================
# TREATMENT DESIGN  (offer amounts — fully configurable)
# ===========================================================
#
# ┌─────────────────────────────────────────────────────────┐
# │  HOW TO ADD / REMOVE OFFER ARMS                         │
# │                                                         │
# │  1. Edit TREATMENT_COMPONENTS:                          │
# │     • Key 0 must always be the control arm (offer=$0).  │
# │     • Other keys are arm IDs (any positive integers).   │
# │     • Values are the offer dollar amounts.              │
# │                                                         │
# │  Examples:                                              │
# │    Add $1000 arm:   4: 1000                             │
# │    Remove $100 arm: delete the  1: 100  entry           │
# │                                                         │
# │  2. (Optional) Set TREATMENT_PROBS to a list of floats  │
# │     summing to 1.0, one per arm, in the same order as   │
# │     TREATMENT_COMPONENTS.  Leave it as None to use      │
# │     equal allocation across all arms.                   │
# │                                                         │
# │  Everything else (labels, offer list, model loops,      │
# │  output columns) is derived automatically.              │
# └─────────────────────────────────────────────────────────┘
#
# remail is NOT a treatment arm — it is a covariate/predictor toggled
# via OPTIMIZATION_SCENARIOS at the optimization step.
# stipulation is a 5-level categorical covariate; its cost is zero and
# it cannot be globally toggled (it varies per prospect independently).
#
# Treatment ID 0 is always the control arm.
# ===========================================================

TREATMENT_COMPONENTS = {
    0: 0,    # Control  (offer = $0)
    1: 100,  # $100 offer
    2: 400,  # $400 offer
    3: 500,  # $500 offer
    4: 1000, # $1000 offer
}

# Human-readable label for each arm (auto-derived — do not edit)
TREATMENTS = {
    k: ('Control' if v == 0 else f'${v}')
    for k, v in TREATMENT_COMPONENTS.items()
}

# Offer dollar amounts, excluding control (auto-derived — do not edit)
OFFER_AMOUNTS = [v for k, v in TREATMENT_COMPONENTS.items() if k != 0]

# -----------------------------------------------------------
# Treatment allocation probabilities
# Set to None for equal allocation across all arms.
# Set to a list of floats (one per arm, same order as
# TREATMENT_COMPONENTS, summing to 1.0) for custom splits.
# -----------------------------------------------------------
TREATMENT_PROBS = None   # None → equal split; or e.g. [0.20, 0.25, 0.30, 0.25]

# Resolve probabilities (equal split when TREATMENT_PROBS is None)
if TREATMENT_PROBS is None:
    _n_arms = len(TREATMENT_COMPONENTS)
    _equal  = round(1.0 / _n_arms, 10)
    TREATMENT_PROBS = [_equal] * _n_arms
    # Correct floating-point rounding on the last element
    TREATMENT_PROBS[-1] = round(1.0 - sum(TREATMENT_PROBS[:-1]), 10)

# Quick sanity checks at import time
assert abs(sum(TREATMENT_PROBS) - 1.0) < 1e-9, \
    f"TREATMENT_PROBS must sum to 1.0 (got {sum(TREATMENT_PROBS):.6f})"

assert len(TREATMENT_PROBS) == len(TREATMENT_COMPONENTS), \
    ("TREATMENT_PROBS length must match TREATMENT_COMPONENTS — "
     "set TREATMENT_PROBS = None for automatic equal allocation.")

# ===========================================================
# STIPULATION LEVELS
# ===========================================================
# stipulation is a 5-level categorical predictor assigned per
# prospect.  The same offer arm can receive different stipulation
# levels.  Cost is zero; stipulation is not toggled in scenarios.
# Rename levels here to match your business nomenclature.
# ===========================================================

STIPULATION_LEVELS = ['none', 'basic', 'standard', 'enhanced', 'premium']

# ===========================================================
# OPTIMIZATION SCENARIOS
# ===========================================================
# In the optimization step we run the net-value optimizer for
# each scenario defined here.  remail is set globally for the
# entire mailing in each scenario.  stipulation is excluded —
# it varies per prospect and has zero cost.
#
# Keys   : short scenario label (used in output files / charts)
# Values : dict with 'remail' flag (0 or 1)
# ===========================================================

OPTIMIZATION_SCENARIOS = {
    'no_remail': {'remail': 0},
    'remail_on': {'remail': 1},
}

# ===========================================================
# COST PARAMETERS
# ===========================================================
OFFER_COST_RATE  = 0.60   # 10% of the offer dollar amount
REMAIL_COST      = 3.0   # Fixed $3 per prospect when remail = 1
# stipulation cost is zero — no entry needed

# ===========================================================
# TREATMENT EFFECT PARAMETERS (data generation)
# ===========================================================
# remail boosts the offer effect (multiplicative).
# stipulation has a graded ordinal effect keyed to STIPULATION_LEVELS.
REMAIL_EFFECT_BOOST = 0.05   # +5% of base offer effect

# Ordinal CATE multiplier per stipulation level (same order as STIPULATION_LEVELS).
# Each value is added to the prospect's CATE as a fraction of the base CATE.
STIPULATION_EFFECT_BY_LEVEL = {
    'none':     0.00,
    'basic':    0.05,
    'standard': 0.10,
    'enhanced': 0.15,
    'premium':  0.20,
}

# ===========================================================
# BIAS CORRECTION METHOD
# ===========================================================
# Selects how (if at all) selection bias is corrected before
# training the X-Learner.
#
#   'psm'     → Propensity Score Matching (nearest-neighbour 1:1).
#               PSM diagnostic plots are always saved.
#               Set USE_MATCHED_DATA_FOR_XLEARNER = True to train
#               the X-Learner on the matched sub-sample instead of
#               the full dataset.
#               Limitation: some arms may have zero matches when
#               propensity overlap is poor (small/rare arms).
#
#   'iptw'    → Inverse Probability of Treatment Weighting.
#               All rows are kept; each row is re-weighted by
#               w = P(T=t) / P(T=t | X)  (stabilised, recommended).
#               Weights are passed as sample_weight to the X-Learner.
#               Limitation: weights can blow up with highly skewed
#               arm sizes or extreme propensity scores.
#
#   'overlap' → Overlap Weighting (Li, Morgan & Zaslavsky 2018).
#               All rows are kept; each row is re-weighted by
#               w = h(x) / e_k(x)  where h(x) is the harmonic mean
#               of propensity scores across all arms.
#               Weights are bounded in (0, 1] — variance is always
#               finite, no arm gets zero weight, and no aggressive
#               trimming is needed.  Recommended for production data
#               with heavily skewed arm sizes.
#               Note: estimates ATE on the overlap population (units
#               where treatment assignment was genuinely ambiguous).
#
#   'none'    → No bias correction.  X-Learner trains on the full
#               unweighted dataset.
# ===========================================================

BIAS_CORRECTION_METHOD = 'psm'   # 'psm' | 'iptw' | 'overlap' | 'none'

# ===========================================================
# PROPENSITY SCORE MATCHING (PSM)
# ===========================================================
# PSM is run once per treatment arm (arm vs. control) after
# Boruta-SHAP feature selection.  Diagnostics are always saved;
# matched data can optionally replace the full dataset for the
# X-Learner.
#
# PSM_METHOD  : propensity score estimator used for matching.
#   'catboost'  → CatBoostClassifier (recommended — handles
#                 mixed/missing data natively, no scaling needed,
#                 consistent with the X-Learner's base learners)
#   'logistic'  → Logistic Regression (fast fallback; requires
#                 scaling and cannot handle missing values)
#   'xgboost'   → XGBoost binary classifier (flexible, requires
#                 xgboost package)
#
# PSM_CALIPER : max allowed difference in propensity score
#              (in SD units of the log-odds score)
# USE_MATCHED_DATA_FOR_XLEARNER : if True, X-Learner trains on
#               PSM-matched data instead of the full dataset.
# PSM_KEY_COVARIATES : highlighted in covariate balance plots.
# ===========================================================

PSM_METHOD       = 'catboost'   # 'catboost' | 'logistic' | 'xgboost'
PSM_CALIPER      = 0.01          # SD units of log-odds
PSM_RANDOM_STATE = RANDOM_SEED

USE_MATCHED_DATA_FOR_XLEARNER = False  # True → train on matched data
XLEARNER_RETAINED_ONLY        = False  # True → train only on on_book_month9 == 1

# ===========================================================
# INVERSE PROBABILITY OF TREATMENT WEIGHTING (IPTW)
# ===========================================================
# Used when BIAS_CORRECTION_METHOD = 'iptw'.
#
# IPTW_PS_METHOD : propensity score estimator for IPTW weights.
#   'catboost'  → CatBoostClassifier (recommended — handles
#                 mixed/missing data natively, consistent with
#                 the X-Learner's base learners)
#   'logistic'  → Multinomial / one-vs-rest LR (fast fallback)
#   'xgboost'   → One-vs-rest XGBoost (flexible; requires xgboost)
#
# IPTW_STABILIZED : True  → stabilised weights  w = P(T) / P(T|X)
#                   False → raw weights          w = 1    / P(T|X)
#                   Stabilised weights are strongly recommended;
#                   they have the same mean as raw weights but
#                   lower variance.
# IPTW_TRIM_PERCENTILE : Trim weights below this percentile and
#                        above (100 - this percentile) to limit
#                        extreme values.  Set to 0 to disable.
# IPTW_RANDOM_STATE : RNG seed for the PS estimator.
# ===========================================================

IPTW_PS_METHOD        = 'catboost'  # 'catboost' | 'logistic' | 'xgboost'
IPTW_STABILIZED       = True         # stabilised weights (recommended)
IPTW_TRIM_PERCENTILE  = 1.0          # trim extreme weights at each tail (%)
IPTW_RANDOM_STATE     = RANDOM_SEED

# ===========================================================
# OVERLAP WEIGHTING
# ===========================================================
# Used when BIAS_CORRECTION_METHOD = 'overlap'.
#
# OVERLAP_PS_METHOD : propensity score estimator (same options as IPTW).
#   'catboost'  → recommended (handles missing/mixed data natively)
#   'logistic'  → fast fallback
#   'xgboost'   → flexible; requires xgboost package
#
# OVERLAP_TRIM_PERCENTILE : optional per-arm trimming.  Because overlap
#   weights are naturally bounded in (0, 1], extreme values are rare.
#   Set to 0 to disable (recommended default for overlap weighting).
# ===========================================================

OVERLAP_PS_METHOD       = 'catboost'  # 'catboost' | 'logistic' | 'xgboost'
OVERLAP_TRIM_PERCENTILE = 0.0         # 0 = no trimming (weights already bounded)

# ===========================================================
# ENTROPY BALANCING REFINEMENT
# ===========================================================
# Optional post-processing step applied after IPTW or overlap
# weighting.  For any feature still above BALANCE_REFINE_SMD_THRESHOLD
# after the initial weights, entropy balancing directly optimises
# the treated-group weights to satisfy balance constraints.
#
# Objective  : minimise KL divergence from initial weights
#              (keeps weights as close to 1.0 as possible while
#               achieving balance — avoids extreme weight collapse)
# Constraints: weighted mean of each imbalanced feature in the
#              treated group equals weighted mean in control group
# Solver     : scipy SLSQP (handles linear equality + bound constraints)
#
# If a feature cannot converge (zero overlap), a warning is printed
# and the original weights are kept for that arm — no crash.
# ===========================================================

BALANCE_REFINE               = False  # True → run entropy balancing after initial weights
BALANCE_REFINE_SMD_THRESHOLD = 0.10   # refine features with |SMD| above this value

# Covariates highlighted in balance plots (must survive feature selection)
PSM_KEY_COVARIATES = [
    'estimated_income',
    'credit_score_modeled',
    'age',
    'estimated_net_worth',
    'liquid_assets_estimated',
]

# ===========================================================
# NUMERIC NaN IMPUTATION (propensity score / sklearn paths)
# ===========================================================
# Controls how numeric NaNs are filled in _encode_cats() before
# passing data to logistic regression or XGBoost propensity models.
# CatBoost paths are NOT affected (CatBoost handles NaNs natively).
#
#   'median'  → replace NaN with the column median  (default; robust to outliers)
#   'mean'    → replace NaN with the column mean
#   <number>  → replace NaN with that fixed value, e.g. 0 or -1
# ===========================================================
PS_NUMERIC_NAN_IMPUTE = 'median'   # 'median' | 'mean' | <number>

# ===========================================================
# FEATURE SELECTION PARAMETERS
# ===========================================================
# Step 1: Initial Pruning
VARIANCE_THRESHOLD    = 0.01   # Remove features below this variance
CORRELATION_THRESHOLD = 0.95   # Remove one of two correlated features

# Step 2: Boruta-SHAP
BORUTA_N_TRIALS      = 50
BORUTA_PERCENTILE    = 100     # Use max of shadow features
BORUTA_RANDOM_STATE  = RANDOM_SEED

# ---------------------------------------------------------------------------
# TOP-N GUARANTEE
# ---------------------------------------------------------------------------
# When set to an integer, Boruta-SHAP guarantees *at least* that many features
# are returned for the respective model path.  Features are added in priority
# order:
#   1. Confirmed (accepted) features — always included first.
#   2. Tentative features, ranked by Boruta hit-rate (highest first) — the
#      "rough-fix" tier; added until the Top-N floor is met.
#   3. Any remaining features (including normally-rejected ones), ranked by
#      their Boruta importance / hit-rate score — added until Top-N is met.
#
# Set to None to disable (default Boruta behaviour — include confirmed +
# tentative only).
# ---------------------------------------------------------------------------
BORUTA_BALANCE_TOP_N   = None   # e.g. 30  →  keep at least 30 for X-Learner
BORUTA_ATTRITION_TOP_N = None   # e.g. 20  →  keep at least 20 for AttritionModel

# ---------------------------------------------------------------------------
# FORCE-INCLUDE FEATURES
# ---------------------------------------------------------------------------
# Features listed here are *always* added to the final feature set for the
# respective model path, regardless of whether Leshy (Boruta) accepted or
# rejected them.  Use this to guarantee that domain-critical variables are
# never silently dropped.
#
# • Any column not present in the post-Step-1 feature matrix will be added as
#   an all-zeros column with a warning (inference / transform mode).
# • Lists may be empty ([]) to disable the override.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FORCE-INCLUDE CATEGORICAL FEATURES
# ---------------------------------------------------------------------------
# Categorical columns in this list are forced into EVERY model path
# (both the X-Learner balance model and the attrition classifier),
# regardless of Leshy's verdict.
#
# These 6 columns are generated by src/data_generation.py and passed
# through Step 1 (InitialPruning) unchanged.  Leshy may elect to reject
# some of them if the encoded integer codes show low SHAP importance —
# the entries below guarantee they are always available to CatBoost, which
# handles them natively.
#
# Add or remove entries to control which categorical columns are forced in.
# Comment out any column you want to let Leshy decide on instead.
# ---------------------------------------------------------------------------
FORCE_INCLUDE_CATEGORICAL = [
    'risk_tier',            # A / B / C / D  — derived from credit score
    'region',               # Northeast / Midwest / South / West
    'occupation_category',  # 8 broad occupational groups
    'channel_preference',   # Mail / Online / Branch / Mobile
    'life_stage_category',  # Young Single / Young Family / Mature Family / ...
    'stipulation',          # 5-level categorical campaign variable
    # 'state',              # uncomment to also force-include the 50-state column
]

# Per-model-path force-include lists.
# Extend with additional numeric features specific to each path if needed.
BORUTA_BALANCE_FORCE_INCLUDE   = list(FORCE_INCLUDE_CATEGORICAL) + [
    # e.g. 'estimated_income',
]
BORUTA_ATTRITION_FORCE_INCLUDE = list(FORCE_INCLUDE_CATEGORICAL) + [
    # e.g. 'age',
]

# ===========================================================
# MODEL PARAMETERS
# ===========================================================

# -----------------------------------------------------------
# Log-transform toggle  (IMPORTANT for skewed balance targets)
# -----------------------------------------------------------
# Opening balance is heavily right-skewed in real data:
#   ~90% of prospects have balances < $10,000, while the top
#   ~10% account for ~92% of total balance volume.
#
# How CatBoost handles skewness:
#   • Symmetric-tree splits partition the feature space without
#     assuming Gaussian errors, so CatBoost is more robust to
#     outliers than linear models.
#   • Leaf-wise L2 regularisation (l2_leaf_reg) naturally shrinks
#     leaf values toward the mean, dampening extreme predictions.
#   • However, with RMSE as the objective, very large balances
#     dominate the loss and the model may over-fit to the top
#     decile while under-representing treatment effects for
#     lower-balance prospects.
#
# Recommended settings for production data with extreme skew:
#
#   Option 1 — LOG TRANSFORM (recommended for severe skew)
#     Set LOG_TRANSFORM_TARGET = True.
#     y → log1p(y) before fitting; CATEs are back-transformed
#     via expm1() automatically.  Benefits:
#       - Compresses the tail; RMSE loss treats all balance
#         levels more equally.
#       - CATE pseudo-outcomes become approximately symmetric,
#         improving CATE model fit.
#       - Avoids one large outlier dominating the gradients.
#     Caveat: CATE is estimated on the log scale and then
#       back-transformed, which gives a *multiplicative* effect;
#       interpret as "% lift" rather than "$ lift" until
#       converted back.
#
#   Option 2 — WINSORISING (alternative or complement)
#     Clip y at a high percentile (e.g. 99th) before fitting,
#     or use CatBoost's built-in quantile/MAE objective:
#       CATBOOST_REGRESSOR_PARAMS['loss_function'] = 'Quantile:alpha=0.9'
#     This directly optimises the median/high-quantile effect
#     rather than the mean.
#
#   Option 3 — RAW TARGET with increased regularisation (current)
#     Set LOG_TRANSFORM_TARGET = False and rely on CatBoost's
#     robust tree splits + l2_leaf_reg.  Safe for moderate skew;
#     may under-fit the bulk of lower-balance prospects when
#     skew is extreme (top-10% / 92% of volume scenario).
#
# For production deployment, consider:
#   1. Enabling LOG_TRANSFORM_TARGET = True  (best general choice)
#   2. Raising l2_leaf_reg to 5–10 in CATBOOST_REGRESSOR_PARAMS
#   3. Evaluating CATE calibration separately for low-balance
#      and high-balance cohorts (e.g. by decile of opening_balance)
# -----------------------------------------------------------
LOG_TRANSFORM_TARGET = False   # Set True for heavily right-skewed production balances

# -----------------------------------------------------------
# Monotonic constraints for Epsilon wealth features.
#   Keys must match feature column names that survive feature
#   selection.  Set to {} to disable all constraints.
#     1  → prediction must be non-decreasing in that feature
#    -1  → prediction must be non-increasing in that feature
# -----------------------------------------------------------
MONOTONE_CONSTRAINTS = {
    'estimated_net_worth':    1,   # higher net worth → higher balance
    'liquid_asset_indicator': 1,   # higher liquid assets → higher balance
}

# -----------------------------------------------------------
# XGBoost base learner parameters (kept for reference / fallback)
#   All keys are passed directly to XGBRegressor(**XGBOOST_PARAMS).
# -----------------------------------------------------------
XGBOOST_PARAMS = {
    'objective':    'reg:squarederror',
    'n_estimators': 200,
    'max_depth':    5,
    'learning_rate': 0.05,
    'reg_alpha':    10.0,
    'reg_lambda':   1.0,
    'subsample':    0.7,
    'random_state': RANDOM_SEED,
    'n_jobs':       -1,
}

# -----------------------------------------------------------
# CatBoost — Attrition / retention classifier
#   Used in AttritionModel (src/models/attrition_model.py).
#   CatBoost handles categorical features and missing values
#   natively; no pre-encoding or imputation is required.
# -----------------------------------------------------------
CATBOOST_CLASSIFIER_PARAMS = {
    'iterations':        800,
    'depth':             6,
    'learning_rate':     0.05,
    'l2_leaf_reg':       3.0,
    'eval_metric':       'AUC',
    'od_type':           'Iter',         # early stopping criterion
    'od_wait':           50,             # patience (iterations)
    'random_seed':       RANDOM_SEED,
    'thread_count':      -1,
    'verbose':           False,
    'allow_writing_files': False,
    # nan_mode='Min' treats missing values as below any observed value
    # (safe default for mixed financial/demographic data)
    'nan_mode':          'Min',
}

# -----------------------------------------------------------
# CatBoost — X-Learner outcome & CATE regressors
#   Used for µ₀, µₖ, τ̂ₜ, τ̂_c in XLearnerUplift.
#   CATE pseudo-outcomes can be negative → use RMSE objective.
# -----------------------------------------------------------
CATBOOST_REGRESSOR_PARAMS = {
    'iterations':        600,
    'depth':             6,
    'learning_rate':     0.05,
    'l2_leaf_reg':       3.0,
    'eval_metric':       'RMSE',
    'od_type':           'Iter',
    'od_wait':           50,
    'random_seed':       RANDOM_SEED,
    'thread_count':      -1,
    'verbose':           False,
    'allow_writing_files': False,
    'nan_mode':          'Min',
}

# -----------------------------------------------------------
# CatBoost — Propensity score classifier (inside X-Learner)
#   Used to estimate P(T=k | X) for blending τ̂ₜ / τ̂_c.
# -----------------------------------------------------------
CATBOOST_PROPENSITY_PARAMS = {
    'iterations':        400,
    'depth':             5,
    'learning_rate':     0.05,
    'l2_leaf_reg':       3.0,
    'eval_metric':       'AUC',
    'od_type':           'Iter',
    'od_wait':           40,
    'random_seed':       RANDOM_SEED,
    'thread_count':      -1,
    'verbose':           False,
    'allow_writing_files': False,
    'nan_mode':          'Min',
}

# Train-Test Split
TEST_SIZE       = 0.3
VALIDATION_SIZE = 0.15   # Out of training set

# ===========================================================
# MODEL REGISTRY DIRECTORY
# ===========================================================
# Serialized model artefacts for handoff / deployment.
# Created automatically by pipeline.py (Step 5) and
# consumed by src/scoring/score_new_data.py.
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Decile targeting parameters (used by scoring script)
N_DECILES      = 10   # Total deciles for prospect ranking
TOP_N_DECILES  = 3    # Top deciles that receive a letter of offer

# ===========================================================
# Create output directories
# ===========================================================
os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)
