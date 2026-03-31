# Configuration for Incremental Campaign Uplift Modeling Project
# Treatment Design: offer amounts only ($100, $400, $500)
# remail and stipulation are treated as predictors / covariates,
# not as treatment arms. Optimization runs scenarios with remail/stipulation ON/OFF.

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
# TREATMENT DESIGN  (offer amounts only)
# ===========================================================
# Each treatment is identified by its offer dollar amount.
#   0  → Control  (no offer)
#   1  → $100 offer
#   2  → $400 offer
#   3  → $500 offer
#
# remail and stipulation are NOT treatment arms any more —
# they are covariates/predictors in the model and are toggled
# via OPTIMIZATION_SCENARIOS at the optimization step.
#
# Treatment ID 0 is always the control arm.
# ===========================================================

TREATMENT_COMPONENTS = {
    0: 0,    # Control  (offer = $0)
    1: 100,  # $100 offer
    2: 400,  # $400 offer
    3: 500,  # $500 offer
}

# Human-readable label for each arm
TREATMENTS = {
    k: ('Control' if v == 0 else f'${v}')
    for k, v in TREATMENT_COMPONENTS.items()
}

# Offer dollar amounts (excluding control)
OFFER_AMOUNTS = [100, 400, 500]

# -----------------------------------------------------------
# Treatment allocation probabilities
# Must sum to 1.0 and have one entry per arm in TREATMENT_COMPONENTS.
# -----------------------------------------------------------
TREATMENT_PROBS = [
    0.20,   # 0: Control
    0.25,   # 1: $100
    0.30,   # 2: $400
    0.25,   # 3: $500
]

# Quick sanity check at import time
assert abs(sum(TREATMENT_PROBS) - 1.0) < 1e-9, \
    f"TREATMENT_PROBS must sum to 1.0 (got {sum(TREATMENT_PROBS):.6f})"

assert len(TREATMENT_PROBS) == len(TREATMENT_COMPONENTS), \
    "TREATMENT_PROBS length must match TREATMENT_COMPONENTS"

# ===========================================================
# OPTIMIZATION SCENARIOS
# ===========================================================
# In the optimization step we run the net-value optimizer for
# each scenario defined here.  remail and stipulation are set
# globally for the entire mailing in each scenario.
#
# Keys   : short scenario label (used in output files / charts)
# Values : dict with 'remail' and 'stipulation' flags (0 or 1)
# ===========================================================

OPTIMIZATION_SCENARIOS = {
    'no_remail_no_stip':  {'remail': 0, 'stipulation': 0},
    'remail_on':          {'remail': 1, 'stipulation': 0},
    'stipulation_on':     {'remail': 0, 'stipulation': 1},
    'both_on':            {'remail': 1, 'stipulation': 1},
}

# ===========================================================
# COST PARAMETERS
# ===========================================================
OFFER_COST_RATE  = 0.5   # 10% of the offer dollar amount
STIPULATION_COST = 5.0    # Fixed $5 per prospect when stipulation = 1
REMAIL_COST      = 3.0    # Fixed $3 per prospect when remail = 1

# ===========================================================
# TREATMENT EFFECT PARAMETERS (data generation)
# ===========================================================
# remail / stipulation boost the offer effect (multiplicative).
# These drive the DGP so the model can learn the interactions
# from the predictor set.
STIPULATION_EFFECT_BOOST = 0.10   # +10% of base offer effect
REMAIL_EFFECT_BOOST      = 0.05   # +5% of base offer effect

# ===========================================================
# BIAS CORRECTION METHOD
# ===========================================================
# Selects how (if at all) selection bias is corrected before
# training the X-Learner.
#
#   'psm'  → Propensity Score Matching (nearest-neighbour 1:1).
#             PSM diagnostic plots are always saved.
#             Set USE_MATCHED_DATA_FOR_XLEARNER = True to train
#             the X-Learner on the matched sub-sample instead of
#             the full dataset.
#
#   'iptw' → Inverse Probability of Treatment Weighting.
#             All rows are kept; each row is re-weighted by
#             w = P(T=t) / P(T=t | X)  (stabilised, recommended).
#             Weights are passed as sample_weight to the X-Learner.
#             Diagnostic plots analogous to the PSM Love plots are
#             saved (weighted SMD, weight distributions, ESS).
#
#   'none' → No bias correction.  X-Learner trains on the full
#             unweighted dataset.
# ===========================================================

BIAS_CORRECTION_METHOD = 'psm'   # 'psm' | 'iptw' | 'none'

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
#               (in SD units of the log-odds score)
# USE_MATCHED_DATA_FOR_XLEARNER : if True, X-Learner trains on
#               PSM-matched data instead of the full dataset.
# PSM_KEY_COVARIATES : highlighted in covariate balance plots.
# ===========================================================

PSM_METHOD       = 'catboost'   # 'catboost' | 'logistic' | 'xgboost'
PSM_CALIPER      = 0.01          # SD units of log-odds
PSM_RANDOM_STATE = RANDOM_SEED

USE_MATCHED_DATA_FOR_XLEARNER = False  # True → train on matched data

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

# Covariates highlighted in balance plots (must survive feature selection)
PSM_KEY_COVARIATES = [
    'estimated_income',
    'credit_score_modeled',
    'age',
    'estimated_net_worth',
    'liquid_assets_estimated',
]

# ===========================================================
# FEATURE SELECTION PARAMETERS
# ===========================================================
# Step 1: Initial Pruning
VARIANCE_THRESHOLD    = 0.01   # Remove features below this variance
CORRELATION_THRESHOLD = 0.95   # Remove one of two correlated features

# Step 2: Boruta-SHAP
BORUTA_N_TRIALS      = 100
BORUTA_PERCENTILE    = 100     # Use max of shadow features
BORUTA_RANDOM_STATE  = RANDOM_SEED

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
