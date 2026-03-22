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
OFFER_COST_RATE  = 0.10   # 10% of the offer dollar amount
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
# PROPENSITY SCORE MATCHING (PSM)
# ===========================================================
# PSM is run once per treatment arm (arm vs. control) after
# Boruta-SHAP feature selection.  Diagnostics are always saved;
# matched data can optionally replace the full dataset for the
# X-Learner.
#
# PSM_METHOD       : 'logistic' (fast) or 'xgboost' (more flexible)
# PSM_CALIPER      : max allowed difference in propensity score
#                    (in SD units of the log-odds score)
# USE_MATCHED_DATA : if True, X-Learner trains on PSM-matched data
# PSM_KEY_COVARIATES: highlighted in covariate balance plots
# ===========================================================

PSM_METHOD       = 'logistic'   # 'logistic' | 'xgboost'
PSM_CALIPER      = 0.01          # SD units of log-odds
PSM_RANDOM_STATE = RANDOM_SEED

USE_MATCHED_DATA_FOR_XLEARNER = False  # True → train on matched data

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
# Log-transform toggle
#   True  → y = np.log1p(y) before fitting the X-Learner;
#            CATE predictions are back-transformed via np.expm1()
#            so all downstream code still works in dollar-space.
#   False → raw target, no transformation.
# -----------------------------------------------------------
LOG_TRANSFORM_TARGET = False   # squarederror objective handles negative CATE pseudo-outcomes directly

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
# XGBoost base learner parameters
#   All keys are passed directly to XGBRegressor(**XGBOOST_PARAMS).
#   Edit any value here; the model code reads from this dict.
# -----------------------------------------------------------
XGBOOST_PARAMS = {
    # reg:squarederror is required here because the X-Learner's
    # second-stage models fit on CATE pseudo-outcomes which can be
    # negative. Tweedie/Poisson objectives require non-negative labels.
    'objective':    'reg:squarederror',
    'n_estimators': 200,
    'max_depth':    5,
    'learning_rate': 0.05,
    'reg_alpha':    10.0,   # L1 — suppresses noise features
    'reg_lambda':   1.0,    # L2
    'subsample':    0.7,
    'random_state': RANDOM_SEED,
    'n_jobs':       -1,
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
