# Incremental Campaign Uplift Modeling

A comprehensive machine learning pipeline for modeling treatment effects in bank marketing campaigns using the **3-Step Sieve** feature selection method, **X-Learner** for uplift modeling, and **attrition prediction** with net-value offer optimisation.

## 📋 Project Overview

This project models the opening balance of prospects in an incremental bank campaign that offers three treatment levels ($100, $400, and $500) plus a control group. The pipeline addresses two key business questions:

1. **What is the causal effect of each offer on opening balance?** (Uplift Modeling)
2. **Who will be on-book at month 9 vs. attrited?** (Attrition Prediction)
3. **Which offer maximises net value per prospect?** (Net Value Optimisation)

The project uses **Epsilon-like synthetic data** with 1,800 variables and employs a sophisticated feature selection process, bias correction, and a multi-scenario offer optimiser.

---

## 🎯 Key Features

### 1. **3-Step Sieve Feature Selection**

#### **Step 1: Initial Pruning**
- **Variance Thresholding**: Removes near-constant features (variance < 0.01)
- **Correlation Filter**: Eliminates redundant features with Pearson correlation > 0.95
- **Reduces**: ~1,800 → ~500–700 features

#### **Step 2: Boruta-SHAP**
- Creates "shadow features" (shuffled versions) and competes them against originals
- Uses SHAP values for better interaction detection
- Only keeps features that significantly outperform their randomised "evil twins"
- **Reduces**: ~500–700 → ~100–200 features
- *Compatibility patches applied*: `np.NaN → np.nan` (NumPy 2.x) and `scipy.stats.binom_test` (SciPy 1.12+) are monkey-patched on import so the BorutaShap package works without modification

#### **Step 3: L1 Regularization (LASSO)**
- Embedded directly in XGBoost base learners (`reg_alpha` parameter)
- Automatically shrinks less-useful coefficients to zero

### 2. **Bias Correction (configurable)**

Controlled by `BIAS_CORRECTION_METHOD` in `config.py`:

#### **Propensity Score Matching (PSM)** (`'psm'`)
- Runs 1:1 nearest-neighbour matching for each treatment arm vs. control
- Caliper matching on the log-odds scale
- Balance diagnostics via Standardised Mean Difference (SMD) Love plots
- By default, diagnostics only — set `USE_MATCHED_DATA_FOR_XLEARNER = True` to train on matched data

#### **Inverse Probability of Treatment Weighting (IPTW)** (`'iptw'`)
- Stabilised weights: `w = P(T) / P(T | X)` (recommended)
- Multinomial logistic regression or XGBoost propensity estimator
- Extreme weight trimming at configurable percentile
- Weighted SMD Love plots and Effective Sample Size (ESS) diagnostics
- *Note*: `multi_class='multinomial'` was removed from `LogisticRegression` for scikit-learn ≥ 1.5 compatibility — multinomial is now the default

#### **None** (`'none'`)
- No bias correction; X-Learner trains on full unweighted data

### 3. **X-Learner for Uplift Modeling**

- Multi-treatment CATE estimation (3 treatment arms vs. control)
- Identifies heterogeneous treatment effects by customer segment
- L1-regularised XGBoost base learners with optional monotonic constraints
- Optional log-transform of the target variable

### 4. **Attrition Prediction Model**

Binary classification for month-9 retention using XGBoost. The synthetic data generating process (DGP) now drives `on_book_month9` with **12 predictors**:

| Predictor | Direction | Rationale |
|-----------|-----------|-----------|
| `credit_score_modeled` | + | Better credit → stays longer |
| `opening_balance` | + | Higher balance → more committed |
| Treatment arm (1/2/3) | − | Incentive offers attract price-sensitive customers |
| `age` | + | Older customers more loyal/stable |
| `debt_to_income_ratio` | − | Financial strain → churn |
| `revolving_utilization` | − | High util → less stable |
| `direct_mail_responsiveness` | + | Responsive → more engaged |
| `homeowner_flag` | + | Homeowners more stable |
| `financial_stress_index` | − | Stress → churn risk |
| `checking_account_balance_est` | + | Higher balance → more engaged |
| `income × arm3` interaction | − | High-income + $500 offer → more selective |
| `remail` / `stipulation` | + (small) | Re-contact slightly improves engagement |

### 5. **Net Value Optimisation & Scenario Analysis**

- For each prospect, computes net value under every offer arm:
  `NV = P(retention) × (Baseline + CATE) − offer_cost − scenario_costs`
- Assigns the **optimal offer** that maximises net value
- Runs **4 scenarios** (remail × stipulation ON/OFF combinations)
- Benchmarks personalised strategy vs. "everyone gets same offer" and random assignment
- **Decile targeting**: evaluates mailing only top-N deciles vs. the full population
- AUUC (Area Under Uplift Curve) computed across all strategies

---

## 📊 Synthetic Data Design

The data generating process creates realistic heterogeneous treatment effects so the optimiser distributes offers across all three arms:

| Offer Arm | Sweet-spot customer profile |
|-----------|-----------------------------|
| **$100** | Low income + high financial stress + high revolving utilisation |
| **$400** | Mid income + high direct-mail responsiveness + homeowner |
| **$500** | High income + high net worth + holds investment account |

**Retention direction** is correctly specified: larger offers attract more price-sensitive customers, so retention probability *decreases* with offer size (Control ~68% → $500 ~52%).

---

## 📁 Project Structure

```
CATE_Model/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Centralised configuration
├── pipeline.py                        # Main orchestration script
│
├── data/                              # Generated / uploaded data
│   └── epsilon_synthetic.csv
│
├── models/                            # Serialised model artefacts
│   ├── attrition_model.joblib
│   ├── step1_pruner.joblib
│   ├── step2_boruta.joblib
│   ├── xlearner_uplift.joblib
│   ├── feature_names.json
│   ├── pipeline_config.json
│   └── MANIFEST.txt
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data_generation.py             # Synthetic Epsilon-like data generator
│   ├── utils.py
│   │
│   ├── feature_selection/
│   │   ├── __init__.py
│   │   ├── step1_initial_pruning.py   # Variance + Correlation filter
│   │   └── step2_boruta_shap.py       # Boruta-SHAP (with NumPy/SciPy patches)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── propensity_matching.py     # PSM diagnostics
│   │   ├── iptw.py                    # IPTW weights
│   │   ├── xlearner_uplift.py         # X-Learner CATE
│   │   ├── attrition_model.py         # Retention classifier
│   │   ├── net_value_strategy.py      # Offer optimisation & scenarios
│   │   └── model_registry.py          # Serialisation helpers
│   │
│   └── scoring/
│       ├── __init__.py
│       └── score_new_data.py          # Inference on new prospects
│
├── results/                           # All output plots and CSVs
└── notebooks/                         # Optional exploration
```

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python pipeline.py
```

This will:
1. Generate synthetic Epsilon-like data (20,000 prospects × 1,800 features)
2. Run Step 1 pruning (variance + correlation)
3. Run Step 2 Boruta-SHAP
4. Apply bias correction (PSM / IPTW / none — set in `config.py`)
5. Train X-Learner for CATE estimation
6. Train attrition (retention) model
7. Run net-value optimisation across all 4 remail × stipulation scenarios
8. Run decile targeting analysis
9. Save all results to `results/`

**Expected runtime**: 10–25 minutes (hardware dependent; reduce `BORUTA_N_TRIALS` for faster testing)

---

## 📊 Key Outputs

### Feature Selection
| File | Description |
|------|-------------|
| `step1_pruning_report.txt` | Features removed by variance / correlation |
| `step2_boruta_report.txt` | Boruta accepted / tentative / rejected features |
| `selected_features.txt` | Final feature list |

### PSM Diagnostics
| File | Description |
|------|-------------|
| `psm_love_plot_arm{N}.png` | Love plot — |SMD| before vs. after matching |
| `psm_propensity_overlap_before/after.png` | PS overlap histograms |

### IPTW Diagnostics
| File | Description |
|------|-------------|
| `iptw_love_plot_arm{N}.png` | Love plot — |SMD| before vs. after weighting |
| `iptw_weight_distribution.png` | Raw vs. trimmed weight histograms |
| `iptw_balance_summary.csv` | Per-arm, per-feature weighted SMD |
| `iptw_effective_sample_sizes.csv` | ESS per arm |

### Uplift Modeling
| File | Description |
|------|-------------|
| `uplift_curves.png` | Cumulative gain curves |
| `auuc_comparison.png` | AUUC metric comparison |
| `cumulative_gain.png` | Gain chart |

### Attrition Model
| File | Description |
|------|-------------|
| `attrition_roc_curve.png` | ROC curve (AUC) |
| `attrition_pr_curve.png` | Precision-Recall curve |
| `attrition_feature_importance.png` | Top predictors |

### Net Value Optimisation
| File | Description |
|------|-------------|
| `optimal_offer_distribution.png` | Offer mix pie/bar chart |
| `personalized_qini_curve.png` | Qini curve with AUUC |
| `personalized_cumulative_net_value.png` | Cumulative NV chart |
| `strategy_comparison_bar.png` | Personalised vs. benchmarks |
| `scenario_comparison_bar.png` | remail × stipulation scenarios |
| `scenario_offer_distributions.png` | Offer mix per scenario |
| `decile_distribution.png` | Avg NV by decile |
| `decile_vs_everyone_comparison.png` | Decile targeting vs. offer everyone |
| `net_value_by_offer_boxplot.png` | NV distribution by assigned arm |

---

## 🔧 Configuration (`config.py`)

Key settings:

```python
# Data
N_SAMPLES = 20000
N_FEATURES = 1800

# Bias correction method
BIAS_CORRECTION_METHOD = 'psm'   # 'psm' | 'iptw' | 'none'

# PSM
PSM_METHOD = 'logistic'
PSM_CALIPER = 0.01
USE_MATCHED_DATA_FOR_XLEARNER = False

# IPTW
IPTW_PS_METHOD = 'logistic'
IPTW_STABILIZED = True
IPTW_TRIM_PERCENTILE = 1.0

# Feature selection
BORUTA_N_TRIALS = 100         # Reduce for faster runs
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'reg_alpha': 10.0,        # L1 — suppresses noise features
    'reg_lambda': 1.0,
    ...
}

# Costs
OFFER_COST_RATE  = 0.5        # 50% of offer dollar amount
STIPULATION_COST = 5.0        # $5 per prospect
REMAIL_COST      = 3.0        # $3 per prospect
```

---

## 📈 Example Results

### Treatment Effects on Opening Balance
```
Arm   Treatment    Mean Opening Balance
────────────────────────────────────────
0     Control      $5,200
1     $100 offer   $5,310  (+$110)
2     $400 offer   $5,490  (+$290)
3     $500 offer   $5,640  (+$440)
```

### Retention Rates by Treatment (correct direction)
```
Arm   Treatment    Month-9 Retention
────────────────────────────────────────
0     Control      ~68%
1     $100 offer   ~63%   (−5 pp)
2     $400 offer   ~57%   (−11 pp)
3     $500 offer   ~52%   (−16 pp)
```
*Higher offers attract more price-sensitive customers who are less likely to stay.*

### Feature Reduction
```
Step                         Features
──────────────────────────────────────
Original Epsilon data        1,800
After Step 1 (Pruning)         ~550   (70% reduction)
After Step 2 (Boruta-SHAP)     ~120   (93% reduction)
```

---

## 🧪 Testing Individual Modules

```bash
python src/data_generation.py
python src/feature_selection/step1_initial_pruning.py
python src/feature_selection/step2_boruta_shap.py
python src/models/propensity_matching.py
python src/models/iptw.py
python src/scoring/score_new_data.py
```

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `xgboost` | Gradient boosting (X-Learner base learners, attrition) |
| `lightgbm` | Fast trees for Boruta-SHAP |
| `shap` | SHAP importance values |
| `BorutaShap` | Boruta algorithm (patched for NumPy 2.x + SciPy 1.12+) |
| `scikit-learn` | PSM, IPTW, preprocessing |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `tqdm` | Progress bars (single-line in-place) |
| `scipy` | Statistical tests (binomtest wrapper) |

---

## 📚 References

- **Boruta Algorithm**: Kursa & Rudnicki (2010). *Journal of Statistical Software*, 36(11)
- **SHAP Values**: Lundberg & Lee (2017). *NeurIPS*
- **X-Learner**: Künzel et al. (2019). *PNAS*
- **Propensity Score Matching**: Rosenbaum & Rubin (1983). *Biometrika*, 70(1)
- **IPTW**: Robins et al. (2000). *Epidemiology*
- **Love Plot / SMD**: Austin (2011). *Multivariate Behavioral Research*, 46(3)

---

## ⚠️ Notes

- **Synthetic data only** — replace `data_generation.py` with your data loader for real Epsilon files
- **Boruta-SHAP** can be slow; reduce `BORUTA_N_TRIALS` in `config.py` for faster testing
- **NumPy 2.x** and **SciPy 1.12+** compatibility patches are applied automatically in `step2_boruta_shap.py` — no package edits required
- **scikit-learn ≥ 1.5** compatibility: `multi_class='multinomial'` removed from `LogisticRegression` (now default behaviour)

---

**Happy Modeling! 🚀**
