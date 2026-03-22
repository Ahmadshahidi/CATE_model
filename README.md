# Incremental Campaign Uplift Modeling

A comprehensive machine learning pipeline for modeling treatment effects in bank marketing campaigns using the **3-Step Sieve** feature selection method, **X-Learner** for uplift modeling, and **attrition prediction**.

## 📋 Project Overview

This project models the opening balance of prospects in an incremental bank campaign that offers three treatment levels ($100, $400, and $500) plus a control group. The pipeline addresses two key business questions:

1. **What is the causal effect of each offer on opening balance?** (Uplift Modeling)
2. **Who will be on-book at month 9 vs. attrited?** (Attrition Prediction)

The project uses **Epsilon-like data** with 1,800 variables and employs a sophisticated 3-step feature selection process to handle high-dimensional data effectively.

---

## 🎯 Key Features

### 1. **3-Step Sieve Feature Selection**

The industry gold standard approach for high-dimensional marketing datasets:

#### **Step 1: Initial Pruning**
- **Variance Thresholding**: Removes near-constant features (e.g., "Luxury Car Owner" that is 0 for 99.9% of data)
- **Correlation Filter**: Eliminates redundant features with Pearson correlation > 0.95
- **Reduces**: ~1,800 → ~500-700 features

#### **Step 2: Boruta-SHAP**
- Creates "shadow features" (shuffled versions) and competes them against originals
- Uses SHAP values for better interaction detection
- Only keeps features that significantly outperform their randomized "evil twins"
- **Reduces**: ~500-700 → ~100-200 features

#### **Step 3: L1 Regularization (LASSO)**
- Embedded directly in XGBoost base learners (`reg_alpha` parameter)
- Automatically shrinks less-useful coefficients to zero
- Ensures final models are parsimonious and avoid overfitting

### 2. **X-Learner for Uplift Modeling**

- **Multi-treatment** handling (3 treatments vs. control)
- Estimates **Conditional Average Treatment Effects (CATE)** for each prospect
- Identifies heterogeneous treatment effects (who benefits most from which offer)
- Uses L1-regularized XGBoost as base learners

### 3. **Attrition Prediction Model**

- Binary classification for month-9 retention
- XGBoost classifier with L1 regularization
- Produces probability scores for targeting

---

## 📁 Project Structure

```
Ab_exam_practices/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                          # Centralized configuration
├── pipeline.py                        # Main orchestration script
│
├── data/                              # Generated/uploaded data
│   └── epsilon_synthetic.csv
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── data_generation.py             # Synthetic data generator
│   │
│   ├── feature_selection/             # 3-step sieve modules
│   │   ├── __init__.py
│   │   ├── step1_initial_pruning.py   # Variance + Correlation
│   │   ├── step2_boruta_shap.py       # Boruta-SHAP
│   │   └── step3_l1_regularization.py # (Embedded in models)
│   │
│   └── models/                        # Modeling modules
│       ├── __init__.py
│       ├── xlearner_uplift.py         # X-Learner for CATE
│       └── attrition_model.py         # Retention classifier
│
├── results/                           # Model outputs
│   ├── step1_pruning_report.txt
│   ├── step2_boruta_report.txt
│   ├── selected_features.txt
│   ├── cate_predictions.csv
│   ├── retention_predictions.csv
│   ├── combined_insights.csv
│   ├── uplift_curves.png
│   ├── attrition_roc_curve.png
│   ├── attrition_pr_curve.png
│   └── attrition_feature_importance.png
│
└── notebooks/                         # Optional exploration
    └── exploration.ipynb
```

---

## 🚀 Quick Start

### Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd Ab_exam_practices
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Full Pipeline

```bash
python pipeline.py
```

This will:
1. Generate synthetic Epsilon-like data (10,000 prospects, 1,800 features)
2. Run the 3-step sieve feature selection
3. Train X-Learner uplift models for each treatment
4. Train the attrition prediction model
5. Generate evaluation reports and visualizations
6. Save all results to the `results/` directory

**Expected runtime**: 5-15 minutes (depending on hardware)

---

## 📊 Key Outputs

### 1. Feature Selection Reports
- **`step1_pruning_report.txt`**: Shows which features were removed due to low variance or high correlation
- **`step2_boruta_report.txt`**: Lists selected features and their Boruta hit rates
- **`selected_features.txt`**: Final list of ~100-200 features used in models

### 2. Uplift Model Results
- **`cate_predictions.csv`**: Individual-level treatment effect estimates for each prospect
- **`uplift_curves.png`**: Cumulative gain curves showing targeting efficiency

### 3. Attrition Model Results
- **`retention_predictions.csv`**: Probability of being on-book at month 9
- **`attrition_roc_curve.png`**: ROC curve (AUC metric)
- **`attrition_pr_curve.png`**: Precision-Recall curve
- **`attrition_feature_importance.png`**: Top predictors of retention

### 4. Combined Insights
- **`combined_insights.csv`**: Master file with:
  - Treatment assignments
  - Actual outcomes
  - CATE predictions for all treatments
  - Retention probabilities
  - Customer value scores (balance × retention)

---

## 🔧 Configuration

All hyperparameters and settings are centralized in **`config.py`**:

```python
# Data generation
N_SAMPLES = 10000
N_FEATURES = 1800

# Feature selection thresholds
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95

# Boruta parameters
BORUTA_N_TRIALS = 100
BORUTA_PERCENTILE = 100

# XGBoost regularization
XGBOOST_REG_ALPHA = 1.0  # L1 penalty
XGBOOST_REG_LAMBDA = 1.0  # L2 penalty
```

Modify these values to experiment with different configurations.

---

## 📈 Example Results

### Treatment Effects on Opening Balance
```
Treatment           Mean Opening Balance
─────────────────────────────────────────
Control             $5,234
$100 Offer          $5,345  (+$111)
$400 Offer          $5,687  (+$453)
$500 Offer          $5,812  (+$578)
```

### Retention Rates by Treatment
```
Treatment           Month-9 On-Book Rate
─────────────────────────────────────────
Control             68.2%
$100 Offer          71.5%  (+3.3 pp)
$400 Offer          78.9%  (+10.7 pp)
$500 Offer          82.1%  (+13.9 pp)
```

### Feature Reduction
```
Step                          Features
──────────────────────────────────────
Original Epsilon data         1,800
After Step 1 (Pruning)          542  (70% reduction)
After Step 2 (Boruta-SHAP)      127  (93% reduction)
After Step 3 (L1 in models)      ~80  (embedded selection)
```

---

## 🧪 Testing Individual Modules

Each module can be tested independently:

```bash
# Test data generation
python src/data_generation.py

# Test Step 1 (Initial Pruning)
python src/feature_selection/step1_initial_pruning.py

# Test Step 2 (Boruta-SHAP)
python src/feature_selection/step2_boruta_shap.py
```

---

## 📚 Technical Details

### Synthetic Data Characteristics

The generated Epsilon-like dataset includes:

- **Demographics** (~200 vars): Age, income, education, occupation, household composition
- **Financial** (~300 vars): Credit scores, banking products, debt ratios, assets
- **Behavioral** (~400 vars): Purchase propensities, channel preferences, online activity
- **Geographic** (~200 vars): ZIP-level demographics, regional indicators
- **Psychographic** (~400 vars): Modeled scores, life stages, wealth segments
- **Noise** (~300 vars): Near-constant features and highly correlated duplicates

**Outcome generation**:
- **Opening Balance**: Heterogeneous treatment effects modeled with income interaction
  - Lower income → $100 offer more effective
  - Higher income → $500 offer more effective
- **Month-9 Retention**: Driven by credit score, opening balance, and treatment assignment

### Why This Approach?

1. **Variance Thresholding**: Epsilon data often has "flag fields" that are constant across 99%+ of records
2. **Correlation Filtering**: Epsilon providers duplicate variables with slight variations (e.g., "Estimated Income" vs "Estimated HH Income")
3. **Boruta-SHAP**: Superior to traditional methods for detecting complex interactions between offers and demographics
4. **L1 Regularization**: Ensures the final model doesn't overfit to noise, even with 100+ features

---

## 🛠️ Dependencies

Core packages:
- `causalml`: X-Learner implementation
- `xgboost`: Gradient boosting with regularization
- `lightgbm`: Fast tree-based models for Boruta
- `shap`: Shapley values for feature importance
- `BorutaShap`: Boruta algorithm with SHAP
- `scikit-learn`: ML utilities
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization

---

## 📖 References

- **Boruta Algorithm**: Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package. *Journal of Statistical Software*, 36(11)
- **SHAP Values**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*
- **X-Learner**: Künzel, S. R., et al. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*
- **CausalML Library**: https://causalml.readthedocs.io/

---

## 📝 License

This project is for educational and research purposes.

---

## 👤 Author

Created for bank campaign modeling and Epsilon data analysis.

---

## 🤝 Contributing

To extend this project:
1. Add train/test splitting for proper validation
2. Implement cross-validation in Boruta
3. Add Qini coefficient metrics for uplift evaluation
4. Include calibration plots for attrition model
5. Add SHAP explanations for individual predictions

---

## ⚠️ Notes

- This uses **synthetic data**. For real Epsilon data, replace `data_generation.py` with your data loader
- **Boruta-SHAP can be time-consuming** with 100 trials × 500+ features. Reduce `BORUTA_N_TRIALS` in config.py for faster testing
- The pipeline assumes **randomized treatment assignment**. For observational data, add propensity score weighting

---

**Happy Modeling! 🚀**
