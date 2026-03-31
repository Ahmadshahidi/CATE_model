"""
Attrition / Retention Prediction Model
========================================
Binary classifier that predicts P(prospect is on-book at month 9).

Model : CatBoostClassifier
  - Handles categorical features natively (no encoding needed)
  - Handles missing values natively (no imputation needed)
  - Robust to unseen categorical levels at inference time
Target: on_book_month9  (1 = retained, 0 = attrited)
Features: Boruta-SHAP selected features run against the ATTRITION target
          (separate from the balance-target Boruta run used by X-Learner)
          + treatment arm indicator.

Usage:
    from src.models.attrition_model import train_attrition_model
    attrition_model = train_attrition_model(X_attrition, y, treatment, save_results_dir)
    proba = attrition_model.predict_proba(X_new, treatment=treatment_new)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, roc_curve, auc,
                              precision_recall_curve)
from catboost import CatBoostClassifier, Pool

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_cat_feature_indices(X: pd.DataFrame) -> list:
    """
    Return a list of column positions that CatBoost should treat as
    categorical.  We detect them as object / category dtype columns.
    Integer-encoded categoricals (e.g. homeowner_flag) can also be
    passed; CatBoost will hash-encode them automatically.
    """
    return [i for i, col in enumerate(X.columns)
            if X[col].dtype.name in ('object', 'category')]


class AttritionModel:
    """
    CatBoost-based binary retention classifier.

    The model is trained on [X_attrition_features | treatment_indicator] so
    that individual-level retention probabilities reflect offer assignment.
    Features are selected by a Boruta-SHAP run against the attrition target
    (on_book_month9), distinct from the balance-target selection used by the
    X-Learner.

    CatBoost natively handles:
      • Categorical features — passed as cat_features indices at training time
        and stored in the model; used automatically at inference.
      • Missing values      — treated via nan_mode='Min' (see config).
      • Unseen categories   — handled by hash-based fallback (no KeyError).

    Attributes
    ----------
    model              : CatBoostClassifier (fitted)
    feature_names      : list[str]  feature columns (excluding treatment)
    cat_feature_names  : list[str]  subset of feature_names that are categorical
    threshold          : float      classification threshold (default 0.5)
    """

    def __init__(self):
        self.model             = None
        self.feature_names     = []
        self.cat_feature_names = []
        self.threshold         = 0.5

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, treatment=None,
            val_size: float = None, random_state: int = None):
        """
        Fit the CatBoostClassifier on training data.

        Parameters
        ----------
        X         : pd.DataFrame  Features (Boruta-SHAP selected, attrition target).
        y         : array-like    Binary outcome (on_book_month9).
        treatment : array-like    Treatment arm IDs (optional; appended as feature).
        val_size  : float         Validation fraction for early stopping.
        random_state : int

        Returns
        -------
        self
        """
        val_size     = val_size     or config.VALIDATION_SIZE
        random_state = random_state or config.RANDOM_SEED

        self.feature_names = X.columns.tolist()

        # Augment features with treatment indicator
        X_aug = X.copy()
        if treatment is not None:
            X_aug['treatment_indicator'] = np.array(treatment)

        y_arr = np.array(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_aug, y_arr,
            test_size    = val_size,
            stratify     = y_arr,
            random_state = random_state,
        )

        # Identify categorical feature indices (after augmentation)
        cat_indices = _get_cat_feature_indices(X_aug)
        self.cat_feature_names = [X_aug.columns[i] for i in cat_indices]

        # Build CatBoost Pools (required to pass cat_features correctly)
        train_pool = Pool(
            data         = X_tr,
            label        = y_tr,
            cat_features = cat_indices,
        )
        val_pool = Pool(
            data         = X_val,
            label        = y_val,
            cat_features = cat_indices,
        )

        params = dict(config.CATBOOST_CLASSIFIER_PARAMS)
        # Override seed if explicitly provided
        if random_state != config.RANDOM_SEED:
            params['random_seed'] = random_state

        self.model = CatBoostClassifier(**params)
        self.model.fit(
            train_pool,
            eval_set   = val_pool,
            use_best_model = True,
        )

        print(f"    CatBoost trained for "
              f"{self.model.get_best_iteration()} iterations "
              f"(best out of {params['iterations']})")
        if self.cat_feature_names:
            print(f"    Categorical features : {len(self.cat_feature_names)}")

        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame, treatment=None) -> np.ndarray:
        """
        Return P(on-book at month 9) for each prospect.

        Parameters
        ----------
        X         : pd.DataFrame  Same feature columns as used in training.
        treatment : array-like    Optional; appended as feature if present.

        Returns
        -------
        np.ndarray  shape (n,)  probabilities in [0, 1]
        """
        X_aug = X[self.feature_names].copy()
        if treatment is not None:
            X_aug['treatment_indicator'] = np.array(treatment)
        elif 'treatment_indicator' in self.model.feature_names_:
            # If trained with treatment, default to 0 (no offer) at inference
            X_aug['treatment_indicator'] = 0

        return self.model.predict_proba(X_aug)[:, 1]

    def predict(self, X: pd.DataFrame, treatment=None) -> np.ndarray:
        """Return binary predictions using self.threshold."""
        return (self.predict_proba(X, treatment) >= self.threshold).astype(int)

    # ------------------------------------------------------------------
    def evaluate(self, X: pd.DataFrame, y, treatment=None,
                 save_dir: str = None) -> dict:
        """
        Evaluate the model and optionally save diagnostic plots.

        Returns
        -------
        dict with keys: roc_auc, avg_precision, classification_report
        """
        proba = self.predict_proba(X, treatment)
        preds = (proba >= self.threshold).astype(int)

        roc_auc  = roc_auc_score(y, proba)
        avg_prec = average_precision_score(y, proba)
        cr       = classification_report(y, preds)

        print(f"\n  Attrition Model Performance:")
        print(f"    ROC-AUC        : {roc_auc:.4f}")
        print(f"    Avg Precision  : {avg_prec:.4f}")
        print(f"\n{cr}")

        if save_dir:
            self._plot_roc(y, proba, roc_auc, save_dir)
            self._plot_pr(y, proba, avg_prec, save_dir)
            self._plot_feature_importance(save_dir)

        return {
            'roc_auc':               roc_auc,
            'avg_precision':         avg_prec,
            'classification_report': cr,
        }

    # ------------------------------------------------------------------
    def _plot_roc(self, y_true, y_score, roc_auc, save_dir):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, color='#2E86AB',
                label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], '--', color='grey', label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('Attrition Model — ROC Curve',
                     fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, 'attrition_roc_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ROC curve saved to: {path}")

    def _plot_pr(self, y_true, y_score, avg_prec, save_dir):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, lw=2, color='#F18F01',
                label=f'PR (AP = {avg_prec:.3f})')
        baseline = y_true.mean() if hasattr(y_true, 'mean') else np.mean(y_true)
        ax.axhline(y=baseline, linestyle='--', color='grey',
                   label=f'Baseline ({baseline:.2f})')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Attrition Model — Precision-Recall Curve',
                     fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, 'attrition_pr_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  PR curve saved to: {path}")

    def _plot_feature_importance(self, save_dir, top_n: int = 30):
        importance  = self.model.get_feature_importance()
        feat_names  = self.model.feature_names_

        imp_df = (pd.DataFrame({'feature': feat_names, 'importance': importance})
                  .sort_values('importance', ascending=False)
                  .head(top_n)
                  .sort_values('importance', ascending=True))

        fig, ax = plt.subplots(figsize=(10, max(6, len(imp_df) * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
        ax.barh(imp_df['feature'], imp_df['importance'], color=colors)
        ax.set_xlabel('Feature Importance (CatBoost PredictionValuesChange)', fontsize=11)
        ax.set_title(f'Attrition Model — Top {top_n} Features',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        path = os.path.join(save_dir, 'attrition_feature_importance.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Feature importance saved to: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def train_attrition_model(X: pd.DataFrame,
                           y,
                           treatment=None,
                           save_results_dir: str = None) -> AttritionModel:
    """
    Train the attrition model and optionally save evaluation outputs.

    Parameters
    ----------
    X                : pd.DataFrame  Boruta-SHAP selected features
                                     (selected against the ATTRITION target,
                                      separate from the balance-target selection
                                      used by the X-Learner).
    y                : array-like    Binary retention label (on_book_month9).
    treatment        : array-like    Treatment arm IDs (optional predictor).
    save_results_dir : str           If provided, save plots + metrics here.

    Returns
    -------
    AttritionModel  (fitted)
    """
    print("\n" + "="*60)
    print("ATTRITION MODEL TRAINING  (CatBoostClassifier)")
    print("="*60)
    print(f"\n  Samples  : {len(X):,}")
    print(f"  Features : {X.shape[1]}  (attrition-target Boruta selection)")
    print(f"  Target   : on_book_month9  "
          f"(mean retention = {np.mean(y):.1%})")

    cat_cols = [c for c in X.columns if X[c].dtype.name in ('object', 'category')]
    print(f"  Categorical features in input: {len(cat_cols)}")

    model = AttritionModel()
    model.fit(X, y, treatment=treatment)

    print(f"\n  ✓ Model trained")

    # Evaluate on full dataset (in-sample diagnostic)
    metrics = model.evaluate(X, y, treatment=treatment,
                              save_dir=save_results_dir)

    if save_results_dir:
        metrics_path = os.path.join(save_results_dir, 'attrition_metrics.csv')
        pd.DataFrame([{
            'roc_auc':       metrics['roc_auc'],
            'avg_precision': metrics['avg_precision'],
        }]).to_csv(metrics_path, index=False)
        print(f"  Metrics saved to: {metrics_path}")

    print("="*60 + "\n")
    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    from src.feature_selection.step2_boruta_shap import run_boruta_shap
    df  = generate_epsilon_data()
    exc = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9', 'offer']
    X   = df[[c for c in df.columns if c not in exc]]
    X1, _ = run_initial_pruning(X, y=df['opening_balance'])
    # Use attrition target for feature selection
    X_att, _ = run_boruta_shap(X1, df['on_book_month9'], task='classification')
    model  = train_attrition_model(X_att, df['on_book_month9'],
                                    treatment=df['treatment'],
                                    save_results_dir=config.RESULTS_DIR)
    proba  = model.predict_proba(X_att, treatment=df['treatment'])
    print(f"\nRetention proba: mean={proba.mean():.3f}, std={proba.std():.3f}")
