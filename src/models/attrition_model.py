"""
Attrition / Retention Prediction Model
========================================
Binary classifier that predicts P(prospect is on-book at month 9).

Model : XGBClassifier with L1 regularisation (Step 3 of the 3-step sieve).
Target: on_book_month9  (1 = retained, 0 = attrited)
Features: Boruta-SHAP selected features + treatment arm indicator.

Usage:
    from src.models.attrition_model import train_attrition_model
    attrition_model = train_attrition_model(X, y, treatment, save_results_dir)
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
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config


class AttritionModel:
    """
    XGBoost-based binary retention classifier.

    The model is trained on [X_features | treatment_indicator] so that
    individual-level retention probabilities reflect offer assignment.

    Attributes
    ----------
    model         : XGBClassifier (fitted)
    feature_names : list[str]  feature columns (excluding treatment)
    threshold     : float      classification threshold (default 0.5)
    """

    def __init__(self):
        self.model         = None
        self.feature_names = []
        self.threshold     = 0.5

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y, treatment=None,
            val_size: float = None, random_state: int = None):
        """
        Fit the XGBClassifier on training data.

        Parameters
        ----------
        X         : pd.DataFrame  Features (Boruta-SHAP selected).
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
            X_aug = X_aug.copy()
            X_aug['treatment_indicator'] = np.array(treatment)

        y_arr = np.array(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_aug, y_arr,
            test_size    = val_size,
            stratify     = y_arr,
            random_state = random_state,
        )

        xgb_params = {k: v for k, v in config.XGBOOST_PARAMS.items()
                      if k != 'objective'}
        self.model = XGBClassifier(
            objective    = 'binary:logistic',
            eval_metric  = 'auc',
            use_label_encoder = False,
            **xgb_params,
        )
        self.model.fit(
            X_tr, y_tr,
            eval_set              = [(X_val, y_val)],
            verbose               = False,
        )

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
            X_aug = X_aug.copy()
            X_aug['treatment_indicator'] = np.array(treatment)
        elif 'treatment_indicator' in self.model.feature_names_in_:
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
        importance = self.model.feature_importances_
        feat_names = self.model.feature_names_in_

        imp_df = (pd.DataFrame({'feature': feat_names, 'importance': importance})
                  .sort_values('importance', ascending=False)
                  .head(top_n)
                  .sort_values('importance', ascending=True))

        fig, ax = plt.subplots(figsize=(10, max(6, len(imp_df) * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
        ax.barh(imp_df['feature'], imp_df['importance'], color=colors)
        ax.set_xlabel('Feature Importance (XGBoost gain)', fontsize=11)
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
    X                : pd.DataFrame  Boruta-SHAP selected features.
    y                : array-like    Binary retention label (on_book_month9).
    treatment        : array-like    Treatment arm IDs (optional predictor).
    save_results_dir : str           If provided, save plots + metrics here.

    Returns
    -------
    AttritionModel  (fitted)
    """
    print("\n" + "="*60)
    print("ATTRITION MODEL TRAINING")
    print("="*60)
    print(f"\n  Samples  : {len(X):,}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Target   : on_book_month9  "
          f"(mean retention = {np.mean(y):.1%})")

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
    X2, _ = run_boruta_shap(X1, df['opening_balance'])
    model  = train_attrition_model(X2, df['on_book_month9'],
                                    treatment=df['treatment'],
                                    save_results_dir=config.RESULTS_DIR)
    proba  = model.predict_proba(X2, treatment=df['treatment'])
    print(f"\nRetention proba: mean={proba.mean():.3f}, std={proba.std():.3f}")
