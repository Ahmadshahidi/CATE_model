"""
Shared utility helpers for the Uplift Modeling pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ──────────────────────────────────────────────────────────────────────────────
# General helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory if it does not exist; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def set_plot_style():
    """Apply a clean, consistent matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   'white',
        'axes.grid':        True,
        'grid.alpha':       0.3,
        'grid.linestyle':   '--',
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'font.size':        10,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
    })


def format_dollar(x, pos=None):
    """Matplotlib formatter: $1,234."""
    return f'${x:,.0f}'


def format_millions(x, pos=None):
    """Matplotlib formatter: $1.2M."""
    return f'${x/1e6:.1f}M'


# ──────────────────────────────────────────────────────────────────────────────
# DataFrame helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_df_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print shape, dtypes summary, and first few rows."""
    print(f"\n{'─'*60}")
    print(f"{name}  —  shape: {df.shape}")
    print(f"{'─'*60}")
    print(f"  dtype counts: {dict(df.dtypes.value_counts())}")
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"  Nulls: {null_counts[null_counts > 0].to_dict()}")
    else:
        print("  No nulls.")
    print(f"{'─'*60}\n")


def split_features_outcomes(df: pd.DataFrame, outcome_cols: list,
                              extra_exclude: list = None):
    """
    Split a DataFrame into features X and a dict of outcome Series.

    Parameters
    ----------
    df : pd.DataFrame
    outcome_cols : list[str]   Columns to treat as outcomes.
    extra_exclude : list[str]  Additional columns to drop from X.

    Returns
    -------
    X : pd.DataFrame
    outcomes : dict[str → pd.Series]
    """
    exclude = set(outcome_cols) | set(extra_exclude or [])
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()
    outcomes = {c: df[c] for c in outcome_cols if c in df.columns}
    return X, outcomes


# ──────────────────────────────────────────────────────────────────────────────
# Train / test helpers
# ──────────────────────────────────────────────────────────────────────────────

def stratified_treatment_split(X: pd.DataFrame,
                                treatment: pd.Series,
                                test_size: float = 0.3,
                                random_state: int = 42):
    """
    Train/test split preserving treatment arm proportions.

    Returns
    -------
    X_train, X_test, t_train, t_test  (all pd.DataFrame / pd.Series)
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, treatment,
        test_size    = test_size,
        stratify     = treatment,
        random_state = random_state,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Uplift evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def qini_coefficient(y_true: np.ndarray,
                     uplift_score: np.ndarray,
                     treatment: np.ndarray) -> float:
    """
    Compute the Qini coefficient for a single treatment arm vs. control.

    Parameters
    ----------
    y_true      : binary or continuous outcome
    uplift_score: predicted CATE / uplift for this arm
    treatment   : 1 = treated, 0 = control

    Returns
    -------
    float — Qini coefficient (area between Qini and random curves,
            normalised by the perfect-model area)
    """
    df = pd.DataFrame({
        'y': y_true, 'score': uplift_score, 't': treatment
    }).sort_values('score', ascending=False).reset_index(drop=True)

    n_treated = df['t'].sum()
    n_control = len(df) - n_treated

    qini_vals, random_vals = [0.0], [0.0]
    n_obs = len(df)
    cum_treated = 0
    cum_control = 0
    prev_rand = 0.0

    for i, row in df.iterrows():
        if row['t'] == 1:
            cum_treated += row['y']
        else:
            cum_control += row['y']

        pct = (i + 1) / n_obs
        qini_vals.append(cum_treated - cum_control * (n_treated / n_control
                                                       if n_control > 0 else 0))
        prev_rand += (n_treated / n_obs) * (df['y'].mean())
        random_vals.append(prev_rand)

    q_arr = np.array(qini_vals)
    r_arr = np.array(random_vals)
    return float(np.trapz(q_arr - r_arr) / (len(qini_vals) - 1))


# ──────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame,
                             title: str = "Feature Importance",
                             top_n: int = 30,
                             save_path: str = None):
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame  with columns 'feature' and 'importance'
    title         : chart title
    top_n         : restrict to top-N features
    save_path     : if provided, save to this path
    """
    df = (importance_df
          .sort_values('importance', ascending=False)
          .head(top_n)
          .sort_values('importance', ascending=True))

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df['feature'], df['importance'], color=colors)

    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_pr_curves(y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       model_name: str = "Model",
                       save_dir: str = None):
    """Plot and optionally save ROC and Precision-Recall curves."""
    from sklearn.metrics import (roc_curve, auc,
                                  precision_recall_curve, average_precision_score)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc     = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap                   = average_precision_score(y_true, y_pred_proba)

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color='#2E86AB', label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} — ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(); plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'attrition_roc_curve.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    # PR curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, color='#F18F01',
            label=f'PR (AP = {ap:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'{model_name} — Precision-Recall Curve',
                 fontsize=13, fontweight='bold')
    ax.legend(); plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'attrition_pr_curve.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()

    return roc_auc, ap
