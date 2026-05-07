"""
Stakeholder Validation — Uplift Decile Ranking + Attrition ROC/Confusion Matrix

Evaluates the trained CATE pipeline on a held-out test set with observed
outcomes. Two artefacts:

1. Uplift decile ranking
   - Score every row with the X-Learner; collapse multi-arm CATEs into a
     single `predicted_uplift = max(cate_treatment_*)` per row.
   - Rank ALL rows (treated + control together) by predicted_uplift descending,
     split into 10 deciles (decile 1 = highest predicted uplift).
   - Within each decile, observed lift is
        mean(actual_opening_balance | offer_amount != 0)
      − mean(actual_opening_balance | offer_amount == 0).

2. Attrition ROC + confusion matrix
   - Uses retention probabilities from the AttritionModel against the actual
     `actual_on_book9` outcome. ROC + AUC + Youden-optimal threshold + 2x2
     confusion matrix at that threshold (sklearn.metrics).

Input CSV must contain:
  - The 16 balance model features (see models/balance_feature_names.json).
  - The 15 attrition model features (see models/attrition_feature_names.json).
  - One of `offer`, `treatment`, or `offer_amount` (control = 0).
  - `actual_opening_balance` (observed continuous outcome).
  - Optional `actual_on_book9` (1 = retained at month 9, 0 = attrited).
    Attrition ROC/CM are skipped if absent.

Outputs (saved to --output_dir):
  - uplift_by_decile.png
  - attrition_roc_curve.png         (if actual_on_book9 present)
  - attrition_confusion_matrix.png  (if actual_on_book9 present)
  - decile_breakdown.csv
  - attrition_metrics.csv           (if actual_on_book9 present)
  - scored_test_set.csv

Usage:
  python src/evaluation/validate_stakeholder.py \\
      --input data/stakeholder_data.csv \\
      --output_dir results/stakeholder_validation/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.scoring.score_new_data import _align_features
from src.models.model_registry import load_pipeline
from src.utils import set_plot_style, ensure_dir


# ─────────────────────────────────────────────────────────────
# 1. Load and validate input
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n  Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")

    for col in ('offer_amount', 'offer', 'treatment'):
        if col in df.columns:
            if col != 'offer_amount':
                df = df.rename(columns={col: 'offer_amount'})
            break
    else:
        raise ValueError(
            "Input must contain an 'offer', 'treatment', or 'offer_amount' column."
        )

    if 'actual_opening_balance' not in df.columns:
        raise ValueError("Input must contain an 'actual_opening_balance' column.")

    if 'actual_on_book9' not in df.columns:
        print("  Note: 'actual_on_book9' missing — attrition evaluation will be skipped.")

    n_ctrl = int((df['offer_amount'] == 0).sum())
    n_trt  = int((df['offer_amount'] != 0).sum())
    print(f"  Control ($0): {n_ctrl:,}  |  Treated (any offer): {n_trt:,}")
    for amt in sorted(df['offer_amount'].unique()):
        if amt == 0:
            continue
        print(f"    ${int(amt)}: {int((df['offer_amount'] == amt).sum()):,} prospects")

    return df


# ─────────────────────────────────────────────────────────────
# 2. Score with X-Learner + attrition model
# ─────────────────────────────────────────────────────────────

def score(df: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    """
    Run X-Learner CATE prediction (all arms) and attrition probability prediction.
    Collapses arm-level CATEs into a single `predicted_uplift = max across arms`
    per row, attaches `retention_probability`, and returns a copy of df with
    the new columns.
    """
    pkg = load_pipeline(models_dir)
    xlearner                = pkg['xlearner']
    attrition_model         = pkg['attrition']
    balance_feature_names   = pkg['balance_feature_names']
    attrition_feature_names = pkg['attrition_feature_names']

    X_balance   = _align_features(df, balance_feature_names,   label='Balance (X-Learner)')
    X_attrition = _align_features(df, attrition_feature_names, label='Attrition')

    cates_df = xlearner.predict_all_cates(X_balance)
    cate_cols = [c for c in cates_df.columns if c.startswith('cate_treatment_')]
    if not cate_cols:
        raise RuntimeError(
            f"X-Learner returned no cate_treatment_* columns; got {list(cates_df.columns)}"
        )

    offer_col       = df['offer_amount'].values.astype(int)
    retention_proba = attrition_model.predict_proba(X_attrition, treatment=offer_col)

    scored = df.copy()
    for c in cate_cols:
        scored[c] = cates_df[c].values
    scored['predicted_uplift']     = cates_df[cate_cols].max(axis=1).values
    scored['retention_probability'] = retention_proba
    scored['is_treated']           = (offer_col != 0).astype(int)
    return scored


# ─────────────────────────────────────────────────────────────
# 3. Decile ranking
# ─────────────────────────────────────────────────────────────

def compute_deciles(scored: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    """
    Rank ALL rows by predicted_uplift descending → split into n_deciles.
    Decile 1 = highest predicted uplift. Uses rank(method='first') to break
    ties deterministically without jitter.
    """
    out = scored.copy()
    out['decile'] = pd.qcut(
        out['predicted_uplift'].rank(method='first', ascending=False),
        q=n_deciles,
        labels=range(1, n_deciles + 1),
    ).astype(int)
    return out


def decile_uplift_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-decile actual lift (treated − control) and average predicted uplift."""
    def _stats(g):
        treated_mask = g['is_treated'] == 1
        avg_t = g.loc[treated_mask, 'actual_opening_balance'].mean()
        avg_c = g.loc[~treated_mask, 'actual_opening_balance'].mean()
        return pd.Series({
            'n_treated':           int(treated_mask.sum()),
            'n_control':           int((~treated_mask).sum()),
            'avg_balance_treated': avg_t,
            'avg_balance_control': avg_c,
            'actual_lift':         avg_t - avg_c,
            'expected_lift':       g['predicted_uplift'].mean(),
        })

    table = scored.groupby('decile').apply(_stats).reset_index()

    print("\n  Decile breakdown (decile 1 = highest predicted uplift):")
    print(
        table[['decile', 'n_treated', 'n_control',
               'avg_balance_treated', 'avg_balance_control',
               'actual_lift', 'expected_lift']].to_string(index=False)
    )
    return table


# ─────────────────────────────────────────────────────────────
# 4. Uplift bar chart
# ─────────────────────────────────────────────────────────────

def plot_uplift_deciles(table: pd.DataFrame, output_dir: str) -> None:
    set_plot_style()

    deciles  = table['decile'].tolist()
    actual   = table['actual_lift'].values
    expected = table['expected_lift'].values
    n_dec    = len(deciles)

    finite = actual[~np.isnan(actual)]
    if finite.size == 0:
        print("  Warning: no deciles with actual lift to plot.")
        return

    cmap   = plt.cm.Blues
    colors = [
        '#cccccc' if np.isnan(v)
        else cmap(0.85 - 0.5 * i / max(n_dec - 1, 1))
        for i, v in enumerate(actual)
    ]
    bar_heights = np.where(np.isnan(actual), 0.0, actual)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(deciles, bar_heights, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.7, label='Actual lift (treated − control)')

    ax.plot(deciles, expected, marker='o', linewidth=2.2, color='#E63946',
            markersize=7, label='Expected lift (mean predicted_uplift)')

    mean_actual = float(np.nanmean(actual))
    ax.axhline(mean_actual, color='#1d3557', linestyle='--', linewidth=1.4,
               alpha=0.75, label=f'Mean actual lift  ${mean_actual:,.0f}')

    data_range = float(np.nanmax(actual) - np.nanmin(actual)) or abs(float(np.nanmax(actual))) or 1.0
    offset     = data_range * 0.015
    y_bottom   = min(float(np.nanmin(actual)), 0.0) - data_range * 0.14
    ax.set_ylim(bottom=y_bottom)

    for bar, row in zip(bars, table.itertuples()):
        x_mid = bar.get_x() + bar.get_width() / 2
        if np.isnan(row.actual_lift):
            ax.text(x_mid, y_bottom + data_range * 0.02, 'no data',
                    ha='center', va='bottom', fontsize=7.5,
                    color='gray', style='italic')
        else:
            h = bar.get_height()
            ax.text(x_mid, h + (offset if h >= 0 else -offset * 3),
                    f'${row.actual_lift:,.0f}',
                    ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=8.5, fontweight='bold')
        ax.text(x_mid, y_bottom,
                f'T:{row.n_treated}\nC:{row.n_control}',
                ha='center', va='top', fontsize=6.5, color='#555555')

    ax.set_xlabel('Predicted-Uplift Decile  (1 = highest, 10 = lowest)', fontsize=11)
    ax.set_ylabel('Actual Opening Balance: Treated − Control  ($)', fontsize=11)
    ax.set_title(
        'Uplift Validation by Decile\n'
        'Rank by max-arm CATE; within each decile compare observed treated vs control balances',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.set_xticks(deciles)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()

    path = os.path.join(output_dir, 'uplift_by_decile.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Uplift bar chart saved: {path}")


# ─────────────────────────────────────────────────────────────
# 5. Attrition: ROC + confusion matrix
# ─────────────────────────────────────────────────────────────

def evaluate_attrition(scored: pd.DataFrame) -> dict | None:
    """
    sklearn-based ROC curve + AUC + Youden-optimal threshold + confusion matrix.
    Returns None if `actual_on_book9` is missing or single-class.
    """
    if 'actual_on_book9' not in scored.columns:
        return None

    mask = scored['actual_on_book9'].notna() & scored['retention_probability'].notna()
    y_true = scored.loc[mask, 'actual_on_book9'].astype(int).values
    y_prob = scored.loc[mask, 'retention_probability'].values

    if np.unique(y_true).size < 2:
        print("  Warning: only one class present in actual_on_book9 — AUC undefined.")
        return None

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = float(roc_auc_score(y_true, y_prob))

    # Youden's J = TPR − FPR (skip the leading inf threshold sklearn returns)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    opt_thr  = float(thresholds[best_idx])

    preds_opt = (y_prob >= opt_thr).astype(int)
    cm = confusion_matrix(y_true, preds_opt, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]),
                      int(cm[1, 0]), int(cm[1, 1]))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f"\n  Attrition Model Evaluation  (n={int(mask.sum()):,})")
    print(f"    AUC                  : {auc:.4f}")
    print(f"    Optimal threshold    : {opt_thr:.4f}  (Youden's J)")
    print(f"    Sensitivity (recall) : {sensitivity:.4f}")
    print(f"    Specificity          : {specificity:.4f}")
    print(f"    Precision            : {precision:.4f}")
    print("    Confusion matrix @ optimal threshold:")
    print(f"      TP={tp:,}  FP={fp:,}")
    print(f"      FN={fn:,}  TN={tn:,}")

    return {
        'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds,
        'auc': auc, 'opt_thr': opt_thr,
        'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n':     int(mask.sum()),
        'n_pos': int((y_true == 1).sum()),
        'n_neg': int((y_true == 0).sum()),
    }


def plot_roc(metrics: dict, output_dir: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    fpr, tpr = metrics['fpr'], metrics['tpr']
    opt_fpr  = 1.0 - metrics['specificity']
    opt_tpr  = metrics['sensitivity']

    ax.plot(fpr, tpr, linewidth=2.5, color='#2E86AB',
            label=f'Attrition Model  (AUC = {metrics["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], '--', linewidth=1.5, color='gray',
            alpha=0.7, label='Random')
    ax.scatter([opt_fpr], [opt_tpr], s=90, zorder=5, color='#E63946',
               label=(f'Optimal threshold {metrics["opt_thr"]:.3f}\n'
                      f'Sens={opt_tpr:.3f}, Spec={metrics["specificity"]:.3f}'))

    ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=11)
    ax.set_title(
        'Attrition Model: ROC Curve\n'
        'retention_probability vs. actual_on_book9',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, 'attrition_roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Attrition ROC curve saved : {path}")


def plot_confusion_matrix_chart(metrics: dict, output_dir: str) -> None:
    set_plot_style()

    cm = np.array([[metrics['tn'], metrics['fp']],
                   [metrics['fn'], metrics['tp']]])
    labels = ['Not Open (0)', 'Open (1)']

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_title(
        f'Attrition Model: Confusion Matrix\n'
        f'Threshold = {metrics["opt_thr"]:.3f}  |  AUC = {metrics["auc"]:.4f}',
        fontsize=13, fontweight='bold', pad=12,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, 'attrition_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Attrition confusion matrix: {path}")


# ─────────────────────────────────────────────────────────────
# 6. Save CSV outputs
# ─────────────────────────────────────────────────────────────

def save_outputs(scored: pd.DataFrame,
                 table: pd.DataFrame,
                 metrics: dict | None,
                 output_dir: str) -> None:
    table.to_csv(os.path.join(output_dir, 'decile_breakdown.csv'), index=False)
    scored.to_csv(os.path.join(output_dir, 'scored_test_set.csv'), index=False)

    if metrics is not None:
        summary = {
            'AUC':             round(metrics['auc'], 4),
            'Optimal_Threshold': round(metrics['opt_thr'], 4),
            'Sensitivity':     round(metrics['sensitivity'], 4),
            'Specificity':     round(metrics['specificity'], 4),
            'Precision':       round(metrics['precision'], 4),
            'TP':              metrics['tp'],
            'FP':              metrics['fp'],
            'FN':              metrics['fn'],
            'TN':              metrics['tn'],
            'N':               metrics['n'],
            'N_Positive':      metrics['n_pos'],
            'N_Negative':      metrics['n_neg'],
        }
        pd.DataFrame([summary]).to_csv(
            os.path.join(output_dir, 'attrition_metrics.csv'), index=False
        )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate CATE pipeline on a test set with observed outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to test-set CSV.')
    parser.add_argument('--output_dir', '-o', default='results/stakeholder_validation',
                        help='Output directory for charts and CSVs.')
    parser.add_argument('--models_dir', '-m', default=None,
                        help='Model package directory (default: config.MODELS_DIR).')
    parser.add_argument('--n_deciles', type=int, default=10,
                        help='Number of deciles (default: 10).')
    return parser.parse_args()


def main():
    import config
    args = _parse_args()

    print("\n" + "=" * 60)
    print("STAKEHOLDER MODEL VALIDATION")
    print("=" * 60)

    models_dir = args.models_dir or config.MODELS_DIR
    output_dir = args.output_dir
    ensure_dir(output_dir)

    df       = load_data(args.input)
    scored   = score(df, models_dir)
    scored   = compute_deciles(scored, n_deciles=args.n_deciles)
    table    = decile_uplift_table(scored)
    plot_uplift_deciles(table, output_dir)

    metrics = evaluate_attrition(scored)
    if metrics is not None:
        plot_roc(metrics, output_dir)
        plot_confusion_matrix_chart(metrics, output_dir)

    save_outputs(scored, table, metrics, output_dir)

    print(f"\n  All outputs saved to: {output_dir}/")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
