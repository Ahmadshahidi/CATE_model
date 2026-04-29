"""
Standalone Uplift Evaluation Script
=====================================
Reads a combined_insights CSV (output of pipeline.py) and generates:
  - Uplift curves per arm
  - AUUC bar chart per arm
  - Qini curves per arm
  - Cumulative gain chart
  - Metrics CSV (AUUC + Qini coefficient per arm)

Reuses:
  XLearnerUplift._compute_auuc_arm()  — static AUUC computation (no model needed)
  qini_coefficient()                  — Qini scalar from src/utils.py
  set_plot_style()                    — consistent plot styling
  ensure_dir()                        — directory creation

Usage
-----
  # Evaluate the held-out test split
  python src/evaluation/evaluate_uplift.py \\
      --input      results/combined_insights.csv \\
      --output_dir results/eval_test \\
      --split      test \\
      --label      _test

  # Evaluate training split (in-sample diagnostic)
  python src/evaluation/evaluate_uplift.py \\
      --input      results/combined_insights.csv \\
      --output_dir results/eval_train \\
      --split      train \\
      --label      _train

  # Evaluate any CSV (no split column required)
  python src/evaluation/evaluate_uplift.py \\
      --input      results/combined_insights_test.csv \\
      --output_dir results/eval_test
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import config
from src.models.xlearner_uplift import XLearnerUplift
from src.utils import qini_coefficient, set_plot_style, ensure_dir


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_uplift_from_csv(
    df: pd.DataFrame,
    output_dir: str,
    label: str = '',
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df         : DataFrame containing cate_treatment_{arm_id} columns,
                 opening_balance_actual (outcome), and treatment (arm ID).
    output_dir : Directory where all PNGs and the metrics CSV are saved.
    label      : Suffix appended to every output filename, e.g. '_test'.

    Returns
    -------
    metrics_df : one row per arm — AUUC and Qini metrics.
    """
    ensure_dir(output_dir)
    set_plot_style()

    arm_ids   = sorted(k for k in config.TREATMENT_COMPONENTS if k != 0)
    y_arr     = df['opening_balance_actual'].values.astype(float)
    t_arr     = df['treatment'].values.astype(int)
    ctrl_mask = (t_arr == 0)
    cmap      = plt.get_cmap('tab10')

    rows = []

    for arm_id in arm_ids:
        cate_col = f'cate_treatment_{arm_id}'
        if cate_col not in df.columns:
            print(f"  ⚠  Column '{cate_col}' not found — skipping arm {arm_id}")
            continue

        arm_mask = (t_arr == arm_id)
        sub_mask = arm_mask | ctrl_mask

        cate_sub = df.loc[sub_mask, cate_col].values
        y_sub    = y_arr[sub_mask]
        t_sub    = arm_mask[sub_mask].astype(int)

        # AUUC — reuses existing @staticmethod, no model instance needed
        auuc, auuc_rand, auuc_lift = XLearnerUplift._compute_auuc_arm(
            y_sub, cate_sub, t_sub)

        # Qini coefficient
        qini_coef = qini_coefficient(y_sub, cate_sub, t_sub)

        rows.append({
            'arm_id':    arm_id,
            'arm_name':  config.TREATMENTS[arm_id],
            'n_rows':    int(sub_mask.sum()),
            'n_treated': int(t_sub.sum()),
            'n_control': int((1 - t_sub).sum()),
            'auuc':      auuc,
            'auuc_rand': auuc_rand,
            'auuc_lift': auuc_lift,
            'auuc_norm': auuc_lift / auuc_rand if auuc_rand else 0.0,
            'qini_coef': qini_coef,
        })

    if not rows:
        print("  ⚠  No arm data found. Check column names in the input CSV.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(rows)

    # Print summary
    print(f"\n  AUUC / Qini Summary{' (' + label.strip('_') + ')' if label else ''}:")
    for _, r in metrics_df.iterrows():
        print(f"    {r['arm_name']:<10}  AUUC lift={r['auuc_lift']:+,.0f}  "
              f"auuc_norm={r['auuc_norm']:.3f}  Qini={r['qini_coef']:.4f}")

    # Save metrics CSV
    metrics_path = os.path.join(output_dir, f'eval_metrics{label}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved to: {metrics_path}")

    # Generate charts
    _plot_uplift_curves(df, arm_ids, y_arr, t_arr, cmap, output_dir, label)
    _plot_auuc_bar(metrics_df, output_dir, label)
    _plot_qini_curves(df, arm_ids, y_arr, t_arr, cmap, output_dir, label)
    _plot_cumulative_gain(df, arm_ids, y_arr, t_arr, cmap, output_dir, label)

    return metrics_df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plot_uplift_curves(df, arm_ids, y_arr, t_arr, cmap, output_dir, label):
    """Uplift curves: mean(treated) - mean(control) at each % targeted."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ctrl_mask = (t_arr == 0)

    for i, arm_id in enumerate(arm_ids):
        cate_col = f'cate_treatment_{arm_id}'
        if cate_col not in df.columns:
            continue

        arm_mask = (t_arr == arm_id)
        sub_mask = arm_mask | ctrl_mask
        cate_sub = df.loc[sub_mask, cate_col].values
        y_sub    = y_arr[sub_mask]
        t_sub    = arm_mask[sub_mask].astype(int)

        order = np.argsort(-cate_sub)
        n     = len(order)
        pcts, gains = [], []
        cum_t = cum_c = cum_tn = cum_cn = 0.0

        for j, idx in enumerate(order):
            if t_sub[idx] == 1:
                cum_t += y_sub[idx]; cum_tn += 1
            else:
                cum_c += y_sub[idx]; cum_cn += 1
            pcts.append(100 * (j + 1) / n)
            gains.append((cum_t / cum_tn if cum_tn else 0) -
                         (cum_c / cum_cn if cum_cn else 0))

        ax.plot(pcts, gains, color=cmap(i), lw=2, label=config.TREATMENTS[arm_id])

    ax.axhline(0, color='black', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlabel('% of Population Targeted (by CATE score)', fontsize=11)
    ax.set_ylabel('Mean Treated − Mean Control (Uplift)', fontsize=11)
    title_suffix = f' — {label.strip("_").title()}' if label else ''
    ax.set_title(f'Uplift Curves per Arm{title_suffix}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'uplift_curves{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_auuc_bar(metrics_df, output_dir, label):
    """Horizontal bar chart of AUUC lift per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    arm_names = metrics_df['arm_name'].tolist()
    y_pos     = np.arange(len(arm_names))
    colors    = plt.get_cmap('tab10')(np.linspace(0, 0.9, len(arm_names)))

    # Panel 1: absolute AUUC lift
    ax = axes[0]
    bars = ax.barh(y_pos, metrics_df['auuc_lift'], color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(arm_names, fontsize=10)
    ax.set_xlabel('AUUC Lift (vs. Random)', fontsize=11)
    ax.set_title('AUUC Lift by Arm', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, metrics_df['auuc_lift']):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f'  {val:+,.0f}', va='center', fontsize=9)

    # Panel 2: normalised AUUC
    ax = axes[1]
    bars2 = ax.barh(y_pos, metrics_df['auuc_norm'], color=colors)
    ax.set_yticks(y_pos); ax.set_yticklabels(arm_names, fontsize=10)
    ax.set_xlabel('Normalised AUUC (lift / random)', fontsize=11)
    ax.set_title('Normalised AUUC by Arm', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars2, metrics_df['auuc_norm']):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f'  {val:.3f}', va='center', fontsize=9)

    title_suffix = f' — {label.strip("_").title()}' if label else ''
    fig.suptitle(f'AUUC Comparison{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'auuc_comparison{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_qini_curves(df, arm_ids, y_arr, t_arr, cmap, output_dir, label):
    """Qini curves: cumulative treated gain adjusted for control fraction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ctrl_mask = (t_arr == 0)

    for i, arm_id in enumerate(arm_ids):
        cate_col = f'cate_treatment_{arm_id}'
        if cate_col not in df.columns:
            continue

        arm_mask  = (t_arr == arm_id)
        sub_mask  = arm_mask | ctrl_mask
        cate_sub  = df.loc[sub_mask, cate_col].values
        y_sub     = y_arr[sub_mask]
        t_sub     = arm_mask[sub_mask].astype(int)
        n         = len(t_sub)
        n_t       = t_sub.sum()
        n_c       = n - n_t

        order  = np.argsort(-cate_sub)
        pcts   = [0.0]
        qvals  = [0.0]
        cum_t  = cum_c = 0.0

        for j, idx in enumerate(order):
            if t_sub[idx] == 1:
                cum_t += y_sub[idx]
            else:
                cum_c += y_sub[idx]
            pcts.append(100 * (j + 1) / n)
            qvals.append(cum_t - cum_c * (n_t / n_c if n_c > 0 else 0))

        rand_end = (y_sub[t_sub == 1].mean() * n_t
                    - y_sub[t_sub == 0].mean() * n_t
                    if n_t and n_c else 0)
        rand_line = [0, rand_end]
        rand_pcts = [0, 100]

        color = cmap(i)
        ax.plot(pcts, qvals, color=color, lw=2, label=config.TREATMENTS[arm_id])
        ax.plot(rand_pcts, rand_line, color=color, lw=1, linestyle='--', alpha=0.5)

    ax.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax.set_xlabel('% of Population Targeted (by CATE score)', fontsize=11)
    ax.set_ylabel('Cumulative Qini Value', fontsize=11)
    title_suffix = f' — {label.strip("_").title()}' if label else ''
    ax.set_title(f'Qini Curves per Arm{title_suffix}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'qini_curves{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def _plot_cumulative_gain(df, arm_ids, y_arr, t_arr, cmap, output_dir, label):
    """Cumulative gain: fraction of total treated outcome captured at each %."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ctrl_mask = (t_arr == 0)

    for i, arm_id in enumerate(arm_ids):
        cate_col = f'cate_treatment_{arm_id}'
        if cate_col not in df.columns:
            continue

        arm_mask = (t_arr == arm_id)
        sub_mask = arm_mask | ctrl_mask
        cate_sub = df.loc[sub_mask, cate_col].values
        y_sub    = y_arr[sub_mask]
        t_sub    = arm_mask[sub_mask].astype(int)

        treated_y   = y_sub[t_sub == 1]
        total_treat = treated_y.sum() if treated_y.sum() != 0 else 1.0

        order = np.argsort(-cate_sub)
        n     = len(order)
        pcts  = [0.0]
        gains = [0.0]
        cum   = 0.0

        for j, idx in enumerate(order):
            if t_sub[idx] == 1:
                cum += y_sub[idx]
            pcts.append(100 * (j + 1) / n)
            gains.append(cum / total_treat)

        ax.plot(pcts, gains, color=cmap(i), lw=2, label=config.TREATMENTS[arm_id])

    ax.plot([0, 100], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlabel('% of Population Targeted (by CATE score)', fontsize=11)
    ax.set_ylabel('Fraction of Total Treated Outcome Captured', fontsize=11)
    title_suffix = f' — {label.strip("_").title()}' if label else ''
    ax.set_title(f'Cumulative Gain per Arm{title_suffix}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'cumulative_gain{label}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate uplift evaluation charts and metrics from a combined_insights CSV.')
    p.add_argument('--input',      required=True,
                   help='Path to combined_insights CSV.')
    p.add_argument('--output_dir', required=True,
                   help='Directory where charts and metrics CSV are saved.')
    p.add_argument('--split',      default=None,
                   help="Filter rows where the 'split' column equals this value "
                        "(e.g. 'test' or 'train'). Omit to use all rows.")
    p.add_argument('--label',      default='',
                   help="Suffix appended to output filenames (e.g. '_test').")
    return p.parse_args()


def main():
    args = _parse_args()
    df   = pd.read_csv(args.input)
    print(f"\nLoaded {len(df):,} rows from {args.input}")

    if args.split and 'split' in df.columns:
        df = df[df['split'] == args.split].copy()
        print(f"  Filtered to split='{args.split}': {len(df):,} rows")
    elif args.split and 'split' not in df.columns:
        print(f"  ⚠  --split '{args.split}' requested but no 'split' column found — "
              f"using all rows.")

    evaluate_uplift_from_csv(df, args.output_dir, label=args.label)
    print(f"\nEvaluation complete. Outputs in: {args.output_dir}")


if __name__ == '__main__':
    main()
