"""
Stakeholder Proof Chart — Validate CATE Model on Known Treatment/Outcome Data

Demonstrates that the model's personalized net value strategy correctly identifies
prospects who respond most to treatment. Uses net_value_gain_vs_ctrl (the
production ranking score) as the model signal, then compares it against actual
observed opening balances.

Input CSV must contain:
  - The 16 balance model features (see models/balance_feature_names.json) — required
    for CATE prediction. Missing feature columns are NaN-filled, which will produce
    invalid scores. The 6 categorical features (risk_tier, region, occupation_category,
    channel_preference, life_stage_category, stipulation) and remail must be present.
  - The 15 attrition model features (see models/attrition_feature_names.json) — required
    for retention probability. Same caveat applies.
  - 'offer', 'treatment', or 'offer_amount' column — control must be 0; treated arms
    can be any dollar amounts within the trained model's range [$100–$1000]. Amounts
    not in the trained set are automatically interpolated.
  - 'opening_balance' column (actual observed outcome)

Outputs (saved to --output_dir):
  - decile_validation.png  : observed lift by model-score decile (main proof chart)
  - uplift_curve.png       : aggregated Qini curve vs random baseline
  - proof_metrics.csv      : summary table of key validation metrics

Usage:
  python src/evaluation/validate_stakeholder.py \\
      --input data/stakeholder_data.csv \\
      --output_dir results/stakeholder_validation/

  # Override detected offers (e.g. if the column has noise or you want a subset):
  python src/evaluation/validate_stakeholder.py \\
      --input data/stakeholder_data.csv \\
      --offers 400 750 1000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.scoring.score_new_data import score_prospects
from src.utils import set_plot_style, ensure_dir

MIN_GROUP_SIZE = 5  # minimum treated or control per decile to plot


# ─────────────────────────────────────────────────────────────
# 1. Load and validate input
# ─────────────────────────────────────────────────────────────

def load_data(path: str, offer_override: list = None) -> tuple:
    """
    Returns (df, campaign_offers) where campaign_offers is the list of
    non-zero offer amounts to pass to the model (auto-detected or overridden).
    """
    df = pd.read_csv(path)
    print(f"\n  Loaded {len(df):,} rows × {df.shape[1]} columns from {path}")

    for col in ('offer_amount', 'offer', 'treatment'):
        if col in df.columns:
            if col != 'offer_amount':
                df = df.rename(columns={col: 'offer_amount'})
            break
    else:
        raise ValueError("Input must contain an 'offer', 'treatment', or 'offer_amount' column.")

    if 'opening_balance' not in df.columns:
        raise ValueError("Input must contain an 'opening_balance' column.")

    if 'open_on_9m' not in df.columns:
        print("  Note: 'open_on_9m' column not found — attrition evaluation will be skipped.")

    # Warn if key model features are missing — predictions will be invalid without them
    import json, os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    for fname, label in [('balance_feature_names.json', 'Balance (X-Learner)'),
                          ('attrition_feature_names.json', 'Attrition')]:
        fpath = _os.path.join(_root, 'models', fname)
        if _os.path.exists(fpath):
            with open(fpath) as f:
                required = json.load(f)
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  WARNING: {len(missing)} {label} feature(s) missing from input — "
                      f"predictions will be degraded. First 5: {missing[:5]}")

    # Detect offer amounts from data (or use override)
    if offer_override:
        campaign_offers = sorted(offer_override)
        print(f"  Offer amounts (overridden): {campaign_offers}")
    else:
        campaign_offers = sorted(int(v) for v in df['offer_amount'].unique() if v > 0)
        print(f"  Offer amounts (auto-detected): {campaign_offers}")

    n_ctrl = (df['offer_amount'] == 0).sum()
    print(f"  Control ($0): {n_ctrl:,}  |  Treated:")
    for amt in campaign_offers:
        n = (df['offer_amount'] == amt).sum()
        print(f"    ${amt}: {n:,} prospects")

    return df, campaign_offers


# ─────────────────────────────────────────────────────────────
# 2. Score with model and compute ranking score
# ─────────────────────────────────────────────────────────────

def run_model_scoring(df: pd.DataFrame, models_dir: str,
                      campaign_offers: list = None) -> pd.DataFrame:
    # Always include control (0) in the campaign offers passed to the scorer
    offers = sorted({0} | set(campaign_offers or []))
    scored = score_prospects(
        df_raw=df,
        models_dir=models_dir,
        campaign_offers=offers,
    )

    # Ranking score: net_value_gain_vs_ctrl = optimal_NV - control_NV
    # Note: actual opening_balance cancels in the difference, so this is
    # purely model-driven (p_ret * CATE - offer_cost).
    scored['net_value_gain_vs_ctrl'] = (
        scored['optimal_net_value'] - scored['net_value_offer_0']
    )

    scored['actual_offer']   = df['offer_amount'].values
    scored['actual_balance'] = df['opening_balance'].values
    scored['is_treated']     = (scored['actual_offer'] > 0).astype(int)

    if 'open_on_9m' in df.columns:
        scored['actual_open_on_9m'] = df['open_on_9m'].values

    return scored


# ─────────────────────────────────────────────────────────────
# 3. Decile validation
# ─────────────────────────────────────────────────────────────

def compute_decile_validation(scored: pd.DataFrame,
                               n_deciles: int = 10) -> tuple:
    scored = scored.copy()
    scored['nv_decile'] = pd.qcut(
        scored['net_value_gain_vs_ctrl'],
        q=n_deciles,
        labels=list(range(1, n_deciles + 1)),
        duplicates='drop',
    ).astype(int)

    is_treated = scored['is_treated'] == 1
    rows = []
    for d in range(1, n_deciles + 1):
        mask_d = scored['nv_decile'] == d
        t_mask = mask_d & is_treated
        c_mask = mask_d & ~is_treated

        n_t = int(t_mask.sum())
        n_c = int(c_mask.sum())
        avg_t = scored.loc[t_mask, 'actual_balance'].mean() if n_t > 0 else np.nan
        avg_c = scored.loc[c_mask, 'actual_balance'].mean() if n_c > 0 else np.nan

        rows.append({
            'decile':              d,
            'n_treated':           n_t,
            'n_control':           n_c,
            'avg_balance_treated': avg_t,
            'avg_balance_control': avg_c,
            'observed_lift':       avg_t - avg_c if (n_t >= MIN_GROUP_SIZE and n_c >= MIN_GROUP_SIZE) else np.nan,
            'avg_nv_gain':         scored.loc[mask_d, 'net_value_gain_vs_ctrl'].mean(),
        })

    return scored, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 4. Qini curve computation
# ─────────────────────────────────────────────────────────────

def compute_qini_curve(scored: pd.DataFrame) -> dict:
    """Aggregate Qini: treated (any offer) vs control, ranked by model score."""
    df = (scored[['net_value_gain_vs_ctrl', 'is_treated', 'actual_balance']]
          .sort_values('net_value_gain_vs_ctrl', ascending=False)
          .reset_index(drop=True))

    n_total   = len(df)
    n_treated = int(df['is_treated'].sum())
    n_control = n_total - n_treated
    ratio     = n_treated / n_control if n_control > 0 else 1.0

    pcts, qini_vals = [0.0], [0.0]
    cum_t = 0.0
    cum_c = 0.0

    for i, row in df.iterrows():
        if row['is_treated']:
            cum_t += row['actual_balance']
        else:
            cum_c += row['actual_balance']
        pcts.append(100.0 * (i + 1) / n_total)
        qini_vals.append(cum_t - cum_c * ratio)

    max_qini    = qini_vals[-1]
    random_vals = [max_qini * (p / 100.0) for p in pcts]

    pcts_arr = np.array(pcts)
    q_arr    = np.array(qini_vals)
    r_arr    = np.array(random_vals)
    auuc_lift = float(np.trapz(q_arr - r_arr, pcts_arr))

    return {
        'percentiles': pcts,
        'qini':        qini_vals,
        'random':      random_vals,
        'auuc_lift':   auuc_lift,
        'n_treated':   n_treated,
        'n_control':   n_control,
    }


# ─────────────────────────────────────────────────────────────
# 5. Attrition model evaluation
# ─────────────────────────────────────────────────────────────

def compute_attrition_metrics(scored: pd.DataFrame) -> dict | None:
    """
    Evaluates retention_probability against actual open_on_9m (0/1).
    Returns a dict with ROC curve arrays, AUC, optimal threshold, and
    confusion matrix counts at that threshold. Returns None if the
    required columns are absent.
    """
    if 'actual_open_on_9m' not in scored.columns or 'retention_probability' not in scored.columns:
        return None

    mask = scored['actual_open_on_9m'].notna() & scored['retention_probability'].notna()
    y_true = scored.loc[mask, 'actual_open_on_9m'].astype(int).values
    y_prob = scored.loc[mask, 'retention_probability'].values

    if len(np.unique(y_true)) < 2:
        print("  Warning: only one class present in open_on_9m — AUC undefined.")
        return None

    # ROC curve via sorted thresholds
    thresholds = np.concatenate(([1.0], np.sort(np.unique(y_prob))[::-1]))
    tpr_list, fpr_list = [0.0], [0.0]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    for thr in thresholds[1:]:
        preds = (y_prob >= thr).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    # Append (1,1) to close the curve
    tpr_arr = np.append(tpr_arr, 1.0)
    fpr_arr = np.append(fpr_arr, 1.0)
    thresholds = np.append(thresholds, 0.0)

    auc = float(np.trapz(tpr_arr, fpr_arr))
    # trapz may be negative if fpr is decreasing; take abs
    if auc < 0:
        auc = -auc

    # Optimal threshold via Youden's J (max TPR - FPR)
    j = tpr_arr - fpr_arr
    best_idx = int(np.argmax(j))
    opt_thr = float(thresholds[best_idx])

    preds_opt = (y_prob >= opt_thr).astype(int)
    tp = int(((preds_opt == 1) & (y_true == 1)).sum())
    fp = int(((preds_opt == 1) & (y_true == 0)).sum())
    tn = int(((preds_opt == 0) & (y_true == 0)).sum())
    fn = int(((preds_opt == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f"\n  Attrition Model Evaluation  (n={mask.sum():,})")
    print(f"    AUC                     : {auc:.4f}")
    print(f"    Optimal threshold       : {opt_thr:.4f}  (Youden's J)")
    print(f"    Sensitivity (recall)    : {sensitivity:.4f}")
    print(f"    Specificity             : {specificity:.4f}")
    print(f"    Precision               : {precision:.4f}")
    print(f"    Confusion matrix (opt threshold):")
    print(f"      TP={tp:,}  FP={fp:,}")
    print(f"      FN={fn:,}  TN={tn:,}")

    return {
        'fpr':         fpr_arr,
        'tpr':         tpr_arr,
        'thresholds':  thresholds,
        'auc':         auc,
        'opt_thr':     opt_thr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n':           int(mask.sum()),
        'n_pos':       int(n_pos),
        'n_neg':       int(n_neg),
    }


def plot_attrition_roc(attrition_metrics: dict, output_dir: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    fpr = attrition_metrics['fpr']
    tpr = attrition_metrics['tpr']
    auc = attrition_metrics['auc']
    opt_fpr = 1 - attrition_metrics['specificity']
    opt_tpr = attrition_metrics['sensitivity']

    ax.plot(fpr, tpr, linewidth=2.5, color='#2E86AB',
            label=f'Attrition Model  (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], '--', linewidth=1.5, color='gray', alpha=0.7, label='Random')
    ax.scatter([opt_fpr], [opt_tpr], s=90, zorder=5, color='#E63946',
               label=f'Optimal threshold {attrition_metrics["opt_thr"]:.3f}\n'
                     f'Sens={opt_tpr:.3f}, Spec={attrition_metrics["specificity"]:.3f}')

    ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=11)
    ax.set_title(
        'Attrition Model: ROC Curve\n'
        'Predicted retention_probability vs. actual open_on_9m',
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


def plot_attrition_confusion_matrix(attrition_metrics: dict, output_dir: str) -> None:
    set_plot_style()
    tp = attrition_metrics['tp']
    fp = attrition_metrics['fp']
    fn = attrition_metrics['fn']
    tn = attrition_metrics['tn']

    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['Not Open (0)', 'Open (1)']

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
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
        f'Threshold = {attrition_metrics["opt_thr"]:.3f}  |  '
        f'AUC = {attrition_metrics["auc"]:.4f}',
        fontsize=13, fontweight='bold', pad=12,
    )
    plt.tight_layout()

    path = os.path.join(output_dir, 'attrition_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Attrition confusion matrix: {path}")


# ─────────────────────────────────────────────────────────────
# 6. Charts
# ─────────────────────────────────────────────────────────────

def plot_decile_validation(decile_df: pd.DataFrame, output_dir: str) -> None:
    set_plot_style()

    valid = decile_df.dropna(subset=['observed_lift'])
    if valid.empty:
        print("  Warning: no deciles with sufficient treated+control to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    deciles = decile_df['decile'].tolist()
    lifts   = decile_df['observed_lift'].tolist()
    n_dec   = len(deciles)

    # Gradient colors: light blue → dark blue
    cmap   = plt.cm.Blues
    colors = [cmap(0.3 + 0.55 * (d - 1) / max(n_dec - 1, 1)) for d in deciles]

    bars = ax.bar(deciles, lifts, color=colors, edgecolor='white', linewidth=0.8,
                  width=0.7)

    # Population mean lift as reference line
    all_treated = decile_df['avg_balance_treated'] * decile_df['n_treated']
    all_control = decile_df['avg_balance_control'] * decile_df['n_control']
    pop_treated_mean = all_treated.sum() / decile_df['n_treated'].sum()
    pop_control_mean = all_control.sum() / decile_df['n_control'].sum()
    pop_mean_lift    = pop_treated_mean - pop_control_mean

    ax.axhline(pop_mean_lift, color='#E63946', linestyle='--', linewidth=1.8,
               label=f'Population avg lift  ${pop_mean_lift:,.0f}')

    # Compute a fixed offset from data range so annotations don't rely on live ylim
    finite_lifts = [v for v in lifts if v is not None and not (isinstance(v, float) and np.isnan(v))]
    data_range   = max(finite_lifts) - min(finite_lifts) if len(finite_lifts) > 1 else abs(finite_lifts[0]) if finite_lifts else 1
    offset       = data_range * 0.015
    y_bottom     = min(min(finite_lifts, default=0), 0) - data_range * 0.12

    ax.set_ylim(bottom=y_bottom)

    # Annotate bars with observed lift and sample sizes
    for bar, row in zip(bars, decile_df.itertuples()):
        x_mid = bar.get_x() + bar.get_width() / 2
        if np.isnan(row.observed_lift):
            ax.text(x_mid, y_bottom + data_range * 0.02,
                    f'n<{MIN_GROUP_SIZE}', ha='center', va='bottom',
                    fontsize=7.5, color='gray', style='italic')
        else:
            h = bar.get_height()
            ax.text(x_mid,
                    h + (offset if h >= 0 else -offset * 3),
                    f'${row.observed_lift:,.0f}',
                    ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=8.5, fontweight='bold')
        # Sample size footnote below the x-axis
        ax.text(x_mid, y_bottom,
                f'T:{row.n_treated}\nC:{row.n_control}',
                ha='center', va='top', fontsize=6.5, color='#555555')

    ax.set_xlabel('Model Score Decile  (1 = lowest predicted NV gain, 10 = highest)',
                  fontsize=11)
    ax.set_ylabel('Observed Lift vs. Control  (avg opening balance, $)', fontsize=11)
    ax.set_title(
        'Proof Chart: Model Score Decile vs. Actual Observed Lift\n'
        'Prospects ranked by personalized net value gain — higher decile → higher actual response',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.set_xticks(deciles)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(output_dir, 'decile_validation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Decile validation chart saved: {path}")


def plot_uplift_curve(qini_data: dict, output_dir: str) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    pcts = qini_data['percentiles']
    q    = qini_data['qini']
    r    = qini_data['random']

    ax.plot(pcts, q, linewidth=2.5, color='#2E86AB', label='Model (net value ranking)')
    ax.plot(pcts, r, '--', linewidth=1.8, color='gray', alpha=0.7, label='Random targeting')
    ax.fill_between(
        pcts, q, r,
        where=np.array(q) >= np.array(r),
        alpha=0.25, color='#2E86AB', label='AUUC gain',
    )

    auuc = qini_data['auuc_lift']
    sign = '+' if auuc >= 0 else ''
    ax.text(
        0.97, 0.05,
        f"AUUC Lift: {sign}${auuc:,.0f}\n"
        f"N treated: {qini_data['n_treated']:,}\n"
        f"N control: {qini_data['n_control']:,}",
        transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )

    ax.set_xlabel('% of Population Targeted (ranked by model score)', fontsize=11)
    ax.set_ylabel('Cumulative Lift vs. Control  ($)', fontsize=11)
    ax.set_title(
        'Aggregated Uplift Curve: Model vs. Random Targeting\n'
        'All treated arms combined — ranked by personalized net value gain',
        fontsize=13, fontweight='bold', pad=12,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

    path = os.path.join(output_dir, 'uplift_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Uplift curve saved: {path}")


# ─────────────────────────────────────────────────────────────
# 7. Summary metrics
# ─────────────────────────────────────────────────────────────

def save_proof_metrics(decile_df: pd.DataFrame, qini_data: dict,
                        output_dir: str,
                        attrition_metrics: dict = None) -> None:
    valid = decile_df.dropna(subset=['observed_lift'])

    top_decile_lift = decile_df.loc[decile_df['decile'] == decile_df['decile'].max(),
                                     'observed_lift'].values[0]
    bot_decile_lift = decile_df.loc[decile_df['decile'] == decile_df['decile'].min(),
                                     'observed_lift'].values[0]

    ratio = (top_decile_lift / bot_decile_lift
             if (not np.isnan(bot_decile_lift) and bot_decile_lift != 0) else np.nan)

    metrics = {
        'AUUC_Lift_$':            round(qini_data['auuc_lift'], 2),
        'Decile_10_Observed_Lift': round(top_decile_lift, 2) if not np.isnan(top_decile_lift) else None,
        'Decile_1_Observed_Lift':  round(bot_decile_lift, 2) if not np.isnan(bot_decile_lift) else None,
        'Decile_10_to_1_Ratio':    round(ratio, 2) if not np.isnan(ratio) else None,
        'N_Treated':              qini_data['n_treated'],
        'N_Control':              qini_data['n_control'],
        'N_Deciles_Plotted':      int(valid['decile'].nunique()),
    }

    if attrition_metrics is not None:
        metrics.update({
            'Attrition_AUC':              round(attrition_metrics['auc'], 4),
            'Attrition_Optimal_Threshold': round(attrition_metrics['opt_thr'], 4),
            'Attrition_Sensitivity':       round(attrition_metrics['sensitivity'], 4),
            'Attrition_Specificity':       round(attrition_metrics['specificity'], 4),
            'Attrition_Precision':         round(attrition_metrics['precision'], 4),
            'Attrition_TP':               attrition_metrics['tp'],
            'Attrition_FP':               attrition_metrics['fp'],
            'Attrition_FN':               attrition_metrics['fn'],
            'Attrition_TN':               attrition_metrics['tn'],
            'Attrition_N':                attrition_metrics['n'],
        })

    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'proof_metrics.csv'), index=False)
    decile_df.to_csv(os.path.join(output_dir, 'decile_breakdown.csv'), index=False)

    print("\n" + "="*60)
    print("PROOF METRICS SUMMARY")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k:<32}: {v}")
    print("="*60)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate CATE model on stakeholder data with known outcomes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to stakeholder CSV file.')
    parser.add_argument('--output_dir', '-o', default='results/stakeholder_validation',
                        help='Output directory for charts and metrics.')
    parser.add_argument('--models_dir', '-m', default=None,
                        help='Model package directory (default: config.MODELS_DIR).')
    parser.add_argument('--n_deciles', type=int, default=10,
                        help='Number of deciles (default: 10).')
    parser.add_argument('--offers', nargs='+', type=int, default=None,
                        metavar='AMOUNT',
                        help=(
                            'Non-zero offer amounts in the data, e.g. --offers 400 750 1000. '
                            'If omitted, amounts are auto-detected from the offer column. '
                            'Must be within the trained range [$100–$1000].'
                        ))
    return parser.parse_args()


def main():
    import config
    args = _parse_args()

    print("\n" + "="*60)
    print("STAKEHOLDER MODEL VALIDATION")
    print("="*60)

    models_dir = args.models_dir or config.MODELS_DIR
    output_dir = args.output_dir
    ensure_dir(output_dir)

    # 1. Load
    df, campaign_offers = load_data(args.input, offer_override=args.offers)

    # 2. Score
    scored = run_model_scoring(df, models_dir, campaign_offers=campaign_offers)

    # 3. Decile validation
    scored, decile_df = compute_decile_validation(scored, n_deciles=args.n_deciles)
    print("\n  Decile breakdown (observed lift = avg_balance_treated - avg_balance_control):")
    print(decile_df[['decile', 'n_treated', 'n_control', 'observed_lift', 'avg_nv_gain']]
          .to_string(index=False))

    # 4. Qini curve
    qini_data = compute_qini_curve(scored)

    # 5. Attrition evaluation (only if open_on_9m is present)
    attrition_metrics = compute_attrition_metrics(scored)

    # 6. Charts
    plot_decile_validation(decile_df, output_dir)
    plot_uplift_curve(qini_data, output_dir)
    if attrition_metrics is not None:
        plot_attrition_roc(attrition_metrics, output_dir)
        plot_attrition_confusion_matrix(attrition_metrics, output_dir)

    # 7. Metrics
    save_proof_metrics(decile_df, qini_data, output_dir, attrition_metrics=attrition_metrics)

    # Save full scored file for downstream use
    scored.to_csv(os.path.join(output_dir, 'scored_stakeholder.csv'), index=False)

    print(f"\n  All outputs saved to: {output_dir}/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
