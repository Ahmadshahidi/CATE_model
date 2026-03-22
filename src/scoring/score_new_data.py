"""
Standalone Scoring Script — Score New Prospects for Letter-of-Offer Campaign

Loads the serialized model package from ``models/`` and applies the full
inference pipeline to a CSV of new prospects.

Output columns added to the input file
---------------------------------------
  cate_treatment_1/2/3     CATE (uplift $) for each offer arm
  retention_probability     P(prospect is on-book at month 9)
  net_value_arm_0/1/2/3    Net value of each arm for this prospect
  optimal_offer_arm         Arm ID of the best offer (0 = control)
  optimal_offer_name        Human-readable label, e.g. '$400'
  optimal_net_value         Net value of the best arm
  decile                    Score decile  1 (lowest) – 10 (highest)
  mail_flag                 1 = top-3 decile → send letter of offer
                            0 = suppress mailing

Usage
-----
  python src/scoring/score_new_data.py \\
      --input  data/new_prospects.csv \\
      --output results/scored_prospects.csv

  # Optional overrides:
  #   --models_dir  models/          (default: config.MODELS_DIR)
  #   --n_deciles   10               (default: config.N_DECILES)
  #   --top_n       3                (default: config.TOP_N_DECILES)
  #   --scenario    no_remail_no_stip  (default: observed features, no override)
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd

# Ensure project root is on the path regardless of where the script is called from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import config
from src.models.model_registry import load_pipeline
from src.models.net_value_strategy import _arm_base_cost


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring function (importable for notebook / API use)
# ──────────────────────────────────────────────────────────────────────────────

def score_prospects(
    df_raw: pd.DataFrame,
    models_dir: str = None,
    n_deciles: int = None,
    top_n_deciles: int = None,
    scenario: dict = None,
) -> pd.DataFrame:
    """
    Score a DataFrame of new prospects end-to-end.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw prospect file.  Columns must include all features expected by
        the trained pipeline (i.e., the 1800 Epsilon-like columns).
        ``opening_balance`` is used as the baseline balance if present;
        if absent a column of zeros is used and a warning is printed.
    models_dir : str, optional
        Directory created by ``save_pipeline()``.
        Defaults to ``config.MODELS_DIR``.
    n_deciles : int, optional
        Total number of score deciles (default ``config.N_DECILES`` = 10).
    top_n_deciles : int, optional
        How many top deciles receive a letter (default ``config.TOP_N_DECILES`` = 3).
    scenario : dict, optional
        If provided, override 'remail' and/or 'stipulation' columns before
        predicting CATEs.  Example: ``{'remail': 1, 'stipulation': 0}``.
        If None, feature columns are used as-is (observed values).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with score columns appended.
    """
    t0 = time.time()

    models_dir   = models_dir   or config.MODELS_DIR
    n_deciles     = n_deciles    or config.N_DECILES
    top_n_deciles = top_n_deciles or config.TOP_N_DECILES

    print("\n" + "="*70)
    print("PROSPECT SCORING PIPELINE")
    print("="*70)
    print(f"\n  Input prospects : {len(df_raw):,}  rows × {df_raw.shape[1]} cols")
    print(f"  Model directory : {models_dir}")
    print(f"  Decile cut-off  : top {top_n_deciles} of {n_deciles} deciles")
    if scenario:
        print(f"  Scenario        : {scenario}")
    else:
        print("  Scenario        : (observed feature values — no override)")

    # ── STEP 1: Load models ────────────────────────────────────────
    pkg = load_pipeline(models_dir)
    pruner         = pkg['pruner']
    xlearner       = pkg['xlearner']
    attrition      = pkg['attrition']
    feature_names  = pkg['feature_names']
    cfg_snap       = pkg['config']

    # ── STEP 2: Feature alignment ──────────────────────────────────
    print("\n" + "─"*70)
    print("STEP 2: FEATURE ALIGNMENT")
    print("─"*70)

    # Keep only columns that were present at training time; add zeros for any missing
    missing_cols = [c for c in feature_names if c not in df_raw.columns]
    extra_cols   = [c for c in df_raw.columns if c not in feature_names
                    and c not in ['opening_balance', 'treatment', 'treatment_name',
                                  'on_book_month9', 'offer']]

    if missing_cols:
        print(f"\n  ⚠  {len(missing_cols)} expected feature(s) not found in input data.")
        print(f"     Missing columns will be filled with 0.")
        print(f"     First 10 missing: {missing_cols[:10]}")
        for col in missing_cols:
            df_raw[col] = 0

    if extra_cols:
        print(f"\n  ℹ  {len(extra_cols)} extra column(s) in input (will be ignored by models).")

    # Apply saved feature selection  (transform only — no re-fitting)
    X = df_raw[feature_names].copy()
    print(f"\n  Features after alignment: {X.shape[1]}")

    # ── STEP 3: Baseline balance ───────────────────────────────────
    if 'opening_balance' in df_raw.columns:
        baseline = df_raw['opening_balance'].values.astype(float)
        print("\n  Baseline balance : read from 'opening_balance' column")
    else:
        baseline = np.zeros(len(df_raw))
        print("\n  ⚠  'opening_balance' not found — using 0 as baseline balance.")
        print("     Net values will be relative to zero; decile ranking is still valid.")

    # ── STEP 4: CATE prediction ───────────────────────────────────
    print("\n" + "─"*70)
    print("STEP 3: CATE PREDICTION  (X-Learner)")
    print("─"*70)

    if scenario:
        cates_df = xlearner.predict_cate_scenario(X, scenario)
    else:
        cates_df = xlearner.predict_all_cates(X)

    print(f"\n  CATE columns: {list(cates_df.columns)}")
    for col in cates_df.columns:
        print(f"    {col}: mean={cates_df[col].mean():+.2f}, "
              f"std={cates_df[col].std():.2f}")

    # ── STEP 5: Retention prediction ─────────────────────────────
    print("\n" + "─"*70)
    print("STEP 4: RETENTION PREDICTION  (Attrition Model)")
    print("─"*70)

    # Attrition model was trained with 'treatment' as an optional column.
    # For new prospecting data we don't have a treatment assignment yet;
    # we score under the assumption of no offer (treatment = 0) for the
    # base retention probability, then let CATEs drive offer optimisation.
    treatment_col = (
        df_raw['treatment'].values.astype(int)
        if 'treatment' in df_raw.columns
        else np.zeros(len(df_raw), dtype=int)
    )

    retention_proba = attrition.predict_proba(X, treatment=treatment_col)
    print(f"\n  Retention probability: mean={retention_proba.mean():.3f}, "
          f"std={retention_proba.std():.3f}")

    # ── STEP 6: Net value computation ─────────────────────────────
    print("\n" + "─"*70)
    print("STEP 5: NET VALUE COMPUTATION")
    print("─"*70)

    scored = df_raw.copy()
    scored['retention_probability'] = retention_proba
    for col in cates_df.columns:
        scored[col] = cates_df[col].values

    p_ret = retention_proba
    arm_ids = sorted(int(k) for k in cfg_snap['treatment_components'])

    nv_cols = []
    for arm_id in arm_ids:
        arm_str = str(arm_id)
        offer   = cfg_snap['treatment_components'][arm_str]

        if arm_id == 0:
            nv = p_ret * baseline
        else:
            cate_col = f'cate_treatment_{arm_id}'
            cost     = cfg_snap['offer_cost_rate'] * offer
            if scenario:
                cost += (cfg_snap['stipulation_cost'] * scenario.get('stipulation', 0)
                         + cfg_snap['remail_cost']    * scenario.get('remail', 0))
            nv = p_ret * (baseline + scored[cate_col].values) - cost

        col_name = f'net_value_arm_{arm_id}'
        scored[col_name] = nv
        nv_cols.append(col_name)
        print(f"  Arm {arm_id} ({cfg_snap['treatments'][arm_str]:<8}) "
              f"offer=${offer:<5}  mean NV = ${nv.mean():,.2f}")

    # ── STEP 7: Optimal arm assignment ────────────────────────────
    nv_matrix       = scored[nv_cols].values
    optimal_indices = np.argmax(nv_matrix, axis=1)
    optimal_arm_ids = np.array(arm_ids)[optimal_indices]
    optimal_nv      = nv_matrix[np.arange(len(scored)), optimal_indices]

    scored['optimal_offer_arm']  = optimal_arm_ids
    scored['optimal_offer_name'] = [
        cfg_snap['treatments'][str(a)] for a in optimal_arm_ids
    ]
    scored['optimal_net_value']  = optimal_nv

    print(f"\n  Optimal offer distribution:")
    dist = pd.Series(scored['optimal_offer_name']).value_counts().sort_index()
    for name, cnt in dist.items():
        print(f"    {name}: {cnt:,}  ({100*cnt/len(scored):.1f}%)")

    # ── STEP 8: Decile + mail flag ────────────────────────────────
    print("\n" + "─"*70)
    print("STEP 6: DECILE ASSIGNMENT & MAIL FLAG")
    print("─"*70)

    scored['decile'] = pd.qcut(
        scored['optimal_net_value'],
        q=n_deciles,
        labels=list(range(1, n_deciles + 1)),
        duplicates='drop',
    ).astype(int)

    top_cutoff = n_deciles - top_n_deciles   # e.g. 7 for top-3 of 10
    scored['mail_flag'] = (scored['decile'] > top_cutoff).astype(int)

    n_mailed = scored['mail_flag'].sum()
    print(f"\n  Decile cut-off  : > {top_cutoff}  "
          f"(deciles {top_cutoff+1}–{n_deciles})")
    print(f"  Prospects mailed: {n_mailed:,}  ({100*n_mailed/len(scored):.1f}%)")
    print(f"  Prospects held  : {len(scored)-n_mailed:,}  "
          f"({100*(len(scored)-n_mailed)/len(scored):.1f}%)")

    # Decile summary table
    print("\n  Decile summary:")
    decile_summary = (
        scored.groupby('decile')
        .agg(
            n_prospects      = ('optimal_net_value', 'count'),
            avg_net_value    = ('optimal_net_value', 'mean'),
            pct_mailed       = ('mail_flag', 'mean'),
        )
        .round(2)
    )
    print(decile_summary.to_string())

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"SCORING COMPLETE  ({elapsed:.1f}s)")
    print(f"  {len(scored):,} prospects scored")
    print(f"  {n_mailed:,} flagged for mailing  ({100*n_mailed/len(scored):.1f}%)")
    print(f"  Total expected NV (mailed only): "
          f"${scored.loc[scored['mail_flag']==1,'optimal_net_value'].sum():,.2f}")
    print(f"{'='*70}\n")

    return scored


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Score new prospects for letter-of-offer campaign.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to input CSV file of new prospects.',
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Path for output scored CSV (default: <input>_scored.csv).',
    )
    parser.add_argument(
        '--models_dir', '-m', default=None,
        help=f'Model package directory (default: {config.MODELS_DIR}).',
    )
    parser.add_argument(
        '--n_deciles', type=int, default=None,
        help=f'Total deciles for ranking (default: {config.N_DECILES}).',
    )
    parser.add_argument(
        '--top_n', type=int, default=None,
        help=f'Number of top deciles to mail (default: {config.TOP_N_DECILES}).',
    )
    parser.add_argument(
        '--scenario', default=None,
        choices=list(config.OPTIMIZATION_SCENARIOS.keys()),
        help='Optional remail/stipulation scenario to apply during CATE prediction.',
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Load input file ────────────────────────────────────────────
    print(f"\nLoading prospects from: {args.input}")
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    df_raw = pd.read_csv(args.input)
    print(f"  Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

    # ── Resolve scenario ───────────────────────────────────────────
    scenario = None
    if args.scenario:
        scenario = config.OPTIMIZATION_SCENARIOS[args.scenario]
        print(f"  Scenario: {args.scenario}  ({scenario})")

    # ── Run scoring ────────────────────────────────────────────────
    scored = score_prospects(
        df_raw        = df_raw,
        models_dir    = args.models_dir,
        n_deciles     = args.n_deciles,
        top_n_deciles = args.top_n,
        scenario      = scenario,
    )

    # ── Save output ────────────────────────────────────────────────
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        output_path = f"{base}_scored.csv"
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"  ✓ Scored prospects saved to: {output_path}")

    # Print a quick column guide
    score_cols = [
        c for c in scored.columns
        if c.startswith(('cate_', 'net_value_', 'optimal_', 'retention_',
                         'decile', 'mail_flag'))
    ]
    print(f"\n  Score columns added ({len(score_cols)}):")
    for col in score_cols:
        print(f"    {col}")


if __name__ == "__main__":
    main()
