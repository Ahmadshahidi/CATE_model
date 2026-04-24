"""
Standalone Scoring Script — Score New Prospects for Letter-of-Offer Campaign

Loads the serialized model package from ``models/`` and applies the full
inference pipeline to a CSV of new prospects.

Two separate feature sets are now used at inference time:
  • balance_feature_names   — features selected against opening_balance
                              → fed to the X-Learner (CATE prediction)
  • attrition_feature_names — features selected against on_book_month9
                              → fed to the AttritionModel (retention prob)

All models (X-Learner, AttritionModel, propensity) are CatBoost-based and
handle categorical features and missing values natively.  No pre-encoding
or imputation is required; unseen categorical levels are handled via
CatBoost's hash-based fallback.

Output columns added to the input file
---------------------------------------
Standard path (no --offers):
  cate_treatment_1/2/…     CATE (uplift $) for each trained offer arm
  retention_probability     P(prospect is on-book at month 9)
  net_value_arm_0/1/2/…    Net value of each trained arm
  optimal_offer_arm         Arm ID of the best offer (0 = control)
  optimal_offer_name        Human-readable label, e.g. '$400'
  optimal_net_value         Net value of the best arm

Campaign-offers path (--offers supplied):
  cate_treatment_1/2/…     Raw per-arm CATEs (kept for transparency)
  net_value_offer_0         Control net value
  net_value_offer_{amount}  One column per requested campaign offer
  optimal_offer_amount      Best offer dollar amount (0 = control)
  optimal_offer_name        Human-readable label, e.g. '$700'
  optimal_net_value         Net value of the best offer

Both paths:
  decile                    Score decile  1 (lowest) – 10 (highest)
  mail_flag                 1 = top-3 decile → send letter of offer
                            0 = suppress mailing

Usage
-----
  python src/scoring/score_new_data.py \\
      --input  data/new_prospects.csv \\
      --output results/scored_prospects.csv

  # Score with a custom campaign offer structure (interpolation applied as needed):
  python src/scoring/score_new_data.py \\
      --input  data/new_prospects.csv \\
      --offers 500 700

  # Optional overrides:
  #   --models_dir  models/          (default: config.MODELS_DIR)
  #   --n_deciles   10               (default: config.N_DECILES)
  #   --top_n       3                (default: config.TOP_N_DECILES)
  #   --scenario    no_remail        (default: observed features, no override)
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
# Helper: align a raw DataFrame to a target feature list
# ──────────────────────────────────────────────────────────────────────────────

def _align_features(df_raw: pd.DataFrame, feature_names: list,
                    label: str = '') -> pd.DataFrame:
    """
    Return a DataFrame that contains exactly ``feature_names`` columns in order.

    Missing columns are filled with NaN (CatBoost handles NaN natively).
    Extra columns are silently dropped.
    """
    missing = [c for c in feature_names if c not in df_raw.columns]
    if missing:
        print(f"\n  ⚠  {label}: {len(missing)} expected feature(s) not found "
              f"in input.  Missing columns filled with NaN.")
        print(f"     First 10 missing: {missing[:10]}")
        df = df_raw.copy()
        for col in missing:
            df[col] = np.nan
    else:
        df = df_raw

    return df[feature_names].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Offer interpolation helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_offer_plan(requested_amounts: list, trained_components: dict) -> list:
    """
    Map each requested offer amount to an exact arm or an interpolation spec.

    Parameters
    ----------
    requested_amounts : list[int]
        Dollar amounts the campaign wants to score (e.g. [500, 700]).
    trained_components : dict
        cfg_snap['treatment_components'] — str arm_id → int offer amount.

    Returns
    -------
    list of dicts, one per requested amount:
        Exact:       {'amount': 500, 'method': 'exact', 'arm_id': 3}
        Interpolate: {'amount': 700, 'method': 'interpolate',
                      'lower_arm': 3, 'upper_arm': 4,
                      'lower_offer': 500, 'upper_offer': 1000, 'alpha': 0.40}

    Raises
    ------
    ValueError if any amount is outside [min_trained_offer, max_trained_offer].
    """
    # Build sorted list of non-zero trained arms: [(arm_id_int, offer), ...]
    trained = sorted(
        [(int(k), v) for k, v in trained_components.items() if v > 0],
        key=lambda x: x[1],
    )
    offer_to_arm = {offer: arm_id for arm_id, offer in trained}
    min_offer    = trained[0][1]
    max_offer    = trained[-1][1]

    plan = []
    for amount in requested_amounts:
        if amount < min_offer or amount > max_offer:
            raise ValueError(
                f"${amount} is outside the trained offer range "
                f"[${min_offer}–${max_offer}]. Extrapolation is not supported."
            )
        if amount in offer_to_arm:
            plan.append({'amount': amount, 'method': 'exact',
                         'arm_id': offer_to_arm[amount]})
        else:
            # Find bracketing arms
            lower_arm, lower_offer = max(
                ((aid, off) for aid, off in trained if off < amount),
                key=lambda x: x[1],
            )
            upper_arm, upper_offer = min(
                ((aid, off) for aid, off in trained if off > amount),
                key=lambda x: x[1],
            )
            alpha = (amount - lower_offer) / (upper_offer - lower_offer)
            plan.append({
                'amount':      amount,
                'method':      'interpolate',
                'lower_arm':   lower_arm,
                'upper_arm':   upper_arm,
                'lower_offer': lower_offer,
                'upper_offer': upper_offer,
                'alpha':       alpha,
            })
    return plan


def _cate_for_offer(spec: dict, all_cates_df) -> 'np.ndarray':
    """Return per-prospect CATE array for a resolved offer spec."""
    if spec['method'] == 'exact':
        return all_cates_df[f'cate_treatment_{spec["arm_id"]}'].values
    lower = all_cates_df[f'cate_treatment_{spec["lower_arm"]}'].values
    upper = all_cates_df[f'cate_treatment_{spec["upper_arm"]}'].values
    return (1 - spec['alpha']) * lower + spec['alpha'] * upper


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring function (importable for notebook / API use)
# ──────────────────────────────────────────────────────────────────────────────

def score_prospects(
    df_raw: pd.DataFrame,
    models_dir: str = None,
    n_deciles: int = None,
    top_n_deciles: int = None,
    scenario: dict = None,
    campaign_offers: list = None,
    optimize_remail: bool = False,
) -> pd.DataFrame:
    """
    Score a DataFrame of new prospects end-to-end.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw prospect file.  Must include the features used during training.
        Columns not needed are ignored; missing feature columns are filled
        with NaN (CatBoost handles them natively).
    models_dir : str, optional
        Directory created by ``save_pipeline()``.
        Defaults to ``config.MODELS_DIR``.
    n_deciles : int, optional
        Total number of score deciles (default ``config.N_DECILES`` = 10).
    top_n_deciles : int, optional
        How many top deciles receive a letter (default ``config.TOP_N_DECILES`` = 3).
    scenario : dict, optional
        If provided, override 'remail' column before predicting CATEs.
        Example: ``{'remail': 1}``.
        If None, feature columns are used as-is (observed values).
    campaign_offers : list[int], optional
        Dollar amounts for this campaign run, e.g. [500, 700].
        Amounts not in the trained set are interpolated between adjacent arms.
        Must be within [min_trained_offer, max_trained_offer].
        If None, all trained arms are scored (existing behaviour).
    optimize_remail : bool, optional
        If True, score each prospect under both remail=0 and remail=1 and
        assign whichever maximises individual net value.  Adds
        ``optimal_remail_flag`` to the output.  Mutually exclusive with
        ``scenario`` (raises ValueError if both are set).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with score columns appended.
    """
    t0 = time.time()

    if optimize_remail and scenario:
        raise ValueError(
            "--optimize-remail and --scenario cannot be used together. "
            "Remail optimization scores every prospect under both remail states "
            "independently; a fixed scenario override is redundant."
        )

    models_dir    = models_dir    or config.MODELS_DIR
    n_deciles     = n_deciles     or config.N_DECILES
    top_n_deciles = top_n_deciles or config.TOP_N_DECILES

    print("\n" + "="*70)
    print("PROSPECT SCORING PIPELINE")
    print("="*70)
    print(f"\n  Input prospects : {len(df_raw):,}  rows × {df_raw.shape[1]} cols")
    print(f"  Model directory : {models_dir}")
    print(f"  Decile cut-off  : top {top_n_deciles} of {n_deciles} deciles")
    if optimize_remail:
        print("  Remail mode     : per-prospect optimization (remail=0 vs remail=1)")
    elif scenario:
        print(f"  Scenario        : {scenario}")
    else:
        print("  Scenario        : (observed feature values — no override)")

    # ── STEP 1: Load models ────────────────────────────────────────
    pkg = load_pipeline(models_dir)
    xlearner                = pkg['xlearner']
    attrition               = pkg['attrition']
    balance_feature_names   = pkg['balance_feature_names']
    attrition_feature_names = pkg['attrition_feature_names']
    cfg_snap                = pkg['config']

    # ── Campaign offer resolution (if custom offers requested) ────────
    offer_plan = None
    if campaign_offers is not None:
        trained_offers_sorted = sorted(
            v for k, v in cfg_snap['treatment_components'].items() if int(v) > 0
        )
        trained_str = '  '.join(f'${o}' for o in trained_offers_sorted)
        campaign_str = '  '.join(f'${a}' for a in campaign_offers)
        print(f"\n  Trained offer arms : {trained_str}")
        print(f"  Campaign offers    : {campaign_str}\n")

        offer_plan = resolve_offer_plan(campaign_offers, cfg_snap['treatment_components'])

        for spec in offer_plan:
            if spec['method'] == 'exact':
                print(f"    ${spec['amount']:<6} →  exact match     "
                      f"(arm {spec['arm_id']})")
            else:
                print(f"    ${spec['amount']:<6} →  interpolated    "
                      f"between ${spec['lower_offer']} (arm {spec['lower_arm']}) "
                      f"and ${spec['upper_offer']} (arm {spec['upper_arm']})  "
                      f"[alpha={spec['alpha']:.2f}]")

    # ── STEP 2: Build two separate feature matrices ─────────────────
    print("\n" + "─"*70)
    print("STEP 2: FEATURE ALIGNMENT  (dual feature sets)")
    print("─"*70)

    # Balance features → X-Learner (CATE + propensity)
    X_balance = _align_features(df_raw, balance_feature_names,
                                 label='Balance features (X-Learner)')
    print(f"\n  Balance feature matrix   : {X_balance.shape[1]} columns")

    # Attrition features → AttritionModel (retention probability)
    X_attrition = _align_features(df_raw, attrition_feature_names,
                                   label='Attrition features (AttritionModel)')
    print(f"  Attrition feature matrix : {X_attrition.shape[1]} columns")

    # Apply scenario override to balance features (CATEs only)
    if scenario:
        for col, val in scenario.items():
            if col in X_balance.columns:
                X_balance[col] = val

    # ── STEP 3: Baseline balance ───────────────────────────────────
    if 'opening_balance' in df_raw.columns:
        baseline = df_raw['opening_balance'].values.astype(float)
        print("\n  Baseline balance : read from 'opening_balance' column")
    else:
        baseline = np.zeros(len(df_raw))
        print("\n  ⚠  'opening_balance' not found — using 0 as baseline balance.")

    # ── STEP 4: CATE prediction ────────────────────────────────────
    print("\n" + "─"*70)
    print("STEP 3: CATE PREDICTION  (X-Learner, balance features)")
    print("─"*70)

    if optimize_remail:
        print("\n  Predicting CATEs under remail=0 ...")
        cates_r0 = xlearner.predict_cate_scenario(X_balance, {'remail': 0})
        print("  Predicting CATEs under remail=1 ...")
        cates_r1 = xlearner.predict_cate_scenario(X_balance, {'remail': 1})
        cates_df = cates_r0   # used only for column listing below
    else:
        cates_r0 = cates_r1 = None
        cates_df = xlearner.predict_all_cates(X_balance)

    print(f"\n  CATE columns: {list(cates_df.columns)}")
    for col in cates_df.columns:
        print(f"    {col}: mean={cates_df[col].mean():+.2f}, "
              f"std={cates_df[col].std():.2f}")

    # ── STEP 5: Retention prediction ──────────────────────────────
    print("\n" + "─"*70)
    print("STEP 4: RETENTION PREDICTION  (AttritionModel, attrition features)")
    print("─"*70)

    # Use actual treatment assignment if available; otherwise assume control (0)
    treatment_col = (
        df_raw['treatment'].values.astype(int)
        if 'treatment' in df_raw.columns
        else np.zeros(len(df_raw), dtype=int)
    )

    retention_proba = attrition.predict_proba(X_attrition, treatment=treatment_col)
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

    p_ret         = retention_proba
    remail_cost   = cfg_snap.get('remail_cost', 0)
    offer_cost_rt = cfg_snap['offer_cost_rate']
    scenario_remail_cost = remail_cost * (scenario.get('remail', 0) if scenario else 0)

    nv_cols = []

    if optimize_remail:
        # ── Per-prospect remail optimization path ─────────────────────
        arm_ids = sorted(int(k) for k in cfg_snap['treatment_components'])
        combos  = []   # (arm_id, remail_flag, temp_col)

        nv_ctrl = p_ret * baseline
        scored['_nv_0_r0'] = nv_ctrl
        combos.append((0, 0, '_nv_0_r0'))

        for arm_id in arm_ids:
            if arm_id == 0:
                continue
            arm_str   = str(arm_id)
            offer     = cfg_snap['treatment_components'][arm_str]
            base_cost = offer_cost_rt * offer
            cate_col  = f'cate_treatment_{arm_id}'

            nv_r0 = p_ret * (baseline + cates_r0[cate_col].values) - base_cost
            col_r0 = f'_nv_{arm_id}_r0'
            scored[col_r0] = nv_r0
            combos.append((arm_id, 0, col_r0))

            nv_r1 = (p_ret * (baseline + cates_r1[cate_col].values)
                     - base_cost - remail_cost)
            col_r1 = f'_nv_{arm_id}_r1'
            scored[col_r1] = nv_r1
            combos.append((arm_id, 1, col_r1))

        combo_cols = [c for _, _, c in combos]
        nv_matrix  = scored[combo_cols].values
        best_idx   = np.argmax(nv_matrix, axis=1)

        all_amounts = [combos[i][0] for i in best_idx]
        scored['optimal_offer_arm']   = all_amounts
        scored['optimal_remail_flag'] = [combos[i][1] for i in best_idx]
        scored['optimal_offer_name']  = [
            cfg_snap['treatments'][str(a)] for a in all_amounts
        ]
        scored['optimal_net_value']   = nv_matrix[np.arange(len(scored)), best_idx]
        scored.drop(columns=combo_cols, inplace=True)

        n_remail = int((scored['optimal_remail_flag'] == 1).sum())
        print(f"\n  Remail assigned   : {n_remail:,}  "
              f"({100*n_remail/len(scored):.1f}% of prospects)")
        print(f"\n  Optimal offer distribution:")
        dist = scored['optimal_offer_name'].value_counts().sort_index()
        for name, cnt in dist.items():
            print(f"    {name}: {cnt:,}  ({100*cnt/len(scored):.1f}%)")

    elif offer_plan is None:
        # ── Standard path: score all trained arms ─────────────────────
        arm_ids = sorted(int(k) for k in cfg_snap['treatment_components'])

        for arm_id in arm_ids:
            arm_str = str(arm_id)
            offer   = cfg_snap['treatment_components'][arm_str]

            if arm_id == 0:
                nv = p_ret * baseline
            else:
                cate_col = f'cate_treatment_{arm_id}'
                cost     = offer_cost_rt * offer + scenario_remail_cost
                nv       = p_ret * (baseline + scored[cate_col].values) - cost

            col_name = f'net_value_arm_{arm_id}'
            scored[col_name] = nv
            nv_cols.append(col_name)
            print(f"  Arm {arm_id} ({cfg_snap['treatments'][arm_str]:<8}) "
                  f"offer=${offer:<5}  mean NV = ${nv.mean():,.2f}")

        # Optimal assignment (trained-arm path)
        nv_matrix       = scored[nv_cols].values
        optimal_indices = np.argmax(nv_matrix, axis=1)
        optimal_arm_ids = np.array(arm_ids)[optimal_indices]
        optimal_nv      = nv_matrix[np.arange(len(scored)), optimal_indices]

        scored['optimal_offer_arm']  = optimal_arm_ids
        scored['optimal_offer_name'] = [
            cfg_snap['treatments'][str(a)] for a in optimal_arm_ids
        ]
        scored['optimal_net_value']  = optimal_nv

    else:
        # ── Campaign-offers path: control + requested amounts ──────────
        # Control
        nv_ctrl = p_ret * baseline
        scored['net_value_offer_0'] = nv_ctrl
        nv_cols.append('net_value_offer_0')
        print(f"  Control  offer=$0      mean NV = ${nv_ctrl.mean():,.2f}")

        # Each requested offer (exact or interpolated)
        for spec in offer_plan:
            amount = spec['amount']
            cate   = _cate_for_offer(spec, cates_df)
            cost   = offer_cost_rt * amount + scenario_remail_cost
            nv     = p_ret * (baseline + cate) - cost

            col_name = f'net_value_offer_{amount}'
            scored[col_name] = nv
            nv_cols.append(col_name)

            method_tag = 'exact' if spec['method'] == 'exact' else 'interpolated'
            print(f"  ${amount:<6}  ({method_tag:<12})  "
                  f"mean NV = ${nv.mean():,.2f}")

        # Optimal assignment (campaign-offers path)
        all_amounts = [0] + [s['amount'] for s in offer_plan]
        nv_matrix       = scored[nv_cols].values
        optimal_indices = np.argmax(nv_matrix, axis=1)
        optimal_amounts = np.array(all_amounts)[optimal_indices]
        optimal_nv      = nv_matrix[np.arange(len(scored)), optimal_indices]

        scored['optimal_offer_amount'] = optimal_amounts
        scored['optimal_offer_name']   = [
            'Control' if a == 0 else f'${a}' for a in optimal_amounts
        ]
        scored['optimal_net_value']    = optimal_nv

    print(f"\n  Optimal offer distribution:")
    dist = pd.Series(scored['optimal_offer_name']).value_counts().sort_index()
    for name, cnt in dist.items():
        print(f"    {name}: {cnt:,}  ({100*cnt/len(scored):.1f}%)")

    # ── STEP 8: Decile + mail flag ─────────────────────────────────
    print("\n" + "─"*70)
    print("STEP 6: DECILE ASSIGNMENT & MAIL FLAG")
    print("─"*70)

    scored['decile'] = pd.qcut(
        scored['optimal_net_value'],
        q=n_deciles,
        labels=list(range(1, n_deciles + 1)),
        duplicates='drop',
    ).astype(int)

    top_cutoff = n_deciles - top_n_deciles
    scored['mail_flag'] = (scored['decile'] > top_cutoff).astype(int)

    n_mailed = scored['mail_flag'].sum()
    print(f"\n  Decile cut-off  : > {top_cutoff}  "
          f"(deciles {top_cutoff+1}–{n_deciles})")
    print(f"  Prospects mailed: {n_mailed:,}  ({100*n_mailed/len(scored):.1f}%)")
    print(f"  Prospects held  : {len(scored)-n_mailed:,}  "
          f"({100*(len(scored)-n_mailed)/len(scored):.1f}%)")

    print("\n  Decile summary:")
    decile_summary = (
        scored.groupby('decile')
        .agg(
            n_prospects   = ('optimal_net_value', 'count'),
            avg_net_value = ('optimal_net_value', 'mean'),
            pct_mailed    = ('mail_flag', 'mean'),
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
        help='Optional remail scenario to apply during CATE prediction.',
    )
    parser.add_argument(
        '--offers', nargs='+', type=int, default=None,
        metavar='AMOUNT',
        help=(
            'Campaign offer amounts for this run, e.g. --offers 500 700. '
            'Amounts not in the trained set are interpolated between adjacent arms. '
            'Must be within the trained offer range. '
            'If omitted, all trained arms are scored.'
        ),
    )
    parser.add_argument(
        '--optimize-remail', action='store_true', default=False,
        help=(
            'Score each prospect under both remail=0 and remail=1 and assign '
            'whichever maximises individual net value. Adds optimal_remail_flag '
            'to output. Cannot be combined with --scenario.'
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    print(f"\nLoading prospects from: {args.input}")
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    df_raw = pd.read_csv(args.input)
    print(f"  Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

    scenario = None
    if args.scenario:
        scenario = config.OPTIMIZATION_SCENARIOS[args.scenario]
        print(f"  Scenario: {args.scenario}  ({scenario})")

    scored = score_prospects(
        df_raw           = df_raw,
        models_dir       = args.models_dir,
        n_deciles        = args.n_deciles,
        top_n_deciles    = args.top_n,
        scenario         = scenario,
        campaign_offers  = args.offers,
        optimize_remail  = args.optimize_remail,
    )

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        output_path = f"{base}_scored.csv"
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"  ✓ Scored prospects saved to: {output_path}")

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
