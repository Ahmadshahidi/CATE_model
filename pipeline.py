"""
Main Pipeline — Offer-Only Uplift Modeling with Scenario Analysis
Treatment arms: Control, $100, $400, $500
remail and stipulation are PREDICTORS (not treatment arms).

Workflow:
  0. Data generation / loading
  1. Step 1 sieve: variance + correlation pruning
  2. Step 2 sieve: Boruta-SHAP feature selection
     (remail and stipulation are included in the predictor pool)
  3. Model 1: X-Learner (3 offer arms vs. control)
  4. Model 2: Attrition / retention prediction
  5. Model 3: Net value optimization — baseline run
  6. Scenario analysis: run each (remail × stipulation) scenario,
     predict counterfactual CATEs, and compare portfolio net values
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

import config
from src.data_generation import generate_epsilon_data, save_data
from src.feature_selection.step1_initial_pruning import run_initial_pruning
from src.feature_selection.step2_boruta_shap import run_boruta_shap
from src.models.xlearner_uplift import train_xlearner
from src.models.attrition_model import train_attrition_model
from src.models.net_value_strategy import optimize_offers, NetValueOptimizer
from src.models.propensity_matching import run_propensity_matching
from src.models.iptw import run_iptw
from src.models.model_registry import save_pipeline


def main():
    print("\n" + "="*70)
    print(" "*10 + "INCREMENTAL CAMPAIGN UPLIFT MODELING")
    print(" "*5 + "Offer-Only Treatment  |  remail & stipulation as predictors")
    print(" "*10 + f"({len(config.TREATMENT_COMPONENTS)} arms: "
          f"{len(config.TREATMENT_COMPONENTS)-1} offer arms + control)")
    print("="*70 + "\n")

    start_time = time.time()

    # helper ─────────────────────────────────────────────────────────
    def _step_header(title: str, step_num: int, t_ref: float) -> float:
        """Print a step banner and return the current time as a step clock."""
        elapsed_total = time.time() - t_ref
        print(f"\n{'█'*70}")
        print(f"STEP {step_num}: {title}")
        print(f"  (pipeline elapsed: {elapsed_total:.1f}s)")
        print(f"{'█'*70}")
        return time.time()

    def _step_done(title: str, t_step: float) -> None:
        """Print a step completion banner with elapsed time."""
        elapsed = time.time() - t_step
        print(f"\n  ✓ {title} complete  [{elapsed:.1f}s]")
        print(f"{'─'*70}")
    # ─────────────────────────────────────────────────────────────────

    # ================================================================
    # STEP 0: DATA PREPARATION
    # ================================================================
    t_step = _step_header("DATA PREPARATION", 0, start_time)

    data_path = os.path.join(config.DATA_DIR, 'epsilon_synthetic.csv')

    if os.path.exists(data_path):
        print(f"\nLoading existing data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {df.shape[0]:,} samples with {df.shape[1]} columns")

        # If the old 13-arm data is present (treatment IDs > 3),
        # regenerate with the new 4-arm design.
        if df['treatment'].max() > 3:
            print("\n⚠  Saved data has old 13-arm design — regenerating ...")
            df = generate_epsilon_data()
            save_data(df)
        elif not {'stipulation', 'remail'}.issubset(df.columns):
            print("\n⚠  Saved data lacks remail/stipulation columns — regenerating ...")
            df = generate_epsilon_data()
            save_data(df)
    else:
        print("\nGenerating synthetic Epsilon-like data ...")
        df = generate_epsilon_data()
        save_data(df)

    # ------------------------------------------------------------------
    # Split columns:
    #   features  = all columns EXCEPT outcomes and 'offer'
    #               → remail and stipulation ARE included as features
    #   outcomes  = opening_balance, on_book_month9, treatment, treatment_name
    #   'offer'   = excluded (it is directly encoded by the treatment arm ID)
    # ------------------------------------------------------------------
    outcome_cols = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9']
    exclude_cols = outcome_cols + ['offer']   # offer is redundant with treatment

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X           = df[feature_cols]
    y_balance   = df['opening_balance']
    y_attrition = df['on_book_month9']
    treatment   = df['treatment']
    offers_col  = df['offer']
    stip_col    = df['stipulation']
    remail_col  = df['remail']

    print(f"\nData Summary:")
    print(f"  Total samples  : {X.shape[0]:,}")
    print(f"  Feature columns: {X.shape[1]}")
    print(f"  Includes remail      in features: {'remail' in X.columns}")
    print(f"  Includes stipulation in features: {'stipulation' in X.columns}")
    print(f"  Outcome 1 (Opening Balance)  : mean ${y_balance.mean():.2f}, "
          f"std ${y_balance.std():.2f}")
    print(f"  Outcome 2 (On-book Month 9)  : {y_attrition.mean():.1%} retention")

    print(f"\n  Treatment Distribution ({len(config.TREATMENT_COMPONENTS)} arms):")
    for arm_id in sorted(config.TREATMENT_COMPONENTS):
        cnt   = (treatment == arm_id).sum()
        pct   = 100 * cnt / len(treatment)
        offer = config.TREATMENT_COMPONENTS[arm_id]
        print(f"    Arm {arm_id}: {config.TREATMENTS[arm_id]:<12}  "
              f"offer=${offer:<5}  {cnt:>5,} ({pct:>5.1f}%)")

    print(f"\n  Remail / Stipulation (treated prospects):")
    treated_mask = offers_col > 0
    print(f"    Remail=1      : {remail_col[treated_mask].sum():,}  "
          f"({100*remail_col[treated_mask].mean():.1f}%)")
    print(f"    Stipulation=1 : {stip_col[treated_mask].sum():,}  "
          f"({100*stip_col[treated_mask].mean():.1f}%)")

    _step_done("DATA PREPARATION", t_step)

    # ================================================================
    # STEP 1: INITIAL PRUNING
    # ================================================================
    t_step = _step_header("SIEVE STEP 1 — INITIAL PRUNING  (Variance + Correlation)", 1, start_time)

    step1_report_path = os.path.join(config.RESULTS_DIR, 'step1_pruning_report.txt')
    X_step1, pruner = run_initial_pruning(
        X,
        y=y_balance,
        treatment=treatment,
        save_report_path=step1_report_path,
    )
    print(f"\n  Step 1 Complete: {X.shape[1]}  →  {X_step1.shape[1]} features")
    _step_done("INITIAL PRUNING", t_step)

    # ================================================================
    # STEP 2: BORUTA-SHAP
    # ================================================================
    t_step = _step_header("SIEVE STEP 2 — BORUTA-SHAP FEATURE SELECTION", 2, start_time)

    step2_report_path = os.path.join(config.RESULTS_DIR, 'step2_boruta_report.txt')
    X_step2, boruta = run_boruta_shap(
        X_step1,
        y=y_balance,
        task='regression',
        save_report_path=step2_report_path,
    )
    print(f"\n  Step 2 Complete: {X_step1.shape[1]}  →  {X_step2.shape[1]} features")

    # Report whether remail / stipulation survived selection
    for col in ['remail', 'stipulation']:
        survived = col in X_step2.columns
        print(f"  {col:>12} in selected features: {survived}")

    _step_done("BORUTA-SHAP FEATURE SELECTION", t_step)

    # Sieve summary
    print("\n" + "="*70)
    print("3-STEP SIEVE SUMMARY")
    print("="*70)
    print(f"  Original features         : {X.shape[1]:>6}")
    print(f"  After Step 1 (Pruning)    : {X_step1.shape[1]:>6}  "
          f"({100*X_step1.shape[1]/X.shape[1]:.1f}%)")
    print(f"  After Step 2 (Boruta)     : {X_step2.shape[1]:>6}  "
          f"({100*X_step2.shape[1]/X.shape[1]:.1f}%)")
    print(f"  Total reduction           :   "
          f"{100*(1-X_step2.shape[1]/X.shape[1]):.1f}%")
    print(f"  Step 3 (L1 in XGBoost)   : embedded inside models")
    print("="*70)

    selected_features_path = os.path.join(config.RESULTS_DIR, 'selected_features.txt')
    with open(selected_features_path, 'w') as f:
        f.write(f"Selected Features ({len(X_step2.columns)}):\n")
        f.write("="*60 + "\n\n")
        for feat in sorted(X_step2.columns):
            f.write(f"  {feat}\n")
    print(f"\n  Selected features saved to: {selected_features_path}")

    # ================================================================
    # STEP 2.5: BIAS CORRECTION  (PSM / IPTW / none)
    # ================================================================
    bias_method = getattr(config, 'BIAS_CORRECTION_METHOD', 'psm').lower()
    t_step = _step_header(f"BIAS CORRECTION  [{bias_method.upper()}]", '2.5', start_time)

    # Initialise defaults — overridden below depending on method
    X_for_xlearner   = X_step2
    t_for_xlearner   = treatment
    y_for_xlearner   = y_balance
    sample_weight_xl = None          # IPTW weights (None → unweighted)

    # ------------------------------------------------------------------
    if bias_method == 'psm':
        print("\n  PSM checks whether the offer arms differ systematically in")
        print("  pre-treatment covariates (selection bias).")
        print("  Visual outputs: Love plots, PS overlap, covariate balance boxplots.\n")

        psm = run_propensity_matching(
            X                = X_step2,
            treatment        = treatment,
            save_results_dir = config.RESULTS_DIR,
        )

        use_matched = getattr(config, 'USE_MATCHED_DATA_FOR_XLEARNER', False)

        if use_matched:
            print("\n  ℹ  USE_MATCHED_DATA_FOR_XLEARNER = True")
            print("     Building combined matched dataset for X-Learner ...")

            X_for_xlearner = pd.DataFrame()
            t_for_xlearner = pd.Series(dtype=int)

            for arm_id, match_df in psm.matched_data.items():
                feat_cols   = [c for c in match_df.columns
                               if c != 'matched_binary_treatment']
                binary_t    = match_df['matched_binary_treatment'].values
                arm_ids_row = np.where(binary_t == 1, arm_id, 0)

                X_for_xlearner = pd.concat(
                    [X_for_xlearner, match_df[feat_cols]], ignore_index=True)
                t_for_xlearner = pd.concat(
                    [t_for_xlearner, pd.Series(arm_ids_row)], ignore_index=True)

            print("  ⚠  y_balance aligned from full dataset by treatment IDs (approximate).")
            y_for_xlearner = y_balance.values[:len(X_for_xlearner)]

            print(f"  Matched dataset: {len(X_for_xlearner):,} rows  "
                  f"({len(X_for_xlearner)/len(X_step2)*100:.1f}% of original)")
        else:
            print("\n  ℹ  USE_MATCHED_DATA_FOR_XLEARNER = False")
            print("     PSM diagnostics saved; X-Learner uses the full (unmatched) dataset.")

        print(f"\n  PSM output files saved to: {config.RESULTS_DIR}")
        print(f"    • psm_propensity_overlap_before.png  — PS distributions pre-match")
        print(f"    • psm_propensity_overlap_after.png   — PS distributions post-match")
        print(f"    • psm_love_plot_arm{{N}}.png           — Love plot / SMD per arm")
        print(f"    • psm_covariate_balance_arm{{N}}.png   — Key covariate boxplots")
        print(f"    • propensity_balance_summary.csv     — Full balance metrics table")

    # ------------------------------------------------------------------
    elif bias_method == 'iptw':
        print("\n  IPTW re-weights every observation by the inverse of its")
        print("  probability of receiving the treatment it actually received.")
        print("  The full dataset is kept; weights are passed to the X-Learner.")
        print(f"\n  Settings:")
        print(f"    PS estimator     : {config.IPTW_PS_METHOD}")
        print(f"    Stabilised       : {config.IPTW_STABILIZED}")
        print(f"    Trim percentile  : {config.IPTW_TRIM_PERCENTILE}%  (each tail)\n")

        iptw_result = run_iptw(
            X                = X_step2,
            treatment        = treatment,
            save_results_dir = config.RESULTS_DIR,
        )
        sample_weight_xl = iptw_result.weights  # passed to train_xlearner below

        print(f"\n  IPTW output files saved to: {config.RESULTS_DIR}")
        print(f"    • iptw_weight_distribution.png   — weight histograms per arm")
        for arm_id in sorted(k for k in config.TREATMENT_COMPONENTS if k != 0):
            print(f"    • iptw_love_plot_arm{arm_id}.png       — weighted Love plot")
        print(f"    • iptw_balance_summary.csv       — weighted SMD table")
        print(f"    • iptw_effective_sample_sizes.csv — ESS per arm")

    # ------------------------------------------------------------------
    elif bias_method == 'none':
        print("\n  ℹ  BIAS_CORRECTION_METHOD = 'none'")
        print("     No bias correction applied.")
        print("     X-Learner trains on the full unweighted dataset.")

    # ------------------------------------------------------------------
    else:
        raise ValueError(
            f"Unknown BIAS_CORRECTION_METHOD='{bias_method}'. "
            f"Valid values: 'psm', 'iptw', 'none'."
        )

    _step_done(f"BIAS CORRECTION [{bias_method.upper()}]", t_step)

    # ================================================================
    # MODEL 1: X-LEARNER  (3 offer arms vs. control)
    # ================================================================
    t_step = _step_header("MODEL 1 — X-LEARNER UPLIFT  (Offer-Only Treatment)", 3, start_time)

    xlearner_model, auuc_df = train_xlearner(
        X                = X_for_xlearner,
        y                = y_for_xlearner,
        treatment        = t_for_xlearner,
        sample_weight    = sample_weight_xl,
        save_results_dir = config.RESULTS_DIR,
    )

    # Predict CATEs on observed features (baseline — no scenario override)
    cates = xlearner_model.predict_all_cates(X_step2)
    cates_path = os.path.join(config.RESULTS_DIR, 'cate_predictions.csv')
    cates.to_csv(cates_path, index=False)
    print(f"\n  CATE predictions saved to: {cates_path}")
    print(f"  Columns: {list(cates.columns)}")
    _step_done("X-LEARNER UPLIFT", t_step)

    # ================================================================
    # MODEL 2: ATTRITION PREDICTION
    # ================================================================
    t_step = _step_header("MODEL 2 — ATTRITION PREDICTION MODEL", 4, start_time)

    attrition_model = train_attrition_model(
        X                = X_step2,
        y                = y_attrition,
        treatment        = treatment,
        save_results_dir = config.RESULTS_DIR,
    )

    retention_proba = attrition_model.predict_proba(X_step2, treatment=treatment)
    retention_df = pd.DataFrame({
        'retention_probability': retention_proba,
        'predicted_on_book':    (retention_proba >= 0.5).astype(int),
    })
    retention_path = os.path.join(config.RESULTS_DIR, 'retention_predictions.csv')
    retention_df.to_csv(retention_path, index=False)
    print(f"\n  Retention predictions saved to: {retention_path}")
    _step_done("ATTRITION PREDICTION MODEL", t_step)

    # ================================================================
    # COMBINED INSIGHTS
    # ================================================================
    t_step = _step_header("COMBINED INSIGHTS: UPLIFT + RETENTION", '4b', start_time)

    insights = pd.DataFrame({
        'treatment':                 treatment.values,
        'treatment_name':            df['treatment_name'].values,
        'offer':                     offers_col.values,
        'stipulation':               stip_col.values,
        'remail':                    remail_col.values,
        'opening_balance_actual':    y_balance.values,
        'retention_actual':          y_attrition.values,
        'retention_predicted_proba': retention_proba,
    }, index=X_step2.index)

    # Attach baseline CATE columns
    insights = pd.concat([insights, cates], axis=1)

    insights_path = os.path.join(config.RESULTS_DIR, 'combined_insights.csv')
    insights.to_csv(insights_path, index=False)
    print(f"\n  Combined insights saved to: {insights_path}")

    # Campaign summary by offer arm
    print("\n" + "="*70)
    print("CAMPAIGN PERFORMANCE SUMMARY  (by offer amount)")
    print("="*70)
    summary = insights.groupby('offer').agg(
        n_prospects               = ('opening_balance_actual', 'count'),
        mean_opening_balance      = ('opening_balance_actual', 'mean'),
        std_opening_balance       = ('opening_balance_actual', 'std'),
        mean_retention_actual     = ('retention_actual', 'mean'),
        mean_retention_predicted  = ('retention_predicted_proba', 'mean'),
    ).round(3)
    print(summary.to_string())

    # Campaign summary by stipulation × remail (treated prospects)
    print("\n" + "="*70)
    print("CAMPAIGN SUMMARY  (by stipulation × remail — treated only)")
    print("="*70)
    summary2 = insights[insights['offer'] > 0].groupby(['stipulation', 'remail']).agg(
        n_prospects          = ('opening_balance_actual', 'count'),
        mean_opening_balance = ('opening_balance_actual', 'mean'),
        mean_retention       = ('retention_actual', 'mean'),
    ).round(3)
    print(summary2.to_string())
    _step_done("COMBINED INSIGHTS", t_step)

    # ================================================================
    # MODEL 3: NET VALUE OPTIMIZATION  (baseline run)
    # ================================================================
    t_step = _step_header("MODEL 3 — NET VALUE OPTIMIZATION & PERSONALIZED STRATEGY", 5, start_time)

    optimizer, net_value_results, strategy_comparison, qini_data, auuc_metrics = \
        optimize_offers(
            combined_insights_df = insights,
            save_results_dir     = config.RESULTS_DIR,
        )
    _step_done("NET VALUE OPTIMIZATION (baseline)", t_step)

    # ================================================================
    # STEP 3b: DECILE TARGETING STRATEGY
    # ================================================================
    t_step = _step_header("STEP 3b — DECILE TARGETING STRATEGY  (top-3 deciles mailed)", '5b', start_time)
    print("\nUsing the optimal_net_value score from the personalised optimiser,")
    print("we rank all prospects into 10 deciles and send letters only to those")
    print("in the top 3 deciles (~30% of the population).  Performance is")
    print("measured alongside the 'Offer Everyone' baseline.\n")

    decile_strat_df, decile_breakdown_df = optimizer.evaluate_decile_targeting_strategy(
        df          = net_value_results,
        n_deciles   = 10,
        top_n_deciles = 3,
        save_dir    = config.RESULTS_DIR,
    )
    _step_done("DECILE TARGETING STRATEGY", t_step)

    # Update comparison plots with the personalized strategy overlay
    print("\n" + "="*60)
    print("UPDATING COMPARISON PLOTS")
    print("="*60)

    xlearner_model.plot_auuc_comparison(
        auuc_df,
        net_value_auuc = auuc_metrics,
        save_path      = os.path.join(config.RESULTS_DIR, 'auuc_comparison.png'),
    )
    print("  ✓ auuc_comparison.png updated")

    xlearner_model.plot_cumulative_gain(
        X_step2, y_balance, treatment,
        net_value_qini_data = qini_data,
        save_path           = os.path.join(config.RESULTS_DIR, 'cumulative_gain.png'),
    )
    print("  ✓ cumulative_gain.png updated")

    # ================================================================
    # STEP 4: SCENARIO ANALYSIS
    # ================================================================
    t_step = _step_header("STEP 4 — SCENARIO ANALYSIS  (remail × stipulation toggles)", 6, start_time)
    print("\nFor each scenario, counterfactual CATEs are predicted by overriding")
    print("the remail/stipulation columns in the feature matrix, then the")
    print("net-value optimizer picks the best offer with scenario-adjusted costs.\n")

    scen_summary, scenario_dfs = optimizer.run_all_scenarios(
        X_features    = X_step2,
        insights_df   = insights,
        xlearner_model = xlearner_model,
        save_dir      = config.RESULTS_DIR,
    )

    # Save per-scenario detailed results
    for scen_name, df_opt in scenario_dfs.items():
        scen_path = os.path.join(
            config.RESULTS_DIR, f'scenario_{scen_name}_results.csv')
        df_opt.to_csv(scen_path, index=False)
    print(f"\n  Per-scenario result CSVs saved to: {config.RESULTS_DIR}")
    _step_done("SCENARIO ANALYSIS", t_step)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - start_time

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"\n  ✓ Total time : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"\n  📁 Results  : {config.RESULTS_DIR}")
    print(f"\n  Key outputs:")
    print(f"    Feature selection reports  :")
    print(f"      {step1_report_path}")
    print(f"      {step2_report_path}")
    print(f"    Selected features          : {selected_features_path}")
    print(f"    PSM diagnostics (Step 2.5) :")
    print(f"      {os.path.join(config.RESULTS_DIR, 'psm_propensity_overlap_before.png')}")
    print(f"      {os.path.join(config.RESULTS_DIR, 'psm_propensity_overlap_after.png')}")
    for arm_id in sorted(k for k in config.TREATMENT_COMPONENTS if k != 0):
        print(f"      {os.path.join(config.RESULTS_DIR, f'psm_love_plot_arm{arm_id}.png')}")
        print(f"      {os.path.join(config.RESULTS_DIR, f'psm_covariate_balance_arm{arm_id}.png')}")
    print(f"      {os.path.join(config.RESULTS_DIR, 'propensity_balance_summary.csv')}")
    print(f"    CATE predictions (baseline): {cates_path}")
    print(f"    Retention predictions      : {retention_path}")
    print(f"    Combined insights          : {insights_path}")
    print(f"    AUUC metrics (by arm)      : "
          f"{os.path.join(config.RESULTS_DIR, 'auuc_metrics.csv')}")
    print(f"    Net value strategy results : "
          f"{os.path.join(config.RESULTS_DIR, 'net_value_strategy_results.csv')}")
    print(f"    Scenario comparison        : "
          f"{os.path.join(config.RESULTS_DIR, 'scenario_comparison.csv')}")
    print(f"    Scenario bar chart         : "
          f"{os.path.join(config.RESULTS_DIR, 'scenario_comparison_bar.png')}")
    print(f"    Scenario offer dist chart  : "
          f"{os.path.join(config.RESULTS_DIR, 'scenario_offer_distributions.png')}")
    print(f"    Decile strategy comparison : "
          f"{os.path.join(config.RESULTS_DIR, 'decile_strategy_comparison.csv')}")
    print(f"    Decile breakdown           : "
          f"{os.path.join(config.RESULTS_DIR, 'decile_distribution.csv')}")
    print(f"    Decile distribution chart  : "
          f"{os.path.join(config.RESULTS_DIR, 'decile_distribution.png')}")
    print(f"    Decile vs everyone chart   : "
          f"{os.path.join(config.RESULTS_DIR, 'decile_vs_everyone_comparison.png')}")

    # Print decile strategy winner
    best_decile_row = decile_strat_df.loc[
        decile_strat_df['total_net_value'].idxmax()
    ]
    print(f"\n{'='*70}")
    print("DECILE TARGETING STRATEGY SUMMARY")
    print(f"{'='*70}")
    for _, row in decile_strat_df.iterrows():
        print(f"  {row['strategy']}")
        print(f"    Mailed         : {int(row['n_prospects_mailed']):,}  "
              f"({row['pct_population_mailed']:.1f}%)")
        print(f"    Total NV       : ${row['total_net_value']:,.2f}")
        print(f"    Lift vs ctrl   : ${row['lift_vs_control']:,.2f}")
        print(f"    Offer cost     : ${row['total_offer_cost']:,.2f}")
        print(f"    Lift / $ spent : ${row['lift_per_dollar_spent']:.4f}")
    print(f"{'='*70}\n")

    # Print winning scenario
    best_row = scen_summary.iloc[0]
    print(f"\n{'='*70}")
    print("BEST SCENARIO  (highest total portfolio net value)")
    print(f"{'='*70}")
    print(f"  Scenario      : {best_row['scenario']}")
    print(f"  remail        : {int(best_row['remail'])}")
    print(f"  stipulation   : {int(best_row['stipulation'])}")
    print(f"  Total NV      : ${best_row['total_net_value']:,.2f}")
    print(f"  Lift vs ctrl  : ${best_row['lift_vs_control']:,.2f}")
    print(f"  Extra cost/pp : ${best_row['extra_cost_per_prospect']:.2f}")
    print(f"{'='*70}\n")

    # ================================================================
    # BUSINESS RECOMMENDATIONS
    # ================================================================
    print("\n" + "="*70)
    print("BUSINESS RECOMMENDATIONS  (top-quartile CATE × retention segments)")
    print("="*70)

    for arm_id in sorted(xlearner_model.models.keys()):
        cate_col = f'cate_treatment_{arm_id}'
        if cate_col not in insights.columns:
            continue

        high_cate      = insights[cate_col].quantile(0.75)
        high_retention = insights['retention_predicted_proba'].quantile(0.75)

        seg = (
            (insights[cate_col] >= high_cate) &
            (insights['retention_predicted_proba'] >= high_retention)
        )

        offer = config.TREATMENT_COMPONENTS[arm_id]
        print(f"\n  Arm {arm_id}: {config.TREATMENTS[arm_id]}  (offer=${offer})")
        print(f"    Segment size                 : {seg.sum():,} ({100*seg.mean():.1f}%)")
        print(f"    Avg CATE in segment          : ${insights.loc[seg, cate_col].mean():.2f}")
        print(f"    Avg P(retention) in segment  : "
              f"{insights.loc[seg, 'retention_predicted_proba'].mean():.1%}")
        base_cost = config.OFFER_COST_RATE * offer
        print(f"    Base arm cost                : ${base_cost:.2f}")
        print(f"    → Add remail  (+${config.REMAIL_COST:.2f}): "
              f"total ${base_cost + config.REMAIL_COST:.2f}")
        print(f"    → Add stip    (+${config.STIPULATION_COST:.2f}): "
              f"total ${base_cost + config.STIPULATION_COST:.2f}")
        print(f"    → Add both    (+${config.REMAIL_COST + config.STIPULATION_COST:.2f}): "
              f"total ${base_cost + config.REMAIL_COST + config.STIPULATION_COST:.2f}")

    print("\n" + "="*70 + "\n")

    # ================================================================
    # STEP 5: SAVE MODEL PACKAGE  (for handoff / deployment)
    # ================================================================
    t_step = _step_header("STEP 5 — SAVING MODEL PACKAGE  (for handoff / deployment)", 7, start_time)
    print(f"  Serializing all pipeline artefacts to: {config.MODELS_DIR}\n")

    save_pipeline(
        pruner          = pruner,
        boruta          = boruta,
        xlearner_model  = xlearner_model,
        attrition_model = attrition_model,
        save_dir        = config.MODELS_DIR,
        feature_names   = X_step2.columns.tolist(),
    )

    print(f"\n  To score a new prospect file:")
    print(f"    python src/scoring/score_new_data.py \\")
    print(f"        --input  data/new_prospects.csv \\")
    print(f"        --output results/scored_prospects.csv")
    print(f"\n  Model package directory : {config.MODELS_DIR}")
    print(f"    step1_pruner.joblib       — variance + correlation pruner")
    print(f"    step2_boruta.joblib       — Boruta-SHAP feature selector")
    print(f"    xlearner_uplift.joblib    — X-Learner CATE models (3 arms)")
    print(f"    attrition_model.joblib    — XGBClassifier retention predictor")
    print(f"    feature_names.json        — {len(X_step2.columns)} selected feature names")
    print(f"    pipeline_config.json      — cost params, arm map, decile settings")
    print(f"    MANIFEST.txt              — human-readable package summary")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
