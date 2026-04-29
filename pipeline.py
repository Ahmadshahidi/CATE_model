"""
Main Pipeline — Offer-Only Uplift Modeling with Scenario Analysis
Treatment arms: Control, $100, $400, $500
remail and stipulation are PREDICTORS (not treatment arms).
stipulation is a 5-level categorical; its cost is zero and it is not toggled in scenarios.

Workflow:
  0. Data generation / loading
  1. Step 1 sieve: variance + correlation pruning
  2. Step 2 sieve: Boruta-SHAP feature selection
     (remail and stipulation are included in the predictor pool)
  3. Model 1: X-Learner (3 offer arms vs. control)
  4. Model 2: Attrition / retention prediction
  5. Model 3: Net value optimization — baseline run
  6. Scenario analysis: run each remail scenario,
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
    print(" "*5 + "(stipulation: 5-level categorical, zero cost, not toggled in scenarios)")
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
        elif config.PROSPECT_ID_COL not in df.columns:
            print(f"\n⚠  Saved data lacks '{config.PROSPECT_ID_COL}' column — regenerating ...")
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
    #                 (stipulation as 5-level categorical, zero cost)
    #   outcomes  = opening_balance, on_book_month9, treatment, treatment_name
    #   'offer'   = excluded (it is directly encoded by the treatment arm ID)
    # ------------------------------------------------------------------
    outcome_cols = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9']
    exclude_cols = outcome_cols + ['offer', config.PROSPECT_ID_COL]

    # Extract row IDs before building feature matrix
    id_col = (df[config.PROSPECT_ID_COL] if config.PROSPECT_ID_COL in df.columns
              else pd.Series(df.index, name=config.PROSPECT_ID_COL))

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X           = df[feature_cols]
    y_balance   = df['opening_balance']
    y_attrition = df['on_book_month9']
    treatment   = df['treatment']
    offers_col  = df['offer']
    remail_col  = df['remail']
    stip_col    = df['stipulation']

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

    treated_mask = offers_col > 0
    print(f"\n  Remail (treated prospects):")
    print(f"    Remail=1 : {remail_col[treated_mask].sum():,}  "
          f"({100*remail_col[treated_mask].mean():.1f}%)")
    print(f"\n  Stipulation distribution (all prospects):")
    stip_counts = stip_col.value_counts().reindex(config.STIPULATION_LEVELS, fill_value=0)
    for lvl, cnt in stip_counts.items():
        print(f"    {lvl:<12}: {cnt:,}  ({100*cnt/len(stip_col):.1f}%)")

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
    # STEP 2a: BORUTA-SHAP  —  Balance target  (→ X-Learner)
    # ================================================================
    t_step = _step_header(
        "SIEVE STEP 2a — BORUTA-SHAP  (Balance target → X-Learner)", 2, start_time)

    step2_balance_report_path = os.path.join(
        config.RESULTS_DIR, 'step2_boruta_balance_report.txt')
    X_balance, boruta_balance = run_boruta_shap(
        X_step1,
        y=y_balance,
        task='regression',
        save_report_path=step2_balance_report_path,
        top_n=getattr(config, 'BORUTA_BALANCE_TOP_N', None),
        force_include=getattr(config, 'BORUTA_BALANCE_FORCE_INCLUDE', []),
    )
    print(f"\n  Step 2a Complete (balance): "
          f"{X_step1.shape[1]}  →  {X_balance.shape[1]} features")

    for col in ['remail', 'stipulation']:
        print(f"  {col:>12} in balance features : {col in X_balance.columns}")

    _step_done("BORUTA-SHAP FEATURE SELECTION (balance)", t_step)

    # ================================================================
    # STEP 2b: BORUTA-SHAP  —  Attrition target  (→ AttritionModel)
    # ================================================================
    t_step = _step_header(
        "SIEVE STEP 2b — BORUTA-SHAP  (Attrition target → AttritionModel)", '2b', start_time)

    step2_attrition_report_path = os.path.join(
        config.RESULTS_DIR, 'step2_boruta_attrition_report.txt')
    X_attrition, boruta_attrition = run_boruta_shap(
        X_step1,
        y=y_attrition,
        task='classification',
        save_report_path=step2_attrition_report_path,
        top_n=getattr(config, 'BORUTA_ATTRITION_TOP_N', None),
        force_include=getattr(config, 'BORUTA_ATTRITION_FORCE_INCLUDE', []),
    )
    print(f"\n  Step 2b Complete (attrition): "
          f"{X_step1.shape[1]}  →  {X_attrition.shape[1]} features")

    for col in ['remail', 'stipulation']:
        print(f"  {col:>12} in attrition features : {col in X_attrition.columns}")

    _step_done("BORUTA-SHAP FEATURE SELECTION (attrition)", t_step)

    # Sieve summary — dual Boruta paths
    print("\n" + "="*70)
    print("FEATURE SELECTION SIEVE SUMMARY  (dual Boruta paths)")
    print("="*70)
    print(f"  Original features                    : {X.shape[1]:>6}")
    print(f"  After Step 1 (Pruning)               : {X_step1.shape[1]:>6}  "
          f"({100*X_step1.shape[1]/X.shape[1]:.1f}%)")
    print(f"  After Step 2a (Boruta — balance)     : {X_balance.shape[1]:>6}  "
          f"({100*X_balance.shape[1]/X.shape[1]:.1f}%)")
    print(f"  After Step 2b (Boruta — attrition)   : {X_attrition.shape[1]:>6}  "
          f"({100*X_attrition.shape[1]/X.shape[1]:.1f}%)")
    print(f"  Balance total reduction              :   "
          f"{100*(1-X_balance.shape[1]/X.shape[1]):.1f}%")
    print(f"  Attrition total reduction            :   "
          f"{100*(1-X_attrition.shape[1]/X.shape[1]):.1f}%")
    print("="*70)

    # Save both feature lists
    balance_features_path = os.path.join(config.RESULTS_DIR, 'balance_selected_features.txt')
    with open(balance_features_path, 'w') as f:
        f.write(f"Balance Selected Features ({len(X_balance.columns)}):\n")
        f.write("="*60 + "\n\n")
        for feat in sorted(X_balance.columns):
            f.write(f"  {feat}\n")
    print(f"\n  Balance features saved to  : {balance_features_path}")

    attrition_features_path = os.path.join(config.RESULTS_DIR, 'attrition_selected_features.txt')
    with open(attrition_features_path, 'w') as f:
        f.write(f"Attrition Selected Features ({len(X_attrition.columns)}):\n")
        f.write("="*60 + "\n\n")
        for feat in sorted(X_attrition.columns):
            f.write(f"  {feat}\n")
    print(f"  Attrition features saved to: {attrition_features_path}")

    # Keep backwards-compatible alias so downstream code that references
    # selected_features_path / X_step2 still works.
    selected_features_path = balance_features_path
    X_step2 = X_balance          # alias used in bias-correction and plotting

    # ================================================================
    # EVAL HOLD-OUT SPLIT  (post-feature-selection, pre-bias-correction)
    # ================================================================
    eval_test_size = getattr(config, 'EVAL_TEST_SIZE', 0.0)
    if eval_test_size > 0:
        print(f"\n  Evaluation hold-out split: {eval_test_size:.0%} test "
              f"(stratified on treatment arm)")
        train_idx, test_idx = train_test_split(
            X_step2.index,
            test_size    = eval_test_size,
            stratify     = treatment.loc[X_step2.index],
            random_state = config.RANDOM_SEED,
        )
        print(f"  Train rows: {len(train_idx):,}   Test rows: {len(test_idx):,}")
    else:
        train_idx = X_step2.index
        test_idx  = pd.Index([], dtype=X_step2.index.dtype)

    # ================================================================
    # STEP 2.5: BIAS CORRECTION  (PSM / IPTW / none)
    # ================================================================
    bias_method = getattr(config, 'BIAS_CORRECTION_METHOD', 'psm').lower()
    t_step = _step_header(f"BIAS CORRECTION  [{bias_method.upper()}]", '2.5', start_time)

    # Initialise defaults — restricted to train_idx when EVAL_TEST_SIZE > 0
    _train_X     = X_step2.loc[train_idx]
    _train_t     = treatment.loc[train_idx]
    _train_y     = y_balance.loc[train_idx]
    _train_y_att = y_attrition.loc[train_idx]

    X_for_xlearner            = _train_X
    t_for_xlearner            = _train_t
    y_for_xlearner            = _train_y
    sample_weight_xl          = None          # IPTW weights (None → unweighted)
    y_attrition_for_xlearner  = _train_y_att

    # ------------------------------------------------------------------
    if bias_method == 'psm':
        print("\n  PSM checks whether the offer arms differ systematically in")
        print("  pre-treatment covariates (selection bias).")
        print("  Visual outputs: Love plots, PS overlap, covariate balance boxplots.\n")

        psm = run_propensity_matching(
            X                = _train_X,
            treatment        = _train_t,
            save_results_dir = config.RESULTS_DIR,
        )

        use_matched = getattr(config, 'USE_MATCHED_DATA_FOR_XLEARNER', False)

        if use_matched:
            print("\n  ℹ  USE_MATCHED_DATA_FOR_XLEARNER = True")
            print("     Building combined matched dataset for X-Learner ...")

            X_for_xlearner      = pd.DataFrame()
            t_for_xlearner      = pd.Series(dtype=int)
            y_balance_xl_list   = []
            y_attrition_xl_list = []

            for arm_id, match_df in psm.matched_data.items():
                feat_cols   = [c for c in match_df.columns
                               if c != 'matched_binary_treatment']
                binary_t    = match_df['matched_binary_treatment'].values
                arm_ids_row = np.where(binary_t == 1, arm_id, 0)

                X_for_xlearner = pd.concat(
                    [X_for_xlearner, match_df[feat_cols]], ignore_index=True)
                t_for_xlearner = pd.concat(
                    [t_for_xlearner, pd.Series(arm_ids_row)], ignore_index=True)
                y_balance_xl_list.append(y_balance.loc[match_df.index].values)
                y_attrition_xl_list.append(y_attrition.loc[match_df.index].values)

            y_for_xlearner           = np.concatenate(y_balance_xl_list)
            y_attrition_for_xlearner = pd.Series(np.concatenate(y_attrition_xl_list))

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
            X                = _train_X,
            treatment        = _train_t,
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
    elif bias_method == 'overlap':
        print("\n  Overlap weighting re-weights every observation by")
        print("  w = h(x) / e_k(x)  where h(x) is the harmonic mean of")
        print("  propensity scores across arms.  Weights are bounded in")
        print("  (0, 1] — robust to skewed arm sizes and extreme PS values.")
        print(f"\n  Settings:")
        print(f"    PS estimator     : {config.OVERLAP_PS_METHOD}")
        print(f"    Trim percentile  : {config.OVERLAP_TRIM_PERCENTILE}%  (each tail)\n")

        from src.models.iptw import run_overlap
        overlap_result = run_overlap(
            X                = _train_X,
            treatment        = _train_t,
            save_results_dir = config.RESULTS_DIR,
        )
        sample_weight_xl = overlap_result.weights

        print(f"\n  Overlap weighting output files saved to: {config.RESULTS_DIR}")
        print(f"    • overlap_weight_distribution.png  — weight histograms per arm")
        for arm_id in sorted(k for k in config.TREATMENT_COMPONENTS if k != 0):
            print(f"    • overlap_love_plot_arm{arm_id}.png    — weighted Love plot")
        print(f"    • overlap_balance_summary.csv      — weighted SMD table")
        print(f"    • overlap_effective_sample_sizes.csv — ESS per arm")

    # ------------------------------------------------------------------
    elif bias_method == 'none':
        print("\n  ℹ  BIAS_CORRECTION_METHOD = 'none'")
        print("     No bias correction applied.")
        print("     X-Learner trains on the full unweighted dataset.")

    # ------------------------------------------------------------------
    else:
        raise ValueError(
            f"Unknown BIAS_CORRECTION_METHOD='{bias_method}'. "
            f"Valid values: 'psm', 'iptw', 'overlap', 'none'."
        )

    _step_done(f"BIAS CORRECTION [{bias_method.upper()}]", t_step)

    # Optional: restrict X-Learner training to retained customers only
    if getattr(config, 'XLEARNER_RETAINED_ONLY', False):
        retained_mask = np.array(y_attrition_for_xlearner) == 1
        n_before = len(X_for_xlearner)
        X_for_xlearner   = X_for_xlearner[retained_mask]
        y_for_xlearner   = np.asarray(y_for_xlearner)[retained_mask]
        t_for_xlearner   = np.asarray(t_for_xlearner)[retained_mask]
        if sample_weight_xl is not None:
            sample_weight_xl = np.asarray(sample_weight_xl)[retained_mask]
        print(f"\n  XLEARNER_RETAINED_ONLY=True: "
              f"{n_before:,} → {len(X_for_xlearner):,} rows "
              f"({100*retained_mask.mean():.1f}% retained)")

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

    # Predict CATEs on all rows (train + test) for combined insights
    cates = xlearner_model.predict_all_cates(X_step2)

    # Held-out test AUUC (out-of-sample evaluation)
    if eval_test_size > 0:
        print(f"\n  Computing held-out test AUUC ({len(test_idx):,} rows) ...")
        auuc_df_test = xlearner_model.compute_auuc(
            X_step2.loc[test_idx],
            y_balance.loc[test_idx],
            treatment.loc[test_idx],
        )
        auuc_test_path = os.path.join(config.RESULTS_DIR, 'auuc_metrics_test.csv')
        auuc_df_test.to_csv(auuc_test_path, index=False)
        print(f"  Test-set AUUC metrics saved to: {auuc_test_path}")

    # Attach prospect_id and save CATE predictions
    cates_out = cates.copy()
    cates_out.insert(0, config.PROSPECT_ID_COL, id_col.loc[X_step2.index].values)
    cates_path = os.path.join(config.RESULTS_DIR, 'cate_predictions.csv')
    cates_out.to_csv(cates_path, index=False)
    print(f"\n  CATE predictions saved to: {cates_path}")
    print(f"  Columns: {list(cates.columns)}")
    _step_done("X-LEARNER UPLIFT", t_step)

    # ================================================================
    # MODEL 2: ATTRITION PREDICTION
    # ================================================================
    t_step = _step_header("MODEL 2 — ATTRITION PREDICTION MODEL", 4, start_time)

    # NOTE: X_attrition uses features selected against the ATTRITION target
    # (on_book_month9) — a separate Boruta-SHAP run from the balance path.
    # Training is restricted to train_idx rows; predictions run on all rows.
    attrition_model = train_attrition_model(
        X                = X_attrition.loc[train_idx],
        y                = y_attrition.loc[train_idx],
        treatment        = treatment.loc[train_idx],
        save_results_dir = config.RESULTS_DIR,
    )

    retention_proba = attrition_model.predict_proba(X_attrition, treatment=treatment)
    retention_df = pd.DataFrame({
        config.PROSPECT_ID_COL:  id_col.values,
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
        config.PROSPECT_ID_COL:      id_col.loc[X_step2.index].values,
        'treatment':                 treatment.values,
        'treatment_name':            df['treatment_name'].values,
        'offer':                     offers_col.values,
        'remail':                    remail_col.values,
        'stipulation':               stip_col.values,
        'opening_balance_actual':    y_balance.values,
        'retention_actual':          y_attrition.values,
        'retention_predicted_proba': retention_proba,
    }, index=X_step2.index)

    # Attach baseline CATE columns
    insights = pd.concat([insights, cates], axis=1)

    # Mark train/test split when EVAL_TEST_SIZE > 0
    if eval_test_size > 0:
        insights['split'] = np.where(insights.index.isin(train_idx), 'train', 'test')

    insights_path = os.path.join(config.RESULTS_DIR, 'combined_insights.csv')
    insights.to_csv(insights_path, index=False)
    print(f"\n  Combined insights saved to: {insights_path}")

    # Save separate train/test sub-files when hold-out split is active
    if eval_test_size > 0:
        train_insights_path = os.path.join(config.RESULTS_DIR, 'combined_insights_train.csv')
        test_insights_path  = os.path.join(config.RESULTS_DIR, 'combined_insights_test.csv')
        insights.loc[train_idx].to_csv(train_insights_path, index=False)
        insights.loc[test_idx].to_csv(test_insights_path,  index=False)
        print(f"  Train insights ({len(train_idx):,} rows): {train_insights_path}")
        print(f"  Test  insights ({len(test_idx):,} rows) : {test_insights_path}")

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

    # Campaign summary by stipulation level × remail (treated prospects)
    print("\n" + "="*70)
    print("CAMPAIGN SUMMARY  (by stipulation level × remail — treated only)")
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

    # ================================================================
    # STEP 3c: TEST SET EVALUATION  (hold-out — net value + decile strategy)
    # ================================================================
    if eval_test_size > 0:
        t_step = _step_header(
            "STEP 3c — TEST SET EVALUATION  (held-out net value + decile strategy)",
            '5c', start_time)
        print(f"\n  Running net value optimisation + decile strategy on the held-out")
        print(f"  test set ({len(test_idx):,} rows).  All model outputs are out-of-sample.\n")

        test_eval_dir = os.path.join(config.RESULTS_DIR, 'test_eval')
        os.makedirs(test_eval_dir, exist_ok=True)

        test_insights = insights.loc[test_idx].copy()

        # Run through the same net value maximisation pipeline as the training set
        test_nv = optimizer.compute_net_values(test_insights)
        test_nv = optimizer.assign_optimal_offers(test_nv)

        # Decile targeting strategy — identical call to the training evaluation
        optimizer.evaluate_decile_targeting_strategy(
            df            = test_nv,
            n_deciles     = 10,
            top_n_deciles = 3,
            save_dir      = test_eval_dir,
        )

        # Cumulative net value chart for the test set
        qini_test = optimizer.compute_qini_curve_combined(test_nv)
        optimizer.plot_cumulative_net_value(
            qini_test,
            save_path=os.path.join(test_eval_dir, 'cumulative_net_value_test.png'),
        )

        print(f"\n  Test evaluation outputs saved to: {test_eval_dir}/")
        print(f"    • decile_distribution.png")
        print(f"    • decile_strategy_comparison.csv / .png")
        print(f"    • cumulative_net_value_test.png")
        _step_done("TEST SET EVALUATION", t_step)

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
    # STEP 4: PER-PROSPECT REMAIL OPTIMIZATION
    # ================================================================
    t_step = _step_header("STEP 4 — PER-PROSPECT REMAIL OPTIMIZATION", 6, start_time)
    print("\nCATEs are predicted under both remail=0 and remail=1.  For each")
    print("prospect the (offer arm, remail flag) combination that maximises")
    print("individual net value is selected.  This strictly dominates any")
    print("global remail policy.\n")

    remail_opt_df = optimizer.run_remail_optimization(
        X_features     = X_step2,
        insights_df    = insights,
        xlearner_model = xlearner_model,
        save_dir       = config.RESULTS_DIR,
    )
    remail_opt_path = os.path.join(config.RESULTS_DIR, 'remail_optimization_results.csv')
    _step_done("PER-PROSPECT REMAIL OPTIMIZATION", t_step)

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
    print(f"      {step2_balance_report_path}  (balance → X-Learner)")
    print(f"      {step2_attrition_report_path}  (attrition → retention model)")
    print(f"    Balance selected features  : {balance_features_path}")
    print(f"    Attrition selected features: {attrition_features_path}")
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
    print(f"    Remail opt. results        : {remail_opt_path}")
    print(f"    Remail comparison chart    : "
          f"{os.path.join(config.RESULTS_DIR, 'remail_optimization_comparison.png')}")
    print(f"    Remail offer dist. chart   : "
          f"{os.path.join(config.RESULTS_DIR, 'remail_offer_distribution.png')}")
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

    # Remail optimization summary
    n_remail = int((remail_opt_df['optimal_remail_flag'] == 1).sum())
    pct_r    = 100 * n_remail / len(remail_opt_df)
    total_nv = remail_opt_df['optimal_net_value'].sum()
    print(f"\n{'='*70}")
    print("PER-PROSPECT REMAIL OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Remail assigned     : {n_remail:,}  ({pct_r:.1f}% of prospects)")
    print(f"  Total portfolio NV  : ${total_nv:,.2f}")
    print(f"  Avg NV / prospect   : ${remail_opt_df['optimal_net_value'].mean():,.2f}")
    print(f"  Lift vs no offer    : ${remail_opt_df['net_value_gain_vs_ctrl'].sum():,.2f}")
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
        print(f"    (stipulation cost = $0)")

    print("\n" + "="*70 + "\n")

    # ================================================================
    # STEP 5: SAVE MODEL PACKAGE  (for handoff / deployment)
    # ================================================================
    t_step = _step_header("STEP 5 — SAVING MODEL PACKAGE  (for handoff / deployment)", 7, start_time)
    print(f"  Serializing all pipeline artefacts to: {config.MODELS_DIR}\n")

    save_pipeline(
        pruner                  = pruner,
        boruta_balance          = boruta_balance,
        boruta_attrition        = boruta_attrition,
        xlearner_model          = xlearner_model,
        attrition_model         = attrition_model,
        balance_feature_names   = X_balance.columns.tolist(),
        attrition_feature_names = X_attrition.columns.tolist(),
        save_dir                = config.MODELS_DIR,
    )

    print(f"\n  To score a new prospect file:")
    print(f"    python src/scoring/score_new_data.py \\")
    print(f"        --input  data/new_prospects.csv \\")
    print(f"        --output results/scored_prospects.csv")
    print(f"\n  Model package directory : {config.MODELS_DIR}")
    print(f"    step1_pruner.joblib              — variance + correlation pruner")
    print(f"    step2_boruta_balance.joblib       — Boruta-SHAP (balance target)")
    print(f"    step2_boruta_attrition.joblib     — Boruta-SHAP (attrition target)")
    n_offer_arms = len(config.TREATMENT_COMPONENTS) - 1
    print(f"    xlearner_uplift.joblib            — X-Learner CATE models ({n_offer_arms} arms)")
    print(f"    attrition_model.joblib            — CatBoostClassifier retention predictor")
    print(f"    balance_feature_names.json        — {len(X_balance.columns)} balance features")
    print(f"    attrition_feature_names.json      — {len(X_attrition.columns)} attrition features")
    print(f"    pipeline_config.json              — cost params, arm map, decile settings")
    print(f"    MANIFEST.txt                      — human-readable package summary")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
