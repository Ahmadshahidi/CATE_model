"""
Net Value Strategy for Optimal Offer Assignment — Offer-Only Treatment Design
Offers: $100, $400, $500  + Control

Net Value per arm = P(on_book) × (Baseline + CATE)
                  − OFFER_COST_RATE × offer
                  − STIPULATION_COST × scenario['stipulation']
                  − REMAIL_COST      × scenario['remail']

remail and stipulation are no longer treatment arms — they are toggled
via OPTIMIZATION_SCENARIOS in config.py.  For each scenario the model
predicts counterfactual CATEs (feature override) and computes scenario-
adjusted costs, returning a full comparison table across scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


# ------------------------------------------------------------------
# Arm cost helpers
# ------------------------------------------------------------------

def _arm_base_cost(arm_id: int) -> float:
    """Offer-only base cost for an arm (no remail / stipulation)."""
    offer = config.TREATMENT_COMPONENTS[arm_id]
    return config.OFFER_COST_RATE * offer


def _scenario_extra_cost(scenario: dict) -> float:
    """Additional cost from remail / stipulation flags in a scenario."""
    return (
        config.STIPULATION_COST * scenario.get('stipulation', 0)
        + config.REMAIL_COST    * scenario.get('remail', 0)
    )


def _arm_cost(arm_id: int, scenario: dict = None) -> float:
    """Total arm cost for a given scenario (or base cost if no scenario)."""
    base = _arm_base_cost(arm_id)
    if scenario:
        base += _scenario_extra_cost(scenario)
    return base


class NetValueOptimizer:
    """
    Optimize offer assignment based on net value maximization.

    Net Value = P(retention) × (Baseline_Balance + CATE) − Cost(arm, scenario)

    Usage:
        # Single run (observed features / no scenario override):
        optimizer.generate_full_report(insights_df, save_dir=...)

        # Scenario analysis (remail / stipulation ON / OFF combinations):
        optimizer.run_all_scenarios(X_features, insights_df, xlearner_model, save_dir=...)
    """

    def __init__(self):
        self.results_df = None

    # ------------------------------------------------------------------
    # STEP 1: Net Value Computation
    # ------------------------------------------------------------------
    def compute_net_values(self, df, cate_df=None, scenario=None,
                           baseline_balance_col='opening_balance_actual',
                           retention_col='retention_predicted_proba'):
        """
        Compute net value for every prospect under every arm.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain retention_predicted_proba and opening_balance_actual.
        cate_df : pd.DataFrame, optional
            CATE columns (cate_treatment_{arm_id}).  If None, the columns
            are read from df directly.
        scenario : dict, optional
            {'remail': 0/1, 'stipulation': 0/1}
            When supplied, arm costs include remail/stipulation charges.
        baseline_balance_col : str
        retention_col : str

        Returns
        -------
        pd.DataFrame  (copy of df with net_value_arm_{arm_id} columns)
        """
        print("\n" + "="*60)
        if scenario:
            scen_label = ', '.join(f"{k}={v}" for k, v in scenario.items())
            print(f"NET VALUE COMPUTATION  (Scenario: {scen_label})")
        else:
            print("NET VALUE COMPUTATION  (Observed features)")
        print("="*60)

        df = df.copy()
        if cate_df is not None:
            # Merge scenario CATEs in
            for col in cate_df.columns:
                df[col] = cate_df[col].values

        p_ret    = df[retention_col]
        baseline = df[baseline_balance_col]

        for arm_id, offer in config.TREATMENT_COMPONENTS.items():
            if arm_id == 0:
                df['net_value_arm_0'] = p_ret * baseline
                print(f"\nArm 0 (Control):  mean NV = ${df['net_value_arm_0'].mean():,.2f}")
            else:
                cate_col = f'cate_treatment_{arm_id}'
                cost     = _arm_cost(arm_id, scenario)
                predicted_balance = baseline + df[cate_col]

                col      = f'net_value_arm_{arm_id}'
                df[col]  = p_ret * predicted_balance - cost

                print(f"\nArm {arm_id}: {config.TREATMENTS[arm_id]}")
                print(f"  Arm cost (incl. scenario) = ${cost:.2f}  |  "
                      f"mean NV = ${df[col].mean():,.2f}")

        print("="*60)
        return df

    # ------------------------------------------------------------------
    # STEP 2: Assign Optimal Arm
    # ------------------------------------------------------------------
    def assign_optimal_offers(self, df):
        """
        For each prospect, choose the arm that maximises net value.

        Adds columns:
            optimal_offer_arm      : arm ID
            optimal_offer_name     : human-readable label
            optimal_net_value      : net value of chosen arm
            net_value_gain_vs_ctrl : improvement over arm-0 (control)
        """
        print("\n" + "="*60)
        print("OPTIMAL OFFER ASSIGNMENT")
        print("="*60)

        df       = df.copy()
        all_arms = sorted(config.TREATMENT_COMPONENTS.keys())
        nv_cols  = [f'net_value_arm_{a}' for a in all_arms]

        nv_matrix          = df[nv_cols].values
        optimal_indices    = np.argmax(nv_matrix, axis=1)
        optimal_arm_ids    = np.array(all_arms)[optimal_indices]
        optimal_net_values = nv_matrix[np.arange(len(df)), optimal_indices]

        df['optimal_offer_arm']      = optimal_arm_ids
        df['optimal_offer_name']     = [config.TREATMENTS[a] for a in optimal_arm_ids]
        df['optimal_net_value']      = optimal_net_values
        df['net_value_gain_vs_ctrl'] = optimal_net_values - df['net_value_arm_0']

        print("\nOptimal Offer Distribution:")
        dist = df['optimal_offer_name'].value_counts().sort_index()
        for name, cnt in dist.items():
            pct = 100 * cnt / len(df)
            print(f"  {name}: {cnt:,}  ({pct:.1f}%)")

        print(f"\nNet Value Summary:")
        print(f"  Total portfolio net value : ${df['optimal_net_value'].sum():,.2f}")
        print(f"  Average per prospect      : ${df['optimal_net_value'].mean():,.2f}")
        print(f"  Total gain vs. control    : ${df['net_value_gain_vs_ctrl'].sum():,.2f}")
        print("="*60)

        self.results_df = df
        return df

    # ------------------------------------------------------------------
    # STEP 3: Strategy Benchmarks
    # ------------------------------------------------------------------
    def evaluate_strategy_vs_benchmarks(self, df):
        """Compare personalised strategy against benchmarks."""
        print("\n" + "="*60)
        print("STRATEGY COMPARISON VS. BENCHMARKS")
        print("="*60)

        strategies  = []
        nv_cols     = [f'net_value_arm_{a}' for a in sorted(config.TREATMENT_COMPONENTS)]
        ctrl_total  = df['net_value_arm_0'].sum()
        pers_total  = df['optimal_net_value'].sum()

        strategies.append({
            'strategy':                   'Personalized (Net Value Opt.)',
            'total_net_value':            pers_total,
            'avg_net_value_per_prospect': pers_total / len(df),
            'lift_vs_control':            pers_total - ctrl_total,
        })
        strategies.append({
            'strategy':                   'Everyone Control (No Offers)',
            'total_net_value':            ctrl_total,
            'avg_net_value_per_prospect': ctrl_total / len(df),
            'lift_vs_control':            0,
        })

        for arm_id in sorted(config.TREATMENT_COMPONENTS):
            if arm_id == 0:
                continue
            col   = f'net_value_arm_{arm_id}'
            total = df[col].sum()
            strategies.append({
                'strategy':                   f'Everyone {config.TREATMENTS[arm_id]}',
                'total_net_value':            total,
                'avg_net_value_per_prospect': total / len(df),
                'lift_vs_control':            total - ctrl_total,
            })

        random_total = df[nv_cols].mean(axis=1).sum()
        strategies.append({
            'strategy':                   'Random Assignment',
            'total_net_value':            random_total,
            'avg_net_value_per_prospect': random_total / len(df),
            'lift_vs_control':            random_total - ctrl_total,
        })

        comp_df = (pd.DataFrame(strategies)
                   .sort_values('total_net_value', ascending=False)
                   .reset_index(drop=True))

        print("\n" + comp_df.to_string(index=False))

        single_arm_rows = comp_df[comp_df['strategy'].str.startswith('Everyone $')]
        if not single_arm_rows.empty:
            best_single = single_arm_rows['total_net_value'].max()
            imp_vs_best = 100 * (pers_total - best_single) / best_single if best_single else 0
            imp_vs_ctrl = 100 * (pers_total - ctrl_total)  / ctrl_total  if ctrl_total else 0

            print(f"\n{'='*60}")
            print("PERSONALIZED STRATEGY PERFORMANCE")
            print(f"{'='*60}")
            print(f"  Improvement vs. Control          : {imp_vs_ctrl:+.2f}%")
            print(f"  Improvement vs. Best Single Arm  : {imp_vs_best:+.2f}%")
            print(f"  Total portfolio net value        : ${pers_total:,.2f}")
            print(f"{'='*60}\n")

        return comp_df

    # ------------------------------------------------------------------
    # STEP 4: Qini curve & AUUC for the personalized strategy
    # ------------------------------------------------------------------
    def compute_qini_curve_combined(self, df):
        """Cumulative net value gain Qini for the personalized strategy."""
        df_sorted = df.sort_values('optimal_net_value', ascending=False).reset_index(drop=True)
        n_total   = len(df_sorted)

        percentiles, qini_values, cum_nv = [], [], []
        for pct in range(0, 101, 1):
            n_targeted = int(n_total * pct / 100)
            if n_targeted == 0:
                percentiles.append(pct); qini_values.append(0); cum_nv.append(0)
                continue
            targeted = df_sorted.iloc[:n_targeted]
            percentiles.append(pct)
            qini_values.append(targeted['net_value_gain_vs_ctrl'].sum())
            cum_nv.append(targeted['optimal_net_value'].sum())

        max_gain        = qini_values[-1]
        random_baseline = [max_gain * (p / 100) for p in percentiles]

        return {
            'percentiles':          percentiles,
            'qini':                 qini_values,
            'cumulative_net_value': cum_nv,
            'random':               random_baseline,
            'max_gain':             max_gain,
        }

    def compute_auuc_combined(self, qini_data):
        """AUUC for the personalized strategy."""
        pcts  = np.array(qini_data['percentiles'])
        qvals = np.array(qini_data['qini'])
        rvals = np.array(qini_data['random'])

        auuc      = integrate.trapezoid(qvals, pcts)
        auuc_rand = integrate.trapezoid(rvals, pcts)
        auuc_lift = auuc - auuc_rand
        auuc_norm = (auuc - auuc_rand) / auuc_rand if auuc_rand else 0

        return {
            'AUUC':            auuc,
            'AUUC_random':     auuc_rand,
            'AUUC_lift':       auuc_lift,
            'AUUC_normalized': auuc_norm,
        }

    # ------------------------------------------------------------------
    # SCENARIO ANALYSIS
    # ------------------------------------------------------------------
    def run_all_scenarios(self, X_features, insights_df, xlearner_model, save_dir=None):
        """
        Run the net-value optimizer for every scenario defined in
        config.OPTIMIZATION_SCENARIOS and return a comparison table.

        For each scenario:
          1. Predict counterfactual CATEs by overriding remail/stipulation
             in the feature matrix (via XLearnerUplift.predict_cate_scenario).
          2. Compute net values using scenario-adjusted arm costs.
          3. Assign optimal offers.
          4. Collect summary metrics.

        Parameters
        ----------
        X_features : pd.DataFrame
            Feature matrix passed to the X-Learner (same columns as fit).
        insights_df : pd.DataFrame
            Must contain opening_balance_actual and retention_predicted_proba.
        xlearner_model : XLearnerUplift
            Already-fitted X-Learner instance.
        save_dir : str, optional
            If provided, saves scenario_comparison.csv and a bar chart.

        Returns
        -------
        pd.DataFrame
            One row per scenario with portfolio-level metrics.
        """
        print("\n" + "="*70)
        print("SCENARIO ANALYSIS  (remail × stipulation toggles)")
        print("="*70)

        scenario_rows   = []
        scenario_dfs    = {}

        for scen_name, scenario in config.OPTIMIZATION_SCENARIOS.items():
            scen_label = ', '.join(f"{k}={v}" for k, v in scenario.items())
            print(f"\n{'─'*60}")
            print(f"  Scenario: {scen_name}  ({scen_label})")
            print(f"{'─'*60}")

            # 1. Counterfactual CATE predictions under this scenario
            scen_cate_df = xlearner_model.predict_cate_scenario(X_features, scenario)

            # 2. Net value computation with scenario costs
            df_nv = self.compute_net_values(
                insights_df,
                cate_df  = scen_cate_df,
                scenario = scenario,
            )

            # 3. Optimal offer assignment
            df_opt = self.assign_optimal_offers(df_nv)
            scenario_dfs[scen_name] = df_opt

            # 4. Collect metrics
            total_nv  = df_opt['optimal_net_value'].sum()
            ctrl_nv   = df_opt['net_value_arm_0'].sum()
            lift      = total_nv - ctrl_nv
            avg_nv    = df_opt['optimal_net_value'].mean()

            dist = df_opt['optimal_offer_name'].value_counts()

            row = {
                'scenario':                scen_name,
                'remail':                  scenario.get('remail', 0),
                'stipulation':             scenario.get('stipulation', 0),
                'total_net_value':         total_nv,
                'avg_net_value':           avg_nv,
                'lift_vs_control':         lift,
                'extra_cost_per_prospect': _scenario_extra_cost(scenario),
            }
            # Offer share columns
            for arm_id in sorted(config.TREATMENT_COMPONENTS):
                arm_name = config.TREATMENTS[arm_id]
                row[f'pct_{arm_name}'] = 100 * dist.get(arm_name, 0) / len(df_opt)
            scenario_rows.append(row)

        scen_summary = pd.DataFrame(scenario_rows).sort_values(
            'total_net_value', ascending=False
        ).reset_index(drop=True)

        print("\n" + "="*70)
        print("SCENARIO COMPARISON SUMMARY")
        print("="*70)
        display_cols = ['scenario', 'remail', 'stipulation',
                        'total_net_value', 'avg_net_value',
                        'lift_vs_control', 'extra_cost_per_prospect']
        print(scen_summary[display_cols].to_string(index=False))
        print("\nBest scenario  :", scen_summary.iloc[0]['scenario'])
        print(f"Total NV       : ${scen_summary.iloc[0]['total_net_value']:,.2f}")
        print("="*70)

        if save_dir:
            # CSV
            csv_path = os.path.join(save_dir, 'scenario_comparison.csv')
            scen_summary.to_csv(csv_path, index=False)
            print(f"\n  Scenario comparison saved to: {csv_path}")

            # Bar chart
            self._plot_scenario_comparison(scen_summary, save_dir)

            # Per-scenario offer distribution chart
            self._plot_scenario_offer_distributions(scenario_dfs, save_dir)

        return scen_summary, scenario_dfs

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    def _plot_scenario_comparison(self, scen_summary, save_dir):
        """Grouped bar chart: total portfolio NV and lift vs. control by scenario."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cmap = plt.get_cmap('tab10')

        scenarios = scen_summary['scenario'].tolist()
        colors    = [cmap(i) for i in range(len(scenarios))]

        # Panel 1: total net value
        ax = axes[0]
        bars = ax.bar(scenarios, scen_summary['total_net_value'] / 1e6, color=colors)
        ax.set_title('Total Portfolio Net Value by Scenario',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Net Value ($ millions)', fontsize=10)
        ax.set_xlabel('Scenario', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'${h:.2f}M',
                    ha='center', va='bottom', fontsize=9)

        # Panel 2: lift vs. control
        ax = axes[1]
        bars = ax.bar(scenarios, scen_summary['lift_vs_control'] / 1e3, color=colors)
        ax.set_title('Lift vs. Control (No Offers) by Scenario',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Lift vs. Control ($ thousands)', fontsize=10)
        ax.set_xlabel('Scenario', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha='right')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'${h:,.0f}K',
                    ha='center', va='bottom', fontsize=9)

        plt.suptitle('Scenario Analysis: remail × stipulation toggles',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'scenario_comparison_bar.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Scenario comparison chart saved to: {save_path}")
        plt.close()

    def _plot_scenario_offer_distributions(self, scenario_dfs, save_dir):
        """One pie chart per scenario showing optimal offer mix."""
        n       = len(scenario_dfs)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]

        cmap = plt.get_cmap('tab10')
        arm_names = [config.TREATMENTS[a] for a in sorted(config.TREATMENT_COMPONENTS)]

        for ax, (scen_name, df_opt) in zip(axes, scenario_dfs.items()):
            counts = df_opt['optimal_offer_name'].value_counts()
            vals   = [counts.get(n, 0) for n in arm_names]
            colors = [cmap(i) for i in range(len(arm_names))]

            ax.pie(vals, labels=arm_names, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title(scen_name, fontsize=10, fontweight='bold')

        plt.suptitle('Optimal Offer Distribution by Scenario',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'scenario_offer_distributions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Scenario offer distribution chart saved to: {save_path}")
        plt.close()

    def plot_qini_curve_combined(self, qini_data, auuc_metrics, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(qini_data['percentiles'], qini_data['qini'],
                linewidth=2.5, label='Personalized Strategy', color='#2E86AB')
        ax.plot(qini_data['percentiles'], qini_data['random'],
                '--', linewidth=2, label='Random Assignment', color='gray', alpha=0.7)
        ax.fill_between(qini_data['percentiles'],
                        qini_data['qini'], qini_data['random'],
                        where=np.array(qini_data['qini']) >= np.array(qini_data['random']),
                        alpha=0.3, color='#2E86AB', label='AUUC Gain')

        ax.text(0.98, 0.02,
                f"AUUC Lift: ${auuc_metrics['AUUC_lift']:,.0f}\n"
                f"Normalised: {auuc_metrics['AUUC_normalized']:.2f}x",
                transform=ax.transAxes, fontsize=10,
                va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('% of Population Targeted (by Net Value)', fontsize=12)
        ax.set_ylabel('Cumulative Net Value Gain vs. Control', fontsize=12)
        ax.set_title('Qini Curve: Personalized Multi-Arm Strategy',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Qini curve saved to: {save_path}")
        plt.close()

    def plot_cumulative_net_value(self, qini_data, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))
        pcts = qini_data['percentiles']
        cum  = qini_data['cumulative_net_value']

        ax.plot(pcts, cum, linewidth=2.5, color='#F18F01')
        ax.fill_between(pcts, 0, cum, alpha=0.3, color='#F18F01')
        ax.text(0.98, 0.98, f"Total Portfolio Value: ${cum[-1]:,.0f}",
                transform=ax.transAxes, fontsize=11,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax.set_xlabel('% of Population Targeted', fontsize=12)
        ax.set_ylabel('Cumulative Net Value ($)', fontsize=12)
        ax.set_title('Cumulative Net Value: Personalized Strategy',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Cumulative net value saved to: {save_path}")
        plt.close()

    def plot_offer_distribution(self, df, save_path=None):
        """Pie + bar chart of optimal offer arm distribution."""
        counts = df['optimal_offer_name'].value_counts().sort_index()
        n      = len(counts)
        cmap   = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(n)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 1.5), 6))

        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax1.set_title('Optimal Offer Distribution', fontsize=12, fontweight='bold')

        ax2.bar(range(n), counts.values, color=colors)
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(counts.index, rotation=30, ha='right', fontsize=9)
        ax2.set_ylabel('Number of Prospects', fontsize=11)
        ax2.set_title('Optimal Offer Counts', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(counts.values):
            ax2.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Offer distribution saved to: {save_path}")
        plt.close()

    def plot_strategy_comparison(self, comp_df, save_path=None):
        """Horizontal bar chart for strategy comparison."""
        comp_df = comp_df.head(20)
        colors  = ['#2E86AB' if 'Personalized' in s else '#A8DADC'
                   for s in comp_df['strategy']]

        fig, ax = plt.subplots(figsize=(12, max(6, len(comp_df) * 0.55)))
        ax.barh(comp_df['strategy'],
                comp_df['total_net_value'] / 1e6,
                color=colors)

        ax.set_xlabel('Total Net Value ($ Millions)', fontsize=12)
        ax.set_title('Strategy Comparison: Total Portfolio Net Value',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar in ax.patches:
            w = bar.get_width()
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'${w:.2f}M', va='center', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Strategy comparison saved to: {save_path}")
        plt.close()

    def plot_net_value_by_offer(self, df, save_path=None):
        """Box plots: net value distribution by assigned arm."""
        offers = sorted(df['optimal_offer_name'].unique())
        n      = len(offers)
        cmap   = plt.get_cmap('tab20')
        data   = [df[df['optimal_offer_name'] == o]['optimal_net_value'].values
                  for o in offers]

        fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
        bp = ax.boxplot(data, labels=offers, patch_artist=True)
        for patch, i in zip(bp['boxes'], range(n)):
            patch.set_facecolor(cmap(i % 20))
            patch.set_alpha(0.6)

        ax.set_xlabel('Optimal Arm', fontsize=12)
        ax.set_ylabel('Net Value ($)', fontsize=12)
        ax.set_title('Net Value Distribution by Assigned Arm',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Net value boxplot saved to: {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # DECILE TARGETING STRATEGY
    # ------------------------------------------------------------------
    def evaluate_decile_targeting_strategy(self, df, n_deciles=10, top_n_deciles=3,
                                           save_dir=None):
        """
        Decile all prospects by their ``optimal_net_value`` score, then
        compare two mailing strategies:

        Strategy A — "Offer Everyone":
            Every prospect receives the offer assigned by the personalised
            net-value optimiser (current pipeline behaviour).

        Strategy B — "Top-{top_n_deciles} Decile Targeting":
            Only prospects in the highest ``top_n_deciles`` deciles (e.g.
            top 30% when top_n_deciles=3 out of 10) are mailed an offer.
            The rest are treated as control (arm 0 — no offer, no cost).

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``assign_optimal_offers()``.  Must contain:
            ``optimal_net_value``, ``optimal_net_value``, ``net_value_arm_0``,
            ``net_value_gain_vs_ctrl``, ``optimal_offer_name``.
        n_deciles : int
            Number of deciles to create (default 10).
        top_n_deciles : int
            Number of top deciles that receive offers (default 3).
        save_dir : str, optional
            If provided, saves CSV and PNG outputs here.

        Returns
        -------
        (strategy_comparison_df, decile_breakdown_df)
        """
        print("\n" + "="*70)
        print(f"DECILE TARGETING STRATEGY  (top {top_n_deciles} of {n_deciles} deciles mailed)")
        print("="*70)

        df = df.copy()

        # ── Assign decile labels (1 = lowest score, n_deciles = highest) ──
        df['decile'] = pd.qcut(
            df['optimal_net_value'],
            q=n_deciles,
            labels=list(range(1, n_deciles + 1)),
            duplicates='drop',
        ).astype(int)

        top_decile_threshold = n_deciles - top_n_deciles   # e.g. 7 for top-3 of 10
        df['in_target_deciles'] = df['decile'] > top_decile_threshold

        print(f"\n  Decile cut-off  : decile > {top_decile_threshold}  "
              f"(deciles {top_decile_threshold + 1}–{n_deciles})")
        print(f"  Prospects mailed: {df['in_target_deciles'].sum():,}  "
              f"({100 * df['in_target_deciles'].mean():.1f}%)")
        print(f"  Prospects held  : {(~df['in_target_deciles']).sum():,}  "
              f"({100 * (~df['in_target_deciles']).mean():.1f}%)")

        # ── Per-decile breakdown ──────────────────────────────────────────
        # Offer cost for Strategy A (everyone): base arm cost (no remail/stip overhead
        # — those are scenario-level; here we use the base offer cost only)
        def _lookup_arm_cost(arm_name):
            for arm_id, label in config.TREATMENTS.items():
                if label == arm_name:
                    return _arm_base_cost(arm_id)
            return 0.0

        df['_offer_cost_everyone'] = df['optimal_offer_name'].apply(_lookup_arm_cost)

        decile_rows = []
        for d in range(1, n_deciles + 1):
            mask = df['decile'] == d
            sub  = df[mask]
            is_targeted = d > top_decile_threshold

            # Strategy A — everyone gets optimal offer
            nv_a    = sub['optimal_net_value'].sum()
            cost_a  = sub['_offer_cost_everyone'].sum()

            # Strategy B — targeted prospects get optimal offer; others get control NV
            if is_targeted:
                nv_b   = sub['optimal_net_value'].sum()
                cost_b = sub['_offer_cost_everyone'].sum()
            else:
                nv_b   = sub['net_value_arm_0'].sum()
                cost_b = 0.0

            decile_rows.append({
                'decile':                    d,
                'n_prospects':               len(sub),
                'mailed_in_targeting':       is_targeted,
                'total_nv_offer_everyone':   nv_a,
                'avg_nv_offer_everyone':     nv_a / len(sub) if len(sub) else 0,
                'total_cost_offer_everyone': cost_a,
                'total_nv_top3_targeting':   nv_b,
                'avg_nv_top3_targeting':     nv_b / len(sub) if len(sub) else 0,
                'total_cost_top3_targeting': cost_b,
            })

        decile_df = pd.DataFrame(decile_rows)

        print("\n" + "="*70)
        print("DECILE BREAKDOWN")
        print("="*70)
        display_cols = ['decile', 'n_prospects', 'mailed_in_targeting',
                        'avg_nv_offer_everyone', 'avg_nv_top3_targeting',
                        'total_cost_offer_everyone', 'total_cost_top3_targeting']
        print(decile_df[display_cols].to_string(index=False))

        # ── Portfolio-level strategy comparison ───────────────────────────
        ctrl_total_nv = df['net_value_arm_0'].sum()

        # Strategy A: everyone gets personalised offer
        total_nv_a    = df['optimal_net_value'].sum()
        total_cost_a  = df['_offer_cost_everyone'].sum()
        lift_a        = total_nv_a - ctrl_total_nv
        n_mailed_a    = len(df)

        # Strategy B: only top-N deciles mailed
        targeted_mask = df['in_target_deciles']
        total_nv_b    = (
            df.loc[targeted_mask,  'optimal_net_value'].sum()
            + df.loc[~targeted_mask, 'net_value_arm_0'].sum()
        )
        total_cost_b  = df.loc[targeted_mask, '_offer_cost_everyone'].sum()
        lift_b        = total_nv_b - ctrl_total_nv
        n_mailed_b    = targeted_mask.sum()

        strategies = [
            {
                'strategy':                       f'Offer Everyone (100% mailed)',
                'n_prospects_mailed':             n_mailed_a,
                'pct_population_mailed':          100.0,
                'total_net_value':                total_nv_a,
                'avg_net_value_per_prospect':     total_nv_a / len(df),
                'avg_nv_per_mailed_prospect':     total_nv_a / n_mailed_a,
                'total_offer_cost':               total_cost_a,
                'cost_per_mailed_prospect':       total_cost_a / n_mailed_a if n_mailed_a else 0,
                'lift_vs_control':                lift_a,
                'lift_per_dollar_spent':          lift_a / total_cost_a if total_cost_a else 0,
            },
            {
                'strategy':                       f'Top-{top_n_deciles} Decile Targeting ({100*top_n_deciles/n_deciles:.0f}% mailed)',
                'n_prospects_mailed':             n_mailed_b,
                'pct_population_mailed':          100 * n_mailed_b / len(df),
                'total_net_value':                total_nv_b,
                'avg_net_value_per_prospect':     total_nv_b / len(df),
                'avg_nv_per_mailed_prospect':     total_nv_b / n_mailed_b if n_mailed_b else 0,
                'total_offer_cost':               total_cost_b,
                'cost_per_mailed_prospect':       total_cost_b / n_mailed_b if n_mailed_b else 0,
                'lift_vs_control':                lift_b,
                'lift_per_dollar_spent':          lift_b / total_cost_b if total_cost_b else 0,
            },
        ]

        strat_df = pd.DataFrame(strategies)

        print("\n" + "="*70)
        print("STRATEGY COMPARISON  (Offer Everyone  vs.  Decile Targeting)")
        print("="*70)
        for _, row in strat_df.iterrows():
            print(f"\n  {row['strategy']}")
            print(f"    Prospects mailed        : {int(row['n_prospects_mailed']):,}  "
                  f"({row['pct_population_mailed']:.1f}%)")
            print(f"    Total net value         : ${row['total_net_value']:,.2f}")
            print(f"    Avg NV / all prospects  : ${row['avg_net_value_per_prospect']:,.2f}")
            print(f"    Avg NV / mailed prospect: ${row['avg_nv_per_mailed_prospect']:,.2f}")
            print(f"    Total offer cost        : ${row['total_offer_cost']:,.2f}")
            print(f"    Cost / mailed prospect  : ${row['cost_per_mailed_prospect']:,.2f}")
            print(f"    Lift vs. control        : ${row['lift_vs_control']:,.2f}")
            print(f"    Lift per $ spent        : ${row['lift_per_dollar_spent']:.4f}")

        # Incremental diff
        nv_diff   = total_nv_b   - total_nv_a
        cost_saved = total_cost_a - total_cost_b
        lift_diff  = lift_b       - lift_a
        print(f"\n  ── Incremental Impact of Switching to Decile Targeting ──")
        print(f"    NV change          : ${nv_diff:+,.2f}")
        print(f"    Cost saved         : ${cost_saved:+,.2f}")
        print(f"    Lift change        : ${lift_diff:+,.2f}")
        print("="*70)

        if save_dir:
            # CSV outputs
            strat_path  = os.path.join(save_dir, 'decile_strategy_comparison.csv')
            decile_path = os.path.join(save_dir, 'decile_distribution.csv')
            strat_df.to_csv(strat_path,  index=False)
            decile_df.to_csv(decile_path, index=False)
            print(f"\n  Strategy comparison CSV : {strat_path}")
            print(f"  Decile breakdown CSV    : {decile_path}")

            # Plots
            self._plot_decile_distribution(decile_df, top_n_deciles, n_deciles, save_dir)
            self._plot_decile_strategy_comparison(strat_df, decile_df,
                                                   top_n_deciles, n_deciles, save_dir)

        return strat_df, decile_df

    def _plot_decile_distribution(self, decile_df, top_n_deciles, n_deciles, save_dir):
        """Bar chart of average net value per decile, top deciles highlighted."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        deciles = decile_df['decile'].tolist()
        colors_a = [
            '#2E86AB' if d > (n_deciles - top_n_deciles) else '#A8DADC'
            for d in deciles
        ]

        # Panel 1: avg NV per decile (Offer Everyone)
        ax = axes[0]
        bars = ax.bar(deciles, decile_df['avg_nv_offer_everyone'], color=colors_a)
        ax.set_xlabel('Decile (1=lowest score, 10=highest)', fontsize=11)
        ax.set_ylabel('Average Net Value per Prospect ($)', fontsize=11)
        ax.set_title('Average Net Value by Decile\n(Offer Everyone)',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(deciles)
        ax.grid(True, alpha=0.3, axis='y')
        # Shade / label top deciles
        cutoff = n_deciles - top_n_deciles
        ax.axvline(x=cutoff + 0.5, color='red', linestyle='--', linewidth=1.5,
                   label=f'Top-{top_n_deciles} cut-off')
        ax.legend(fontsize=9)
        for bar, val in zip(bars, decile_df['avg_nv_offer_everyone']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'${val:,.0f}',
                    ha='center', va='bottom', fontsize=7.5, rotation=45)

        # Panel 2: prospect count per decile
        ax = axes[1]
        colors_b = [
            '#F18F01' if d > (n_deciles - top_n_deciles) else '#FFD6A5'
            for d in deciles
        ]
        bars2 = ax.bar(deciles, decile_df['n_prospects'], color=colors_b)
        ax.set_xlabel('Decile', fontsize=11)
        ax.set_ylabel('Number of Prospects', fontsize=11)
        ax.set_title('Prospect Count by Decile\n(with Top-3 highlighted)',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(deciles)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axvline(x=cutoff + 0.5, color='red', linestyle='--', linewidth=1.5,
                   label=f'Top-{top_n_deciles} cut-off')
        ax.legend(fontsize=9)
        for bar, val in zip(bars2, decile_df['n_prospects']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{val:,}',
                    ha='center', va='bottom', fontsize=8)

        plt.suptitle(
            f'Decile Distribution  —  highlights = top {top_n_deciles} deciles mailed',
            fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'decile_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Decile distribution chart : {save_path}")
        plt.close()

    def _plot_decile_strategy_comparison(self, strat_df, decile_df,
                                          top_n_deciles, n_deciles, save_dir):
        """Four-panel comparison: total NV, total cost, lift, lift-per-$ spent."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes_flat = axes.flatten()

        labels  = strat_df['strategy'].tolist()
        colors  = ['#2E86AB', '#F18F01']

        metrics = [
            ('total_net_value',           'Total Portfolio Net Value ($)',       'Total Net Value'),
            ('total_offer_cost',          'Total Offer Cost ($)',                'Total Offer Cost'),
            ('lift_vs_control',           'Lift vs. Control (No Offers) ($)',    'Lift vs. Control'),
            ('lift_per_dollar_spent',     'Lift per $ of Offer Cost',            'Lift per $'),
        ]

        for ax, (col, ylabel, title) in zip(axes_flat, metrics):
            vals = strat_df[col].tolist()
            bars = ax.bar(range(len(labels)), vals, color=colors)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9, rotation=10, ha='right')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, v in zip(bars, vals):
                fmt = f'${v:,.2f}' if col != 'lift_per_dollar_spent' else f'${v:.4f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        fmt,
                        ha='center', va='bottom', fontsize=9)

        # Overlay decile curve on the "Total Net Value" panel (panel 0)
        ax0   = axes_flat[0]
        ax0_r = ax0.twinx()
        cutoff = n_deciles - top_n_deciles
        cum_nv_everyone  = decile_df.sort_values('decile', ascending=False)[
            'total_nv_offer_everyone'].cumsum().values
        cum_nv_targeting = decile_df.sort_values('decile', ascending=False)[
            'total_nv_top3_targeting'].cumsum().values
        x_ticks = list(range(1, n_deciles + 1))
        ax0_r.plot(x_ticks, cum_nv_everyone[::-1],
                   'o--', color='steelblue', linewidth=1.5, markersize=4,
                   label='Cumul. NV — Everyone', alpha=0.8)
        ax0_r.plot(x_ticks, cum_nv_targeting[::-1],
                   's--', color='darkorange', linewidth=1.5, markersize=4,
                   label='Cumul. NV — Targeting', alpha=0.8)
        ax0_r.set_ylabel('Cumulative NV (from top decile)', fontsize=8)
        ax0_r.legend(fontsize=7, loc='lower right')

        plt.suptitle(
            'Strategy Performance Comparison\n'
            f'"Offer Everyone"  vs.  "Top-{top_n_deciles} Decile Targeting"',
            fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(save_dir, 'decile_vs_everyone_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Strategy comparison chart : {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # FULL REPORT  (baseline: observed features / no scenario override)
    # ------------------------------------------------------------------
    def generate_full_report(self, df, save_dir, cate_df=None, scenario=None):
        """
        Run the complete net value optimisation pipeline and save all outputs.

        Parameters
        ----------
        df       : combined insights DataFrame
        save_dir : output directory
        cate_df  : optional pre-computed scenario CATE DataFrame
        scenario : optional scenario dict for cost computation

        Returns
        -------
        (results_df, comparison_df, qini_data, auuc_metrics)
        """
        df          = self.compute_net_values(df, cate_df=cate_df, scenario=scenario)
        df          = self.assign_optimal_offers(df)
        comp_df     = self.evaluate_strategy_vs_benchmarks(df)
        qini_data   = self.compute_qini_curve_combined(df)
        auuc_metrics = self.compute_auuc_combined(qini_data)

        print("\n" + "="*60)
        print("AUUC METRICS  (Personalized Strategy)")
        print("="*60)
        print(f"  AUUC              : ${auuc_metrics['AUUC']:,.2f}")
        print(f"  AUUC (Random)     : ${auuc_metrics['AUUC_random']:,.2f}")
        print(f"  AUUC Lift         : ${auuc_metrics['AUUC_lift']:,.2f}")
        print(f"  Normalised AUUC   : {auuc_metrics['AUUC_normalized']:.2f}x")
        print("="*60 + "\n")

        # --- Plots ---
        self.plot_qini_curve_combined(
            qini_data, auuc_metrics,
            save_path=os.path.join(save_dir, 'personalized_qini_curve.png'))

        self.plot_cumulative_net_value(
            qini_data,
            save_path=os.path.join(save_dir, 'personalized_cumulative_net_value.png'))

        self.plot_offer_distribution(
            df,
            save_path=os.path.join(save_dir, 'optimal_offer_distribution.png'))

        self.plot_strategy_comparison(
            comp_df,
            save_path=os.path.join(save_dir, 'strategy_comparison_bar.png'))

        self.plot_net_value_by_offer(
            df,
            save_path=os.path.join(save_dir, 'net_value_by_offer_boxplot.png'))

        # --- CSV outputs ---
        df.to_csv(os.path.join(save_dir, 'net_value_strategy_results.csv'), index=False)
        comp_df.to_csv(os.path.join(save_dir, 'strategy_comparison.csv'), index=False)
        pd.DataFrame([auuc_metrics]).to_csv(
            os.path.join(save_dir, 'personalized_auuc_metrics.csv'), index=False)

        print("  Net value strategy CSV files saved.")
        return df, comp_df, qini_data, auuc_metrics


def optimize_offers(combined_insights_df, save_results_dir=None,
                    cate_df=None, scenario=None):
    """
    Convenience wrapper for the full net value optimisation (baseline run).

    Returns
    -------
    (NetValueOptimizer, results_df, comp_df, qini_data, auuc_metrics)
    """
    optimizer = NetValueOptimizer()
    results_df, comp_df, qini_data, auuc_metrics = optimizer.generate_full_report(
        combined_insights_df,
        save_dir  = save_results_dir or '.',
        cate_df   = cate_df,
        scenario  = scenario,
    )
    return optimizer, results_df, comp_df, qini_data, auuc_metrics


if __name__ == "__main__":
    print("Run pipeline.py to test the full net value workflow")
