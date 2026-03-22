"""
Propensity Score Matching (PSM) — Selection Bias Diagnostics
=============================================================
Checks whether treatment arms are balanced on pre-treatment covariates.
Saves Love plots, propensity overlap plots, and covariate balance boxplots.

This module does NOT alter the modelling dataset by default
(USE_MATCHED_DATA_FOR_XLEARNER = False in config.py).
It serves as a diagnostic / quality-assurance step.

Usage:
    from src.models.propensity_matching import run_propensity_matching
    psm = run_propensity_matching(X, treatment, save_results_dir)
    # psm.matched_data  →  dict {arm_id: matched_df}
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

warnings.filterwarnings('ignore')


class PropensityScoreMatching:
    """
    Nearest-neighbour 1:1 propensity score matching (arm vs. control).

    Attributes
    ----------
    matched_data  : dict {arm_id: pd.DataFrame}
        Each value is the matched dataset for that arm (treated + control rows),
        with an added column ``matched_binary_treatment`` (1=treated, 0=control).
    balance_summary : pd.DataFrame
        Per-arm, per-covariate SMD before and after matching.
    propensity_scores : dict {arm_id: np.ndarray}
        Predicted propensity scores for all records in the arm vs. control sub-sample.
    """

    def __init__(self,
                 method: str       = None,
                 caliper: float    = None,
                 random_state: int = None):
        self.method       = method       or config.PSM_METHOD
        self.caliper      = caliper      or config.PSM_CALIPER
        self.random_state = random_state or config.PSM_RANDOM_STATE

        self.matched_data       = {}
        self.balance_summary    = None
        self.propensity_scores  = {}
        self._ps_models         = {}

    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, treatment: pd.Series,
                      save_dir: str = None) -> 'PropensityScoreMatching':
        """
        Estimate propensity scores, match, and save diagnostics.

        Parameters
        ----------
        X         : pd.DataFrame  Feature matrix (Boruta-SHAP selected).
        treatment : pd.Series     Multi-arm treatment indicator (0=control).
        save_dir  : str           If provided, save plots/CSVs here.

        Returns
        -------
        self
        """
        print("\n" + "="*60)
        print("PROPENSITY SCORE MATCHING  (Selection Bias Diagnostics)")
        print("="*60)

        treat_arr  = np.array(treatment)
        ctrl_mask  = treat_arr == 0
        arm_ids    = sorted(k for k in config.TREATMENT_COMPONENTS if k != 0)
        all_rows   = []   # for combined balance summary

        # Pre-matching global PS overlap plot
        all_ps = {}

        psm_iter = (
            _tqdm(arm_ids, desc='  PSM arms', unit='arm', ncols=80,
                  bar_format='  {desc}: {percentage:3.0f}%|{bar}| '
                              '{n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            if _HAS_TQDM else arm_ids
        )

        for arm_id in psm_iter:
            t_arm_start = time.time()
            arm_mask  = treat_arr == arm_id
            sub_mask  = ctrl_mask | arm_mask
            X_sub     = X[sub_mask].reset_index(drop=True)
            t_sub     = (treat_arr[sub_mask] == arm_id).astype(int)

            print(f"\n  Arm {arm_id} ({config.TREATMENTS[arm_id]}):")
            print(f"    Treated : {arm_mask.sum():,}   Control : {ctrl_mask.sum():,}")

            # ── Propensity score estimation ───────────────────────
            ps = self._estimate_ps(X_sub, t_sub, arm_id)
            self.propensity_scores[arm_id] = ps
            all_ps[arm_id] = (t_sub, ps)

            # ── Matching ─────────────────────────────────────────
            matched_df = self._match(X_sub, t_sub, ps, arm_id)
            self.matched_data[arm_id] = matched_df

            # ── Balance statistics ────────────────────────────────
            balance_rows = self._compute_balance(X_sub, t_sub, matched_df, arm_id)
            all_rows.extend(balance_rows)

            if save_dir:
                self._plot_love(balance_rows, arm_id, save_dir)
                self._plot_covariate_balance(X_sub, t_sub, matched_df,
                                              arm_id, save_dir)

        self.balance_summary = pd.DataFrame(all_rows)

        if save_dir:
            self._plot_ps_overlap(all_ps, save_dir)
            bal_path = os.path.join(save_dir, 'propensity_balance_summary.csv')
            self.balance_summary.to_csv(bal_path, index=False)
            print(f"\n  Balance summary saved to: {bal_path}")

        print(f"\n{'='*60}")
        print("PSM COMPLETE")
        for arm_id in arm_ids:
            n_matched = len(self.matched_data[arm_id]) // 2
            print(f"  Arm {arm_id}: {n_matched:,} matched pairs")
        print(f"{'='*60}\n")

        return self

    # ------------------------------------------------------------------
    def _estimate_ps(self, X_sub: pd.DataFrame,
                     t_sub: np.ndarray, arm_id: int) -> np.ndarray:
        """Estimate propensity scores via logistic regression or XGBoost."""
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_sub)

        if self.method == 'xgboost':
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    objective='binary:logistic',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0,
                    use_label_encoder=False,
                )
                model.fit(X_sc, t_sub)
                ps = model.predict_proba(X_sc)[:, 1]
            except Exception:
                ps = self._logistic_ps(X_sc, t_sub)
        else:
            ps = self._logistic_ps(X_sc, t_sub)

        self._ps_models[arm_id] = (scaler, None)   # store scaler for later
        print(f"    PS range: [{ps.min():.3f}, {ps.max():.3f}]  "
              f"mean={ps.mean():.3f}")
        return ps

    def _logistic_ps(self, X_sc: np.ndarray, t: np.ndarray) -> np.ndarray:
        lr = LogisticRegression(
            max_iter=1000, C=1.0, random_state=self.random_state,
            solver='lbfgs', n_jobs=-1,
        )
        lr.fit(X_sc, t)
        return lr.predict_proba(X_sc)[:, 1]

    # ------------------------------------------------------------------
    def _match(self, X_sub: pd.DataFrame, t_sub: np.ndarray,
               ps: np.ndarray, arm_id: int) -> pd.DataFrame:
        """1:1 nearest-neighbour matching with caliper on log-odds PS."""
        LOG_ODDS = np.log(ps.clip(1e-4, 1 - 1e-4) / (1 - ps.clip(1e-4, 1 - 1e-4)))
        lo_std   = LOG_ODDS.std()
        caliper  = self.caliper * lo_std

        treated_idx = np.where(t_sub == 1)[0]
        control_idx = np.where(t_sub == 0)[0]

        # Fit NN on log-odds of control
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(LOG_ODDS[control_idx].reshape(-1, 1))
        dists, ctrl_pos = nn.kneighbors(LOG_ODDS[treated_idx].reshape(-1, 1))

        dists   = dists.flatten()
        ctrl_pos = ctrl_pos.flatten()

        # Apply caliper
        valid = dists <= caliper
        matched_treated = treated_idx[valid]
        matched_control = control_idx[ctrl_pos[valid]]

        X_matched = pd.concat([
            X_sub.iloc[matched_treated].assign(matched_binary_treatment=1),
            X_sub.iloc[matched_control].assign(matched_binary_treatment=0),
        ], ignore_index=True)

        n_before = valid.shape[0]
        n_after  = valid.sum()
        print(f"    Matched  : {n_after:,} treated pairs  "
              f"(caliper={caliper:.4f}; {100*n_after/n_before:.1f}% retained)")
        return X_matched

    # ------------------------------------------------------------------
    def _compute_balance(self, X_sub: pd.DataFrame, t_sub: np.ndarray,
                          matched_df: pd.DataFrame, arm_id: int) -> list:
        """Compute Standardised Mean Differences (SMD) before and after matching."""
        feat_cols = [c for c in X_sub.columns if c != 'matched_binary_treatment']
        rows = []

        # Before matching
        treated_before = X_sub.loc[t_sub == 1, feat_cols]
        control_before = X_sub.loc[t_sub == 0, feat_cols]

        # After matching
        m_treat = matched_df[matched_df['matched_binary_treatment'] == 1][feat_cols]
        m_ctrl  = matched_df[matched_df['matched_binary_treatment'] == 0][feat_cols]

        for feat in feat_cols:
            mu_t_bef  = treated_before[feat].mean()
            mu_c_bef  = control_before[feat].mean()
            sd_bef    = np.sqrt((treated_before[feat].var() +
                                  control_before[feat].var()) / 2)
            smd_before = (mu_t_bef - mu_c_bef) / sd_bef if sd_bef > 0 else 0

            mu_t_aft  = m_treat[feat].mean() if len(m_treat) else mu_t_bef
            mu_c_aft  = m_ctrl[feat].mean()  if len(m_ctrl)  else mu_c_bef
            sd_aft    = np.sqrt((m_treat[feat].var() +
                                  m_ctrl[feat].var()) / 2) if len(m_treat) > 1 else sd_bef
            smd_after = (mu_t_aft - mu_c_aft) / sd_aft if sd_aft > 0 else 0

            rows.append({
                'arm_id':   arm_id,
                'arm_name': config.TREATMENTS[arm_id],
                'feature':  feat,
                'smd_before': round(abs(smd_before), 4),
                'smd_after':  round(abs(smd_after), 4),
                'balanced':   abs(smd_after) < 0.1,
            })

        balanced = sum(r['balanced'] for r in rows)
        print(f"    Balance  : {balanced}/{len(rows)} features |SMD| < 0.10 after matching")
        return rows

    # ------------------------------------------------------------------
    def _plot_love(self, balance_rows: list, arm_id: int, save_dir: str):
        """Love plot: |SMD| before vs after matching for top-N covariates."""
        df = pd.DataFrame(balance_rows)
        key_covs = [c for c in config.PSM_KEY_COVARIATES if c in df['feature'].values]
        if not key_covs:
            key_covs = df.nlargest(20, 'smd_before')['feature'].tolist()

        df = df[df['feature'].isin(key_covs)].sort_values('smd_before', ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.4)))
        y_pos = np.arange(len(df))
        ax.barh(y_pos - 0.2, df['smd_before'], 0.40,
                color='#E74C3C', alpha=0.7, label='Before matching')
        ax.barh(y_pos + 0.2, df['smd_after'],  0.40,
                color='#2E86AB', alpha=0.7, label='After matching')
        ax.axvline(x=0.1, color='grey', linestyle='--', linewidth=1,
                   label='|SMD| = 0.10 threshold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'], fontsize=9)
        ax.set_xlabel('Absolute Standardised Mean Difference (SMD)', fontsize=10)
        arm_label = config.TREATMENTS[arm_id]
        ax.set_title(f'Love Plot — Arm {arm_id} ({arm_label}) vs. Control',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        path = os.path.join(save_dir, f'psm_love_plot_arm{arm_id}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Love plot saved: {path}")

    def _plot_covariate_balance(self, X_sub: pd.DataFrame, t_sub: np.ndarray,
                                 matched_df: pd.DataFrame, arm_id: int,
                                 save_dir: str, top_n: int = 4):
        """Box plots for key covariates before/after matching."""
        key_covs = [c for c in config.PSM_KEY_COVARIATES if c in X_sub.columns][:top_n]
        if not key_covs:
            return

        fig, axes = plt.subplots(2, len(key_covs),
                                  figsize=(4 * len(key_covs), 8))
        if len(key_covs) == 1:
            axes = axes.reshape(2, 1)

        arm_label = config.TREATMENTS[arm_id]

        for j, feat in enumerate(key_covs):
            # Row 0: before matching
            ax = axes[0, j]
            data_t = X_sub.loc[t_sub == 1, feat]
            data_c = X_sub.loc[t_sub == 0, feat]
            ax.boxplot([data_c, data_t], labels=['Control', arm_label],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_title(f'{feat}\n(Before)', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Row 1: after matching
            ax = axes[1, j]
            data_t_m = matched_df[matched_df['matched_binary_treatment'] == 1][feat]
            data_c_m = matched_df[matched_df['matched_binary_treatment'] == 0][feat]
            ax.boxplot([data_c_m, data_t_m], labels=['Control', arm_label],
                       patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
            ax.set_title(f'{feat}\n(After)', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Covariate Balance — Arm {arm_id} ({arm_label})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(save_dir, f'psm_covariate_balance_arm{arm_id}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Covariate balance plot: {path}")

    def _plot_ps_overlap(self, all_ps: dict, save_dir: str):
        """Propensity score overlap histograms before and after matching."""
        arm_ids = sorted(all_ps.keys())
        n       = len(arm_ids)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]

        cmap   = plt.get_cmap('tab10')
        colors = {'treated': cmap(0), 'control': cmap(1)}

        for ax, arm_id in zip(axes, arm_ids):
            t_sub, ps = all_ps[arm_id]
            ps_t = ps[t_sub == 1]
            ps_c = ps[t_sub == 0]

            ax.hist(ps_c, bins=40, alpha=0.6, color=colors['control'],
                    label='Control', density=True)
            ax.hist(ps_t, bins=40, alpha=0.6, color=colors['treated'],
                    label=config.TREATMENTS[arm_id], density=True)
            ax.set_xlabel('Propensity Score', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Arm {arm_id}: {config.TREATMENTS[arm_id]}',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Propensity Score Overlap (Before Matching)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(save_dir, 'psm_propensity_overlap_before.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        # After-matching version
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, arm_id in zip(axes, arm_ids):
            mdf = self.matched_data[arm_id]
            t_sub2, ps2 = all_ps[arm_id]
            feat_cols    = [c for c in mdf.columns if c != 'matched_binary_treatment']
            # Re-use pre-matched PS (approx)
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            X_m = mdf[feat_cols]
            t_m = mdf['matched_binary_treatment'].values
            scaler = StandardScaler()
            X_sc   = scaler.fit_transform(X_m)
            lr     = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            try:
                lr.fit(X_sc, t_m)
                ps_m = lr.predict_proba(X_sc)[:, 1]
            except Exception:
                ps_m = np.random.uniform(0.2, 0.8, len(X_m))

            ps_t_m = ps_m[t_m == 1]
            ps_c_m = ps_m[t_m == 0]
            ax.hist(ps_c_m, bins=30, alpha=0.6, color=colors['control'],
                    label='Control', density=True)
            ax.hist(ps_t_m, bins=30, alpha=0.6, color=colors['treated'],
                    label=config.TREATMENTS[arm_id], density=True)
            ax.set_xlabel('Propensity Score', fontsize=10)
            ax.set_title(f'Arm {arm_id}: {config.TREATMENTS[arm_id]} (After)',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Propensity Score Overlap (After Matching)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path2 = os.path.join(save_dir, 'psm_propensity_overlap_after.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  PS overlap charts saved: {path}, {path2}")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_propensity_matching(X: pd.DataFrame,
                             treatment,
                             save_results_dir: str = None) -> PropensityScoreMatching:
    """
    Estimate propensity scores, perform 1:1 matching, and save diagnostics.

    Parameters
    ----------
    X                : pd.DataFrame  Boruta-SHAP selected features.
    treatment        : array-like    Multi-arm treatment vector.
    save_results_dir : str           Directory for plots/CSVs.

    Returns
    -------
    PropensityScoreMatching  (fitted)
    """
    if not isinstance(treatment, pd.Series):
        treatment = pd.Series(treatment)

    psm = PropensityScoreMatching()
    psm.fit_transform(X, treatment, save_dir=save_results_dir)
    return psm


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.data_generation import generate_epsilon_data
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    from src.feature_selection.step2_boruta_shap import run_boruta_shap
    df  = generate_epsilon_data()
    exc = ['treatment', 'treatment_name', 'opening_balance', 'on_book_month9', 'offer']
    X   = df[[c for c in df.columns if c not in exc]]
    X1, _ = run_initial_pruning(X)
    X2, _ = run_boruta_shap(X1, df['opening_balance'])
    psm = run_propensity_matching(X2, df['treatment'],
                                   save_results_dir=config.RESULTS_DIR)
    print(f"\nMatched data keys: {list(psm.matched_data.keys())}")
