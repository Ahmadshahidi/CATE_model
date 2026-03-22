"""
Inverse Probability of Treatment Weighting (IPTW)
==================================================
Estimates multi-arm propensity scores and computes stabilised (or raw)
IPTW weights for each observation.  The weights are then passed as
``sample_weight`` to the X-Learner's XGBoost base learners, making the
weighted sample equivalent to a pseudo-randomised population.

Algorithm
---------
1. Fit a multi-class propensity score model:
       ê(t | x) = P(T = t | X = x)
   using multinomial logistic regression (or one-vs-rest XGBoost).

2. Compute per-observation weights:
   - Stabilised (recommended):
       w_i = P(T = t_i)  /  ê(t_i | x_i)
     where P(T = t) is the marginal treatment probability.
   - Unstabilised:
       w_i = 1  /  ê(t_i | x_i)

3. Trim extreme weights at the configured percentile (each tail).

4. Save diagnostics:
   - iptw_weight_distribution.png   — per-arm weight histograms
   - iptw_love_plot_arm{N}.png      — weighted SMD comparison (Love plots)
   - iptw_balance_summary.csv       — full weighted balance table
   - iptw_effective_sample_sizes.csv — ESS = (Σw)² / Σw² per arm

Usage
-----
    from src.models.iptw import run_iptw
    result = run_iptw(X, treatment, save_results_dir=config.RESULTS_DIR)
    # result.weights  →  np.ndarray, shape (n_samples,)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class IPTWWeighting:
    """
    Inverse Probability of Treatment Weighting for multi-arm studies.

    Attributes
    ----------
    weights : np.ndarray, shape (n_samples,)
        Final (trimmed) IPTW weight for every observation in the
        dataset, in the original row order.
    propensity_scores : np.ndarray, shape (n_samples, n_arms_total)
        Estimated P(T=k | X) for every unit and every arm (columns
        ordered by sorted arm_id).
    arm_ids : list[int]
        Sorted list of all arm IDs (including control = 0).
    balance_summary : pd.DataFrame
        Per-arm, per-covariate unweighted and weighted SMD.
    ess : dict {arm_id: float}
        Effective Sample Size for the treated units in each arm.
    """

    def __init__(self,
                 ps_method: str    = None,
                 stabilized: bool  = None,
                 trim_pct: float   = None,
                 random_state: int = None):
        self.ps_method    = ps_method    if ps_method    is not None else config.IPTW_PS_METHOD
        self.stabilized   = stabilized   if stabilized   is not None else config.IPTW_STABILIZED
        self.trim_pct     = trim_pct     if trim_pct     is not None else config.IPTW_TRIM_PERCENTILE
        self.random_state = random_state if random_state is not None else config.IPTW_RANDOM_STATE

        self.weights           = None
        self.propensity_scores = None
        self.arm_ids           = []
        self.balance_summary   = None
        self.ess               = {}
        self._ps_model         = None
        self._scaler           = None

    # ------------------------------------------------------------------
    # Public fit entry point
    # ------------------------------------------------------------------
    def fit_transform(self, X: pd.DataFrame, treatment,
                      save_dir: str = None) -> 'IPTWWeighting':
        """
        Estimate propensity scores, compute IPTW weights, and save
        diagnostics.

        Parameters
        ----------
        X         : pd.DataFrame  Feature matrix (Boruta-SHAP selected).
        treatment : array-like    Multi-arm treatment indicator (0 = control).
        save_dir  : str           If provided, save plots/CSVs here.

        Returns
        -------
        self
        """
        print("\n" + "="*60)
        print("INVERSE PROBABILITY OF TREATMENT WEIGHTING (IPTW)")
        print("="*60)
        print(f"\n  PS estimator : {self.ps_method}")
        print(f"  Stabilised   : {self.stabilized}")
        print(f"  Trim %ile    : {self.trim_pct:.1f}%  (each tail)")

        t_arr = np.array(treatment, dtype=int)
        self.arm_ids = sorted(config.TREATMENT_COMPONENTS.keys())   # [0,1,2,3]

        n = len(t_arr)
        print(f"\n  Dataset: {n:,} observations, "
              f"{len(self.arm_ids)} arms  {self.arm_ids}")

        # ── Step 1: estimate multi-class propensity scores ────────────
        print("\n  Estimating multi-class propensity scores ...")
        self.propensity_scores = self._estimate_ps(X, t_arr)

        # ── Step 2: compute raw weights ───────────────────────────────
        # Marginal treatment probabilities (from sample frequencies)
        marginal_probs = {
            arm: (t_arr == arm).mean() for arm in self.arm_ids
        }

        weights_raw = np.zeros(n, dtype=float)
        for i, arm_id in enumerate(self.arm_ids):
            mask = t_arr == arm_id
            ps_col = self.propensity_scores[:, i]    # P(T=arm_id | X)
            ps_col_clipped = np.clip(ps_col, 1e-6, 1 - 1e-6)
            if self.stabilized:
                weights_raw[mask] = marginal_probs[arm_id] / ps_col_clipped[mask]
            else:
                weights_raw[mask] = 1.0 / ps_col_clipped[mask]

        # ── Step 3: trim extreme weights ──────────────────────────────
        if self.trim_pct > 0:
            lo = np.percentile(weights_raw, self.trim_pct)
            hi = np.percentile(weights_raw, 100.0 - self.trim_pct)
            weights_trimmed = np.clip(weights_raw, lo, hi)
            n_trimmed = ((weights_raw < lo) | (weights_raw > hi)).sum()
            print(f"\n  Weight trimming: clipped {n_trimmed:,} observations "
                  f"to [{lo:.4f}, {hi:.4f}]")
        else:
            weights_trimmed = weights_raw.copy()

        self.weights = weights_trimmed

        # ── Step 4: diagnostics ───────────────────────────────────────
        self._print_weight_summary(t_arr, weights_trimmed)
        self._compute_ess(t_arr, weights_trimmed)
        balance_rows = self._compute_balance(X, t_arr, weights_trimmed)
        self.balance_summary = pd.DataFrame(balance_rows)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._plot_weight_distributions(t_arr, weights_raw, weights_trimmed,
                                             save_dir)
            for arm_id in sorted(k for k in self.arm_ids if k != 0):
                self._plot_love(balance_rows, arm_id, save_dir)

            bal_path = os.path.join(save_dir, 'iptw_balance_summary.csv')
            self.balance_summary.to_csv(bal_path, index=False)
            print(f"\n  Balance summary saved to: {bal_path}")

            ess_df = pd.DataFrame([
                {'arm_id': arm_id,
                 'arm_name': config.TREATMENTS[arm_id],
                 'n_observed': int((t_arr == arm_id).sum()),
                 'ess': round(self.ess.get(arm_id, 0), 2)}
                for arm_id in self.arm_ids
            ])
            ess_path = os.path.join(save_dir, 'iptw_effective_sample_sizes.csv')
            ess_df.to_csv(ess_path, index=False)
            print(f"  Effective sample sizes saved to: {ess_path}")

        print(f"\n{'='*60}")
        print("IPTW COMPLETE")
        for arm_id in self.arm_ids:
            n_arm = (t_arr == arm_id).sum()
            ess_arm = self.ess.get(arm_id, np.nan)
            print(f"  Arm {arm_id} ({config.TREATMENTS[arm_id]:<10}): "
                  f"n={n_arm:>5,}  ESS={ess_arm:>7,.1f}  "
                  f"(ratio={ess_arm/n_arm:.2%})")
        print(f"{'='*60}\n")

        return self

    # ------------------------------------------------------------------
    # Propensity score estimation
    # ------------------------------------------------------------------
    def _estimate_ps(self, X: pd.DataFrame, t_arr: np.ndarray) -> np.ndarray:
        """
        Fit a multi-class classifier and return the full probability matrix
        P(T=k | X) with columns ordered by self.arm_ids.

        Returns
        -------
        np.ndarray, shape (n_samples, n_arms)
        """
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        if self.ps_method == 'xgboost':
            proba = self._xgboost_ps(X_sc, t_arr)
        else:
            proba = self._logistic_ps(X_sc, t_arr)

        # Ensure columns are ordered by self.arm_ids
        # sklearn's predict_proba returns columns ordered by model.classes_
        return proba

    def _logistic_ps(self, X_sc: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
        """Multinomial logistic regression (softmax)."""
        n_classes = len(self.arm_ids)
        solver = 'lbfgs' if n_classes <= 10 else 'saga'
        lr = LogisticRegression(
            solver=solver,
            max_iter=1000,
            C=1.0,
            random_state=self.random_state,
            n_jobs=-1,
        )
        lr.fit(X_sc, t_arr)
        self._ps_model = lr

        # Reorder columns to match self.arm_ids
        proba = lr.predict_proba(X_sc)          # columns = lr.classes_
        col_order = [list(lr.classes_).index(a) for a in self.arm_ids]
        proba = proba[:, col_order]

        for i, arm_id in enumerate(self.arm_ids):
            ps_col = proba[:, i]
            print(f"    Arm {arm_id} PS: "
                  f"range=[{ps_col.min():.4f}, {ps_col.max():.4f}]  "
                  f"mean={ps_col.mean():.4f}")
        return proba

    def _xgboost_ps(self, X_sc: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
        """One-vs-rest XGBoost propensity model."""
        try:
            from xgboost import XGBClassifier
            n_classes = len(self.arm_ids)
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                objective='multi:softprob',
                num_class=n_classes,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric='mlogloss',
            )
            # Re-map arm IDs to consecutive 0..K-1 for XGBoost
            arm_map   = {arm: idx for idx, arm in enumerate(self.arm_ids)}
            t_remapped = np.array([arm_map[a] for a in t_arr])
            model.fit(X_sc, t_remapped)
            self._ps_model = model

            proba = model.predict_proba(X_sc)   # columns 0..K-1 = arm_ids order

            for i, arm_id in enumerate(self.arm_ids):
                ps_col = proba[:, i]
                print(f"    Arm {arm_id} PS: "
                      f"range=[{ps_col.min():.4f}, {ps_col.max():.4f}]  "
                      f"mean={ps_col.mean():.4f}")
            return proba

        except Exception as exc:
            print(f"  ⚠  XGBoost PS failed ({exc}); falling back to logistic.")
            return self._logistic_ps(X_sc, t_arr)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _print_weight_summary(self, t_arr: np.ndarray,
                               weights: np.ndarray) -> None:
        print("\n  Weight summary (per arm):")
        print(f"  {'Arm':<10} {'n':>6}  {'min':>8}  {'p25':>8}  "
              f"{'median':>8}  {'p75':>8}  {'max':>8}  {'mean':>8}")
        print("  " + "-"*72)
        for arm_id in self.arm_ids:
            mask = t_arr == arm_id
            w    = weights[mask]
            print(f"  {config.TREATMENTS[arm_id]:<10} {mask.sum():>6,}  "
                  f"{w.min():>8.3f}  {np.percentile(w,25):>8.3f}  "
                  f"{np.median(w):>8.3f}  {np.percentile(w,75):>8.3f}  "
                  f"{w.max():>8.3f}  {w.mean():>8.3f}")

    def _compute_ess(self, t_arr: np.ndarray,
                      weights: np.ndarray) -> None:
        """Effective Sample Size = (Σw)² / Σw² for each arm."""
        for arm_id in self.arm_ids:
            mask = t_arr == arm_id
            w    = weights[mask]
            ess  = (w.sum() ** 2) / (w ** 2).sum() if len(w) > 0 else 0.0
            self.ess[arm_id] = ess

    def _compute_balance(self, X: pd.DataFrame, t_arr: np.ndarray,
                          weights: np.ndarray) -> list:
        """
        Compute Standardised Mean Differences (SMD) before and after
        IPTW weighting for each arm vs. control.
        """
        feat_cols  = X.columns.tolist()
        ctrl_mask  = t_arr == 0
        rows       = []
        arm_ids_no_ctrl = sorted(k for k in self.arm_ids if k != 0)

        for arm_id in arm_ids_no_ctrl:
            arm_mask = t_arr == arm_id
            sub_mask = ctrl_mask | arm_mask

            X_sub   = X.values[sub_mask]
            t_sub   = arm_mask[sub_mask]
            w_sub   = weights[sub_mask]

            t_idx = np.where(t_sub)[0]
            c_idx = np.where(~t_sub)[0]

            balanced_count = 0
            for j, feat in enumerate(feat_cols):
                col = X_sub[:, j]

                # Unweighted SMD
                mu_t_uw = col[t_idx].mean()
                mu_c_uw = col[c_idx].mean()
                sd_pool  = np.sqrt((col[t_idx].var() + col[c_idx].var()) / 2.0)
                smd_uw   = abs(mu_t_uw - mu_c_uw) / sd_pool if sd_pool > 0 else 0.0

                # Weighted SMD (IPTW-adjusted)
                w_t = w_sub[t_idx]
                w_c = w_sub[c_idx]

                wmu_t = np.average(col[t_idx], weights=w_t) if w_t.sum() > 0 else mu_t_uw
                wmu_c = np.average(col[c_idx], weights=w_c) if w_c.sum() > 0 else mu_c_uw
                wvar_t = (np.average((col[t_idx] - wmu_t) ** 2, weights=w_t)
                          if w_t.sum() > 0 else col[t_idx].var())
                wvar_c = (np.average((col[c_idx] - wmu_c) ** 2, weights=w_c)
                          if w_c.sum() > 0 else col[c_idx].var())
                wsd_pool = np.sqrt((wvar_t + wvar_c) / 2.0)
                smd_w    = abs(wmu_t - wmu_c) / wsd_pool if wsd_pool > 0 else 0.0

                balanced = smd_w < 0.1
                if balanced:
                    balanced_count += 1

                rows.append({
                    'arm_id':        arm_id,
                    'arm_name':      config.TREATMENTS[arm_id],
                    'feature':       feat,
                    'smd_unweighted': round(smd_uw, 4),
                    'smd_weighted':   round(smd_w, 4),
                    'balanced':       balanced,
                })

            print(f"  Arm {arm_id} ({config.TREATMENTS[arm_id]}): "
                  f"{balanced_count}/{len(feat_cols)} features |SMD| < 0.10 "
                  f"after IPTW")

        return rows

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def _plot_weight_distributions(self, t_arr: np.ndarray,
                                    weights_raw: np.ndarray,
                                    weights_trimmed: np.ndarray,
                                    save_dir: str):
        """Histogram of IPTW weights per arm (raw vs. trimmed)."""
        arm_ids_no_ctrl = sorted(k for k in self.arm_ids if k != 0)
        n_arms = len(arm_ids_no_ctrl)
        cmap   = plt.get_cmap('tab10')

        fig, axes = plt.subplots(2, n_arms,
                                  figsize=(5 * n_arms, 8),
                                  sharey=False)
        if n_arms == 1:
            axes = axes.reshape(2, 1)

        for col_i, arm_id in enumerate(arm_ids_no_ctrl):
            arm_mask = t_arr == arm_id
            color    = cmap(col_i)
            arm_label = config.TREATMENTS[arm_id]

            # Row 0: control weights in this arm's sub-sample
            ctrl_mask = t_arr == 0
            for row_i, (mask, label) in enumerate([
                (ctrl_mask, 'Control'),
                (arm_mask,  arm_label),
            ]):
                ax_raw  = axes[0, col_i]
                ax_trim = axes[1, col_i]
                if row_i == 0:
                    ax_raw.hist(weights_raw[mask],  bins=50, alpha=0.5,
                                color='grey', label='Control', density=True)
                    ax_trim.hist(weights_trimmed[mask], bins=50, alpha=0.5,
                                 color='grey', label='Control', density=True)
                else:
                    ax_raw.hist(weights_raw[mask],  bins=50, alpha=0.7,
                                color=color, label=arm_label, density=True)
                    ax_trim.hist(weights_trimmed[mask], bins=50, alpha=0.7,
                                 color=color, label=arm_label, density=True)

            axes[0, col_i].set_title(f'Arm {arm_id}: {arm_label}\n(Raw weights)',
                                      fontsize=10, fontweight='bold')
            axes[1, col_i].set_title(f'Arm {arm_id}: {arm_label}\n(Trimmed weights)',
                                      fontsize=10, fontweight='bold')
            for ax in (axes[0, col_i], axes[1, col_i]):
                ax.set_xlabel('IPTW Weight', fontsize=9)
                ax.set_ylabel('Density', fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.suptitle('IPTW Weight Distributions (Before and After Trimming)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(save_dir, 'iptw_weight_distribution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Weight distribution plot saved: {path}")

    def _plot_love(self, balance_rows: list, arm_id: int, save_dir: str):
        """Love plot: unweighted vs. weighted |SMD| for key covariates."""
        df = pd.DataFrame(balance_rows)
        df = df[df['arm_id'] == arm_id].copy()
        if df.empty:
            return

        key_covs = [c for c in config.PSM_KEY_COVARIATES
                    if c in df['feature'].values]
        if not key_covs:
            # Fall back to top-20 by unweighted SMD
            key_covs = (df.nlargest(20, 'smd_unweighted')
                          ['feature'].tolist())

        df = (df[df['feature'].isin(key_covs)]
                .sort_values('smd_unweighted', ascending=True))

        fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))
        y_pos = np.arange(len(df))

        ax.barh(y_pos - 0.2, df['smd_unweighted'], 0.38,
                color='#E74C3C', alpha=0.75, label='Unweighted')
        ax.barh(y_pos + 0.2, df['smd_weighted'],   0.38,
                color='#2E86AB', alpha=0.75, label='IPTW-weighted')

        ax.axvline(x=0.1, color='grey', linestyle='--', linewidth=1.2,
                   label='|SMD| = 0.10 threshold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'], fontsize=9)
        ax.set_xlabel('Absolute Standardised Mean Difference (SMD)', fontsize=10)
        arm_label = config.TREATMENTS[arm_id]
        ax.set_title(f'IPTW Love Plot — Arm {arm_id} ({arm_label}) vs. Control\n'
                     f'Stabilised={self.stabilized}  Trim={self.trim_pct}%',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        path = os.path.join(save_dir, f'iptw_love_plot_arm{arm_id}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    IPTW Love plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_iptw(X: pd.DataFrame,
             treatment,
             save_results_dir: str = None) -> IPTWWeighting:
    """
    Compute IPTW weights for the given feature matrix and treatment vector.

    Parameters
    ----------
    X                : pd.DataFrame  Boruta-SHAP selected features.
    treatment        : array-like    Multi-arm treatment vector (0 = control).
    save_results_dir : str           Directory for diagnostic plots/CSVs.

    Returns
    -------
    IPTWWeighting  (fitted)
        Access the per-row weights via  result.weights  (np.ndarray).
    """
    if not isinstance(treatment, pd.Series):
        treatment = pd.Series(treatment)

    iptw = IPTWWeighting()
    iptw.fit_transform(X, treatment, save_dir=save_results_dir)
    return iptw


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────
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

    result = run_iptw(X2, df['treatment'],
                      save_results_dir=config.RESULTS_DIR)

    print(f"\nWeights shape : {result.weights.shape}")
    print(f"Weights range : [{result.weights.min():.4f}, {result.weights.max():.4f}]")
    print(f"ESS per arm   : {result.ess}")
