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


# ---------------------------------------------------------------------------
def _encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* ready for sklearn estimators
    (StandardScaler, LogisticRegression, XGBoost):

    1. Categorical / object columns → integer codes  (NaN → -1)
    2. Numeric columns with NaN     → imputed per config.PS_NUMERIC_NAN_IMPUTE
         'median'  → column median  (default)
         'mean'    → column mean
         <number>  → fixed constant (e.g. 0)

    Both steps are required so that StandardScaler and linear models never
    receive non-finite values.  CatBoost paths do NOT call this function.
    """
    out = df.copy()

    # Step 1 — encode categoricals
    for col in out.columns:
        if (pd.api.types.is_categorical_dtype(out[col])
                or pd.api.types.is_object_dtype(out[col])):
            out[col] = pd.Categorical(out[col]).codes.astype('int16')

    # Step 2 — impute remaining NaN in numeric columns
    num_cols_with_nan = [c for c in out.columns if out[c].isna().any()]
    if num_cols_with_nan:
        strategy = config.PS_NUMERIC_NAN_IMPUTE
        if strategy == 'median':
            fill_values = out[num_cols_with_nan].median()
        elif strategy == 'mean':
            fill_values = out[num_cols_with_nan].mean()
        else:
            fill_values = float(strategy)
        out[num_cols_with_nan] = out[num_cols_with_nan].fillna(fill_values)

    return out
# ---------------------------------------------------------------------------


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class IPTWWeighting:
    """
    Propensity-score weighting for multi-arm studies.

    Supports two weighting schemes via `weighting_type`:

    'iptw'    — Inverse Probability of Treatment Weighting.
                w_i = P(T=k) / P(T=k | X)  (stabilised, recommended)
                     or  1 / P(T=k | X)    (unstabilised).
                Variance can blow up when propensity scores are near 0/1
                (common with heavily skewed arm sizes).

    'overlap' — Overlap Weighting (Li, Morgan & Zaslavsky 2018).
                w_i = h(x_i) / e_k(x_i)
                where h(x) = 1 / Σ_k(1/e_k(x))  (harmonic mean of PS).
                Weights are bounded in (0, 1], so variance is always
                finite — no trimming required for skewed data.
                Estimates the ATE on the overlap population (units
                where treatment assignment was genuinely ambiguous).

    Attributes
    ----------
    weights : np.ndarray, shape (n_samples,)
        Final (trimmed) weight for every observation in the dataset,
        in the original row order.
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
                 weighting_type: str = 'iptw',
                 ps_method: str    = None,
                 stabilized: bool  = None,
                 trim_pct: float   = None,
                 random_state: int = None):
        if weighting_type not in ('iptw', 'overlap'):
            raise ValueError(f"weighting_type must be 'iptw' or 'overlap', got '{weighting_type}'")
        self.weighting_type = weighting_type

        if weighting_type == 'overlap':
            self.ps_method    = ps_method    if ps_method    is not None else config.OVERLAP_PS_METHOD
            self.trim_pct     = trim_pct     if trim_pct     is not None else config.OVERLAP_TRIM_PERCENTILE
            self.stabilized   = False        # not applicable for overlap
        else:
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
        Estimate propensity scores, compute weights, and save diagnostics.

        Parameters
        ----------
        X         : pd.DataFrame  Feature matrix (Boruta-SHAP selected).
        treatment : array-like    Multi-arm treatment indicator (0 = control).
        save_dir  : str           If provided, save plots/CSVs here.

        Returns
        -------
        self
        """
        method_label = ('OVERLAP WEIGHTING' if self.weighting_type == 'overlap'
                        else 'INVERSE PROBABILITY OF TREATMENT WEIGHTING (IPTW)')
        print("\n" + "="*60)
        print(method_label)
        print("="*60)
        print(f"\n  Weighting type : {self.weighting_type.upper()}")
        print(f"  PS estimator   : {self.ps_method}")
        if self.weighting_type == 'iptw':
            print(f"  Stabilised     : {self.stabilized}")
        print(f"  Trim %ile      : {self.trim_pct:.1f}%  (each tail)")

        t_arr = np.array(treatment, dtype=int)
        self.arm_ids = sorted(config.TREATMENT_COMPONENTS.keys())   # [0,1,2,3]

        n = len(t_arr)
        print(f"\n  Dataset: {n:,} observations, "
              f"{len(self.arm_ids)} arms  {self.arm_ids}")

        # ── Step 1: estimate multi-class propensity scores ────────────
        print("\n  Estimating multi-class propensity scores ...")
        self.propensity_scores = self._estimate_ps(X, t_arr)

        # ── Step 2: compute raw weights ───────────────────────────────
        weights_raw = np.zeros(n, dtype=float)

        if self.weighting_type == 'overlap':
            # Overlap weights (Li, Morgan & Zaslavsky 2018):
            #   h(x) = 1 / Σ_k(1/e_k(x))   ← harmonic mean of PS across arms
            #   w_i  = h(x_i) / e_k(x_i)   ← for unit i assigned to arm k
            # Weights are in (0, 1] so variance is always finite — no need
            # for aggressive trimming even with severely skewed arm sizes.
            ps_clipped = np.clip(self.propensity_scores, 1e-6, 1 - 1e-6)
            harmonic_mean = 1.0 / np.sum(1.0 / ps_clipped, axis=1)   # shape (n,)
            for i, arm_id in enumerate(self.arm_ids):
                mask = t_arr == arm_id
                weights_raw[mask] = harmonic_mean[mask] / ps_clipped[mask, i]
        else:
            # IPTW weights:  P(T=k) / P(T=k|X)  (stabilised)  or  1/P(T=k|X)
            marginal_probs = {arm: (t_arr == arm).mean() for arm in self.arm_ids}
            for i, arm_id in enumerate(self.arm_ids):
                mask = t_arr == arm_id
                ps_col = self.propensity_scores[:, i]    # P(T=arm_id | X)
                ps_col_clipped = np.clip(ps_col, 1e-6, 1 - 1e-6)
                if self.stabilized:
                    weights_raw[mask] = marginal_probs[arm_id] / ps_col_clipped[mask]
                else:
                    weights_raw[mask] = 1.0 / ps_col_clipped[mask]

        # ── Step 3: trim extreme weights per arm ─────────────────────
        # Trimming is done within each arm separately so that rare arms
        # (which produce large weights) are not over-trimmed by the
        # global percentiles driven by dominant arms.
        weights_trimmed = weights_raw.copy()
        if self.trim_pct > 0:
            n_trimmed_total = 0
            for arm_id in self.arm_ids:
                arm_mask = t_arr == arm_id
                arm_w    = weights_raw[arm_mask]
                lo = np.percentile(arm_w, self.trim_pct)
                hi = np.percentile(arm_w, 100.0 - self.trim_pct)
                clipped  = np.clip(arm_w, lo, hi)
                n_arm_trimmed = ((arm_w < lo) | (arm_w > hi)).sum()
                weights_trimmed[arm_mask] = clipped
                n_trimmed_total += n_arm_trimmed
                print(f"  Arm {arm_id} weight trim: "
                      f"[{lo:.4f}, {hi:.4f}]  clipped={n_arm_trimmed:,}")
            print(f"\n  Total observations trimmed: {n_trimmed_total:,}")

        # ── Step 4: entropy balancing refinement (optional) ──────────
        weights_pre_ebal = None
        if getattr(config, 'BALANCE_REFINE', False):
            weights_pre_ebal = weights_trimmed.copy()
            weights_trimmed  = self._entropy_balance_refine(
                X, t_arr, weights_trimmed)

        self.weights = weights_trimmed

        # ── Step 5: diagnostics ───────────────────────────────────────
        self._print_weight_summary(t_arr, weights_trimmed)
        self._compute_ess(t_arr, weights_trimmed)
        balance_rows = self._compute_balance(X, t_arr, weights_trimmed,
                                              weights_pre_refine=weights_pre_ebal)
        self.balance_summary = pd.DataFrame(balance_rows)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._plot_weight_distributions(t_arr, weights_raw, weights_trimmed,
                                             save_dir)
            for arm_id in sorted(k for k in self.arm_ids if k != 0):
                self._plot_love(balance_rows, arm_id, save_dir)

            prefix = 'overlap' if self.weighting_type == 'overlap' else 'iptw'
            bal_path = os.path.join(save_dir, f'{prefix}_balance_summary.csv')
            self.balance_summary.to_csv(bal_path, index=False)
            print(f"\n  Balance summary saved to: {bal_path}")

            ess_df = pd.DataFrame([
                {'arm_id': arm_id,
                 'arm_name': config.TREATMENTS[arm_id],
                 'n_observed': int((t_arr == arm_id).sum()),
                 'ess': round(self.ess.get(arm_id, 0), 2)}
                for arm_id in self.arm_ids
            ])
            ess_path = os.path.join(save_dir, f'{prefix}_effective_sample_sizes.csv')
            ess_df.to_csv(ess_path, index=False)
            print(f"  Effective sample sizes saved to: {ess_path}")

        done_label = 'OVERLAP WEIGHTING COMPLETE' if self.weighting_type == 'overlap' else 'IPTW COMPLETE'
        print(f"\n{'='*60}")
        print(done_label)
        for arm_id in self.arm_ids:
            n_arm = (t_arr == arm_id).sum()
            ess_arm = self.ess.get(arm_id, np.nan)
            print(f"  Arm {arm_id} ({config.TREATMENTS[arm_id]:<10}): "
                  f"n={n_arm:>5,}  ESS={ess_arm:>7,.1f}  "
                  f"(ratio={ess_arm/n_arm:.2%})")
        print(f"{'='*60}\n")

        return self

    # ------------------------------------------------------------------
    # Entropy balancing refinement
    # ------------------------------------------------------------------
    def _entropy_balance_refine(self, X: pd.DataFrame,
                                 t_arr: np.ndarray,
                                 weights_init: np.ndarray) -> np.ndarray:
        """
        Post-process weights via entropy balancing (ebal-py, Hainmueller 2012).

        Two modes:
        - 'ATC' (default): per-arm, reweights each treated arm to match control
        - 'ATT': global, reweights control group to match all treated arms combined
        """
        from ebal import ebal_bin

        threshold = getattr(config, 'BALANCE_REFINE_SMD_THRESHOLD', 0.10)
        tolerance = getattr(config, 'BALANCE_REFINE_TOLERANCE', 1e-4)
        estimand  = getattr(config, 'BALANCE_REFINE_ESTIMAND', 'ATC').upper()
        weights   = weights_init.copy()
        ctrl_mask = t_arr == 0
        feat_cols = X.columns.tolist()

        print(f"\n  Entropy balancing refinement  (mode={estimand}, threshold |SMD| > {threshold})")

        # ── ATT mode: reweight control to match all treated arms combined ──
        if estimand == 'ATT':
            treated_mask = t_arr != 0
            ctrl_idx = np.where(ctrl_mask)[0]
            trt_idx  = np.where(treated_mask)[0]

            if len(trt_idx) == 0 or len(ctrl_idx) == 0:
                print("    WARNING: ATT mode requires both treated and control observations")
                return weights

            # Numeric columns only (skip categorical for ebal compatibility)
            numeric_cols = [i for i, col in enumerate(feat_cols)
                           if not (isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object)]
            if not numeric_cols:
                print("    WARNING: no numeric features found for ATT balance")
                return weights

            X_numeric = X.values[:, numeric_cols].copy().astype(float)

            # Impute NaNs
            strategy = getattr(config, 'PS_NUMERIC_NAN_IMPUTE', 'median')
            for j in range(X_numeric.shape[1]):
                col = X_numeric[:, j]
                if np.isnan(col).any():
                    fill = (np.nanmedian(col) if strategy == 'median'
                            else np.nanmean(col) if strategy == 'mean'
                            else float(strategy))
                    X_numeric[:, j] = np.where(np.isnan(col), fill, col)

            T_global = treated_mask.astype(int)  # 1=treated, 0=control
            Y_dummy  = np.zeros(len(t_arr))
            w_ctrl_init = weights[ctrl_mask].copy()

            # ── remove collinear features from control matrix before ebal ──
            # ebal raises ValueError if the control X has zero-variance or
            # near-perfectly-correlated columns (singular constraint matrix).
            X_ctrl_only = X_numeric[T_global == 0]
            var_ctrl = X_ctrl_only.var(axis=0)
            keep_local = np.where(var_ctrl > 1e-8)[0]
            n_dropped_var = X_numeric.shape[1] - len(keep_local)
            if n_dropped_var:
                print(f"    Dropped {n_dropped_var} near-zero-variance feature(s) from control")

            if len(keep_local) > 1:
                corr_mat = np.abs(np.corrcoef(X_ctrl_only[:, keep_local].T))
                np.fill_diagonal(corr_mat, 0.0)
                drop_corr: set = set()
                for ci in range(corr_mat.shape[0]):
                    if ci not in drop_corr:
                        for cj in range(ci + 1, corr_mat.shape[1]):
                            if corr_mat[ci, cj] > 0.95:
                                drop_corr.add(cj)
                if drop_corr:
                    keep_local = keep_local[[i for i in range(len(keep_local))
                                             if i not in drop_corr]]
                    print(f"    Dropped {len(drop_corr)} near-collinear feature(s) from control")

            # map local indices back to numeric_cols for SMD reporting
            ebal_cols_local = keep_local          # indices into X_numeric columns
            ebal_cols_global = [numeric_cols[i] for i in ebal_cols_local]  # indices into feat_cols
            X_ebal = X_numeric[:, ebal_cols_local]

            try:
                e = ebal_bin(
                    max_iterations=500,
                    constraint_tolerance=tolerance,
                    print_level=0,
                    effect='ATT',
                    PCA=False,
                )
                out = e.ebalance(T_global, X_ebal, Y_dummy, base_weight=w_ctrl_init)
            except Exception as exc:
                print(f"    WARNING: ebal(ATT) raised {type(exc).__name__}: {exc} "
                      f"— original weights kept")
                return weights

            if out['converged']:
                # ATT reweights T=0 (control); extract those weights
                new_ctrl_w = out['w'][T_global == 0]
                # preserve weight sum
                new_ctrl_w = new_ctrl_w * (w_ctrl_init.sum() / new_ctrl_w.sum())
                new_ctrl_w = np.maximum(new_ctrl_w, 1e-8)
                weights[ctrl_idx] = new_ctrl_w

                # Report SMD improvement for the features ebal was balanced on
                X_sub_ctrl = X.values[ctrl_mask][:, ebal_cols_global].astype(float)
                X_sub_trt  = X.values[treated_mask][:, ebal_cols_global].astype(float)
                _strat = getattr(config, 'PS_NUMERIC_NAN_IMPUTE', 'median')
                for local_j, global_j in enumerate(ebal_cols_global):
                    col_ctrl = X_sub_ctrl[:, local_j].copy()
                    col_trt  = X_sub_trt[:, local_j].copy()
                    for arr in (col_ctrl, col_trt):
                        if np.isnan(arr).any():
                            fill = (np.nanmedian(arr) if _strat == 'median'
                                    else np.nanmean(arr) if _strat == 'mean'
                                    else float(_strat))
                            arr[:] = np.where(np.isnan(arr), fill, arr)

                    sd_pool = np.sqrt((col_ctrl.var() + col_trt.var()) / 2.0)
                    smd_before = (abs(np.average(col_ctrl, weights=w_ctrl_init) -
                                      np.average(col_trt))
                                  / sd_pool if sd_pool > 0 else 0.0)
                    smd_after  = (abs(np.average(col_ctrl, weights=new_ctrl_w) -
                                      np.average(col_trt))
                                  / sd_pool if sd_pool > 0 else 0.0)
                    print(f"    {feat_cols[global_j]:<35}  SMD {smd_before:.4f} → {smd_after:.4f}")

                return weights
            else:
                print(f"    WARNING: ebal(ATT) did not converge (maxdiff={out.get('maxdiff', '?'):.4g}) "
                      f"— original weights kept")
                return weights

        # ── ATC mode (default): per-arm refinement ──────────────────────────
        for arm_id in sorted(k for k in self.arm_ids if k != 0):
            arm_mask = t_arr == arm_id
            sub_mask = ctrl_mask | arm_mask

            X_sub = X.values[sub_mask].copy()  # keep as-is (mixed numeric/categorical)
            t_sub = arm_mask[sub_mask]          # True = arm, False = control
            w_sub = weights[sub_mask]

            t_idx = np.where(t_sub)[0]
            c_idx = np.where(~t_sub)[0]

            # ── impute NaNs and identify numeric features above threshold ──
            imbal_j = []
            for j, feat in enumerate(feat_cols):
                if isinstance(X[feat].dtype, pd.CategoricalDtype) or X[feat].dtype == object:
                    continue
                col = X_sub[:, j].astype(float)  # convert to float only after skipping categorical
                if np.isnan(col).any():
                    strategy = getattr(config, 'PS_NUMERIC_NAN_IMPUTE', 'median')
                    fill = (np.nanmedian(col) if strategy == 'median'
                            else np.nanmean(col) if strategy == 'mean'
                            else float(strategy))
                    col = np.where(np.isnan(col), fill, col)
                    X_sub[:, j] = col
                w_t = w_sub[t_idx]
                w_c = w_sub[c_idx]
                wmu_t   = np.average(col[t_idx], weights=w_t)
                wmu_c   = np.average(col[c_idx], weights=w_c)
                sd_pool = np.sqrt((col[t_idx].var() + col[c_idx].var()) / 2.0)
                smd_w   = abs(wmu_t - wmu_c) / sd_pool if sd_pool > 0 else 0.0
                if smd_w > threshold:
                    imbal_j.append(j)

            if not imbal_j:
                print(f"    Arm {arm_id}: all features already balanced — skipping")
                continue

            print(f"    Arm {arm_id}: {len(imbal_j)} feature(s) above threshold — optimising")

            # ── entropy balancing via ebal_bin(ATC) ───────────────────────
            # ATC reweights the treated arm (T=1) to match control (T=0) means.
            # base_weight = current IPTW arm weights (length = n_arm).
            # ebal internally inverts T for ATC, so base_weight is for T=1 arm.
            X_imbal  = X_sub[:, imbal_j].astype(float)    # (n, K) — numeric only
            T_sub    = t_sub.astype(int)                   # 1=arm, 0=control
            w_t_init = w_sub[t_idx].copy()
            w_c      = w_sub[c_idx]
            # Y_dummy: ebal requires an outcome array but we only use weights
            Y_dummy  = np.zeros(len(t_sub))

            try:
                e = ebal_bin(
                    max_iterations=500,
                    constraint_tolerance=tolerance,
                    print_level=0,
                    effect="ATC",
                    PCA=False,
                )
                out = e.ebalance(T_sub, X_imbal, Y_dummy, base_weight=w_t_init)
            except Exception as exc:
                print(f"    WARNING Arm {arm_id}: ebal raised {type(exc).__name__}: {exc} "
                      f"— original weights kept")
                continue

            if out['converged']:
                # out['w'][T_sub==1] = refined arm weights (may have different sum)
                new_arm_w = out['w'][T_sub == 1]
                # preserve the original weight sum so downstream ESS is comparable
                new_arm_w = new_arm_w * (w_t_init.sum() / new_arm_w.sum())
                new_arm_w = np.maximum(new_arm_w, 1e-8)

                full_positions = np.where(sub_mask)[0][t_idx]
                weights[full_positions] = new_arm_w

                # report per-feature SMD improvement
                w_c_full = weights[np.where(sub_mask)[0][c_idx]]
                for j in imbal_j:
                    col     = X_sub[:, j]
                    sd_pool = np.sqrt((col[t_idx].var() + col[c_idx].var()) / 2.0)
                    smd_before = (abs(np.average(col[t_idx], weights=w_t_init) -
                                      np.average(col[c_idx], weights=w_c))
                                  / sd_pool if sd_pool > 0 else 0.0)
                    smd_after  = (abs(np.average(col[t_idx], weights=new_arm_w) -
                                      np.average(col[c_idx], weights=w_c_full))
                                  / sd_pool if sd_pool > 0 else 0.0)
                    print(f"      {feat_cols[j]:<35}  SMD {smd_before:.4f} → {smd_after:.4f}")
            else:
                print(f"    WARNING Arm {arm_id}: ebal did not converge "
                      f"(maxdiff={out.get('maxdiff', '?'):.4g}) — original weights kept")

        return weights

    # ------------------------------------------------------------------
    # Propensity score estimation
    # ------------------------------------------------------------------
    def _estimate_ps(self, X: pd.DataFrame, t_arr: np.ndarray) -> np.ndarray:
        """
        Fit a multi-class classifier and return the full probability matrix
        P(T=k | X) with columns ordered by self.arm_ids.

        StandardScaler is applied ONLY for the non-CatBoost paths (logistic /
        XGBoost).  CatBoost handles categorical columns and missing values
        natively and must receive the raw DataFrame — applying StandardScaler
        to a DataFrame that contains pd.Categorical columns raises a
        ValueError before we ever reach the CatBoost branch.

        Returns
        -------
        np.ndarray, shape (n_samples, n_arms)
        """
        if self.ps_method == 'catboost':
            proba = self._catboost_ps(X, t_arr)   # uses raw X, no scaling
        elif self.ps_method == 'xgboost':
            X_num = _encode_cats(X)
            self._scaler = StandardScaler()
            X_sc  = self._scaler.fit_transform(X_num)
            proba = self._xgboost_ps(X_sc, t_arr)
        else:
            X_num = _encode_cats(X)
            self._scaler = StandardScaler()
            X_sc  = self._scaler.fit_transform(X_num)
            proba = self._logistic_ps(X_sc, t_arr)

        return proba

    def _catboost_ps(self, X: pd.DataFrame, t_arr: np.ndarray) -> np.ndarray:
        """
        Multi-class CatBoostClassifier propensity model.

        Uses raw (unscaled) X — CatBoost handles feature scaling,
        missing values, and mixed dtypes internally.  Columns are
        reordered to match self.arm_ids before returning.

        Note: CatBoost requires cat_feature values to be int or str —
        float NaN is not allowed.  We replace NaN in categorical columns
        with the string '__NA__' before building the Pool.
        """
        from catboost import CatBoostClassifier, Pool

        # Replace NaN in categorical columns with '__NA__' string
        # (CatBoost rejects float NaN in cat_features)
        X_pool = X.copy()
        for col in X_pool.columns:
            if X_pool[col].dtype.name in ('object', 'category'):
                X_pool[col] = X_pool[col].astype(object).fillna('__NA__')

        # Map arm IDs to consecutive labels 0..K-1 for CatBoost
        arm_map     = {arm: idx for idx, arm in enumerate(self.arm_ids)}
        t_remapped  = np.array([arm_map[a] for a in t_arr], dtype=int)

        cat_indices = [i for i, col in enumerate(X_pool.columns)
                       if X_pool[col].dtype.name == 'object']
        pool = Pool(data=X_pool, label=t_remapped, cat_features=cat_indices)

        params = dict(config.CATBOOST_PROPENSITY_PARAMS)
        # Multi-class IPTW: override to MultiClass objective
        params['loss_function']  = 'MultiClass'
        params['eval_metric']    = 'MultiClass'
        params['classes_count']  = len(self.arm_ids)

        model = CatBoostClassifier(**params)
        model.fit(pool)
        self._ps_model = model

        proba = model.predict_proba(X_pool)   # use NaN-replaced X_pool; shape (n, K)

        for i, arm_id in enumerate(self.arm_ids):
            ps_col = proba[:, i]
            print(f"    Arm {arm_id} PS (CatBoost): "
                  f"range=[{ps_col.min():.4f}, {ps_col.max():.4f}]  "
                  f"mean={ps_col.mean():.4f}")
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
                          weights: np.ndarray,
                          weights_pre_refine: np.ndarray = None) -> list:
        """
        Compute Standardised Mean Differences (SMD) before and after
        IPTW weighting for each arm vs. control.
        When weights_pre_refine is supplied (entropy-balance was applied),
        a third smd_iptw column is included showing the pre-ebal SMD.
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
                # Skip categorical / object columns — SMD is undefined for them
                if (pd.api.types.is_categorical_dtype(X[feat])
                        or pd.api.types.is_object_dtype(X[feat])):
                    continue

                col = X_sub[:, j].astype(float)

                # Impute numeric NaN using the same strategy as _encode_cats()
                if np.isnan(col).any():
                    strategy = config.PS_NUMERIC_NAN_IMPUTE
                    if strategy == 'median':
                        fill = np.nanmedian(col)
                    elif strategy == 'mean':
                        fill = np.nanmean(col)
                    else:
                        fill = float(strategy)
                    col = np.where(np.isnan(col), fill, col)

                # Unweighted SMD
                mu_t_uw = col[t_idx].mean()
                mu_c_uw = col[c_idx].mean()
                sd_pool  = np.sqrt((col[t_idx].var() + col[c_idx].var()) / 2.0)
                smd_uw   = abs(mu_t_uw - mu_c_uw) / sd_pool if sd_pool > 0 else 0.0

                # Weighted SMD (IPTW-adjusted)
                # Denominator uses the same unweighted sd_pool as above so
                # smd_unweighted and smd_weighted share a common scale and
                # are directly comparable on Love plots (Austin & Stuart 2015).
                w_t = w_sub[t_idx]
                w_c = w_sub[c_idx]

                wmu_t = np.average(col[t_idx], weights=w_t) if w_t.sum() > 0 else mu_t_uw
                wmu_c = np.average(col[c_idx], weights=w_c) if w_c.sum() > 0 else mu_c_uw
                smd_w = abs(wmu_t - wmu_c) / sd_pool if sd_pool > 0 else 0.0

                # SMD for IPTW/overlap weights before entropy balance
                smd_iptw = None
                if weights_pre_refine is not None:
                    w_pre = weights_pre_refine[sub_mask]
                    wmu_t_pre = (np.average(col[t_idx], weights=w_pre[t_idx])
                                 if w_pre[t_idx].sum() > 0 else mu_t_uw)
                    wmu_c_pre = (np.average(col[c_idx], weights=w_pre[c_idx])
                                 if w_pre[c_idx].sum() > 0 else mu_c_uw)
                    smd_iptw = abs(wmu_t_pre - wmu_c_pre) / sd_pool if sd_pool > 0 else 0.0

                balanced = smd_w < 0.1
                if balanced:
                    balanced_count += 1

                rows.append({
                    'arm_id':         arm_id,
                    'arm_name':       config.TREATMENTS[arm_id],
                    'feature':        feat,
                    'smd_unweighted': round(smd_uw,   4),
                    'smd_iptw':       round(smd_iptw, 4) if smd_iptw is not None else None,
                    'smd_weighted':   round(smd_w,    4),
                    'balanced':       balanced,
                })

            stage = "IPTW+ebal" if weights_pre_refine is not None else "IPTW"
            print(f"  Arm {arm_id} ({config.TREATMENTS[arm_id]}): "
                  f"{balanced_count}/{len(feat_cols)} features |SMD| < 0.10 "
                  f"after {stage}")

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
        prefix = 'overlap' if self.weighting_type == 'overlap' else 'iptw'
        path = os.path.join(save_dir, f'{prefix}_weight_distribution.png')
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

        weighted_label = ('Overlap-weighted' if self.weighting_type == 'overlap'
                          else 'IPTW-weighted')
        has_ebal = ('smd_iptw' in df.columns) and df['smd_iptw'].notna().any()

        fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.45)))
        y_pos = np.arange(len(df))

        if has_ebal:
            bar_w = 0.25
            ax.barh(y_pos - bar_w, df['smd_unweighted'], bar_w,
                    color='#E74C3C', alpha=0.75, label='Unweighted')
            ax.barh(y_pos,          df['smd_iptw'],       bar_w,
                    color='#F39C12', alpha=0.75, label=f'{weighted_label} only')
            ax.barh(y_pos + bar_w,  df['smd_weighted'],   bar_w,
                    color='#2E86AB', alpha=0.75, label=f'{weighted_label} + Entropy-balanced')
        else:
            ax.barh(y_pos - 0.2, df['smd_unweighted'], 0.38,
                    color='#E74C3C', alpha=0.75, label='Unweighted')
            ax.barh(y_pos + 0.2, df['smd_weighted'],   0.38,
                    color='#2E86AB', alpha=0.75, label=weighted_label)

        ax.axvline(x=0.1, color='grey', linestyle='--', linewidth=1.2,
                   label='|SMD| = 0.10 threshold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'], fontsize=9)
        ax.set_xlabel('Absolute Standardised Mean Difference (SMD)', fontsize=10)
        arm_label = config.TREATMENTS[arm_id]
        prefix = 'overlap' if self.weighting_type == 'overlap' else 'iptw'
        method_label = ('Overlap' if self.weighting_type == 'overlap' else 'IPTW')
        subtitle = (f'Trim={self.trim_pct}%' if self.weighting_type == 'overlap'
                    else f'Stabilised={self.stabilized}  Trim={self.trim_pct}%')
        if has_ebal:
            estimand = getattr(config, 'BALANCE_REFINE_ESTIMAND', 'ATC').upper()
            subtitle = f'{subtitle}  +  Entropy-balanced ({estimand})'
        ax.set_title(f'{method_label} Love Plot — Arm {arm_id} ({arm_label}) vs. Control\n'
                     f'{subtitle}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        path = os.path.join(save_dir, f'{prefix}_love_plot_arm{arm_id}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    {method_label} Love plot saved: {path}")


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

    iptw = IPTWWeighting(weighting_type='iptw')
    iptw.fit_transform(X, treatment, save_dir=save_results_dir)
    return iptw


def run_overlap(X: pd.DataFrame,
                treatment,
                save_results_dir: str = None) -> IPTWWeighting:
    """
    Compute overlap weights for the given feature matrix and treatment vector.

    Overlap weights  w_i = h(x_i) / e_k(x_i)  are bounded in (0, 1], so
    variance is always finite even when propensity scores are near 0 or 1.
    This makes them robust for skewed arm sizes where IPTW fails.

    Parameters
    ----------
    X                : pd.DataFrame  Boruta-SHAP selected features.
    treatment        : array-like    Multi-arm treatment vector (0 = control).
    save_results_dir : str           Directory for diagnostic plots/CSVs.

    Returns
    -------
    IPTWWeighting  (fitted, weighting_type='overlap')
        Access the per-row weights via  result.weights  (np.ndarray).
    """
    if not isinstance(treatment, pd.Series):
        treatment = pd.Series(treatment)

    ow = IPTWWeighting(weighting_type='overlap')
    ow.fit_transform(X, treatment, save_dir=save_results_dir)
    return ow


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
