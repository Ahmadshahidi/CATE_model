"""
Optuna Tuning — X-Learner CatBoost Regressors
===============================================
Optimises the CatBoost hyper-parameters used for the *outcome models*
(µ₀, µ₁) and *CATE models* (τ̂ₜ, τ̂_c) inside the X-Learner.

Objective
---------
Maximise AUUC lift on a held-out validation fold.
Each trial trains a single-arm X-Learner (arm 1 vs. control) with the
suggested parameters; the remaining arms share the same parameter set
so one trial is a reliable proxy for the full model.

Storage
-------
Study is persisted to  tuning/studies/xlearner_study.db  (SQLite) so it
can be paused and resumed with  --resume.

Output
------
tuning/best_params/xlearner_params.json   — best CatBoost regressor params
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Directories ─────────────────────────────────────────────────────────────
_STUDY_DIR   = os.path.join(os.path.dirname(__file__), 'studies')
_PARAMS_DIR  = os.path.join(os.path.dirname(__file__), 'best_params')
_STORAGE     = f"sqlite:///{os.path.join(_STUDY_DIR, 'xlearner_study.db')}"
_OUT_JSON    = os.path.join(_PARAMS_DIR, 'xlearner_params.json')
_STUDY_NAME  = 'xlearner_catboost_regressors'


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_data():
    """Load or generate training data and run feature selection."""
    from src.data_generation import generate_epsilon_data
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    from src.feature_selection.step2_boruta_shap import run_boruta_shap

    data_path = os.path.join(config.DATA_DIR, 'epsilon_synthetic.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("  Generating synthetic data ...")
        df = generate_epsilon_data()

    outcome_cols = ['treatment', 'treatment_name', 'opening_balance',
                    'on_book_month9', 'offer']
    X = df[[c for c in df.columns if c not in outcome_cols]]
    y = df['opening_balance']
    t = df['treatment']

    # Try to load already-selected balance features from the saved model
    bal_feat_path = os.path.join(config.MODELS_DIR, 'balance_feature_names.json')
    if os.path.exists(bal_feat_path):
        with open(bal_feat_path) as fh:
            feats = json.load(fh)
        feats = [f for f in feats if f in X.columns]
        if feats:
            print(f"  Using saved balance features ({len(feats)}) for tuning.")
            return X[feats], y, t

    # Fall back to fresh feature selection (slower)
    print("  Running feature selection for tuning ...")
    X1, _ = run_initial_pruning(X, y=y, treatment=t)
    X2, _ = run_boruta_shap(X1, y=y, task='regression')
    return X2, y, t


def _compute_auuc(y, cate_scores, t_binary):
    """Scalar AUUC lift for one arm."""
    from scipy import integrate
    order  = np.argsort(-cate_scores)
    n_t    = t_binary.sum()
    n_c    = len(t_binary) - n_t
    if n_t == 0 or n_c == 0:
        return 0.0

    cum_uplift, cum_rand = [0.0], [0.0]
    ct = cc = ctn = ccn = 0.0
    mean_uplift = y[t_binary == 1].mean() - y[t_binary == 0].mean()

    for i, idx in enumerate(order):
        if t_binary[idx]:
            ct += y[idx]; ctn += 1
        else:
            cc += y[idx]; ccn += 1
        up   = (ct / ctn if ctn else 0) - (cc / ccn if ccn else 0)
        cum_uplift.append(up * (i + 1))
        cum_rand.append(mean_uplift * (i + 1))

    pcts = np.linspace(0, 100, len(cum_uplift))
    auuc = integrate.trapezoid(cum_uplift, pcts)
    rand = integrate.trapezoid(cum_rand,   pcts)
    return auuc - rand


def _build_params(trial: optuna.Trial) -> dict:
    """Sample CatBoost regressor hyperparameters from the trial."""
    return {
        'iterations':       trial.suggest_int('iterations',       200, 1000, step=100),
        'depth':            trial.suggest_int('depth',             4,   8),
        'learning_rate':    trial.suggest_float('learning_rate',   0.01, 0.20, log=True),
        'l2_leaf_reg':      trial.suggest_float('l2_leaf_reg',     1.0,  20.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',  1,    50),
        'border_count':     trial.suggest_int('border_count',      32,   255),
        # Fixed keys — not tuned
        'eval_metric':         'RMSE',
        'od_type':             'Iter',
        'od_wait':             40,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }


# ── Objective ────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, X: pd.DataFrame, y, treatment) -> float:
    """
    Train a single-arm (arm=1 vs. control) X-Learner on the training split
    and return -AUUC_lift on the validation split.
    """
    from catboost import CatBoostRegressor, CatBoostClassifier, Pool

    params = _build_params(trial)

    # ── Data split ─────────────────────────────────────────────────────
    t_arr = np.array(treatment, dtype=int)
    y_arr = np.array(y, dtype=float)

    # Keep only arm-1 and control for a fast single-arm proxy
    mask_proxy = (t_arr == 0) | (t_arr == 1)
    X_p = X[mask_proxy].reset_index(drop=True)
    y_p = y_arr[mask_proxy]
    t_p = (t_arr[mask_proxy] == 1).astype(int)

    X_tr, X_val, y_tr, y_val, t_tr, t_val = train_test_split(
        X_p, y_p, t_p,
        test_size    = 0.25,
        stratify     = t_p,
        random_state = config.RANDOM_SEED,
    )

    feat_names  = X_p.columns.tolist()
    cat_indices = [i for i, c in enumerate(feat_names)
                   if X_p[c].dtype.name in ('object', 'category')]

    def _pool(X_arr, y_arr=None, w=None):
        df = pd.DataFrame(X_arr, columns=feat_names)
        return Pool(df, label=y_arr, cat_features=cat_indices, weight=w)

    ctrl_tr = t_tr == 0
    arm_tr  = t_tr == 1

    # Stage 1 — outcome models
    mu0 = CatBoostRegressor(**params)
    mu0.fit(_pool(X_tr[ctrl_tr], y_tr[ctrl_tr]))

    mu1 = CatBoostRegressor(**params)
    mu1.fit(_pool(X_tr[arm_tr], y_tr[arm_tr]))

    # Stage 2 — pseudo-outcomes
    mu0_pred_all_tr = mu0.predict(_pool(X_tr))
    mu1_pred_ctrl   = mu1.predict(_pool(X_tr[ctrl_tr]))

    D_treated = y_tr[arm_tr]  - mu0_pred_all_tr[arm_tr]
    D_control = mu1_pred_ctrl - y_tr[ctrl_tr]

    # Stage 3 — CATE models
    tau_t = CatBoostRegressor(**params)
    tau_t.fit(_pool(X_tr[arm_tr], D_treated))

    tau_c = CatBoostRegressor(**params)
    tau_c.fit(_pool(X_tr[ctrl_tr], D_control))

    # Propensity (simple logistic for speed; fixed params)
    ps_model = CatBoostClassifier(
        iterations=200, depth=4, learning_rate=0.05, verbose=False,
        allow_writing_files=False, random_seed=config.RANDOM_SEED,
    )
    ps_model.fit(_pool(X_tr, t_tr))

    # ── Validation AUUC ────────────────────────────────────────────────
    X_val_arr = np.array(X_val)
    pool_val  = _pool(X_val_arr)

    e_x      = ps_model.predict_proba(pool_val)[:, 1].clip(0.05, 0.95)
    tau_t_v  = tau_t.predict(pool_val)
    tau_c_v  = tau_c.predict(pool_val)
    cate_val = e_x * tau_t_v + (1 - e_x) * tau_c_v

    auuc_lift = _compute_auuc(y_val, cate_val, t_val)
    return -auuc_lift     # Optuna minimises


# ── Main ─────────────────────────────────────────────────────────────────────

def run_study(n_trials: int = 50, resume: bool = False) -> dict:
    os.makedirs(_STUDY_DIR,  exist_ok=True)
    os.makedirs(_PARAMS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("OPTUNA TUNING — X-LEARNER (CatBoost Regressors)")
    print("="*65)
    print(f"  Trials    : {n_trials}")
    print(f"  Storage   : {_STORAGE}")
    print(f"  Study name: {_STUDY_NAME}")
    print("="*65 + "\n")

    print("Loading data ...")
    X, y, treatment = _load_data()
    print(f"  Features : {X.shape[1]}   Rows : {len(X):,}\n")

    study = optuna.create_study(
        study_name       = _STUDY_NAME,
        storage          = _STORAGE,
        sampler          = TPESampler(seed=config.RANDOM_SEED),
        direction        = 'minimize',
        load_if_exists   = resume,
    )

    study.optimize(
        lambda trial: objective(trial, X, y, treatment),
        n_trials  = n_trials,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n  Best trial : #{best.number}  "
          f"AUUC lift = {-best.value:,.2f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:<22} : {v}")

    # Build full param dict (merge tuned values with fixed keys)
    full_params = {
        'iterations':          best.params['iterations'],
        'depth':               best.params['depth'],
        'learning_rate':       round(best.params['learning_rate'], 6),
        'l2_leaf_reg':         round(best.params['l2_leaf_reg'], 4),
        'min_data_in_leaf':    best.params['min_data_in_leaf'],
        'border_count':        best.params['border_count'],
        'eval_metric':         'RMSE',
        'od_type':             'Iter',
        'od_wait':             50,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }

    with open(_OUT_JSON, 'w') as fh:
        json.dump(full_params, fh, indent=2)
    print(f"\n  Best params saved to: {_OUT_JSON}")

    # Param importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n  Parameter importances:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:8]:
            print(f"    {name:<22} : {imp:.3f}")
    except Exception:
        pass

    return full_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optuna tuning for X-Learner CatBoost regressors')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials (default 50)')
    parser.add_argument('--resume',   action='store_true',
                        help='Load existing study from storage and continue')
    args = parser.parse_args()
    run_study(n_trials=args.n_trials, resume=args.resume)
