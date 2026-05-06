"""
Optuna Tuning — Attrition (Retention) CatBoost Classifier
==========================================================
Optimises the CatBoost hyper-parameters for the binary retention model
(on_book_month9 target).

Objective
---------
Maximise mean ROC-AUC across 5 stratified cross-validation folds.

Storage
-------
Study is persisted to  tuning/studies/attrition_study.db  (SQLite).

Output
------
tuning/best_params/attrition_params.json  — best CatBoost classifier params
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

_STUDY_DIR  = os.path.join(os.path.dirname(__file__), 'studies')
_PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'best_params')
_STORAGE    = f"sqlite:///{os.path.join(_STUDY_DIR, 'attrition_study.db')}"
_OUT_JSON   = os.path.join(_PARAMS_DIR, 'attrition_params.json')
_STUDY_NAME = 'attrition_catboost_classifier'


def _load_data():
    """Load data and return attrition-target feature matrix."""
    data_path = os.path.join(config.DATA_DIR, config.RAW_DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Set RAW_DATA_FILE in config.py and place your data in {config.DATA_DIR}."
        )
    df = pd.read_csv(data_path)

    outcome_cols = ['treatment', 'treatment_name', 'opening_balance',
                    'on_book_month9', 'offer']
    X_raw  = df[[c for c in df.columns if c not in outcome_cols]]
    y      = df['on_book_month9']
    t      = df['treatment']

    # Try saved attrition feature names first
    att_feat_path = os.path.join(config.MODELS_DIR, 'attrition_feature_names.json')
    if os.path.exists(att_feat_path):
        with open(att_feat_path) as fh:
            feats = json.load(fh)
        feats = [f for f in feats if f in X_raw.columns]
        if feats:
            print(f"  Using saved attrition features ({len(feats)}) for tuning.")
            X = X_raw[feats].copy()
            X['treatment_indicator'] = t.values
            return X, y, t

    # Fall back to fresh feature selection
    from src.feature_selection.step1_initial_pruning import run_initial_pruning
    from src.feature_selection.step2_boruta_shap import run_boruta_shap
    print("  Running feature selection for attrition tuning ...")
    X1, _ = run_initial_pruning(X_raw, y=y, treatment=t)
    X2, _ = run_boruta_shap(X1, y=y, task='classification')
    X2['treatment_indicator'] = t.values
    return X2, y, t


def _build_params(trial: optuna.Trial) -> dict:
    return {
        'iterations':          trial.suggest_int('iterations',      200, 1200, step=100),
        'depth':               trial.suggest_int('depth',            4,   8),
        'learning_rate':       trial.suggest_float('learning_rate',  0.01, 0.20, log=True),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg',    1.0, 20.0, log=True),
        'min_data_in_leaf':    trial.suggest_int('min_data_in_leaf', 1,   50),
        'border_count':        trial.suggest_int('border_count',     32,  255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        # Fixed keys
        'eval_metric':         'AUC',
        'od_type':             'Iter',
        'od_wait':             50,
        'random_seed':         config.RANDOM_SEED,
        'thread_count':        -1,
        'verbose':             False,
        'allow_writing_files': False,
        'nan_mode':            'Min',
    }


def objective(trial: optuna.Trial, X: pd.DataFrame, y) -> float:
    """5-fold stratified CV → return mean -ROC-AUC."""
    from catboost import CatBoostClassifier, Pool

    params      = _build_params(trial)
    y_arr       = np.array(y, dtype=int)
    cat_indices = [i for i, c in enumerate(X.columns)
                   if X[c].dtype.name in ('object', 'category')]

    skf    = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=config.RANDOM_SEED)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_arr)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx],  y_arr[val_idx]

        tr_pool  = Pool(X_tr, label=y_tr,  cat_features=cat_indices)
        val_pool = Pool(X_val, label=y_val, cat_features=cat_indices)

        model = CatBoostClassifier(**params)
        model.fit(tr_pool, eval_set=val_pool, use_best_model=True)

        proba = model.predict_proba(val_pool)[:, 1]
        scores.append(roc_auc_score(y_val, proba))

        # Pruning: report intermediate value after each fold
        trial.report(np.mean(scores), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return -np.mean(scores)   # minimise


def run_study(n_trials: int = 50, resume: bool = False) -> dict:
    os.makedirs(_STUDY_DIR,  exist_ok=True)
    os.makedirs(_PARAMS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("OPTUNA TUNING — ATTRITION MODEL (CatBoost Classifier)")
    print("="*65)
    print(f"  Trials    : {n_trials}")
    print(f"  Storage   : {_STORAGE}")
    print(f"  Study name: {_STUDY_NAME}")
    print("="*65 + "\n")

    print("Loading data ...")
    X, y, _ = _load_data()
    print(f"  Features : {X.shape[1]}   Rows : {len(X):,}   "
          f"Retention rate : {y.mean():.1%}\n")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study  = optuna.create_study(
        study_name     = _STUDY_NAME,
        storage        = _STORAGE,
        sampler        = TPESampler(seed=config.RANDOM_SEED),
        pruner         = pruner,
        direction      = 'minimize',
        load_if_exists = resume,
    )

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    best = study.best_trial
    print(f"\n  Best trial : #{best.number}  ROC-AUC = {-best.value:.4f}")
    print("  Best params:")
    for k, v in best.params.items():
        print(f"    {k:<26} : {v}")

    full_params = {
        'iterations':          best.params['iterations'],
        'depth':               best.params['depth'],
        'learning_rate':       round(best.params['learning_rate'], 6),
        'l2_leaf_reg':         round(best.params['l2_leaf_reg'], 4),
        'min_data_in_leaf':    best.params['min_data_in_leaf'],
        'border_count':        best.params['border_count'],
        'bagging_temperature': round(best.params['bagging_temperature'], 4),
        'eval_metric':         'AUC',
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

    try:
        importance = optuna.importance.get_param_importances(study)
        print("\n  Parameter importances:")
        for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:8]:
            print(f"    {name:<26} : {imp:.3f}")
    except Exception:
        pass

    return full_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optuna tuning for Attrition CatBoost classifier')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--resume',   action='store_true')
    args = parser.parse_args()
    run_study(n_trials=args.n_trials, resume=args.resume)
