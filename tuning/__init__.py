"""
Optuna-based hyperparameter tuning for the CATE pipeline.

Modules
-------
tune_xlearner   — Tune CatBoost regressors (outcome & CATE models).
tune_attrition  — Tune CatBoost attrition classifier.
tune_propensity — Tune CatBoost propensity score classifier.
apply_best_params — Write best params from JSON back to config.py.

Quick start
-----------
    # Tune all models (80 trials each by default):
    python tuning/tune_xlearner.py   --n-trials 80
    python tuning/tune_attrition.py  --n-trials 80
    python tuning/tune_propensity.py --n-trials 60

    # Write best params to config.py:
    python tuning/apply_best_params.py

Studies are persisted in tuning/studies/*.db (SQLite) so they can be
resumed if interrupted.  Best parameters for each stage are saved to
tuning/best_params/*.json.
"""
