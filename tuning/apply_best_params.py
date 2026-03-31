"""
Apply Best Optuna Parameters → config.py
=========================================
Reads the JSON files produced by the three tuning scripts and updates the
corresponding parameter dicts in config.py using regex-based in-place
replacement.

Supported targets
-----------------
  tuning/best_params/xlearner_params.json   → CATBOOST_REGRESSOR_PARAMS
  tuning/best_params/attrition_params.json  → CATBOOST_CLASSIFIER_PARAMS
  tuning/best_params/propensity_params.json → CATBOOST_PROPENSITY_PARAMS

The original config.py is backed up to  config.py.bak  before any changes.

Usage
-----
    # Apply all three (skips missing JSON files):
    python tuning/apply_best_params.py

    # Apply only the X-Learner params:
    python tuning/apply_best_params.py --only xlearner

    # Dry run (print diff, do not write):
    python tuning/apply_best_params.py --dry-run
"""

import os
import sys
import re
import json
import shutil
import argparse
import difflib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

_REPO_ROOT  = os.path.join(os.path.dirname(__file__), '..')
_CONFIG     = os.path.join(_REPO_ROOT, 'config.py')
_PARAMS_DIR = os.path.join(os.path.dirname(__file__), 'best_params')

_TARGETS = {
    'xlearner':   ('xlearner_params.json',   'CATBOOST_REGRESSOR_PARAMS'),
    'attrition':  ('attrition_params.json',  'CATBOOST_CLASSIFIER_PARAMS'),
    'propensity': ('propensity_params.json',  'CATBOOST_PROPENSITY_PARAMS'),
}

# Keys whose values should be written as Python literals rather than JSON
# strings — i.e., booleans and None stay unquoted.
_BOOL_NONE_KEYS = {'verbose', 'allow_writing_files', 'nan_mode'}


def _format_value(key: str, val) -> str:
    """Format a parameter value as a Python literal for insertion into config.py."""
    if isinstance(val, bool):
        return 'True' if val else 'False'
    if val is None:
        return 'None'
    if isinstance(val, str):
        return f"'{val}'"
    if isinstance(val, float):
        # Preserve enough precision without scientific notation
        return f"{val:.8g}"
    return str(val)


def _build_dict_block(var_name: str, params: dict) -> str:
    """Return the full Python dict assignment block for config.py."""
    lines = [f'{var_name} = {{']
    max_key_len = max(len(k) for k in params)
    for k, v in params.items():
        padding = ' ' * (max_key_len - len(k))
        lines.append(f"    '{k}':{padding}  {_format_value(k, v)},")
    lines.append('}')
    return '\n'.join(lines)


def _replace_dict_in_config(config_text: str, var_name: str,
                              params: dict) -> str:
    """
    Replace the block

        VAR_NAME = {
            ...
        }

    in config_text with the new params dict.  The regex handles arbitrary
    whitespace and multi-line comment blocks between the assignment and the
    closing brace.
    """
    pattern = (
        rf'^({re.escape(var_name)}\s*=\s*\{{)'  # opening line
        rf'.*?'                                   # anything (non-greedy)
        rf'(^\}})'                                # closing brace at line start
    )
    new_block = _build_dict_block(var_name, params)
    replaced, n = re.subn(pattern, new_block, config_text,
                           count=1, flags=re.MULTILINE | re.DOTALL)
    if n == 0:
        raise ValueError(f"Could not find '{var_name} = {{...}}' block in config.py")
    return replaced


def apply(targets: list[str], dry_run: bool = False) -> None:
    with open(_CONFIG, 'r') as fh:
        original = fh.read()

    updated = original
    applied = []

    for target in targets:
        json_file, var_name = _TARGETS[target]
        json_path = os.path.join(_PARAMS_DIR, json_file)

        if not os.path.exists(json_path):
            print(f"  ⚠  {json_file} not found — skipping {target}.")
            continue

        with open(json_path) as fh:
            params = json.load(fh)

        print(f"\n  Applying {json_file} → {var_name}")
        for k, v in params.items():
            print(f"    {k:<26}: {v}")

        updated = _replace_dict_in_config(updated, var_name, params)
        applied.append(target)

    if not applied:
        print("\n  Nothing to apply — no JSON files found.")
        return

    if dry_run:
        print("\n  --- DRY RUN DIFF ---")
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile='config.py (original)',
            tofile='config.py (updated)',
        )
        sys.stdout.writelines(diff)
        print("  --- END DIFF ---")
        print("\n  Dry run: config.py was NOT modified.")
        return

    # Backup original
    bak = _CONFIG + '.bak'
    shutil.copy2(_CONFIG, bak)
    print(f"\n  Backup written to: {bak}")

    with open(_CONFIG, 'w') as fh:
        fh.write(updated)

    print(f"\n  ✅  config.py updated with best params for: {', '.join(applied)}")
    print("  Re-run the pipeline to train with the new hyperparameters.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write Optuna best params from JSON back to config.py')
    parser.add_argument(
        '--only', nargs='+',
        choices=list(_TARGETS.keys()),
        default=list(_TARGETS.keys()),
        help='Which model(s) to update (default: all three)')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show diff without writing config.py')
    args = parser.parse_args()
    apply(targets=args.only, dry_run=args.dry_run)
