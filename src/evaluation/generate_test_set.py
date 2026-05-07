"""
Generate a stakeholder test set from epsilon_synthetic.csv.

Samples rows across all treatment arms, renames outcome columns to the
names expected by validate_stakeholder.py, and saves to
  results/test_stakeholder.csv

Usage:
  python src/evaluation/generate_test_set.py [--n_per_arm N] [--seed S]
"""

import argparse
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import config

RENAME = {
    'offer':         'offer_amount',       # dollar offer (0 = control)
    'opening_balance': 'actual_opening_balance',
    'on_book_month9':  'actual_on_book9',
}


def generate(n_per_arm: int = 400, seed: int = 42) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(config.DATA_DIR, config.RAW_DATA_FILE))
    print(f"Loaded {len(raw):,} rows from {config.RAW_DATA_FILE}")

    arms = sorted(raw['treatment'].unique())
    parts = []
    for arm in arms:
        subset = raw[raw['treatment'] == arm].sample(
            n=min(n_per_arm, (raw['treatment'] == arm).sum()),
            random_state=seed,
        )
        parts.append(subset)
        print(f"  arm {arm} (${subset['offer'].iloc[0]:>4}): {len(subset):,} rows sampled")

    test_df = pd.concat(parts, ignore_index=True)
    test_df = test_df.drop(columns=['treatment', 'treatment_name'], errors='ignore')
    test_df = test_df.rename(columns=RENAME)
    return test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_arm', type=int, default=400,
                        help='Rows to sample per treatment arm (default: 400)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None,
                        help='Output path (default: results/test_stakeholder.csv)')
    args = parser.parse_args()

    out_path = args.output or os.path.join(config.RESULTS_DIR, 'test_stakeholder.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    test_df = generate(n_per_arm=args.n_per_arm, seed=args.seed)
    test_df.to_csv(out_path, index=False)

    n_ctrl = int((test_df['offer_amount'] == 0).sum())
    n_trt  = int((test_df['offer_amount'] != 0).sum())
    print(f"\nTest set saved: {out_path}")
    print(f"  Rows: {len(test_df):,}  |  Control: {n_ctrl:,}  |  Treated: {n_trt:,}")
    print(f"  Columns: {list(test_df.columns[:6])} ... ({test_df.shape[1]} total)")


if __name__ == '__main__':
    main()
