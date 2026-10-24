"""Final V10 score: equal-weight rank blend of two run_v3.py outputs (V5 fits hard-tier positives, V7 does not)."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run import validate_predictions
from run_v3 import blend_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True, help='predictions from run_v3.py --keep-hard (V5)')
    parser.add_argument('--b', type=Path, required=True, help='predictions from run_v3.py (V7)')
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    args = parser.parse_args()
    expected = pq.read_table(args.test / 'transactions.parquet', columns=['TRANSACTION_HASH']).to_pandas().TRANSACTION_HASH
    a = pd.read_csv(args.a).set_index('TRANSACTION_HASH').loc[expected].score.to_numpy()
    b = pd.read_csv(args.b).set_index('TRANSACTION_HASH').loc[expected].score.to_numpy()
    predictions = pd.DataFrame({'TRANSACTION_HASH': expected, 'score': blend_scores(a, b)})
    validate_predictions(predictions, expected)
    predictions.to_csv(args.output, index=False, float_format='%.17g')
    validate_predictions(pd.read_csv(args.output), expected)
    print(json.dumps({'rows': len(predictions), 'unique_scores': int(predictions.score.nunique()),
                      'csv_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
