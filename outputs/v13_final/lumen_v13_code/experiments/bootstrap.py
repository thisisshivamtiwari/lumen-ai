"""Paired group bootstrap of the AP gain of the 50/50 rank blend over V1 (annotation groups as blocks)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'solution'))
from experiment import load, OUT
from evaluate_v1 import groups_for

f, a = load()
ctx = pd.read_csv(OUT / 'context_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
rank = lambda s: pd.Series(s).groupby(a.fold.to_numpy()).rank(pct=True).to_numpy()
v1, blend = rank(a.score.to_numpy()), 0.5 * rank(a.score.to_numpy()) + 0.5 * rank(ctx)
y = a.is_suspicious.to_numpy()
codes, uniq = pd.factorize(groups_for(a))
members = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_numpy()
rng = np.random.default_rng(20261024)
deltas = []
for _ in range(300):
    idx = np.concatenate(members[rng.integers(0, len(members), len(members))])
    deltas.append(average_precision_score(y[idx], blend[idx]) - average_precision_score(y[idx], v1[idx]))
d = np.array(deltas)
print(f'pooled AP gain {average_precision_score(y, blend) - average_precision_score(y, v1):.4f}; 95% CI [{np.quantile(d, .025):.4f}, {np.quantile(d, .975):.4f}]; share >0 {np.mean(d > 0):.3f}')
