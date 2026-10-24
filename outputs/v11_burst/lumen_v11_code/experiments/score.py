"""Score confirmation-split variants (last-two-day labels) with paired group-bootstrap intervals."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/v9_confirm'
sys.path.insert(0, str(ROOT / 'solution'))
from evaluate_v1 import groups_for  # noqa: E402

labels = pd.read_csv(ROOT / 'data/training/train/labels.csv')
s = {v: pd.read_csv(OUT / f'{v}_scores.csv').set_index('TRANSACTION_HASH').score for v in ['v1', 'v5', 'v7', 'v8']}
a = labels.set_index('TRANSACTION_HASH').loc[s['v1'].index].reset_index()
y = a.is_suspicious.to_numpy()
S = {v: s[v].loc[a.TRANSACTION_HASH].to_numpy() for v in s}
tier = np.where(a.family.eq('ambiguous'), 'ambiguous', a.difficulty)
k = int(np.ceil(0.01 * len(y)))
rows = []
for v, x in S.items():
    order = np.lexsort((a.TRANSACTION_HASH.to_numpy(), -x))
    cat = {c: AP(y[(y == 0) | (tier == c)], x[(y == 0) | (tier == c)]) for c in ['easy', 'medium', 'hard']}
    rows.append(dict(variant=v, ap=AP(y, x), tp_top1pct=int(y[order[:k]].sum()), **{f'{c}_ap': cat[c] for c in cat}))
print(f'{len(y)} rows, {int(y.sum())} positives, top-1% budget {k}')
print(pd.DataFrame(rows).round(4).to_string(index=False))
codes, _ = pd.factorize(groups_for(a))
members = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_numpy()
rng = np.random.default_rng(20261024)
pairs = [('v5', 'v1'), ('v7', 'v5'), ('v8', 'v7'), ('v8', 'v5')]
d = {p: [] for p in pairs}
for _ in range(500):
    idx = np.concatenate(members[rng.integers(0, len(members), len(members))])
    for p in pairs:
        d[p].append(AP(y[idx], S[p[0]][idx]) - AP(y[idx], S[p[1]][idx]))
for p in pairs:
    x = 100 * np.array(d[p])
    print(f'{p[0]} - {p[1]}: {100 * (AP(y, S[p[0]]) - AP(y, S[p[1]])):+.2f} pts, 95% CI [{np.quantile(x, .025):+.2f}, {np.quantile(x, .975):+.2f}], share>0 {np.mean(x > 0):.3f}')
