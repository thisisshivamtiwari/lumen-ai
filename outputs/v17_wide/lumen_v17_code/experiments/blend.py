"""Rank blends of saved V1 and V3 out-of-fold scores, ranked within each fold (no refitting)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import load, summary, OUT

f, a = load()
ctx = pd.read_csv(OUT / 'context_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
v1 = a.score.to_numpy()
rank = lambda s: pd.Series(s).groupby(a.fold.to_numpy()).rank(pct=True).to_numpy()
rows = [summary('v1', a, v1), summary('context', a, ctx)]
for w in (0.3, 0.5, 0.7):
    rows.append(summary(f'blend_ctx{w}', a, (1 - w) * rank(v1) + w * rank(ctx)))
table = pd.DataFrame([{**{k: v for k, v in r.items() if k not in ('fold_ap', 'category_ap')},
                       **{f'fold{i}': x for i, x in enumerate(r['fold_ap'])}, **r['category_ap']} for r in rows])
table.to_csv(OUT / 'blend_metrics.csv', index=False)
print(table.round(4).to_string(index=False))
