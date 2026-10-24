"""Second-stage context features: first-stage scores of payments sharing a wallet nearby in time.

Scores are converted to within-split percentiles so out-of-fold training scores and full-model
test scores sit on the same scale. No labels enter these features; only model scores do.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from payments import load_payments  # noqa: E402

WINDOWS = (1, 6, 24, 72)
DIRECTED = (6, 72)
HIGH = 0.97


def load_edges(directory):
    p = pd.DataFrame(load_payments(directory))[['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS']]
    p['TIMESTAMP'] = p.TIMESTAMP.astype('int64')
    return p


def context_features(edges, ids, scores, bursts=None, normalized=False):
    """edges: payment edges for one split; ids/scores: first-stage scores for the same transactions.
    bursts: optional frame of TRANSACTION_HASH plus one burst-id column per motif scale."""
    s = pd.Series(np.asarray(scores, float), index=pd.Index(ids, name='TRANSACTION_HASH'))
    e = edges.set_index('TRANSACTION_HASH').loc[s.index].reset_index()
    e['pct'] = s.rank(pct=True, method='average').to_numpy()
    n = len(e)
    e['i'] = np.arange(n)
    inc = pd.concat([e[['i', 'TIMESTAMP', 'pct']].assign(w=e.FROM_ADDRESS, out=1),
                     e[['i', 'TIMESTAMP', 'pct']].assign(w=e.TO_ADDRESS, out=0)], ignore_index=True)
    # Pair every payment with every other payment touching the same wallet within the widest window.
    pairs = inc.merge(inc, on='w', suffixes=('', '_n'))
    pairs = pairs[(pairs.i != pairs.i_n)]
    pairs['dt'] = (pairs.TIMESTAMP_n - pairs.TIMESTAMP) / 3600
    pairs = pairs[pairs.dt.abs() <= max(WINDOWS)]
    out = {}
    nodup = pairs.drop_duplicates(['i', 'i_n'])
    for w in WINDOWS:
        q = nodup[nodup.dt.abs() <= w].groupby('i').pct_n
        out[f'ctx_{w}h_max'] = q.max()
        out[f'ctx_{w}h_mean'] = q.mean()
        out[f'ctx_{w}h_count'] = q.size()
        out[f'ctx_{w}h_high'] = q.apply(lambda v: int((v >= HIGH).sum()))
        out[f'ctx_{w}h_top2'] = q.apply(lambda v: np.sort(v.to_numpy())[-2] if len(v) > 1 else np.nan)
    # Directional: money into the sender before this payment, money out of the receiver after it.
    for w in DIRECTED:
        up = pairs[(pairs.out == 1) & (pairs.out_n == 0) & (pairs.dt <= 0) & (pairs.dt >= -w)].groupby('i').pct_n
        down = pairs[(pairs.out == 0) & (pairs.out_n == 1) & (pairs.dt >= 0) & (pairs.dt <= w)].groupby('i').pct_n
        for name, q in (('up', up), ('down', down)):
            out[f'ctx_{name}_{w}h_max'] = q.max()
            out[f'ctx_{name}_{w}h_mean'] = q.mean()
            out[f'ctx_{name}_{w}h_high'] = q.apply(lambda v: int((v >= HIGH).sum()))
    f = pd.DataFrame(out).reindex(np.arange(n))
    for c in f.columns:
        f[c] = f[c].fillna(0.0)
    # Whole-week wallet context.
    wmax = inc.groupby('w').pct.max()
    wmean = inc.groupby('w').pct.mean()
    f['ctx_sender_week_max'] = e.FROM_ADDRESS.map(wmax).to_numpy()
    f['ctx_receiver_week_max'] = e.TO_ADDRESS.map(wmax).to_numpy()
    f['ctx_sender_week_mean'] = e.FROM_ADDRESS.map(wmean).to_numpy()
    f['ctx_receiver_week_mean'] = e.TO_ADDRESS.map(wmean).to_numpy()
    f['ctx_self_pct'] = e.pct.to_numpy()
    if normalized:
        # Exposure-normalised evidence: busy wallets meet more high-scoring payments by chance.
        base = (e.FROM_ADDRESS.map(wmean).to_numpy() + e.TO_ADDRESS.map(wmean).to_numpy()) / 2
        for w in WINDOWS:
            f[f'ctx_{w}h_high_frac'] = f[f'ctx_{w}h_high'] / np.maximum(f[f'ctx_{w}h_count'], 1)
            f[f'ctx_{w}h_mean_excess'] = np.where(f[f'ctx_{w}h_count'] > 0, f[f'ctx_{w}h_mean'] - base, 0.0)
            f[f'ctx_{w}h_max_excess'] = np.where(f[f'ctx_{w}h_count'] > 0, f[f'ctx_{w}h_max'] - base, 0.0)
        whigh = inc.assign(h=(inc.pct >= HIGH).astype(float)).groupby('w').h.mean()
        f['ctx_sender_week_high_frac'] = e.FROM_ADDRESS.map(whigh).to_numpy()
        f['ctx_receiver_week_high_frac'] = e.TO_ADDRESS.map(whigh).to_numpy()
    if bursts is not None:
        # Same-burst context: scores of the other payments in this payment's motif burst (self excluded).
        b = bursts.set_index('TRANSACTION_HASH').loc[e.TRANSACTION_HASH].reset_index(drop=True)
        pct = e.pct
        for col in b.columns:
            g = pct.groupby(b[col].to_numpy())
            size = g.transform('size').to_numpy()
            others = np.maximum(size - 1, 1)
            top1 = g.transform('max').to_numpy()
            top2 = g.transform(lambda v: np.sort(v.to_numpy())[-2] if len(v) > 1 else 0.0).to_numpy()
            f[f'burst_{col}_mean'] = np.where(size > 1, (g.transform('sum').to_numpy() - pct.to_numpy()) / others, 0.0)
            f[f'burst_{col}_max'] = np.where(size > 1, np.where(pct.to_numpy() >= top1, top2, top1), 0.0)
            f[f'burst_{col}_min'] = np.where(size > 1, g.transform('min').to_numpy(), 0.0)
            f[f'burst_{col}_high'] = g.transform(lambda v: (v >= HIGH).sum()).to_numpy() - (pct.to_numpy() >= HIGH)
    f.insert(0, 'TRANSACTION_HASH', e.TRANSACTION_HASH.to_numpy())
    assert len(f) == n and f.TRANSACTION_HASH.is_unique and np.isfinite(f.drop(columns='TRANSACTION_HASH').to_numpy()).all()
    return f
