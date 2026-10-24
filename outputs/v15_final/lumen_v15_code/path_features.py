"""Slow scatter-gather path features: value-compatible two-hop paths X -> M -> Y anchored on the
same source and destination, grouped non-transitively. Label-free; identical code for train and test.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from payments import load_payments  # noqa: E402

HORIZON = 6 * 3600
RATIO = (0.6, 1.25)
WINDOWS_H = (1, 6)
MEASURES = ('paths', 'ratio_median', 'kept_median', 'value_cv', 'hold_median_h', 'hold_cv', 'span_h', 'converted_share')


def load(directory):
    p = pd.DataFrame(load_payments(directory))[['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE']]
    p['TIMESTAMP'] = p.TIMESTAMP.astype('int64')
    p['VALUE'] = p.VALUE.astype(float)
    legs = pq.read_table(Path(directory) / 'transfers.parquet', columns=['TRANSACTION_HASH']).to_pandas().groupby('TRANSACTION_HASH').size()
    p['converts'] = (p.TRANSACTION_HASH.map(legs).fillna(1) > 1).astype(float)
    return p.reset_index(drop=True)


def build_hops(p):
    a = p.rename(columns={'TRANSACTION_HASH': 'h1', 'TIMESTAMP': 't1', 'FROM_ADDRESS': 'X', 'TO_ADDRESS': 'M', 'VALUE': 'v1', 'converts': 'c1'})
    b = p.rename(columns={'TRANSACTION_HASH': 'h2', 'TIMESTAMP': 't2', 'FROM_ADDRESS': 'M', 'TO_ADDRESS': 'Y', 'VALUE': 'v2', 'converts': 'c2'})
    h = a.merge(b, on='M')
    hold = h.t2 - h.t1
    ratio = h.v2 / np.maximum(h.v1, 1e-9)
    h = h[(hold > 0) & (hold <= HORIZON) & (ratio >= RATIO[0]) & (ratio <= RATIO[1]) & (h.M != h.X) & (h.M != h.Y) & (h.X != h.Y)].copy()
    h['hold'] = (h.t2 - h.t1) / 3600
    h['ratio'] = h.v2 / np.maximum(h.v1, 1e-9)
    h['dev'] = (h.ratio - 1).abs()
    h['conv'] = np.maximum(h.c1, h.c2)
    h = h.sort_values('dev').drop_duplicates(['h1', 'Y']).drop_duplicates(['h2', 'X']).reset_index(drop=True)
    h['hop'] = np.arange(len(h))
    return h


def group_measures(h, W):
    hw = h[h.hold <= W][['hop', 'X', 'Y', 'M', 't1', 't2', 'v1', 'hold', 'ratio', 'conv']]
    j = hw.merge(hw, on=['X', 'Y'], suffixes=('', '_n'))
    j = j[(j.t1_n - j.t1).abs() <= W * 3600]
    g = j.groupby('hop')
    m = pd.DataFrame({'paths': g.M_n.nunique(), 'ratio_median': g.ratio_n.median(), 'hold_median_h': g.hold_n.median(),
                      'converted_share': g.conv_n.mean(), 'span_h': (g.t2_n.max() - g.t1_n.min()) / 3600,
                      'value_cv': (g.v1_n.std() / g.v1_n.mean()).fillna(0.0), 'hold_cv': (g.hold_n.std() / g.hold_n.mean()).fillna(0.0)})
    m['kept_median'] = 1 - m.ratio_median
    return m.reindex(columns=MEASURES)


def path_features(directory):
    p = load(directory)
    n = len(p)
    h = build_hops(p)
    out = pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH})
    order = pd.Series(np.arange(n), index=p.TRANSACTION_HASH)
    for W in WINDOWS_H:
        m = group_measures(h, W)
        hm = h[['hop', 'h1', 'h2', 'dev']].merge(m, left_on='hop', right_index=True)
        # Each payment's best hop: largest paths, then ratio closest to 1; as leg 1 or leg 2.
        both = pd.concat([hm.assign(hash=hm.h1, role='scatter'), hm.assign(hash=hm.h2, role='gather')])
        both = both.sort_values(['paths', 'dev'], ascending=[False, True])
        best = both.drop_duplicates('hash').set_index('hash')
        idx = order.loc[best.index].to_numpy()
        for meas in MEASURES:
            col = np.full(n, -1.0)
            col[idx] = best[meas].to_numpy()
            out[f'sg_{W}h_{meas}'] = col
        out[f'sg_{W}h_paths'] = np.where(out[f'sg_{W}h_paths'] < 0, 0.0, out[f'sg_{W}h_paths'])
        for role in ('scatter', 'gather'):
            r = both[both.role == role].drop_duplicates('hash').set_index('hash').paths
            col = np.zeros(n)
            col[order.loc[r.index].to_numpy()] = r.to_numpy()
            out[f'sg_{W}h_as_{role}'] = col
    assert len(out) == n and out.TRANSACTION_HASH.is_unique and np.isfinite(out.drop(columns='TRANSACTION_HASH').to_numpy()).all()
    out.attrs['hops'] = len(h)
    return out
