"""Multi-scale temporal motif features, label-free.

For each time scale D, payments are linked when they touch the same wallet and are consecutive in
that wallet's history within D minutes. Connected components approximate one burst of related
activity. Each payment gets its burst's size, duration, shape and flow, plus its own role in it.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from payments import load_payments  # noqa: E402

SCALES = (2, 5, 10, 20)


def load(directory):
    p = pd.DataFrame(load_payments(directory))[['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE']]
    p['TIMESTAMP'] = p.TIMESTAMP.astype('int64')
    p['VALUE'] = p.VALUE.astype(float)
    return p.reset_index(drop=True)


def components(p, minutes):
    n = len(p)
    inc = pd.DataFrame({'i': np.r_[np.arange(n), np.arange(n)], 'w': np.r_[p.FROM_ADDRESS, p.TO_ADDRESS],
                        't': np.r_[p.TIMESTAMP, p.TIMESTAMP]}).sort_values(['w', 't', 'i'])
    same = inc.w.to_numpy()[1:] == inc.w.to_numpy()[:-1]
    close = np.diff(inc.t.to_numpy()) <= minutes * 60
    keep = same & close
    a, b = inc.i.to_numpy()[:-1][keep], inc.i.to_numpy()[1:][keep]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def motif_features(directory):
    p = load(directory)
    n = len(p)
    out = pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH})
    logv = np.log1p(p.VALUE)
    for d in SCALES:
        c = components(p, d)
        g = pd.DataFrame({'c': c, 't': p.TIMESTAMP, 'v': p.VALUE, 'lv': logv, 's': p.FROM_ADDRESS, 'r': p.TO_ADDRESS})
        grp = g.groupby('c')
        size = grp.t.transform('size')
        span = (grp.t.transform('max') - grp.t.transform('min')) / 60
        senders = g.groupby('c').s.nunique()
        receivers = g.groupby('c').r.nunique()
        wallets = pd.concat([g[['c', 's']].rename(columns={'s': 'w'}), g[['c', 'r']].rename(columns={'r': 'w'})]).drop_duplicates()
        nw = wallets.groupby('c').size()
        # Wallet roles inside the burst.
        outdeg = g.groupby(['c', 's']).size().rename('out')
        indeg = g.groupby(['c', 'r']).size().rename('in')
        roles = pd.concat([outdeg.rename_axis(['c', 'w']), indeg.rename_axis(['c', 'w'])], axis=1).fillna(0)
        roles['mid'] = (roles['out'] > 0) & (roles['in'] > 0)
        rc = roles.groupby(level=0)
        n_mid = rc.mid.sum()
        max_out, max_in = rc['out'].max(), rc['in'].max()
        n_src = rc.apply(lambda r: int(((r['out'] > 0) & (r['in'] == 0)).sum()))
        n_sink = rc.apply(lambda r: int(((r['in'] > 0) & (r['out'] == 0)).sum()))
        # Flow through intermediates: value in vs value out for wallets that both receive and send.
        vin = g.groupby(['c', 'r']).v.sum().rename_axis(['c', 'w'])
        vout = g.groupby(['c', 's']).v.sum().rename_axis(['c', 'w'])
        flow = pd.concat([vin.rename('vin'), vout.rename('vout')], axis=1).fillna(0)
        flow = flow[roles.mid.reindex(flow.index).fillna(False).to_numpy()]
        fsum = flow.groupby(level=0).sum()
        conserve = (fsum.vout / fsum.vin.replace(0, np.nan)).clip(0, 5)
        pairs = g.groupby('c').apply(lambda x: x[['s', 'r']].drop_duplicates().shape[0]) if False else \
            g.drop_duplicates(['c', 's', 'r']).groupby('c').size()
        key = g.c
        pre = f'motif_{d}m_'
        out[pre + 'size'] = size.to_numpy()
        out[pre + 'log_size'] = np.log1p(size.to_numpy())
        out[pre + 'span_min'] = span.to_numpy()
        out[pre + 'rate'] = (size / (span + 1)).to_numpy()
        out[pre + 'wallets'] = key.map(nw).to_numpy()
        out[pre + 'senders'] = key.map(senders).to_numpy()
        out[pre + 'receivers'] = key.map(receivers).to_numpy()
        out[pre + 'cyclomatic'] = (size - key.map(nw) + 1).to_numpy()
        out[pre + 'repeat_pairs'] = (size - key.map(pairs)).to_numpy()
        out[pre + 'intermediates'] = key.map(n_mid).to_numpy()
        out[pre + 'sources'] = key.map(n_src).to_numpy()
        out[pre + 'sinks'] = key.map(n_sink).to_numpy()
        out[pre + 'max_out'] = key.map(max_out).to_numpy()
        out[pre + 'max_in'] = key.map(max_in).to_numpy()
        out[pre + 'conservation'] = key.map(conserve).fillna(-1).to_numpy()
        out[pre + 'log_value_sum'] = np.log1p(grp.v.transform('sum')).to_numpy()
        out[pre + 'value_cv'] = (grp.v.transform('std') / grp.v.transform('mean')).fillna(0).to_numpy()
        out[pre + 'value_rel'] = (logv - grp.lv.transform('mean')).to_numpy()
        out[pre + 'position'] = ((g.t - grp.t.transform('min')) / 60 / (span + 1e-9)).fillna(0).clip(0, 1).to_numpy()
        out[pre + 'gap_median'] = grp.t.transform(lambda t: np.median(np.diff(np.sort(t.to_numpy()))) / 60 if len(t) > 1 else -1).to_numpy()
    f = out.drop(columns='TRANSACTION_HASH')
    assert len(out) == n and out.TRANSACTION_HASH.is_unique and np.isfinite(f.to_numpy(dtype=float)).all()
    return out
