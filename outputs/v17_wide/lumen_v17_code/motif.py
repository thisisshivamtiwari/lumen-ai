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
FLOW_SCALES = (60, 120)


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


def flow_components(p, minutes):
    """Link a payment into a wallet to a later payment out of it within D minutes carrying 5-105% of its value."""
    n = len(p)
    ins = pd.DataFrame({'a': np.arange(n), 'w': p.TO_ADDRESS, 'ta': p.TIMESTAMP, 'va': p.VALUE})
    outs = pd.DataFrame({'b': np.arange(n), 'w': p.FROM_ADDRESS, 'tb': p.TIMESTAMP, 'vb': p.VALUE})
    j = ins.merge(outs, on='w')
    dt = j.tb - j.ta
    keep = (dt >= 0) & (dt <= minutes * 60) & (j.vb <= 1.05 * j.va) & (j.vb >= 0.05 * j.va) & (j.a != j.b)
    graph = coo_matrix((np.ones(int(keep.sum())), (j.a[keep], j.b[keep])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def oneoff_components(p, minutes):
    """Bursts built only from payments whose sender-receiver pair occurs once all week; other payments are singletons.
    Recurring relationships (most background and relay traffic) cannot chain planted scenarios together."""
    n = len(p)
    pair_n = p.groupby(['FROM_ADDRESS', 'TO_ADDRESS']).TRANSACTION_HASH.transform('size').to_numpy()
    idx = np.flatnonzero(pair_n == 1)
    sub = p.iloc[idx].reset_index(drop=True)
    c_sub = components(sub, minutes)
    c = np.arange(n) + n
    c[idx] = c_sub
    return c


def motif_features(directory, flow=False, oneoff=()):
    p = load(directory)
    n = len(p)
    out = pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH})
    logv = np.log1p(p.VALUE)
    scales = [(f'{d}m', components(p, d)) for d in SCALES]
    if flow:
        scales += [(f'flow{d}m', flow_components(p, d)) for d in FLOW_SCALES]
    scales += [(f'oneoff{d}m', oneoff_components(p, d)) for d in oneoff]
    for name, c in scales:
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
        pre = f'motif_{name}_'
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


EGO_SCALES = (2, 5, 10)


def ego_features(directory):
    """Payments within +-D minutes sharing a wallet with this payment (1 hop), and payments within
    +-D minutes of the centre sharing a wallet with any of those (2 hops). Not transitive, so busy
    wallets cannot chain unrelated activity into the description."""
    p = load(directory)
    n = len(p)
    inc = pd.DataFrame({'i': np.r_[np.arange(n), np.arange(n)], 'w': np.r_[p.FROM_ADDRESS, p.TO_ADDRESS],
                        't': np.r_[p.TIMESTAMP, p.TIMESTAMP], 'side': np.r_[np.zeros(n), np.ones(n)].astype(int)})
    pairs = inc.merge(inc, on='w', suffixes=('', '_n'))
    pairs = pairs[(pairs.i != pairs.i_n) & ((pairs.t_n - pairs.t).abs() <= max(EGO_SCALES) * 60)]
    pairs = pairs.drop_duplicates(['i', 'i_n'])[['i', 'i_n', 't', 't_n']]
    out = {'TRANSACTION_HASH': p.TRANSACTION_HASH.to_numpy()}
    value = p.VALUE.to_numpy()
    for d in EGO_SCALES:
        one = pairs[(pairs.t_n - pairs.t).abs() <= d * 60][['i', 'i_n']]
        hop1 = one.groupby('i').size().reindex(np.arange(n), fill_value=0).to_numpy()
        two = one.merge(one.rename(columns={'i': 'i_n', 'i_n': 'i_2'}), on='i_n')
        two = two[two.i_2 != two.i]
        two = two.assign(t0=p.TIMESTAMP.to_numpy()[two.i], t2=p.TIMESTAMP.to_numpy()[two.i_2])
        two = two[(two.t2 - two.t0).abs() <= d * 60]
        reach = pd.concat([one.rename(columns={'i_n': 'j'}), two[['i', 'i_2']].rename(columns={'i_2': 'j'})]).drop_duplicates()
        hop2 = reach.groupby('i').size().reindex(np.arange(n), fill_value=0).to_numpy()
        reach = reach.assign(v=value[reach.j], s=p.FROM_ADDRESS.to_numpy()[reach.j], r=p.TO_ADDRESS.to_numpy()[reach.j])
        rg = reach.groupby('i')
        wallets = pd.concat([reach[['i', 's']].rename(columns={'s': 'w'}), reach[['i', 'r']].rename(columns={'r': 'w'})]).drop_duplicates().groupby('i').size()
        out[f'ego_{d}m_hop1'] = hop1
        out[f'ego_{d}m_hop2'] = hop2
        out[f'ego_{d}m_wallets'] = wallets.reindex(np.arange(n), fill_value=0).to_numpy()
        out[f'ego_{d}m_value_cv'] = (rg.v.std() / rg.v.mean()).reindex(np.arange(n)).fillna(0).to_numpy()
        out[f'ego_{d}m_value_ratio'] = (rg.v.sum().reindex(np.arange(n)).fillna(0).to_numpy() / np.maximum(value, 1e-9))
    f = pd.DataFrame(out)
    assert len(f) == n and np.isfinite(f.drop(columns='TRANSACTION_HASH').to_numpy(dtype=float)).all()
    return f


def chain_features(directory, hop_minutes=10, low=0.5, high=1.05):
    """Longest time-respecting, value-compatible path through each payment, and one-off fan counts."""
    p = load(directory)
    n = len(p)
    t, v = p.TIMESTAMP.to_numpy(), p.VALUE.to_numpy()
    ins = pd.DataFrame({'a': np.arange(n), 'w': p.TO_ADDRESS})
    outs = pd.DataFrame({'b': np.arange(n), 'w': p.FROM_ADDRESS})
    j = ins.merge(outs, on='w')
    dt = t[j.b.to_numpy()] - t[j.a.to_numpy()]
    ratio = v[j.b.to_numpy()] / np.maximum(v[j.a.to_numpy()], 1e-9)
    j = j[(dt >= 0) & (dt <= hop_minutes * 60) & (ratio >= low) & (ratio <= high) & (j.a != j.b).to_numpy()]
    succ = j.groupby('a').b.apply(np.array).to_dict()
    pred = j.groupby('b').a.apply(np.array).to_dict()
    order = np.lexsort((np.arange(n), t))
    fwd = np.ones(n, int)
    for i in order[::-1]:
        nxt = succ.get(i)
        if nxt is not None:
            fwd[i] = 1 + fwd[nxt].max()
    bwd = np.ones(n, int)
    for i in order:
        prv = pred.get(i)
        if prv is not None:
            bwd[i] = 1 + bwd[prv].max()
    out = pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH, 'chain_fwd': fwd, 'chain_bwd': bwd, 'chain_len': fwd + bwd - 1,
                        'chain_out_degree': np.array([len(succ.get(i, ())) for i in range(n)]),
                        'chain_in_degree': np.array([len(pred.get(i, ())) for i in range(n)])})
    # One-off fans: payments within +-5 minutes into the same receiver / out of the same sender whose pair occurs once all week.
    pair_n = p.groupby(['FROM_ADDRESS', 'TO_ADDRESS']).TRANSACTION_HASH.transform('size').to_numpy()
    p = p.assign(i=np.arange(n), oneoff=(pair_n == 1).astype(int))
    for side, col in (('in', 'TO_ADDRESS'), ('out', 'FROM_ADDRESS')):
        x = p[['i', col, 'TIMESTAMP', 'oneoff']].rename(columns={col: 'w'})
        m = x.merge(x, on='w', suffixes=('', '_n'))
        m = m[(m.i != m.i_n) & ((m.TIMESTAMP_n - m.TIMESTAMP).abs() <= 300)]
        g = m.groupby('i')
        out[f'fan5m_{side}_count'] = g.size().reindex(np.arange(n), fill_value=0).to_numpy()
        out[f'fan5m_{side}_oneoff'] = g.oneoff_n.sum().reindex(np.arange(n), fill_value=0).to_numpy()
    out['pair_oneoff'] = p.oneoff.to_numpy()
    return out


def schedule_features(directory):
    """Recurring-relationship signals: interval regularity and exact native-amount repeats per
    directed pair and per wallet (payroll, subscriptions and settlements repeat; planted motifs rarely do)."""
    import json
    import pyarrow.parquet as pq
    p = load(directory)
    raw = pq.read_table(Path(directory) / 'transactions.parquet', columns=['TRANSACTION_HASH', 'DELIVERED_AMOUNT']).to_pandas()
    p = p.merge(raw, on='TRANSACTION_HASH', validate='1:1')
    p['amt'] = p.DELIVERED_AMOUNT.map(lambda s: json.dumps(json.loads(s), sort_keys=True))
    p = p.sort_values(['TIMESTAMP', 'TRANSACTION_HASH']).reset_index(drop=True)
    p['hour'] = (p.TIMESTAMP % 86400) / 3600
    out = pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH})

    def gaps(keys):
        g = p.groupby(keys).TIMESTAMP
        gap = g.diff() / 3600
        stats = gap.groupby([p[k] for k in keys]).agg(['median', 'std', 'mean', 'count'])
        stats['cv'] = stats['std'] / stats['mean']
        stats['daily'] = ((stats['median'] / 24 - (stats['median'] / 24).round()).abs() < 0.05) & (stats['median'] >= 20)
        idx = pd.MultiIndex.from_frame(p[keys]) if len(keys) > 1 else p[keys[0]]
        return stats.reindex(idx)

    for name, keys in (('pair', ['FROM_ADDRESS', 'TO_ADDRESS']), ('sender_out', ['FROM_ADDRESS']), ('receiver_in', ['TO_ADDRESS'])):
        s = gaps(keys)
        out[f'sched_{name}_gap_median'] = s['median'].fillna(-1).to_numpy()
        out[f'sched_{name}_gap_cv'] = s['cv'].fillna(-1).to_numpy()
        out[f'sched_{name}_daily'] = s['daily'].fillna(False).astype(int).to_numpy()
        same_amt = p.groupby(keys + ['amt']).TRANSACTION_HASH.transform('size')
        total = p.groupby(keys).TRANSACTION_HASH.transform('size')
        out[f'sched_{name}_amount_repeats'] = (same_amt - 1).to_numpy()
        out[f'sched_{name}_amount_repeat_share'] = (p.assign(r=(same_amt > 1).astype(float)).groupby(keys).r.transform('mean')).to_numpy()
        out[f'sched_{name}_hour_std'] = p.groupby(keys).hour.transform('std').fillna(-1).to_numpy()
        out[f'sched_{name}_count'] = total.to_numpy()
    f = out
    assert f.TRANSACTION_HASH.is_unique and np.isfinite(f.drop(columns='TRANSACTION_HASH').to_numpy(dtype=float)).all()
    return f
