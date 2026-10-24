"""Edge-aware one-off features: was a one-off relationship given time to repeat?

Legitimate hub relationships recur every day or two. A pair first seen near the end of the
observed week (or last seen near its start) has not had time to repeat, so "one-off" is an
artefact of the window edge. Each wallet's typical repeat gap says how much time was needed.
Label-free, computed from one split's own edges, identical code for train and test.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from payments import load_payments  # noqa: E402

WINDOW = 600
DEFAULT_GAP_H = 168.0


def load(directory):
    p = pd.DataFrame(load_payments(directory))[['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE']]
    p['TIMESTAMP'] = p.TIMESTAMP.astype('int64')
    return p.reset_index(drop=True)


def wallet_repeat_gap(p):
    """Median, over a wallet's repeating directed pairs (either direction), of each pair's median gap in hours."""
    q = p.sort_values(['FROM_ADDRESS', 'TO_ADDRESS', 'TIMESTAMP'])
    gap = q.groupby(['FROM_ADDRESS', 'TO_ADDRESS']).TIMESTAMP.diff() / 3600
    pair_gap = gap.dropna().groupby([q.FROM_ADDRESS[gap.notna()], q.TO_ADDRESS[gap.notna()]]).median().rename('g').reset_index()
    both = pd.concat([pair_gap[['FROM_ADDRESS', 'g']].rename(columns={'FROM_ADDRESS': 'w'}),
                      pair_gap[['TO_ADDRESS', 'g']].rename(columns={'TO_ADDRESS': 'w'})])
    return both.groupby('w').g.median().clip(1.0, DEFAULT_GAP_H)


def side_share(p, wallet_col, flag):
    """Share of the wallet's other payments on this side within +-WINDOW that carry the flag; -1 if none."""
    n = len(p)
    t = p.TIMESTAMP.to_numpy()
    share = np.full(n, -1.0)
    wallet_share = np.zeros(n)
    for w, idx in p.groupby(wallet_col).indices.items():
        idx = idx[np.argsort(t[idx], kind='stable')]
        tw, fw = t[idx], flag[idx].astype(float)
        c1, cf = np.r_[0, np.cumsum(np.ones(len(idx)))], np.r_[0, np.cumsum(fw)]
        lo = np.searchsorted(tw, tw - WINDOW, side='left')
        hi = np.searchsorted(tw, tw + WINDOW, side='right')
        others = c1[hi] - c1[lo] - 1
        flagged = cf[hi] - cf[lo] - fw
        share[idx] = np.where(others > 0, flagged / np.maximum(others, 1), -1.0)
        wallet_share[idx] = fw.mean()
    return share, wallet_share


def edge_context_features(directory):
    p = load(directory)
    n = len(p)
    t = p.TIMESTAMP.to_numpy()
    L = (t - t.min()) / 3600
    R = (t.max() - t) / 3600
    gaps = wallet_repeat_gap(p)
    gs = p.FROM_ADDRESS.map(gaps).fillna(DEFAULT_GAP_H).to_numpy()
    gr = p.TO_ADDRESS.map(gaps).fillna(DEFAULT_GAP_H).to_numpy()
    G = np.maximum(gs, gr)
    pair_count = p.groupby(['FROM_ADDRESS', 'TO_ADDRESS']).TRANSACTION_HASH.transform('size').to_numpy()
    oneoff = (pair_count == 1)
    edge = np.minimum(L, R)
    unexplained = oneoff & (edge >= G)
    f = pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH.to_numpy(),
                      'ec_sender_gap_h': gs, 'ec_receiver_gap_h': gr,
                      'ec_left_over_gap': np.minimum(L / G, 20.0), 'ec_right_over_gap': np.minimum(R / G, 20.0),
                      'ec_edge_ratio': np.minimum(edge / G, 20.0),
                      'ec_oneoff_unexplained': unexplained.astype(float),
                      'ec_oneoff_strength': oneoff * np.minimum(1.0, edge / G)})
    for side, col in (('sender_out', 'FROM_ADDRESS'), ('receiver_in', 'TO_ADDRESS')):
        share, wallet_share = side_share(p, col, unexplained)
        f[f'ec_{side}_other_unexplained_share'] = share
        f[f'ec_{side}_unexplained_vs_wallet'] = np.where(share >= 0, share - wallet_share, 0.0)
    assert len(f) == n and f.TRANSACTION_HASH.is_unique and np.isfinite(f.drop(columns='TRANSACTION_HASH').to_numpy()).all()
    return f
