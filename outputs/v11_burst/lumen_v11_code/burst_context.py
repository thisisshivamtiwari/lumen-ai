"""Burst-context features: is this payment's ten-minute burst normal for this wallet?

Label-free, computed from one split's own edges. For each payment, two sides are described: the
sender's outgoing payments and the receiver's incoming payments within +-10 minutes. A legitimate
hub's one-off payment during a busy spell looks like a planted fan-out leg to the existing features;
these features say whether one-off counterparties and bursts are usual for that wallet.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from payments import load_payments  # noqa: E402

WINDOW = 600  # seconds either side


def load(directory):
    p = pd.DataFrame(load_payments(directory))[['TRANSACTION_HASH', 'TIMESTAMP', 'FROM_ADDRESS', 'TO_ADDRESS', 'VALUE']]
    p['TIMESTAMP'] = p.TIMESTAMP.astype('int64')
    p['VALUE'] = p.VALUE.astype(float)
    return p.reset_index(drop=True)


def side_features(p, wallet_col, name, oneoff, span_seconds):
    n = len(p)
    t, v = p.TIMESTAMP.to_numpy(), p.VALUE.to_numpy()
    out = {k: np.zeros(n) for k in ('n', 'other_oneoff_share', 'wallet_oneoff_share', 'rate_ratio', 'amount_cv', 'amount_cv_vs_wallet')}
    for w, idx in p.groupby(wallet_col).indices.items():
        idx = idx[np.argsort(t[idx], kind='stable')]
        tw, vw, ow = t[idx], v[idx], oneoff[idx].astype(float)
        c1, cv, cv2, co = (np.r_[0, np.cumsum(x)] for x in (np.ones(len(idx)), vw, vw * vw, ow))
        lo = np.searchsorted(tw, tw - WINDOW, side='left')
        hi = np.searchsorted(tw, tw + WINDOW, side='right')
        cnt = c1[hi] - c1[lo]
        s, s2 = cv[hi] - cv[lo], cv2[hi] - cv2[lo]
        mean = s / cnt
        var = np.maximum(s2 / cnt - mean * mean, 0.0)
        cvw = np.where(cnt > 1, np.sqrt(var) / np.maximum(mean, 1e-9), 0.0)
        others = cnt - 1
        oo = co[hi] - co[lo] - ow
        wallet_share = ow.mean()
        wallet_cv = vw.std() / max(vw.mean(), 1e-9) if len(vw) > 1 else 0.0
        expected = len(idx) * (2 * WINDOW) / span_seconds
        out['n'][idx] = cnt
        out['other_oneoff_share'][idx] = np.where(others > 0, oo / np.maximum(others, 1), -1.0)
        out['wallet_oneoff_share'][idx] = wallet_share
        out['rate_ratio'][idx] = np.log1p(cnt) - np.log1p(expected)
        out['amount_cv'][idx] = cvw
        out['amount_cv_vs_wallet'][idx] = cvw - wallet_cv
    f = pd.DataFrame({f'bc_{name}_{k}': val for k, val in out.items()})
    f[f'bc_{name}_oneoff_surprise'] = oneoff.astype(float) - f[f'bc_{name}_wallet_oneoff_share']
    f[f'bc_{name}_burst_vs_wallet_oneoff'] = np.where(f[f'bc_{name}_other_oneoff_share'] >= 0,
                                                      f[f'bc_{name}_other_oneoff_share'] - f[f'bc_{name}_wallet_oneoff_share'], 0.0)
    return f


def burst_context_features(directory):
    p = load(directory)
    pair_count = p.groupby(['FROM_ADDRESS', 'TO_ADDRESS']).TRANSACTION_HASH.transform('size').to_numpy()
    oneoff = pair_count == 1
    span = float(p.TIMESTAMP.max() - p.TIMESTAMP.min())
    f = pd.concat([side_features(p, 'FROM_ADDRESS', 'sender_out', oneoff, span),
                   side_features(p, 'TO_ADDRESS', 'receiver_in', oneoff, span)], axis=1)
    f.insert(0, 'TRANSACTION_HASH', p.TRANSACTION_HASH.to_numpy())
    assert len(f) == len(p) and f.TRANSACTION_HASH.is_unique and np.isfinite(f.drop(columns='TRANSACTION_HASH').to_numpy()).all()
    return f
