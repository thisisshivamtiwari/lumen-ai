"""Confirmation split: fit on labels from the first five days, evaluate on labels from the last two.

Features use the full authorised seven-day window (as at test time); only labels are split by time.
Every variant is rebuilt from scratch on this split, including its inner folds for context/aux scores.
Run from the project root: .venv/bin/python outputs/v9_confirm/source/confirm.py <variant>
Variants: v1, v5, v7, v8.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/v9_confirm'
sys.path.insert(0, str(ROOT / 'solution'))
from context import load_edges, context_features  # noqa: E402
from evaluate_v1 import groups_for  # noqa: E402
from run_v3 import blend_scores  # noqa: E402

CUTOFF = pd.Timestamp('2026-09-06 00:00:00', tz='UTC').value // 10**9
CONFIG = json.loads((ROOT / 'solution/config.json').read_text())
PARAMS = dict(CONFIG['params'])


def load():
    f = pd.read_parquet(ROOT / 'outputs/features_v1/train.parquet')
    m = pd.read_parquet(ROOT / 'outputs/v3_context/motif_ego_train.parquet').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
    a = pd.read_csv(ROOT / 'data/training/train/labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index()
    t = load_edges(ROOT / 'data/training/train').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].TIMESTAMP.to_numpy()
    return f, m, a, t


def fit(X, y):
    return CatBoostClassifier(**PARAMS).fit(X, y)


def main():
    variant = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    f, m, a, t = load()
    y = a.is_suspicious.to_numpy()
    train, test = t < CUTOFF, t >= CUTOFF
    X = f[CONFIG['features']].reset_index(drop=True)
    if variant != 'v1':
        X = pd.concat([X, m], axis=1)
    fit_rows = train.copy()
    if variant in ('v7', 'v8'):
        fit_rows &= ~((y == 1) & a.difficulty.eq('hard').to_numpy())
    # Inner grouped folds over the labelled (first five days) rows only.
    tr_idx = np.flatnonzero(train)
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=PARAMS['random_seed']).split(
        tr_idx, y[tr_idx], groups_for(a.iloc[tr_idx])))
    inner = [(tr_idx[i], tr_idx[j]) for i, j in inner]
    if variant == 'v8':
        relay = a.family.isin(['aggregate_relay', 'distributed_relay']).astype(int).to_numpy()
        aux_params = dict(PARAMS, iterations=500, learning_rate=0.08)
        aux = np.full(len(y), np.nan)
        for i, j in inner:
            aux[j] = CatBoostClassifier(**aux_params).fit(X.iloc[i], relay[i]).predict_proba(X.iloc[j])[:, 1]
        aux[test] = CatBoostClassifier(**aux_params).fit(X[train], relay[train]).predict_proba(X[test])[:, 1]
        X = X.assign(aux_relay=aux)
    stage1 = fit(X[fit_rows], y[fit_rows])
    s1 = np.full(len(y), np.nan)
    s1[test] = stage1.predict_proba(X[test])[:, 1]
    if variant == 'v1':
        score = s1[test]
    else:
        for i, j in inner:
            keep = i[fit_rows[i]]
            s1[j] = fit(X.iloc[keep], y[keep]).predict_proba(X.iloc[j])[:, 1]
        assert np.isfinite(s1).all()
        ctx = context_features(load_edges(ROOT / 'data/training/train'), f.TRANSACTION_HASH, s1).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
        Z = pd.concat([X, ctx], axis=1)
        s2 = fit(Z[fit_rows], y[fit_rows]).predict_proba(Z[test])[:, 1]
        score = blend_scores(s1[test], s2)
    pd.DataFrame({'TRANSACTION_HASH': f.TRANSACTION_HASH[test].to_numpy(), 'score': score}).to_csv(OUT / f'{variant}_scores.csv', index=False)
    print(variant, 'done', int(train.sum()), 'label rows,', int(test.sum()), 'evaluated', flush=True)


if __name__ == '__main__':
    main()
