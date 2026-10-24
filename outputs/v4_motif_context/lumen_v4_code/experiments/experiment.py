"""V3 experiments on the exact V1 grouped folds.

context: nested two-stage model. For outer fold f, first-stage scores feeding the context features
come only from models that never saw fold-f labels (single-fold models for the other folds, the
V1 fold model for fold f). The second stage is fitted on the other folds and scored on fold f.
seedbag: V1 features and parameters, averaged over five seeds.
Run from the project root: .venv/bin/python outputs/v3_context/source/experiment.py {context|seedbag}
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/v3_context'
sys.path.insert(0, str(ROOT / 'solution'))
from context import load_edges, context_features  # noqa: E402

CONFIG = json.loads((ROOT / 'solution/config.json').read_text())
PARAMS = dict(CONFIG['params'])
COLS = CONFIG['features']


def load():
    f = pd.read_parquet(ROOT / 'outputs/features_v1/train.parquet')
    a = pd.read_csv(ROOT / 'outputs/strategy3_v1/graph_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index()
    return f, a


def fit(X, y, seed=None):
    params = dict(PARAMS)
    if seed is not None:
        params['random_seed'] = seed
    return CatBoostClassifier(**params).fit(X, y)


def summary(name, a, oof):
    y = a.is_suspicious.to_numpy()
    ids = a.TRANSACTION_HASH.to_numpy()
    folds = [float(average_precision_score(y[a.fold == i], oof[a.fold == i])) for i in range(3)]
    k = int(np.ceil(0.01 * len(y)))
    order = np.lexsort((ids.astype(str), -oof))
    cat = np.where(y == 0, 'legit', np.where(a.family.eq('ambiguous'), 'ambiguous', a.difficulty))
    cats = {}
    for c in ['easy', 'medium', 'hard']:
        m = (y == 0) | (cat == c)
        cats[c] = float(average_precision_score(y[m], oof[m]))
    return dict(name=name, fold_ap=folds, mean_fold_ap=float(np.mean(folds)), pooled_ap=float(average_precision_score(y, oof)),
                tp_top402=int(y[order[:k]].sum()), tp_top1250=int(y[order[:1250]].sum()), category_ap=cats)


def with_motif(f):
    m = pd.read_parquet(OUT / 'motif_train.parquet').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
    return pd.concat([f[COLS].reset_index(drop=True), m], axis=1)


def run_grouped(X, a):
    y = a.is_suspicious.to_numpy()
    fold = a.fold.to_numpy()
    oof = np.full(len(y), np.nan)
    for i in range(3):
        model = fit(X[fold != i], y[fold != i])
        oof[fold == i] = model.predict_proba(X[fold == i])[:, 1]
        print(f'fold {i} AP {average_precision_score(y[fold == i], oof[fold == i]):.4f}', flush=True)
    pd.Series(model.get_feature_importance(), index=X.columns).sort_values(ascending=False).head(40).to_csv(OUT / 'motif_importance.csv')
    return oof


def run_context(f, a, X=None, base=None):
    y = a.is_suspicious.to_numpy()
    fold = a.fold.to_numpy()
    X = f[COLS] if X is None else X
    base = a.score.to_numpy() if base is None else base
    edges = load_edges(ROOT / 'data/training/train')
    single = {}
    for g in range(3):
        t = time.perf_counter()
        single[g] = fit(X[fold == g], y[fold == g]).predict_proba(X)[:, 1]
        print(f'single-fold model {g} {time.perf_counter() - t:.0f}s', flush=True)
    oof = np.full(len(y), np.nan)
    importances = []
    for f_out in range(3):
        g, h = [i for i in range(3) if i != f_out]
        s = np.where(fold == f_out, base, np.where(fold == g, single[h], single[g]))
        ctx = context_features(edges, f.TRANSACTION_HASH, s).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
        Z = pd.concat([X.reset_index(drop=True), ctx], axis=1)
        tr, te = fold != f_out, fold == f_out
        model = fit(Z[tr], y[tr])
        oof[te] = model.predict_proba(Z[te])[:, 1]
        imp = pd.Series(model.get_feature_importance(), index=Z.columns)
        importances.append(imp)
        print(f'outer fold {f_out} AP {average_precision_score(y[te], oof[te]):.4f} (V1 {average_precision_score(y[te], a.score[te]):.4f})', flush=True)
    pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).head(40).to_csv(OUT / f'context_importance_{len(X.columns)}.csv')
    return oof


def run_seedbag(f, a):
    y = a.is_suspicious.to_numpy()
    fold = a.fold.to_numpy()
    X = f[COLS]
    preds = [a.score.to_numpy()]
    for seed in range(PARAMS['random_seed'] + 1, PARAMS['random_seed'] + 5):
        p = np.full(len(y), np.nan)
        for i in range(3):
            p[fold == i] = fit(X[fold != i], y[fold != i], seed).predict_proba(X[fold == i])[:, 1]
        preds.append(p)
        print(f'seed {seed} pooled AP {average_precision_score(y, p):.4f}', flush=True)
    np.save(OUT / 'seedbag_members.npy', np.vstack(preds))
    return np.mean(preds, axis=0)


def main():
    name = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    f, a = load()
    assert len(f) == 40131 and a.fold.between(0, 2).all()
    t = time.perf_counter()
    if name == 'context':
        oof = run_context(f, a)
    elif name == 'seedbag':
        oof = run_seedbag(f, a)
    elif name == 'motif':
        oof = run_grouped(with_motif(f), a)
    elif name == 'motif_context':
        base = pd.read_csv(OUT / 'motif_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
        oof = run_context(f, a, with_motif(f), base)
    assert np.isfinite(oof).all()
    pd.DataFrame({'TRANSACTION_HASH': f.TRANSACTION_HASH, 'fold': a.fold, 'score': oof}).to_csv(OUT / f'{name}_oof.csv', index=False)
    result = {'v1': summary('v1', a, a.score.to_numpy()), name: summary(name, a, oof), 'seconds': time.perf_counter() - t}
    (OUT / f'{name}_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
