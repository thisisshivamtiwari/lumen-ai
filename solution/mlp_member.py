"""Neural blend member: a multilayer perceptron on the V14 feature table, as a diversity partner for V15.

OOF on the identical grouped folds (hard-tier positives not fitted); if the rank blend with the V15
not-fitted-hard OOF improves mean fold AP on at least two folds, fits on all training rows and scores
the test split. Run from the project root: .venv/bin/python solution/mlp_member.py
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v16_mlp'
sys.path.insert(0, str(ROOT / 'solution'))
sys.path.insert(0, str(ROOT / 'outputs/v3_context/source'))
from experiment import load, summary, OUT as EXP  # noqa: E402
from run_v3 import blend_scores, with_motif, with_burst, with_paths  # noqa: E402
from features import build_features  # noqa: E402

SEED = 20261024
CONFIG = json.loads((ROOT / 'solution/config.json').read_text())


def mlp():
    return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, batch_size=256, learning_rate_init=1e-3,
                         max_iter=40, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6, random_state=SEED)


def fit_predict(Xtr, ytr, Xte):
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=SEED).fit(Xtr)
    m = mlp().fit(qt.transform(Xtr), ytr)
    return m.predict_proba(qt.transform(Xte))[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    f, a = load()
    fold = a.fold.to_numpy()
    y = a.is_suspicious.to_numpy()
    cache = pd.read_parquet(EXP / 'bcp_train.parquet').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
    X = pd.concat([f[CONFIG['features']].reset_index(drop=True), cache], axis=1)
    keep = ~((y == 1) & a.difficulty.eq('hard').to_numpy())
    oof = np.full(len(y), np.nan)
    for i in range(3):
        tr = (fold != i) & keep
        oof[fold == i] = fit_predict(X[tr].to_numpy(), y[tr], X[fold == i].to_numpy())
        print(f'fold {i} MLP AP {average_precision_score(y[fold == i], oof[fold == i]):.4f}  ({time.perf_counter() - start:.0f}s)', flush=True)
    pd.DataFrame({'TRANSACTION_HASH': f.TRANSACTION_HASH, 'fold': fold, 'score': oof}).to_csv(OUT / 'mlp_oof.csv', index=False)
    get = lambda n: pd.read_csv(EXP / f'{n}_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()

    def bl(x, z, w=0.5):
        out = np.zeros(len(x))
        for i in range(3):
            m = fold == i
            r1 = pd.Series(x[m]).rank(pct=True).to_numpy(); r2 = pd.Series(z[m]).rank(pct=True).to_numpy()
            out[m] = (1 - w) * r1 + w * r2 + 1e-7 * (x[m] + z[m])
        return out
    v11 = bl(get('bc_nohard'), get('bc_nohard_context')); v12 = bl(get('nh_bcg'), get('nh_bcg_context')); v14 = bl(get('nh_bcp'), get('nh_bcp_context'))
    v15 = bl(bl(v11, v12), v14)
    rows = {'V15 half': v15, 'MLP alone': oof, 'V15+MLP w=0.25': bl(v15, oof, 0.25), 'V15+MLP w=0.5': bl(v15, oof, 0.5)}
    res = {k: summary(k, a, s) for k, s in rows.items()}
    for k, r in res.items():
        print(f"{k:16s} mean {100 * r['mean_fold_ap']:.2f} pooled {100 * r['pooled_ap']:.2f} top402 {r['tp_top402']} folds {[round(100 * x, 2) for x in r['fold_ap']]} easy {r['category_ap']['easy']:.3f} med {r['category_ap']['medium']:.3f}", flush=True)
    base = res['V15 half']
    best_w, best = None, base
    for k, w in (('V15+MLP w=0.25', 0.25), ('V15+MLP w=0.5', 0.5)):
        r = res[k]
        better = sum(x > b for x, b in zip(r['fold_ap'], base['fold_ap']))
        if r['mean_fold_ap'] > best['mean_fold_ap'] and better >= 2:
            best_w, best = w, r
    (OUT / 'mlp_metrics.json').write_text(json.dumps({'results': res, 'accepted_weight': best_w}, indent=2) + '\n')
    if best_w is None:
        print('MLP member rejected: no blend weight improves mean fold AP on >=2 folds', flush=True)
        return
    print(f'MLP member accepted at weight {best_w}; fitting on all training rows and scoring test', flush=True)
    train_dir, test_dir = ROOT / 'data/training/train', ROOT / 'data/test'
    test = build_features(test_dir)
    test = with_paths(with_burst(with_motif(test, test_dir), test_dir), test_dir)
    T = test[X.columns]
    assert T.columns.tolist() == X.columns.tolist() and np.isfinite(T.to_numpy()).all()
    s = fit_predict(X[keep].to_numpy(), y[keep], T.to_numpy())
    mlp_test = pd.DataFrame({'TRANSACTION_HASH': test.TRANSACTION_HASH, 'score': s})
    mlp_test.to_csv(OUT / 'mlp_test.csv', index=False)
    (OUT / 'accepted_weight.txt').write_text(str(best_w))
    print(f'done {time.perf_counter() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
