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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/v3_context'
sys.path.insert(0, str(ROOT / 'solution'))
from context import load_edges, context_features  # noqa: E402
import motif  # noqa: E402


def burst_ids(directory, scales=(2, 5, 10)):
    p = motif.load(directory)
    return pd.DataFrame({'TRANSACTION_HASH': p.TRANSACTION_HASH, **{f'{d}m': motif.components(p, d) for d in scales}})

CONFIG = json.loads((ROOT / 'solution/config.json').read_text())
PARAMS = dict(CONFIG['params'])
COLS = CONFIG['features']


def load():
    f = pd.read_parquet(ROOT / 'outputs/features_v1/train.parquet')
    a = pd.read_csv(ROOT / 'outputs/strategy3_v1/graph_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index()
    return f, a


WEIGHTS = None


def fit(X, y, seed=None):
    params = dict(PARAMS)
    if seed is not None:
        params['random_seed'] = seed
    if WEIGHTS is None:
        return CatBoostClassifier(**params).fit(X, y)
    w = WEIGHTS.loc[X.index].to_numpy()
    keep = w > 0
    return CatBoostClassifier(**params).fit(X[keep], np.asarray(y)[keep], sample_weight=w[keep])


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


def with_motif(f, cache='motif_train.parquet'):
    m = pd.read_parquet(OUT / cache).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
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


def run_context(f, a, X=None, base=None, bursts=None):
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
        ctx = context_features(edges, f.TRANSACTION_HASH, s, bursts).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
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
    global WEIGHTS
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
    elif name == 'motif_context_burst':
        base = pd.read_csv(OUT / 'motif_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
        oof = run_context(f, a, with_motif(f), base, burst_ids(ROOT / 'data/training/train'))
    elif name == 'motif_ego':
        oof = run_grouped(with_motif(f, 'motif_ego_train.parquet'), a)
    elif name == 'motif_ego_context':
        base = pd.read_csv(OUT / 'motif_ego_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
        oof = run_context(f, a, with_motif(f, 'motif_ego_train.parquet'), base)
    elif name == 'motif_ego_hardneg':
        # Training weights only (annotations never become features): look-alike negatives x2, background x0.7.
        lookalike = (a.is_suspicious == 0) & (a.family != 'background')
        WEIGHTS = pd.Series(np.where(lookalike, 2.0, np.where(a.family == 'background', 0.7, 1.0)), index=pd.RangeIndex(len(a)))
        oof = run_grouped(with_motif(f, 'motif_ego_train.parquet'), a)
    elif name == 'hgb_motif_ego':
        from sklearn.ensemble import HistGradientBoostingClassifier
        X = with_motif(f, 'motif_ego_train.parquet')
        y = a.is_suspicious.to_numpy(); fold = a.fold.to_numpy(); oof = np.full(len(y), np.nan)
        for i in range(3):
            hgb = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                                 l2_regularization=1.0, max_features=0.5, early_stopping=False, random_state=20261024)
            oof[fold == i] = hgb.fit(X[fold != i], y[fold != i]).predict_proba(X[fold == i])[:, 1]
            print(f'fold {i} AP {average_precision_score(y[fold == i], oof[fold == i]):.4f}', flush=True)
    elif name == 'pseudo_motif_ego':
        # Simulated transductive self-training: the held-out fold plays the unlabelled test set.
        X = with_motif(f, 'motif_ego_train.parquet')
        y = a.is_suspicious.to_numpy(); fold = a.fold.to_numpy(); oof = np.full(len(y), np.nan)
        base = pd.read_csv(OUT / 'motif_ego_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
        for i in range(3):
            tr, te = fold != i, fold == i
            s = base[te]
            hi, lo = np.quantile(s, 1 - 0.015), np.quantile(s, 0.5)
            pick = (s >= hi) | (s <= lo)
            Xp = pd.concat([X[tr], X[te][pick]]); yp = np.r_[y[tr], (s[pick] >= hi).astype(int)]
            oof[te] = fit(Xp.reset_index(drop=True), yp).predict_proba(X[te])[:, 1]
            print(f'fold {i} AP {average_precision_score(y[te], oof[te]):.4f} (base {average_precision_score(y[te], s):.4f}; pseudo positives {int((s >= hi).sum())}, precision {y[te][s >= hi].mean():.3f})', flush=True)
    elif name in ('oneoff_nohard', 'oneoff_nohard_context'):
        WEIGHTS = pd.Series(np.where((a.is_suspicious == 1) & (a.difficulty == 'hard'), 0.0, 1.0), index=pd.RangeIndex(len(a)))
        X = with_motif(f, 'oneoff_train.parquet')
        if name == 'oneoff_nohard':
            oof = run_grouped(X, a)
        else:
            base = pd.read_csv(OUT / 'oneoff_nohard_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
            oof = run_context(f, a, X, base)
    elif name in ('bc_nohard', 'bc_nohard_context', 'bc_keephard', 'bc_keephard_context'):
        # Burst-context features on top of motif+ego; hard-tier positives not fitted unless keephard.
        if 'nohard' in name:
            WEIGHTS = pd.Series(np.where((a.is_suspicious == 1) & (a.difficulty == 'hard'), 0.0, 1.0), index=pd.RangeIndex(len(a)))
        X = with_motif(f, 'motif_ego_bc_train.parquet')
        if name.endswith('_context'):
            stem = name[:-8]
            base = pd.read_csv(OUT / f'{stem}_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
            oof = run_context(f, a, X, base)
        else:
            oof = run_grouped(X, a)
    elif name in ('nohard', 'nohard_context'):
        # Training-row selection only: hard relay and ambiguous positives are left out of fitting
        # (they are still scored and evaluated); annotations never become features.
        WEIGHTS = pd.Series(np.where((a.is_suspicious == 1) & (a.difficulty == 'hard'), 0.0, 1.0), index=pd.RangeIndex(len(a)))
        X = with_motif(f, 'motif_ego_train.parquet')
        if name == 'nohard':
            oof = run_grouped(X, a)
        else:
            base = pd.read_csv(OUT / 'nohard_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
            oof = run_context(f, a, X, base)
    elif name == 'aux':
        # Auxiliary relay-ness target (training supervision only), nested by fold like the context stage.
        WEIGHTS = pd.Series(np.where((a.is_suspicious == 1) & (a.difficulty == 'hard'), 0.0, 1.0), index=pd.RangeIndex(len(a)))
        X = with_motif(f, 'motif_ego_train.parquet')
        y = a.is_suspicious.to_numpy(); fold = a.fold.to_numpy()
        relay = a.family.isin(['aggregate_relay', 'distributed_relay']).astype(int).to_numpy()
        aux_params = dict(PARAMS, iterations=500, learning_rate=0.08)
        auxfit = lambda m: CatBoostClassifier(**aux_params).fit(X[m], relay[m])
        single = {g: auxfit(fold == g).predict_proba(X)[:, 1] for g in range(3)}
        oof = np.full(len(y), np.nan); auxcols = {}
        for i in range(3):
            g, h = [k for k in range(3) if k != i]
            aux = np.where(fold == g, single[h], np.where(fold == h, single[g], auxfit(fold != i).predict_proba(X)[:, 1]))
            print(f'aux fold {i} relay AUC on held-out fold {roc_auc_score(relay[fold == i], aux[fold == i]):.4f}', flush=True)
            Z = X.assign(aux_relay=aux)
            tr = (fold != i) & (WEIGHTS.to_numpy() > 0)
            oof[fold == i] = CatBoostClassifier(**PARAMS).fit(Z[tr], y[tr]).predict_proba(Z[fold == i])[:, 1]
            auxcols[i] = aux
            print(f'fold {i} AP {average_precision_score(y[fold == i], oof[fold == i]):.4f}', flush=True)
        np.save(OUT / 'aux_nested.npy', np.vstack([auxcols[i] for i in range(3)]))
    elif name in ('aux_context', 'aux_context_norm'):
        # Nested context stage on top of the aux stage-1 model; per outer fold the aux column is the nested one.
        WEIGHTS = pd.Series(np.where((a.is_suspicious == 1) & (a.difficulty == 'hard'), 0.0, 1.0), index=pd.RangeIndex(len(a)))
        X = with_motif(f, 'motif_ego_train.parquet')
        y = a.is_suspicious.to_numpy(); fold = a.fold.to_numpy(); keep = WEIGHTS.to_numpy() > 0
        aux = np.load(OUT / 'aux_nested.npy')
        base = pd.read_csv(OUT / 'aux_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
        edges = load_edges(ROOT / 'data/training/train')
        oof = np.full(len(y), np.nan)
        for i in range(3):
            g, h = [k for k in range(3) if k != i]
            Z = X.assign(aux_relay=aux[i])
            sg = CatBoostClassifier(**PARAMS).fit(Z[(fold == g) & keep], y[(fold == g) & keep]).predict_proba(Z)[:, 1]
            sh = CatBoostClassifier(**PARAMS).fit(Z[(fold == h) & keep], y[(fold == h) & keep]).predict_proba(Z)[:, 1]
            s = np.where(fold == i, base, np.where(fold == g, sh, sg))
            ctx = context_features(edges, f.TRANSACTION_HASH, s, normalized=name.endswith('_norm')).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
            Z2 = pd.concat([Z.reset_index(drop=True), ctx], axis=1)
            tr = (fold != i) & keep
            oof[fold == i] = CatBoostClassifier(**PARAMS).fit(Z2[tr], y[tr]).predict_proba(Z2[fold == i])[:, 1]
            print(f'outer fold {i} AP {average_precision_score(y[fold == i], oof[fold == i]):.4f}', flush=True)
    elif (OUT / f'{name}_train.parquet').exists():
        oof = run_grouped(with_motif(f, f'{name}_train.parquet'), a)
    elif name.endswith('_context') and (OUT / f'{name[:-8]}_train.parquet').exists():
        stem = name[:-8]
        base = pd.read_csv(OUT / f'{stem}_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].score.to_numpy()
        oof = run_context(f, a, with_motif(f, f'{stem}_train.parquet'), base)
    elif name == 'motif_long':
        PARAMS.update(iterations=2500, learning_rate=0.03)
        oof = run_grouped(with_motif(f), a)
    assert np.isfinite(oof).all()
    pd.DataFrame({'TRANSACTION_HASH': f.TRANSACTION_HASH, 'fold': a.fold, 'score': oof}).to_csv(OUT / f'{name}_oof.csv', index=False)
    result = {'v1': summary('v1', a, a.score.to_numpy()), name: summary(name, a, oof), 'seconds': time.perf_counter() - t}
    (OUT / f'{name}_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
