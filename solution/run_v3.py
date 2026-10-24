"""V3: V1 CatBoost blended with a second stage using first-stage scores of nearby payments sharing a wallet.

Stage 1 is the V1 model. Training context uses V1 grouped out-of-fold scores, so no training row's
context comes from a model fitted on its own fold; test context uses the full V1 model's test scores.
Both are converted to within-split percentiles before building context features.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from features import build_features
from context import load_edges, context_features
from evaluate_v1 import groups_for
from motif import motif_features, ego_features
from burst_context import burst_context_features, burst_context_features_windowed, reconverge_features
from path_features import path_features
from run import validate_predictions


def blend_scores(stage1, stage2):
    """Equal-weight blend of stage-1 and stage-2 percentile ranks (selected on grouped OOF).
    Averaging two ranks creates exact ties; the mean raw probability breaks them without reordering."""
    rank = 0.5 * pd.Series(stage1).rank(pct=True).to_numpy() + 0.5 * pd.Series(stage2).rank(pct=True).to_numpy()
    return rank * (1 - 1e-6) + 1e-7 * (np.asarray(stage1) + np.asarray(stage2)) / 2


def with_motif(frame, directory, ego=True):
    m = motif_features(directory)
    if ego:
        m = m.merge(ego_features(directory), on='TRANSACTION_HASH', validate='1:1')
    m = m.set_index('TRANSACTION_HASH').loc[frame.TRANSACTION_HASH].reset_index(drop=True)
    return pd.concat([frame.reset_index(drop=True), m], axis=1)


def with_burst(frame, directory, extended=True, wide=False):
    b = burst_context_features(directory)
    if extended:
        b = b.merge(burst_context_features_windowed(directory), on='TRANSACTION_HASH', validate='1:1')
        if wide:
            b = b.merge(burst_context_features_windowed(directory, windows=(21600, 86400)), on='TRANSACTION_HASH', validate='1:1')
        b = b.merge(reconverge_features(directory), on='TRANSACTION_HASH', validate='1:1')
    b = b.set_index('TRANSACTION_HASH').loc[frame.TRANSACTION_HASH].reset_index(drop=True)
    return pd.concat([frame.reset_index(drop=True), b], axis=1)


def with_paths(frame, directory):
    s = path_features(directory).set_index('TRANSACTION_HASH').loc[frame.TRANSACTION_HASH].reset_index(drop=True)
    return pd.concat([frame.reset_index(drop=True), s], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--no-motif', action='store_true', help='V3 variant without multi-scale motif features')
    parser.add_argument('--no-ego', action='store_true', help='V4 variant without ego-network features')
    parser.add_argument('--keep-hard', action='store_true', help='V5 variant: also fit on hard-tier positives')
    parser.add_argument('--no-burst', action='store_true', help='V10 variant without burst-context features')
    parser.add_argument('--basic-burst', action='store_true', help='V11 variant: +-10 min burst context only, no wider windows or re-convergence')
    parser.add_argument('--no-paths', action='store_true', help='V12 variant without slow scatter-gather path features')
    parser.add_argument('--wide-windows', action='store_true', help='V17 stretch: add +-6 h and +-24 h burst-context windows (stage-1 ablation +0.28, below acceptance bar)')
    parser.add_argument('--aux', action='store_true', help='Experimental V8: auxiliary relay-ness feature trained on the family annotation (off by default)')
    args = parser.parse_args()
    start = time.perf_counter()
    config = json.loads(args.config.read_text())
    params, columns = config['params'], config['features']
    train = build_features(args.train)
    if not args.no_motif:
        train = with_motif(train, args.train, not args.no_ego)
        columns = columns + [c for c in train.columns if c.startswith(('motif_', 'ego_'))]
    if not args.no_burst:
        train = with_burst(train, args.train, not args.basic_burst, args.wide_windows)
        columns = columns + [c for c in train.columns if c.startswith('bc')]
    if not args.no_paths:
        train = with_paths(train, args.train)
        columns = columns + [c for c in train.columns if c.startswith('sg_')]
    labels = pd.read_csv(args.train / 'labels.csv').set_index('TRANSACTION_HASH').loc[train.TRANSACTION_HASH].reset_index()
    y = labels.is_suspicious.to_numpy()
    X = train[columns]
    assert np.isfinite(X.to_numpy()).all()

    # Rows used for fitting. By default the hard-tier positives (planted historical relays and the
    # deliberately ambiguous payments) are left out of fitting: they are rarely rankable and teach the
    # models to promote ~20,000 legitimate relay look-alikes. They are still scored. Selection only;
    # annotations never become features.
    fit_rows = np.ones(len(y), bool) if args.keep_hard else ~((y == 1) & labels.difficulty.eq('hard').to_numpy())

    test = build_features(args.test)
    if not args.no_motif:
        test = with_motif(test, args.test, not args.no_ego)
    if not args.no_burst:
        test = with_burst(test, args.test, not args.basic_burst, args.wide_windows)
    if not args.no_paths:
        test = with_paths(test, args.test)
    T = test[columns]
    assert T.columns.tolist() == X.columns.tolist() and T.dtypes.equals(X.dtypes)

    groups = groups_for(labels)
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=params['random_seed']).split(X, y, groups))
    if args.aux:
        # Auxiliary relay-ness model: target is whether a training payment belongs to relay-type activity
        # (aggregate or distributed relay family, either label). The family annotation is a training target
        # only; at test time the feature is this model's prediction from observable features.
        relay = labels.family.isin(['aggregate_relay', 'distributed_relay']).astype(int).to_numpy()
        aux_params = dict(params, iterations=500, learning_rate=0.08)
        aux_oof = np.full(len(y), np.nan)
        for a, b in folds:
            aux_oof[b] = CatBoostClassifier(**aux_params).fit(X.iloc[a], relay[a]).predict_proba(X.iloc[b])[:, 1]
        aux_model = CatBoostClassifier(**aux_params).fit(X, relay)
        X = X.assign(aux_relay=aux_oof)
        T = T.assign(aux_relay=aux_model.predict_proba(T)[:, list(aux_model.classes_).index(1)])

    # Stage 1 out-of-fold scores on the V1 grouped folds (annotation groups only define folds).
    oof = np.full(len(y), np.nan)
    for a, b in folds:
        a = a[fit_rows[a]]
        oof[b] = CatBoostClassifier(**params).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    assert np.isfinite(oof).all()
    stage1 = CatBoostClassifier(**params).fit(X[fit_rows], y[fit_rows])
    test_stage1 = stage1.predict_proba(T)[:, list(stage1.classes_).index(1)]

    ctx_train = context_features(load_edges(args.train), train.TRANSACTION_HASH, oof).set_index('TRANSACTION_HASH').loc[train.TRANSACTION_HASH].reset_index(drop=True)
    ctx_test = context_features(load_edges(args.test), test.TRANSACTION_HASH, test_stage1).set_index('TRANSACTION_HASH').loc[test.TRANSACTION_HASH].reset_index(drop=True)
    Z = pd.concat([X.reset_index(drop=True), ctx_train], axis=1)
    ZT = pd.concat([T.reset_index(drop=True), ctx_test], axis=1)
    assert Z.columns.tolist() == ZT.columns.tolist()
    stage2 = CatBoostClassifier(**params).fit(Z[fit_rows], y[fit_rows])
    test_stage2 = stage2.predict_proba(ZT)[:, list(stage2.classes_).index(1)]
    scores = blend_scores(test_stage1, test_stage2)

    expected = pq.read_table(args.test / 'transactions.parquet', columns=['TRANSACTION_HASH']).to_pandas().TRANSACTION_HASH
    predictions = pd.DataFrame({'TRANSACTION_HASH': test.TRANSACTION_HASH, 'score': scores})
    predictions = predictions.set_index('TRANSACTION_HASH').loc[expected].reset_index()
    validate_predictions(predictions, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False, float_format='%.17g')
    reloaded = pd.read_csv(args.output)
    validate_predictions(reloaded, expected)
    assert reloaded.TRANSACTION_HASH.equals(predictions.TRANSACTION_HASH)
    assert args.output.stat().st_size <= 12 * 1024 ** 2
    report = {'model': 'V3 two-stage context, 50/50 rank blend with stage 1', 'motif_features': not args.no_motif, 'ego_features': not (args.no_motif or args.no_ego), 'hard_positives_fitted': args.keep_hard, 'aux_relay_feature': args.aux, 'burst_context_features': not args.no_burst, 'extended_burst_context': not (args.no_burst or args.basic_burst), 'path_features': not args.no_paths, 'fitted_rows': int(fit_rows.sum()),'training_rows': len(train), 'test_rows': len(test),
              'features': Z.shape[1], 'elapsed_seconds': time.perf_counter() - start,
              'csv_bytes': args.output.stat().st_size, 'csv_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'score_min': float(scores.min()), 'score_max': float(scores.max()), 'unique_scores': int(len(np.unique(scores))),
              'spearman_stage2_vs_stage1': float(pd.Series(test_stage2).corr(pd.Series(test_stage1), method='spearman'))}
    args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
