"""Reproducible first benchmark on organiser label-free starter features.

Annotation fields are used exclusively for split construction and evaluation.
Full unlabelled split context is allowed by the released README.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from payments import load_payments
from starter_features import make_features


SEED = 20261024


def model_for(name):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=0))
    return CatBoostClassifier(iterations=700, depth=5, learning_rate=0.04,
                              loss_function="Logloss", random_seed=SEED,
                              thread_count=4, verbose=False, allow_writing_files=False)


def metrics(y, scores, hashes):
    y = np.asarray(y)
    scores = np.asarray(scores)
    n = len(y)
    k = max(1, int(np.ceil(n * .01)))
    order = np.lexsort((np.asarray(hashes, dtype=str), -scores))
    captured = int(y[order[:k]].sum())
    return dict(rows=n, positives=int(y.sum()),
                ap=float(average_precision_score(y, scores)),
                score=float(100 * average_precision_score(y, scores)),
                roc_auc=float(roc_auc_score(y, scores)),
                review_budget=k, precision_at_1pct=float(captured/k),
                recall_at_1pct=float(captured/y.sum()))


def validate_export(pred, expected):
    assert list(pred.columns) == ["TRANSACTION_HASH", "score"]
    assert len(pred) == len(expected)
    assert pred.TRANSACTION_HASH.is_unique
    assert set(pred.TRANSACTION_HASH) == set(expected)
    assert np.isfinite(pred.score).all() and pred.score.between(0, 1).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/training/train"))
    parser.add_argument("--test", type=Path, default=Path("data/test"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    payments = pd.DataFrame(load_payments(args.train))
    f = make_features(payments)
    labels = pd.read_csv(args.train/"labels.csv").set_index("TRANSACTION_HASH").loc[f.TRANSACTION_HASH].reset_index()
    assert labels.TRANSACTION_HASH.equals(f.TRANSACTION_HASH)
    y = labels.is_suspicious.to_numpy()
    X = f.drop(columns="TRANSACTION_HASH")
    assert set(X.columns).isdisjoint({"family", "difficulty", "scenario_id", "cohort_id", "is_suspicious", "split"})
    assert np.isfinite(X.to_numpy()).all()
    ids = f.TRANSACTION_HASH.to_numpy()
    groups = np.where(labels.cohort_id.notna(), "cohort:"+labels.cohort_id.fillna(""),
                      np.where(labels.scenario_id.notna(), "scenario:"+labels.scenario_id.fillna(""), "row:"+labels.TRANSACTION_HASH))
    grouped = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(X, y, groups))
    for a, b in grouped:
        assert not set(groups[a]).intersection(groups[b])
    random_fit, random_check = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=0)
    report = {"seed": SEED, "features": X.columns.tolist(),
              "context": "Full label-free window, separately per supplied split; explicitly permitted by README.",
              "groups": "cohort_id where supplied, else scenario_id, else individual background row; annotations are not predictors.",
              "models": {}}
    test = pd.DataFrame(load_payments(args.test))
    tf = make_features(test)
    assert tf.drop(columns="TRANSACTION_HASH").columns.tolist() == X.columns.tolist()
    for name in ["logistic", "catboost"]:
        print("Evaluating", name, flush=True)
        model = model_for(name)
        model.fit(X.iloc[random_fit], y[random_fit])
        random_scores = model.predict_proba(X.iloc[random_check])[:, 1]
        random_result = metrics(y[random_check], random_scores, ids[random_check])
        oof = np.full(len(y), np.nan)
        folds = []
        fold_ids = np.full(len(y), -1)
        for i, (a, b) in enumerate(grouped):
            model = model_for(name)
            model.fit(X.iloc[a], y[a])
            oof[b] = model.predict_proba(X.iloc[b])[:, 1]
            fold_ids[b] = i
            m = metrics(y[b], oof[b], ids[b]);folds.append(m)
            print(name, "grouped fold", i, "AP score", round(m["score"], 3), flush=True)
        assert np.isfinite(oof).all()
        pooled = metrics(y, oof, ids)
        annotations = labels.copy()
        annotations["fold"] = fold_ids
        annotations["score"] = oof
        annotations.to_csv(args.output/f"{name}_grouped_oof.csv", index=False)
        # Compare each positive difficulty against the same full legitimate background.
        slices = {}
        for difficulty in ["easy", "medium", "hard"]:
            mask = (y == 0) | labels.difficulty.eq(difficulty).to_numpy()
            slices[difficulty] = metrics(y[mask], oof[mask], ids[mask])
        model = model_for(name)
        model.fit(X, y)
        predictions = pd.DataFrame({"TRANSACTION_HASH": tf.TRANSACTION_HASH,
                                    "score": model.predict_proba(tf[X.columns])[:, 1]})
        validate_export(predictions, test.TRANSACTION_HASH)
        target = args.output/f"{name}_predictions.csv"
        predictions.to_csv(target, index=False, float_format="%.17g")
        loaded = pd.read_csv(target)
        validate_export(loaded, test.TRANSACTION_HASH)
        assert np.allclose(predictions.score, loaded.score, rtol=0, atol=1e-15)
        joblib.dump(model, args.output/f"{name}.joblib")
        reloaded = joblib.load(args.output/f"{name}.joblib")
        check = reloaded.predict_proba(tf[X.columns].iloc[:100])[:, 1]
        assert np.allclose(check, predictions.score.iloc[:100], rtol=0, atol=1e-14)
        report["models"][name] = dict(random_holdout=random_result, grouped_folds=folds,
            grouped_mean_ap=float(np.mean([m["ap"] for m in folds])), grouped_pooled=pooled,
            positive_difficulty_with_common_negative_background=slices,
            predictions=str(target), csv_bytes=target.stat().st_size,
            csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print(name, "random", round(random_result["score"], 3), "grouped pooled", round(pooled["score"], 3), flush=True)
        (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    best = max(report["models"], key=lambda n: report["models"][n]["grouped_mean_ap"])
    report["provisional_choice_by_grouped_mean_ap"] = best
    report["elapsed_seconds"] = time.perf_counter()-start
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Provisional choice:", best, "elapsed", round(report["elapsed_seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
