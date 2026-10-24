"""Train the fixed selected Strategy3 model and export a validated test CSV."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from features import build_features


def validate_predictions(predictions, expected):
    assert predictions.columns.tolist()==['TRANSACTION_HASH','score']
    assert expected.notna().all() and expected.is_unique
    assert len(predictions)==len(expected) and predictions.TRANSACTION_HASH.is_unique
    assert set(predictions.TRANSACTION_HASH)==set(expected)
    assert np.isfinite(predictions.score).all() and predictions.score.between(0,1).all()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('predictions.csv'))
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--save-model',type=Path)
    args=parser.parse_args();start=time.perf_counter()
    config=json.loads(args.config.read_text())
    train=build_features(args.train)
    labels=pd.read_csv(args.train/'labels.csv',usecols=['TRANSACTION_HASH','is_suspicious'])
    assert labels.TRANSACTION_HASH.notna().all() and labels.TRANSACTION_HASH.is_unique
    assert set(labels.TRANSACTION_HASH)==set(train.TRANSACTION_HASH)
    y=labels.set_index('TRANSACTION_HASH').loc[train.TRANSACTION_HASH,'is_suspicious'].to_numpy()
    assert set(np.unique(y))=={0,1}
    columns=config['features'];assert len(columns)==len(set(columns))
    forbidden={'TRANSACTION_HASH','family','difficulty','scenario_id','cohort_id','is_suspicious','split'}
    assert not set(columns)&forbidden
    X=train[columns]
    assert np.isfinite(X.to_numpy()).all()
    model=CatBoostClassifier(**config['params']);model.fit(X,y)
    test=build_features(args.test);T=test[columns]
    assert T.columns.tolist()==X.columns.tolist() and T.dtypes.equals(X.dtypes)
    scores=model.predict_proba(T)[:,list(model.classes_).index(1)]
    expected=pq.read_table(args.test/'transactions.parquet',columns=['TRANSACTION_HASH']).to_pandas().TRANSACTION_HASH
    predictions=pd.DataFrame({'TRANSACTION_HASH':test.TRANSACTION_HASH,'score':scores})
    predictions=predictions.set_index('TRANSACTION_HASH').loc[expected].reset_index()
    validate_predictions(predictions,expected)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output,index=False,float_format='%.17g')
    reloaded=pd.read_csv(args.output);validate_predictions(reloaded,expected)
    assert reloaded.TRANSACTION_HASH.equals(predictions.TRANSACTION_HASH)
    assert np.allclose(reloaded.score,predictions.score,rtol=0,atol=1e-15)
    assert args.output.stat().st_size<=12*1024**2
    if args.save_model:
        args.save_model.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(args.save_model))
        restored=CatBoostClassifier();restored.load_model(str(args.save_model))
        assert np.allclose(restored.predict_proba(T.iloc[:1000])[:,1],scores[:1000],rtol=0,atol=1e-14)
    report={'training_rows':len(train),'test_rows':len(test),'features':len(columns),
            'feature_set':config['feature_set'],'elapsed_seconds':time.perf_counter()-start,
            'csv_bytes':args.output.stat().st_size,'csv_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'score_min':float(scores.min()),'score_max':float(scores.max()),'unique_scores':int(len(np.unique(scores))),
            'tests':{'exact_ids':'PASS','one_score_per_id':'PASS','finite_range':'PASS','csv_roundtrip':'PASS'},
            'competition_score':None,'leaderboard_submitted':False}
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
