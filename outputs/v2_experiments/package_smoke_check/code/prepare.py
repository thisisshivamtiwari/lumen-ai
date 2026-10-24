"""Build clean-workspace caches; never overwrite existing V1 artifacts.
Use --reference-root to point the V2 scripts at these regenerated reference files.
The primary scripts accept LUMEN_WORKSPACE; this creates its expected layout.
"""
from pathlib import Path
import argparse,json,os
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from features import build_features
from extra_features import build_extra
from evaluate_v1 import PARAMS,groups_for,columns_for
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--train',type=Path,required=True);p.add_argument('--test',type=Path,required=True);args=p.parse_args();R=args.workspace.resolve();assert not R.exists(),'Use a new empty workspace to protect V1';R.mkdir(parents=True);(R/'data/training').mkdir(parents=True);os.symlink(args.train.resolve(),R/'data/training/train');os.symlink(args.test.resolve(),R/'data/test');(R/'outputs/features_v1').mkdir(parents=True);(R/'outputs/strategy3_v1').mkdir();(R/'outputs/v2_experiments/runs').mkdir(parents=True)
f=build_features(args.train);f.to_parquet(R/'outputs/features_v1/train.parquet',index=False);build_extra(args.train).to_parquet(R/'outputs/v2_experiments/extra_train.parquet',index=False)
a=pd.read_csv(args.train/'labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index();y=a.is_suspicious.to_numpy();X=f.drop(columns='TRANSACTION_HASH');splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261024).split(X,y,groups_for(a)));report={'candidates':{}}
for name in ['behaviour','graph']:
 z=X[columns_for(X.columns,name)];pred=np.zeros(len(y));fold=np.zeros(len(y),int);aps=[]
 for k,(tr,va) in enumerate(splits):
  m=CatBoostClassifier(**PARAMS);m.fit(z.iloc[tr],y[tr]);pred[va]=m.predict_proba(z.iloc[va])[:,1];fold[va]=k;aps.append(average_precision_score(y[va],pred[va]))
 o=a.copy();o['fold']=fold;o['score']=pred;o.to_csv(R/'outputs/strategy3_v1'/f'{name}_oof.csv',index=False);report['candidates'][name]={'mean_ap':float(np.mean(aps)),'pooled':{'ap':float(average_precision_score(y,pred))},'folds':[{'ap':float(v)} for v in aps]}
(R/'outputs/strategy3_v1/evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print('Prepared fresh workspace',R,flush=True)
