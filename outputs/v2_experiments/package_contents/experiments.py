"""Finite predeclared candidate comparison on exact fixed Track A folds."""
from pathlib import Path
import json,time,resource,sys,os
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from evaluate_v1 import PARAMS,groups_for,columns_for
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments'
f=pd.read_parquet(ROOT/'outputs/features_v1/train.parquet');X=f.drop(columns='TRANSACTION_HASH');a=pd.read_csv(ROOT/'data/training/train/labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH];y=a.is_suspicious.to_numpy();fold=np.load(O/'locked_splits.npz')['fold_A'];extra=pd.read_parquet(O/'extra_train.parquet').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True);XX=pd.concat([X.reset_index(drop=True),extra],axis=1)
params=dict(PARAMS)
def model(kind,seed=20261024):
 if kind=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2500,solver='lbfgs'))
 pp=dict(params,random_seed=seed)
 if kind=='depth4':pp.update(depth=4)
 if kind=='depth8':pp.update(depth=8)
 return CatBoostClassifier(**pp)
def data(kind):
 if kind in ['chrono','chrono_no_static']:return XX if kind=='chrono' else XX[columns_for(XX.columns,'behaviour')]
 return X
specs=[('logistic','Regularized linear sanity check','Scaled numeric V1 features; C=.1','Tests nonlinearity; may underfit interactions'),('depth4','Stronger regularization','CatBoost depth4; other V1 settings fixed','Reduce variance; may underfit motifs'),('depth8','Richer interactions','CatBoost depth8; other V1 settings fixed','Capture mixed context; risk overfit'),('chrono','Temporal value compatibility and local diversity','V1 plus 190 local diversity / ordered event-chain features','Separate repeated relay background from motifs; dense chance paths may dominate'),('chrono_no_static','Static graph block ablation','Chrono candidate with static paths/neighbour aggregates removed','Check incremental static graph value')]
rows=[];regs=[]
base=pd.read_csv(ROOT/'outputs/strategy3_v1/graph_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH,'score'].to_numpy();base_ap=[average_precision_score(y[fold==k],base[fold==k]) for k in range(3)]
rows.append(dict(experiment='v1',track='A',seed=20261024,features=len(X.columns),mean_fold_ap=np.mean(base_ap),pooled_ap=average_precision_score(y,base),fold0=base_ap[0],fold1=base_ap[1],fold2=base_ap[2],runtime_seconds=0,decision='protected baseline'))
selected=sys.argv[1:]
if selected:
 rows=pd.read_csv(O/'model_comparison.csv').to_dict('records');regs=pd.read_csv(O/'experiment_registry.csv').to_dict('records')
for kind,hyp,change,risk in specs:
 if selected and kind not in selected:continue
 rows=[r for r in rows if r['experiment']!=kind];regs=[r for r in regs if r['experiment']!=kind]
 start=time.perf_counter();z=data(kind);pred=np.full(len(y),np.nan);aps=[]
 for k in range(3):
  tr=fold!=k;va=fold==k;m=model(kind);m.fit(z.loc[tr],y[tr]);pred[va]=m.predict_proba(z.loc[va])[:,1];ap=average_precision_score(y[va],pred[va]);aps.append(ap);print(kind,k,round(ap,6),flush=True)
 elapsed=time.perf_counter()-start
 pd.DataFrame({'TRANSACTION_HASH':f.TRANSACTION_HASH,'fold':fold,'score':pred}).to_csv(O/'runs'/f'{kind}_oof.csv',index=False)
 row=dict(experiment=kind,track='A',seed=20261024,features=z.shape[1],mean_fold_ap=float(np.mean(aps)),pooled_ap=average_precision_score(y,pred),fold0=aps[0],fold1=aps[1],fold2=aps[2],runtime_seconds=elapsed,decision='pending confirmation');rows.append(row)
 regs.append(dict(experiment=kind,hypothesis=hyp,change=change,benefit_and_failure_mode=risk,track='A',mean_fold_ap=row['mean_fold_ap'],pooled_ap=row['pooled_ap'],runtime_seconds=elapsed,peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),decision='pending'))
 pd.DataFrame(rows).to_csv(O/'model_comparison.csv',index=False);pd.DataFrame(regs).to_csv(O/'experiment_registry.csv',index=False)
config={'baseline_params':params,'logistic':{'C':.1,'max_iter':2500,'solver':'lbfgs','scaler':'StandardScaler'},'feature_columns':list(XX.columns),'source_of_baseline':'immutable cached raw-derived features, exactly from verified packaged builder','candidate_order':[s[0] for s in specs]}
(O/'experiment_config.json').write_text(json.dumps(config,indent=2)+'\n');print(pd.DataFrame(rows).to_string(index=False),flush=True)
