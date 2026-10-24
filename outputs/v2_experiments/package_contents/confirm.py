"""Frozen finalist evaluation, no further model selection after holdout inspection."""
from pathlib import Path
import os,json,time
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from evaluate_v1 import PARAMS,groups_for,columns_for
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments'
f=pd.read_parquet(ROOT/'outputs/features_v1/train.parquet');a=pd.read_csv(ROOT/'data/training/train/labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH];y=a.is_suspicious.to_numpy();X=f.drop(columns='TRANSACTION_HASH');ex=pd.read_parquet(O/'extra_train.parquet').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True);XX=pd.concat([X.reset_index(drop=True),ex],axis=1);splits=np.load(O/'locked_splits.npz');fold=splits['fold_A']
table=pd.read_csv(O/'model_comparison.csv');dev=table[table.track.eq('A')&~table.experiment.eq('v1')];best=dev.sort_values('mean_fold_ap',ascending=False).iloc[0].experiment
frozen={'candidate':best,'selection':'highest mean A AP among predeclared candidates; frozen before B/C/confirmation seed','candidate_mean_ap':float(dev.set_index('experiment').loc[best,'mean_fold_ap']),'no_further_tuning':True};(O/'frozen_finalist.json').write_text(json.dumps(frozen,indent=2)+'\n');print('Frozen finalist',best,flush=True)
def data(kind):
 if kind=='chrono':return XX
 if kind=='chrono_no_static':return XX[columns_for(XX.columns,'behaviour')]
 return X

def fit(kind,tr,va,seed,track,stress=None):
 start=time.perf_counter();pp=dict(PARAMS,random_seed=seed)
 if kind=='depth4':pp['depth']=4
 if kind=='depth8':pp['depth']=8
 z=data(kind);m=CatBoostClassifier(**pp);m.fit(z.iloc[tr],y[tr]);ss=m.predict_proba(z.iloc[va])[:,1];ap=average_precision_score(y[va],ss)
 rec={'experiment':kind,'track':track,'seed':seed,'fit_rows':len(tr),'validation_rows':len(va),'positives':int(y[va].sum()),'prevalence':float(y[va].mean()),'ap':float(ap),'runtime_seconds':time.perf_counter()-start}
 if stress is not None:
  common=f.TRANSACTION_HASH.iloc[va].isin(stress.index).to_numpy();ix=va[common];thin=stress.loc[f.TRANSACTION_HASH.iloc[ix],z.columns];ts=m.predict_proba(thin)[:,1];rec['sparse_rows']=len(ix);rec['same_targets_full_ap']=float(average_precision_score(y[ix],ss[common]));rec['same_targets_sparse_ap']=float(average_precision_score(y[ix],ts));rec['sparse_ap_delta']=rec['same_targets_sparse_ap']-rec['same_targets_full_ap']
 pd.DataFrame({'TRANSACTION_HASH':f.TRANSACTION_HASH.iloc[va],'score':ss}).to_csv(O/'runs'/f'{kind}_{track}_seed{seed}.csv',index=False)
 print(kind,track,seed,round(ap,6),flush=True);return ss,rec
# Logistic is a sanity baseline only. If it wins unexpectedly, fail closed rather than change confirmation family.
assert best!='logistic','Unexpected linear winner: needs separately declared confirmation.'
results=[];conf={}
for kind in ['v1',best]:
 pred=np.full(len(y),np.nan)
 for k in range(3):
  va=np.flatnonzero(fold==k);ss,rec=fit(kind,np.flatnonzero(fold!=k),va,20261025,f'A_confirm_fold{k}');pred[va]=ss;results.append(rec)
 conf[kind]=pred
 pd.DataFrame(results).to_csv(O/'confirmation_results.csv',index=False)
primary_gain=frozen['candidate_mean_ap']-float(table[table.experiment.eq('v1')&table.track.eq('A')].iloc[0].mean_fold_ap)
tracks=['B0','C']+(['B1'] if primary_gain>=.01 else [])
for track in tracks:
 stress=pd.read_parquet(O/'sparse_context_features.parquet').set_index('TRANSACTION_HASH') if track=='C' else None
 for kind in ['v1',best]:
  _,rec=fit(kind,np.flatnonzero(splits[track]==0),np.flatnonzero(splits[track]==1),20261024,track,stress);results.append(rec)
 pd.DataFrame(results).to_csv(O/'confirmation_results.csv',index=False)
# Paired group bootstrap on fixed OOF predictions estimates conditional sampling uncertainty.
bas=pd.read_csv(ROOT/'outputs/strategy3_v1/graph_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH,'score'].to_numpy();new=pd.read_csv(O/'runs'/f'{best}_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH,'score'].to_numpy();g=groups_for(a.reset_index());ug,gi=np.unique(g,return_inverse=True);rng=np.random.default_rng(20261024);diff=[]
for b in range(300):
 weights=np.bincount(rng.integers(len(ug),size=len(ug)),minlength=len(ug))[gi];ds=[]
 for k in range(3):
  mask=fold==k
  if np.dot(weights[mask],y[mask])==0:break
  ds.append(average_precision_score(y[mask],new[mask],sample_weight=weights[mask])-average_precision_score(y[mask],bas[mask],sample_weight=weights[mask]))
 if len(ds)==3:diff.append(np.mean(ds))
basefold=np.array([average_precision_score(y[fold==k],bas[fold==k]) for k in range(3)]);newfold=np.array([average_precision_score(y[fold==k],new[fold==k]) for k in range(3)]);seed_delta=np.mean([average_precision_score(y[fold==k],conf[best][fold==k])-average_precision_score(y[fold==k],conf['v1'][fold==k]) for k in range(3)])
checks={'absolute_gain_at_least_0.01':bool(np.mean(newfold-basefold)>=.01),'improves_at_least_2_folds':bool(np.sum(newfold>basefold)>=2),'paired_group_bootstrap_lower_positive':bool(np.quantile(diff,.025)>0),'confirmation_seed_gain_positive':bool(seed_delta>0)}
deltas={}
for track in tracks:
 r=pd.DataFrame(results);q=r[r.track.eq(track)].set_index('experiment');delta=float(q.loc[best,'ap']-q.loc['v1','ap']);deltas[track]=delta;checks[track+'_no_regression_over_0.02']=delta>=-.02
verdict={'candidate':best,'accepted_by_local_evidence':all(checks.values()),'checks':checks,'absolute_mean_ap_delta':float(np.mean(newfold-basefold)),'relative_mean_ap_delta':float(np.mean(newfold-basefold)/np.mean(basefold)),'fold_deltas':(newfold-basefold).tolist(),'paired_group_bootstrap_delta_95pct':np.quantile(diff,[.025,.975]).tolist(),'bootstrap_replicates':len(diff),'confirmation_seed_mean_ap_delta':float(seed_delta),'holdout_deltas':deltas,'reserved_B1_used':bool('B1' in tracks),'reserved_B1_note':'Opened only if primary >=1 point gain; otherwise preserved for future work.','warning':'Conditional group bootstrap excludes retraining uncertainty; two seeds and two random node partitions cannot establish hidden-test performance.'}
(O/'acceptance.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict,indent=2),flush=True)
