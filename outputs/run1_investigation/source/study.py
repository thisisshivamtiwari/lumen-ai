from common import *
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
import time
f,a,X,y,fold,base=load();ids=f.TRANSACTION_HASH.to_numpy();groups=groups_for(a);locked=np.load(ROOT/'outputs/v2_experiments/locked_splits.npz');assert np.array_equal(locked['fold_A'],fold)
for k,(_,b) in enumerate(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261024).split(X,y,groups)):assert (fold[b]==k).all()
ex=pd.read_parquet(O/'flexible_train.parquet').set_index('TRANSACTION_HASH').loc[ids].reset_index(drop=True);Z=pd.concat([X,ex],axis=1);drop=[c for c in X if '_300s' in c or c in ['starter_sender_burst_10m','starter_receiver_burst_10m']]
configs={'weight3':{'hypothesis':'Medium has modest positive evidence above a strongly negative base logit; weighting all positives may learn weak context interactions. May increase lawful-motif false positives.','features':X.columns.tolist(),'params':dict(PARAMS,class_weights=[1,3])},'without5min':{'hypothesis':'Lawful false positives receive large short-burst/5-minute-flow boosts absent in most medium cases. Remove this block to test shortcut dependence; likely loss of easy-case recall.','features':[c for c in X if c not in drop],'params':PARAMS},'flexible_sums':{'hypothesis':'Fixed whole-window totals can miss multi-payment split/merge compatibility in noisy history. Add 33 bounded cumulative-prefix compatibility features. They may create chance matches and are not proof of provenance.','features':Z.columns.tolist(),'params':PARAMS}}
plan={'created_after':'Run 1 reproduction, OOF feature profiles and held-out SHAP diagnosis; before candidate fits','candidates':configs,'fixed_primary':'Exact V1 SGKF folds, no changed target or annotation-derived inputs','maximum_development_candidates':3,'retained_promotion_criteria':json.loads((ROOT/'outputs/v2_experiments/experiment_plan.json').read_text())['promotion_rule'],'additional_guard':'No loss in pooled OOF true positives at same global top-402 cutoff; P/R both reported. No slice-only promotion.','confirmation_seed':20261025,'wallet_and_time':'Same locked B0/C for every candidate, B1 only frozen qualifying finalist. Previously inspected protocols; no independent labelled final set.','test_feedback':'Run1 rounded aggregate results used to motivate research; no private labels or new leaderboard query as a tuning oracle; no test features used in model selection.'};(O/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
rows=[];preds={};baseline_rows=[]
for k in range(3):baseline_rows.append(dict(experiment='v1',track='A',fold=k,seed=20261024,**metrics(y[fold==k],base[fold==k],ids[fold==k])))
baseline_rows.append(dict(experiment='v1',track='A_pooled',fold=-1,seed=20261024,**metrics(y,base,ids)))
for track in ['B0','C']:
 q=pd.read_csv(ROOT/'outputs/v2_experiments/runs'/f'v1_{track}_seed20261024.csv').set_index('TRANSACTION_HASH');ii=np.flatnonzero(locked[track]==1);assert set(q.index)==set(ids[ii]);baseline_rows.append(dict(experiment='v1',track=track,fold=-1,seed=20261024,**metrics(y[ii],q.loc[ids[ii],'score'].to_numpy(),ids[ii])))
rows.extend(baseline_rows)
def train(name,track,seed,k=-1):
 conf=configs[name];z=Z[conf['features']];tr=np.flatnonzero(fold!=k) if track=='A' else np.flatnonzero(locked[track]==0);va=np.flatnonzero(fold==k) if track=='A' else np.flatnonzero(locked[track]==1);params=dict(conf['params'],random_seed=seed);start=time.perf_counter();m=CatBoostClassifier(**params);m.fit(z.iloc[tr],y[tr]);ss=m.predict_proba(z.iloc[va])[:,1];m.save_model(str(O/'runs'/f'{name}_{track}_{k}_seed{seed}.cbm'));r=dict(experiment=name,track=track,fold=k,seed=seed,features=len(conf['features']),runtime_seconds=time.perf_counter()-start,fit_rows=len(tr),**metrics(y[va],ss,ids[va]));print(name,track,k,seed,'AP',round(r['ap'],6),'P1',round(r['precision_top1'],4),'R1',round(r['recall_top1'],4),flush=True);pd.DataFrame({'TRANSACTION_HASH':ids[va],'score':ss}).to_csv(O/'runs'/f'{name}_{track}_{k}_seed{seed}.csv',index=False);return va,ss,r
for name in configs:
 pred=np.full(len(y),np.nan)
 for k in range(3):
  va,ss,r=train(name,'A',20261024,k);rows.append(r);pred[va]=ss;pd.DataFrame(rows).to_csv(O/'all_metrics.csv',index=False)
 preds[name]=pred;pd.DataFrame({'TRANSACTION_HASH':ids,'fold':fold,'score':pred}).to_csv(O/'runs'/f'{name}_oof.csv',index=False);r=dict(experiment=name,track='A_pooled',seed=20261024,fold=-1,**metrics(y,pred,ids));rows.append(r);pd.DataFrame(rows).to_csv(O/'all_metrics.csv',index=False)
 # Exactly the same category rows, negative background and global budget across candidates.
 c=category_metrics(a,pred,ids);c['experiment']=name;c.to_csv(O/f'{name}_categories.csv',index=False)
# Freeze one candidate solely on primary A mean AP; no selecting after B/C results.
summary=pd.DataFrame(rows);mean=summary[summary.track.eq('A')].groupby('experiment').ap.mean();best=mean.drop('v1').idxmax();(O/'frozen_finalist.json').write_text(json.dumps({'candidate':best,'primary_mean_ap':mean.to_dict(),'rule':'highest fixed A mean AP among 3 predeclared candidates; no later tuning or alternate winner selected using B/C'},indent=2)+'\n');print('Frozen finalist',best,flush=True)
for track in ['B0','C']:
 for name in configs:
  _,_,r=train(name,track,20261024);rows.append(r);pd.DataFrame(rows).to_csv(O/'all_metrics.csv',index=False)
# Seed robustness only for the frozen finalist; protected V1 confirmation was already run.
seed_base=np.full(len(y),np.nan);new_seed=np.full(len(y),np.nan)
for k in range(3):
 ii=np.flatnonzero(fold==k);q=pd.read_csv(ROOT/'outputs/v2_experiments/runs'/f'v1_A_confirm_fold{k}_seed20261025.csv').set_index('TRANSACTION_HASH');seed_base[ii]=q.loc[ids[ii],'score'];rows.append(dict(experiment='v1',track='A',fold=k,seed=20261025,**metrics(y[ii],seed_base[ii],ids[ii])));va,ss,r=train(best,'A',20261025,k);new_seed[va]=ss;rows.append(r);pd.DataFrame(rows).to_csv(O/'all_metrics.csv',index=False)
# Paired resampling of the original annotation protection groups; annotations never enter predictors.
ug,gi=np.unique(groups,return_inverse=True);rng=np.random.default_rng(20261024);deltas={n:[] for n in configs}
for _ in range(300):
 w=np.bincount(rng.integers(len(ug),size=len(ug)),minlength=len(ug))[gi]
 if any(np.dot(w[fold==k],y[fold==k])==0 for k in range(3)):continue
 for name in configs:
  delta=[]
  for k in range(3):
   mask=fold==k;delta.append(average_precision_score(y[mask],preds[name][mask],sample_weight=w[mask])-average_precision_score(y[mask],base[mask],sample_weight=w[mask]))
  deltas[name].append(np.mean(delta))
final=[];df=pd.DataFrame(rows)
for name in configs:
 av=df[(df.experiment==name)&df.track.eq('A')&df.seed.eq(20261024)].sort_values('fold');baseav=df[(df.experiment=='v1')&df.track.eq('A')&df.seed.eq(20261024)].sort_values('fold');pool=df[(df.experiment==name)&df.track.eq('A_pooled')].iloc[0];bp=df[(df.experiment=='v1')&df.track.eq('A_pooled')].iloc[0];ci=np.quantile(deltas[name],[.025,.975]);gain=float(av.ap.mean()-baseav.ap.mean());holdout={tr:float(df[(df.experiment==name)&df.track.eq(tr)].iloc[0].ap-df[(df.experiment=='v1')&df.track.eq(tr)].iloc[0].ap) for tr in ['B0','C']}
 checks={'mean_ap_gain_ge_0.01':gain>=.01,'at_least_two_folds_better':int(np.sum(av.ap.to_numpy()>baseav.ap.to_numpy()))>=2,'paired_group_ci_lower_positive':ci[0]>0,'B0_no_ap_regression_over_0.02':holdout['B0']>=-.02,'C_no_ap_regression_over_0.02':holdout['C']>=-.02,'global_top1_no_TP_loss':int(pool.true_positives_top1)>=int(bp.true_positives_top1)}
 seed_delta=None
 if name==best:
  seed_delta=float(np.mean([average_precision_score(y[fold==k],new_seed[fold==k])-average_precision_score(y[fold==k],seed_base[fold==k]) for k in range(3)]));checks['confirmation_seed_improves']=seed_delta>0
 else:checks['confirmation_seed_improves']=False
 accepted=all(checks.values());final.append({'experiment':name,'mean_fold_ap':float(av.ap.mean()),'pooled_ap':float(pool.ap),'mean_fold_precision_top1':float(av.precision_top1.mean()),'mean_fold_recall_top1':float(av.recall_top1.mean()),'pooled_precision_top1':float(pool.precision_top1),'pooled_recall_top1':float(pool.recall_top1),'pooled_true_positives_top1':int(pool.true_positives_top1),'absolute_mean_ap_delta':gain,'relative_mean_ap_delta':gain/float(baseav.ap.mean()),'improving_folds':int(np.sum(av.ap.to_numpy()>baseav.ap.to_numpy())),'ci_delta_low':float(ci[0]),'ci_delta_high':float(ci[1]),'B0_ap_delta':holdout['B0'],'C_ap_delta':holdout['C'],'confirmation_seed_mean_ap_delta':seed_delta,'local_gates_passed':accepted,'checks':{kk:bool(vv) for kk,vv in checks.items()}})
# Only an otherwise-qualifying frozen finalist can spend the reserved B1 protocol.
for rec in final:
 if rec['experiment']==best and rec['local_gates_passed']:
  qb=CatBoostClassifier(**PARAMS);tr=np.flatnonzero(locked['B1']==0);va=np.flatnonzero(locked['B1']==1);qb.fit(X.iloc[tr],y[tr]);bscore=qb.predict_proba(X.iloc[va])[:,1];_,ss,r=train(best,'B1',20261024);rows.append(r);delta=float(average_precision_score(y[va],ss)-average_precision_score(y[va],bscore));rec['B1_ap_delta']=delta;rec['local_gates_passed']=delta>=-.02;rec['requires_clean_raw_reproduction_before_promotion']=True
(O/'acceptance.json').write_text(json.dumps({'candidates':final,'frozen_finalist':best,'source_reproduction_required_for_promotion':True,'independent_label_holdout_available':False,'leaderboard_used_for_tuning':False,'submitted':False},indent=2)+'\n');pd.DataFrame([{k:v for k,v in z.items() if k!='checks'} for z in final]).to_csv(O/'experiment_table.csv',index=False);pd.DataFrame(rows).to_csv(O/'all_metrics.csv',index=False);print(json.dumps(final,indent=2),flush=True)
