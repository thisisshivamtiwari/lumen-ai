from pathlib import Path
import sys,json,hashlib,os
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_v1 import groups_for
from payments import load_payments
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';rng=np.random.default_rng(20261024)
f=pd.read_parquet(ROOT/'outputs/features_v1/train.parquet');X=f.drop(columns='TRANSACTION_HASH')
a=pd.read_csv(ROOT/'data/training/train/labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index()
p=pd.DataFrame(load_payments(ROOT/'data/training/train')).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index()
y=a.is_suspicious.to_numpy();g=groups_for(a);score=pd.read_csv(ROOT/'outputs/strategy3_v1/graph_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH,'score'].to_numpy()
nodes=np.array(sorted(set(p.FROM_ADDRESS)|set(p.TO_ADDRESS)));s=np.searchsorted(nodes,p.FROM_ADDRESS);r=np.searchsorted(nodes,p.TO_ADDRESS)
splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=20261024).split(X,y,g));fold=np.full(len(y),-1)
fold_audit=[]
for k,(tr,va) in enumerate(splits):
 fold[va]=k;nt=set(s[tr])|set(r[tr]);nv=set(s[va])|set(r[va]);pairs_tr=set(zip(s[tr],r[tr]));pairs_va=list(zip(s[va],r[va]))
 fold_audit.append(dict(fold=k,train_rows=len(tr),validation_rows=len(va),groups_overlap=len(set(g[tr])&set(g[va])),wallet_overlap=len(nt&nv),validation_wallets=len(nv),validation_pair_seen_fraction=float(np.mean([v in pairs_tr for v in pairs_va]))))
adj=csr_matrix((np.ones(len(s)),(s,r)),shape=(len(nodes),len(nodes)));nc,components=connected_components(adj,directed=False)
# Lock two node partitions now. No labels are used to choose wallets.
plan={'primary':'A: exact V1 folds, mean fold AP; pooled AP separate','promotion_rule':'Require >=0.01 absolute mean AP gain, positive gain in >=2/3 folds, positive paired group-bootstrap 95% lower bound, confirmation seed improvement, no >0.02 absolute B or C regression. These local thresholds are governance choices, not significance guarantees.', 'selection':'A used for limited development; B0 and C checked after candidate selection, B1 kept untouched until frozen finalist. B splits use disjoint supervised endpoint wallets and full permitted unlabelled context; not graph-isolated.', 'candidate_budget':['regularized logistic baseline','CatBoost depth4','CatBoost depth8','CatBoost added chronological flow and local counterparty features','added features without static graph (ablation)'], 'confirmation_seed':20261025,'context':'All tracks use full authorized label-free window; C is offline temporal-label stress, not causal streaming validation.'}
(O/'experiment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
arrays={'fold_A':fold}
for k,seed in enumerate([20261029,20261103]):
 selected=np.random.default_rng(seed).permutation(len(nodes))[:len(nodes)//2];held=np.zeros(len(nodes),bool);held[selected]=True
 tr=(~held[s])&(~held[r]);va=held[s]&held[r];arrays['B'+str(k)]=np.where(tr,0,np.where(va,1,-1))
cut=1788652800 # overwrite with parsed timestamp to avoid constant mistakes
cut=int(pd.Timestamp('2026-09-06',tz='UTC').timestamp());arrays['C']=(p.TIMESTAMP.to_numpy(dtype=np.int64)>=cut).astype(int)
np.savez(O/'locked_splits.npz',**arrays)
report={'folds':fold_audit,'customer_nodes':len(nodes),'weak_components':nc,'largest_component_nodes':int(np.bincount(components).max()),'feature_columns':len(X.columns),'nonfinite_features':int((~np.isfinite(X.to_numpy())).sum()),'constant_columns':X.columns[X.nunique().eq(1)].tolist(),'duplicate_numeric_columns':int(X.T.duplicated().sum()),'B':{k:{'fit_rows':int((v==0).sum()),'holdout_rows':int((v==1).sum()),'holdout_positives':int(y[v==1].sum()),'dropped_cross_edges':int((v==-1).sum()),'wallet_overlap':0} for k,v in arrays.items() if k.startswith('B')}}
(O/'validation_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
# All audit slices are annotations-only; never passed to fitting.
def metrics(mask):
 yy=y[mask];ss=score[mask];n=len(yy);out={'n':n,'positives':int(yy.sum()),'prevalence':float(yy.mean()),'ap':float(average_precision_score(yy,ss)) if yy.sum() else None}
 for frac in [.005,.01,.03,.05,.1]:
  count=max(1,int(np.ceil(n*frac)));tp=int(yy[np.lexsort((f.TRANSACTION_HASH.to_numpy()[mask],-ss))[:count]].sum());out[f'precision_top_{frac:g}']=tp/count;out[f'recall_top_{frac:g}']=tp/max(1,yy.sum())
 return out
rows=[]
for k in [-1,0,1,2]:
 mask=np.ones(len(y),bool) if k==-1 else fold==k;rows.append({'slice':'pooled' if k==-1 else f'fold{k}',**metrics(mask)})
for field in ['difficulty','family']:
 for v in sorted(a.loc[y==1,field].unique()):
  mask=(y==0)|a[field].eq(v).to_numpy();rows.append({'slice':field+':'+v,**metrics(mask)})
for name,cat in [('day',pd.to_datetime(p.TIMESTAMP.astype('int64'),unit='s',utc=True).dt.strftime('%Y-%m-%d')),('amount_quintile',pd.qcut(X.log_delivered_usd,5,duplicates='drop').astype(str)),('sender_degree_quartile',pd.qcut(X.sender_out_full_unique,4,duplicates='drop').astype(str)),('execution_legs',X.execution_legs.astype(str))]:
 for v in sorted(cat.unique()):rows.append({'slice':name+':'+str(v),**metrics(np.asarray(cat==v))})
pd.DataFrame(rows).to_csv(O/'error_slices.csv',index=False)
ug,gi=np.unique(g,return_inverse=True)
def ci(mask,reps):
 vals=[]
 for _ in range(reps):
  w=np.bincount(rng.integers(len(ug),size=len(ug)),minlength=len(ug))[gi][mask]
  if np.dot(w,y[mask])>0:vals.append(average_precision_score(y[mask],score[mask],sample_weight=w))
 return {'ap_95pct_group_bootstrap':np.quantile(vals,[.025,.975]).tolist(),'replicates':len(vals),'resampling_unit':'cohort, otherwise scenario, otherwise background row; conditional on fitted OOF scores'}
intervals={'pooled':ci(np.ones(len(y),bool),200)}
for v in ['easy','medium','hard']:intervals[v]=ci((y==0)|a.difficulty.eq(v).to_numpy(),120)
(O/'uncertainty.json').write_text(json.dumps(intervals,indent=2)+'\n')
# Explicit top 1% errors, pooled and each fold, with annotation analysis only.
errs=[]
for k in [-1,0,1,2]:
 ix=np.arange(len(y)) if k==-1 else np.flatnonzero(fold==k);picked=np.zeros(len(y),bool);picked[ix[np.lexsort((f.TRANSACTION_HASH.to_numpy()[ix],-score[ix]))[:int(np.ceil(len(ix)*.01))]]]=True
 for typ,mask in [('FP',(y==0)&picked),('FN',(y==1)&~picked&np.isin(np.arange(len(y)),ix))]:
  d=a.loc[mask,['family','difficulty']].fillna('none').value_counts().reset_index(name='count');d['fold']=k;d['error']=typ;errs.append(d)
pd.concat(errs).to_csv(O/'review_budget_errors.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for k in [-1,0,1,2]:
 m=np.ones(len(y),bool) if k==-1 else fold==k;pr,re,_=precision_recall_curve(y[m],score[m]);ax[0].plot(re,pr,label='pooled' if k==-1 else f'fold {k}')
ax[0].set(xlabel='Recall',ylabel='Precision',title='V1 OOF precision-recall');ax[0].legend()
for lab in [0,1]:ax[1].hist(np.log10(np.maximum(score[y==lab],1e-8)),bins=45,density=True,alpha=.5,label=f'label {lab}')
ax[1].set(xlabel='log10 score',ylabel='Density',title='V1 score distributions');ax[1].legend();fig.tight_layout();fig.savefig(O/'baseline_diagnostics.png',dpi=150);plt.close(fig)
rep=json.loads((O/'baseline_reproduction.json').read_text()) if (O/'baseline_reproduction.json').exists() else None
if rep: (O/'baseline_reproduction_report.md').write_text(f'''# Verified V1 reproduction\n\nPackaged source extracted into a new directory, Python 3.12.13 virtual environment created and exact requirements installed. Command and environment are in v1_reproduction_command.json. Fresh raw-input run took {rep['runtime_seconds']:.2f} seconds. Export is byte-identical: SHA256 `{rep['original_sha256']}`. No hidden labels were accessed.\n\nIndependent OOF recalculation: mean fold AP 0.5448294042; pooled OOF AP 0.5371864804. Fold APs 0.5983379784, 0.5839404460, 0.4522097882. These are local estimates; no leaderboard result exists. Original annotations were realigned by unique transaction hash before evaluation and fixed SGKF memberships independently matched.\n\nChecks: official package checksum entries, duplicate keys, composite ledger joins, timestamps, transfer coverage and reference-value units were rechecked against released audit evidence. The reproduction uses raw data, not cached features. Source labels are read only for training target; annotations are absent from feature columns. Output schema, finite range, unique IDs, exact test ordering and archive limits independently checked again by verify_submission.py. Further machine evidence: baseline_integrity_recheck.json, validation_evidence.json.\n\nV1 is preserved and compared against its pre-experiment hash manifest at completion.\n''')
(O/'validation_audit.md').write_text(f'''# Validation audit\n\nTrack A exactly reproduces V1's three shuffled StratifiedGroupKFold folds (seed 20261024). Group priority is cohort, then scenario, then individual background row. There is no group overlap. It is an internal protocol, not an official leaderboard protocol.\n\nAll 700 customer wallets are visible across train/validation folds. The full customer graph has {nc} connected component(s), largest {report['largest_component_nodes']} nodes. Graph-component splitting would collapse to one group. Shared wallets, pairs and dense graph context make Track A weak evidence of unseen-wallet generalization even though no IDs or labels enter predictors. See validation_evidence.json for repeated-pair fractions.\n\nTrack B uses pre-locked random wallet halves. Fit labels belong only to edges whose two endpoints are in the fit half; evaluation labels belong only to edges whose two endpoints are in the other half. Cross-half edges have no supervised role. Each side still sees full authorized unlabelled graph context. Thus it tests disjoint supervised entities, not independent graphs or complete scenario separation. Both B partitions were locked before experimenting; B1 is final confirmation only. Smaller samples and truncated supervised density make B harder than the final fit; prevalence is recorded.\n\nTrack C fits the first five days of labels, evaluates the last two days. Full seven-day unlabelled features are allowed by the release; this is offline label generalization, not real-time causal performance. The previous V1 temporal diagnostic used this same cutoff, so it is not an untouched holdout. Cohorts can span the cutoff.\n\nTrack D uses annotations exclusively for error slices. Difficulty-slice AP holds all negatives fixed; positive counts and prevalence differ, so raw AP across difficulties is not a fair comparison. Group bootstrap intervals reflect group sampling conditional on fitted predictions, not training uncertainty or a calibrated confidence guarantee. Ambiguous positives have intentionally unobservable labels.\n\nNo target encoding, label propagation or pseudo labels. Static paths count adjacency, not chronological tracing. Validation-period future features are permitted offline context, but would be leakage in a streaming deployment. Repeated use of A for feature/model selection risks selection overfit; the finite experiment plan, seed confirmation and untouched B1 reduce but cannot eliminate it.\n\nRules: only supplied data; no private answers/generator access; no publication or submission in this task. The public terms require original submission material to be public by the deadline and require contributor/AI disclosure; these remain user-approved release steps. No raw-data publication is assumed permissible.\n''')
summary=pd.DataFrame(rows);d=summary.set_index('slice')
(O/'error_analysis.md').write_text('''# V1 error analysis before V2 changes\n\nMetrics are recomputed from saved OOF scores, aligned to original labels. See error_slices.csv for each fold and pooled AP, precision and recall at 0.5%, 1%, 3%, 5% and 10% review budgets, plus day, amount quintile, sender-degree and execution-leg slices. baseline_diagnostics.png contains per-fold PR curves and pooled positive/negative score distributions. review_budget_errors.csv records false positives and missed positives at the top 1% by family and difficulty, each fold and pooled.\n\n'''+d.loc[['pooled','fold0','fold1','fold2','difficulty:easy','difficulty:medium','difficulty:hard'],['n','positives','prevalence','ap','precision_top_0.01','recall_top_0.01']].to_string()+'''\n\nDifficulty slices include all 38,881 negatives and only positives from the named slice. Easy has 800 positives, medium 225, hard 225 including 25 deliberately ambiguous positives. AP is therefore prevalence-dependent. uncertainty.json gives paired-group-compatible bootstrap intervals; small hard-positive cohort count makes uncertainty broad.\n\nVerified weaknesses: medium/hard positives are ranked near the negative mass; pooled easy performance dominates total AP. A high-scoring false-positive tail exists in legitimate relay and motif families, so increasing all relay scores would harm precision. Static graph features add only about 0.42 percentage points in mean fold AP over behavioural features and fail to improve every fold; dense adjacency does not establish funds chronology. Fold 2 is materially weaker.\n\nPrioritized hypotheses: (1) simpler/deeper trees test under/overfitting without inventing signals; (2) local distinct-counterparty and direction-correct value matching may separate bursts from legitimate repeated activity; (3) strictly ordered, value-compatible multi-event chains may improve medium motifs, with explicit caveat that reference-value compatibility is not asset-level provenance. A sparse-history stress will remove 30% of unlabelled economic events and recompute features for retained targets; this is inference distribution-shift stress rather than a retrained ablation. No attempt will be made to memorize annotation identities or infer intentionally unobservable labels.\n''')
print(json.dumps(report,indent=2),flush=True)

with (O/'validation_audit.md').open('a') as stream:stream.write('\nB1 independence caveat: its metrics can remain uninspected, but its labels already participated in Track A development. It is not an independent final labelled test. No untouched labelled dataset is available.\n')
