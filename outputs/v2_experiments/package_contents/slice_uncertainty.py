from pathlib import Path
import os,json
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
from evaluate_v1 import groups_for
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';f=pd.read_parquet(ROOT/'outputs/features_v1/train.parquet');a=pd.read_csv(ROOT/'data/training/train/labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index();y=a.is_suspicious.to_numpy();oo=pd.read_csv(ROOT/'outputs/strategy3_v1/graph_oof.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH];s=oo.score.to_numpy();fold=oo.fold.to_numpy();g=groups_for(a);ug,gi=np.unique(g,return_inverse=True);rng=np.random.default_rng(20261024)
# Audit-only row labels for the exact same diagnostic slices created in audit.py.
from payments import load_payments
p=pd.DataFrame(load_payments(ROOT/'data/training/train')).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH]
fields={'day':pd.to_datetime(p.TIMESTAMP.astype('int64'),unit='s',utc=True).dt.strftime('%Y-%m-%d').to_numpy(),'amount_quintile':pd.qcut(f.log_delivered_usd,5,duplicates='drop').astype(str).to_numpy(),'sender_degree_quartile':pd.qcut(f.sender_out_full_unique,4,duplicates='drop').astype(str).to_numpy(),'execution_legs':f.execution_legs.astype(str).to_numpy()}
d=pd.read_csv(O/'error_slices.csv');lo=[];hi=[]
for name in d.slice:
 if name=='pooled':m=np.ones(len(y),bool)
 elif name.startswith('fold'):m=fold==int(name[-1])
 else:
  field,value=name.split(':',1);m=((y==0)|a[field].eq(value).to_numpy()) if field in ['family','difficulty'] else fields[field]==value
 vals=[]
 for _ in range(100):
  w=np.bincount(rng.integers(len(ug),size=len(ug)),minlength=len(ug))[gi][m]
  if np.dot(w,y[m])>0:vals.append(average_precision_score(y[m],s[m],sample_weight=w))
 if vals:l,h=np.quantile(vals,[.025,.975])
 else:l=h=np.nan
 lo.append(l);hi.append(h)
d['ap_group_bootstrap_low']=lo;d['ap_group_bootstrap_high']=hi;d['bootstrap_replicates_requested']=100;d.to_csv(O/'error_slices.csv',index=False)
rows=[]
for k in [-1,0,1,2]:
 for label in [0,1]:
  m=(y==label)&((fold==k) if k>=0 else True);z=s[m];rows.append({'fold':k,'label':label,'n':len(z),'score_mean':float(z.mean()),'score_q05':float(np.quantile(z,.05)),'score_median':float(np.median(z)),'score_q95':float(np.quantile(z,.95))})
pd.DataFrame(rows).to_csv(O/'score_distributions.csv',index=False)
(O/'error_analysis.md').write_text((O/'error_analysis.md').read_text()+'\nAll error_slices.csv rows also include 100-replicate group-bootstrap AP intervals; these are approximate conditional intervals. score_distributions.csv gives each fold and pooled positive/negative score quantiles.\n')
print('All diagnostic slice uncertainty complete',flush=True)
