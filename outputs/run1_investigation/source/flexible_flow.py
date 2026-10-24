"""Label-free cumulative economic-flow compatibility, not fund provenance."""
from common import ROOT,O
import sys
sys.path.insert(0,str(ROOT/'solution'))
from payments import load_payments,load_rates,XRP
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq

def prefix_match(values,ages,reference,max_terms=8):
 if not len(values):return [10.,0.,24.,0.,0.]
 n=min(len(values),max_terms);totals=np.cumsum(values[:n]);errors=np.abs(totals-reference)/reference;j=int(np.argmin(errors));return [min(10,float(errors[j])),float(j+1),float(ages[j]/3600),float(np.mean(errors<=.05)),float(np.mean(errors<=.2))]

def build(directory):
 d=Path(directory);p=pd.DataFrame(load_payments(d)).sort_values(['TIMESTAMP','TRANSACTION_INDEX','TRANSACTION_HASH']).reset_index(drop=True);t=p.TIMESTAMP.to_numpy(np.int64);v=p.VALUE.to_numpy(float);rates=load_rates(d);tr=pq.read_table(d/'transfers.parquet').to_pandas().sort_values(['TRANSACTION_HASH','TRANSFER_INDEX']);first=tr.drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH').loc[p.TRANSACTION_HASH];asset=np.where(first.TOKEN_ADDRESS.eq(XRP),'XRP',first.TOKEN_ADDRESS+':'+first.SYMBOL);cost=first.VALUE.to_numpy(float)*np.array([rates[(int(tt)//3600*3600,a)]/1e6 for tt,a in zip(t,asset)]);nodes=pd.Index(sorted(set(p.FROM_ADDRESS)|set(p.TO_ADDRESS)));s=nodes.get_indexer(p.FROM_ADDRESS);r=nodes.get_indexer(p.TO_ADDRESS);n=len(p);f={}
 for role,targets,context,values,ref,before in [('sender_prior_in',s,r,v,cost,True),('receiver_future_out',r,s,cost,v,False)]:
  for w in [3600,21600,86400]:
   z=np.tile([10.,0.,24.,0.,0.],(n,1))
   for node in range(len(nodes)):
    ix=np.flatnonzero(context==node);qi=np.flatnonzero(targets==node);tt=t[ix]
    lo=np.searchsorted(tt,t[qi]-w if before else t[qi],'left' if before else 'right');hi=np.searchsorted(tt,t[qi] if before else t[qi]+w,'left' if before else 'right')
    for j,l,h in zip(qi,lo,hi):
     ni=ix[l:h][::-1] if before else ix[l:h];z[j]=prefix_match(values[ni],abs(t[ni]-t[j]),ref[j])
   for k,name in enumerate(['min_prefix_error','best_prefix_terms','best_prefix_lag_hours','prefix_match05_fraction','prefix_match20_fraction']):f[f'flex_{role}_{w}_{name}']=z[:,k]
 for w in [3600,21600,86400]:
  a=f[f'flex_sender_prior_in_{w}_min_prefix_error'];b=f[f'flex_receiver_future_out_{w}_min_prefix_error'];f[f'flex_{w}_two_sided_error']=np.maximum(a,b)
 z=pd.DataFrame(f,dtype=np.float32);assert np.isfinite(z.to_numpy()).all();z.insert(0,'TRANSACTION_HASH',p.TRANSACTION_HASH);return z
if __name__=='__main__':
 assert prefix_match(np.array([50,50]),np.array([10,20]),100)[:2]==[0.,2.]
 assert prefix_match(np.array([]),np.array([]),100)[1]==0
 assert prefix_match(np.array([100,100]),np.array([10,20]),100)[1]==1
 z=build(ROOT/'data/training/train');z.to_parquet(O/'flexible_train.parquet',index=False);print('Flexible flow features',len(z),z.shape[1]-1,flush=True)
