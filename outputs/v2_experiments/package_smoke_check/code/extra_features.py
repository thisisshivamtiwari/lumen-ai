"""Label-free chronology and local turnover; one customer event per Payment.
Compatible reference-valued events are candidates, never proof of asset provenance.
No label, annotation, hash value or raw entity ID is returned as a predictor.
"""
from pathlib import Path
import time
import numpy as np,pandas as pd,pyarrow.parquet as pq
from payments import load_payments,load_rates,XRP

def build_extra(directory):
 start=time.perf_counter();d=Path(directory);p=pd.DataFrame(load_payments(d)).sort_values(['TIMESTAMP','TRANSACTION_INDEX','TRANSACTION_HASH']).reset_index(drop=True)
 t=p.TIMESTAMP.to_numpy(dtype=np.int64);v=p.VALUE.to_numpy(dtype=float);n=len(p)
 tr=pq.read_table(d/'transfers.parquet').to_pandas().sort_values(['TRANSACTION_HASH','TRANSFER_INDEX']);first=tr.drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH').loc[p.TRANSACTION_HASH];rates=load_rates(d)
 asset=np.where(first.TOKEN_ADDRESS.eq(XRP),'XRP',first.TOKEN_ADDRESS+':'+first.SYMBOL);cost=first.VALUE.to_numpy(dtype=float)*np.array([rates[(int(ts)//3600*3600,a)]/1e6 for ts,a in zip(t,asset)])
 nodes=pd.Index(sorted(set(p.FROM_ADDRESS)|set(p.TO_ADDRESS)));s=nodes.get_indexer(p.FROM_ADDRESS);r=nodes.get_indexer(p.TO_ADDRESS)
 incoming=[np.flatnonzero(r==i) for i in range(len(nodes))];outgoing=[np.flatnonzero(s==i) for i in range(len(nodes))];f={}
 # Short-window diversity absent from V1. Strict seconds exclude simultaneous events.
 for role,target in [('sender',s),('receiver',r)]:
  for side,groups,partners,vals in [('in',incoming,s,v),('out',outgoing,r,cost)]:
   for window in [3600,21600,86400]:
    for before in [True,False]:
     fields={k:np.zeros(n) for k in ['unique','repeat_fraction','largest_partner_share','median_over_current','max_share','current_partner_fraction']}
     for node,ix in enumerate(groups):
      qi=np.flatnonzero(target==node);tt=t[ix]
      if not len(ix):continue
      lo=np.searchsorted(tt,t[qi]-window if before else t[qi],side='left' if before else 'right');hi=np.searchsorted(tt,t[qi] if before else t[qi]+window,side='left' if before else 'right')
      for j,l,h in zip(qi,lo,hi):
       if h<=l:continue
       ni=ix[l:h];_,counts=np.unique(partners[ni],return_counts=True);vv=vals[ni];den=cost[j] if side=='out' else v[j]
       fields['unique'][j]=len(counts);fields['repeat_fraction'][j]=1-len(counts)/(h-l);fields['largest_partner_share'][j]=counts.max()/(h-l);fields['median_over_current'][j]=min(1000,np.median(vv)/den);fields['max_share'][j]=vv.max()/vv.sum();partner=r[j] if role=='sender' else s[j];fields['current_partner_fraction'][j]=np.mean(partners[ni]==partner)
     for k,z in fields.items():f[f'local_{role}_{side}_{"before" if before else "after"}_{window}_{k}']=z
 # Chronological event DAG. At each economic wallet, compare actual outgoing cost
 # with previous delivered value. No liquidity-provider leg is a customer hop.
 for window in [3600,21600]:
  for tolerance in [.01,.05,.15]:
   pred=[[] for _ in range(n)];succ=[[] for _ in range(n)];besterr=np.full(n,10.);bestlag=np.full(n,168.)
   for node,ins in enumerate(incoming):
    outs=outgoing[node];tt=t[ins]
    if not len(ins) or not len(outs):continue
    lo=np.searchsorted(tt,t[outs]-window,'left');hi=np.searchsorted(tt,t[outs],'left')
    for j,l,h in zip(outs,lo,hi):
     if h<=l:continue
     cand=ins[l:h];err=abs(v[cand]-cost[j])/max(cost[j],1e-8);b=np.argmin(err);besterr[j]=min(10,err[b]);bestlag[j]=(t[j]-t[cand[b]])/3600
     good=cand[err<=tolerance];pred[j]=good.tolist()
     for i in good:succ[i].append(int(j))
   back=np.zeros(n);forward=np.zeros(n);rev=np.zeros(n)
   for j in range(n):
    if len(pred[j]):back[j]=min(8,1+back[pred[j]].max());rev[j]=np.mean(s[np.asarray(pred[j])]==r[j])
   for j in range(n-1,-1,-1):
    if len(succ[j]):forward[j]=min(8,1+forward[succ[j]].max())
   pre=f'chrono_{window}_{tolerance:g}'
   f[pre+'_predecessors']=np.array([len(z) for z in pred]);f[pre+'_successors']=np.array([len(z) for z in succ]);f[pre+'_back_depth']=back;f[pre+'_forward_depth']=forward;f[pre+'_span_depth']=np.minimum(12,back+forward);f[pre+'_two_sided']=np.minimum(back,forward);f[pre+'_return_fraction']=rev
  f[f'chrono_{window}_best_prior_error']=besterr;f[f'chrono_{window}_best_prior_lag']=bestlag
 result=pd.DataFrame(f,dtype=np.float32);assert np.isfinite(result.to_numpy()).all();result.insert(0,'TRANSACTION_HASH',p.TRANSACTION_HASH.to_numpy());print(f'extra features {len(f)} rows {n} seconds {time.perf_counter()-start:.2f}',flush=True);return result

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--data',required=True);a.add_argument('--output',required=True);args=a.parse_args();build_extra(args.data).to_parquet(args.output,index=False)
