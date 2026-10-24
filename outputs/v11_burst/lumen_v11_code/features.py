"""Label-free, full-window customer-payment features for Strategy3 v1.

Never reads labels.csv. Raw identifiers are used for alignment/grouping only.
Window summaries are strict before/after in timestamp seconds; static graph
features are deliberately separate from chronological matching features.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from payments import load_payments, load_rates, XRP
from starter_features import make_features as starter_features

WINDOWS = (300, 3600, 21600, 86400, 259200)


def safe_ratio(a, b):
    return np.clip(np.asarray(a)/(np.asarray(b)+1e-8), 0, 1000)


def build_features(directory):
    started = time.perf_counter()
    directory = Path(directory)
    p = pd.DataFrame(load_payments(directory))
    p = p.sort_values(["TIMESTAMP", "TRANSACTION_INDEX", "TRANSACTION_HASH"]).reset_index(drop=True)
    n = len(p)
    t = p.TIMESTAMP.to_numpy(dtype=np.int64)
    vin = p.VALUE.to_numpy(dtype=np.float64)
    tx = pq.read_table(directory/"transactions.parquet").to_pandas().set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH]
    tr = pq.read_table(directory/"transfers.parquet").to_pandas()
    tr = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    first = tr.drop_duplicates("TRANSACTION_HASH", keep="first").set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH]
    last = tr.drop_duplicates("TRANSACTION_HASH", keep="last").set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH]
    assert first.FROM_ADDRESS.to_numpy().tolist() == p.FROM_ADDRESS.tolist()
    assert last.TO_ADDRESS.to_numpy().tolist() == p.TO_ADDRESS.tolist()
    rates = load_rates(directory)
    source_asset = np.where(first.TOKEN_ADDRESS.eq(XRP), "XRP", first.TOKEN_ADDRESS+":"+first.SYMBOL)
    cost = first.VALUE.to_numpy(dtype=float) * np.array([rates[(int(ts)//3600*3600, a)]/1e6 for ts, a in zip(t, source_asset)])
    assert np.isfinite(cost).all() and (cost > 0).all()
    # Customer nodes only: conversion liquidity-provider execution legs stay separate.
    nodes = pd.Index(sorted(set(p.FROM_ADDRESS)|set(p.TO_ADDRESS)))
    s = nodes.get_indexer(p.FROM_ADDRESS);r = nodes.get_indexer(p.TO_ADDRESS);nn=len(nodes)
    outgoing=[np.flatnonzero(s==a) for a in range(nn)]
    incoming=[np.flatnonzero(r==a) for a in range(nn)]
    values_out=cost;values_in=vin
    f={}
    basic=starter_features(p).set_index("TRANSACTION_HASH").loc[p.TRANSACTION_HASH]
    for c in basic: f['starter_'+c]=basic[c].to_numpy()
    f['log_paid_usd']=np.log1p(cost)
    f['log_delivered_usd']=np.log1p(vin)
    # No date/hour/sequence/hash/issuer ID predictors: avoid split and generator shortcuts.
    f['execution_legs']=p.TRANSACTION_HASH.map(tr.groupby('TRANSACTION_HASH').size()).to_numpy()
    f['valuation_cost_ratio']=safe_ratio(cost,vin)
    f['self_payment']=(s==r).astype(float)
    f['left_exposure_hours']=(t-t.min())/3600
    f['right_exposure_hours']=(t.max()-t)/3600
    full={}
    for side,groups,vals,partners in [('in',incoming,values_in,s),('out',outgoing,values_out,r)]:
        stats={k:np.zeros(nn) for k in ['count','sum','mean','std','median','iqr','unique','span','gap_mean','gap_std','entropy','largest_share']}
        for a,ix in enumerate(groups):
            if not len(ix):continue
            vv=vals[ix];tt=t[ix];_,freq=np.unique(partners[ix],return_counts=True);q=freq/freq.sum()
            stats['count'][a]=len(ix);stats['sum'][a]=vv.sum();stats['mean'][a]=vv.mean()
            stats['std'][a]=vv.std();stats['median'][a]=np.median(vv);stats['iqr'][a]=np.quantile(vv,.75)-np.quantile(vv,.25)
            stats['unique'][a]=len(freq);stats['span'][a]=(tt[-1]-tt[0])/3600
            stats['entropy'][a]=-np.sum(q*np.log(q));stats['largest_share'][a]=q.max()
            if len(ix)>1:stats['gap_mean'][a]=np.diff(tt).mean()/3600;stats['gap_std'][a]=np.diff(tt).std()/3600
        full[side]=stats
        stats['cv']=safe_ratio(stats['std'],stats['mean'])
        stats['repeat_fraction']=1-safe_ratio(stats['unique'],stats['count'])
        for role,targets in [('sender',s),('receiver',r)]:
            for k,v in stats.items():f[f'{role}_{side}_full_{k}']=v[targets]
            f[f'{role}_{side}_amount_over_median']=safe_ratio(vin,stats['median'][targets])
    for role,targets in [('sender',s),('receiver',r)]:
        f[f'{role}_full_out_in_count']=safe_ratio(full['out']['count'][targets],full['in']['count'][targets])
        f[f'{role}_full_out_in_value']=safe_ratio(full['out']['sum'][targets],full['in']['sum'][targets])
        f[f'{role}_full_net_fraction']=(full['in']['sum'][targets]-full['out']['sum'][targets])/(full['in']['sum'][targets]+full['out']['sum'][targets]+1e-8)
    # All target events query each endpoint's complete unlabelled split context.
    window={}
    nearest_records={}
    for role,targets in [('sender',s),('receiver',r)]:
        target_by_node=[np.flatnonzero(targets==a) for a in range(nn)]
        for side,groups,vals in [('in',incoming,values_in),('out',outgoing,values_out)]:
            near={k:np.zeros(n) for k in ['prev_hours','next_hours','prev_relative_error','next_relative_error']}
            for direction in ['before','after']:
                for w in WINDOWS:
                    count=np.zeros(n);total=np.zeros(n);stdev=np.zeros(n);unique=np.zeros(n)
                    for a,ix in enumerate(groups):
                        qi=target_by_node[a]
                        if not len(qi) or not len(ix):continue
                        tt=t[ix];vv=vals[ix]
                        if direction=='before':lo=np.searchsorted(tt,t[qi]-w,'left');hi=np.searchsorted(tt,t[qi],'left')
                        else:lo=np.searchsorted(tt,t[qi],'right');hi=np.searchsorted(tt,t[qi]+w,'right')
                        cs=np.r_[0.,np.cumsum(vv)];cs2=np.r_[0.,np.cumsum(vv*vv)]
                        cc=hi-lo;ss=cs[hi]-cs[lo]
                        count[qi]=cc;total[qi]=ss
                        stdev[qi]=np.sqrt(np.maximum(0,(cs2[hi]-cs2[lo])/np.maximum(1,cc)-(ss/np.maximum(1,cc))**2))
                    prefix=f'{role}_{side}_{direction}_{w}s'
                    mean=total/np.maximum(count,1)
                    f[prefix+'_count']=count;f[prefix+'_log_sum']=np.log1p(total)
                    f[prefix+'_cv']=safe_ratio(stdev,mean)
                    f[prefix+'_sum_over_payment']=safe_ratio(total,vin)
                    window[(role,side,direction,w)]=(count,total)
            # Nearest timing and nearest-size compatibility across bounded candidate intervals.
            for a,ix in enumerate(groups):
                qi=target_by_node[a]
                if not len(qi):continue
                near['prev_hours'][qi]=168;near['next_hours'][qi]=168
                near['prev_relative_error'][qi]=10;near['next_relative_error'][qi]=10
                if not len(ix):continue
                tt=t[ix];vv=vals[ix];prev=np.searchsorted(tt,t[qi],'left')-1;nxt=np.searchsorted(tt,t[qi],'right')
                ok=prev>=0;j=qi[ok];k=prev[ok]
                near['prev_hours'][j]=np.minimum(168,(t[j]-tt[k])/3600)
                near['prev_relative_error'][j]=np.minimum(10,np.abs(vv[k]-vin[j])/(vin[j]+1e-8))
                ok=nxt<len(ix);j=qi[ok];k=nxt[ok]
                near['next_hours'][j]=np.minimum(168,(tt[k]-t[j])/3600)
                near['next_relative_error'][j]=np.minimum(10,np.abs(vv[k]-vin[j])/(vin[j]+1e-8))
            for k,v in near.items():f[f'{role}_{side}_{k}']=v
            # Match a payment to other *economic* events; no label-derived risk propagation.
            for w in [3600,21600,86400]:
                for direction in ['before','after']:
                    errors=np.full(n,10.);bestlag=np.full(n,168.);capped=np.zeros(n)
                    for a,ix in enumerate(groups):
                        qi=target_by_node[a]
                        if not len(qi) or not len(ix):continue
                        tt=t[ix];vv=vals[ix]
                        if direction=='before':lo=np.searchsorted(tt,t[qi]-w,'left');hi=np.searchsorted(tt,t[qi],'left')
                        else:lo=np.searchsorted(tt,t[qi],'right');hi=np.searchsorted(tt,t[qi]+w,'right')
                        for j,l,h in zip(qi,lo,hi):
                            if h<=l:continue
                            if h-l>256:capped[j]=1;l=max(l,h-256) if direction=='before' else l;h=min(h,l+256)
                            e=np.abs(vv[l:h]-vin[j])/(vin[j]+1e-8);best=int(np.argmin(e))
                            errors[j]=min(10,e[best]);bestlag[j]=abs(tt[l+best]-t[j])/3600
                    prefix=f'{role}_{side}_{direction}_match_{w}s'
                    f[prefix+'_rel_error']=errors;f[prefix+'_lag_hours']=bestlag;f[prefix+'_capped']=capped
    for w in WINDOWS:
        for role in ['sender','receiver']:
            for direction in ['before','after']:
                ic,iv=window[(role,'in',direction,w)];oc,ov=window[(role,'out',direction,w)]
                f[f'{role}_{direction}_{w}s_out_in_count']=safe_ratio(oc,ic)
                f[f'{role}_{direction}_{w}s_flow_imbalance']=(iv-ov)/(iv+ov+1e-8)
        si,sv=window[('sender','in','before',w)];ro,rv=window[('receiver','out','after',w)]
        f[f'flow_{w}s_prior_in_error']=np.minimum(10,np.abs(sv-cost)/(cost+1e-8))
        f[f'flow_{w}s_later_out_error']=np.minimum(10,np.abs(rv-vin)/(vin+1e-8))
        f[f'flow_{w}s_two_sided_count']=np.minimum(si,ro)
    # Repeated pair statistics and chronological recurrence.
    pair_codes=s*nn+r
    pair_stats={k:np.zeros(n) for k in ['count','span_hours','value_cv','value_median','prev_hours','next_hours']}
    pair_aggregate={}
    for code in np.unique(pair_codes):
        ix=np.flatnonzero(pair_codes==code);vv=vin[ix];tt=t[ix]
        pair_stats['count'][ix]=len(ix);pair_stats['span_hours'][ix]=(tt[-1]-tt[0])/3600
        pair_stats['value_cv'][ix]=vv.std()/(vv.mean()+1e-8);pair_stats['value_median'][ix]=np.median(vv)
        pair_stats['prev_hours'][ix]=np.r_[168,np.diff(tt)/3600]
        pair_stats['next_hours'][ix]=np.r_[np.diff(tt)/3600,168]
    for k,v in pair_stats.items():f['pair_'+k]=v
    f['pair_amount_relative']=safe_ratio(vin,pair_stats['value_median'])
    # Distribution of each pair's multi-hour flow role over its complete history.
    for w in [21600,86400]:
        for role,side,direction in [('sender','in','before'),('receiver','out','after')]:
            count,total=window[(role,side,direction,w)]
            ratio=np.minimum(10,total/(vin+1e-8));med=np.zeros(n);cv=np.zeros(n)
            for code in np.unique(pair_codes):
                ix=np.flatnonzero(pair_codes==code);q=ratio[ix];med[ix]=np.median(q);cv[ix]=q.std()/(q.mean()+1e-8)
            f[f'pair_{role}_{w}s_flow_ratio_median']=med;f[f'pair_{role}_{w}s_flow_ratio_cv']=cv
    adj=csr_matrix((np.ones(n),(s,r)),shape=(nn,nn));binary=adj.copy();binary.data[:]=1
    two=binary@binary;three=two@binary
    f['graph_pair_count']=np.asarray(adj[s,r]).ravel()
    f['graph_reverse_count']=np.asarray(adj[r,s]).ravel()
    f['graph_twohop_parallel']=np.asarray(two[s,r]).ravel()
    f['graph_threehop_parallel']=np.asarray(three[s,r]).ravel()
    f['graph_cycle3_static']=np.asarray(two[r,s]).ravel()
    f['graph_cycle4_static']=np.asarray(three[r,s]).ravel()
    undirected=((binary+binary.T)>0).astype(float)
    for stat in ['count','unique','cv','entropy']:
        for side in ['in','out']:
            z=full[side][stat]
            for direction,A in [('out',binary),('in',binary.T),('both',undirected)]:
                degree=np.asarray(A.sum(axis=1)).ravel()
                mean=np.asarray(A@z).ravel()/np.maximum(degree,1)
                for role,targets in [('sender',s),('receiver',r)]:f[f'{role}_neighbor_{direction}_{side}_{stat}_mean']=mean[targets]
    result=pd.DataFrame(f).replace([np.inf,-np.inf],np.nan).fillna(0).astype(np.float32)
    assert np.isfinite(result.to_numpy()).all()
    result.insert(0,'TRANSACTION_HASH',p.TRANSACTION_HASH.to_numpy())
    assert result.TRANSACTION_HASH.is_unique and len(result)==n
    print(f'Features {directory}: {n:,} rows, {len(f)} numeric columns, {time.perf_counter()-started:.1f}s',flush=True)
    return result
