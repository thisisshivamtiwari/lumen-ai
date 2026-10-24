from common import *
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
f,a,X,y,fold,s=load();category=cats(a);g=groups_for(a);ug,gi=np.unique(g,return_inverse=True);rng=np.random.default_rng(20261024);d=pd.read_csv(O/'category_metrics.csv');low=[];high=[];nreps=[]
for _,row in d.iterrows():
 m=np.ones(len(y),bool) if row.fold==-1 else fold==row.fold;mask=m&((y==0)|(category==row.category));vs=[];reps=200 if row.fold==-1 else 60
 for _ in range(reps):
  w=np.bincount(rng.integers(len(ug),size=len(ug)),minlength=len(ug))[gi][mask]
  if np.dot(w,y[mask])>0:vs.append(average_precision_score(y[mask],s[mask],sample_weight=w))
 l,h=np.quantile(vs,[.025,.975]);low.append(l);high.append(h);nreps.append(len(vs))
d['ap_group_bootstrap_low']=low;d['ap_group_bootstrap_high']=high;d['bootstrap_replicates']=nreps;d.to_csv(O/'category_metrics.csv',index=False)
p=pd.read_csv(O/'shap_group_contributions.csv');p['block']=np.where(p.feature.str.contains('_300s')|p.feature.isin(['starter_receiver_burst_10m','starter_sender_burst_10m']),'short_burst_flow',np.where(p.feature.str.startswith('pair_')|p.feature.eq('graph_pair_count'),'pair_history','other'));b=p.groupby(['fold','group','block']).agg(mean_logit_contribution=('mean_shap_log_odds','sum'),rows=('rows','first')).reset_index();b.to_csv(O/'attribution_blocks.csv',index=False)
order=np.lexsort((f.TRANSACTION_HASH,-s));top=np.zeros(len(y),bool);top[order[:402]]=True;rules={'one_off_pair':X.pair_count.eq(1),'receiver_burst_gt1':X.starter_receiver_burst_10m.gt(1),'prior_flow_error_le0.1':X.flow_300s_prior_in_error.le(.1),'later_flow_error_le0.1':X.flow_300s_later_out_error.le(.1)};records=[]
for k in [-1,0,1,2]:
 fm=np.ones(len(y),bool) if k==-1 else fold==k
 for group,mask in [('high_FP',(y==0)&top),('medium',category=='medium'),('easy',category=='easy'),('legitimate',y==0)]:
  mask=mask&fm
  for rule,flag in rules.items():records.append({'fold':k,'group':group,'rule':rule,'rows':int(mask.sum()),'count':int((mask&flag).sum()),'fraction':float(flag[mask].mean())})
pd.DataFrame(records).to_csv(O/'observable_pattern_rates.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for cat in ['legitimate','easy','medium','observable-hard','ambiguous']:
 z=s[category==cat];ax[0].hist(np.log10(np.maximum(z,1e-8)),bins=np.linspace(-5,0,50),histtype='step',density=True,label=cat)
ax[0].axvline(np.log10(s[order[401]]),color='black',linestyle='--',label='global top1 threshold');ax[0].set(xlabel='log10 OOF score',ylabel='Density',title='V1 category distributions');ax[0].legend(fontsize=7)
for group in ['medium','high_FP']:
 z=b[b.group.eq(group)];values=z.groupby('block').apply(lambda q:np.average(q.mean_logit_contribution,weights=q.rows));x=np.arange(3)+(0 if group=='medium' else .35);ax[1].bar(x,values.reindex(['pair_history','short_burst_flow','other']),width=.35,label=group)
ax[1].set_xticks(np.arange(3)+.175,['Pair history','Short flow/burst','Other']);ax[1].set(ylabel='Mean SHAP log-odds contribution',title='Held-out model attribution');ax[1].legend();fig.tight_layout();fig.savefig(O/'diagnostic.png',dpi=160);plt.close(fig)
print('Category uncertainty and observable-pattern evidence saved')
