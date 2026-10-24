from pathlib import Path
import os,json
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';table=pd.read_csv(O/'model_comparison.csv');labels=pd.read_csv(ROOT/'data/training/train/labels.csv').set_index('TRANSACTION_HASH');rows=[];fig,ax=plt.subplots(1,2,figsize=(11,4))
for _,z in table.iterrows():
 kind=z.experiment;path=ROOT/'outputs/strategy3_v1/graph_oof.csv' if kind=='v1' else O/'runs'/f'{kind}_oof.csv';q=pd.read_csv(path);y=labels.loc[q.TRANSACTION_HASH,'is_suspicious'].to_numpy();score=q.score.to_numpy();order=np.lexsort((q.TRANSACTION_HASH.to_numpy(),-score));p,r,_=precision_recall_curve(y,score);ax[0].plot(r,p,label=kind,linewidth=2 if kind=='v1' else 1,alpha=.85)
 for fraction in [.005,.01,.03,.05,.1]:
  k=int(np.ceil(len(q)*fraction));tp=int(y[order[:k]].sum());rows.append({'experiment':kind,'fraction':fraction,'review_count':k,'positives':int(y.sum()),'true_positives':tp,'precision':tp/k,'recall':tp/y.sum()})
for i,name in enumerate(['v1','depth8','chrono','chrono_no_static']):
 z=table[table.experiment.eq(name)].iloc[0];ax[1].plot([0,1,2],[100*z.fold0,100*z.fold1,100*z.fold2],marker='o',label=name)
ax[0].set(xlabel='Recall',ylabel='Precision',title='Pooled OOF ranking comparison');ax[0].legend(fontsize=8);ax[1].set(xlabel='Fixed Track A fold',ylabel='Average precision (%)',title='Fold stability');ax[1].set_xticks([0,1,2]);ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(O/'candidate_diagnostics.png',dpi=150);plt.close(fig);pd.DataFrame(rows).to_csv(O/'candidate_review_budgets.csv',index=False)
print('Candidate diagnostics saved')
