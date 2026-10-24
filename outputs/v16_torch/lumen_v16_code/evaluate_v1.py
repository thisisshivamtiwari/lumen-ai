"""Evaluate two predetermined label-free feature blocks on fixed grouped folds."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedGroupKFold
from baseline import metrics, SEED


PARAMS=dict(iterations=1200,depth=6,learning_rate=.035,l2_leaf_reg=10,
            loss_function='Logloss',random_seed=SEED,thread_count=4,
            verbose=False,allow_writing_files=False,rsm=.85)


def groups_for(labels):
    return np.where(labels.cohort_id.notna(),'cohort:'+labels.cohort_id.fillna(''),
        np.where(labels.scenario_id.notna(),'scenario:'+labels.scenario_id.fillna(''),'row:'+labels.TRANSACTION_HASH))


def columns_for(all_columns, name):
    if name=='behaviour':return [c for c in all_columns if not c.startswith('graph_') and '_neighbor_' not in c]
    return list(all_columns)


def main():
    start=time.perf_counter();out=Path('outputs/strategy3_v1');out.mkdir(exist_ok=True)
    f=pd.read_parquet('outputs/features_v1/train.parquet')
    labels=pd.read_csv('data/training/train/labels.csv').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index()
    X=f.drop(columns='TRANSACTION_HASH');y=labels.is_suspicious.to_numpy();ids=f.TRANSACTION_HASH.to_numpy();groups=groups_for(labels)
    splits=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED).split(X,y,groups))
    report={'params':PARAMS,'context':'Full authorised label-free split window; no label-derived features.', 'candidates':{}}
    for name in ['behaviour','graph']:
        cols=columns_for(X.columns,name);oof=np.full(len(y),np.nan);fold=np.full(len(y),-1);results=[]
        print('Candidate',name,'columns',len(cols),flush=True)
        for i,(a,b) in enumerate(splits):
            assert not set(groups[a])&set(groups[b])
            model=CatBoostClassifier(**PARAMS);model.fit(X[cols].iloc[a],y[a])
            oof[b]=model.predict_proba(X[cols].iloc[b])[:,1];fold[b]=i
            m=metrics(y[b],oof[b],ids[b]);results.append(m)
            print(name,'fold',i,'AP score',round(m['score'],3),flush=True)
        assert np.isfinite(oof).all()
        slices={}
        for field,values in [('difficulty',['easy','medium','hard']),('family',sorted(labels.loc[labels.is_suspicious.eq(1),'family'].unique()))]:
            for value in values:
                mask=(y==0)|labels[field].eq(value).to_numpy()
                slices[f'{field}:{value}']=metrics(y[mask],oof[mask],ids[mask])
        ann=labels.copy();ann['fold']=fold;ann['score']=oof;ann.to_csv(out/f'{name}_oof.csv',index=False)
        report['candidates'][name]={'columns':cols,'folds':results,'mean_ap':float(np.mean([m['ap'] for m in results])),
            'pooled':metrics(y,oof,ids),'slices_with_common_negative_background':slices}
        (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    chosen=max(report['candidates'],key=lambda k:report['candidates'][k]['mean_ap'])
    report['chosen']=chosen;report['elapsed_seconds']=time.perf_counter()-start
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    config={'model':'CatBoostClassifier','params':PARAMS,'feature_set':chosen,'features':report['candidates'][chosen]['columns']}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Selected',chosen,'mean grouped AP score',report['candidates'][chosen]['mean_ap']*100,flush=True)


if __name__=='__main__':main()
