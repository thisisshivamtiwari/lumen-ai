from common import *
from catboost import CatBoostClassifier,Pool
f,a,X,y,fold,s=load();category=cats(a);ids=f.TRANSACTION_HASH.to_numpy();order=np.lexsort((ids,-s));top=np.zeros(len(y),bool);top[order[:int(np.ceil(len(y)*.01))]]=True;pos=np.flatnonzero(y==1);bottom=np.zeros(len(y),bool);bottom[pos[np.argsort(s[pos])[:125]]]=True;rng=np.random.default_rng(20261024)
# Exhaustive medium/high-FP/bottom-positive rows, plus fixed label-stratified reference samples.
sample=np.unique(np.r_[np.flatnonzero(category=='medium'),np.flatnonzero((y==0)&top),np.flatnonzero(bottom),rng.choice(np.flatnonzero(category=='easy'),200,replace=False),rng.choice(np.flatnonzero(y==0),400,replace=False)])
rows=[];importance=[];checks=[]
for k in range(3):
 m=CatBoostClassifier(**PARAMS);tr=fold!=k;va=fold==k;m.fit(X.loc[tr],y[tr]);p=m.predict_proba(X.loc[va])[:,1];delta=float(np.max(abs(p-s[va])));assert delta<1e-12;checks.append({'fold':k,'max_original_oof_difference':delta});m.save_model(str(O/'runs'/f'v1_fold{k}.cbm'))
 qi=sample[fold[sample]==k];sh=m.get_feature_importance(Pool(X.iloc[qi]),type='ShapValues',thread_count=4);raw=m.predict(X.iloc[qi],prediction_type='RawFormulaVal');assert np.allclose(sh[:,:-1].sum(axis=1)+sh[:,-1],raw,atol=1e-10)
 groups={'medium':category[qi]=='medium','easy_reference':category[qi]=='easy','legitimate_reference':y[qi]==0,'high_FP':(y[qi]==0)&top[qi],'lowest_125_positive':bottom[qi]}
 for group,mask in groups.items():
  if not mask.any():continue
  for j,col in enumerate(X):rows.append({'fold':k,'group':group,'rows':int(mask.sum()),'feature':col,'mean_shap_log_odds':float(sh[mask,j].mean()),'mean_abs_shap':float(abs(sh[mask,j]).mean()),'expected_log_odds':float(sh[mask,-1].mean())})
 for col,weight in zip(X,m.feature_importances_):importance.append({'fold':k,'feature':col,'importance':float(weight)})
 print('Explained fold',k,'rows',len(qi),'exact OOF parity',delta,flush=True)
pd.DataFrame(rows).to_csv(O/'shap_group_contributions.csv',index=False);pd.DataFrame(importance).to_csv(O/'feature_importance.csv',index=False);(O/'shap_checks.json').write_text(json.dumps({'checks':checks,'SHAP_sum_to_raw_logit':'PASS','scope':'Fold-held-out rows only; explanatory attribution of fitted model, not causal evidence. Legitimate reference also contains all top-1% FPs; other legitimate rows are a fixed 400-row sample. All medium cases included.','sample_rows':len(sample)},indent=2)+'\n')
