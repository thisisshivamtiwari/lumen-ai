from common import *
import hashlib
from sklearn.metrics import roc_auc_score
f,a,X,y,fold,s=load();category=cats(a);ids=f.TRANSACTION_HASH.to_numpy();order=np.lexsort((ids,-s));top=np.zeros(len(y),bool);top[order[:int(np.ceil(len(y)*.01))]]=True;bottom=np.zeros(len(y),bool);pos=np.flatnonzero(y==1);bottom[pos[np.argsort(s[pos])[:125]]]=True
portal=json.loads((O/'run1_public_response.json').read_text()) if (O/'run1_public_response.json').exists() else json.loads((O/'leaderboard_public_response.txt').read_text());run=[v for v in portal['submissions'] if v['login']=='Lumen'];assert len(run)==1
local={str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in [ROOT/'outputs/final_submission/predictions.csv',ROOT/'outputs/final_submission/lumen_strategy3_v1_code.zip',ROOT/'outputs/submission_v1/predictions.csv',ROOT/'outputs/submission_v1/lumen_strategy3_v1_code.zip']}
(O/'run1_provenance.json').write_text(json.dumps({'official_public_run':run[0],'public_dataset_metadata':portal['datasets'],'portal_fields_available':list(run[0]),'server_file_hash_available':False,'local_artifacts':local,'linkage_conclusion':'The two designated final local artifacts match frozen V1 exactly and their source reproduced the CSV. Public Run 1 metadata contains no file digest, uploaded filenames, or original uploads; exact linkage to server bytes cannot be independently confirmed.','rounded_breakdown':'Difficulty recall rounded to nearest 10%; 0% does not establish zero true positives; overall score/precision/recall also displayed rounded.'},indent=2)+'\n')
rows=[]
for k in [-1,0,1,2]:
 mask=np.ones(len(y),bool) if k==-1 else fold==k;z=category_metrics(a.loc[mask].reset_index(drop=True),s[mask],ids[mask]);z['fold']=k;rows.append(z)
cm=pd.concat(rows);cm.to_csv(O/'category_metrics.csv',index=False)
masks={'legitimate':y==0,'medium':category=='medium','easy':category=='easy','observable-hard':category=='observable-hard','ambiguous':category=='ambiguous','high_FP':(y==0)&top,'lowest_125_positive':bottom}
profiles=[]
for name,mask in masks.items():
 for col in X:
  z=X.loc[mask,col];profiles.append({'group':name,'rows':int(mask.sum()),'feature':col,'median':float(z.median()),'q25':float(z.quantile(.25)),'q75':float(z.quantile(.75)),'mean':float(z.mean())})
pd.DataFrame(profiles).to_csv(O/'observable_feature_profiles.csv',index=False)
# Descriptive separation, not feature selection or hypothesis-test significance.
effects=[]
for name,mask0,mask1 in [('medium_vs_legitimate',y==0,category=='medium'),('medium_vs_high_FP',(y==0)&top,category=='medium'),('high_FP_vs_legitimate',y==0,(y==0)&top)]:
 m=mask0|mask1;target=mask1[m].astype(int)
 for col in X:
  auc=roc_auc_score(target,X.loc[m,col]);effects.append({'comparison':name,'feature':col,'auc':auc,'separation':abs(auc-.5),'group0_rows':int(mask0.sum()),'group1_rows':int(mask1.sum())})
pd.DataFrame(effects).sort_values(['comparison','separation'],ascending=[True,False]).to_csv(O/'feature_separation.csv',index=False)
errors=[]
for name,mask in [('high_FP',(y==0)&top),('lowest_125_positive',bottom),('missed_medium',(category=='medium')&~top)]:
 z=a.loc[mask].copy();z['score']=s[mask];z['category']=category[mask];z['error_set']=name;errors.append(z)
pd.concat(errors).to_csv(O/'local_error_rows.csv',index=False)
counts=pd.concat(errors).groupby(['error_set','family','difficulty'],dropna=False).size().reset_index(name='rows');counts.to_csv(O/'error_pattern_counts.csv',index=False)
# Recall and score summaries at the actual global 1% review threshold.
summary={'pooled':metrics(y,s,ids),'global_score_threshold':float(s[order[int(np.ceil(len(y)*.01))-1]]),'global_top1_category_counts':pd.Series(category[top]).value_counts().to_dict(),'high_FP_rows':int(((y==0)&top).sum()),'lowest_positive_selection':'bottom 125 positives (10% of all positives), fixed before candidate testing'};(O/'baseline_cutoff.json').write_text(json.dumps(summary,indent=2)+'\n')
print(cm[cm.fold.eq(-1)].to_string(index=False));print(counts.to_string(index=False));print(json.dumps(summary,indent=2))
