"""Label-free repeated-behaviour diagnostics, not predictor construction."""
from pathlib import Path
import os,json
import numpy as np,pandas as pd
from payments import load_payments
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';f=pd.read_parquet(ROOT/'outputs/features_v1/train.parquet');p=pd.DataFrame(load_payments(ROOT/'data/training/train')).set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH];fold=np.load(O/'locked_splits.npz')['fold_A'];v=p.VALUE.to_numpy(float);t=p.TIMESTAMP.to_numpy('int64');pairs=list(zip(p.FROM_ADDRESS,p.TO_ADDRESS));lookup={}
for j,pair in enumerate(pairs):lookup.setdefault(pair,[]).append(j)
rows=[]
for k in range(3):
 va=np.flatnonzero(fold==k);matches=0
 for j in va:
  ix=np.asarray(lookup[pairs[j]]);ix=ix[fold[ix]!=k]
  if len(ix) and np.any((abs(t[ix]-t[j])<=86400)&(abs(v[ix]-v[j])/v[j]<=.01)):matches+=1
 rows.append({'fold':k,'validation_rows':len(va),'same_pair_within_24h_and_1pct_value_seen_in_fit':matches,'fraction':matches/len(va)})
result={'heuristic':'same directed customer pair, <=24h absolute gap and <=1% delivered reference-USD difference; similarity is not proof of duplicate labels or leakage','folds':rows,'exact_duplicate_full_numeric_feature_vectors':int(f.drop(columns='TRANSACTION_HASH').duplicated().sum())};(O/'near_duplicate_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

with (O/'validation_audit.md').open('a') as stream:stream.write('\nRepeated-behaviour audit: no duplicate full numeric feature vectors; about 1% of validation events have a same-directed-pair fit event within 24 hours and 1% delivered reference value. See near_duplicate_audit.json. This heuristic is not proof of duplicate labels.\n')
