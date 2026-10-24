"""Predeclared 30% event-history thinning; no labels used for retention."""
from pathlib import Path
import tempfile,os,json,time,hashlib
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow as pa
from features import build_features
from extra_features import build_extra
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';d=ROOT/'data/training/train'
tx=pq.read_table(d/'transactions.parquet');keep=np.array([int(hashlib.sha256(h.encode()).hexdigest()[:8],16)%10>=3 for h in tx.column('TRANSACTION_HASH').to_pylist()]);ids=set(tx.column('TRANSACTION_HASH').to_numpy()[keep]);tr=pq.read_table(d/'transfers.parquet');tk=np.array([h in ids for h in tr.column('TRANSACTION_HASH').to_pylist()]);start=time.perf_counter()
with tempfile.TemporaryDirectory(prefix='lumen-thinned-') as tmp:
 td=Path(tmp);pq.write_table(tx.filter(pa.array(keep)),td/'transactions.parquet');pq.write_table(tr.filter(pa.array(tk)),td/'transfers.parquet');os.symlink(d/'market_rates.csv',td/'market_rates.csv')
 x=build_features(td);e=build_extra(td);x.merge(e,on='TRANSACTION_HASH',validate='one_to_one').to_parquet(O/'sparse_context_features.parquet',index=False)
(O/'sparse_context.json').write_text(json.dumps({'history_retention':float(keep.mean()),'rows':int(keep.sum()),'selection':'SHA256 transaction hash modulo 10 >=3; hash used only for fixed label-blind sampling, never as predictor','effect':'30% of economic events removed; remaining targets scored with less unlabelled context. Compare same retained evaluation targets with full context. This is inference covariate-shift stress, not missing-feature imputation.','seconds':time.perf_counter()-start},indent=2)+'\n')
print('Sparse history complete',flush=True)
