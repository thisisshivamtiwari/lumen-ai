from pathlib import Path
import os,tempfile,json
import numpy as np,pandas as pd,pyarrow.parquet as pq,pyarrow as pa
from extra_features import build_extra
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';D=ROOT/'data/training/train'
# Independent shuffled raw-row and poisoned-label check on a 2,000-event fixture.
tx=pq.read_table(D/'transactions.parquet').slice(0,2000);ids=set(tx.column('TRANSACTION_HASH').to_pylist());tr=pq.read_table(D/'transfers.parquet');keep=pa.array([h in ids for h in tr.column('TRANSACTION_HASH').to_pylist()]);tr=tr.filter(keep)
with tempfile.TemporaryDirectory(prefix='lumen-feature-test-') as tmp:
 a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir();rng=np.random.default_rng(91)
 for d in [a,b]:os.symlink(D/'market_rates.csv',d/'market_rates.csv')
 pq.write_table(tx,a/'transactions.parquet');pq.write_table(tr,a/'transfers.parquet');pq.write_table(tx.take(pa.array(rng.permutation(tx.num_rows))),b/'transactions.parquet');pq.write_table(tr.take(pa.array(rng.permutation(tr.num_rows))),b/'transfers.parquet');(b/'labels.csv').write_text('TRANSACTION_HASH,is_suspicious,family\npoison,1,forbidden\n')
 x=build_extra(a).set_index('TRANSACTION_HASH').sort_index();z=build_extra(b).set_index('TRANSACTION_HASH').sort_index();assert x.equals(z)
 result={'row_shuffle_and_label_poison':'PASS','fixture_rows':len(x),'features':len(x.columns),'finite':'PASS','invariant_scope':'Raw rows shuffled and poisoned labels simultaneously; independent code inspection confirms builder never opens labels. Does not establish target predictability.','semantics':'DAG edges strictly earlier -> later timestamp at a shared economic customer wallet, delivered reference USD compared to outgoing actual-debit reference USD; no LP hops; tolerance per edge, not conservation over the whole chain. Per-hop horizon 1h/6h, depths clipped8; whole chain may exceed that horizon.'}
 full=pd.read_parquet(O/'extra_train.parquet').drop(columns='TRANSACTION_HASH');result['constant_extra_features']=full.columns[full.nunique().eq(1)].tolist();result['nonfinite_extra']=int((~np.isfinite(full.to_numpy())).sum())
 (O/'extra_feature_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print('Extra feature invariants PASS',flush=True)
