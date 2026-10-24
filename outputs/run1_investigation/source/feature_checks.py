from common import *
import flexible_flow as ff
import tempfile,os
from unittest.mock import patch
import pyarrow as pa,pyarrow.parquet as pq

def fixture(times):
 p=pd.DataFrame({'TRANSACTION_HASH':['a','b','c'],'TIMESTAMP':times,'TRANSACTION_INDEX':[0,1,2],'FROM_ADDRESS':['a','b','hub'],'TO_ADDRESS':['hub','hub','sink'],'VALUE':[50,50,100]});tr=pd.DataFrame({'TRANSACTION_HASH':['a','b','c'],'TRANSFER_INDEX':[0,0,0],'TOKEN_ADDRESS':[ff.XRP]*3,'SYMBOL':['XRP']*3,'VALUE':[50,50,100]})
 with tempfile.TemporaryDirectory() as tmp:
  pq.write_table(pa.Table.from_pandas(tr),Path(tmp)/'transfers.parquet')
  with patch.object(ff,'load_payments',return_value=p.to_dict('records')),patch.object(ff,'load_rates',return_value={(0,'XRP'):1000000}):return ff.build(tmp).set_index('TRANSACTION_HASH')
z=fixture([0,5,10]);assert z.loc['c','flex_sender_prior_in_3600_min_prefix_error']==0;assert z.loc['c','flex_sender_prior_in_3600_best_prefix_terms']==2
z=fixture([0,10,10]);assert z.loc['c','flex_sender_prior_in_3600_min_prefix_error']==.5;assert z.loc['c','flex_sender_prior_in_3600_best_prefix_terms']==1
D=ROOT/'data/training/train';tx=pq.read_table(D/'transactions.parquet').slice(0,2000);ids=set(tx.column('TRANSACTION_HASH').to_pylist());tr=pq.read_table(D/'transfers.parquet');tr=tr.filter(pa.array([h in ids for h in tr.column('TRANSACTION_HASH').to_pylist()]));rng=np.random.default_rng(18)
with tempfile.TemporaryDirectory() as tmp:
 da=Path(tmp)/'a';db=Path(tmp)/'b';da.mkdir();db.mkdir()
 for dd in [da,db]:os.symlink(D/'market_rates.csv',dd/'market_rates.csv')
 pq.write_table(tx,da/'transactions.parquet');pq.write_table(tr,da/'transfers.parquet');pq.write_table(tx.take(pa.array(rng.permutation(tx.num_rows))),db/'transactions.parquet');pq.write_table(tr.take(pa.array(rng.permutation(tr.num_rows))),db/'transfers.parquet');(db/'labels.csv').write_text('poison\n1\n');x=ff.build(da).set_index('TRANSACTION_HASH').sort_index();y=ff.build(db).set_index('TRANSACTION_HASH').sort_index();assert x.equals(y)
(O/'feature_checks.json').write_text(json.dumps({'prefix_sum_known_answer':'PASS','same_timestamp_excluded':'PASS','raw_row_shuffle_and_poisoned_label_fixture':'PASS','fixture_rows':2000,'features':33,'scope':'Synthetic tiny unit fixtures and raw-input subset invariance only; never added to model fitting. Cumulative values are compatibility candidates, not proof of funds provenance.'},indent=2)+'\n');print('Flexible-flow checks PASS')
