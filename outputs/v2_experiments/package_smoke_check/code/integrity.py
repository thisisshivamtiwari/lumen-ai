from pathlib import Path
import json,hashlib,os
import numpy as np,pandas as pd,pyarrow.parquet as pq
from payments import load_payments,load_rates,XRP
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));out=ROOT/'outputs/v2_experiments';result={}
manifest=json.loads((ROOT/'data/training/checksums.json').read_text());assert all(hashlib.sha256((ROOT/'data/training'/p).read_bytes()).hexdigest()==v for p,v in manifest.items());result['official_checksums_passed']=len(manifest)
wallets={};hashes={}
for name,d in [('train','data/training/train'),('practice','data/training/validation'),('test','data/test')]:
 d=ROOT/d;tx=pq.read_table(d/'transactions.parquet').to_pandas();tr=pq.read_table(d/'transfers.parquet').to_pandas();b=pq.read_table(d/'blocks.parquet').to_pandas();p=pd.DataFrame(load_payments(d));rates=load_rates(d)
 assert tx.TRANSACTION_HASH.is_unique and not tx.TRANSACTION_HASH.isna().any();assert not tr.duplicated(['TRANSACTION_HASH','TRANSFER_INDEX']).any();assert set(tx.TRANSACTION_HASH)==set(tr.TRANSACTION_HASH)
 assert not b.duplicated(['LEDGER_INDEX','LEDGER_HASH']).any();joined=tx.merge(b[['LEDGER_INDEX','LEDGER_HASH','TIMESTAMP']],on=['LEDGER_INDEX','LEDGER_HASH'],how='left',suffixes=('','_block'),validate='many_to_one');assert joined.TIMESTAMP.eq(joined.TIMESTAMP_block).all()
 ts=tx.TIMESTAMP.astype('int64');assert np.array_equal(pd.to_datetime(ts,unit='s',utc=True).astype('datetime64[ns, UTC]').astype('int64'),tx.DATETIME.astype('datetime64[ns, UTC]').astype('int64'))
 assert tx.SUCCESS.all() and tx.SOURCE_FEE.astype(float).between(10,15).all() and np.allclose(tx.FEE.astype(float)*1e6,tx.SOURCE_FEE.astype(float));assert np.isfinite(tr.VALUE.astype(float)).all() and tr.VALUE.gt(0).all();assert p.VALUE.gt(0).all()
 order=tr.sort_values(['TRANSACTION_HASH','TRANSFER_INDEX']);first=order.drop_duplicates('TRANSACTION_HASH').set_index('TRANSACTION_HASH').loc[p.TRANSACTION_HASH];last=order.drop_duplicates('TRANSACTION_HASH',keep='last').set_index('TRANSACTION_HASH').loc[p.TRANSACTION_HASH]
 assert first.FROM_ADDRESS.to_list()==p.FROM_ADDRESS.to_list();assert last.TO_ADDRESS.to_list()==p.TO_ADDRESS.to_list()
 asset=np.where(last.TOKEN_ADDRESS.eq(XRP),'XRP',last.TOKEN_ADDRESS+':'+last.SYMBOL);value=last.VALUE.astype(float).to_numpy()*np.array([rates[(int(t)//3600*3600,a)]/1e6 for t,a in zip(p.TIMESTAMP,asset)])
 assert np.allclose(value,p.VALUE,rtol=2e-5,atol=1e-5)
 legs=tr.groupby('TRANSACTION_HASH').TRANSFER_INDEX.apply(lambda v: sorted(map(int,v))==list(range(len(v))));assert legs.all()
 result[name]={'transactions':len(tx),'transfer_rows':len(tr),'ledger_rows':len(b),'missing_required_keys':int(tx[['TRANSACTION_HASH','TIMESTAMP','ACCOUNT']].isna().sum().sum()),'max_projection_transfer_valuation_relative_error':float(np.max(abs(value-p.VALUE)/p.VALUE)),'ledger_join':'PASS','timestamp_check':'PASS','transfer_leg_order':'PASS','projection_endpoints':'PASS','null_counts':{str(c):int(v) for c,v in tx.isna().sum().items() if v}}
 wallets[name]=set(p.FROM_ADDRESS)|set(p.TO_ADDRESS);hashes[name]=set(p.TRANSACTION_HASH)
for a,b in [('train','practice'),('train','test'),('practice','test')]:assert not wallets[a]&wallets[b];assert not hashes[a]&hashes[b]
pred=pd.read_csv(ROOT/'outputs/submission_v1/predictions.csv');expected=pq.read_table(ROOT/'data/test/transactions.parquet',columns=['TRANSACTION_HASH']).to_pandas().TRANSACTION_HASH;assert pred.TRANSACTION_HASH.equals(expected);result['exact_test_order']='PASS';result['disjoint_splits']='PASS'
(out/'baseline_integrity_recheck.json').write_text(json.dumps(result,indent=2)+'\n');print('Integrity PASS',flush=True)
