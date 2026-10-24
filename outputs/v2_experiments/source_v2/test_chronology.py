"""Tiny synthetic unit fixtures only; never used in fitting or scoring."""
from pathlib import Path
from unittest.mock import patch
import tempfile
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
import extra_features as e

def make(times,values):
 p=pd.DataFrame({'TRANSACTION_HASH':['a','b','c'],'TIMESTAMP':times,'TRANSACTION_INDEX':[0,1,2],'FROM_ADDRESS':['n0','n1','n2'],'TO_ADDRESS':['n1','n2','n3'],'VALUE':values})
 tr=pd.DataFrame({'TRANSACTION_HASH':['a','b','c'],'TRANSFER_INDEX':[0,0,0],'TOKEN_ADDRESS':[e.XRP]*3,'SYMBOL':['XRP']*3,'VALUE':values})
 with tempfile.TemporaryDirectory() as d:
  pq.write_table(pa.Table.from_pandas(tr),Path(d)/'transfers.parquet')
  with patch.object(e,'load_payments',return_value=p.to_dict('records')),patch.object(e,'load_rates',return_value={(0,'XRP'):1000000}):return e.build_extra(d)
f=make([0,60,120],[100,100,100]);assert f['chrono_3600_0.01_back_depth'].tolist()==[0,1,2];assert f['chrono_3600_0.01_forward_depth'].tolist()==[2,1,0];assert f['chrono_3600_0.01_two_sided'].tolist()==[0,1,0]
f=make([0,60,60],[100,100,100]);assert f['chrono_3600_0.01_back_depth'].tolist()==[0,1,0], 'same-second events cannot establish chronology'
f=make([0,60,120],[100,500,100]);assert f['chrono_3600_0.01_back_depth'].tolist()==[0,0,0], 'incompatible values must break chain'
print('Chronology semantics: 3 controlled unit fixtures PASS')
