"""Validate exported predictions and the code archive without hidden labels."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--code-zip',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    pred=pd.read_csv(args.predictions)
    expected=pq.read_table(args.test/'transactions.parquet',columns=['TRANSACTION_HASH']).to_pandas().TRANSACTION_HASH
    assert pred.columns.tolist()==['TRANSACTION_HASH','score']
    assert expected.is_unique and expected.notna().all()
    assert len(pred)==len(expected) and len(pred)<=100000
    assert pred.TRANSACTION_HASH.is_unique and pred.TRANSACTION_HASH.notna().all()
    assert set(pred.TRANSACTION_HASH)==set(expected)
    assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
    assert args.predictions.stat().st_size<=12*1024**2
    assert args.code_zip.stat().st_size<=8*1024**2
    with zipfile.ZipFile(args.code_zip) as z:
        assert z.testzip() is None
        files=z.infolist();assert len(files)<=2000
        unpacked=sum(f.file_size for f in files);assert unpacked<=64*1024**2
        for f in files:
            path=Path(f.filename)
            assert not path.is_absolute() and '..' not in path.parts
            assert not f.flag_bits&1 and f.extract_version<45
            assert not any(p in {'.venv','node_modules','data','__pycache__'} for p in path.parts)
            assert path.suffix not in {'.parquet','.joblib','.cbm'}
        names={f.filename for f in files}
        assert {"run.py",'features.py','config.json','requirements.txt','README.md','LICENSE','AI_DISCLOSURE.md','methodology.pdf'}<=names
    report={'status':'PASS','rows':len(pred),'unique_scores':int(pred.score.nunique()),
        'csv_bytes':args.predictions.stat().st_size,'code_zip_bytes':args.code_zip.stat().st_size,
        'code_uncompressed_bytes':unpacked,'code_files':len(files),
        'csv_sha256':hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        'code_zip_sha256':hashlib.sha256(args.code_zip.read_bytes()).hexdigest(),
        'scope':'Local artifact checks only; not organiser acceptance or hidden-test performance.'}
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
