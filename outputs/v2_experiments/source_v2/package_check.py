"""Reproduce copied baseline inference from an extracted research source ZIP."""
from pathlib import Path
import os,zipfile,json,subprocess,time,hashlib
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';src=Path(__file__).resolve().parent;target=O/'package_smoke_check';target.mkdir(exist_ok=True);code=target/'code';code.mkdir(exist_ok=True)
with zipfile.ZipFile(target/'source_smoke.zip','w',zipfile.ZIP_DEFLATED,allowZip64=False) as z:
 for p in sorted(src.iterdir()):
  if p.is_file() and p.suffix in ['.py','.json','.txt','.md']:z.write(p,p.name)
with zipfile.ZipFile(target/'source_smoke.zip') as z:z.extractall(code)
command=[str(O/'v1_clean_reproduction/.venv/bin/python'),str(code/'run.py'),'--train',str(ROOT/'data/training/train'),'--test',str(ROOT/'data/test'),'--output',str(target/'reference_predictions.csv')];start=time.perf_counter();r=subprocess.run(command,cwd=code,capture_output=True,text=True);(target/'stdout.log').write_text(r.stdout);(target/'stderr.log').write_text(r.stderr);assert r.returncode==0,r.stderr
original=ROOT/'outputs/submission_v1/predictions.csv';same=original.read_bytes()==(target/'reference_predictions.csv').read_bytes();assert same
record={'status':'PASS','byte_identical_baseline_csv':same,'csv_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'command':command,'scope':'Research source snapshot extracted into a new directory, executed with previously created clean pinned baseline environment; no new V2 candidate CSV is certified. Final archive must retain these inference file hashes.','inference_source_hashes':{n:hashlib.sha256((code/n).read_bytes()).hexdigest() for n in ['run.py','features.py','payments.py','starter_features.py','config.json','requirements.txt']}};(O/'source_package_reproduction.json').write_text(json.dumps(record,indent=2)+'\n');print('Research source baseline reproduction PASS',flush=True)
