"""Create a research-only source/config/report archive, never publish it."""
from pathlib import Path
import os,json,hashlib,zipfile,shutil
ROOT=Path(os.environ.get('LUMEN_WORKSPACE',Path.cwd()));O=ROOT/'outputs/v2_experiments';src=Path(__file__).resolve().parent;stage=O/'package_contents';stage.mkdir(exist_ok=True)
for p in src.iterdir():
 if p.is_file() and (p.suffix in ['.py','.txt','.md','.json'] or p.name=='LICENSE'):shutil.copy2(p,stage/p.name)
for name in ['methodology.md','methodology.pdf']:shutil.copy2(O/name,stage/name)
e=stage/'evidence';e.mkdir(exist_ok=True)
names=['acceptance.json','frozen_finalist.json','experiment_plan.json','experiment_config.json','model_comparison.csv','experiment_registry.csv','feature_ablation.csv','candidate_difficulty_comparison.csv','confirmation_results.csv','baseline_reproduction_report.md','validation_audit.md','error_analysis.md','baseline_recomputed_metrics.json','baseline_reproduction.json','v1_reproduction_command.json','baseline_artifact_recheck.json','baseline_integrity_recheck.json','validation_evidence.json','near_duplicate_audit.json','uncertainty.json','error_slices.csv','review_budget_errors.csv','review_budget_tie_check.json','score_distributions.csv','fold_difficulty_analysis.csv','sparse_context.json','extra_feature_checks.json','source_package_reproduction.json','v1_preservation_check.json','environment.json','baseline_diagnostics.png','candidate_diagnostics.png','candidate_review_budgets.csv']
for name in names:
 if (O/name).is_file():shutil.copy2(O/name,e/name)
# Preserve exact inference file bytes tested from the earlier extracted source ZIP.
proof=json.loads((O/'source_package_reproduction.json').read_text())
for name,digest in proof['inference_source_hashes'].items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest
manifest={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='source_manifest.json'};(stage/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'lumen_strategy3_v2_research_code.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,allowZip64=False) as z:
 for p in sorted(stage.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(stage).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;assert len(z.infolist())<=2000;assert sum(i.file_size for i in z.infolist())<=64*1024**2;assert all(not n.endswith(('.parquet','.cbm','.joblib')) and not n.startswith('data/') for n in z.namelist())
 assert json.loads(z.read('source_manifest.json'))==manifest
 for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
assert archive.stat().st_size<=8*1024**2
r={'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':len(manifest)+1,'all_manifest_hashes_verified':True,'inference_files_match_extracted_smoke_test':True,'scope':'Research archive; no accepted V2 CSV; not uploaded or public.'};(O/'release_manifest.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
