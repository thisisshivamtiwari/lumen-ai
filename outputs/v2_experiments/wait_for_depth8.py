from pathlib import Path
import time,os,signal
p=Path('outputs/v2_experiments/model_comparison.csv')
for _ in range(180):
 if p.exists() and '\ndepth8,' in p.read_text():
  os.kill(99893,signal.SIGTERM);Path('outputs/v2_experiments/implementation_fix_note.txt').write_text('Paused after non-feature candidate runs to correct the current-counterparty reference for sender-in and receiver-out local features. No label or validation-result-guided change. Extra-feature candidates run only with corrected build and are the reported results.\n');break
 time.sleep(2)
else:raise RuntimeError('No completion within six minutes')
