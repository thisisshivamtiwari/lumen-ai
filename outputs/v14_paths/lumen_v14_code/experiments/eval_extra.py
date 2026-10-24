"""Extra metrics for stage-1 OOF candidates: flagged = top K of its fold, K = fold's positives.
Usage: eval_extra.py <oof name> [<oof name> ...]  (reads outputs/v3_context/<name>_oof.csv)"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
labels = pd.read_csv(ROOT / 'data/training/train/labels.csv')
feats = pd.read_parquet(ROOT / 'outputs/features_v1/train.parquet', columns=['TRANSACTION_HASH', 'left_exposure_hours', 'right_exposure_hours'])
rows = []
for name in sys.argv[1:]:
    o = pd.read_csv(ROOT / f'outputs/v3_context/{name}_oof.csv').merge(labels, on='TRANSACTION_HASH').merge(feats, on='TRANSACTION_HASH')
    m = json.load(open(ROOT / f'outputs/v3_context/{name}_metrics.json'))[name]
    flagged = np.zeros(len(o), bool)
    for i, g in o.groupby('fold'):
        k = int(g.is_suspicious.sum())
        flagged[g.nlargest(k, 'score').index] = True
    o['flagged'] = flagged
    fp = o[(o.is_suspicious == 0) & o.flagged]
    relay = fp[fp.family.isin(['aggregate_relay', 'distributed_relay'])]
    easy = o[(o.is_suspicious == 1) & (o.difficulty == 'easy')]
    near = easy[np.minimum(easy.left_exposure_hours, easy.right_exposure_hours) < 48]
    rows.append({'candidate': name, 'mean_fold_ap': round(100 * m['mean_fold_ap'], 2), 'folds': [round(100 * x, 2) for x in m['fold_ap']],
                 'tp_top1pct': m['tp_top402'], 'easy_ap': round(m['category_ap']['easy'], 3), 'medium_ap': round(m['category_ap']['medium'], 3),
                 'hard_ap': round(m['category_ap']['hard'], 3), 'fp_total': len(fp), 'fp_relay': len(relay),
                 'fp_bipartite': int((fp.family == 'bipartite').sum()), 'fp_by_family': fp.family.value_counts().to_dict(),
                 'relay_fp_median_right_h': round(float(relay.right_exposure_hours.median()), 1) if len(relay) else None,
                 'easy_catch': round(easy.flagged.mean(), 3), 'easy_catch_near_edge': round(near.flagged.mean(), 3), 'easy_near_edge_n': len(near)})
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).drop(columns='fp_by_family').to_string(index=False))
for r in rows:
    print(r['candidate'], 'FP by family:', r['fp_by_family'])
