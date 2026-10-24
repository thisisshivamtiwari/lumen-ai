"""Exact V1 evaluation definitions, with no executable V1-mutating entrypoint."""
import numpy as np
SEED=20261024
PARAMS=dict(iterations=1200,depth=6,learning_rate=.035,l2_leaf_reg=10,loss_function='Logloss',random_seed=SEED,thread_count=4,verbose=False,allow_writing_files=False,rsm=.85)
def groups_for(labels):
 return np.where(labels.cohort_id.notna(),'cohort:'+labels.cohort_id.fillna(''),np.where(labels.scenario_id.notna(),'scenario:'+labels.scenario_id.fillna(''),'row:'+labels.TRANSACTION_HASH))
def columns_for(all_columns,name):
 return [c for c in all_columns if not c.startswith('graph_') and '_neighbor_' not in c] if name=='behaviour' else list(all_columns)
