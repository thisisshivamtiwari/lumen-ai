"""PyTorch neural blend member (UNVALIDATED stretch run): residual MLP on the V14 feature table.

Trains on all labelled rows except the hard-tier positives (as the tree halves do), averages three
seeds, scores the test split and writes outputs/v16_torch/torch_test.csv. No out-of-fold validation is
performed here by design (time-boxed); any blend with it is unvalidated.
Run from the project root: .venv/bin/python solution/torch_member.py
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/v16_torch'
sys.path.insert(0, str(ROOT / 'solution'))
sys.path.insert(0, str(ROOT / 'outputs/v3_context/source'))
from experiment import load, OUT as EXP  # noqa: E402
from run_v3 import with_motif, with_burst, with_paths  # noqa: E402
from features import build_features  # noqa: E402

SEEDS = (20261024, 20261025, 20261026)
CONFIG = json.loads((ROOT / 'solution/config.json').read_text())
EPOCHS, BATCH = 40, 512


class ResMLP(nn.Module):
    def __init__(self, d, width=512, depth=3, drop=0.2):
        super().__init__()
        self.inp = nn.Linear(d, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(drop),
                                                   nn.Linear(width, width), nn.Dropout(drop)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h).squeeze(-1)


def train_predict(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(8)
    xt, yt, xe = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
    model = ResMLP(xt.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * ((len(xt) + BATCH - 1) // BATCH))
    pos_weight = torch.tensor((len(ytr) - ytr.sum()) / ytr.sum() / 4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe)).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    f, a = load()
    y = a.is_suspicious.to_numpy()
    cache = pd.read_parquet(EXP / 'bcp_train.parquet').set_index('TRANSACTION_HASH').loc[f.TRANSACTION_HASH].reset_index(drop=True)
    X = pd.concat([f[CONFIG['features']].reset_index(drop=True), cache], axis=1)
    keep = ~((y == 1) & a.difficulty.eq('hard').to_numpy())
    test_dir = ROOT / 'data/test'
    test = build_features(test_dir)
    test = with_paths(with_burst(with_motif(test, test_dir), test_dir), test_dir)
    T = test[X.columns]
    assert T.columns.tolist() == X.columns.tolist() and np.isfinite(T.to_numpy()).all()
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=SEEDS[0]).fit(X[keep])
    Xtr, Xte = qt.transform(X[keep]), qt.transform(T)
    print(f'features ready {time.perf_counter() - start:.0f}s; train rows {keep.sum()}, dim {Xtr.shape[1]}', flush=True)
    preds = []
    for s in SEEDS:
        preds.append(train_predict(Xtr, y[keep], Xte, s))
        print(f'seed {s} done {time.perf_counter() - start:.0f}s', flush=True)
    score = np.mean(preds, axis=0)
    pd.DataFrame({'TRANSACTION_HASH': test.TRANSACTION_HASH, 'score': score}).to_csv(OUT / 'torch_test.csv', index=False)
    print(f'written {OUT / "torch_test.csv"} {time.perf_counter() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
