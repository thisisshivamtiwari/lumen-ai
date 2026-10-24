# Lumen V15 - final: equal rank blend of V13 (V11 + V12) and V14, each a blend of two fitting modes

This package trains CatBoost models from the organiser's labelled training split, then scores every test transaction. Only supplied competition data is used. It does not query a service or upload anything.

## Exact reproduction command

Python 3.12.13 was used on an Apple Silicon laptop (8 CPU cores). From the extracted code ZIP directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
T=/abs/path/to/train; S=/abs/path/to/test-data
# V11 halves (+-10 min burst context only, no path features)
.venv/bin/python run_v3.py --basic-burst --no-paths --keep-hard --train $T --test $S --output v11a.csv
.venv/bin/python run_v3.py --basic-burst --no-paths             --train $T --test $S --output v11b.csv
# V12 halves (extended burst context, no path features)
.venv/bin/python run_v3.py --no-paths --keep-hard --train $T --test $S --output v12a.csv
.venv/bin/python run_v3.py --no-paths             --train $T --test $S --output v12b.csv
# V14 halves (extended burst context + slow scatter-gather paths)
.venv/bin/python run_v3.py --keep-hard --train $T --test $S --output v14a.csv
.venv/bin/python run_v3.py             --train $T --test $S --output v14b.csv
.venv/bin/python blend_runs.py --a v11a.csv --b v11b.csv --test $S --output v11.csv
.venv/bin/python blend_runs.py --a v12a.csv --b v12b.csv --test $S --output v12.csv
.venv/bin/python blend_runs.py --a v14a.csv --b v14b.csv --test $S --output v14.csv
.venv/bin/python blend_runs.py --a v11.csv  --b v12.csv  --test $S --output v13.csv
.venv/bin/python blend_runs.py --a v13.csv  --b v14.csv  --test $S --output predictions.csv
```

Each `run_v3.py` call takes about 8-10 minutes (two can run in parallel on 8 cores); each blend takes seconds. No GPU is required. `run.py` is the earlier V1 single-model path, kept for reference. `--aux` is an experimental option (family annotation as an auxiliary training target); it is **off by default and not used**. `edge_context.py` is an evaluated-and-rejected module, not used.

**Annotation use.** No annotation (family, difficulty, scenario, cohort) is ever an inference input. The `labels.csv` difficulty column is used once, for training-row selection in the not-fitted-hard halves (the 225 hard-tier positives are scored but not fitted); the `--keep-hard` halves use only `is_suspicious`. Both halves are blended, and this choice is disclosed here and in the code.

**Path-feature amounts.** `payments.py` values each payment by its delivered amount at the public hourly reference price. The path ratio (leg-2 value / leg-1 value in reference USD) is a value-compatibility signal for ranking, not evidence that the same funds moved; conversion legs are flagged separately (`converted_share`).

## Approach

1. **Payments and V1 features** (`payments.py`, `features.py`): one economic sender-to-recipient edge per transaction, valued in reference USD; 457 label-free features (wallet and pair statistics, multi-window flows, amount/time compatibility, bounded static graph roles).
2. **Multi-scale motif features** (`motif.py`, new in V4): payments are linked when they touch the same wallet and are consecutive in that wallet's history within D minutes (D = 2, 5, 10, 20). Connected components approximate one burst of related activity. Each payment gets its burst's size, duration, rate, wallet roles (sources, sinks, intermediates, max in/out degree), cyclomatic number, repeated pairs, flow conservation through intermediates, value spread and its own position in the burst (80 features).
3. **Ego-network features** (`motif.ego_features`, new in V5): for D = 2, 5, 10 minutes, the number of payments sharing a wallet with this payment within +-D (1 hop) and reachable through one more shared wallet within +-D of it (2 hops), their wallet count, value spread and value relative to this payment. Unlike bursts, these are not transitive, so busy wallets cannot chain unrelated activity into them (15 features).
4. **Burst-context features** (`burst_context.py`, new in V11): for the sender's outgoing and the receiver's incoming payments within +-10 minutes: burst size, share of one-off counterparties among the other burst payments, the wallet's whole-week one-off share, the surprise of this payment being one-off for this wallet, burst-vs-wallet one-off difference, burst rate against the wallet's expected rate, amount spread in the burst and against the wallet (16 features). They answer "is this burst normal for this wallet?", which separates a legitimate hub's busy spell from a planted fan. V12 adds the same eight side features at +-30 min and +-2 h windows (`bc30_`, `bc120_`; slower planted scenarios) and two re-convergence features: the largest number of distinct burst counterparties that pay the same downstream wallet within 6 h (sender side) or received from the same upstream wallet within 6 h (receiver side). `--basic-burst` restores V11.
5. **Slow scatter-gather path features** (`path_features.py`, new in V14): value-compatible two-hop paths X -> M -> Y (hold 0-6 h, delivered/received ratio 0.6-1.25, distinct wallets), each payment keeping its best partner per far endpoint. Hops with the same source and destination whose first legs fall within +-1 h / +-6 h form a non-transitive group; each payment gets its best group's number of distinct middlemen, ratio and kept-share medians, value spread, hold median and spread, span, converted share, and its scatter/gather role (20 features). They target planted scatter-gather scenarios spread over 1-6 h, which the short burst blocks cannot see. `--no-paths` restores V12.
6. **Stage 1**: CatBoost on V1 + motif + ego features (fixed parameters in `config.json`).
7. **Fitting rows** (new in V7): the 225 hard-tier positives (planted historical relays and the deliberately ambiguous payments) are left out of model fitting, but still scored. Fitting them taught the models to promote ~20,000 legitimate relay look-alikes while almost never ranking the hard relays themselves. This is training-row selection only; no annotation is a feature. `--keep-hard` restores V5 behaviour.
8. **Context stage** (`context.py`): for each payment, first-stage scores (as within-split percentiles) of other payments sharing its sender or receiver within 1/6/24/72 hours, directional up/downstream windows and whole-week wallet context. Training context uses grouped out-of-fold stage-1 scores; test context uses the full stage-1 model. Stage 2 is CatBoost on stage-1 features + context.
9. **Final score**: equal-weight blend of stage-1 and stage-2 percentile ranks, with the mean raw probability as a tie-breaker.

No raw account/hash/issuer ID, split ID, absolute date, scenario, family, cohort or difficulty is a predictor. Annotation fields define validation folds and evaluation slices only.

## Local evaluation

Three-fold grouped evaluation (cohort, else scenario, else row) on the training split, identical folds for every candidate. For the context stage, every first-stage score feeding a held-out fold's context comes from models that never saw that fold's labels (nested). `experiments/` contains the evaluation scripts.

| Candidate | Mean fold AP x100 | Pooled AP x100 | Folds better than V1 | Top-1% true positives |
|---|---:|---:|---:|---:|
| V1 (portal run 1) | 54.5 | 53.7 | - | 326 |
| V1 + context blend | 57.3 | 57.1 | 3/3 | 326 |
| V1 + motif | 59.8 | 58.7 | 3/3 | 349 |
| V4: motif + context blend (portal 66.5) | 61.6 | 61.5 | 3/3 | 355 |
| V5: + ego-network features (portal 66.6) | 62.0 | 62.0 | 3/3 | 356 |
| V7: hard-tier positives not fitted (portal 66.9) | 63.0 | 62.8 | 3/3 | 353 |
| V10: equal rank blend of V5 and V7 (portal 67.3) | 63.2 | - | 3/3 vs V7 | - |
| V11: V10 with burst-context features (portal 68.2) | see below | | | |
| V12: V11 with wider windows and re-convergence (portal 68.3) | see below | | | |
| V14: V12 plus slow scatter-gather paths (portal pending at packaging) | see below | | | |
| **V15: equal rank blend of V13 and V14 (this package)** | see below | | | |

Full not-fitted-hard pipelines (stage 1 + nested context stage, blended) on the identical grouped folds, aligned by transaction hash, top-1% budget = ceil(1%) = 402 with hash tie-break: V12 64.17 (67.5 / 68.0 / 57.0; 349 TP, 53 FP in top 402), V13 = V11+V12 64.40 (68.1 / 68.0 / 57.1; 350 / 52), V14 64.21 (68.2 / 67.7 / 56.8; 351 / 51), **V15 = V13+V14 64.75** (69.1 / 67.9 / 57.2; 352 / 50). V14 alone does not beat V13 through the full pipeline; it is included only through the blend, chosen on these folds from the two blends considered (V13+V14 and V12+V14 = 64.59). No separate time-based or wallet-disjoint check was run for V14/V15 (not run); the V5/V7 time-based confirmation split is in `experiments/confirm.py`. Local figures are optimistic: the same folds have been reused for selection throughout the day.

Burst-context stage-1 ablation on the same folds (hard-tier positives not fitted): without 60.98 (65.7 / 63.0 / 54.2), with **62.26** (66.7 / 64.9 / 55.2); with hard positives fitted: 60.27 -> 61.02. Full V7-style pipeline (stage 1 + nested context stage, blended) with burst context: mean fold AP 63.82 against 62.95 without (67.9 / 67.0 / 56.5 vs 66.9 / 65.8 / 56.2, better on all three folds). `--no-burst` restores V10. Stage-1 ablation for V12 on the same folds: V11 features 62.26 (66.7 / 64.9 / 55.2) -> wider windows 62.98 -> re-convergence 62.89 -> both **63.61** (66.9 / 66.6 / 57.4), medium-tier AP 0.081 -> 0.123. V14 stage-1 ablation on the same folds: V12 features 63.61 (66.9 / 66.6 / 57.4) -> with path features **63.96** (67.7 / 66.8 / 57.4); planted scatter-gather scenarios spanning 1-6 h caught 32% -> 55% at K = positives, legit scatter-gather look-alikes flagged 34% -> 49%, relay look-alike false positives 130 -> 102. An edge-of-window one-off correction (`edge_context.py`, `ec_` features) was evaluated and rejected: 63.13, better on one fold only; it is included for reference and not used.

A time-based confirmation split (labels from the first five days fitted, last two days evaluated, full-window features) gives AP 64.97 for V10 against 63.18 for V7 (paired group-bootstrap difference +1.79, 95% CI +0.39 to +3.44) and 66.50 for V5; V5's advantage there comes from hard relay scenarios that span the cutoff, which cannot occur on the test split (disjoint wallets). V10 hedges between the two. Scripts: `experiments/confirm.py`, `experiments/score.py`.

Control: averaging V1 over five seeds changed mean fold AP by +0.1, so the gains are not an ensembling artefact. The blend weight (0.5) was chosen on the same folds from three tried values. Easy AP rose from 0.73 to 0.84 and medium from 0.03 to 0.09 (V5 vs V1); burst-level context, longer training, value-compatible flow bursts and chain-depth features were tried and rejected (no gain on the same folds); observable-hard relays remain essentially undetected and the 25 ambiguous labels are independent of observables by design.

These are local metrics, not competition scores. Validation is not fully wallet-disjoint (only 700 wallets exist per split) and folds were reused across experiments, so gains can be optimistic.

## Contents and provenance

See `methodology.pdf` (V1 methodology; V4 changes are described above), `THIRD_PARTY.md`, `AI_DISCLOSURE.md` and `LICENSE`. The ZIP excludes datasets, predictions, model binaries and installed dependencies. Original solution code is MIT licensed; organiser material and libraries retain their original terms.

## V17 (stretch run, below acceptance bar): +-6 h and +-24 h burst-context windows

`run_v3.py --wide-windows` adds the eight burst-context side features at +-6 h and +-24 h. Stage-1 ablation on the identical grouped folds: 63.61 -> 63.89 (67.9 / 66.9 / 56.8 vs 66.9 / 66.6 / 57.4), better on two folds, below the 1-point bar used all day; no full-pipeline validation was run. V17 = equal rank blend of V15 and the wide-window pair of runs (`--wide-windows --keep-hard`, `--wide-windows`, blended with `blend_runs.py`). Submitted as a stretch run at the deadline; V15 remains the evidence-backed entry.
