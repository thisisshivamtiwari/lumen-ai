# Lumen Strategy3 - first submission

This package trains a fixed CatBoost model from the organiser's labelled training split, then scores every test transaction. Only supplied competition data is used. It does not query a service or upload anything.

## Exact reproduction command

Python 3.12.13 was used on an Apple Silicon laptop with 8 GB RAM. From the extracted code ZIP directory:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --train /absolute/path/to/train --test /absolute/path/to/test-data --output predictions.csv
```

The training directory contains the released labelled tables and labels.csv. The test directory contains the separate test tables and rates. Approximately one minute is needed after installation on the reference laptop; runtime varies. Four CPU model threads are used. No GPU is required.

The unlabelled validation/ directory is practice data, not the scored test. Use the actual 40,101-transaction test directory for the final CSV. A machine-readable output report is saved beside predictions.csv. To optionally save the native model, append --save-model model.cbm.

## Approach and fixed configuration

- One customer sender-to-recipient economic payment per transaction, using the organiser helper and delivered reference USD value.
- First executed transfer leg supplies actual paid value, converted with the issuer-aware containing-hour reference rate. Liquidity-provider legs are not independent customer payments.
- 457 numeric features cover starter rules, directional wallet and pair statistics, 5-minute/1-hour/6-hour/24-hour/72-hour flows, amount/time compatibility and bounded static graph roles.
- The feature builder reads transactions, transfers and rates, never labels. No raw account/hash/issuer ID, split-ID, absolute-date, scenario, family, cohort or difficulty is a predictor.
- Full unlabelled within-window context is permitted by the released README. Each supplied split is processed independently, with identical feature definitions.
- Fixed CatBoost parameters and ordered predictor names are in config.json. No cached features, saved model, test labels, external transaction data or synthetic replicas are required.

## Evaluation and limitations

Three-fold grouped evaluation keeps each supplied cohort together, otherwise each scenario together, otherwise each background row in its own group. Annotation fields are used only for evaluation. This is not a completely wallet-disjoint validation.

| Candidate | Mean grouped AP x100 | Pooled OOF AP x100 |
|---|---:|---:|
| Starter-feature CatBoost | 35.820 | 34.271 |
| Behaviour features | 54.063 | 53.407 |
| Selected graph-enhanced features | 54.483 | 53.719 |

A first-five-day-labels / last-two-day-labels diagnostic scores 56.269 AP x100, using the full authorised unlabelled seven-day context. This is an offline label holdout, not a causal detection experiment or independent final test.

These are local metrics, not competition scores. Medium/hard historical relays remain weaknesses; gains are mainly on easy cases. The organiser states that 25 hard positives per split have labels independent of observable features. No probability calibration, guaranteed hidden-test score or real-world AML readiness is claimed. Difficulty slices use a common negative background; different slice prevalence prevents direct raw-AP comparisons.

## Contents and provenance

run.py is the final reproduction path. features.py implements new features. payments.py is the supplied helper; starter_features.py extracts the organiser notebook's feature definitions. baseline.py, evaluate_v1.py and audit_data.py retain experiment/audit source, expect the development data/outputs layout in their paths, and are not required by the reproduction command above.

See methodology.pdf, validation_summary.json, THIRD_PARTY.md, AI_DISCLOSURE.md and LICENSE. The ZIP excludes datasets, predictions, model binaries and installed dependencies. Original solution code is MIT licensed; organiser material and libraries retain their original terms.

## Upload

Upload the separate predictions.csv and code ZIP. No script submits automatically. The validator checks local format and archive constraints; final acceptance is determined by the organiser's loader and rules.
