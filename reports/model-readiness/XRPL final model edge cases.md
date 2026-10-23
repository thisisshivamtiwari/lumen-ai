# XRPL final model review: edge cases before submission

Researched 23 October 2026. Scope: Strategy3's synthetic XRP Ledger transaction-ranking challenge. This is a proposed acceptance plan for an already trained model, not a completed model audit. The workspace currently contains planning documents, transcripts and the training deck, but no trained model, pipeline, real dataset or prediction artifacts. Every check starts as **NOT RUN**.

## What I would sign off as a senior ML engineer

A good overall score is insufficient. I would require evidence that each score belongs to the right transaction, the payment reconstruction is economically meaningful, validation labels cannot influence prediction, inference reproduces training feature semantics, and the model's weaknesses are visible on meaningful cohorts. I would block submission for broken identity, numerical output, schema, leakage or reproducibility. I would document bounded robustness weaknesses rather than pretend every rare case can be solved before the deadline.

These are two different kinds of tests:

- **Correctness invariants:** an irrelevant change must not change the intended representation or result; for example, shuffling stored rows while retaining canonical event order must not change a hash's score. A failed invariant is a bug or an explicitly documented design mismatch.
- **Behavioural stress tests:** change real context and observe performance; for example, a slower laundering chain or a new issuer. There is no universal expected score or guarantee of unchanged AP. Report the result and its practical impact.

The test design below is our engineering synthesis. Sources establish API behaviour and ledger semantics, not that this model has any particular defect. Google's ML Test Score rubric motivates testing data, model and system together; it is not a checklist that certifies this competition model. [Google Research](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)

## Before running the checks

Freeze the model and decision policy. Save model/configuration hashes, dependency versions, random seeds, exact feature names/order/types, preprocessing and category mappings, chosen tree iteration, context visibility and allowed graph sources. Identify a labelled confirmation set that has not been used repeatedly to select features. If none remains, state that limitation; do not rename the tuning set an independent test.

Keep the entire target transaction ID spine. For this deck, the final test has 40,101 transactions; reconcile the count and exact membership with the actual release. Preserve test labels as unavailable. Use training holdouts for AP and supervised error review. On the unlabelled competition test set, only schema, distribution, graph, runtime and output checks are available.

Retain a baseline and last known-good submission. New fixes may require retraining and fresh evaluation; do not patch test scores by hand. If synthetic fixtures are used, keep truth/scenario metadata out of model inputs and do not treat fixture AP as evidence of hidden-test generalisation.

Priority meanings: **P0** = release-blocking correctness or trustworthy-evaluation requirement when applicable; **P1** = required risk review, with a recorded disposition rather than an invented universal numerical threshold; **P2** = conditional deeper test if the feature or time budget justifies it. Mark inapplicable checks **N/A with a reason**, not PASS.

## 1. Identity, schema and inference correctness

The first failure mode is evaluating good predictions and then exporting them against the wrong rows. Test the saved model through the actual final pipeline, not just a notebook estimator. CatBoost documents the need to align predictor features, and LightGBM exposes feature validation for supported DataFrame inputs. pandas also matches null merge keys to one another and can check join cardinality. [CatBoost prediction interface](https://catboost.ai/docs/en/concepts/python-reference_catboost_predict), [LightGBM Booster.predict](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html), [pandas.merge](https://pandas.pydata.org/docs/reference/api/pandas.merge.html)

| ID | Priority | Edge case and test | Required outcome / disposition |
|---|---|---|---|
| IO01 | P0 | **Correct count, wrong IDs.** Compare the exported hash set and multiplicities with the target spine; include missing, extra, duplicate, whitespace-altered and null hashes. | Exact target membership, one row per required hash, no accidental ID mutation; count alone is insufficient. |
| IO02 | P0 | **Prediction/label misalignment.** Independently shuffle feature rows and label rows; join by hash before scoring. Reorder the export and reload it. | Hash-aligned scores and evaluation remain the same. Never zip unrelated arrays by position. |
| IO03 | P0 | **Many-to-many and null-key joins.** Insert duplicate asset/hour lookup keys, null sender/asset/ledger keys and orphan rows into a small fixture. | Reject ambiguous lookup duplicates; do not match unrelated null records. Target spine stays intact; legitimate one-to-many legs are aggregated deliberately. |
| IO04 | P0 | **Feature order/type drift.** Remove/add/reorder a predictor, change numeric to string and change categorical codes. | Input schema either rejects the mismatch or explicitly restores the saved mapping. Silent misinterpretation is forbidden. Check pipeline behaviour, not a library version assumption. |
| IO05 | P0 | **Unknown or missing input.** Both unseen wallets, unseen asset/issuer, categorical missing value, all optional fields absent, NaN and extreme numeric values. | Documented fallback or clear rejection for invalid required data; finite valid prediction for supported target rows. Model-supported missing features need not be imputed to zero. |
| IO06 | P0 | **Wrong score channel.** Check class labels/order, probability column, raw margin, hard class output and selected iteration. | Larger submitted scores mean the documented suspicious target. Export scalar scores in [0,1], not class labels or unconverted margins. Do not clip invalid outputs as a substitute for fixing the path. |
| IO07 | P0 | **Saved-model disagreement.** Predict the same fixed feature matrix before save, after reload and in a fresh process using the final command. | Match within a declared environment-appropriate numerical tolerance; explain any rank/top-k change. Save preprocessing with the model. |
| IO08 | P0 | **Chunked inference changes context.** Build one canonical feature table, then score it whole, shuffled and in chunks. Separately compare whole-context versus chunked feature construction. | Estimator scores agree by hash for fixed features. Feature chunking must preserve intended history/state. Recomputing local degrees or ranks per chunk must not silently change the model's meaning. |

## 2. Ledger and economic-reconstruction cases

XRPL Payment metadata distinguishes requested from actually delivered value; a payment can convert currency, including back to its own sender. Transaction results and ledger ordering are separate metadata. Tokens have issuer identity and their own numerical representation; issuer transfer fees differ from XRP transaction cost. These checks are conditional on the released synthetic schema: do not invent absent fields or add support for unrelated real-ledger features. [Payment reference](https://xrpl.org/docs/references/protocol/transactions/types/payment), [partial payments](https://xrpl.org/docs/concepts/payment-types/partial-payments), [transaction metadata](https://xrpl.org/docs/references/protocol/transactions/metadata), [currency formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats), [transfer fees](https://xrpl.org/docs/concepts/tokens/fungible-tokens/transfer-fees)

| ID | Priority | Edge case and test | Required outcome / disposition |
|---|---|---|---|
| XR01 | P0 | **Million-fold amount error.** Use 1,000,000 drops, one-drop amounts, dust and large valid amounts; compare exact reconciliation with model values. | 1,000,000 drops = 1 XRP. Keep integer drops for exact residual checks and avoid overflow/unsafe float equality. |
| XR02 | P0 | **Same symbol, different issuer.** Construct two USD assets with different issuers and rates. | Asset/rate joins preserve the documented issuer+currency identity. Unknown issuer does not inherit another issuer's rate by symbol alone. |
| XR03 | P0 | **Two/three conversion legs.** Reconcile source paid, destination delivered and helper endpoints with raw legs. Shuffle physical leg order. | One economic payment edge; sequential legs do not double/triple volume or create extra independent laundering hops. Use documented leg order, not file position. |
| XR04 | P0 | **Partial receipt / SendMax ceiling.** On an allowed issued-token fixture, requested receipt is 100, actual delivered is 20 and spending ceiling is larger than actual cost. | Incoming flow uses actual delivered 20; outgoing paid value comes from supported evidence. Ceiling/request fields may remain separate features. |
| XR05 | P0 | **Failed/no-transfer target.** Include an eligible transaction with no completed payment evidence but a recorded fee/status. | It remains in the scored target spine; no invented completed transfer. Success comes from documented result semantics, not a Flags value or mere ledger inclusion. |
| XR06 | P0 | **Rate boundaries and holes.** Exercise exactly 10:00 UTC, just before 11:00, missing rate and duplicate asset/hour rate. | Join the containing documented UTC hour. Flag missing rates, reject ambiguous duplicates, avoid outside feeds or nearest-future substitution. |
| XR07 | P1 | **Decimal/JSON edge cases.** Equivalent valid token decimal/scientific formats, empty/null JSON, unexpected object shape and malformed numeric text. | Supported equivalent representations parse consistently. Invalid forms remain explicit; schema violations do not masquerade as zero or completed flows. |
| XR08 | P1 | **Fee/spread mismatch.** Compare same-asset forwarding, token fees, asset conversions and partial forwarding. | Conservation is a graded compatibility signal. Do not apply one fixed 10–15-drop tolerance to all assets and conversions. |
| XR09 | P1 | **Self-conversion.** Account = destination with legitimate asset exchange. | Representation retains the event but distinguishes atomic self-conversion from a multi-transaction laundering cycle. Review score in context; no automatic whitelist. |
| XR10 | P1 | **Tags: null, zero and shared numbers.** Same destination tag at different service addresses; optional field missing versus zero. | Do not collapse missing and zero or assert global customer identity. Tags are address-scoped, service-defined context if present. |

Do not prioritise the legacy `delivered_amount = unavailable` case unless the schema or timestamps support it. The official documentation places that special partial-payment case before 20 January 2014; it is not a reason to complicate a fictional modern seven-day dataset. MPT/NFT/AccountDelete cases likewise become required only if relevant records appear. [Partial-payment historical caveat](https://xrpl.org/docs/concepts/payment-types/partial-payments)

## 3. Leakage and graph-state checks

The central principle is information availability: learned preprocessing must respect its training split, while unlabelled graph visibility depends on the authorised batch/causal setting. Do not infer either setting solely from a static leaderboard. The graph-specific neighbour test below is our synthesis, extending ordinary split hygiene. [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls)

| ID | Priority | Edge case and test | Required outcome / disposition |
|---|---|---|---|
| LG01 | P0 | **Held-out-label intervention.** Freeze model/configuration; delete or permute only validation labels, then rebuild validation features and predictions, with caches invalidated. | Features and predictions are identical; only evaluation changes. Training labels remain available as permitted. Never reuse this modified evaluation to select the model. |
| LG02 | P0 | **Leakage hidden in fitting or caches.** Audit imputers, feature selectors, target encoders, category statistics, neighbour-risk tables and cache provenance. | Learned transforms exclude held-out labels/data as required. Cache keys include source/fold/context/configuration; stale full-data features cannot enter a split. Distinguish permitted unlabelled batch context from forbidden labels. |
| LG03 | P0 | **Future append in causal mode.** Append strictly later events, rebuild features for earlier targets and compare. | Earlier causal features/predictions remain unchanged. In batch mode the same invariance is not required; document that separate context contract. |
| LG04 | P0 | **Inductive node holdout uses full-graph features.** Remove selected nodes and incident transactions, rebuild the training context; add validation context only as the chosen experiment permits. | Features reflect the declared graph visibility. A removed node with incident events still embedded in training aggregates invalidates an inductive claim. Wallet overlap alone is not proof of leakage. |
| LG05 | P0 if used | **Neighbour OOF scores leak indirectly.** Trace the training-label provenance of every neighbour score used by an outer-validation edge. | Its outer held-out labels are excluded from all underlying first-stage fits. Use inner cross-fitting for stage-two training; otherwise drop the stage or withdraw the claimed validation. |
| LG06 | P1 | **Scenario/pair copies across folds.** Audit exact duplicates, near-identical motifs, split scenario IDs, repeated pairs and the largest hub-connected components. | Record overlap and evaluate a representative held-out-group stress test. Do not automatically delete legitimate repeated payments or assume components equal scenarios. |
| LG07 | P1 | **Random row score versus temporal/entity score.** Compare fixed random, proposed five-day/two-day temporal and feasible wallet/pair-disjoint checks. | Explain differences; choose the primary check by actual inference setting. No universal requirement that disjoint and random AP match. Same-wallet future payments may be legitimate test cases. |
| LG08 | P2 | **Label permutation diagnostic.** If time permits, retrain a small end-to-end variant using permuted training targets under the same split and evaluate untouched real validation targets. | Look for repeated above-chance behaviour suggesting leakage or artifacts. Judge against a permutation/null distribution and prevalence, not an invented exact AP value or one run. It is a diagnostic, not a complete leakage proof. |

For graph storage, parallel payments must remain distinct even when sender and recipient are identical. A directed multigraph is one suitable representation. Enumerators may count motifs differently from transaction multiplicity, so verify the definition with known-answer fixtures rather than assuming a graph library's count matches your feature. [NetworkX MultiDiGraph](https://networkx.org/documentation/stable/reference/classes/multidigraph.html)

## 4. Behavioural cases that aggregate AP can conceal

These tests inspect **transactions**, not permanently criminal wallets. A real-world-looking shape is not proof of intent. Define genuine correctness fixtures for structure, but use holdout labels to judge the actual classifier. Handwritten plausible scenarios do not establish the organiser's labels or a monotonic relationship between one signal and suspicion.

| ID | Priority | Cohort / perturbation | What to inspect |
|---|---|---|---|
| BH01 | P1 | **Legitimate hubs:** payroll fan-out, merchant fan-in, exchange gather-scatter, treasury split/merge and refunds. | Highest-ranked false positives and the fraction of the top 402 consumed by each cohort. Trace contextual differences rather than whitelist all hubs. |
| BH02 | P1 | **Mixed wallet:** one account has normal payments and scenario-labelled suspicious payments. | Can edge-specific features distinguish them? Check over-propagation to the wallet's unrelated transactions. If features are identical, acknowledge the representational limit. |
| BH03 | P1 | **Cold start:** unseen sender, unseen recipient, both unseen, new pair and unseen issuer. | Support counts, fallback usage, AP where labels exist, score distribution and top-k misses. Never collapse every unseen address into one shared graph node. |
| BH04 | P1 | **Incomplete history:** first/last day, clipped path, funded before observation and exit after the window. | Separate observed net flow from balance; first observed from newly created. Crop context and measure degradation, not forced score invariance. |
| BH05 | P1 | **Non-template shapes:** change fan width, chain length, duration, overlap and counterparty turnover. | Check blind reliance on four-to-seven senders or exactly five-minute windows. Synthetic fixture definitions are illustrative, not proof of hidden-test effectiveness. |
| BH06 | P1 | **Amount tails and ambiguity:** tiny repeated loops, high-volume routine payments, slow partial forwarding and cross-asset returns. | Look for absolute amount/asset exemptions and over-sensitive conservation assumptions. Do not require scores to rise monotonically with amount or hop count. |
| BH07 | P1 | **Static versus chronological cycle:** preserve topology, vary supported event order; include atomic conversion legs separately. | Static participation may remain while chronological participation changes correctly. Do not demand the same final score if the model intentionally uses both. |
| BH08 | P1 | **High-degree hub / giant component.** Run worst observed hub, dense pair activity and overlapping routes. | Peak memory, total feature runtime, capped searches, capped-row fraction and cohort ranking. Search truncation must not be silent evidence of no motif. |
| BH09 | P2 | **Structure-preserving renaming.** Bijectively rename non-semantic fixture wallet IDs and transaction hashes across all tables, preserving roles and joins. | Structural features should be invariant. For models explicitly using identity, investigate expected changes instead of declaring all changes bugs. Do not rename fixed currency issuers used by rate joins arbitrarily. |
| BH10 | P2 | **Equivalent input representation.** Permute physical rows, JSON key order or documented equivalent decimal encodings without changing event order/economic values. | Canonical features remain equivalent. Existing estimator predictions follow saved schema/tolerance. Time-dependent features cannot use incidental file order as chronology. |

## 5. Ranking, statistics and final artifacts

AP is a threshold-weighted precision-recall summary, not trapezoidal PR-AUC. It requires the correct positive target and score direction. The competition's source of truth is its scorer; use scikit-learn as an explicitly checked implementation. [Average Precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

| ID | Priority | Edge case and test | Required outcome / disposition |
|---|---|---|---|
| RK01 | P0 | **Scorer parity and ties.** Use tiny perfect, reversed, tied and constant-score examples, plus the deck's two-positive example. | Match the official AP/tie convention and positive label. For all-tied constant scores and both classes, AP equals prevalence; never implement a tie-breaking rank formula that invents information. |
| RK02 | P0 | **Export changes ranking.** Compute metrics and rank groups on in-memory versus CSV-reloaded predictions. | Preserve finite bounded scores and adequate decimal precision. Investigate new ties, clipping, rounding and class-score inversion; do not add random jitter to manufacture AP. |
| RK03 | P1 | **Misleading slices and folds.** Evaluate pattern/difficulty cohorts, folds and pooled OOF predictions with support/prevalence. | Report positives/negatives. All-positive or no-positive subsets cannot establish useful discrimination. Compare models within identical cohorts; pooled AP and mean fold AP answer different questions. |
| RK04 | P1 | **One scenario dominates improvement.** Inspect scenario contribution, worst meaningful cohort and top-402 precision/recall; use group-aware uncertainty if enough independent groups exist. | Final candidate improves the predefined comparison with understood regressions. No made-up universal AP delta; no independent evidence if only one positive group exists. |
| RK05 | P1 | **Ensemble degrades best member / unstable top-k.** Compare final blend versus strongest member on identical held-out rows and inspect rank-boundary ties. | Keep a blend only with evidence; seed diversity and rank averages are not automatic gains. If inference normalises ranks, use the intended whole target universe, not independently per chunk. |
| OP01 | P0 | **Cache-free reproducibility.** Run the actual inference command from raw inputs in a fresh process without notebook variables, target labels or old graph caches. | Complete predictions, declared runtime/memory and numerical/rank agreement within declared tolerance. Stateful graph processors must reset between splits/runs. |
| OP02 | P0 | **Wrong package or incomplete run.** Open the final ZIP and CSV as the organiser loader would. Verify file size, paths, dependency manifest, model files or retraining path and version hashes. | Deck limits: CSV ≤12 MiB; code ZIP ≤8 MiB; methodology under 10 slides or 4 pages. No installed package directories or secrets. Reconcile release updates; verify completed scoring rather than just upload success. |
| OP03 | P1 | **Stale model/feature pair.** Compare feature-schema hash and preprocessing version with the exact saved artifact and submitted commit. | The final package must be the model evaluated, not a similarly named checkpoint. Archive selected iteration, model/CSV hashes, input fingerprints and uncertainty/limitations. |

## Three concrete tests I would run first

### A. Wrong-ID test

Predict a small set, permute its rows, restore predictions by hash and compare the hash→score mapping. Then permute labels independently and evaluate only after a checked hash join. If this changes results, stop: the model's reported performance or exported output is not reliable.

### B. Forbidden-label test

With the chosen model/configuration frozen, change only outer-validation labels. Force rebuilding of validation features and neighbour-risk inputs. Predictions must not change. This is a narrow intervention: if you intentionally retrain on changed training labels, predictions are expected to change, and that does not constitute leakage.

### C. Chunking test

Compute graph features once using the intended full context, then compare inference on 1, 10 and all-row chunks. With fixed features, a tree estimator should return equivalent per-row results. If chunked feature building changes context, preserve the correct state/history or explicitly forbid that inference mode. Do not claim chunk-invariant predictions for a graph pipeline whose allowed context was actually changed.

## Suggested sign-off order

These are planning allocations, not measured runtime guarantees. A clean raw-to-output run should start early and continue while other checks run.

1. **First 30 minutes:** start a clean inference run; verify ID sets, score channel, feature schema, row/cardinality checks, CSV reload and saved-model parity. Inspect a representative conversion and missing-data row.
2. **Next 30–60 minutes:** forbidden-label/caching checks, graph context/cutoffs, conditional nested stacking audit, chunking/state, scorer parity and top-ranked held-out errors.
3. **Remaining robustness budget:** unseen endpoints/assets, mixed wallets/hubs, temporal and entity stress splits, boundary clipping, exact motifs and high-degree runtime. Avoid broad tuning that consumes the independent confirmation set.
4. **Final reserved hour:** freeze candidate, rerun changed checks, verify package and organiser acceptance. If a late change fails a blocker, use the verified fallback rather than improvise test predictions.

Recommended sign-off evidence consists of:

- A machine-readable test ledger with test ID, applicability, status, artifact path and issue/disposition.
- Hash-aligned held-out predictions and an exact-AP report with class support, cohort results, fold/pooling definitions and top-k diagnostics.
- Reconciled raw→canonical examples for plain payment, conversion, partial receipt if present, and missing/failed flow if eligible.
- A feature/label provenance description and saved split assignments.
- A cache-free final run record and exact submitted artifact hashes.

**Do not sign off** a model with known ID corruption, silent feature-schema mismatch, unsupported score output, held-out-label leakage or irreproducible final inference. A high leaderboard score does not waive these failures. Behavioural weaknesses may be acceptable for a bounded hackathon submission if documented and if they do not conceal a correctness defect; they do not establish readiness for real AML decisions.

## Evidence coverage and source limitations

This review builds on the local [Strategy3](../../Strategies/Strategy3.md), [research report](../XRPL%20strategy%20research.md) and [training deck](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx). Fresh internet research used official XRPL, scikit-learn, pandas, NetworkX, CatBoost, LightGBM and TensorFlow documentation plus Google Research. TFDV distinguishes schema problems, training/serving skew and drift; those are useful categories here, not a requirement to install TensorFlow for a tree model. [TFDV guide](https://tensorflow.github.io/tfx/guide/tfdv/)

The direct web reader could not extract the XRPL ledger-close-time and robust-payment-monitoring pages in this pass; no unsupported subsecond timing or exotic transaction behaviour is assumed from those failed reads. Available Payment and transaction-metadata documentation grounds the relevant checks. Installed package versions were not established, so verify API switches against the actual environment before implementing them.

The companion [test ledger](XRPL%20final%20model%20test%20ledger.csv) contains the same 44 checks, initially NOT RUN. It records a test plan, not test results.
