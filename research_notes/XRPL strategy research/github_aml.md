# GitHub AML implementations, graph blind spots, and practical experiments

## Which implementation ideas genuinely strengthen the existing strategy?

### Takeaway
Keep boosted trees plus engineered directed transaction features as the main path. The valuable research transfer is stronger feature semantics and validation, rather than adopting a full GNN repository the day before submission. This is a recommendation, not a measured improvement on the organiser dataset.

### Cited Findings
- IBM's author-maintained Multi-GNN repository implements GIN, GAT, PNA, and RGCN with edge updates, reverse message passing, ego identifiers, and port numbering; its README declares Apache 2.0. Retrieved through the connected GitHub tool, including actual source rather than search snippets. — [README](https://github.com/IBM/Multi-GNN/blob/main/README.md)
- Its data utility computes separate incoming/outgoing ports and interarrival deltas. Ports order distinct counterparties by earliest observed interaction; they are not simply transaction sequence numbers. — [data_util.py](https://github.com/IBM/Multi-GNN/blob/main/data_util.py)
- The authors' theory concerns suitably powerful directed multigraph networks with the combined adaptations; it does not establish that any ordinary GCN learns every AML motif or that a small trained model achieves that guarantee. — [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/29069/30025)
- The example training pipeline constructs temporal train/validation/test graph snapshots, uses a constant placeholder node feature, and normalizes each snapshot's edge features separately. These are observed implementation choices, not prescriptions to copy. — [data_loading.py](https://github.com/IBM/Multi-GNN/blob/main/data_loading.py)
- The supplied environment includes CUDA 11.8 and PyTorch Geometric compiled packages. Setup is materially heavier than pandas plus a tree library; no installation or runtime benchmark was performed here. — [env.yml](https://github.com/IBM/Multi-GNN/blob/main/env.yml)
- GFP's paper ablations add fan features, then multi-hop motifs, then vertex statistics, with further gains at each stage on its benchmarks. Its reported metric is minority-class F1, not this competition's AP; no score improvement can be transferred numerically. It explicitly distinguishes full-batch offline features from streaming visibility. — [paper sections 4 and Table 5](https://arxiv.org/html/2402.08593v2)

### Inferences
- P0 feature bundle: preserve the directed **multigraph** edge table with a unique row identifier. A simple NetworkX DiGraph silently merges repeated source/target payments and can erase structuring, repeated-payment, and timing evidence.
- Add source and destination, incoming and outgoing summaries separately; for each use unique counterparties, edge count, fan/degree ratio, amount median/IQR, log totals, asset-conditioned amount variability, interarrival median/IQR, first/last activity, reciprocal edge counts, and fraction of new counterparties. These represent behavior rather than wallet identity.
- Port rank and elapsed time since previous transaction should be cheap ablations. Also include normalized rank and a seen-before indicator: raw port number grows with available history and can accidentally identify the train/test window.
- Preserve two feature views: (a) cheap account/pair/asset statistics; (b) bounded edge-participation motif features. Avoid spending the entire budget enumerating large cycles before testing these basics.
- Each scored transaction needs its own motif-role features. Being next to an illicit pattern does not establish participation in it. Source fan-out, intermediate forwarding, sink fan-in, and cycle closure are different roles.
- Prioritize an implementable 2–4 hop bounded search with short/medium/long windows over unbounded cycle enumeration. Log the truncation/degree-cap indicator; otherwise a capped count resembles a genuine low count.
- Full GNN only after a sound baseline and environment smoke test; benchmark against the same folds and visibility, and retain only if AP gain is stable. Directed message passing and edge updates matter more than selecting a fashionable architecture name.

### Gaps
- No organiser data or generator code was provided to this research task; wallet overlap, label prevalence, motif durations, and numerical benefit remain unmeasured.
- Neither published F1 nor model expressivity proves a gain in this XRP hackathon.

## What edge cases and unsafe assumptions need correction?

### Takeaway
The highest-risk blind spots are hub contamination, forcing all transactions through a single homogeneous graph, mistaking static connectivity for actual flow, and treating a wallet's behavior as a permanent label.

### Cited Findings
- AMLSim contains an explicitly normal fan-out transaction implementation. Its normal model sends successive payments to neighboring accounts; fan shape alone is not an illicit label. — [normal FanOutTransactionModel.java](https://github.com/IBM/AMLSim/blob/master/src/main/java/amlsim/model/normal/FanOutTransactionModel.java)
- Its CycleTypology implements both ordered random-interval scheduling and an UNORDERED case that does not sort the steps. Therefore even an established synthetic AML implementation can label static cycles without requiring chronological circulation. This is evidence about external AMLSim only, **not evidence the organiser uses AMLSim or this scheduling behavior**. — [CycleTypology.java](https://github.com/IBM/AMLSim/blob/master/src/main/java/amlsim/model/aml/CycleTypology.java)
- The AMLworld paper describes transaction-level labels, legitimate/illicit fund behavior, and laundering beyond predefined graph shapes. AMLworld and AMLSim are distinct generators with different modeling scope. — [sections 3.3–3.4](https://arxiv.org/html/2306.16424v3)
- AMLSim's README warns that its wiki can lag implementation and advises its master branch. Its Java 8 and older Python/NetworkX dependencies make installing the generator poor use of a short hackathon budget. The actual license file is Apache 2.0. — [README](https://github.com/IBM/AMLSim/blob/master/README.md); [LICENSE](https://github.com/IBM/AMLSim/blob/master/LICENSE)

### Inferences
- Build both static motif features and time-respecting versions; do not delete static evidence solely because a plausible economic flow cannot be reconstructed. Compare their marginal contribution on held-out data.
- A time-respecting 2-edge transition requires shared intermediate wallet, correct direction, increasing timestamps, compatible asset/value semantics, and bounded lag. For equal timestamps retain an ambiguity indicator rather than inventing an order from row position or a lexical transaction hash. If actual ledger transaction order is provided, use that documented order.
- A transition's nominal amount match should allow split/gather totals and partial onward transfers. Exact equality is brittle. Use relative error, lag, asset compatibility, and (where organiser data supports it) conversion-rate/spread/fee allowance as **features**, not deterministic labels.
- Model raw transfer/payment topology separately from economic payment or conversion relations when the supplied schema permits that distinction. Converting currencies through a legitimate liquidity provider can create false cycles and hubs. A generic wallet projection that fuses everything can produce fictitious laundering chains.
- Keep high-degree hubs in the data. Instead of universally removing exchanges/payment processors, add behavior-conditioned features and degree-normalized motif statistics; audit both capped and uncapped local evidence. Illicit transactions can use legitimate hubs.
- Neighbor score aggregation should be role-specific and time/asset gated. A global maximum across all adjacent edges is easily contaminated by one high-scoring transfer at a busy exchange.
- Components may collapse into a giant component through innocent infrastructure. Before GroupKFold, inspect component size distribution and number of positive-bearing groups. If one group dominates, component grouping is not a usable validation scheme. A temporal primary split plus wallet/pair-disjoint stress split is more informative; if scenario IDs exist, use them without exposing them as a target shortcut.
- Window boundaries cause censoring: first-day wallets may have missing prior history, and last-day transactions may lack a future closing edge. Add time-to-window-start/end, observed exposure duration, and history-count indicators; test performance on boundary rows independently.
- Additional fixtures: legitimate payroll fan-out; merchant fan-in; round trip refunds; parallel identical payments; self-transfer; repeated counterparties; conversion loop; split and merge with imperfect conservation; a laundering transaction embedded in an otherwise legitimate hub; disconnected single-edge fraud; partially visible cycle; overlapping motifs; missing asset/amount; very large decimal values; shuffled input row order.

### Gaps
- No proof the organiser labels funds, motif edges, wallet actors, or scenarios consistently with any public generator. Read organiser README before adopting semantic assumptions.
- Whether raw LP/gateway accounts are identifiable or payment paths are supplied must be checked in actual schema. Do not infer XRP production semantics from a synthetic field name.

## How should validation and the final one-day experiment plan change?

### Takeaway
Do not promote stage-two neighbor-score features using a single global OOF vector. Protect an outer holdout, match feature visibility to the authorised inference task, and choose the smallest stable AP improvement.

### Cited Findings
- Official sklearn stacking trains its meta-model on cross-validated base predictions and warns against training a stacking model on in-sample base predictions. This supports out-of-fold discipline but does not by itself solve graph-neighbor contamination. — [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- AP is a recall-weighted sum of precision at distinct score thresholds, accepts continuous decision scores, and differs from trapezoidal PR-AUC. — [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- GFP's official API is stateful: transform inserts input edges, fit clears state, and fit_transform is fit followed by transform on the same edges. An arbitrary batch call is not a guarantee of causal event-by-event features. It supports motif histograms and per-vertex statistics; configuring a time window determines retention. — [GFP API](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html)
- Snap ML is **not Apache-licensed because IBM's other repos are**: current PyPI metadata names IBM's International License Agreement for Non-Warranted Programs and classifies it as Other/Proprietary. Its 1.17.2 distribution lists macOS x86_64 wheels and Linux wheels, with no macOS arm64 wheel shown. Documentation requires OpenMP and says GFP is unsupported on Windows. — [PyPI release metadata](https://pypi.org/project/snapml/); [installation docs](https://snapml.readthedocs.io/en/latest/installation.html)

### Inferences
- **Graph OOF leakage example:** let A be an outer-validation edge and B its neighbor in a different ordinary OOF fold. The model that produced B's OOF score may have trained on A's label. Aggregating B's score into A's features therefore leaks A's label even though A's own score was OOF. Excluding only the central edge is insufficient.
- **Correct evaluation:** for each outer split, exclude all outer-validation labels from every stage. Within outer training, inner-cross-fit the base model and generate score features for the stage-two training rows. Fit the base model on all outer training to score outer-validation/context edges. Create validation neighbor features exclusively from models that never saw outer-validation labels. Evaluate the entire two-stage pipeline on untouched outer-validation labels. Keep context topology only if it matches authorised test visibility. For the smallest budget, omit stage two until this can be implemented and audited.
- **Optional final-fit alignment:** compare inference ensembles trained on inner-fold subsets versus the base model trained on all training data, because an OOF-trained stage two can receive better-calibrated full-data scores at inference. Use outer evaluation to choose; never train stage two on in-sample neighbor predictions and call it cross-fitted.
- **Visibility contract:** if complete seven-day test topology is authorised, offline within-window unlabeled edges can be legitimate. The same visibility must be mirrored in validation. If deployment is online, future topology must be excluded. Maintain two explicit feature builders/configurations so this choice is reviewable. A hidden label is forbidden in either mode.
- **GFP recommendation:** prefer reimplementing a narrow set of documented statistics and bounded patterns with existing local packages. Reuse Snap ML only after checking competition library rules, exact package terms, a compatible wheel, and a tiny known-answer test. Do not copy assumptions from documentation repository licenses. This is a practical dependency choice, not a request for a legal opinion.
- **Cheap validation priorities:** one primary split chosen from organiser generation setup, one temporal or wallet-disjoint stress split, per-fold positive counts, and paired changes for all ablations. Keep a final untouched holdout to reduce repeated-tuning selection bias. If there are few scenarios, report scenario-level uncertainty, not a false narrow interval from independent transaction bootstrap.
- **Metrics:** select checkpoints and ensemble weights by actual AP. Binary cross-entropy is an appropriate training surrogate but a lower log loss does not ensure a better ranking. Report AP plus precision@K only as a diagnosis of top-list errors. Pooled OOF AP and mean-fold AP answer different questions when fold score scales differ; inspect both instead of silently treating them as interchangeable.
- **Rank ensemble correction:** any strictly increasing transform preserves one model's order and tie structure, but averaging ranks can change order and introduce ties. Identical seed rankings cannot gain AP through rank averaging. Different seeds may help through diversity; it is an experiment, not a tie-breaking guarantee. Keep full float precision, no arbitrary jitter, and no global rounding. Compare raw probability averaging, normalized rank averaging, and the strongest individual model on the same held-out rows.
- **First 2 hours:** schema/labels/order audit; baseline AP; visibility contract; wallet/component overlap report; submission ID integrity.
- **Hours 2–5:** cheap directed per-wallet, pair, asset and timing summaries; one fixed tree configuration; paired AP ablation and hard-negative audit.
- **Hours 5–9:** bounded static and temporal motif participation; value-error and conversion-aware transition features where schema allows; degree/hub ablations; boundary slices.
- **Hours 9–12:** optional properly nested stage two only if time permits; otherwise two complementary tree feature views. Small coarse ensemble weight grid on a tuning fold, confirmation on untouched holdout.
- **Remaining time:** freeze code/seed/config, train final models, verify exact test IDs and finite continuous scores, hash artifacts and dry-run submission. No external AML datasets are training inputs; public code/papers are methodological references only.

### Gaps
- All gain claims remain hypotheses until organiser-only experiments run. This note does not identify a proven best model or expected leaderboard score.
- The exact competition rule on use of unlabeled full test topology, third-party packages, and any external generated augmentation still needs organiser confirmation from supplied rules; do not assume permission from general ML practice.
