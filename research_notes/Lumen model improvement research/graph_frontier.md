# Graph and temporal improvements beyond the active ego-network experiment

## What information is genuinely missing after V4?

### Takeaway

The strongest next direction is to distinguish coherent, unusual transaction evidence from routine exposure to busy wallets. The active ego experiment already addresses short-range size/dilution; repeating larger ego counts, training longer, or blindly propagating high scores is unlikely to answer the remaining question. This is an inference from the completed local ablations and the source review below, not a measured improvement.

### Cited Findings

- Local source review: `motif.components` links consecutive events at either shared wallet whenever each gap is below D, then takes an undirected connected component. D is a **link-gap threshold, not an upper bound on the episode's total duration**. `flow_components` additionally checks forward time and an outgoing/incoming delivered-value ratio between 0.05 and 1.05, but again returns undirected transitive components. Existing features summarize component size, role counts, conservation, shape, amount CV and event position. — [Current motif.py](/Applications/Softwares/Lumen.ai/solution/motif.py)
- Local source review: `context.py` already includes 1/6/24/72-hour maximum, mean, count, high-count and second-highest neighbour percentiles; 6/72-hour upstream/downstream aggregates; whole-week endpoint means/maxima; and the transaction's own percentile. The optional same-burst means/maxima/high counts were already tried. It does **not** expose high-score fraction/enrichment against exposure, distinct high-score counterparties, concentration of support by counterparty, or amount-constrained directional score messages. Week endpoint maxima and means also include the current event. — [Current context.py](/Applications/Softwares/Lumen.ai/solution/context.py)
- Local source review: V1 already contains strict before/after endpoint window totals and CVs over 5 min/1 h/6 h/24 h/72 h; nearest/best amount matches; sender/receiver global uniqueness and entropy; repeated-pair timing and value CV; pair history flow-ratio median/CV; static cycle/parallel-path counts. These are not newly proposed features. Rolling window code allocates a `unique` array without using it, and does not emit rolling counterparty-set turnover. — [V1 features.py](/Applications/Softwares/Lumen.ai/solution/features.py)
- Local source review: active `ego_features` caps all reachable events within +/-2/5/10 min of the centre, allows at most two hops, and emits reach size/wallet count/value CV/value ratio. This is useful bounded context, but it does not count ordered direction-specific paths or robust repeated flow episodes. `VALUE` in the projected customer graph is reference USD, not mixed token units. — [motif.py](/Applications/Softwares/Lumen.ai/solution/motif.py), [payments.py](/Applications/Softwares/Lumen.ai/solution/payments.py)
- Completed local results: motif base mean fold AP 0.59773; context-on-motif 0.61143; rejected flow-components 0.59592; long training 0.59056; same-burst context 0.60619. The V4 rank blend is reported in the supplied log as mean 0.6162, pooled 0.6153, top-402 TP 355, top-1250 TP 785. These are validation results; the attachment reports a portal V4 score of 66.5, which was not independently authenticated in this research. — [motif metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_metrics.json), [context metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_context_metrics.json), [flow metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_flow_rejected_metrics.json), [long metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_long_metrics.json), [burst metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_context_burst_metrics.json), [supplied log](/Users/shivamtiwari/.codex/attachments/a8c7f759-9039-41cc-9bfe-1176933a3a5b/Pasted%20text.txt)
- Primary temporal-motif research defines motifs using ordered directed events and a bound on **total event span**. A sequence with individually short gaps can fail that total-span requirement. — [Paranjape, Benson and Leskovec, WSDM 2017 project and definition](https://snap.stanford.edu/temporal-motifs/index.html)
- IBM's Graph Feature Preprocessor uses per-edge graph patterns and direction-specific endpoint statistics with configurable temporal windows. This supports trying graph features with gradient boosting before replacing the learning architecture; its published results do not establish a gain on this competition. — [GFP official documentation](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html), [GFP primary paper](https://arxiv.org/html/2402.08593v2)
- Multi-GNN explicitly separates reverse message passing, port numbering and ego IDs; its port numbering distinguishes counterparties using chronological local first occurrence. Its expressivity results concern those adaptations together, not a claim that arbitrary ego counts or raw wallet identifiers solve AML. — [AAAI 2024 primary paper](https://arxiv.org/html/2306.11586v3), [official IBM implementation](https://github.com/IBM/Multi-GNN)

### Inferences

- The existing flow-component experiment's failure is evidence against that broad transitive formulation; it is not evidence that all temporal flow information is useless.
- Maxima saturate under exposure: if independent neighbour ranks were uniform, the expected maximum of n neighbours would be n/(n+1). Independence is unrealistic here, but the illustration explains why a busy wallet naturally has a near-one maximum without being suspicious.
- Existing observations support continuing feature-level CatBoost ablations. A new GNN stack would introduce dependencies, sampling choices and validation work while leaving the observed dilution and legitimacy questions unresolved.
- Do not say the remaining leaderboard gap “must be below the top 1%.” Equal rounded precision/recall bars do not imply equal ordering inside that budget; AP depends on those ranks too. A single matching public-test improvement is supportive evidence for that change, not proof that later model selection is safe.

### Gaps

- No graph or temporal proposal below has been implemented or measured by this research agent.
- The completed metrics inspected here contain grouped-fold results; a future context candidate still needs its own wallet and temporal checks. Existing V1 checks do not automatically transfer.
- It is not established that lawful and suspicious versions of any specific bipartite or relay scenario are indistinguishable. Similar inspected marginals are insufficient to prove observational equivalence.

## Which small, falsifiable ablations should be tried next?

### Takeaway

Predeclare at most three new feature blocks, evaluate each against the frozen V4 recipe on exactly the same folds, and test a combination only if individual evidence warrants it. First choice is exposure-corrected context; second is ordered bounded path support. Multi-day recurrence is the most relevant remaining hard-relay hypothesis, but may be a later task if there is insufficient time for holdout and reproduction checks.

### Cited Findings

- Temporal pattern counts are encoded per edge in GFP, including fan-in/out, scatter-gather and temporal/simple cycles; configured windows control which graph history participates. Directional source/target statistics remain separate. — [GFP official feature definitions](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html)
- FlowScope models complete flow through a multipartite graph rather than relying on dense subgraph shape alone. This motivates using directed flow evidence, but its account-level published performance is not a prediction of transaction AP here. — [FlowScope, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5906)
- MonLAD/MonLAD-W tracks observed inflow-minus-outflow residual and repeated effective inflow/balancing events. The paper explicitly states initial account state is unknown and residual can be negative; residual is not observed bank balance. — [MonLAD, WSDM 2022, sections 4.1–4.2](https://shenghua-liu.github.io/assets/pdf/papers/wsdm2022-monlad.pdf)
- OddBall identifies deviations from relationships between local graph size and weight, rather than treating raw size as anomalous by itself. This is general anomaly-detection evidence; it does not establish that unusual behaviour is laundering. — [OddBall primary paper](https://www.cs.cmu.edu/~mmcgloho/pubs/pakdd10.pdf)
- Strength-preserving null-model work controls for single-node local characteristics when judging motif over-representation. Its application is information-flow networks, so a fast AML feature inspired by it is an engineering inference, not its validated method. — [Pagnottoni and Spelta primary paper, section 2.3](https://link.springer.com/article/10.1007/s10260-023-00724-y)

### Inferences

#### A1 — Exposure-corrected, leave-one-out score support

**Question:** Does a payment have an unusual *concentration* of high-scoring contextual evidence, or does it merely touch a busy wallet?

For each event e and directional neighbourhood N(e,D), exclude e, de-duplicate events touching both endpoints, and compute:

- n = eligible neighbour count; K = count with cross-fitted percentile >=0.97.
- Smooth high-score fraction H = (K + 5p)/(n + 5), p=0.03. Retain n separately.
- Heuristic excess Z = (K - np)/sqrt(np(1-p)+1). This is **not a calibrated z-test**, because neighbours are dependent.
- Count distinct counterparties supplying high scores; support concentration C = sum over counterparties of (high_count_c / max(K,1))^2. A single high-scoring lawful burst should not appear as many independent corroborating sources.
- Exponentially time-weighted mean high-score excess, sum(w_j*(I[pct_j>=0.97]-p))/max(sum(w_j),1), w_j=exp(-abs(t_j-t_e)/D). Use one predetermined D at 1 h and 6 h, and separate upstream-before and downstream-after support.
- Leave-one-out whole-week endpoint mean and maximum to remove a mechanical echo of the current score. Self percentile already exists, so do not mistake self-score duplication for graph gain.

**New information:** relative concentration, independent-support breadth and per-event exposure correction. Current code has n and K individually but does not explicitly supply these ratios/residuals; trees can approximate interactions, so this may still be redundant. State that uncertainty before trying it.

**Mechanism:** lift medium cases surrounded by a concentrated weak/high evidence pocket while reducing busy-lawful hub propagation. Replacing max with corrected support is a cleaner ablation than just appending dozens of correlated maxima.

**Failure modes:** lawful bipartite/chain episodes may produce many mutually high model scores. Those are correlated errors, not corroborated ground truth. A high fraction in a two-event neighbourhood is unstable; smoothing and n prevent treating it like a large pocket. Restrict exactly how many thresholds are tried.

**Validation cost:** needs the existing nested score protocol, not one global OOF array reused naively inside outer-fold validation. With reusable valid outer-safe score caches, only context aggregation and three stage-2 fits per seed are new; otherwise six first-/second-stage fits are required. No new graph engine or neural dependency. Do not promise wall time without measuring.

#### A2 — Anchor-bounded, ordered flow and motif support

**Question:** Is the centre event part of a coherent ordered route, rather than an unrelated branch in an unbounded burst?

For e=(s,r,t), build a small event DAG of candidate payments within a fixed **total** span D. Start with D=30 min and 120 min; do not scan many scales. A route a->s, s->r, r->b must satisfy t_in < t < t_out and t_out-t_in<=D. Use ledger index / transaction index for same-timestamp ordering only where the released ledger semantics permit it. Keep same-time/ambiguous-order counts separately rather than inventing order from hashes.

Emit a small set of continuous features:

- Ordered upstream/downstream support counts, plus count of complete two-sided three-edge routes containing e.
- Directed route count divided by the count of available upstream × downstream candidates, with explicit zero-denominator flag.
- Best and median route amount mismatch at each intermediary, using received reference USD for incoming events and **actual sender paid reference USD** for outgoing events; output lags and errors rather than a broad pass/fail link.
- True return-to-source temporal-cycle support of length 2/3 inside D, excluding repeated intermediate wallets. Keep it separate from chain support.
- A route's current-event role (start/middle/end), retained value fraction and observed incoming/outgoing exposure.

Bound neighbour candidates at e.g. 32 per endpoint per window with deterministic temporal selection and emit cap flags. For a fan-in/out, use sums over a fixed contiguous, time-bounded group where appropriate rather than asserting one incoming payment funds every outgoing payment. A one-to-one amount match is evidence of compatibility, not identified funds.

**New information:** strict direction/order, total-span boundedness, and route consistency at multiple intermediaries. Active ego has only reach counts and value summary; V1 has one-hop nearest matches, not route-level joint coherence. Existing flow-component features were transitive and allowed a broad 0.05–1.05 edge ratio, so this is a distinct hypothesis.

**Mechanism:** recover low-ranked medium/slow easy patterns without allowing a busy wallet to merge 90 minutes of unrelated activity into a “10-minute” component.

**Failure modes:** lawful payroll, settlement and commerce can form the same motifs. More exact motifs can sharpen false positives as readily as positives. Reference USD conversion, execution spread, pre-existing balances, split/merge paths and window censoring limit conservation claims. Tight amount cutoffs may erase the exact medium patterns being sought. Use continuous mismatch and exposure/cap flags; no pseudo-labels.

**Validation cost:** pure numpy/pandas bounded joins and three stage-1 fits if label-free features are added to the base model. If stacked, nested context evaluation is again required. Start with 10–20 features, not exhaustive all-path enumeration. Do not deploy a different inference feature semantics from validation.

#### A3 — Multi-day turnover and recurring observed-flow episodes

**Question:** Are apparently ordinary relay transactions repeated parts of a coordinated history that single-window counts cannot represent?

For each wallet and each endpoint role, construct 6-hour time bins relative to split start. Use received reference USD inflow I_b and sender-paid reference USD outflow O_b; track coverage rather than assuming no pre-window activity. Candidate features:

- Adjacent-block counterparty turnover: Jaccard(I_neighbours_b, I_neighbours_previous) and the corresponding outgoing set, plus fraction of new counterparties in current 6h versus previous 48h. Include set sizes and observation exposure.
- Recurring split/merge shape: how many distinct active bins have multiple incoming counterparties and outgoing count lower than incoming count, and vice versa; fraction of volume in each role, and robust CV of these ratios across active bins.
- Wallet observed-flow residual C(t)=sum(received_usd)-sum(paid_usd), **not balance**. After demeaning local initial C, extract episode range, duration, inflow/outflow count ratio, residual return error normalized by episode inflow, and number/dispersion of similar episodes during the week.
- Repeated-lag consistency: on populated windows only, compare incoming flow in bin b with outgoing flow in b or b+1, summarize normalized mismatch median and dispersion. Include active-pair count; zero-filled correlation over seven days is easily spurious.
- Optional pair-level recurrence of these episode roles rather than re-emitting V1 pair value CV or pair pre-flow ratio CV.

Thresholds should be relative to robust wallet scale or fixed observables, not learned from “hard” annotations. The classifier sees identical numeric definitions for every transaction. Keep all wallets inside each data split isolated from other splits.

**New information:** which counterparties recur/change and whether local flow episodes repeat with stable lag/role, beyond V1 totals, pair recurrence and one-window imbalance. This targets aggregate/distributed histories rather than short easy bursts.

**Mechanism:** a structured relay operating across days may be evident in repeated flow cycles even when each payment and short local graph look normal.

**Failure modes:** lawful recurring business can look more stable than laundering. Seven days yield few independent episodes; unknown initial balances and varying reference exchange rates break literal inventory conservation. If the cohort/scenario split leaves only six positive hard scenarios, transaction-level sample count exaggerates evidence. Evaluate by independent group, not only row bootstrap.

**Validation cost:** label-free preprocessing plus three base fits; no nested context necessary unless scores are added. Before fitting, audit finite values, episode counts, left/right censoring, and density/cap distributions without looking for generator artifacts. One fixed definition is a reasonable research candidate; a wide search over bin width, phase, thresholds and lags is not.

#### A4 — Conditional magnitude and role residuals, only if A1–A3 show redundancy

A small alternative is to encode deviation of local flow volume from the observed relation with degree/unique counterparties, using robust label-free conditional medians in a handful of exposure/degree bins. For example log1p(local_USD)-median(log1p(local_USD)|degree_bin,exposure_bin). Derive mapping from the permitted training context and apply frozen rules; never let label composition determine bins. This is a feature-engineering inference from OddBall, not its exact algorithm, and may be largely reconstructible by CatBoost from existing columns. It ranks below A1–A3 because it adds less obviously new information.

### Gaps

- None of A1–A4 has an established expected AP gain or guarantee of rescuing hard cases.
- Exact route candidate counts and wall-clock/memory cost are not measured. The 8 GB environment argues for bounded per-wallet arrays, not unrestricted Cartesian joins over 72 hours.
- The appropriate 6h/48h episode settings are hypotheses chosen for parsimony, not evidence of the hidden data's construction. Do not infer or reverse-engineer the competition generator.

## What can responsibly fit before 16:30 London, and what belongs later?

### Takeaway

Today, preserve the submitted V4/V1 bytes and let the active ego experiment finish. Then at most one or two targeted blocks can be investigated with the full existing promotion process; a new architecture or a hard-case rescue campaign without independent holdouts belongs later. Submission remains an explicit separate approval step.

### Cited Findings

- The local experiment harness validates three fixed outer folds and uses single-other-fold models to construct outer-safe context scores. The production runner builds stage-1 OOF train context, full-train test context, then equal-rank blends stages. These differing first-stage training sizes require checking distribution/generalization, especially for new score-derived features. — [experiment.py](/Applications/Softwares/Lumen.ai/outputs/v3_context/source/experiment.py), [run_v3.py](/Applications/Softwares/Lumen.ai/solution/run_v3.py)
- Existing local ablations already make a cost-effective case against more iterations and broad flow/same-burst context as the immediate next step. — [long metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_long_metrics.json), [flow metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_flow_rejected_metrics.json), [burst metrics](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_context_burst_metrics.json)
- The published Multi-GNN implementation requires a separate environment, data preprocessing and architectural adaptations; importing that pipeline is materially more work than a feature ablation. — [official repository setup and usage](https://github.com/IBM/Multi-GNN)

### Inferences

**Recommended sequence today:**

1. Record the active ego candidate's completed results and compare against the fixed V4 blend, not against V1 alone. No new candidate is accepted merely because it beats the older baseline.
2. A1 is the cheapest direct test of the log's medium/context hypothesis. A2 is the stronger new observable-information test if low-ranked easy episodes remain diluted. If there is time for only one, prefer A1 when reliable nested score caches exist; prefer a small label-free A2 when they do not.
3. Pre-register one implementation per block and one final combination. Preserve recipe parameters and equal blend weights rather than tuning them against public scores.
4. Report per-fold and pooled AP, top-402 TP/FP, precision/recall@1%, top-1250 results, and family/difficulty slices for diagnosis only. Use paired group-aware uncertainty and the existing seed confirmation/gates.
5. Check wallet and temporal holdouts for each serious candidate; repeatable AP improvement and no unacceptable budget/holdout regressions remain necessary. Evaluate actual full pipeline context, not only stage-1.
6. Reproduce the exact package from raw inputs, verify hashes/coverage/order and feature semantics, then present the candidate for explicit approval. Do not spend the remaining release/reproduction buffer on unbounded feature search.

**After the deadline / as follow-on research:** A3 with multiple independent periods, true temporal motif histograms at longer horizons, port/order role features, then a directed multigraph architecture or learned temporal encoder if simple features leave demonstrably distinct structure unresolved. GNN paper expressivity results show potential representation capacity, not automatic superiority on this small grouped dataset.

**Additional error controls:** track how many new top-402 rows belong to each lawful scenario; a model can improve average AP while increasing concentration of correlated false alarms. For score propagation, compare against a score-shuffled neighbour control preserving degree/time exposure—diagnostic only, not a submission feature—to see whether gain is genuinely relational or just duplicated self-score/exposure. Never train from annotated scenario IDs or difficulty names.

### Gaps

- No account balances, external transaction histories or new labels are introduced by these recommendations.
- I did not contact or modify the active training agent or its files, and did not run training, generate predictions, upload or submit.
- Time budget here uses the supplied deadline of 24 October 2026, 16:30 Europe/London. It is not permission to relax the frozen promotion gates.

