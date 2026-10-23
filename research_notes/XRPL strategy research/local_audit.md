# Local evidence and XRPL edge-case audit

## 1. What do the actual materials establish, and which earlier strategy claims need correction?

### Takeaway
Keep Strategy1's data integrity and reproducibility discipline; keep Strategy2's domain features and graph context, but remove its unsupported guarantees. The strongest new improvement is to distinguish an economic payment from its atomic conversion execution and to test how graph context behaves at shared service hubs.

### Cited Findings

Local source convention: slide numbers below mean physical PPTX slide positions, not footer numbers. Transcript files contain paragraph separators rather than ordinary newline-delimited prose; paragraph numbers are the result of splitting on U+2029. All three transcripts, both strategy files, and XML text plus notes from all 28 PPTX slides were reviewed. No actual PDF exists in `Pdf`; it contains the PowerPoint. Notes contain only slide numbers, not the promised extended speaker commentary. The two strategy files contain several successive approaches/revisions, explaining the user's reference to three strategies. [Deck](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx); [Strategy1](../../Strategies/Strategy1.txt); [Strategy2](../../Strategies/Strategy2.txt); [Meeting1](../../MeetingTranscribe/Meeting1.txt); [Meeting2](../../MeetingTranscribe/Meeting2.txt); [Meeting3](../../MeetingTranscribe/Meeting3.txt).

| Evidence-backed requirement | Local source | Implication |
|---|---|---|
| Train 40,131 transactions; test 40,101; seven days per split | Deck slide 21 | Preserve exactly one output per test transaction, including transactions with missing or unusable flow evidence. |
| Transfers have 1–3 rows per transaction; conversions involve 1–2 liquidity providers | Slides 8, 21 | Never count all legs as separate scored payments or independent laundering hops. |
| Asset key is TOKEN_ADDRESS + SYMBOL; SOURCE_VALUE is drops for XRP | Slide 22 | Native amount and asset-specific identity must survive conversion to USD. |
| Hourly USD reference rates; join the hour containing the payment | Slides 21–22 | Do not replace with nearest future hour, a symbol-only rate, or external prices. |
| DELIVERED_AMOUNT and EXTRA_TRANSACTION_FIELDS can be JSON text | Slide 22 | Audit parser output and parse-failure coverage. |
| Score is 100 × AP; ties grouped; top-402 precision/recall and ROC-AUC do not rank teams | Slide 24 | Reject 0.5 threshold tuning as primary optimisation; preserve continuous scores. |
| 64/18/18 easy/medium/hard suspicious cases; hidden variations; ambiguous intent | Slide 23 | These are positive difficulty shares, not positive prevalence, difficulty weights, or guaranteed test mixture. |
| All 40,101 test labels used for scoring | Slide 24 | A leaderboard submission is feedback on the evaluation set; repeated tuning can overfit it. |
| No outside transaction or reference data without approval; public docs/papers allowed | Slide 25 | Reading GitHub/Kaggle is appropriate; importing their transaction data or labels needs separate permission. |
| Build on hackathon day; libraries/pretrained models allowed | Slide 25; Meeting2 paragraphs 33–38 | Research is in scope now; no local email proving an early-coding exemption was found. |
| CSV at most 12 MiB; code ZIP at most 8 MiB; method under 10 slides or 4 pages | Slide 26 | The 8 MiB limit attaches to the code ZIP, not an independent methodology upload. |
| Deadline 24 October 2026, 16:30 London; 50 uploads/team/UTC day | Slide 26 | Freeze before deadline and keep a valid fallback; limit use of leaderboard feedback. |
| payments.py and baseline.py supplied in package | Slide 27 | Inspect and use helpers when received; they are not present locally, so their complete behaviour is unverified. |

Specific revisions to earlier planning:

1. Strategy2 lines 4 and 28 propose a 0.5 threshold and isotonic calibration; lines 65–69 correctly retract them. Slide 24 is decisive. Monotone calibration does not improve rank, and non-strict mappings can create ties. Score range is still [0,1] per slide 26. [Strategy2](../../Strategies/Strategy2.txt).
2. Strategy2 line 71 claims all listed parsing concerns are already covered by payments.py. Slide 27 establishes a helper exists, not its corner-case correctness or exact columns. The helper must be audited against README and a small reconciled sample. [Strategy2](../../Strategies/Strategy2.txt); [Deck slide 27](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx).
3. Strategy2 lines 140–141 move from past-only features to declaring whole-week future use permissible. Static AP evaluation alone does not establish this. Retrospective graph context is a strong candidate if README/organiser confirms entire split is available as context; reproduce that availability in validation. Keep a causal variant as a diagnostic or fallback. [Strategy2](../../Strategies/Strategy2.txt); [Deck slides 7, 21, 24, 27](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx).
4. Strategy2 line 27 says hidden test is hard-heavy; transcript says difficulty mixture differs and is unknown. Do not assert it is harder, or upweight difficulty without evidence. A robust strategy tests harder cases without guessing evaluation weights. [Meeting2 paragraphs 30–32](../../MeetingTranscribe/Meeting2.txt); [Strategy2](../../Strategies/Strategy2.txt).
5. Strategy2 line 26 says every feature separating train from test is a generator artefact to drop. Domain shift can also be real covariate change; flag and ablate such features instead of automatic deletion. This is an inference from the strategy's logic, not an observed data finding. [Strategy2](../../Strategies/Strategy2.txt).
6. Strategy2 lines 152–153 infer sparse wallet history from 40k rows. Wallet counts/degree distribution are unavailable: a week can have few dense wallets or many sparse wallets. Profile before deciding wallet context is weak. [Strategy2](../../Strategies/Strategy2.txt); [Deck slide 21](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx).
7. Strategy2 lines 158–159 dismiss temporal validation as meaningless. Seven days limits precision, but a chronological stress split can still reveal temporal artefacts. It is a diagnostic alongside scenario/group holdout, not necessarily the primary offline evaluation. [Strategy2](../../Strategies/Strategy2.txt).
8. Strategy2 lines 86–90 refer to an unseen organiser email approving earlier coding. No email is present. Its existence and scope cannot be used as evidence here. [Strategy2](../../Strategies/Strategy2.txt).
9. The tiny-loop and pass-through anecdotes are real public-ledger illustrations with purpose unknown, not confirmed negatives injected in this challenge. Deck slide 20 says 40 × 0.00001 XRP and 19 seconds / 15 drops. Strategy2 line 6 and transcript numbers differ. Use them as stress-test examples, not universal whitelist rules. [Deck slide 20](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx); [Meeting1 paragraphs 109–114](../../MeetingTranscribe/Meeting1.txt).
10. “New wallet” is first observed in window, not proven on-ledger creation; “nothing else” is observation limited. Avoid equating net recorded inflow with actual account balance. [Deck slides 7, 11, 13, 16](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx).

### Inferences

- The revised approach should have separate evidence tables: observed domain facts, confirmed challenge mechanics, assumptions needing README, and experimental hypotheses.
- The existing strategies are strongest on first-hour traps, but weaker on constructing the right graph, avoiding score propagation contamination, and measuring robustness to novel synthetic variations.
- Full history, component membership, and neighbouring model scores each introduce different validation risks. They must not be collapsed into one generic rule saying either all future data is leakage or all unlabelled context is safe.

### Gaps

- No dataset, README, helper code, submission template, label metadata, official live rules download, or early-coding authorisation email was found locally. Exact columns, leg ordering, missingness, transaction types, scenario IDs and positive prevalence remain unverified.
- The deck promises extended notes, but the actual embedded notes contain no substantive extra commentary.

## 2. Which XRPL semantics are most likely to corrupt the proposed features?

### Takeaway
Use real XRPL documentation to prevent semantic mistakes, while respecting that this is a simplified synthetic ledger. Several useful defensive checks are conditional on the actual schema and should not expand scope into unused ledger transaction types.

### Cited Findings

- **Atomic conversion and self-conversion:** one Payment can exchange multiple assets atomically. Account=Destination can denote an ordinary currency conversion. SendMax is a spending ceiling excluding XRP transaction cost; Paths may contain multiple execution routes. Payment amount issuer fields also have special “any acceptable issuer” cases when issuer matches an endpoint. [Official Payment reference](https://xrpl.org/docs/references/protocol/transactions/types/payment).
- **Partial-payment parsing:** partial payment is a Payment flag, not a separate transaction type. The received value is `delivered_amount`; API v2 uses DeliverMax for the instruction. Direct XRP-to-XRP cannot be partial. Token rounding may cause small differences. Legacy pre-20 January 2014 partial payments may carry the string `unavailable`; that historical case should be tested only if observed. [Official Partial Payments](https://xrpl.org/docs/concepts/payment-types/partial-payments).
- **Failure semantics:** inclusion in a ledger does not prove successful payment; a failure can still incur a transaction cost. Check transaction result rather than interpreting a Flags value as success. Do not create completed money-flow edges from failed instructions. [Official Robustly Monitoring for Payments](https://xrpl.org/docs/concepts/payment-types/robustly-monitoring-for-payments).
- **Ordering:** metadata TransactionIndex gives a transaction's position within its ledger. TransactionResult is separate from transaction flags. [Official Transaction Metadata](https://xrpl.org/docs/references/protocol/transactions/metadata).
- **Timestamp precision:** ledger close time is rounded, usually at 10-second resolution; same-ledger payments should not be assigned artificial subsecond causality. [Official Ledger Close Times](https://xrpl.org/docs/concepts/ledgers/ledger-close-times).
- **Tags:** source and destination tags are unsigned 32-bit fields for off-ledger processing. They can denote hosted customers, return payments, quotes, or disposable deposit identifiers. Their meaning is controlled by the service, not the ledger; identical tag numbers at different addresses do not identify a common actor. [Official Source and Destination Tags](https://xrpl.org/docs/concepts/transactions/source-and-destination-tags).
- **Costs:** token issuers may impose percentage transfer fees; XRP itself has no issuer transfer fee. These are distinct from the burned XRP transaction cost. Consequently, a universal 10–15-drop conservation tolerance is only justified for the challenge's stated simplified cases, not arbitrary cross-asset flows. [Official Transfer Fees](https://xrpl.org/docs/concepts/tokens/fungible-tokens/transfer-fees); [Deck slide 8](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx).
- **Numeric and asset format:** XRP amounts are string integers in drops; token values can use decimal/scientific notation. Currency codes are case-sensitive and may be 40-character hex. Preserve canonical identity rather than blindly uppercasing/decoding symbols or using binary floats for exact fee residuals. MPT is a separate amount shape, only relevant if present. [Official Currency Formats](https://xrpl.org/docs/references/protocol/data-types/currency-formats).

### Inferences

These are proposed checks/features, not claims that corresponding rows exist in the unavailable dataset:

| Priority | Missed edge case / blind spot | Failure it creates | Proposed handling and verification |
|---|---|---|---|
| P0 | Liquidity provider shared by many conversions | Fake laundering hub; giant connected component; misleading cycles/hop depth | Economic graph = sender → ultimate receiver, one edge per transaction; separate execution-leg table for conversion context. Audit helper endpoints against raw JSON/legs. |
| P0 | Summing 2–3 conversion legs as independent volume | Double/triple-count turnover and scenario activity | Store source paid USD and final delivered USD separately; never sum sequential conversion legs into economic payment amount. |
| P0 | Arbitrary parquet row order defines first/last leg | Incorrect actual cost/recipient | Use documented leg index or path endpoint continuity; fail to an explicit unknown state when ambiguous. Never silently choose the largest leg. |
| P0 | Failed or no-transfer transaction omitted by inner join | Missing predictions or invented flows | Left-join all eligible transactions; make successful payment graph from actual transfers; retain missing-flow/status fields for scoring. Confirm label eligibility. |
| P0 | Full components containing legitimate service hubs | All clients grouped together; positive score contaminates unrelated users | Compare economic components, bounded temporal ego-graphs, and hub-aware path features. Do not propagate a component maximum to every member. Measure largest component and fold concentration before grouping. |
| P0 | Unordered static cycles | A→B, B→C, C→A with impossible chronology appears as laundering | Require time-respecting distinct payment edges; exclude same-transaction LP legs as independent steps; record ambiguity for same-time rows. Bound hop/time search. |
| P0 | Predicted-neighbour stage2 leakage | Validation target affects neighbour model whose score returns to that target | Outer validation fold must be absent from every stage1 fit used for stage2 training/evaluation; use inner cross-fitting inside outer-train. Exclude self and duplicate legs; compare to raw structural features. |
| P0 | Rate join by symbol alone or rounded-to-nearest hour | Wrong issuer price; future price; multiplied rows | Validate rate key uniqueness per asset/hour; floor UTC payment time to containing hour; preserve missing/zero/invalid rate separately; no external substitution. |
| P1 | Token fee or changing FX mistaken for retained criminal balance | Conservation features flag benign conversions | Use same-asset native conservation where possible; compare source/destination USD at payment hour; account separately for XRP burn and conversion spread. Preserve mismatch as uncertain feature. |
| P1 | Wallet's existing balance unseen at window start | “Forwarded this deposit” falsely inferred | Label balance as cumulative observed net flow, not actual balance. Pass-through matching is compatibility evidence, not exact provenance. Use amount/time candidate summaries with ambiguity counts. |
| P1 | One deposit pays several exits or vice versa | Nearest-neighbour match misses scatter/gather | Multi-window aggregate conservation and bounded split/merge matching, without assuming strict FIFO or tracing fungible funds exactly. |
| P1 | Same wallet alternates legitimate and suspicious activity | Wallet/component label assignment spreads guilt | Keep transaction-specific context, local contrast, and degree-normalised features; explicitly evaluate mixed neighbourhoods and shared hubs. |
| P1 | Tags treated as global actor IDs | False joins and fabricated flow paths | Use address-scoped tag presence/cardinality/entropy only if fields exist. Optional tag-scoped subgraph is a sensitivity analysis, not asserted identity. Preserve missing versus tag 0. |
| P1 | Self-payment conversion counted as one-node laundering loop | Legitimate exchange/arbitrage pushed high | Separate self-conversion indicator; require multi-transaction contextual evidence for round-trip suspicion. |
| P1 | Sparse / unseen assets and missing JSON | Pipeline defaults unknown to zero or model overfits parse failure | Keep raw audit fields, parse-state masks and fallback row features; check missingness by split/label without target-derived imputation. |
| P1 | Day-1/day-7 clipping | First-seen wallets called newly created; unfinished motifs called clean | Observation-exposure features and left/right censor flags; evaluate interior versus boundary separately. Run cropped-window stress tests. |
| P1 | Fixed 5-minute / 4–7-mule templates | Novel slower/different-width scenarios missed | Continuous counts, duration/amount dispersion and multi-scale 1m/5m/1h/6h/24h/72h/week features; select by ablation rather than all combinations. |
| P1 | Tiny bot loops universally whitelisted | Small-value laundering automatically missed | Treat amount×recurrence×motif as interaction, never absolute exception; inspect tiny-value false negatives. |
| P2 | Token decimal rounding and float fee errors | Near-conservation features unstable | Integer drops and decimal parsing for audit; scale-aware relative/absolute tolerances; logarithms only after valid numeric classification. |
| P2 | Unknown MPT, NFT, AMM, AccountDelete | Wasted effort or unsupported inference | Profile transaction types first; add only observed types with labels/schema support. Technical workshop background is not evidence every type is in challenge. |

### Gaps

- Need helper code and actual rows to verify leg ordering, ultimate receiver extraction, tags, result fields, rate key and whether failures/non-Payment transactions exist.
- Official real-ledger semantics can be richer than challenge generator; issuer transfer fees, special issuer fields, MPT and legacy delivered-amount absence are conditional checks, not assumed feature priorities.

## 3. What concrete strategy improvements follow from the blind spots?

### Takeaway
Spend the first research-to-implementation effort on trustworthy economic graph construction and a validation protocol that matches the allowed prediction context. Only then add graph-derived ranking gains; second-stage smoothing is optional and must earn its place.

### Cited Findings

The deck explicitly recommends loading/joining, building one edge per payment using payments.py, running baseline.py, examining labels, then features and local AP validation. It names normal shopping, peer payments, exchange/payroll hubs and treasury flows, while noting both classes use conversions. Shape slides list legitimate counterparts to all eight motifs. These are the strongest direct priorities. [Deck slides 11–18, 23, 27](../../Pdf/Decode%20%26%20Detect%20AML%20Training%20-%20final.pptx).

### Inferences

Recommended additions to the main strategy:

1. **Two graph views with one target table.** Canonical economic payment edges for actor behaviour, separate execution legs for LP/asset/cost features, and an immutable full transaction key spine for submission coverage.
2. **Validation contract first.** Confirm full-window retrospective context versus causal availability. Primary grouped/scenario holdout when feasible; complementary random edge holdout and temporal/cropping stress tests reveal different risks. Connected components are diagnostics, not guaranteed scenario IDs. Never use test labels or leaderboard to select dozens of features.
3. **Hard-negative contrast rather than extra motif rules.** Compare bursts with each wallet's broader recurrence, counterparty diversity, outside connectivity, tag usage if present, time concentration and amount variation. Normal payroll can fan out; an exchange can collect and distribute; legitimate treasury can conserve value. No single shape or conversion flag is decisive.
4. **Transaction-local context beyond component scores.** Use edge neighbourhood counts and direction, short time-respecting paths, repeated source/destination overlap and role-asymmetry. Avoid a hard rule copying a flow/scenario score to all its transactions until label intent confirms all are positive.
5. **Test generalisation through targeted perturbations.** Use organiser-data-only diagnostics: re-express relative time, permute arbitrary IDs consistently, crop context, vary motif windows, inspect unseen-asset fallbacks, and remove one feature family. Such tests probe stability; they are not independent validation datasets or numerical evidence of real-world AML quality.
6. **Preserve AP-compatible output.** Strong row/structural boosted tree baseline first; raw structural graph features next; constrained score-context model only after safe nested cross-fitting. Rank blending is a candidate, not guaranteed safe/improving; it can reverse useful ordering. Do not add random jitter to ties as a claimed advantage.
7. **Error slices with evidence availability.** Report overall AP and top-1% results plus raw/conversion/self-conversion, hub/non-hub, tiny/large amount, interior/boundary, sparse/dense history, known/unseen asset and missing/complete parser cases. Pattern/difficulty diagnostics only if metadata actually exists. Per-slice AP has a different class prevalence and should be interpreted within that slice, not averaged into an invented overall metric.
8. **Hypothesis-based use of unknown future data.** Separate `before`, `after`, and `full-window` feature families; compare their gains under the same permitted context. This reveals whether benefit comes from real flow completion or distribution-specific timing artefacts.
9. **Stop condition.** If a new family improves one fold but damages hubs, boundary rows or shifted cases, investigate before retaining it. If no stable gain appears, the simpler verified graph-informed tabular baseline is the better submission.

### Gaps

None of these proposed improvements has been implemented or benchmarked, and no honest score gain can be promised without the organiser package. Highest-impact unresolved questions are: allowed observation context; exact helper output/leg ordering; scenario/difficulty metadata; transaction coverage; train/test wallet overlap; and missing-rate/result/JSON behaviour.
