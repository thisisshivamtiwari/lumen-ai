# Generate an evidence-grounded XRPL challenge replica

Act as a senior ML data engineer and XRPL domain specialist. Build and run a reproducible synthetic-data generator that produces the closest defensible development replica of the Decode & Detect challenge described in the supplied materials. Generate the complete related data package, not merely an edge list or a few example rows. Do not train a detector, submit anything, contact organisers or make ledger transactions.

Aim for exact compliance with every stated, testable requirement and economic consistency across files. Do not claim 99% statistical similarity or a near-identical hidden dataset: the organiser's original generator, schema, class prevalence and distributions are unknown. Distinguish documented fidelity from assumed fidelity. A valid replica can match stated counts and semantics exactly without matching the private distribution.

## 1. Read sources before choosing a schema

Use these local inputs, relative to `/Applications/Softwares/Lumen.ai`:

- `MeetingTranscribe/Meeting1.txt`: domain training, eight shapes, approximate fan-in counts, illustrative hubs and real-payment anecdotes.
- `MeetingTranscribe/Meeting2.txt`: synthetic AWS/SonarX-style format, file roles, hidden variation and scoring discussion.
- `MeetingTranscribe/Meeting3.txt`: technical XRPL workshop; treat its general API/NFT/AMM/MPT/account-deletion discussion as background, not a promise those rows are in the challenge.
- `Pdf/Decode & Detect AML Training - final.pptx`: all 28 physical slides, including diagrams and embedded notes. It is a PowerPoint, despite its folder name. Notes may contain only slide numbers.
- `Strategies/Strategy3.md` and `reports/XRPL strategy research.md`: engineering guidance, not organiser facts.

Check whether an actual organiser README, schemas, sample files, `payments.py` or `baseline.py` have since become available. If present, inspect them and use their exact column names, types, null conventions, timestamp units, asset encodings and reconstruction interface. Do not overwrite them. Record any conflict with the earlier deck instead of silently choosing a value.

Read these official references for semantic validation, not transaction downloads:

- https://xrpl.org/docs/references/protocol/transactions/types/payment
- https://xrpl.org/docs/concepts/payment-types/partial-payments
- https://xrpl.org/docs/references/protocol/transactions/metadata
- https://xrpl.org/docs/references/protocol/data-types/currency-formats
- https://xrpl.org/docs/concepts/ledgers/ledger-structure
- https://xrpl.org/docs/concepts/tokens/fungible-tokens/transfer-fees
- https://xrpl.org/docs/concepts/transactions/source-and-destination-tags

Prefer the actual release for dataset structure, the deck for final briefing numbers, and official documentation for ledger semantics. The transcripts contain transcription mistakes. When public documentation is richer than the simplified challenge, do not add every supported XRPL feature to the replica. Do not substitute another vendor's schema and call it AWS/SonarX.

Create an evidence register before generation: `requirement_id`, `source file / physical slide or transcript paragraph`, `statement`, `strength`, `implementation`, `validation`. Strength must distinguish EXACT_STATED, APPROXIMATE_STATED, ILLUSTRATIVE_EXAMPLE, PROTOCOL_RULE and SIMULATOR_ASSUMPTION.

## 2. Exact and bounded requirements from the training deck

Implement these in the primary challenge-like profile unless a newer organiser release explicitly supersedes them:

1. Exactly **40,131 training transaction rows** and **40,101 test transaction rows**. These are transactions, not transfer-leg counts. Allocate complete scenarios before background filling; never truncate a suspicious path to hit a row budget accidentally.
2. Each split has a **seven-day observation interval**. Choose and disclose the actual synthetic start dates. Use `[start, start + 7 days)` consistently; populate all seven days. The original dates and whether windows are consecutive are unknown.
3. Per split create `transactions.parquet`, `transfers.parquet`, `blocks.parquet` and `market_rates.csv`. Create `labels.csv` in training only. The test-facing directory must not contain ground truth.
4. In the primary described data, each transaction has **one to three transfer rows**. A plain transfer has one economic leg; conversions have **two or three execution legs** via one or two liquidity providers. Keep all legs of an atomic conversion under one transaction hash.
5. The join identifier is `TRANSACTION_HASH`.
6. For XRP transfer rows, `SOURCE_VALUE` is measured in **drops**: **1 XRP = 1,000,000 drops**.
7. Asset identity uses **`TOKEN_ADDRESS` plus `SYMBOL`**. Preserve issuer identity. Include XRP and fictional-issued USD, GBP and EUR; include at least two issuers of one token code to exercise the documented distinction. Exact issuer counts are simulator assumptions.
8. Every transaction in this simplified profile burns **10–15 drops of XRP**, including issued-token payments. Choose the exact fee distribution as an assumption; apply the same mechanism to suspicious and legitimate events. Burned fees are not transfer payments to liquidity providers or recipients.
9. `SendMax` is the permitted spending ceiling, not actual expenditure. `DELIVERED_AMOUNT` represents receipt, and the documented first execution leg shows actual payment cost.
10. `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT` are **JSON text** where specified by the release. Do not silently replace them with nested Arrow structs because that is more convenient.
11. `market_rates.csv` supplies an **hourly USD reference rate per asset**; joins use the hour containing the payment. Currency code alone is insufficient when issuers differ.
12. Training suspicious transactions are described as approximately **64% easy, 18% medium and 18% hard**. The deck does not state total positive prevalence, exact counts, or full difficulty definitions. Do not turn this into a 64% suspicious dataset, scenario-count weights, or a guaranteed test mixture.
13. Both legitimate and suspicious activity use conversions. There is substantial shopping, peer payment, exchange, payroll and legitimate treasury activity. Intent labels describe synthetic scenarios, not immutable wallet types.
14. The hidden test has variations absent from training and some ambiguity. Its difficulty mix is unknown; do not assert it is necessarily hard-heavy.
15. The task is transaction ranking by **100 × Average Precision**, not a 0.5 classification threshold. This informs package documentation and label granularity, not a reason to manufacture an easy classifier.

References: deck physical slides 7–8, 21–24 and 27. Match every exact constraint with an assertion and every unprovided choice with a configuration entry.

## 3. Reproduce the actual scenario families

Generate all eight families in the training split and varied counterparts in test. For every family also generate legitimate structural lookalikes, so a motif alone is not the label.

| Family | Suspicious-intent scenario | Legitimate comparison and realism requirement |
|---|---|---|
| Fan-in | Multiple senders feed a collector, followed by one or a few exits; short and varied deposit-to-exit delays. | Merchant, fundraiser and exchange deposits; varied recurrence and outside connectivity. |
| Fan-out | A funded source splits into several recipients, some of whom forward funds. | Payroll, refunds, giveaways and exchange withdrawals; forwarding alone must not be unique to positives. |
| Scatter-gather | Several middle-wallet paths connect a common origin and common destination; compatible aggregate value. | Corporate treasury transfers and exchange rebalancing. |
| Gather-scatter | Several sources and sinks share a central hub, with short holds and low observed retention in some scenarios. | Exchanges, payment processors and payroll providers. |
| Cycle | Several economic payments return funds to the origin, possibly across assets. | Refunds, arbitrage, market-making-like flows and bot/test loops. |
| Layered stack | Multiple intermediary layers separate sources and destinations; support different widths and depths. | Processing chains and related-company payments. |
| Bipartite | Directed many-to-many payments between two groups. | Marketplace buyers/sellers and supplier payments, with overlapping amount and timing ranges. |
| Irregular network | Branches, merges, alternate paths and returns; spans multiple payments and possibly hours/days. | Friends, families and local trading networks. Do not encode intent through disconnected-vs-connected status alone. |

Use the deck's listed signals as generative properties: counterparty counts, participation roles, hold times, value compatibility, outside contacts, layer depth, repeated routes and temporal concentration. These are distributions to model, not deterministic label rules. Allow suspicious participants to make ordinary payments. Do not label every transaction by a suspicious wallet as suspicious, nor label every liquidity provider touched by a suspicious conversion as an offender.

### Approximate transcript hint: fan-in

Meeting1, paragraph beginning “there's, in the training data ... around 18 ...”, says **around 18 fan-in scenarios**, typically **4–7 senders**, paying one collector **within about five minutes**.

Use 18 as a disclosed simulator default for the primary training fan-in scenario count, not an exact organiser guarantee. For those representative scenarios, choose four to seven distinct contributing senders with deposit spread at most about 300 seconds. Include the collector's economically supported exit. Values, exit lags and the number of other scenario families are not supplied: make them configurable and report them. Do not force all legitimate fan-in or every test fan-in to obey this template.

### Concrete illustrations: isolate exact replicas from distribution assumptions

Create a dedicated illustrative-fixture pack with these values and an explicit provenance note. Do not imply these exact public anecdotes are known rows in the organiser's synthetic train/test files.

1. **Fast forwarding, deck physical slide 20:** A sends **50 XRP** to B; **19 seconds** later B sends **49.999985 XRP** to C. Native values are **50,000,000** and **49,999,985 drops**. B's transaction fee is **15 drops**, leaving no XRP from that particular receipt after its forwarding payment and fee. A's separate fee is additional, funded by A. The transcript's “90 seconds” is superseded by the deck. The real example's intent is unknown: store `illustrative_unknown`, not an asserted real-world negative. A separate simulator-created automated-service lookalike can have explicit benign intent.
2. **Tiny loop, same slide:** **40 economic payments**, **0.00001 XRP = 10 drops per payment**, around **ten accounts**. Choose four rounds of a ten-node directed loop as a disclosed implementation of the stated example; its exact original route/timestamps were not supplied. Pay fees separately, with sufficient opening XRP. Do not use 0.0001 XRP from the noisy transcript. The original intent remains unknown. A simulator-labelled bot counterpart may be explicitly benign.
3. **Exchange-style hub, Meeting1 paragraph beginning “the clearest exchange ...”:** construct **94 incoming payments from 66 distinct senders** and **95 outgoing payments to 64 distinct receivers**. The counts are stated in an illustrative exchange-like discussion; exact amounts, timing, overlap of the two groups and presence in the actual challenge are not established. Make any choices explicit. If guaranteeing exact whole-window degree, isolate the fixture hub; do not later attach background edges to it.

Keep these exact fixtures outside the primary row budgets by default. Include benign analogues in the main background, but label their extra numbers as simulator choices. Historical theft totals, UK crime statistics, US illustrative thresholds and generic workshop examples are not generation parameters for this dataset.

## 4. Easy / medium / hard: explicit simulation definitions

Difficulty is evaluation metadata, never a predictor field. These definitions are our assumptions because the organiser has not supplied thresholds:

- **Easy:** compact chronological motif, clear local roles and broadly compatible values, limited interfering activity. Include easy legitimate lookalikes too; do not make fee, JSON format or issuer a class marker.
- **Medium:** more background activity, variable hold times and amounts, partial retention/forwarding, conversions and less isolated wallets. Keep semantics economically valid.
- **Hard:** longer history, changing counterparties, overlapping routes, wider/longer motifs, ordinary payments by scenario participants and plausible legitimate competitors. Difficulty must arise from behavioural overlap, not corrupted files or target leakage.

Parameterise duration, width, depth, value dispersion and overlap continuously. The presence of a cycle or irregular network does not alone assign a difficulty.

Positive prevalence and counts are unknown. Put the chosen total positive budget in `assumptions.json` and config, with a rationale and sensitivity profiles. For an exactly 64/18/18 **transaction-level** training allocation, choose a feasible positive count divisible by 50 and verify `easy = 32m`, `medium = 9m`, `hard = 9m`. If respecting scenario integrity requires a different count or rounded ratio, report the actual integer totals and rounding; never relabel arbitrary filler transactions to fake the proportions. Resolve budgets before generation.

For test, create independent seeds and genuinely held-out parameter combinations, not just renamed copies. Use a configurable, explicitly assumed difficulty mixture; provide nominal and shifted profiles if helpful. Do not claim a particular mix matches the hidden test. Keep every profile self-contained with the documented transaction budgets.

## 5. Make it a closed, economically coherent simulation

Interpret “closed” as internally consistent observations and reproducible state, not access to actual identities or unlimited wallet history.

- All referenced accounts, issuers, asset keys, transaction hashes and ledger references resolve. Generate fictional checksum-valid classic addresses beginning `r` offline where feasible. Do not reuse real victims' addresses, query live balances, generate funded accounts or broadcast anything. Synthetic hashes must be format-valid and unique, but must not be presented as real on-chain transaction hashes.
- Track private opening balances and per-asset debits/credits in simulator state. Prefund XRP fees for token-only participants and prefund necessary liquidity-provider inventory. Never repair insolvency by silently increasing balances mid-scenario. Initial balances are assumptions, not hidden movements added to the public transaction budget.
- When issued assets are moved, preserve their specific issuer. Model issuance/redemption only with explicit accounting; no unexplained token creation. A simplified prefunded-inventory simulator is acceptable if documented as such rather than a full XRPL node emulator.
- For each atomic conversion, retain actual source paid, final destination delivered, all intermediate inventory changes and any explicit spread. Do not sum the legs as economic turnover, fake equal token quantities across assets, or consume balances twice. Fee burn is separate and reduces XRP supply in the simulation.
- Use exact integer drops and decimal-safe token accounting for reconciliation; choose and disclose how Arrow/Parquet stores values when the true schema is absent. Model features may later use floats, but conservation tests must not rely on unsafe exact binary-float equality.
- No cash, KYC identities, real names or ground-truth purpose fields appear in challenge-facing predictor tables. Observed net flow is not wallet balance; first observed is not proven account creation. Hide pre-window state from challenge-facing features.
- Preserve each sender's supported sequence ordering and a deterministic within-ledger transaction order. Use a consistent UTC/event-time mapping. Ledger close times need not be unique; do not invent chronology from physical row order.
- A 19-second **event-time fixture** must not force an impossible ledger-header schedule. Distinguish the documentary payment-time illustration from rounded ledger close-time representation. If exact release timestamp semantics do not support both, retain the exact illustration in its own fixture and document the representational limitation.
- If exporting every consecutive ledger, parent links must be consistent. If exporting only active ledgers, do not falsely claim the previous exported row is the immediate parent ledger. Which ledgers the organiser includes is unknown; disclose the chosen policy.
- Decide train/test wallet overlap, issuer sharing and dates through config. A disjoint-wallet default may be useful for development, but it is not a stated organiser fact. Allow a separate shared-wallet profile for sensitivity analysis; never copy scenario identity across splits as a shortcut.

No full consensus, signature or market microstructure simulator is required. Do not describe format-valid synthetic tables as valid signed ledger transactions or exact protocol-generated hashes.

## 6. Respect Payment semantics without expanding the core data arbitrarily

Use the official Payment reference to validate direct payments, cross-asset execution, self-conversion and optional fields. Important constraints include:

- Use one documented API representation: API v1 `Amount` or API v2 `DeliverMax`, not both as casually duplicated instruction fields. Keep the selected representation explicit and adapt to actual organiser files when provided.
- Direct XRP-to-XRP instructions omit `SendMax` and `Paths`; cross-asset instructions use supported fields consistently. Any provided paths are non-empty valid structures, not strings invented to encode a class.
- Treat conversion legs as atomic under the shared transaction hash, including legitimate self-conversion cases.
- Ensure actual spend stays within a present SendMax in the **source asset**, excluding the separately burned network fee. Avoid comparing token quantities across incompatible assets.

For receipt and failures, use the separate partial-payment and metadata references. A partial payment is a flag, not a new transaction type. Use actual delivered amounts and valid result semantics; do not claim a flags integer means success. Do not manufacture direct XRP-to-XRP partial payments. For tokens, allow documented rounding only through explicit tolerances.

The core deck profile promises one-to-three transfer rows per transaction. Therefore keep failed/no-transfer transactions, malformed JSON, missing rates, unsupported amounts and other deliberately invalid cases in a **separate robustness-fixture pack**, unless the actual organiser schema establishes they belong in the main files. They must not silently violate the primary profile's stated cardinality. Include valid partial-payment or self-conversion examples in the core only if consistent with its schema and stated leg contract.

Tags, if generated, are unsigned service-specific values scoped to addresses; do not treat equal numbers across addresses as a global customer ID. Account reserves, real prices, exact conversion proportions and exotic transaction-type frequencies are unknown. Do not copy garbled reserve figures or default to modern MPT/NFT/AMM records absent evidence.

## 7. Schema fidelity: do not invent an official schema

The deck establishes these exact names: `TRANSACTION_HASH`, `SOURCE_VALUE`, `TOKEN_ADDRESS`, `SYMBOL`, `EXTRA_TRANSACTION_FIELDS`, `DELIVERED_AMOUNT`. It does not provide all columns, case conventions, types, timestamp units, rate headers, native-token address sentinel or label-column name.

If genuine released files/helpers are available, reproduce their complete schemas precisely and document validated compatibility. If absent:

1. Create a minimal, coherent **provisional schema** with the known names preserved.
2. Include transaction sender, destination/amount JSON, fee, type and ledger link; transfer sender/receiver/asset/value and explicit leg order; ledger index/hash/close time; rate asset/hour/USD value; and a clearly chosen train label column.
3. List every unverified name/type/null representation in `schema_provenance.json`. Do not claim these files can feed the real `payments.py` without an adapter or compatibility test.
4. Use one explicit native-XRP token-address sentinel consistently in transfers/rates, documented as assumed if unknown. Use UTC timestamps with declared precision, and document Ripple-epoch conversion only where that field is actually used.
5. Use Parquet for the three documented columnar tables and CSV for rates/labels. Preserve required JSON **strings**. Do not replace the entire package with precomputed graph features.

No external transaction or market-reference rows may be imported. Public documentation can guide representation. Rates themselves are fictional simulation parameters, with reproducible values, not claims about historical XRP or FX prices.

## 8. Prevent simulator fingerprints and accidental target exposure

Randomise opaque IDs and physical output order independently of class while preserving logical chronology. Keep scenario IDs, pattern names, intent, difficulty, wallet role and construction order out of predictor tables and JSON. Do not generate `rFraud...` / `rLegit...` address prefixes, class-coded hashes, asset-exclusive labels, suspicious-only conversion flags or background-only fee values. Apply the same valid serialisation, asset support, fee process and missingness policy to both classes.

Separate simulation mechanisms from labels: infer ground truth from explicitly assigned scenario intent and transaction participation, not a deterministic function of a visible motif-count rule. Some legitimate and suspicious scenarios may be observationally ambiguous. Store ambiguity and label rationale privately. Do not duplicate an identical transaction hash in conflicting scenarios; resolve overlapping event membership deterministically in private truth metadata. Let a wallet participate in both classes.

Do not set an artificial target like “a tree must score 99% AP.” If a cheap classifier perfectly predicts intent from IDs, storage order, JSON shape, rates or file markers, investigate the artifact; do not celebrate it as successful replication. No detector training is required for delivery: a distribution/artifact audit and known-answer feature checks are sufficient.

## 9. Required output structure

Create everything under a new non-overwriting folder such as:

```text
synthetic_replica_v1/
  README.md
  config.yaml
  assumptions.json
  evidence_register.csv
  schema_provenance.json
  fidelity_report.md
  validation_report.json
  manifest.json
  generator/
    ... executable source and pinned dependency manifest ...
  train/
    transactions.parquet
    transfers.parquet
    blocks.parquet
    market_rates.csv
    labels.csv
  test/
    transactions.parquet
    transfers.parquet
    blocks.parquet
    market_rates.csv
  private_truth_DO_NOT_USE_AS_FEATURES/
    train_transaction_truth.parquet
    test_transaction_truth.parquet
    scenario_manifest.parquet
    opening_balances_and_accounting.parquet
  illustrative_fixtures/
    fast_forward_50_xrp_19_seconds/
    forty_ten_drop_payments_ten_accounts/
    exchange_hub_94_in_95_out/
  robustness_fixtures/
    ... separate pathological cases with expected behaviour ...
```

Fixture subdirectories should contain their necessary input tables and separate expected truth, not only a prose description. Test-facing files have no target labels; private synthetic test truth is held outside that directory for development evaluation only. Documentation must clearly warn against importing private truth into feature assembly.

Store source snapshots/retrieval timestamps or version references when available. `manifest.json` records seed/configuration, file row counts, actual schemas, content hashes and generation version. Provide one generation command and one independent validation command. Preserve existing outputs; a repeat seed should reproduce logical records after canonical sorting, even if the Parquet writer's binary metadata prevents byte-identical files.

## 10. Independent acceptance checks

Run validation from exported files, not only in-memory generator objects. Fail loudly for violated core requirements.

1. Exact training/test transaction counts; globally unique transaction hashes; no duplicate target IDs or train/test hash overlap.
2. Every core transaction has the stated one-to-three legs; conversion cardinality and endpoint continuity reconcile; all references resolve.
3. Each observed split fits its declared seven-day interval; rates cover every used asset/hour. For a complete hourly grid, a seven-day interval aligned to hour boundaries has 168 hour buckets per asset. Extra boundary rows, if used, must be explained rather than called required by the deck.
4. `SOURCE_VALUE` XRP units, integer drops, 10–15-drop core fee range, JSON text types, asset identities and actual paid/delivered values reconcile across tables.
5. Every train target has exactly one label. No labels or private metadata are present in test-facing files. No predictor column exposes truth, scenario family or difficulty.
6. Private opening balances plus event updates explain all debits/credits and fees; no unexpected negative balances or silent money creation. Enforce the declared token/inventory accounting model.
7. All eight scenario families and legitimate comparisons exist; report actual scenario and transaction counts separately. Verify the representative approximate 18 fan-in cases if that default is used.
8. Report positive prevalence, training transaction-level difficulty counts/ratios and chosen test mixture as assumed where not stated. Check intended ratio without breaking scenarios.
9. Assert exact illustrative-fixture values: 19-second / 50 XRP forwarding pair with 15-drop B fee; 40 × 10-drop loop payments over ten accounts; 94/66 incoming and 95/64 outgoing hub counts in the declared scope.
10. A held-out test parameter combination must genuinely differ from training; seeds and addresses alone do not establish variation. Report which parameters/family combinations are withheld.
11. Read/write roundtrip, row-order shuffling and documented equivalent amount representations preserve economic meaning. Null-key and duplicate-rate fixtures cannot silently multiply records.
12. A standalone validator can reconstruct one economic payment per core transaction from exported tables. If actual `payments.py` is available, test compatibility directly; otherwise mark helper compatibility UNKNOWN, not PASS.
13. For each documented requirement, produce PASS / FAIL / UNKNOWN / NOT APPLICABLE and an evidence path. Do not convert schema unknowns into successful matches merely because an assumed schema is internally valid.

## 11. Fidelity report and final response

Report three distinct things:

- **Observable requirement compliance:** which exact supplied facts were matched, with measured counts and tests. Aim for all applicable explicit assertions to pass.
- **Protocol and relational consistency:** what was validated versus which aspects of full XRPL consensus/execution were simplified.
- **Unknown statistical fidelity:** prevalence, amount/time distributions, participant counts, scenario frequencies, exact schemas and hidden-test mixture that cannot be verified without organiser samples.

Never compress these into “99% identical to the organiser dataset.” State which unknowns would need original files or aggregate reference statistics to quantify similarity. Do not treat a high model score on self-generated data as fidelity evidence.

At the end, give the output path, exact reproduction/validation commands, actual row counts, difficulty/scenario summaries, validation failures and limitations. Do not dump tens of thousands of rows into chat. Do not train a final model or mix generated records into the competition submission workflow; this package is explicitly synthetic development material.

Proceed end-to-end using transparent configurable defaults for unknowns. Do not stop merely because the full original schema is missing; produce a clearly provisional replica and identify exactly what would need adapting when the real files arrive.
