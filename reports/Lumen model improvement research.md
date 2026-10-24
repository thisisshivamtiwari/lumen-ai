# Improve Lumen Through Selective Transaction Evidence

**The next useful work is to make the existing V4 improvement auditable, then test whether contextual evidence distinguishes lawful look-alikes from suspicious payments.** The saved V4 result is a substantial improvement: mean fold AP rises from **54.4829% to 61.6162%**, and the global top-402 list catches **355 rather than 326 positives**. That result should not be dismissed as leakage: the inspected outer-validation construction excludes the held-out fold's labels. There is, however, a separate feedback path inside meta-training and an unresolved difference between training-context and inference-context score sources. While the existing agent finishes its ego-network work, the best distinct research candidates are exposure-corrected neighbour evidence, strictly ordered bounded paths, and a single capped hard-negative experiment. Multi-day relay episodes belong behind those candidates, subject to time. **No new feature has been implemented or measured by this research, and no training, submission, upload or active-source modification was performed.**

Research snapshot: **24 October 2026, approximately 13:10–13:25 Europe/London**. Active source was changing during inspection; the results and code references below identify the observed versions rather than asserting that development has stopped. This report builds on the released data and current logs; pre-release assumptions in the [earlier strategy research](</Applications/Softwares/Lumen.ai/reports/XRPL strategy research.md>) are superseded wherever the released README or current evidence resolves them.

## V4 improved the ranking, but its remaining errors have changed

The comparable local evidence comes from the same 40,131 labelled transactions, original cohort/scenario grouped folds, and 1,250 positives. The **competition review budget is 402**, calculated as `ceil(0.01 × 40131)`. The exploratory top-1,250 table answers a different question; its precision equals recall only because its budget equals the positive count. Different table totals therefore do not establish a different run. The newer presentation also merges raw families: “Scatter-gather” combines `fan_out_in` and `multi_stage`, while “Layered stack” combines `chain`, `stack` and `distributed_relay`. Keep the raw `family × difficulty` breakdown beside any friendly names. ([Reporting implementation](/Applications/Softwares/Lumen.ai/outputs/fp_fn_report/source/fp_fn_report.py:18), [frozen V1 errors](/Applications/Softwares/Lumen.ai/outputs/validation_error_report/Validation_FP_FN_Report.md))

| Local validation procedure | Mean fold AP | Pooled AP | TP / FP at 402 | Precision / recall at 402 | TP at 1,250 |
|---|---:|---:|---:|---:|---:|
| Frozen V1, raw scores | 54.4829% | 53.7186% | 326 / 76 | 81.0945% / 26.0800% | 699 |
| Saved V4 motif/context rank blend | 61.6162% | 61.5271% | 355 / 47 | 88.3085% / 28.4000% | 785 |

These figures were independently recalculated by the research audit from the [V1 OOF file](/Applications/Softwares/Lumen.ai/outputs/strategy3_v1/graph_oof.csv), [V4 OOF file](/Applications/Softwares/Lumen.ai/outputs/v3_context/blend_motif_context_oof.csv) and released labels aligned by transaction hash. V4's **7.1333 percentage-point mean-fold gain** survives the score-normalization sensitivity check. The attachment reports a **66.5 portal score**, but that server result and uploaded-file identity were not independently authenticated in this research. It is supporting log evidence, not a promised score for another candidate. ([Audit and artifact hashes](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/stacking_audit.md>))

| Category | Positives | V1 caught at 402 | V4 caught at 402 | V4 caught at 1,250 | V4 missed below 1,250 |
|---|---:|---:|---:|---:|---:|
| Easy | 800 | 321 | 339 | 755 | 45 |
| Medium | 225 | 5 | 16 | 26 | 199 |
| Observable-hard | 200 | 0 | 0 | 4 | 196 |
| Ambiguous | 25 | 0 | 0 | 0 | 25 |

The immediate FP target is now **bipartite, fan-out-in and chain**: together they account for **38 of V4's 47 head false positives**. Aggregate relay supplied 12 V1 head false positives but only one in V4; it should no longer dominate the description of current head errors. At ranks 403–1,250, V4 has 430 positives—416 easy, ten medium, four observable-hard—mixed with 418 legitimate payments. This supports investigating both lawful head look-alikes and coherent easy cases just outside the budget. It does not justify suppressing whole annotated families. ([Recomputed counts and source trail](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/ranking_and_budget.md>))

AP also rewards ordering below, and within, that budget. For unique scores, `AP = Σpositive ranks r precision@r / 1250`. A descriptive decomposition of the pooled V4–V1 AP gain attributes about **6.50 points to easy**, **1.20 to medium**, and **0.10 to observable-hard**; this is an accounting identity, not a causal feature attribution. Equal rounded portal precision bars do not prove that all remaining score differences arise below the first 402. A perfect 402-row head could recall at most **32.16%** of 1,250 positives, so 100% review-budget recall is mathematically impossible. ([AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [local decomposition](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/ranking_and_budget.md>))

## Protect every score source before extending the context stack

The inspected context wrapper uses outer fold F as validation and the other folds G/H for fitting. Its validation-score sources and stage-two fit exclude F's labels. **This is meaningful outer protection.** The concern is narrower: a centre j in G receives a neighbour k's score from a model fitted on G, including `y_j`. Thus `y_j → fitted model → score(k) → context(j)` is a possible training shortcut. It is a confirmed dependency, but its numerical effect is unmeasured. It does not establish that the reported outer AP is fabricated or directly contaminated by held-out labels. ([Score-bank construction](/Applications/Softwares/Lumen.ai/outputs/v3_context/source/experiment.py:79))

Ordinary stacking uses cross-validated base predictions for meta-training and full-data base models at inference. Neighbour aggregation adds a dependency that checking the centre's own OOF prediction cannot resolve; that extension is this audit's engineering conclusion. For an outer training set T, split it into grouped inner blocks B. Fit a base model on `T\B`, score the observable window with that one model, and extract context only for centres in B. Assemble those centre-wise features to fit stage two. Fit a base on T to construct validation-centre features, with F excluded from every source fit. Apply the same construction to all training labels for final meta-training, followed by full-training base scores on the separate test window. ([Official stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html))

**That minimal recipe removes the centre-label feedback, but does not make distributions identical.** Some neighbouring training-window rows are in-sample for their source model, whereas every payment in the separate test window is unseen. Source fit sizes also differ. A stricter diagnostic excludes both the centre block B and neighbour block C when producing a message from C to B; at least three inner blocks are needed, with the `B=C` case handled explicitly. This costs more and is a separate candidate, not a silent repair to saved scores. Record `(centre block, neighbour block, source model hash, source fit blocks)` and assert both outer and centre exclusions. A toy dependency-sentinel test can check assignment cheaply; a real permutation/refit tests effects more strongly but costs training time. ([Detailed construction and limitations](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/stacking_audit.md>))

Self inclusion needs precise language too. Time-window context excludes the current event, while whole-week endpoint means/maxima include it, and `ctx_self_pct` is explicit. Those features are not inherently leakage when their score sources are valid; they risk redundant echoes of the centre score and inaccurate “other payments” descriptions. The burst extension's minimum also includes self despite its comment. Correct semantics and compare leave-one-out versions as an ablation, preserving frozen artifacts. ([Current aggregation code](/Applications/Softwares/Lumen.ai/solution/context.py:35), [weekly and burst features](/Applications/Softwares/Lumen.ai/solution/context.py:60))

Two evidence repairs are inexpensive. First, the existing bootstrap loads the older `context_oof.csv` and compares against **percentile-transformed V1**; it is not a V4-versus-raw-V1 interval. Second, ranking each outer fold separately gives V4 pooled AP **61.5271%**, whereas ranking the mixed three-model bank globally gives **60.2800%**. Mean fold AP remains strong under both checks. This sensitivity does not prove inference is wrong: actual inference ranks one full test window, while pooled OOF mixes fitted models. Freeze and name the exact percentile and tie-break procedure, then bootstrap that exact candidate against raw V1. ([Bootstrap source](/Applications/Softwares/Lumen.ai/outputs/v3_context/source/bootstrap.py:13), [packaged blend](/Applications/Softwares/Lumen.ai/outputs/v4_motif_context/lumen_v4_code/run_v3.py:24), [recalculation](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/stacking_audit.md>))

## Three focused experiments add information beyond the active ego work

The active ego experiment already measures bounded short-range reach and value summaries. V1 already has before/after window totals, amount matches, pair recurrence, entropy and static paths; V4 adds connected temporal components and score context. Re-emitting these features is duplication. Existing trials also argue against wider seed, iteration or burst grids: five-seed V1 averaging added only **0.1226 mean-AP points** while losing two head catches; longer motif training and same-burst context underperformed their respective references. ([Source and completed ablations](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/graph_frontier.md>), [ranking audit](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/ranking_and_budget.md>))

| Priority | Proposed new experiment | Distinct hypothesis | Scope and stop rule |
|---|---|---|---|
| 1 — do next | Exposure-corrected neighbour evidence | Concentrated support matters more than touching a busy wallet | One compact context block; 30–45-minute initial investigation if valid score caches exist |
| 2 — alternative next | Ordered, anchor-bounded route evidence | Joint direction, total duration and amount compatibility distinguish actual participation from loose reach | 10–20 label-free features; 45–60-minute initial investigation with deterministic caps |
| 3 — only with compute available | Capped inner-OOF hard-negative emphasis | A few confusing legitimate training examples deserve extra learning weight | One fixed ×2 weighting rule; 30–45-minute initial investigation with honest caches |
| 4 — defer unless earlier work finishes early | Multi-day counterparty turnover and residual episodes | Hard relays reveal repeated roles across days | One fixed episode definition; no width/lag search before freeze |

These are planning allocations, **not measured runtimes or expected gains**. Test at most one or two additional candidates today; full holdouts and reproduction consume time beyond the initial trial. The proposals below are original adaptations motivated by the cited research, not published methods already validated on Lumen.

**Exposure correction:** for an event's directional neighbourhood, exclude self and duplicate events; let n be its event count and K count cross-fitted percentiles at least 0.97. Add `H=(K+5×0.03)/(n+5)` and `E=(K−0.03n)/sqrt(0.03×0.97n+1)`, retaining n. E is a heuristic excess, not a calibrated significance test: neighbouring scores are dependent. Add distinct supporting counterparties and concentration `C=Σc(Kc/max(K,1))²`. Use only fixed 1h/6h upstream-before and downstream-after neighbourhoods. This asks whether support is unusually concentrated and broadly distributed, rather than automatically treating a large maximum as strong evidence. Current features include counts and maxima, but not these explicit exposure/concentration contrasts. CatBoost can approximate some interactions already, so redundancy is a credible failure outcome. ([Existing context](/Applications/Softwares/Lumen.ai/solution/context.py:42))

Use a leave-one-out weekly mean/max in the same small block, preserving the explicit centre score. Test replacement of selected maxima rather than appending a large correlated family. Lawful bipartite bursts can produce many correlated high scores; multiple counterparties are not independent witnesses. Inspect candidate–incumbent changes by exposure and lawful scenario after fitting, and include a neighbour-score shuffle preserving exposure/time strata as a diagnostic of whether relational ordering adds information. A shuffle is a control, never a prediction feature. Valid source provenance is a prerequisite to interpreting this trial.

**Ordered routes:** existing motif components link consecutive events within a gap D, then use transitive connectivity; their total duration can exceed D. The active ego radius also does not establish a directed path. Temporal-motif research instead requires ordered directed edges within a bound on the entire motif span. Implement a route `a→s, s→r, r→b` containing the centre, requiring `t_in<t<t_out` and `t_out−t_in≤D`, for fixed **30/120-minute total spans**. Emit route support, support divided by available upstream×downstream opportunities, lag summaries, and continuous amount-mismatch summaries. Keep true return-to-source cycles separate from chains. ([Temporal motif definition](https://snap.stanford.edu/temporal-motifs/index.html), [current source audit](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/graph_frontier.md>))

Compare incoming received reference USD with actual outgoing sender-paid reference USD at each intermediary: `error=abs(received−paid)/max(received,paid,epsilon)`. Keep cost, delivery and conversion uncertainty separate. Cap candidates at 32 per endpoint/window using deterministic temporal selection and emit cap flags; do not enumerate unrestricted Cartesian joins on 8 GB RAM. Use released ledger order for timestamp ties only where semantics support it, otherwise record ambiguous order. A matching amount proves compatibility, not which fungible funds financed an exit. Payroll and lawful settlement can form the same routes; pre-existing balances, split/gather flows and window censoring can defeat exact conservation. The test is whether continuous joint evidence improves held-out ranking, not whether a tighter hand-coded rule looks persuasive. GFP's per-edge temporal patterns support this feature-level approach without requiring a new graph library. ([GFP feature definitions](https://snapml.readthedocs.io/en/latest/graph_preprocessor.html))

**Hard-negative emphasis:** within each outer training partition only, generate reference scores by inner grouped cross-fitting. Set `w_i=1+1[y_i=0 and inner_percentile(score_i)≥0.968]`; all positives and other negatives retain weight one. Fit one weighted candidate with frozen features/hyperparameters. The weights use permitted binary training labels, never family/difficulty annotations. Do not mine negatives from the outer OOF errors being evaluated, or reuse global OOF scores whose source models saw outer-validation labels. If the reference is stacked, rebuild its entire stack within the training boundary. This is a targeted learning experiment, distinct from the failed ×3 weighting of every positive. It can still demote genuine suspicious look-alikes, and must pass overall AP/head gates. ([CatBoost per-object weights](https://catboost.ai/docs/en/concepts/python-reference_pool), [previous weighting result](/Applications/Softwares/Lumen.ai/outputs/run1_investigation/comparison.csv))

**Deferred relay evidence:** six-hour bins could measure adjacent-bin counterparty-set turnover, repeated split/merge roles and lagged incoming/outgoing compatibility. Observed residual `R(t)=Σreceived_USD−Σpaid_USD` can describe repeated accumulation/forwarding episodes, after accounting for local starting level and observation coverage. **It is not account balance.** MonLAD motivates residual and repeated balancing descriptions, but its account-level stream results do not establish transaction AP gains here. Seven days provide few independent episodes, and lawful businesses repeat too; test independent scenario/cohort support, not only row counts. Defer full GNN adoption, a MAP-ranking-objective trial and new external datasets until simpler features leave a measured unresolved gap. ([MonLAD primary paper](https://shenghua-liu.github.io/assets/pdf/papers/wsdm2022-monlad.pdf), [IBM Multi-GNN implementation](https://github.com/IBM/Multi-GNN))

## Promotion requires the exact candidate, every stage and every original gate

Keep the original grouped memberships and raw V1 comparator frozen. Each candidate needs **at least +1.0 AP percentage point**, improvement in **two of three folds**, a **positive lower paired group-bootstrap bound**, positive gain under the prescribed confirmation seed, and **no wallet or temporal AP regression greater than two points**. Add the existing global-top-402 non-regression requirement; slice improvements cannot override it. Also show incremental benefit over the latest admissible V4/ego incumbent: clearing V1 alone does not justify replacing a stronger candidate. Previously inspected holdouts remain regression diagnostics, not an untouched final test. ([Original gates and later head-budget requirement](/Applications/Softwares/Lumen.ai/outputs/run1_investigation/Run1_Investigation_Report.md))

For a stack, **wallet/temporal isolation applies to every base fit, context score, mining weight and meta-model**, not just the final classifier. Keep permitted full-window unlabelled context distinct from streaming causality; the former is allowed by the released offline task. Do not reuse V1 temporal numbers as evidence for V4. Recompute the exact candidate's confirmation seed and B/C diagnostics; none was found in the scoped V3/V4 metrics at the audit snapshot. ([Released README](/Applications/Softwares/Lumen.ai/data/training/README.md), [audit gaps](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/stacking_audit.md>))

Each result table should include mean/per-fold AP, explicitly labelled pooled AP, global TP/FP/precision/recall@402, fold-budget top134 results, exploratory top1250, and raw-family/difficulty analysis. V4 catches **124/121/111** under separate fold budgets; those sum to 356, while global top402 catches 355. Both are legitimate allocation diagnostics and must stay separate. Resample paired cohort/scenario/background-row blocks within folds for the primary mean-fold AP delta, binding results to source/config/OOF hashes. Keep the saved score procedure fixed during bootstrap and state the resampling policy. Conditional score bootstrap omits retraining and repeated-selection uncertainty. Reusing folds to choose many features can overfit the selection criterion even without label leakage. ([Local fold counts](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/ranking_and_budget.md>), [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html))

Finally, freeze the exact inference recipe and run the **actual extracted ZIP from raw inputs**. Check all 40,101 expected test hashes, uniqueness, finite [0,1] scores, deterministic tie handling, dependency versions and original V1 protection hashes. Packaged evaluation scripts currently contain project-depth assumptions; successful imports do not prove portable evaluation reproduction. The V4 ZIP's validation summary still identifies V1 metrics; label historical evidence and replace stale candidate claims only in a new package. Preserve immutable ZIPs and predictions rather than overwriting them during correction. ([Packaged evaluation path assumptions](/Applications/Softwares/Lumen.ai/outputs/v4_motif_context/lumen_v4_code/experiments/experiment.py:18), [packaged runner](/Applications/Softwares/Lumen.ai/outputs/v4_motif_context/lumen_v4_code/run_v3.py:56), [metadata audit](</Applications/Softwares/Lumen.ai/research_notes/Lumen model improvement research/ranking_and_budget.md>))

## Freeze experiments early enough to verify the result

Using the supplied **16:30 London deadline on 24 October 2026**, allow the active ego task to finish and use roughly **13:25–14:15** for provenance/evidence repair plus at most one extra trial. Freeze the candidate list by **14:15**, then use **14:15–15:00** for confirmation and full-pipeline holdouts. Freeze feature/model semantics by **15:00** and reserve **15:00–15:45** for clean reproduction, methodology and artifact checks. Deliver reviewable files by **16:00**, leaving a contingency buffer before the stated deadline. If a gate or reproducibility check cannot finish, retain the latest fully evidenced incumbent. This schedule is a planning constraint, not independently verified portal availability or permission to submit.

The following handoff can be copied to the working agent after its current task; it deliberately avoids interrupting that task:

```text
Finish your current ego/context experiment first. Keep V1 and all prior
prediction/ZIP artifacts immutable. Do not upload or submit anything.

Create a new timestamped research directory and freeze the exact incumbent
source, configuration, folds, OOF scores and hashes. Reconcile 402 versus
1250 budgets and raw family mappings before comparing reports.

Before extending score context, save source-model fit membership for every
centre/neighbour block. Assert outer-heldout labels are absent everywhere.
Audit the meta-training feedback path from a centre label into neighbour
scores. Evaluate a centre-excluded construction as a separate procedure;
document remaining in-sample-neighbour and fit-size mismatch. Do not call
ordinary per-row OOF alone sufficient for graph context.

Redo the exact finalist-versus-raw-V1 paired group interval; the older V3
context bootstrap is not a V4 interval. Evaluate the full pipeline on wallet
and temporal holdouts, including all base/context/mining stages.

If there is time after the active task, pre-register ONE new candidate:
exposure-corrected context using fixed 1h/6h directional support,
H=(K+0.15)/(n+5), excess/concentration and leave-one-out weekly summaries.
Use only provenance-valid score banks. If such banks are unavailable,
prefer the small label-free ordered-route experiment in the report.
Do not repeat ego reach, seed bags, long training or broad blend searches.

Only with spare compute, test one hard-negative rule using inner-group OOF
scores inside each outer training partition: weight 2 for negatives in the
top 3.2%; weight 1 otherwise. Never mine from outer-validation errors.

Keep every promotion gate: >=1 AP point over raw V1, >=2/3 folds improve,
paired group CI lower bound >0, positive prescribed confirmation-seed gain,
no >2 AP-point wallet/temporal regression, and global-top402 non-regression.
Report incremental benefit over the best admissible incumbent too. Show AP,
TP/FP/P/R@402, per-fold top134 and exploratory top1250; annotations are
analysis/grouping only, never predictors or hard-coded ranking penalties.

Freeze the candidate list by 14:15 London and model semantics by 15:00.
Reproduce the actual extracted ZIP from raw data, validate complete IDs,
hashes/schema/ranges and correct methodology before presenting files.
If checks fail or time expires, retain the latest fully verified artifact.
Return the evidence and recommendation; await explicit submission approval.
```

The main implication is that **selectivity matters more than additional connectivity**. V4 already learned useful contextual structure; the unresolved question is which contextual support survives lawful high-activity patterns and a fully specified inference procedure. Research narrows that question to a few falsifiable tests. It cannot promise a leaderboard increase, identify intent from ambiguous labels, or replace candidate-specific verification.
