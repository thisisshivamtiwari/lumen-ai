# Kaggle competition lessons for the XRPL strategy

## Which Kaggle ideas transfer, and which should be corrected?

### Takeaway
Borrow entity-aware behavioral features and deliberate validation, not competition-specific label propagation. The most useful improvement is an experiment contract that makes leakage and false gains harder to create.

### Cited Findings
- IEEE-CIS winner Chris Deotte's own coauthored account describes combining XGBoost with teammates' CatBoost/LightGBM, testing chronologically, deriving client identifiers to produce behavioral aggregations, and excluding the UID itself to avoid overfitting. Its task used ROC-AUC, not this challenge's AP. Its useful transfer is entity-relative features, not its exact score, model settings, or card-specific features. [Winning participant account](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/).
- The 21st-place author's GitHub account emphasizes sequences of similar transactions rather than isolated records. It reports an offline/online discrepancy caused by grouping train and test separately versus together. This is evidence that aggregation construction must match inference, not permission to use labels from validation/test. [Author repository](https://github.com/white-bird/kaggle-ieee).
- A lower-ranked IEEE-CIS competitor reports reducing adversarial AUC from 1.00 to 0.83 by removing about 150 features. Their reported rank blend of LightGBM/CatBoost scored 0.913026 private AUC versus LightGBM alone at 0.918967. This does not isolate a causal effect of deletion but directly shows that combining model families and rank averaging is no guarantee of improvement. [First-party Kaggle retrospective; indexed text](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111164).
- The 41st-place team's account describes two branches, one with adversarial-filtered features and another retaining all features; their final blend outperformed the filtered branch privately. It also explicitly describes probing individual inferred users using leaderboard feedback and turning resulting inferences into pseudolabels. Those manipulations are task-specific and should not become the XRPL default. [First-party Kaggle solution; indexed text](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/lets-try-not-to-drop-1000-places-41st-place-soluti).

### Inferences
The following are research recommendations, not measured XRPL results:

| Existing proposal | Verdict | Concrete correction |
|---|---|---|
| CatBoost/LightGBM with time and graph features | Keep as primary baseline | Compare raw transaction, behavioral, and motif blocks through the same frozen folds. Feature improvements are likely a better one-day investment than a large hyperparameter search. |
| Group by connected component | Conditional stress test | First inspect component sizes and label counts. A legitimate hub can create a giant component, making this split impractical or unrepresentative. Prefer true scenario IDs if provided, never assume they exist. Compare entity-disjoint and chronological stress tests against actual train/test overlap. |
| Automatically delete high adversarial-importance features | Reject as automatic rule | Classifying train versus test diagnoses distinguishability, not failure of the target relationship. Absolute date differences can make adversarial AUC nearly perfect for an expected later window. Inspect identifiers, boundaries, missingness and normalized behavior separately; delete only with leakage reasoning or supervised holdout ablation. |
| Graph score smoothing | Conditional, low priority | Keep edge-level distinctions and direction/time. Legitimate hubs and mixed benign/malicious accounts violate the assumption that every neighbor shares a label. Compare residual blending of predictions with original model; inspect hub false positives. Never overwrite known-negative-looking edges solely because the account is risky. |
| Difficulty upweighting | Conditional, low priority | If difficulty is organizer-supplied, verify it is available at inference and legitimately usable. If derived from errors, derive it within training folds. Compare global AP and easy-case regressions; hard examples can be noisy/ambiguous rather than merely underweighted. |
| Isotonic calibration for AP | Remove from default | It is a probability quality tool. A strictly increasing remapping leaves ranking and AP unchanged; isotonic creates score ties and can worsen AP. Calibrating separately before averaging can change ensemble ordering, but must earn its place through held-out evaluation. |
| Rank-average ensemble | Conditional | Compare best single model, probability mean and rank mean on identical honest predictions. Average ranks preserve ties shared across all models and can create equal rank sums even without those shared ties. Adding seeds is not sufficient evidence of useful diversity. |
| Two-stage neighbor predictions | Conditional, high leakage risk | Do not fit stage two on in-sample scores. Validate the entire construction under an outer holdout, including how neighbor scores are generated; see details below. |

### Gaps
- No XRPL data were supplied to this research subtask, so no split, feature, weighting, smoothing or ensemble is demonstrated to win here.
- The IEEE-CIS data structure, label persistence, evaluation metric and months-long competition differ from a seven-day synthetic XRPL batch. Its numeric gains cannot be promised here.

## How should AP, validation and stacking be handled?

### Takeaway
Evaluate the organizer's exact average-precision implementation and protect the full feature/prediction pipeline from held-out labels. A good transaction-level score can otherwise be a measurement of shared entities or label contamination.

### Cited Findings
- Scikit-learn defines AP as the sum of precision at thresholds weighted by recall increases; its implementation is not the trapezoidal area under a linearly interpolated precision-recall curve. Both decision scores and probabilities are valid inputs. [Official AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- Scikit-learn's calibration guide says isotonic regression introduces prediction ties and can alter ranking metrics; sigmoid is strictly monotonic and preserves ordering. The guide separately notes isotonic's overfitting risk, especially with small samples. [Official calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
- Official stacking documentation trains the final estimator using cross-validated base predictions. It warns of a very high overfitting risk when prefit base models and the stacking model use predictions from the same training data. [Official stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).
- Kaggle's solution-write-up documentation reproduces the IEEE-CIS winners' discussion of multiple time-gap holdouts, month grouping, and separate analysis of known, unknown and uncertain linked clients, where different model families were strongest in different cohorts. Treat it as Kaggle's reproduction, not a directly fetched full original discussion. [Kaggle documentation; indexed text](https://www.kaggle.com/solution-write-up-documentation).

### Inferences
- Confirm the organizer helper before equating any library's metric called `aucpr`, `PRAUC` or `PR-AUC` with `average_precision_score`. Do not optimize F1 or choose a final threshold for an AP submission; submit continuous scores for every transaction.
- Report global AP, fold AP with positive/negative counts, and subgroup failure diagnostics. AP depends on prevalence, so compare model A versus B on the same subgroup and examples; do not call a rare-pattern AP lower than a common-pattern AP proof that the model understands the former less well. If per-pattern slices contain only positives, AP is degenerate; evaluate against a fixed justified negative background and also report recall at the global operating threshold/top-k.
- Pooled out-of-fold AP and mean fold AP answer different questions because score ordering across folds matters. Save both and inspect score-scale differences; do not mix predictions from different models/folds blindly and present that as a definitive score.
- Create an explicit data-visibility contract: labels always training-only; graph topology and full-window unlabeled test transactions only if batch/transductive use is permitted; historical-only views if the task requires online scoring. Reproduce the chosen visibility regime in validation. A future unlabeled edge is not inherently forbidden in a batch task, but a later label remains forbidden.
- Neighbor prediction leakage is subtler than ordinary row-level OOF. Suppose edge i's own OOF score excludes its label, but neighboring edge j's score comes from a model trained on i. Aggregating j's prediction into i can indirectly carry i's label. For an outer validation fold, train every first-stage model and every label-based graph statistic without any outer validation label, then produce neighbor features and assess the full second stage on that untouched fold. A simpler one-day choice is label-free graph features and a single booster.
- Feature joins must preserve transaction cardinality. Repeated directed edges are separate observations; do not deduplicate them, collapse parallel edges, or map a node risk label to all incident transactions. Track unseen sender, unseen receiver, both unseen, repeated pair and giant-hub cohorts.
- Prefer a small predefined validation matrix: one development split matching likely hidden-test identity/window structure; one entity-disjoint stress split; one chronological boundary stress split when relevant. Group bootstrapping by scenario or meaningful entity can diagnose uncertainty better than treating every related edge as independent, but avoid spending the hackathon on elaborate inference.

### Gaps
- It is unknown whether train/test are independent simulated weeks, chronologically successive windows, share account identities, or reuse scenario generator templates. These questions control the primary split and must be resolved from release files/README and organizer answers.
- A weakly connected component is not necessarily an independent laundering scenario. Component CV can be both too harsh for expected shared entities and too weak for shared generator templates.

## What is the practical Kaggle-informed execution strategy and access coverage?

### Takeaway
Use leaderboard feedback to verify a robust local result and spend scarce time on the data contract, strongest feature blocks, and submission integrity. There is no evidence-based reason to stack every attractive competition trick into a one-day solution.

### Cited Findings
- Dwork and colleagues show that adaptive holdout reuse can lead to overfitting, including when repeated scores guide the next model. [Original research paper](https://arxiv.org/abs/1506.02629).
- Roelofs and colleagues' analysis of more than one hundred Kaggle competitions found little evidence of substantial overfitting in aggregate. Thus leaderboard overfitting is a risk, not an inevitable explanation of every private/public discrepancy. [Original NeurIPS paper](https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html).

### Inferences
Suggested experiment queue, in order:
1. Audit release schema/labels, split semantics, ID overlap, missingness, parallel edges and boundary censoring. Record official scorer and submission contract. Make a valid baseline submission early.
2. Train a compact booster on raw transaction fields plus safe transformations; keep all IDs for joins but exclude arbitrary numeric ID ordering from the robust default. Run an explicit ID-only/row-order artifact audit rather than silently relying on generator artifacts.
3. Add endpoint-relative behavior, pair recurrence, directed degree/amount flow, short time windows and time-respecting motif features as separate blocks. Inspect both added value and hub false positives. Normalize per local window/endpoint where train/test windows differ.
4. Fit a complementary booster only after the feature model is sound. Measure which positives move up, which legitimate-hub edges move down, and top-rank overlap; model-family names alone do not establish diversity.
5. Compare a small set of blends on honest predictions. Keep the best single model available as fallback. Attempt smoothing, weighting or second stages only if stable improvements on the original holdout and key stress cases justify the extra failure modes.
6. Freeze the final feature/score choices before the final test run. Validate row count, exact IDs, order, duplicates, finite bounded scores if required, required columns, and successful loading by organizer helper. Save code/config, dependency versions, run seed and feature visibility notes. Preserve time for final submission instead of one more unreviewed model.

**Proposed retention gate (design choice, not an experimentally established threshold):** add a feature block/postprocessor only when it improves the fixed primary holdout's exact AP, does not produce a material unexplained worst-fold or legitimate-hub regression, and is implementable reproducibly before the freeze. Compare paired predictions on the same rows. If the gain is tiny or comes from one scenario, inspect a paired scenario/entity bootstrap as an optional sensitivity check; an interval overlapping zero means uncertainty, not proof of no effect. Do not set a universal numeric AP-gain threshold before observing fold size, positive counts and run variation. For a final ensemble also require that it beat the strongest included single model on the primary check, with useful error complementarity and no major stress-split collapse. A failed gate retains the previous baseline; leaderboard-only improvement does not override a diagnosed leakage issue.

### Gaps
**Kaggle access coverage on 23 October 2026:** Direct browser research tool opens were attempted for original winner discussions 111284 and 111321, 21st/41st-place writeups, cdeotte's RAPIDS notebook, feature-engineering discussion 108575, retrospective 111164, leaderboard, and solution-write-up documentation. These returned `Total lines: 0` / no extractable body, not a verified authentication error. Kaggle's indexed search responses did expose substantial first-party text for the 41st-place writeup, retrospective 111164, solution-write-up documentation and partial 21st-place writeup. This note uses those indexed excerpts with that qualification. The 21st-place author's own GitHub repository and Deotte's coauthored NVIDIA article were fully readable and used as primary-source fallbacks. No claim is made to have run or inspected the blocked notebooks' full code. Broad AML/graph Kaggle search did not produce a comparable, fully verified competition solution worth importing; do not substitute unsourced portfolio performance claims for evidence.
