# Ranking objectives, review budgets and promotion governance

Read-only research completed 24 October 2026, approximately 13:13 Europe/London. Current time was checked through the clock tool (12:12:53 UTC). No model was trained, no active agent files were changed, and no submission action was taken. The deadline plan below assumes the supplied 16:30 London deadline remains in force.

## 1. What did the latest scores actually improve, and where is headroom?

### Takeaway

V4's saved OOF scores show a substantial improvement over V1, but almost all AP gain still comes from easy cases. The next decision must separate whole-ranking AP, the actual 402-payment review budget, and the exploratory 1,250-payment budget. The older differently named table is reproducible from the same V1 predictions; its differences do **not** establish that it came from another run.

### Cited Findings

**Metric definition.** AP is the sum of threshold precision weighted by recall increments; it is not trapezoidal PR-AUC. With unique scores and binary labels, this becomes `AP = (1/P) sum_{r:y_r=1} TP(r)/r`. Moving positives up anywhere can help AP; merely changing a numerical threshold cannot change AP or top-k membership. PR is particularly informative under skewed class distributions. [Official sklearn AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html); [Davis and Goadrich, ICML 2006](https://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf).

**Direct recalculation from saved files.** All figures below were recomputed by aligning transaction hashes in [frozen V1 OOF](/Applications/Softwares/Lumen.ai/outputs/strategy3_v1/graph_oof.csv), [saved V4 OOF](/Applications/Softwares/Lumen.ai/outputs/v3_context/blend_motif_context_oof.csv), and [training labels](/Applications/Softwares/Lumen.ai/data/training/train/labels.csv). Labels/annotations were used only for analysis. The V4 figures are **local validation**, not an independently verified portal score.

| Local OOF model | Mean fold AP | Pooled AP | TP at 402 | FP at 402 | Precision at 402 | Recall at 402 | TP at 1,250 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen V1 | 54.4829% | 53.7186% | 326 | 76 | 81.0945% | 26.0800% | 699 |
| Saved V4 blend | 61.6162% | 61.5271% | 355 | 47 | 88.3085% | 28.4000% | 785 |

Both sets contain 40,131 rows and 1,250 positives. Therefore `k_1pct = ceil(.01*N)=402`; `precision@k=TP/k`, `recall@k=TP/1250`. At the **different** exploratory cutoff `k=1250`, precision and recall happen to be equal because `k=P`. This does not make 1,250 the competition's operating budget. [OOF sources above; reporting implementation](/Applications/Softwares/Lumen.ai/outputs/fp_fn_report/source/fp_fn_report.py).

| Category | Total positives | V1 TP at 402 | V4 TP at 402 | V1 TP at 1,250 | V4 TP at 1,250 | V4 FN below 1,250 |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 800 | 321 | 339 | 671 | 755 | 45 |
| Medium | 225 | 5 | 16 | 24 | 26 | 199 |
| Observable-hard | 200 | 0 | 0 | 4 | 4 | 196 |
| Ambiguous | 25 | 0 | 0 | 0 | 0 | 25 |
| Total | 1,250 | 326 | 355 | 699 | 785 | 465 |

At V4 ranks 403–1,250, there are **430 positives and 418 legitimate payments**: 416 easy, 10 medium, 4 observable-hard, no ambiguous positives. Below 1,250, 189 positives occur at 1,251–5,000, 99 at 5,001–10,000 and 177 below 10,000. This makes both head reordering and the medium tail concrete targets. [Saved V4 OOF](/Applications/Softwares/Lumen.ai/outputs/v3_context/blend_motif_context_oof.csv); [labels](/Applications/Softwares/Lumen.ai/data/training/train/labels.csv).

The 47 V4 top-402 FPs are: **bipartite 22; fan_out_in 9; chain 7; stack 3; multi_stage 2; background 2; distributed_relay 1; aggregate_relay 1**. The first three groups supply 38/47=80.9% of head FPs. Do not continue describing aggregate relay as the principal *current* head FP source: it supplied 12 V1 head FPs and now supplies one. It remains difficult elsewhere. [V1/V4 OOF and labels above].

**Reconcile naming before drawing conclusions.** The older report maps `fan_out_in + multi_stage` to “Scatter-gather” and `chain + stack + distributed_relay` to “Layered stack.” That mapping yields V1 head counts of 17 scatter-gather FPs and 21 layered-stack FPs, exactly matching raw family counts 8+9 and 16+4+1. It also yields 351 suspicious scatter-gather rows and 287 layered-stack rows, while combining very different easy/medium/hard regimes. The difference between 699 and 785 catches is V1 versus V4; the difference between 326 and 699 is the same V1 score file at two budgets. [Exact mapping and cutoff code](/Applications/Softwares/Lumen.ai/outputs/fp_fn_report/source/fp_fn_report.py); [V1 type table at 402](/Applications/Softwares/Lumen.ai/outputs/fp_fn_report/type_table_top402.csv); [V1 type table at 1,250](/Applications/Softwares/Lumen.ai/outputs/fp_fn_report/type_table_top1250.csv).

**AP contribution calculation.** Using each category's contribution to the full-ranking AP sum, rather than AP on slices with different prevalence, V4–V1 pooled AP gain of 7.8085 percentage points decomposes into **easy +6.5036; medium +1.2040; observable-hard +0.1000; ambiguous +0.0008**. V4 medium TP at 1,250 rises only two, despite its meaningful AP contribution gain: this demonstrates why one cutoff cannot explain the whole AP movement. This is a descriptive decomposition, not causal attribution. [Unique-score OOF artifacts and AP formula above].

### Inferences

- Headroom at the existing head budget is bounded: replacing all remaining 47 FPs could add at most 47 catches, **+11.69 precision points and +3.76 recall points**. Even a perfect 402-payment head cannot recall more than 402/1250=32.16% of all positives. These are arithmetic ceilings, not predicted attainable improvements.
- “The remaining leaderboard gap must come from below the top 1%” is too strong. AP can improve through order changes *within* the first 402 even with identical P@402, as well as through tail changes. Displayed portal percentages may also be rounded. One successful hidden-test gain is encouraging, but does not prove future local gains will transfer.
- Concentrate on evidence that separates lawful high-scoring bipartite/fan-out-in/chain instances from positives across multiple held-out groups. Do not suppress an entire family based on its annotation. Do not claim two inspected lawful examples are provably indistinguishable from positives under every possible observable feature.
- Keep a raw `family × difficulty` table alongside any presentation names. “Layered stack” currently combines ordinary chain/stack patterns with hard distributed relays, making a single proposed fix misleading.

### Gaps

- Portal V4 score 66.5 and category percentages are claims from the supplied log, not newly authenticated here. No hidden-test labels or transaction-level hidden-test explanations were accessed.
- The 25 ambiguous labels are described as independent of available observables in the supplied dataset README. This is a limitation of this release, not a proof that every real-world ambiguous payment is intrinsically unlearnable. [Dataset README](/Applications/Softwares/Lumen.ai/data/training/README.md).

## 2. Which ranking changes are justified beyond the active ego/context work?

### Takeaway

Do not duplicate the active agent's ego/context work. The strongest distinct small experiment is bounded hard-negative emphasis obtained exclusively inside each training fold. Direct AP-aware ranking is a secondary experiment only if time remains after validation corrections; wider parameter or ensemble searches are poorly justified by the available results.

### Cited Findings

**What is already low-value.** Five-seed V1 averaging consumed about 668 seconds and increased mean AP only **0.1226 points**, pooled AP **0.1415 points**, while TP@402 fell 326→324. Longer motif training decreased mean AP to 59.0564% compared with motif's 59.7726%. Burst context mean AP is 60.6192%, below normal motif-context's 61.1432%. These specific experiments do not justify additional seed/depth/iteration grids. [Seed experiment](/Applications/Softwares/Lumen.ai/outputs/v3_context/seedbag_metrics.json); [long-training result](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_long_metrics.json); [burst-context result](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_context_burst_metrics.json).

Uniform positive weighting ×3 already reduced mean AP **54.4829→53.0096**, TP@402 **326→313**. Increasing the weight of *all* positives is not a remedy demonstrated by these data. [Previous fixed experiments](/Applications/Softwares/Lumen.ai/outputs/run1_investigation/comparison.csv).

**Complementarity exists but is not an oracle model.** Motif alone catches 349 positives in the top 402; context alone catches 354. Their union contains 416 distinct positives, including 62 caught only by motif and 67 only by context. The blend catches 355. The union is selected using known labels and is **not an achievable 416-catch prediction or deployable strategy**; it simply suggests diverse errors. A new blend must be assessed on the same outer-held-out records, with its rule fixed inside training data. [Motif OOF](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_oof.csv); [context OOF](/Applications/Softwares/Lumen.ai/outputs/v3_context/motif_context_oof.csv).

CatBoost supports nonnegative per-object `weight` in Pool. Its ranking API also supports explicit pairs. `YetiRank:mode=MAP` is available on CPU; using evaluation metric `MAP` alone as if it were an optimizable loss is incorrect. PairLogit optimizes a pairwise surrogate and does not itself establish AP optimization. [Official Pool documentation](https://catboost.ai/docs/en/concepts/python-reference_pool); [official ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking). Metric-specific ranking surrogates are supported by an original comparison of learning-to-rank methods; that paper does not supply evidence for AML data or this competition. [Lyzhin et al., ICML 2023](https://proceedings.mlr.press/v202/lyzhin23a.html).

**Candidate HN1 — small, capped hard-negative emphasis (proposed; not run).** Freeze the existing best admissible feature/model procedure. For each outer fold:

1. Within its training partition only, generate **inner group-OOF** reference scores. Do not reuse global OOF scores if their producing models saw outer-validation labels.
2. Give weight 2 to legitimate training rows whose inner score is in the top 3.2% of that inner scored population. All positive rows and other negative rows retain weight 1. Optionally normalize all weights by their overall mean; the rule must be frozen before evaluation.
3. Fit exactly one weighted candidate, with the existing features and hyperparameters. If the chosen reference is a two-stage model, rebuild its full label-derived context inside the same outer-training boundary.
4. Evaluate AP and head catches on the untouched outer fold. Check whether it consistently demotes legitimate bipartite/fan-out-in/chain look-alikes without losing more positives. Use annotations only afterward to explain the outcome.

Formula: `w_i = 1 + 1[y_i=0 and inner_percentile(s_i)>=0.968]`. The factor 2 and cutoff 3.2% are a single predeclared hypothesis, not a search grid. Do not choose them from candidate outer errors. This differs from failed ×3 positive weighting because it targets a limited set of confusing *negative training examples*, and differs from hard-coded family penalties because its decisions do not consume family/difficulty annotations. Supported implementation mechanism: [CatBoost Pool weights](https://catboost.ai/docs/en/concepts/python-reference_pool).

**Candidate R1 — objective alignment, lower priority (proposed; not run).** If HN1 and active work leave adequate verification time, try one `CatBoostRanker(loss_function='YetiRank:mode=MAP')` configuration using the same label-free feature matrix and outer splits, with binary relevance `is_suspicious`. Treat the complete training partition as one ranking query and the held-out partition as one evaluation query to align with whole-window AP. Do **not** use `family`, `difficulty`, `scenario_id` or `cohort_id` as query definitions; this would silently change the learning problem and can introduce annotation-based training assistance. Give the smoke test a strict memory/runtime cap. If one large query is impractical, defer rather than arbitrarily subdividing into scenario queries. Head-truncated MAP is not full competition AP; report the actual objective string and full sklearn AP. [Official ranking documentation](https://catboost.ai/docs/en/concepts/loss-functions-ranking); [Lyzhin et al.](https://proceedings.mlr.press/v202/lyzhin23a.html).

### Inferences

- HN1 is the preferable independent experiment because the current head has 47 repeated lawful look-alike errors. It may still fail by sacrificing genuine fast patterns; the original gates must decide. It is a hypothesis, not a promised fix.
- R1 would test whether a different training objective can use existing features better. It cannot recover information absent from those features and could worsen calibration or regress generalization. Keep it to one candidate, not a new tuning campaign.
- Do not perform another broad blending sweep. First measure complementarity at *the actual budget*, and use a predeclared equal-weight rule or a rule learned solely on inner folds. Re-optimizing weights over the repeatedly inspected outer OOF set creates another selected model.
- A rank-blend score around 0.99 means a high combined percentile; it does **not** mean a calibrated 99% laundering probability. A strictly monotonic scalar calibration preserves order, so cannot improve AP/head catches; isotonic calibration introduces ties and may change them. [Official sklearn calibration discussion](https://scikit-learn.org/stable/modules/calibration.html).

### Gaps

- Neither HN1 nor R1 has been implemented or validated in this research. Runtime of a single 26k-row ranking query is unknown here.
- Cached context scores require a complete dependency audit before use for inner hard-negative mining. A filename containing “OOF” is insufficient evidence that a score excludes the current outer validation labels.

## 3. What must be corrected in evaluation, and when should work freeze?

### Takeaway

The large mean-fold V4 gain survives the score-normalization audit; its quoted bootstrap support does not yet target the right comparison. Recompute evidence for the exact shipped procedure, cap further experiments, preserve every original gate, and leave sufficient time to reproduce and package the chosen artifact without submitting it.

### Cited Findings

**Score-mapping audit.** Saved V4 exactly reconstructs as an equal blend of motif and context percentile ranks **computed separately inside each outer fold**, followed by a raw-score tie break. V4 inference instead computes percentiles over the single test window. Per-fold ranking is defensible for evaluating a procedure applied to one held-out window, but pooled OOF numbers mix separate fitted models and require explicit disclosure. [V4 inference blend implementation](/Applications/Softwares/Lumen.ai/outputs/v4_motif_context/lumen_v4_code/run_v3.py); [saved V4 OOF](/Applications/Softwares/Lumen.ai/outputs/v3_context/blend_motif_context_oof.csv).

| Read-only score representation | Mean fold AP | Pooled AP | TP@402 |
|---|---:|---:|---:|
| V1 raw probabilities | 54.4829% | 53.7186% | 326 |
| V1 fold-percentile transform | 54.4829% | 54.1097% | 328 |
| Motif raw probabilities | 59.7726% | 58.6919% | 349 |
| Motif fold-percentile transform | 59.7726% | 59.5692% | 353 |
| Context raw probabilities | 61.1432% | 59.3156% | 354 |
| Context fold-percentile transform | 61.1432% | 60.9401% | 361 |
| Saved V4: within-fold rank blend + tie break | 61.6162% | 61.5271% | 355 |
| Diagnostic global-pool rank blend + tie break | 61.6909% | 60.2800% | 356 |

This table is a procedure sensitivity check, **not a set of newly eligible model candidates**. Global-pool blending mixes models with different score scales; it is not a replacement estimate for applying the blend to a single independent window. However, the 1.2471-point pooled AP difference between blend mappings is material, so 61.5271 must not be presented as protocol-independent. The primary original mean-fold V4–V1 gain is **7.1333 AP points**, not explained by V1 percentile normalization, which leaves mean-fold AP unchanged. [Sources immediately above; AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

**Tie changes are real model changes.** The no-epsilon within-fold blend has 23,801 unique scores and pooled AP 61.5238%; the shipped epsilon tie break produces 40,131 unique scores and 61.5271%. Thus it changes pooled AP by approximately **0.0033 points**. The increment is tiny here, but it must be disclosed as a deterministic secondary ordering signal rather than “unchanged ranking.” The rank step exceeds epsilon in these data, so non-tied rank sums retain their order. [Blend implementation](/Applications/Softwares/Lumen.ai/outputs/v4_motif_context/lumen_v4_code/run_v3.py).

**Bootstrap is stale.** The current bootstrap script loads **`context_oof.csv`**, not `motif_context_oof.csv` or saved V4, and compares its old 50/50 blend with **fold-percentile-ranked V1**, not the untouched raw V1 scores. Its confidence interval therefore cannot be cited as a confidence interval for V4 versus frozen raw V1. [Bootstrap source](/Applications/Softwares/Lumen.ai/outputs/v3_context/source/bootstrap.py).

**Reporting metadata is stale.** The V4 ZIP's `validation_summary.json` reports model “Strategy3 graph-enhanced CatBoost v1”, mean AP54.4829% and V1 temporal diagnostics. This can serve as historical baseline evidence if labelled accordingly, but it is not V4 holdout evidence. [Packaged validation summary](/Applications/Softwares/Lumen.ai/outputs/v4_motif_context/lumen_v4_code/validation_summary.json).

Model-selection overfit can be as large as apparent differences between algorithms. Repeatedly choosing features, weights, depth, seeds and blends on the same folds introduces selection bias; a conditional bootstrap of saved predictions does not undo this. Selection belongs inside the fitting procedure when estimating an unbiased final performance. [Cawley and Talbot, JMLR 2010](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf).

### Inferences

**Evidence to recompute before any promotion statement:** align exact raw V1 and exact candidate OOF hashes; assert original fold IDs and no group overlap; hash the score-producing source and configuration; report per-fold and pooled AP separately; bootstrap the **specified mean-of-fold-AP delta** with paired scenario/cohort blocks sampled within folds, alongside the descriptive pooled delta. State whether scores are re-ranked during bootstrap—default keep the fixed saved procedure's scores and do not search an ordering rule during resampling. The CI describes conditional uncertainty of saved predictions, not all training/model-selection uncertainty. Repeated bootstrap groups do not substitute for seed confirmation.

Wallet/temporal checks must refit the **entire** score-generating and stacking pipeline under their own training boundaries. Reusing first-stage scores from a model fitted with the holdout's labels, then withholding those labels only from the final classifier, is not a valid holdout. Likewise the already completed five-seed V1 experiment is not confirmation of V4 or of a new weighted candidate; the candidate's feature/stack/blend procedure needs its own predeclared confirmation run. A selected OOF improvement remains optimistic to an unknown degree after repeated candidate comparison; even fully passing diagnostics provides no guarantee of hidden-test gain.

| Priority | Action | Why now | Stop rule |
|---|---|---|---|
| 0 | Correct exact-artifact bootstrap and full-pipeline B/C audit | Required evidence is stale/missing, despite strong mean-fold V4 gain | Do not assert promotion gates passed without it |
| 1 | Finish active ego/context experiment | Already in flight; avoids duplicating work | No extra scale/depth search; fixed original gates |
| 2 | One HN1 trial | Targets 47 head false alarms using only nested training labels | Reject after fixed folds if weak; no weight/cutoff sweep |
| 3 | One MAP-objective trial, only with spare time | Distinct training-objective hypothesis | Runtime/memory smoke cap; otherwise defer |
| 4 | Freeze, confirmation, reproducible package | Prevents a last-minute unverified regression | No further model edits after freeze |

| Required decision evidence | Retain original requirement |
|---|---|
| Original fixed grouped folds | Same memberships and identical target rows |
| AP gain against frozen V1 | At least +1.0 percentage point under the originally specified primary statistic |
| Fold consistency | Improvement in at least 2/3 folds |
| Paired group-bootstrap gain | Lower confidence bound above zero for the exact candidate/reference pair |
| Seed stability | Positive gain under the prescribed confirmation seed; never choose the best seed afterward |
| Wallet and temporal checks | No regression larger than 2 AP points; no reuse portrayed as untouched validation |
| AP and head budget | Report AP, TP/FP/P/R@402 and per-fold top134; never approve from a difficulty slice alone |
| Current incumbent comparison | Show incremental candidate–V4 benefit too; clearing V1 alone does not justify replacing an already better admissible incumbent |
| Integrity/reproduction | Labels excluded from features, correct hash coverage, frozen V1 manifest unchanged, clean executable package reproduces predictions |

For V4, original-fold top134 catches are **124,121,111** versus V1 **122,108,98**; summing V4 fold budgets gives 356, whereas globally pooled top402 gives355. This is expected because the two review-allocation rules differ. Report both explicitly, never interchange them. [V1/V4 OOF sources above].

**Time-boxed handoff for today (London time, 24 October):**

1. **13:15–14:15:** finish active ego/context work; correct score/provenance/bootstrap metadata in a new research directory. Permit at most HN1 as the independent extra candidate if compute is available. Pre-register it before running. Stop new architecture/depth/blend grids.
2. **14:15–15:00:** freeze the candidate list and rule; run prescribed confirmation seed, wallet and temporal diagnostics on the chosen candidate. These are regression diagnostics if already inspected repeatedly, not an untouched final-test claim. If gates cannot be completed, retain the latest fully evidenced incumbent.
3. **15:00–15:45:** hard feature/model freeze. Run clean end-to-end reproduction from the actual ZIP, regenerate precise methodology/AI disclosure, validate all required artifacts and hashes, check public-material/submission requirements already identified by the team. Avoid training contention that prevents reproduction finishing.
4. **15:45–16:00:** produce the final reviewable files and manifest for the user. Preserve the rest of the pre-deadline interval for upload/portal failures **only if explicitly authorized**. No automatic upload/submission is authorized by this research request.

**Stop criteria:** discontinue any candidate below the fixed AP threshold, with fewer head catches and no justified overall ranking improvement, that fails a confirmation seed or holdout gate, whose inner labels cannot be proven isolated, or that cannot be reproduced within the freeze window. Do not spend the remaining time chasing ambiguous labels, hand-adjusting family ranks, repeatedly querying the leaderboard, extending training after a rejected capacity test, or interpreting percentile scores as risk probabilities.

### Gaps

- No current V4 wallet/temporal result, confirmation-seed result or correct V4-versus-original-V1 paired confidence interval was found in the scoped artifacts. Existing V1 values in the V4 package do not fill that gap.
- The final 16:30 deadline and portal acceptance requirements were provided by project context, not newly authenticated through the portal in this read-only research. Confirm administratively without using hidden-test diagnostics for tuning.
