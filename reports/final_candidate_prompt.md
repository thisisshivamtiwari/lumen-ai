# Prompt: build the strongest available final Lumen candidate

Act as a senior ML engineer working in `/Applications/Softwares/Lumen.ai`. Your task is to produce the strongest defensible submission package using the supplied competition data and the work already completed. Aim for 80/100, but do not claim or guarantee an unobserved leaderboard score.

## Scope and speed

Skip new validation training runs, new holdout campaigns, repeated bootstraps and broad hyperparameter searches. Reuse existing out-of-fold predictions, metrics and experiment logs. Do not delete validation artifacts, fit on held-out labels while describing the resulting scores as OOF, or conceal uncertainty. A new unvalidated candidate must be identified as unvalidated.

Keep V1 frozen. Work in a new timestamped output directory. Do not overwrite another agent's source, caches or running experiments. Snapshot the exact source before fitting. Do not upload, submit, publish or push anything; return local files for review.

## First establish the actual current state

Read `data/training/README.md`, the latest packaged README, current source and the existing experiment metrics. Inspect the completed V14 outputs before proposing work already finished. Distinguish stage-one ablations, full nested pipelines and final blends; their metrics are not interchangeable.

The current files inspected for this prompt show:

- Official score is **100 × average precision**, grouping tied scores at the same threshold. Top-1% precision and recall are separate diagnostics.
- `nh_bcg_metrics.json`: V12 stage-one mean fold AP 0.6360609, pooled AP 0.6227213, top-402 TP 352.
- `nh_bcp_metrics.json`: V14 path-feature stage-one mean fold AP 0.6395702, pooled AP 0.6293068, top-402 TP 357, top-1250 TP 807.
- V14 adds slow scatter-gather paths. The edge-of-window correction was rejected. These experiments need not be repeated.
- Packaged claims of portal scores are historical evidence; do not describe them as newly verified live results.

Check whether newer completed artifacts supersede these figures. Identify every candidate by prediction hash, source snapshot, configuration and blend recipe.

## Choose the final approach from existing evidence

1. Compare completed full-pipeline candidates using existing identical-fold OOF predictions. Recalculate AP, per-fold AP and top-1% TP/FP, precision and recall without refitting. Align by transaction hash. Use the organiser's exact budget and tie rules; never substitute K = positive count for the top-1% cutoff. Report top-1250 metrics separately.
2. Prefer a completed full pipeline supported by existing overall AP evidence. Do not promote V14 solely because its stage-one AP improved. Report existing wallet/temporal results when available and mark missing checks as not run.
3. Inspect existing prediction disagreement around the ranking head and among medium positives. Any ensemble must combine genuinely complementary models, use the same transformations at evaluation and inference, and avoid a large blend-weight search. Do not infer test labels from leaderboard feedback or prescribe family-specific test scores.
4. If no completed new pipeline has sufficient evidence, package the strongest existing reproducible candidate. If producing an unvalidated alternative, put it in a separate directory and make no claim it improves performance.

## Correctness requirements

Use only supplied observable transaction and graph information as inference inputs. No private test answers, organiser-only files, generator seeds, portal exploitation or other teams' private submissions. Do not use family, difficulty, scenario or cohort annotations as predictors. Audit existing annotation-based row selection and auxiliary targets against the competition rules; disable uncertain auxiliary annotation use in the recommended package.

Retain one economic payment per transaction and handle conversion legs consistently. For path features, verify which amounts represent sender debit versus recipient delivery; do not treat a reference-USD ratio as proof that the same funds moved through a path. Correct any discovered material bug, but label the affected candidate's old evaluation as stale.

Audit source-to-artifact consistency, nested score provenance and train/inference feature construction. Do not describe an outer-fold-protected stack as label leakage without evidence, or ignore training-side score shortcuts. Do not silently change a recipe after evaluating it.

## Build and return

Fit the selected fixed recipe on the authorised labelled training data and generate all 40,101 test scores. Use a new output directory. Check the exact expected transaction-hash set, one row per hash, finite scores in [0,1], supported CSV columns and deterministic tie handling. These file checks remain necessary even though new validation runs are skipped.

Package the exact source, pinned dependencies, commands, methodology, limitations, licence and required AI disclosure. Reuse an existing clean reproduction only if it covers the identical source and configuration; otherwise run the packaged reproduction if feasible, or explicitly mark it unverified.

Return:

1. Absolute paths to `predictions.csv`, the code ZIP and the methodology file.
2. SHA-256 hashes of the predictions and ZIP.
3. A small evidence table: candidate, full-pipeline or stage-one, mean/pooled OOF AP, top-1% TP/FP, existing holdout evidence and reproduction status.
4. The selected recipe, remaining uncertainty and whether it merits a submission based on existing evidence.

Stop after producing the files. State explicitly that no upload or submission was performed. An 80 target is an aspiration, not a fabricated result or a reason to bypass the competition rules.
