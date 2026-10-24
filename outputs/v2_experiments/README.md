# V2 research result: retain V1

V1 reproduced byte for byte. Five predeclared alternatives were evaluated; the best gain (+0.132 AP percentage points) is inconclusive and fails promotion. No V2 predictions were generated. No submission, upload or publication occurred. All 45 protected V1 files remain unchanged.

- [Methodology and decision](methodology.md)
- [Methodology PDF](methodology.pdf)
- [Baseline reproduction](baseline_reproduction_report.md)
- [Validation audit](validation_audit.md)
- [Error analysis](error_analysis.md)
- [Experiment registry](experiment_registry.csv)
- [Model comparison](model_comparison.csv)
- [Feature ablations](feature_ablation.csv)
- [Seed, wallet, time and history-loss diagnostics](confirmation_results.csv)
- [Acceptance evidence](acceptance.json)
- [Source/configuration/report ZIP](lumen_strategy3_v2_research_code.zip)
- [Exact clean-workspace reproduction commands](source_v2/README.md)
- [Archive validation](research_archive_validation.json)
- [V1 preservation check](v1_preservation_check.json)

All AP results are local. No independent labelled final holdout is available: the reserved B1 protocol has uninspected metrics, but its labels already entered Track A. Important weaknesses are medium/hard detection, shared training-validation wallets, realistic relay false positives, unseen batches and missing history. Removing 30% of economic events drops V1 AP by 15.47 percentage points on the same retained temporal targets. Supplied competition windows are complete; this is a stress test.

The ZIP includes source, exact configurations, complete visible task prompt, AI attribution, methodology and aggregate evidence. It excludes datasets, environments, fitted weights and per-transaction training/OOF files. The actual V1 submission artifacts remain in ../submission_v1/. Future release requires team/version metadata and user authorization.
