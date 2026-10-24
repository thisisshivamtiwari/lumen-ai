# AI and external assistance

Provider: OpenAI. Tool/product: Codex accessed through this ChatGPT coding session. The exact deployed model identifier, sampling temperature and hidden inference settings are not exposed to this session; no specific model version or setting is invented here.

AI assistance was used to inspect the organiser's supplied files, draft and implement label-free feature engineering, train/evaluate CatBoost, implement artifact checks and prepare this methodology/package. All computation ran locally on the supplied data. No external labels, private organiser files, generator seeds or test answers were used. No synthetic replica was mixed into training. No external AI service was used to label or score individual test transactions.

Relevant user task prompts in this work:

1. "I got the ... data and here is the data - Competition is started and this is real data." The user supplied the training and test ZIP paths.
2. "Now first read the entire read me and all from /data directory and start bulding on the strategy that we've already discussed and build the first output to submit."

Implementation instructions followed: read the complete supplied README/notebook/helper, preserve one prediction per transaction, reconstruct economic payments separately from conversion legs, use authorised full-window label-free context, keep annotation metadata out of predictors, compare feature blocks on fixed scenario/cohort grouped folds, and package an independently reproducible result. This is a record of visible task instructions, not a claim to expose system prompts or unavailable provider settings.

ML settings are fully specified in `config.json`. The random seed is 20261024; CPU thread count is four. This package has not been submitted automatically to the competition site.

No pre-trained detector is used. CatBoost, NumPy, pandas, PyArrow, SciPy, scikit-learn and joblib are third-party libraries, pinned through requirements. The organiser helper and starter-feature code are attributed in `THIRD_PARTY.md`.

## V4 update (motif features and context stage)

Provider: Anthropic. Tool/product: Claude Code (model Claude Opus 5.5). AI assistance was used to analyse validation errors by type and rank band, design and implement the multi-scale motif features (`motif.py`), the second-stage context features (`context.py`), the V4 pipeline (`run_v3.py`) and the grouped/nested evaluation scripts in `experiments/`, and to update this package's README. All computation ran locally on the supplied data. No external labels, organiser-private files, generator seeds or test answers were used, and no AI service labelled or scored individual test transactions.


V5-V11 updates (ego-network features, hard-tier fitting selection, run blending, burst-context features and the time-based confirmation split) were likewise implemented with Claude Code (Anthropic; models Claude Opus 5.5 and Claude Fable 5.1). The burst-context feature design followed a written specification supplied by the team. The experimental `--aux` option (family annotation as an auxiliary training target) is not used in this submission.

V14 (slow scatter-gather path features; rejected edge-context features) was implemented with Claude Code (Anthropic; Claude Fable 5.1) from a written specification supplied by the team.

V15 (final blend, README consolidation and rules audit) was prepared with Claude Code (Anthropic; Claude Fable 5.1).
