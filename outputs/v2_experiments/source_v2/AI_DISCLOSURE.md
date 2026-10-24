# AI and external assistance

Provider: OpenAI. Tool/product: Codex accessed through this ChatGPT coding session. The exact deployed model identifier, sampling temperature and hidden inference settings are not exposed to this session; no specific model version or setting is invented here.

AI assistance was used to inspect the organiser's supplied files, draft and implement label-free feature engineering, train/evaluate CatBoost, implement artifact checks and prepare this methodology/package. All computation ran locally on the supplied data. No external labels, private organiser files, generator seeds or test answers were used. No synthetic replica was mixed into training. No external AI service was used to label or score individual test transactions.

Relevant user task prompts in this work:

1. "I got the ... data and here is the data - Competition is started and this is real data." The user supplied the training and test ZIP paths.
2. "Now first read the entire read me and all from /data directory and start bulding on the strategy that we've already discussed and build the first output to submit."

Implementation instructions followed: read the complete supplied README/notebook/helper, preserve one prediction per transaction, reconstruct economic payments separately from conversion legs, use authorised full-window label-free context, keep annotation metadata out of predictors, compare feature blocks on fixed scenario/cohort grouped folds, and package an independently reproducible result. This is a record of visible task instructions, not a claim to expose system prompts or unavailable provider settings.

ML settings are fully specified in `config.json`. The random seed is 20261024; CPU thread count is four. This package has not been submitted automatically to the competition site.

No pre-trained detector is used. CatBoost, NumPy, pandas, PyArrow, SciPy, scikit-learn and joblib are third-party libraries, pinned through requirements. The organiser helper and starter-feature code are attributed in `THIRD_PARTY.md`.

V2 research extension: the complete user mission is preserved verbatim in MISSION.txt. AI assistance implemented independent reproduction/integrity checks, fixed Track A/B/C/D evaluations, a finite five-candidate comparison, chronological-flow/local-diversity features, controlled unit fixtures, seed and history-loss stress tests, and the decision report. Tool sampling parameters remain unexposed. Model settings, fixed split seeds and the predeclared promotion policy are in experiment_plan.json, experiment_config.json and locked_splits.npz (regenerable by audit.py from public inputs). Unit fixtures contain fabricated numeric arrays for software checks only; they are never model training data. Test labels were never available. No external transaction or label source was used. Public rules at https://hackathon.ukcbt.org/terms were inspected; no upload or publication is authorized by this research mission.
