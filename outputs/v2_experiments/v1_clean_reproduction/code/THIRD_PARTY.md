# Sources and assistance

- `payments.py` is copied unchanged from the organiser's training download.
- `starter_features.py` extracts the `make_features` cell from the organiser's `starter_notebook.ipynb`; the feature definitions are unchanged. Imports and a provenance docstring were added.
- Original audit, baseline, feature, evaluation, reproduction and verification code was prepared with OpenAI Codex assistance in this session. It uses pandas, NumPy, PyArrow, SciPy, scikit-learn, CatBoost and joblib. See AI_DISCLOSURE.md for visible task instructions and available settings.
- Annotation fields from training labels are used to form evaluation groups and error slices, never as predictor features. All graph/context features here are label-free and computed over each supplied window independently, consistent with the release's offline investigation rule.

This records attribution; it does not relicense organiser material.
