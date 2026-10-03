# Rules for This Project — Notebook Version

These rules are non-negotiable even though the implementation has been simplified.

## 1. Scientific integrity

- Treat AI-EMT as a synthetic, mechanistically generated benchmark calibrated against published epidemiological parameters.
- Never describe the dataset as real EHR data, real hospital operations, NHS data or real patient outcomes.
- Do not make clinical diagnosis, treatment, triage or deployment claims.
- Report predictive performance and associations, not causation.
- A feature contribution, SHAP value, missing laboratory test or hospital characteristic does not prove why an outcome occurred.
- State the synthetic-data limitation prominently in the README and relevant notebooks.

## 2. Data and provenance

- Preserve the original Kaggle files unchanged under ignored `data/raw/ai_emt/`.
- Record the source, download date, Kaggle terms, file sizes, row counts, schemas and important joins in the README or Notebook 01.
- Validate `train.csv` and `supplemental_data.csv` before combining them.
- Check duplicates, IDs, target availability and temporal coverage before modelling.
- Keep raw data, processed healthcare data, feature matrices, model weights, credentials and large generated files out of GitHub.
- Commit only a tiny synthetic demonstration sample when needed.

## 3. Targets and leakage

- Use ED presentation/initial assessment as the prediction timestamp unless the dataset documents a valid alternative.
- For `adverse_outcome`, exclude final ICU admission, death, post-admission information and any other information generated after the prediction point.
- For `readmission_30d`, exclude post-discharge and future-encounter information.
- Treat the leakage audit in Notebook 01/02 as the source of truth.
- Do not derive preprocessing parameters, encodings, feature-selection rules, hospital rates or thresholds from validation or holdout observations.
- Build rolling/cohort features only from earlier encounters and only when exact chronology makes them valid.

## 4. Validation and model selection

- Use chronological primary validation.
- Intended split: 2019–2021 development/training, 2022 validation, 2023–2024 final holdout, subject to confirmation from the downloaded data.
- Do not use a random final train/test split.
- Select preprocessing, feature sets, model parameters, calibration and ranking policies using development/validation data only.
- Evaluate the locked final approach once on the final holdout.
- Do not change the final model because of the final holdout result.

## 5. Metrics and imbalance

- Use PR-AUC as the primary discrimination metric for rare-event targets.
- Also report ROC-AUC, precision, recall, F1, Recall@K, Precision@K, Brier score, ECE, reliability curves and documented confusion matrices.
- Do not make accuracy the headline metric.
- Prefer class weighting and calibrated ranking over naïve resampling.
- If SMOTE or another resampling method is tested, apply it only inside the training portion/folds and document the comparison.

## 6. Missingness analysis

- Treat selected laboratory missingness as an analytical research question.
- Compare:
  1. imputation without indicators
  2. imputation with missingness indicators
  3. models with native missing-value handling where applicable
- Examine missingness by outcome, acuity, setting and time.
- Do not treat a missing test as direct evidence of patient condition.
- Write the conclusions as predictive/process associations, not clinical causal explanations.

## 7. Notebook discipline

- Work in Jupyter notebooks, preferably through VS Code.
- Run notebooks in numerical order.
- Start every notebook with a clear goal and end it with a concise conclusion.
- Keep code readable and grouped into functions when a function is reused inside that notebook.
- Avoid copying a large preprocessing block differently across notebooks. Reuse the same notebook logic by loading saved intermediate outputs where appropriate, or keep a clearly marked shared function section.
- Do not hide major methodological decisions inside unexplained cells.
- Save important tables and figures into `reports/` so the final README does not depend on scrolling through notebooks.

## 8. Model ladder

Use the same order as the original project:

`Constant baseline → Logistic Regression → XGBoost → LightGBM → TensorFlow MLP → TensorFlow TabTransformer`

Optional multi-task learning may be tested after the core model ladder is complete.

An advanced model is not automatically better because it is more complex.

## 9. Advanced models and ablations

- Use the same temporal partitions and feature-availability rules across model families.
- Compare deep-learning models directly against the strongest classical baseline.
- Retain an advanced model only when the validation evidence supports its additional complexity.
- Record rejected experiments as honestly as successful ones.
- Do not tune the project indefinitely. A clean comparison is more valuable than a large collection of weak experiments.

## 10. Calibration, explainability and ranking

- Fit Platt scaling or isotonic calibration on validation data only.
- Report Brier score, ECE and reliability curves.
- Describe risk-ranking outputs as **analytical prioritisation**, not medical decision support.
- Use SHAP language carefully. A feature contributed to a model prediction; it did not prove causation or clinical necessity.
- Note correlated-feature and explanation-stability limitations.

## 11. Subgroups and temporal robustness

Where sample sizes support interpretation, compare:

- age group
- sex
- hospital type
- region
- urban/rural status
- season
- day/night
- COVID regime

Do not draw strong conclusions from sparse groups.

## 12. Operational-pressure proxy

- Do not claim to observe bed occupancy, staffing demand, patient flow or actual hospital capacity unless the dataset contains those measures.
- Create an ED Operational Pressure Index only if exact timestamps or valid aggregation periods are available.
- Fit standardisation and pressure thresholds using training data only.
- Describe it as a **derived analytical pressure proxy based on encounter demand/acuity burden**.
- If the data do not support valid aggregation, omit the component.

## 13. Scope control

Stop after the final report and portfolio-ready README.

Do not add APIs, deployment, Docker, Kubernetes, Kafka, streaming, production databases, cloud infrastructure, monitoring, LLM/RAG, agents or complex CI/CD.
