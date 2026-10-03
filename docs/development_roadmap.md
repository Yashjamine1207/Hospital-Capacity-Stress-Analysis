# Hospital Emergency Department Operational Risk Intelligence
## Simplified Notebook-First Development Roadmap

## Purpose

Build a rigorous healthcare Data Science portfolio project that predicts **adverse outcome** and **30-day ED readmission** from the AI in Emergency Medicine – Turkiye (AI-EMT) dataset.

The project keeps the original analytical ambition but removes unnecessary software engineering overhead. The implementation is centred on Jupyter notebooks, reproducible notebook sections, saved result tables and final figures.

**Project subtitle:** Predicting High-Acuity Outcomes, Readmission Risk and Operational Pressure from 1 Million Emergency-Department Encounters.

The AI-EMT dataset is a realistically simulated, mechanistically generated benchmark calibrated against published epidemiological parameters. It is **not real EHR data** and must never be presented as evidence about real hospitals, NHS operations, patient care or clinical decision-making.

## Final claim

> This project investigates large-scale emergency-department risk prediction under class imbalance, informative laboratory missingness and temporal/institutional heterogeneity, comparing statistical models, gradient boosting and TensorFlow deep-learning models using leakage-safe chronological validation, calibration, explainability and out-of-sample evaluation.

## What stays the same

- Dataset: Kaggle AI in Emergency Medicine – Turkiye (AI-EMT)
- Primary targets: `adverse_outcome` and `readmission_30d`
- Intended chronological split: 2019–2021 development/training, 2022 validation, 2023–2024 final holdout, subject to confirmation from the downloaded data
- Missing laboratory values remain an explicit MNAR research question
- Logistic Regression remains the interpretable baseline
- XGBoost and LightGBM remain the main boosted-tree models
- TensorFlow/Keras MLP remains the main neural baseline
- TensorFlow TabTransformer remains the advanced tabular challenger
- Optional multi-task learning remains an experiment only if time and labels support it
- PR-AUC remains the primary discrimination metric for rare outcomes
- Brier score and ECE remain important probability-quality metrics
- Recall@K and Precision@K remain the main prioritisation metrics
- SHAP, subgroup analysis, temporal analysis and error analysis remain in scope
- The ED Operational Pressure Index is included only when the dataset supports valid temporal aggregation

## What is removed

The following are **not part of the core project**:

- `src/` Python package architecture
- `configs/` YAML configuration files
- standalone training/evaluation scripts
- MLflow experiment infrastructure
- extensive unit-test suites
- Docker or Docker Compose
- FastAPI or Flask
- AWS, Terraform or other cloud infrastructure
- Kafka or streaming
- Kubernetes
- production database/application layers
- monitoring and drift infrastructure
- LLM/RAG or agent systems
- authentication and live clinical scoring

Notebook code should still be organised into clear reusable functions inside the relevant notebook where practical. The goal is reproducibility without creating a second software project inside the Data Science project.

## Scope and boundaries

### In scope

- Two binary prediction tasks: `adverse_outcome` and `readmission_30d`
- Data quality audit and exploratory data analysis
- Laboratory missingness/MNAR analysis
- Point-in-time feature engineering
- Institutional, geographical and temporal comparisons
- Logistic Regression
- XGBoost
- LightGBM
- TensorFlow/Keras MLP
- TensorFlow TabTransformer
- Controlled imbalance experiments
- Probability calibration
- Risk ranking and capacity analysis
- SHAP explainability
- Subgroup and temporal robustness analysis
- False-positive and false-negative analysis
- Conditional ED Operational Pressure Index analysis
- Final research-style reporting through the README and notebook outputs

### Out of scope

- Clinical diagnosis, treatment recommendations or triage automation
- Claims about real-world clinical effectiveness
- Claims about NHS hospital performance
- Bed-occupancy forecasting
- Actual hospital-capacity measurement
- Production software or deployment

## Dataset and provenance

The project uses the AI-EMT dataset covering 2019–2024. The original project description records 1,000,000 ED encounters, 135 encounter features and coverage across 25 cities/all NUTS-2 regions of Türkiye.

| File | Rows | Columns | Use |
|---|---:|---:|---|
| `train.csv` | 700,000 | 135 | Main labelled training/development data |
| `supplemental_data.csv` | 100,000 | 135 | Additional labelled data; verify compatibility before inclusion |
| `test.csv` | 200,000 | 132 | Competition data; do not use as the project holdout unless labels are legitimately available |
| `hospital_reference.csv` | 25 | 13 | Hospital/infrastructure context |
| `icd10_reference.csv` | 21 | 7 | Diagnosis reference/context |

Use the 800,000 labelled encounters only after checking schema compatibility, duplicate/ID overlap, target availability and temporal coverage.

The original documentation lists 29 laboratory variables and several selectively ordered tests with substantial missingness. The approximate missingness values in the source plan must be recalculated from the actual downloaded data before being reported as results.

## Non-negotiable scientific rules

- Use only information available at ED presentation/initial assessment for predictive models.
- Audit every candidate feature for prediction-time availability.
- Exclude post-outcome and post-discharge information.
- Use chronological validation, not a random final train/test split.
- Select features, preprocessing, hyperparameters, calibration and ranking policies using development/validation data only.
- Touch the final holdout only once for the locked final evaluation.
- Build any historical or rolling signal only from strictly prior encounters.
- Report associations and predictive performance, not causation.
- Keep illustrative operational or cost assumptions separate from observed model metrics.
- State the synthetic-data limitation prominently in the README and final notebooks.

# Notebook Plan

The old 15-notebook structure is reduced to **9 main notebooks**. Each notebook should finish with saved figures/tables and a clear written conclusion.

## Notebook 01 — Data Audit, Targets and Chronological Split

**File:** `notebooks/01_data_audit_targets_and_split.ipynb`

### Goal
Confirm the actual dataset structure before modelling.

### Work

- Load the AI-EMT files
- Compare train and supplemental schemas
- Check row counts and duplicate IDs
- Verify target availability and target definitions
- Inspect date/year coverage
- Check categorical values, numerical ranges and obvious invalid values
- Measure target prevalence directly from the data
- Identify candidate reference-table joins
- Create the chronological development/validation/holdout split
- Display the final split sizes and date ranges
- Document the prediction timestamp

### Output

- Data-quality summary table
- Target-prevalence plots
- Year/regime distribution plot
- Final train/validation/holdout table
- Initial leakage candidate list

### Completion condition

The actual data support the intended split and no model training starts before the target and split definitions are documented.

## Notebook 02 — EDA, Missingness and Leakage-Safe Features

**File:** `notebooks/02_eda_missingness_and_features.ipynb`

### Goal
Understand the data and create the model-ready feature matrix without leakage.

### Work

- Demographic EDA
- Vital-sign EDA
- Laboratory EDA
- Comorbidity and severity-score EDA
- Hospital/region EDA
- Missingness rates and missingness patterns
- Missingness by target, acuity, year/regime and hospital context
- Identify selectively ordered laboratories
- Create missingness indicators for selected laboratory variables
- Compare simple imputation strategies
- Create validated derived features such as:
  - Shock Index = HR / SBP
  - Modified Shock Index = HR / MAP
  - Anion Gap = Na - (Cl + HCO3)
- Add carefully justified interactions where inputs exist and prediction-time availability is defensible
- Remove post-outcome/post-discharge leakage fields
- Finalise feature groups for all models

### Output

- EDA figures
- Missingness heatmap
- Missingness-by-outcome plots
- Feature summary table
- Leakage-safe feature list

### Completion condition

One consistent feature definition is used for every model family unless an experiment explicitly changes the feature treatment.

## Notebook 03 — Classical Machine Learning: Logistic Regression, XGBoost and LightGBM

**File:** `notebooks/03_logistic_xgboost_lightgbm.ipynb`

### Goal
Build the complete classical ML model ladder.

### Models

1. Constant-prevalence baseline
2. Regularised Logistic Regression with class weighting
3. XGBoost
4. LightGBM

### Work

- Use the same chronological split and permitted feature set
- Apply preprocessing consistently
- Train both primary targets
- Use reasonable hyperparameter tuning on development/validation data
- Compare model discrimination and probability quality
- Track training time and basic model complexity

### Metrics

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1
- Brier score
- ECE where implemented
- Recall@K
- Precision@K

### Output

- One model-comparison table for both targets
- PR curves
- ROC curves
- Initial ranking results
- Written comparison of Logistic Regression, XGBoost and LightGBM

## Notebook 04 — Imbalance, Ranking and Model Comparison

**File:** `notebooks/04_imbalance_ranking_and_model_comparison.ipynb`

### Goal
Investigate the imbalanced targets without contaminating the validation design.

### Work

- Compare class weighting strategies
- Test threshold/ranking policies
- Test any resampling only inside training data/folds
- Do not apply SMOTE to the full temporal dataset
- Compare top 1%, 5% and 10% analyst-review capacities
- Calculate Recall@K, Precision@K, events captured and false-positive burden
- Compare the strongest Logistic/XGBoost/LightGBM configurations
- Identify the strongest boosted baseline before deep learning

### Output

- Imbalance comparison table
- Recall@K plot
- Precision@K plot
- Capacity/ranking table
- Selected baseline model for each target

## Notebook 05 — TensorFlow/Keras MLP

**File:** `notebooks/05_tensorflow_mlp.ipynb`

### Goal
Test whether a neural tabular model adds value beyond the strong boosted baseline.

### Architecture

- Numerical feature normalisation
- Categorical embeddings where appropriate
- Dense hidden layers
- Batch normalisation
- Dropout
- Early stopping
- Class-aware loss or class weighting

### Work

- Use the same chronological split
- Use the same target definitions
- Train both targets
- Compare with the selected boosted baseline using the same evaluation metrics
- Keep training settings simple enough to repeat
- Run only a small number of architecture/hyperparameter experiments

### Decision rule

The MLP is retained only when it gives a meaningful and reproducible improvement in the pre-specified evidence. Otherwise it is recorded as a valid rejected challenger.

## Notebook 06 — TensorFlow TabTransformer

**File:** `notebooks/06_tensorflow_tabtransformer.ipynb`

### Goal
Test the advanced categorical/numerical tabular deep-learning challenger.

### Work

- Build categorical token/embedding representations
- Combine them with numerical features
- Implement the TensorFlow TabTransformer architecture
- Use early stopping and class-aware training
- Keep the experiment budget controlled
- Evaluate on exactly the same temporal partitions and metrics

### Decision rule

Complexity is not treated as success. The TabTransformer is retained only if the validation evidence justifies its additional complexity and it remains defensible on the locked holdout.

## Notebook 07 — Calibration, Risk Ranking and SHAP

**File:** `notebooks/07_calibration_ranking_and_shap.ipynb`

### Goal
Turn the selected model's scores into credible probability and explainability results.

### Work

- Compare uncalibrated probabilities with Platt scaling and isotonic calibration
- Fit calibration using validation data only
- Produce reliability diagrams
- Calculate Brier score and ECE
- Recalculate ranking metrics after the selected calibration policy
- Produce global SHAP explanations for the selected tree-based model
- Produce local SHAP examples for representative TP, FP, FN and low-risk cases
- Discuss correlated-feature and missingness-indicator limitations

### Output

- Calibration curves
- Calibration metrics table
- Risk-ranking table
- SHAP summary
- Local explanation examples

## Notebook 08 — Temporal, Institutional, Subgroup and Error Analysis

**File:** `notebooks/08_robustness_subgroups_and_errors.ipynb`

### Goal
Understand where the selected model performs differently or fails.

### Work

- Compare chronological performance windows
- Compare 2019/pre-COVID, 2020–2021/COVID and 2022–2024/later periods where meaningful
- Evaluate sufficiently sized subgroups:
  - age group
  - sex
  - hospital type
  - region
  - urban/rural status
  - season
  - day/night
- Compare false negatives and false positives
- Analyse whether missingness patterns are linked to changing model performance
- Perform feasible cross-institution or cross-region holdout experiments
- Add uncertainty/context for small groups

### Output

- Subgroup-performance table
- Temporal-performance table
- Error-profile plots
- Robustness conclusions

## Notebook 09 — Operational Pressure, Final Holdout and Portfolio Results

**File:** `notebooks/09_operational_pressure_and_final_results.ipynb`

### Goal
Finish the project without introducing a new software layer.

### Part A: operational pressure

Only perform this section if the data contain exact timestamps or another valid aggregation period.

- Aggregate encounter demand by valid time period
- Calculate a derived pressure proxy from encounter volume, high-acuity share and adverse-outcome burden if justified
- Fit standardisation and high-pressure thresholds on training data only
- Use a training-only percentile such as the 95th percentile, with sensitivity checks only when useful
- Never call the measure bed occupancy, staffing demand or actual hospital capacity utilisation

If valid time aggregation is not available, explicitly mark this section as **omitted due to data limitations**.

### Part B: final holdout

- Lock the selected model, feature set, missingness strategy and calibration method
- Evaluate once on the 2023–2024 final holdout
- Report all headline metrics
- Do not change the model after seeing the final holdout

### Part C: portfolio summary

Create the final tables and figures required by the README:

- Dataset summary
- Target prevalence
- Model comparison
- Calibration comparison
- Recall@K/Precision@K
- Subgroup robustness
- Error analysis
- SHAP summary
- Optional operational-pressure result

# Model Ladder

The modelling sequence remains:

`Constant baseline → Logistic Regression → XGBoost → LightGBM → TensorFlow MLP → TensorFlow TabTransformer`

Optional experiments:

- Multi-task TensorFlow model with separate heads for `adverse_outcome` and `readmission_30d`
- Additional uncertainty/conformal analysis only if time remains after the core project is complete

The optional components must never delay the complete core project.

# Final Evaluation Protocol

## Primary split

Use the actual dataset years to confirm the intended chronological design:

- **Development/training:** 2019–2021
- **Validation:** 2022
- **Final holdout:** 2023–2024

Do not replace this with a random final train/test split.

## Model-selection rule

All feature choices, preprocessing decisions, model hyperparameters, calibration decisions and ranking policies are selected using development/validation data only.

The final holdout is used once for the locked final evaluation.

## Primary metrics

### Discrimination

- PR-AUC
- ROC-AUC

### Classification diagnostics

- Precision
- Recall
- F1
- Confusion matrix at documented operating points

### Probability quality

- Brier score
- Expected Calibration Error
- Reliability diagram

### Ranking/capacity

- Recall@K
- Precision@K
- Number of events captured
- False-positive burden

Accuracy is supporting information, not the headline metric for imbalanced outcomes.

# Final Stop Point

The project is complete when:

- The data audit is documented
- Targets are defined
- Leakage exclusions are clear
- The chronological split is implemented
- Missingness analysis is complete
- Logistic Regression, XGBoost and LightGBM are compared
- Imbalance and ranking analysis is complete
- TensorFlow MLP is tested
- TensorFlow TabTransformer is tested
- Calibration is evaluated
- SHAP is completed for the final tree model where appropriate
- Subgroup, temporal and error analysis is completed
- Operational-pressure analysis is either validly completed or explicitly omitted
- The locked final model is evaluated once on the final holdout
- The README contains the final results and limitations

**Stop there.** The next hour spent building an API, Docker image, MLflow system or cloud architecture would not improve the core evidence of this portfolio project nearly as much as improving the notebooks, results and explanation.
