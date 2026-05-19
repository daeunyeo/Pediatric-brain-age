# pediatric-brain-age

> Predicting brain developmental age from fMRI functional connectivity
> and generating individualized developmental reports via LLM interpretation.

<img width="1768" height="1452" alt="20번 아동!" src="https://github.com/user-attachments/assets/33ce7f3c-3f58-4fec-a66c-8dbd9c101e5f" />

---

## What This Project Does

- Predicts Brain Age Gap (BAG) from pediatric fMRI functional connectivity
  using a classical ML pipeline (FC matrix, PCA, Ridge regression)
- Translates numeric predictions into structured developmental reports
  readable by caregivers, using an LLM interpretation layer
- Validated on ds000228 (Richardson et al., 2018),
  children aged 3.5 to 12.3 years

### Why Pediatric

Brain development is most rapid and variable during childhood,
making this age range well suited for BAG analysis.
Parents of young children have high demand for accessible,
evidence based developmental information.
This motivated the design of a caregiver readable report
grounded in fMRI connectivity data.

Adults in the dataset were used solely for classification
validation and excluded from the regression and report pipeline.

### Why Brain Age Gap (BAG)

Children of the same chronological age can differ substantially in neural
maturation. BAG is defined as predicted brain age minus chronological age.
It quantifies this individual variation and serves as a potential index
for developmental screening in educational and screening contexts.

### Why LLM

A numeric BAG value alone is not interpretable by non-specialists.
The LLM layer converts model outputs into Korean language developmental
reports for caregivers, incorporating network level connectivity profiles
alongside the BAG estimate.


### Key Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Child/Adult Classification | SVM (linear) | Accuracy | 96.7% |
| Child/Adult Classification | SVM (linear) | Adult F1 | 0.915 |
| Pediatric Brain Age Regression | PCA + Ridge | MAE | 1.58 yrs |
| Pediatric Brain Age Regression | PCA + Ridge | R² | 0.292 |
| Statistical Validation | Permutation Test | p-value | 0.010 |

---

## 1. Dataset & Rationale

### Dataset Overview

| Property | Value |
|---|---|
| Dataset | OpenNeuro ds000228 (Richardson et al., 2018) |
| Paradigm | Naturalistic movie watching (Pixar short film) |
| Total subjects | 155 (122 children, 33 adults) |
| Used in this project | 150 (118 children aged 3.5 - 12.3 yrs, 32 adults) |
| Age range (children) | 3.5 - 12.3 years |
| Age range (adults) | 18.0 - 39.0 years |
| Atlas | Harvard-Oxford cortical atlas (48 ROIs) |

### Why This Dataset?

Three factors informed the selection of ds000228 over other publicly available
fMRI datasets.

**1. Head motion and data quality in children**

Resting state fMRI requires subjects to remain still without any external
stimulus. This is a known practical problem with young children, who show
elevated head motion artifacts under resting conditions. The movie watching
paradigm in ds000228 naturally sustains visual attention, reducing motion
artifacts and improving signal quality in pediatric subjects. Using naturalistic stimuli to sustain attention in young participants is standard practice in developmental neuroimaging.

**2. Alignment with the dataset's original research purpose**

ds000228 was originally designed to study the development of the social brain
(Richardson et al., 2018). The Pixar stimulus was selected by the original
authors to engage social cognition networks (including regions associated with theory of mind and narrative comprehension) known to undergo substantial functional reorganization during childhood. This means the FC
patterns in this dataset are particularly likely to carry developmental signal
relevant to the prediction task pursued here.

**3. Ecological validity and extensibility**

A brain age assessment tool intended for real-world use must be compatible with
protocols that are feasible in clinical and educational settings. A paradigm
built around passive movie watching is substantially more deployable with
children than a resting state scan, which demands extended still compliance.
This makes the pipeline developed here a more realistic candidate for
future extension toward applied developmental screening.
This project takes a step in that direction by generating caregiver
readable reports from the same paradigm.

<img width="683" height="578" alt="fc_matrix_example" src="https://github.com/user-attachments/assets/0bdb777a-63da-4931-854b-3e9e9d089105" />

### Class Imbalance Handling

The original dataset contains roughly 4:1 children-to-adult ratio (122:33).
Rather than reducing the majority class to force equal counts (which would have decreased the training set by approximately 55%), the original distribution was preserved and `class_weight='balanced'` was applied in all
classifiers. This approach retains all available data while correcting for
optimization bias during training.

---

## 2. Preprocessing Pipeline

### Pipeline Overview

```
Raw 4D fMRI (.nii.gz)
      
ROI Time Series Extraction
  · Harvard-Oxford cortical atlas (48 ROIs)
  · NiftiLabelsMasker
  · Per ROI signal z-score normalized (zero mean, unit variance)
  · Output shape: (n_timepoints, 48)
       
Functional Connectivity Matrix
  · Pearson correlation across all ROI pairs
  · np.corrcoef(time_series.T)
  · Output shape: (48, 48)
       
Upper-Triangle Flatten
  · Symmetric matrix → retain upper triangle only (k=1)
  · Removes diagonal (autocorrelation = 1.0) and redundant lower triangle
  · 48×48 = 2304 → 1128 unique features
      
NaN Replacement
  · Source: ROIs with zero variance time series
    (no BOLD signal change → correlation undefined → NaN)
  · Replaced with 0.0 (treated as no connectivity)
       
Feature Matrix X: shape (n_subjects, 1128)
```

### Design Decisions

| Step | Decision | Reason |
|---|---|---|
| Z-score normalization | Per ROI, per subject | Removes scanner-level amplitude differences, preserves relative connectivity pattern |
| Pearson correlation | Over full scan length | Standard FC estimator, computationally stable for this scan duration |
| Upper triangle only | `np.triu_indices(48, k=1)` | FC matrix is symmetric, lower triangle is redundant |
| NaN → 0.0 | `np.nan_to_num` | Undefined correlation treated as absence of connectivity |
| StratifiedKFold | Over KFold | KFold produced NaN accuracy folds due to child/adult imbalance in split, StratifiedKFold enforces class ratio per fold |

### Network Level Analysis

Three functional networks were defined using Harvard-Oxford atlas ROI indices:

| Network | Key ROIs | Function |
|---|---|---|
| Language | IFG, STG, Planum Temporale | Language comprehension and production |
| Social Cognition | mPFC, PCC, Angular Gyrus | Self-referential processing, social cognition |
| Visual Processing | Lateral Occipital Cortex, Cuneal Cortex | Visual processing |

For each subject, within network mean connectivity strength was computed
and converted to a z-score relative to age matched peers within 1.5 years.

Note: BAG and network z-scores are independent measures. BAG reflects
overall FC pattern maturity. Network z-scores reflect relative connectivity
strength within specific functional systems.

---

## 3. Model Comparison & Selection

### Task 1: Child / Adult Classification (n = 150)

All models follow a `StandardScaler → Classifier` pipeline evaluated with
5-fold StratifiedKFold cross-validation.

| Model | Acc (mean ± std) | Child F1 | Adult F1 |
|---|---|---|---|
| SVM (linear, balanced) | 0.967 ± 0.030 | 0.98 | 0.915 |
| Logistic Regression (balanced) | 0.967 ± 0.030 | 0.98 | 0.915 |

**Why SVM?**
In this setting, feature dimensionality (1128) substantially exceeds sample
count (150). Linear SVM is theoretically well-suited to high dimensional,
small sample count problems because it finds a maximum margin hyperplane using
only the support vectors, which limits sensitivity to noise from irrelevant
features.

**Interpreting identical performance**
Both models produced numerically identical results across all folds. This reflects the structure of the data rather than coincidence. When two classes are linearly separable, linear SVM and logistic regression converge to similar decision boundaries. The FC-based representation is sufficient for adult/child discrimination without nonlinear or more complex classifiers.

---

### Task 2a: Full Cohort Age Regression (n = 150)

Pipeline: `StandardScaler → Regressor`, 5-fold KFold.
Upper-triangle features (1128) used throughout.

| Model | MAE (mean ± std) | R² |
|---|---|---|
| SVR (linear) | 4.13 ± 0.49 yrs | 0.479 |
| Ridge | 4.14 ± 0.49 yrs | 0.476 |
| ElasticNet | 4.02 ± 0.47 yrs | 0.523 |

The full cohort spans 3.5–39 years, mixing children and adults. MAE here
reflects cross-group error and should not be directly compared to pediatric
results below, which are single-group.

**Effect of increasing sample size**

| n | Task | MAE |
|---|---|---|
| 40 | Age regression (all ages) | 9.99 yrs |
| 80 | Age regression (all ages) | 4.72 yrs |
| 150 | Age regression (all ages) | ~4.03 yrs |

Doubling the sample from 40 to 80 cut MAE by more than half, without any
change to the model architecture. This underscores that data volume was the
primary bottleneck at small sample sizes.

---

### Task 2b: Pediatric Brain Age Regression (n = 118 children)

Pipeline: `StandardScaler → [PCA] → Regressor`, 5-fold KFold.

| Model | MAE (mean ± std) | R² |
|---|---|---|
| SVR (linear) | 1.73 ± 0.13 yrs | 0.139 |
| ElasticNet | 1.65 ± 0.12 yrs | 0.207 |
| KRR (RBF) | 1.93 ± 0.20 yrs | −0.123 |
| **PCA + Ridge** | **1.58 ± 0.16 yrs** | **0.292** |

<img width="690" height="390" alt="model_comparison" src="https://github.com/user-attachments/assets/24d3d872-cdeb-423c-95ed-78a1fee86678" />

**Why PCA + Ridge?**

PCA + Ridge achieved the lowest MAE and highest R² across all models. The
rationale for this combination is as follows:

- With 118 samples and 1128 features, direct regression risks overfitting.
  PCA (n_components=50) first compresses the feature space into the 50
  directions of maximum variance, discarding noise dimensions.
- Ridge then fits a regularized regression on these 50 components, further
  penalizing large coefficients.
- KRR (RBF) underperformed (R² = −0.123), suggesting that nonlinear kernel
  mapping does not help and likely overfits at this sample size.

---

## 4. Evaluation & Statistical Validation

### Metrics

| Metric | Applied to | Interpretation |
|---|---|---|
| Accuracy | Classification | Overall fraction correct |
| F1 (per class) | Classification | Harmonic mean of precision and recall (critical for imbalanced classes) |
| MAE | Regression | Mean absolute prediction error in years; directly interpretable |
| R² | Regression | Proportion of variance explained (0 = no better than mean prediction) |
| Brain Age Gap | Individual report | Predicted age − actual age (positive = developmentally advanced) |
| Peer Ranking | Individual report | Percentile rank among peers (within 1.5 years), reported as top X% |

### Why Not Accuracy Alone?

At n=40, overall classification accuracy was 85% but adult F1 was only 0.67.
This discrepancy arises because the majority class (child) dominates the
accuracy numerator. Reporting only overall accuracy would have overstated
model performance on the clinically relevant minority class (adults). This is why per-class F1 was used as the primary classification metric.

### Permutation Test

To verify that the PCA + Ridge model learned genuine structure rather than
exploiting random correlations, a permutation test was conducted.

| | MAE |
|---|---|
| Observed model | 1.59 yrs |
| Mean of 100 permutations (shuffled labels) | 2.20 yrs |
| p-value | **0.010** |

The observed MAE falls below the 1st percentile of the null distribution
(p = 0.010), confirming that the model captures real developmental signal
in the FC patterns beyond chance level.

<img width="690" height="390" alt="permutation_test" src="https://github.com/user-attachments/assets/b9cc1973-ebc5-4755-a78f-e982320e2efd" />

Note: R² = 0.292 indicates that approximately 29% of age variance is explained
by the model. While statistically significant, this leaves substantial
unexplained variance (a limitation acknowledged in Section 7).

### Classification Report (n = 150)

```
              precision    recall  f1-score   support

       child       0.97      0.99      0.98       118
       adult       0.97      0.88      0.92        32

    accuracy                           0.97       150
   macro avg       0.97      0.93      0.95       150
weighted avg       0.97      0.97      0.97       150
```

---

## 5. Brain Age Report

The final model generates an individualized Brain Age Report
for any subject in the dataset.

### Report Structure

The report consists of four panels and a text interpretation section.

| Component | Description |
|---|---|
| BAG Distribution | Histogram of BAG among peers (within 1.5 years), with the subject marked |
| Age Prediction | Scatter plot of predicted vs. actual age across all 118 children, with the subject highlighted and a MAE error bar |
| Network Connectivity | Radar chart showing network level z-scores for Language, Social Cognition, and Visual Processing networks relative to peers |
| LLM Report | Structured Korean language developmental report generated by an LLM, incorporating BAG estimate and network connectivity profiles |

### LLM Interpretation Layer

The LLM layer (Mistral API) receives the following inputs:

- BAG and predicted brain age
- Peer ranking
- Network level z-scores (Language, Social Cognition, Visual Processing)
- ROI level connectivity profiles

The output is a structured Korean language report covering:

- Developmental level and peer comparison
- Network connectivity strengths and areas for improvement
- Activity recommendations based on network profiles

Reports are generated in Korean as the target audience is Korean speaking caregivers.

### Example Output

<img width="1768" height="1452" alt="70번 아동!" src="https://github.com/user-attachments/assets/44d12e3c-8419-42f8-9133-8e1032b68889" />

<img width="1768" height="1452" alt="90번 아동!" src="https://github.com/user-attachments/assets/bc8da5bb-d60b-498f-b168-17a13858baa8" />

---

## 6. Reflection

### What Worked
- FC matrix representation was sufficient for both
  classification and age regression on the same pipeline
- LLM interpretation layer successfully translated
  numeric outputs into structured caregiver readable reports
- Network z-score profiles provided interpretable
  individual level connectivity patterns

### Limitations
- Validated on a single dataset (n=118).
  Generalizability to other pediatric fMRI datasets
  remains to be tested.
- R² of 0.292 indicates the model explains approximately
  29% of age variance. Predictions should be interpreted
  alongside the reported uncertainty bounds.
- Network analysis covers three networks only,
  defined based on ds000228 research context.

### Future Directions
- Apply to larger pediatric datasets 
- Expand network definitions beyond three networks
- Validate LLM report accuracy with domain experts
- Support multilingual report generation

---

## 7. Repository Structure

```
pediatric-brain-age/
├── brain_age_pipeline.ipynb
├── brain_age_llm_report.ipynb
├── README.md
└── requirements.txt
```

The ML pipeline dataset fetches automatically via nilearn.
The LLM report pipeline requires a Mistral API key.

---

## References

Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018).
Development of the social brain from age three to twelve years.
*Nature Communications*, 9, 1027.
https://doi.org/10.1038/s41467-018-03399-2

Desikan, R. S., et al. (2006).
An automated labeling system for subdividing the human cerebral cortex
on MRI scans into gyral based regions of interest.
*NeuroImage*, 31(3), 968-980.
https://doi.org/10.1016/j.neuroimage.2006.01.021

Mistral AI. (2024). Mistral API documentation.
https://docs.mistral.ai
