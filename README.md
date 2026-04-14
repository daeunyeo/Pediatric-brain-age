# pediatric-brain-age

> Predicting brain developmental age from fMRI functional connectivity using classical machine learning

---

## Key Results

| Task | Model | Metric | Score |
|---|---|---|---|
| Child/Adult Classification | SVM (linear) | Accuracy | 96.7% |
| Child/Adult Classification | SVM (linear) | Adult F1 | 0.915 |
| Pediatric Brain Age Regression | PCA + Ridge | MAE | 1.58 yrs |
| Pediatric Brain Age Regression | PCA + Ridge | R² | 0.292 |
| Statistical Validation | Permutation Test | p-value | 0.010 |

---

## 1. Motivation & Background

### Problem Statement

The brain does not develop at a uniform pace across individuals. Two children
of the same chronological age may differ substantially in their neural
maturation. Neuroimaging offers a way to quantify this gap, but translating raw fMRI signals into interpretable developmental indices requires a complete ML pipeline from signal extraction to individual-level reporting.

This project addresses the following question:

> **Can functional connectivity (FC) patterns alone predict where an individual
> child stands in their brain development relative to peers?**

### From Classification to Regression

The project was structured as a deliberate two-step progression:

1. **Step 1. Classification**: Can FC patterns distinguish children from
   adults? This verifies that the connectivity matrix carries age-relevant
   information at all.
2. **Step 2. Regression**: Can FC patterns predict continuous age? This
   reframes the same data as a finer-grained estimation problem.

Moving from classification to regression on the same pipeline was intentional: the same FC representation handles both tasks, and the comparison shows what each formulation reveals about the data.

### Why Brain Age Gap?

Predicting chronological age is not the end goal. The **Brain Age Gap**, defined as the residual between predicted brain age and actual age, is the informative quantity. A positive gap indicates that the brain appears more mature than
expected for the child's age; a negative gap indicates the reverse. This
residual has potential value as an individual-level indicator for
developmental evaluation in educational or clinical contexts.

---

## 2. Dataset & Rationale

### Dataset Overview

| Property | Value |
|---|---|
| Dataset | OpenNeuro ds000228 (Richardson et al., 2018) |
| Paradigm | Naturalistic movie-watching (Pixar short film) |
| Total subjects | 155 (122 children, 33 adults) |
| Used in this project | 150 (118 children aged 3.5–12.3 yrs, 32 adults) |
| Age range (children) | 3.5 – 12.3 years |
| Age range (adults) | 18.0 – 39.0 years |
| Atlas | Harvard-Oxford cortical atlas (48 ROIs) |

### Why This Dataset?

Three factors informed the selection of ds000228 over other publicly available
fMRI datasets.

**① Head motion and data quality in children**

Resting-state fMRI requires subjects to remain still without any external
stimulus. This is a known practical problem with young children, who show
elevated head motion artifacts under resting conditions. The movie-watching
paradigm in ds000228 naturally sustains visual attention, reducing motion
artifacts and improving signal quality in pediatric subjects. Using naturalistic stimuli to sustain attention in young participants is standard practice in developmental neuroimaging.

**② Alignment with the dataset's original research purpose**

ds000228 was originally designed to study the development of the social brain
(Richardson et al., 2018). The Pixar stimulus was selected by the original
authors to engage social cognition networks (including regions associated with theory of mind and narrative comprehension) known to undergo substantial functional reorganization during childhood. This means the FC
patterns in this dataset are particularly likely to carry developmental signal
relevant to the prediction task pursued here.

**③ Ecological validity and extensibility**

A brain-age assessment tool intended for real-world use must be compatible with
protocols that are feasible in clinical and educational settings. A paradigm
built around passive movie-watching is substantially more deployable with
children than a resting-state scan, which demands extended still compliance.
This makes the pipeline developed here a more realistic candidate for
future extension toward applied developmental screening.

<img width="683" height="578" alt="fc_matrix_example" src="https://github.com/user-attachments/assets/0bdb777a-63da-4931-854b-3e9e9d089105" />

### Class Imbalance Handling

The original dataset contains roughly 4:1 children-to-adult ratio (122:33).
Rather than reducing the majority class to force equal counts (which would have decreased the training set by approximately 55%), the original distribution was preserved and `class_weight='balanced'` was applied in all
classifiers. This approach retains all available data while correcting for
optimization bias during training.

---

## 3. Preprocessing Pipeline

### Pipeline Overview

```
Raw 4D fMRI (.nii.gz)
      
ROI Time Series Extraction
  · Harvard-Oxford cortical atlas (48 ROIs)
  · NiftiLabelsMasker
  · Per-ROI signal z-score normalized (zero mean, unit variance)
  · Output shape: (n_timepoints, 48)
       
Functional Connectivity Matrix
  · Pearson correlation across all ROI pairs
  · np.corrcoef(time_series.T)
  · Output shape: (48, 48)
       
Upper-Triangle Flatten
  · Symmetric matrix → retain upper triangle only (k=1)
  · Removes diagonal (self-correlation = 1.0) and redundant lower triangle
  · 48×48 = 2304 → 1128 unique features
      
NaN Replacement
  · Source: ROIs with zero-variance time series
    (no BOLD signal change → correlation undefined → NaN)
  · Replaced with 0.0 (treated as no connectivity)
       
Feature Matrix X: shape (n_subjects, 1128)
```

### Design Decisions

| Step | Decision | Reason |
|---|---|---|
| Z-score normalization | Per-ROI, per-subject | Removes scanner-level amplitude differences, preserves relative connectivity pattern |
| Pearson correlation | Over full scan length | Standard FC estimator, computationally stable for this scan duration |
| Upper triangle only | `np.triu_indices(48, k=1)` | FC matrix is symmetric, lower triangle is redundant |
| NaN → 0.0 | `np.nan_to_num` | Undefined correlation treated as absence of connectivity |
| StratifiedKFold | Over KFold | KFold produced NaN accuracy folds due to child/adult imbalance in split, StratifiedKFold enforces class ratio per fold |

---

## 4. Model Comparison & Selection

### Task 1: Child / Adult Classification (n = 150)

All models follow a `StandardScaler → Classifier` pipeline evaluated with
5-fold StratifiedKFold cross-validation.

| Model | Acc (mean ± std) | Child F1 | Adult F1 |
|---|---|---|---|
| SVM (linear, balanced) | 0.967 ± 0.030 | 0.98 | 0.915 |
| Logistic Regression (balanced) | 0.967 ± 0.030 | 0.98 | 0.915 |

**Why SVM?**
In this setting, feature dimensionality (1128) substantially exceeds sample
count (150). Linear SVM is theoretically well-suited to high-dimensional,
low-sample-count problems because it finds a maximum-margin hyperplane using
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

## 5. Evaluation & Statistical Validation

### Metrics

| Metric | Applied to | Interpretation |
|---|---|---|
| Accuracy | Classification | Overall fraction correct |
| F1 (per class) | Classification | Harmonic mean of precision and recall (critical for imbalanced classes) |
| MAE | Regression | Mean absolute prediction error in years; directly interpretable |
| R² | Regression | Proportion of variance explained (0 = no better than mean prediction) |
| Brain Age Gap | Individual report | Predicted age − actual age (positive = developmentally advanced) |

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

## 6. Brain Age Report Demo

The final model (PCA + Ridge, trained on all 118 children) generates an
individual Brain Age Report for any subject in the dataset.

**Example output: subject index 69 (actual age: 5.4 yrs)**

```
Actual age     : 5.4 yrs
Predicted age  : 6.1 yrs  (± 2.10 yrs)
Brain Age Gap  : +0.63 yrs
Peer ranking   : Top 46%  (24th out of 50 age-matched peers)
Interpretation : Brain development approximately 0.6 yrs ahead of peers
```

<img width="1289" height="515" alt="69" src="https://github.com/user-attachments/assets/cb1b8b18-468b-481d-9b82-1e8e70e20c5e" />

The report consists of two panels:

- **Left panel**: Distribution of Brain Age Gap among age-matched peers
  (±1.5 yrs), with the subject's gap marked.
- **Right panel**: Scatter plot of predicted vs. actual age across all 118
  children, with the subject highlighted and a ±MAE error bar shown.

**What makes this output distinct from raw age prediction**

The report does not evaluate whether the model predicted the correct age in
absolute terms. It situates the individual within their peer group. A gap of
+0.63 years is meaningful not because the model was accurate to 0.63 years,
but because it places this child in the upper half of their age-matched cohort.
This peer-relative framing makes the output relevant to developmental evaluation.

---

## 7. Limitations & Future Work

### Limitations

**1. Low R² (0.292)**
The model explains approximately 29% of age variance in the pediatric sample.
The permutation test confirms this is above chance (p = 0.010), but the
majority of developmental variance in FC patterns remains unaccounted for by
this pipeline. Predictions should be interpreted with caution and alongside
the reported uncertainty bounds.

**2. Small single-dataset sample (n = 118 children)**
All pediatric models were trained and evaluated on a single dataset. Results
may not generalize to children scanned under different acquisition parameters,
atlas parcellations, or demographic compositions. Cross-dataset validation was
not performed.

**3. Age-band MAE values are manually set**
The uncertainty bounds displayed in the Brain Age Report (e.g., ±2.10 yrs)
were manually assigned based on observed fold-level error patterns, not derived
from a formal statistical model. These values approximate but do not rigorously
represent prediction intervals.

**4. Temporal information is discarded**
Converting the BOLD time series to a static FC matrix loses all temporal
dynamics. The correlation structure summarizes co-activation but cannot capture
sequential or state-dependent patterns in brain activity.

### Future Work

| Direction | Motivation |
|---|---|
| Apply to larger pediatric datasets (e.g., HCP-D, ABCD) | Improve generalizability (reduce reliance on single-dataset results) |
| Cross-dataset validation | Test whether FC-based brain age generalizes across acquisition sites |
| Alternative atlas parcellations (Schaefer, AAL) | Assess sensitivity of results to ROI definition |
| Use time series directly (e.g., RNN, Transformer) | Recover temporal information lost in FC summarization |
| Formal prediction intervals | Replace manually set MAE bands with statistically derived confidence bounds |
| Partial correlation via `nilearn.ConnectivityMeasure` | More precise FC estimation by controlling for indirect connectivity |

---

## 8. Repository Structure

```
pediatric-brain-age/
├── figures/
│   ├── fc_matrix_example.png
│   ├── model_comparison.png
│   ├── permutation_test.png
│   └── brain_age_report.png
├── brain_age_pipeline.ipynb
├── README.md
└── requirements.txt
```

Clone the repo and run the notebooks in `brain_age_pipeline.ipynb` sequentially (01 → 05).
The dataset fetches automatically on first run via `nilearn` and caches under `~/nilearn_data/`.

---

## References

Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018).
Development of the social brain from age three to twelve years.
*Nature Communications*, 9, 1027.
https://doi.org/10.1038/s41467-018-03399-2
