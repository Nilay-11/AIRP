# Study summary for the team

**Title (working):** Does Deep Learning Outperform Classical Machine Learning for Cardiovascular Disease Classification? An Explainable AI Comparison Using NHANES 2011–2018

Status: all results for the 7 models are final.

---

## 1. Dataset

| Item | Detail |
|---|---|
| Name | National Health and Nutrition Examination Survey (NHANES) |
| Source | U.S. Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS) |
| Link | https://wwwn.cdc.gov/nchs/nhanes/ (public, free) |
| Cycles used | 2011–2012 (G), 2013–2014 (H), 2015–2016 (I), 2017–2018 (J) |
| Files merged | DEMO (demographics), BPX (blood pressure), BMX (body measures), MCQ (medical conditions), TCHOL (total cholesterol), HDL, GHB (HbA1c), BIOPRO (biochemistry), SMQ (smoking), DIQ (diabetes) |
| Population | Adults aged ≥ 20 who completed the physical examination |
| Final sample | **21,552 adults; 2,369 with CVD (11.0% unweighted, 8.9% survey-weighted)** |
| Split | Stratified 80/20: training 17,241 / test 4,311 (474 CVD cases) |
| Excluded | The 2019–2020 cycle (interrupted by COVID-19); 94 people with unknown outcome |

**Outcome (target variable).** CVD = 1 if the person reported ever being told by a doctor they had any of:
congestive heart failure (MCQ160B), coronary heart disease (MCQ160C), angina (MCQ160D), heart attack (MCQ160E) or stroke (MCQ160F).
CVD = 0 if "no" to all five.

---

## 2. Attributes (20 input features)

| Group | Feature | NHANES source | Unit / coding |
|---|---|---|---|
| Demographic | Age | RIDAGEYR | years (top-coded at 80) |
| | Sex | RIAGENDR | 1 = male, 0 = female |
| | Race/ethnicity (4 indicators) | RIDRETH1 | Mexican American, Other Hispanic, Non-Hispanic Black, Other/multiracial (reference: Non-Hispanic White) |
| Socioeconomic | Income-to-poverty ratio | INDFMPIR | ratio, 0–5 |
| | Education | DMDEDUC2 | 1–5 ordinal |
| Blood pressure | Systolic BP | BPXSY1–3 (mean) | mmHg |
| | Diastolic BP | BPXDI1–3 (mean) | mmHg |
| | Pulse rate | BPXPLS | beats/min |
| Anthropometric | BMI | BMXBMI | kg/m² |
| | Waist circumference | BMXWAIST | cm |
| Lipids | Total cholesterol | LBXTC | mg/dL |
| | HDL cholesterol | LBDHDD | mg/dL |
| Glycaemic | HbA1c | LBXGH | % |
| | Diagnosed diabetes | DIQ010 | 1 = yes, 0 = no/borderline |
| Renal/metabolic | Serum creatinine | LBXSCR | mg/dL |
| | Uric acid | LBXSUA | mg/dL |
| Behavioural | Smoking status | SMQ020, SMQ040 | 0 = never, 1 = former, 2 = current |

Medication variables were deliberately **not** used as inputs, because treatment usually follows a CVD diagnosis and would leak the answer.
Preprocessing: median imputation and standardisation, fitted on the training data only.

---

## 3. Models (7)

| # | Model | Family | Library | Selected settings |
|---|---|---|---|---|
| 1 | Logistic Regression (L2) | Classical ML | scikit-learn | C = 0.01 |
| 2 | Random Forest | Classical ML | scikit-learn | 500 trees, depth 12, min leaf 20 |
| 3 | Support Vector Machine (RBF kernel) | Classical ML | scikit-learn | C = 10, γ = 0.001, class-balanced, Platt probabilities |
| 4 | XGBoost | Classical ML | xgboost | 502 trees, depth 5, learning rate 0.011 |
| 5 | Multilayer Perceptron (MLP) | Deep learning | PyTorch | 64-32 hidden units, BatchNorm, ReLU, dropout 0.5, AdamW |
| 6 | Tabular ResNet | Deep learning | PyTorch | 4 residual blocks, width 64, dropout 0.5, AdamW |
| 7 | FT-Transformer | Deep learning | PyTorch | feature tokenizer + 2 Transformer blocks, width 64, 8 attention heads, AdamW lr 3e-4 |

All models were tuned with 5-fold stratified cross-validation on the training set (objective: ROC-AUC).
The three deep models share the same training loop: AdamW optimiser, binary cross-entropy loss and early stopping.
ResNet and FT-Transformer follow Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).

---

## 4. Mathematical and statistical tools for comparison

**Discrimination**
- **ROC-AUC:** the probability that a random CVD case scores higher than a random non-case. Reported with a 95% bootstrap confidence interval (1,000 resamples).
- **PR-AUC (average precision):** precision–recall area, suitable for imbalanced data (11% prevalence).
- **Precision = TP/(TP+FP), Recall = TP/(TP+FN), Specificity = TN/(TN+FP), F1 = 2·P·R/(P+R).** These use a decision threshold chosen to maximise F1 on training data only.

**Calibration (are the predicted probabilities correct?)**
- **Brier score** = mean of (predicted probability − actual outcome)².
- **Expected Calibration Error (ECE)** = Σ (bin size / n) · |observed rate − mean predicted probability|, over 10 risk bins.
- **Calibration slope and intercept:** logistic regression of the outcome on logit(predicted probability). Ideal values are slope 1, intercept 0.

**Statistical significance**
- **DeLong test:** a nonparametric test for the difference between two correlated ROC-AUCs on the same test set. It gives a p-value for each pair of models.

**Explainability**
- **SHAP (SHapley Additive exPlanations):** each feature's contribution to a prediction, based on Shapley values from cooperative game theory. The same permutation-based estimator was used for all 7 models so the results are comparable. It was validated against exact TreeSHAP (Spearman ρ = 0.989).
- **Global importance** = mean |SHAP value| per feature.

**Agreement and stability of explanations**
- **Spearman's ρ** and **Kendall's τ:** rank correlation of feature-importance rankings between two models.
- **Kendall's W:** agreement of rankings across all models at once (0 = none, 1 = perfect).
- **Jaccard index of top-5 features** = (number of features in both top-5 lists) / (number of distinct features in either list).
- **Bootstrap stability:** refit each model on 5 bootstrap samples and compare its SHAP rankings.

**Robustness**
- **Feature-group ablation:** remove one group of features, retrain, and measure ΔAUC (averaged over 5 random splits).
- **Repeated splits:** AUC mean ± SD over 5 different train/test splits.
- **Temporal validation:** train on 2011–2016 and test on 2017–2018.
- **Survey-weighted AUC:** uses the NHANES examination weights (divided by 4 for four cycles).

**Cost:** tuning time, training time, inference time per 1,000 people, and SHAP computation time.

---

## 5. Key results (test set, n = 4,311)

| Model | Family | Test ROC-AUC [95% CI] | PR-AUC | F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic Regression | Classical | 0.839 [0.821–0.855] | 0.392 | 0.455 | 0.080 | 0.010 |
| Random Forest | Classical | 0.839 [0.821–0.856] | 0.407 | 0.447 | 0.080 | 0.010 |
| SVM (RBF) | Classical | 0.839 [0.821–0.854] | 0.396 | 0.447 | 0.080 | 0.007 |
| **XGBoost** | Classical | **0.848 [0.829–0.863]** | 0.418 | 0.451 | 0.078 | 0.009 |
| MLP | Deep | 0.842 [0.825–0.858] | 0.398 | 0.447 | 0.080 | 0.018 |
| ResNet | Deep | 0.837 [0.819–0.855] | 0.402 | 0.439 | 0.080 | 0.021 |
| FT-Transformer | Deep | 0.843 [0.825–0.859] | 0.419 | 0.455 | 0.079 | 0.012 |

- No deep model was significantly better than any classical model (DeLong p ≥ 0.161 wherever a deep model had the higher AUC).
- XGBoost was significantly better than LR, RF, SVM and ResNet. It was not significantly better than the MLP (p = 0.064) or FT-Transformer (p = 0.075).
- SHAP: age was ranked #1 by all 7 models. Kendall's W = 0.75 overall, 0.92 among the deep models and 0.74 among the classical models.
- Cost: FT-Transformer took 5,779 s to tune against 49 s for XGBoost.

Full results are in `RESULTS.md` and the results PDF.

## 6. Abstract (final, about 190 words)

Cardiovascular disease (CVD) is a leading cause of death, yet it remains unclear whether deep learning improves CVD classification over classical machine learning on structured clinical data. Using four cycles of the U.S. National Health and Nutrition Examination Survey (NHANES 2011–2018; n = 21,552 adults, 2,369 with CVD), we compared four classical models (Logistic Regression, Random Forest, Support Vector Machine, XGBoost) with three deep learning models (multilayer perceptron, tabular ResNet, FT-Transformer) using 20 demographic, clinical and laboratory features. Models were evaluated for discrimination (ROC-AUC, PR-AUC, DeLong test), calibration (Brier score, expected calibration error) and computational cost, and interpreted with SHAP. All models performed similarly (ROC-AUC 0.837–0.848). XGBoost achieved the highest AUC, and no deep learning model significantly outperformed any classical model, although the FT-Transformer required 119 times longer tuning than XGBoost. Age was the most influential feature for every model. Feature rankings were broadly concordant (Kendall's W = 0.75) but clustered by the shape of the decision function, with the deep models agreeing closely with each other (W = 0.92). Inverse attributions for cholesterol reflected treatment after diagnosis, indicating association rather than causation. For structured health survey data, deep learning offered no measurable advantage over classical machine learning.

**Keywords:** cardiovascular disease, machine learning, deep learning, NHANES, explainable AI, SHAP
