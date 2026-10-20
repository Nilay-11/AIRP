# IV. Results

*Classical Machine Learning vs. Deep Learning for Cardiovascular Disease Classification: An Explainable AI Study Using NHANES Data. Results section draft: every number comes from the experiment outputs and can be pasted into the paper.*

## A. Study Population

The analytic cohort comprised 21,552 adults from four NHANES cycles (2011–2018), of whom 2,369 reported a physician diagnosis of CVD (11.0% unweighted; 8.9% survey-weighted). Data were split into a training set of 17,241 participants (1,895 cases) and a held-out test set of 4,311 participants (474 cases). Table I compares survey-weighted characteristics by CVD status.

Participants with CVD were older (65.0 vs. 46.1 years), more often male (54.0% vs. 47.5%), had diagnosed diabetes more often (33.0% vs. 8.3%) and were more often former smokers (38.0% vs. 23.1%). They also had higher systolic blood pressure, waist circumference, HbA1c and creatinine. Notably, they had **lower** total cholesterol (177.5 vs. 193.0 mg/dL) and diastolic blood pressure (67.8 vs. 71.3 mmHg), a pattern examined in Section IV-H.

**Table I. Survey-weighted characteristics of the study population**

| Characteristic | Overall | No CVD | CVD |
|---|---|---|---|
| Unweighted n | 21,552 | 19,183 | 2,369 |
| Age, years, mean (SD) | 47.7 (17.1) | 46.1 (16.5) | 65.0 (13.1) |
| Male sex, % | 48.0 | 47.5 | 54.0 |
| Income-to-poverty ratio, mean (SD) | 3.0 (1.7) | 3.0 (1.7) | 2.5 (1.6) |
| Systolic BP, mmHg, mean (SD) | 122.8 (17.4) | 122.0 (16.8) | 130.4 (20.8) |
| Diastolic BP, mmHg, mean (SD) | 71.0 (12.0) | 71.3 (11.7) | 67.8 (14.5) |
| BMI, kg/m², mean (SD) | 29.3 (7.0) | 29.1 (7.0) | 30.8 (7.3) |
| Waist circumference, cm, mean (SD) | 99.9 (16.7) | 99.3 (16.6) | 106.6 (16.2) |
| Total cholesterol, mg/dL, mean (SD) | 191.6 (41.4) | 193.0 (41.0) | 177.5 (43.0) |
| HDL cholesterol, mg/dL, mean (SD) | 53.9 (16.5) | 54.2 (16.5) | 50.4 (16.4) |
| HbA1c, %, mean (SD) | 5.7 (1.0) | 5.6 (0.9) | 6.2 (1.3) |
| Serum creatinine, mg/dL, mean (SD) | 0.9 (0.4) | 0.9 (0.3) | 1.1 (0.6) |
| Diagnosed diabetes, % | 10.5 | 8.3 | 33.0 |
| Current smoker, % | 18.9 | 18.5 | 22.4 |
| Former smoker, % | 24.5 | 23.1 | 38.0 |

## B. Predictive Performance

Table II reports the test-set performance of all seven models. Discrimination was remarkably similar: ROC-AUC ranged only from 0.837 (ResNet) to 0.848 (XGBoost), a spread of 0.011, and every 95% confidence interval overlapped. XGBoost achieved the highest ROC-AUC (0.848, 95% CI 0.829–0.863) and the lowest Brier score (0.078). Among the deep models, the FT-Transformer performed best (ROC-AUC 0.843) and achieved the highest PR-AUC of all models (0.419), followed by the MLP (0.842) and ResNet (0.837). Cross-validation AUCs (0.846–0.850) closely matched the test AUCs, indicating little overfitting.

Threshold-based metrics showed the same parity. F1-scores ranged from 0.439 to 0.455, with LR and the FT-Transformer jointly highest (0.455). The MLP had the highest precision (0.390) and specificity (0.899), while RF and the FT-Transformer had the highest recall (0.586).

**Table II. Test-set performance of the seven models (n = 4,311; 474 CVD cases)**

| Model | Family | CV AUC (SD) | Test ROC-AUC [95% CI] | PR-AUC | Precision | Recall | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Classical | 0.846 (0.009) | 0.839 [0.821–0.855] | 0.392 | 0.383 | 0.561 | 0.888 | **0.455** |
| Random Forest | Classical | 0.847 (0.003) | 0.839 [0.821–0.856] | 0.407 | 0.361 | **0.586** | 0.872 | 0.447 |
| SVM (RBF) | Classical | 0.849 (0.007) | 0.839 [0.821–0.854] | 0.396 | 0.362 | 0.584 | 0.873 | 0.447 |
| **XGBoost** | Classical | 0.850 (0.005) | **0.848 [0.829–0.863]** | 0.418 | 0.373 | 0.572 | 0.881 | 0.451 |
| MLP | Deep | 0.849 (0.007) | 0.842 [0.825–0.858] | 0.398 | **0.390** | 0.523 | **0.899** | 0.447 |
| ResNet | Deep | 0.849 (0.009) | 0.837 [0.819–0.855] | 0.402 | 0.377 | 0.527 | 0.892 | 0.439 |
| FT-Transformer | Deep | 0.850 (0.007) | 0.843 [0.825–0.859] | **0.419** | 0.371 | **0.586** | 0.877 | **0.455** |

*Precision, recall, specificity and F1 use each model's threshold maximising F1 on out-of-fold training predictions. Bold marks the best value in each column.*

[[FIG:fig1_roc_calibration.png|0.98|Fig. 1. (a) ROC curves of the seven models on the test set, with AUC in the legend. (b) Calibration curves by decile of predicted risk. Solid lines are classical models; dashed lines are deep-learning models.]]

## C. Calibration

All models were well calibrated (Fig. 1b, Table III). Expected calibration error (ECE) was at most 0.021, and Brier scores fell between 0.078 and 0.080. The SVM had the lowest ECE (0.007), and the FT-Transformer had the calibration slope closest to the ideal value of 1 (0.99). ResNet (slope 0.87, ECE 0.021) and the MLP (0.91, 0.018) were slightly over-confident, producing predicted risks that were too extreme. Random Forest (1.16) was slightly under-confident.

**Table III. Calibration on the test set**

| Model | Brier score | ECE | Calibration slope | Calibration intercept |
|---|---|---|---|---|
| Logistic Regression | 0.080 | 0.010 | 1.04 | −0.001 |
| Random Forest | 0.080 | 0.010 | 1.16 | −0.009 |
| SVM (RBF) | 0.080 | **0.007** | 0.97 | 0.004 |
| XGBoost | **0.078** | 0.009 | 1.04 | 0.015 |
| MLP | 0.080 | 0.018 | 0.91 | −0.014 |
| ResNet | 0.080 | 0.021 | 0.87 | −0.117 |
| FT-Transformer | 0.079 | 0.012 | **0.99** | −0.100 |

*Ideal values: slope 1, intercept 0.*

## D. Statistical Comparison

Pairwise DeLong tests compared all 21 model pairs (Table IV). After Holm correction for multiple comparisons, XGBoost remained significantly better than RF (adjusted p = 0.012), ResNet (p = 0.036) and SVM (p = 0.040). Its advantage over LR (unadjusted p = 0.005) was no longer significant (adjusted p = 0.095), and it did not differ significantly from the MLP or the FT-Transformer (unadjusted p = 0.064 and 0.075). **No deep-learning model was significantly better than any classical model, with or without correction.** Wherever a deep model had the higher AUC (the MLP and FT-Transformer versus LR, RF and SVM), unadjusted p ranged from 0.161 to 0.460. The MLP–ResNet difference (unadjusted p = 0.031) did not survive correction (adjusted p = 0.525). LR, RF and SVM were statistically indistinguishable from one another (unadjusted p ≥ 0.876). *Table IV below shows unadjusted p-values; Holm-adjusted values are in `results/delong_pairwise_corrected.csv`.*

**Table IV. Pairwise DeLong test p-values for differences in test ROC-AUC**

| | LR | RF | SVM | XGB | MLP | ResNet | FT-T |
|---|---|---|---|---|---|---|---|
| LR | – | 0.944 | 0.876 | **0.005** | 0.321 | 0.675 | 0.169 |
| RF | 0.944 | – | 1.000 | **0.001** | 0.460 | 0.643 | 0.239 |
| SVM | 0.876 | 1.000 | – | **0.002** | 0.274 | 0.547 | 0.161 |
| XGB | **0.005** | **0.001** | **0.002** | – | 0.064 | **0.002** | 0.075 |
| MLP | 0.321 | 0.460 | 0.274 | 0.064 | – | **0.031** | 0.682 |
| ResNet | 0.675 | 0.643 | 0.547 | **0.002** | **0.031** | – | 0.075 |
| FT-T | 0.169 | 0.239 | 0.161 | 0.075 | 0.682 | 0.075 | – |

*Bold: p < 0.05. FT-T = FT-Transformer.*

## E. Robustness

The near-identical performance held up under three further checks (Table V).
- **Repeated splits:** across five random train/test splits, XGBoost had the highest mean AUC (0.846), closely followed by the FT-Transformer (0.845), and every model was within about one standard deviation of the others.
- **Temporal validation:** when trained on 2011–2016 and tested on 2017–2018, all seven models converged to an AUC of 0.837–0.840, and XGBoost's small advantage on the main test set disappeared.
- **Survey weighting:** survey-weighted test AUCs (0.851–0.862) preserved the same ranking, with XGBoost highest.

**Table V. Robustness of discrimination**

| Model | AUC, 5 splits (mean ± SD) | Temporal AUC (2017–18) | Temporal Brier | Survey-weighted test AUC |
|---|---|---|---|---|
| Logistic Regression | 0.842 ± 0.002 | 0.838 | 0.088 | 0.852 |
| Random Forest | 0.841 ± 0.004 | 0.837 | 0.089 | 0.851 |
| SVM (RBF) | 0.843 ± 0.003 | 0.838 | 0.089 | 0.853 |
| XGBoost | **0.846 ± 0.004** | 0.838 | 0.088 | **0.862** |
| MLP | 0.844 ± 0.003 | 0.839 | 0.088 | 0.855 |
| ResNet | 0.841 ± 0.005 | **0.840** | 0.088 | 0.851 |
| FT-Transformer | 0.845 ± 0.003 | 0.837 | 0.089 | 0.853 |

## F. Computational Cost

Although accuracy was similar, computational cost differed by several orders of magnitude (Table VI). The FT-Transformer required 5,779 s (1.6 h) of tuning, 119 times longer than XGBoost (49 s). Its final training took 275.6 s, about 445 times longer than XGBoost and 5,000 times longer than LR. ResNet and the MLP were cheaper, but both still took longer to tune than every classical model. The SVM had the slowest inference (828 ms per 1,000 predictions) and SHAP explanation time (1,615 s).

**Table VI. Computational cost (CPU, wall-clock)**

| Model | Parameters | Tuning (s) | Training (s) | Inference (ms / 1,000) | SHAP, 300 rows (s) |
|---|---|---|---|---|---|
| Logistic Regression | 21 | 5 | 0.05 | 1.1 | 10 |
| Random Forest | 500 trees | 271 | 1.5 | 24.8 | 120 |
| SVM (RBF) | – | 224 | 47.4 | 828 | 1,615 |
| XGBoost | 502 trees | 49 | 0.6 | 2.9 | 10 |
| MLP | 3,649 | 352 | 9.0 | 1.2 | 5 |
| ResNet | 68,353 | 913 | 22.3 | 8.8 | 198 |
| FT-Transformer | 69,589 | 5,779 | 275.6 | 351 | 1,300 |

## G. Feature Attributions and Their Stability

**Most important features.** SHAP analysis identified **age as the most influential feature for every model** (Fig. 2). Total cholesterol, smoking status, diagnosed diabetes and income-to-poverty ratio completed the top five for LR, SVM and all three deep models. RF and XGBoost ranked serum creatinine in their top five instead of one of these. Fig. 3 shows feature effects for the best model, XGBoost. Older age, smoking, diagnosed diabetes and higher creatinine raised predicted risk, whereas higher income-to-poverty ratio and, counter-intuitively, higher total cholesterol lowered it (Section IV-H).

[[FIG:fig2_shap_importance_heatmap.png|0.62|Fig. 2. Mean |SHAP| importance per model, scaled to each model's top feature. Numbers in the cells are within-model ranks; the white line separates classical (left) from deep (right) models.]]

**Agreement across models.** Feature rankings were broadly concordant across all seven models (Kendall's W = 0.752), but agreement followed the shape of the decision function rather than the classical/deep label (Table VII). The two tree ensembles agreed closely with each other (RF–XGBoost ρ = 0.941). LR, SVM and the three deep models formed a second cluster of smooth-boundary models, for example ResNet–FT-Transformer ρ = 0.935, SVM–MLP 0.886 and MLP–ResNet 0.863. The weakest agreement occurred between tree and linear models (LR–RF ρ = 0.474). The three deep models were highly concordant with one another (W = 0.921; mean ρ = 0.882), more so than the four classical models (W = 0.742; mean ρ = 0.657).

**Stability within each model.** Refitting each model on five bootstrap samples showed that the tree ensembles produced the most stable full rankings (RF ρ = 0.973; XGBoost 0.921). The SVM and MLP kept a perfectly stable top five (Jaccard 1.00). The FT-Transformer was the least stable (ρ = 0.761; top-5 Jaccard 0.61).

**Table VII. SHAP ranking agreement between models (Spearman ρ) and within-model stability**

| | LR | RF | SVM | XGB | MLP | ResNet | FT-T |
|---|---|---|---|---|---|---|---|
| RF | 0.474 | | | | | | |
| SVM | 0.656 | 0.653 | | | | | |
| XGB | 0.553 | **0.941** | 0.663 | | | | |
| MLP | 0.743 | 0.629 | **0.886** | 0.633 | | | |
| ResNet | 0.666 | 0.663 | 0.698 | 0.717 | 0.863 | | |
| FT-T | 0.728 | 0.597 | 0.692 | 0.681 | 0.848 | **0.935** | |
| Bootstrap stability (ρ) | 0.846 | 0.973 | 0.732 | 0.921 | 0.805 | 0.804 | 0.761 |
| Top-5 stability (Jaccard) | 0.867 | 0.767 | 1.000 | 0.733 | 1.000 | 0.767 | 0.605 |

[[FIG:fig3_shap_beeswarm_XGB.png|0.62|Fig. 3. SHAP beeswarm plot for XGBoost. Each dot is one test participant; colour shows the feature value (red high, blue low) and position shows its effect on predicted CVD probability. Grey dots are missing values.]]

## H. Feature-Group Ablation and the Treatment Effect

Removing one feature group at a time confirmed the SHAP results (Fig. 4). Removing the **demographic** group caused by far the largest loss for every model (−0.037 to −0.063 AUC). The loss was largest for LR (−0.063), which relies most heavily on age. Lipid, glycaemic, socioeconomic, renal/metabolic and smoking groups each cost between 0.001 and 0.007 AUC. Removing **anthropometric or blood-pressure** features changed AUC by at most 0.003 for every model, which is within the variation between splits. The ordering of groups was nearly identical across all seven models.

[[FIG:fig4_ablation.png|0.80|Fig. 4. Change in test ROC-AUC (× 100) when each feature group is removed and the model retrained, averaged over five random splits.]]

The inverse cholesterol attribution and the near-zero contribution of blood pressure are explained by treatment after diagnosis (Table VIII). Survey-weighted statin use was 58.0% among participants with CVD versus 14.2% among those without, and antihypertensive use was 62.8% versus 20.1%. Among statin users, mean total cholesterol was 166.4 mg/dL in the CVD group versus 183.0 mg/dL in the non-CVD group. Because the outcome is prevalent CVD measured at a single point in time, these treatment effects lower the measured risk factors of diagnosed patients. The models' attributions therefore reflect predictive association, not causal risk.

**Table VIII. Medication use and treated risk factors by CVD status (supplementary check; not model inputs)**

| Measure | No CVD | CVD |
|---|---|---|
| Statin use, % (weighted) | 14.2 | 58.0 |
| BP medication use, % (weighted) | 20.1 | 62.8 |
| Total cholesterol, statin users (mg/dL) | 183.0 | 166.4 |
| Total cholesterol, non-users (mg/dL) | 193.5 | 188.5 |
| Systolic BP, BP-medication users (mmHg) | 134.3 | 135.4 |
| Systolic BP, non-users (mmHg) | 120.7 | 127.6 |

## I. Summary of Findings

1. **Discrimination was nearly identical across all seven models** (ROC-AUC 0.837–0.848). XGBoost was highest on the main test set and across repeated splits, with the FT-Transformer a close second (0.845 vs. 0.846).
2. **No deep-learning model significantly outperformed any classical model** (DeLong p ≥ 0.161 wherever a deep model had the higher AUC).
3. **All models were well calibrated** (ECE ≤ 0.021). The FT-Transformer had the best calibration slope (0.99); ResNet and the MLP were slightly over-confident.
4. **Performance was stable over time:** all models reached an AUC of 0.837–0.840 on the 2017–2018 cycle.
5. **Deep models cost far more:** the FT-Transformer needed 119× longer tuning than XGBoost for no accuracy gain.
6. **Age was the most important feature for every model.** Explanations clustered by the shape of the decision function (tree vs. smooth), not by classical vs. deep. The deep models agreed with one another almost perfectly (W = 0.92).
7. **Cholesterol and blood-pressure attributions were distorted by treatment after diagnosis.** SHAP values should be read as association, not causation.
