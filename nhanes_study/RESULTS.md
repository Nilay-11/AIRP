# Classical ML vs. Deep Learning for CVD Classification on NHANES 2011–2018 — Experimental Results

All numbers below come from the scripts in this folder (run order `01` → `06`) and are reproducible (seed 42).
Tables are in `results/tables/` (CSV + Markdown), figures in `figures/` (300-dpi PNG + vector PDF).

---

## 1. What was done (Methodology section material)

**Data.** Four NHANES cycles: 2011–2012 (G), 2013–2014 (H), 2015–2016 (I), 2017–2018 (J), downloaded from CDC
(files DEMO, BPX, BMX, MCQ, TCHOL, HDL, GHB, BIOPRO, SMQ, DIQ; BPQ used only for the supplementary medication check).
All variable names were verified to exist with identical coding in all four cycles. 2019–2020 was excluded because
collection was interrupted by COVID-19.

**Population.** Adults ≥ 20 y who completed the mobile examination (RIDSTATR = 2) and had a determinable outcome.

| Cycle | Participants | Examined adults ≥20 | Outcome determined | CVD cases |
|---|---:|---:|---:|---:|
| 2011–2012 | 9,756 | 5,319 | 5,302 | 550 |
| 2013–2014 | 10,175 | 5,588 | 5,572 | 574 |
| 2015–2016 | 9,971 | 5,474 | 5,448 | 598 |
| 2017–2018 | 9,254 | 5,265 | 5,230 | 647 |
| **Total** | 39,156 | 21,646 | **21,552** | **2,369** |

**Outcome.** CVD = 1 if the participant reported ever being told by a doctor they had congestive heart failure
(MCQ160B), coronary heart disease (MCQ160C), angina (MCQ160D), heart attack (MCQ160E) or stroke (MCQ160F); 0 if
"no" to all five; excluded otherwise (refused/don't know/missing; n = 94). Prevalence: 11.0 % unweighted,
8.9 % survey-weighted (MEC weight / 4, per NCHS guidance for combining cycles).

**Features (20).** Age; sex; race/ethnicity (4 indicators, ref. non-Hispanic White); income-to-poverty ratio;
education; systolic and diastolic BP (mean of up to 3 readings, zero diastolic treated as missing); pulse rate; BMI;
waist circumference; total cholesterol; HDL cholesterol; HbA1c; serum creatinine; uric acid; smoking status
(never/former/current); diagnosed diabetes. Medication variables were deliberately **excluded** as features because
treatment is frequently a consequence of diagnosed CVD (label leakage). Maximum missingness: income-to-poverty ratio 9.8 %;
all other features ≤ 6.3 %.

**Split and preprocessing.** Stratified 80/20 split (train n = 17,241 / 1,895 cases; test n = 4,311 / 474 cases).
Median imputation and standardisation were fitted inside each pipeline on training folds only (no leakage).

**Models and tuning** (5-fold stratified CV on the training set, objective ROC-AUC):

| Model | Search space | Selected |
|---|---|---|
| Logistic Regression (L2) | C ∈ {1e-3 … 100} | C = 0.01 |
| Random Forest (500 trees) | depth {8,12,None} × min leaf {5,20,50} × max_features {sqrt,0.5} | depth 12, leaf 20, sqrt |
| SVM (RBF, Platt probabilities) | C {0.1,1,10,100} × γ {scale,0.01,0.001,0.0003} × class_weight {None,balanced} | C = 10, γ = 0.001, balanced |
| XGBoost | 40-iteration random search (7 hyperparameters) | 502 trees, depth 5, lr 0.0113 |
| MLP (PyTorch) | 5 architectures × dropout {0.1,0.3,0.5} × weight decay {1e-4,1e-2} × lr {1e-3,3e-4} (60 configs) | 64-32, dropout 0.5, wd 0.01, lr 1e-3 |

MLP: Linear–BatchNorm–ReLU–Dropout blocks, AdamW, BCE loss, batch 256, early stopping on a 15 % internal validation
split (patience 15); final model 3,649 parameters, stopped at epoch 89.

**Decision threshold.** Chosen per model to maximise F1 on out-of-fold training predictions, then applied unchanged to
the test set (the test set was never used for any choice).

**Metrics.** ROC-AUC (1,000-sample bootstrap 95 % CI), PR-AUC, precision, recall, specificity, F1, Brier score, ECE
(10 quantile bins), calibration slope and intercept; DeLong test for paired AUC differences; survey-weighted test AUC;
tuning/fit/inference time; SHAP time.

**SHAP.** One model-agnostic method for all five models (permutation SHAP on the predicted probability; 300 test rows,
50 training background rows, 164 evaluations/row) so attributions are directly comparable. Validated against exact
interventional TreeSHAP for XGBoost: importance-ranking Spearman ρ = **0.989**.

**Stability.** (i) Cross-model: pairwise Spearman/Kendall correlation of mean |SHAP| vectors, top-5 Jaccard,
Kendall's W across all five models. (ii) Within-model: 5 bootstrap resamples of the training set, refit, recompute SHAP,
pairwise Spearman and top-5 Jaccard.

**Ablation.** Leave-one-feature-group-out (8 groups), retrained with tuned hyperparameters, ΔAUC averaged over 5
different random stratified splits. **Robustness:** same 5 splits for full-model AUC variability; **temporal
validation:** train on 2011–2016 (n = 16,322), test on 2017–2018 (n = 5,230).

---

## 2. Results

### Table 1 — Survey-weighted characteristics (`table1_characteristics.md`)
CVD cases were older (65.0 vs 46.1 y), more often had diagnosed diabetes (33.0 % vs 8.3 %) and were more often former
smokers (38.0 % vs 23.1 %). Notably, CVD cases had **lower** total cholesterol (177.5 vs 193.0 mg/dL) and diastolic BP
(67.8 vs 71.3 mmHg) — see §3.

### Table 2 — Test-set performance (n = 4,311; 474 cases)

| Model | CV AUC (SD) | Test AUC [95 % CI] | PR-AUC | Precision | Recall | F1 | Brier | ECE | Cal. slope | DeLong p vs MLP |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.846 (0.009) | 0.839 [0.821–0.855] | 0.392 | 0.383 | 0.561 | 0.455 | 0.080 | 0.010 | 1.04 | 0.321 |
| Random Forest | 0.847 (0.003) | 0.839 [0.821–0.856] | 0.407 | 0.361 | 0.586 | 0.447 | 0.080 | 0.010 | 1.16 | 0.460 |
| SVM (RBF) | 0.849 (0.007) | 0.839 [0.821–0.854] | 0.396 | 0.362 | 0.584 | 0.447 | 0.080 | 0.007 | 0.97 | 0.274 |
| **XGBoost** | 0.850 (0.005) | **0.848 [0.829–0.863]** | **0.418** | 0.373 | 0.572 | 0.451 | **0.078** | 0.009 | 1.04 | 0.064 |
| MLP | 0.849 (0.007) | 0.842 [0.825–0.858] | 0.398 | 0.390 | 0.523 | 0.447 | 0.080 | 0.018 | 0.91 | — |

- XGBoost had the highest AUC; it was significantly better than LR (p = 0.005), RF (p < 0.001) and SVM (p = 0.002),
  but **not** significantly better than the MLP (p = 0.064).
- The MLP was **not** significantly different from any classical model (all DeLong p ≥ 0.064).
- All models span only 0.009 AUC; F1 spans 0.447–0.455.
- All models were well calibrated (ECE ≤ 0.018). The MLP had the largest ECE and a slope < 1 (slightly over-confident);
  RF had a slope > 1 (slightly under-confident).
- Survey-weighted test AUC: 0.851–0.862, same ordering.

**Repeated splits (5 seeds, `repeated_split_auc.md`):** mean AUC LR 0.842 ± 0.002, RF 0.841 ± 0.004,
SVM 0.843 ± 0.003, XGB 0.846 ± 0.004, MLP 0.844 ± 0.003 — differences are within one SD.

**Temporal validation (train 2011–2016 → test 2017–2018, `temporal_validation.md`):** AUC LR 0.838, RF 0.837,
SVM 0.838, XGB 0.838, MLP 0.839; Brier 0.088 for all. No model degraded meaningfully over time and the ranking
advantage of XGBoost disappeared.

### Computational efficiency (`table_efficiency.md`)

| Model | Tuning (s) | Final fit (s) | Inference (ms / 1k rows) | SHAP, 300 rows (s) |
|---|---:|---:|---:|---:|
| LR | 5 | 0.05 | 1.1 | 10 |
| RF | 271 | 1.5 | 24.8 | 120 |
| SVM | 224 | 47.4 | 828 | 1,615 |
| XGB | 49 | 0.6 | 2.9 | 10 |
| MLP | 352 | 9.0 | 1.2 | 5 |

The MLP was the most expensive model to tune and ~170× slower to fit than LR for no significant accuracy gain.
SVM was by far the slowest at inference and explanation. (Wall-clock times, 12-core CPU; tuning ran in parallel.)

### Figure 2 / Table 3 — SHAP feature importance and stability
- **Age was the #1 feature for all five models.** The same five features — age, smoking, total cholesterol,
  diagnosed diabetes, income-to-poverty ratio — form the top 5 for LR, SVM and MLP; RF and XGB swap in serum creatinine.
- Overall cross-model concordance: **Kendall's W = 0.746**.
- **Agreement clusters by model family, not by "classical vs. deep":** RF–XGB ρ = 0.941 (tree ensembles);
  SVM–MLP ρ = 0.886 and LR–MLP ρ = 0.743 (smooth decision functions); cross-family pairs ρ = 0.47–0.66.
  The MLP's attributions are closer to LR/SVM than the tree models are.
- **Within-model stability (bootstrap refits):** RF ρ = 0.973, XGB 0.921, LR 0.846, MLP 0.805, SVM 0.732. Tree models
  give the most stable full rankings, but LR/SVM/MLP keep a perfectly stable top 5 (Jaccard 1.00 for SVM and MLP,
  0.87 for LR) whereas RF/XGB top-5 sets vary (0.77 / 0.73) because several mid-ranked features have near-equal importance.
- Tree models assign more importance to continuous lab/physiological measures (creatinine, HbA1c, systolic BP),
  while LR/SVM/MLP assign more to demographics (ethnicity indicators, smoking).

### Figure 4 / Table 4 — Leave-one-group-out ablation (ΔAUC, mean ± SD over 5 splits)
- Removing **demographics** caused the largest drop for every model (−0.039 to −0.063), largest for LR (−0.063),
  i.e., LR relies most on age; nonlinear models partially recover the signal from correlated clinical variables.
- Lipids, glycaemic, socioeconomic, renal/metabolic and smoking groups each cost 0.002–0.007 AUC.
- **Anthropometric and blood-pressure groups contributed ≈ 0** (|ΔAUC| ≤ 0.0013) for every model.
- The ranking of groups was nearly identical across all five models — ablation agrees with SHAP at the group level.

---

## 3. Discussion points (non-causal)

1. **Deep learning gave no measurable advantage.** The MLP was statistically indistinguishable from LR, RF and SVM
   and non-significantly below XGBoost, consistent with prior evidence that gradient-boosted trees match or exceed neural
   networks on tabular data. Under temporal validation all five converge to AUC ≈ 0.838.
2. **Simplicity is competitive.** LR reached within 0.009 AUC of the best model, was well calibrated, and was ~170×
   cheaper to train than the MLP.
3. **Attribution agreement depends on model family.** Similar AUC does not imply similar explanations; tree ensembles
   and smooth models distribute credit differently among correlated features. Reporting SHAP from a single model would
   give an incomplete picture.
4. **Reverse causation from treatment (important caveat).** SHAP shows *higher* total cholesterol associated with
   *lower* predicted CVD probability in all models. This is explained by treatment after diagnosis
   (`06_medication_check.py`): weighted statin use 58.0 % in CVD cases vs 14.2 % in non-cases; among statin users
   mean cholesterol was 166.4 mg/dL (CVD) vs 183.0 (no CVD). Similarly 62.8 % of cases vs 20.1 % of non-cases take
   BP medication, which likely explains why blood pressure added ≈ 0 AUC. Because the outcome is **prevalent,
   self-reported** CVD measured cross-sectionally, models learn markers of *having been diagnosed and treated*, not
   *risk of developing* CVD. Attributions describe predictive association only.
5. **Limitations.** Cross-sectional, self-reported outcome (recall bias, no incidence); age top-coded at 80; survey
   weights used for descriptive statistics and a weighted AUC sensitivity check, but not in model training; permutation
   SHAP is approximate (validated ρ = 0.989 vs exact TreeSHAP); for the MLP, the selected dropout (0.5) and weight decay
   (0.01) lie on the edge of the search grid, although the selected architecture (64-32) is interior and the CV
   difference between the top configurations is small.

---

## 4. Results-based abstract (replace the "proposed study" abstract)

> *Abstract*— Cardiovascular disease (CVD) classification from routinely collected clinical data offers a setting to
> compare the accuracy and interpretability of machine-learning methods. Using four cycles (2011–2018) of the National
> Health and Nutrition Examination Survey (n = 21,552 adults; 2,369 with self-reported CVD), we compared Logistic
> Regression, Random Forest, Support Vector Machine and XGBoost with a multilayer perceptron (MLP) on 20 harmonized
> demographic, clinical and laboratory features. All models achieved similar discrimination on a held-out test set
> (ROC-AUC 0.839–0.848); XGBoost performed best (0.848, 95 % CI 0.829–0.863), and the MLP (0.842) did not differ
> significantly from any classical model (DeLong p ≥ 0.064). All models were well calibrated (ECE ≤ 0.018), and under
> temporal validation on 2017–2018 data every model reached AUC ≈ 0.838. The MLP required the longest tuning time and
> about 170 times the training time of Logistic Regression. SHAP analysis identified age as the most influential feature
> for every model, and feature rankings were broadly concordant (Kendall's W = 0.75), but agreement clustered by model
> family: tree ensembles agreed closely with each other (ρ = 0.94), as did smooth-boundary models (SVM–MLP ρ = 0.89).
> Feature-group ablation confirmed the dominance of demographic variables. Inverse attributions for cholesterol were
> traced to post-diagnosis treatment, underscoring that attributions reflect predictive association, not causation.
> For structured clinical survey data, deep learning provided no measurable advantage over classical approaches.

---

## 5. Files

| Script | Purpose |
|---|---|
| `01_build_dataset.py` | Merge & harmonize 4 cycles → `data/nhanes_cvd_2011_2018.csv`, `data/cohort_flow.csv` |
| `common.py` | Features, pipelines, PyTorch MLP, metrics, DeLong, bootstrap |
| `02_train_evaluate.py` | Tuning, OOF thresholds, test metrics, DeLong → `results/test_metrics.csv`, `models/` |
| `03_shap_stability.py` | SHAP for 5 models, TreeSHAP check, cross/within-model stability |
| `04_ablation_robustness.py` | Group ablation, repeated splits, temporal validation |
| `05_figures_tables.py` | Figures 1–4 and all tables |
| `06_medication_check.py` | Supplementary statin / BP-medication check |

Figures: `fig1_roc_calibration` (ROC + calibration), `fig2_shap_importance_heatmap`, `fig3_shap_beeswarm_XGB`,
`fig4_ablation`.
