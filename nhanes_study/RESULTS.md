# Classical ML vs. Deep Learning for CVD Classification on NHANES 2011–2018: Experimental Results

All numbers below come from the scripts in this folder (run order `00` → `06`) and are reproducible with seed 42.
Tables are in `results/tables/` (CSV and Markdown) and figures in `figures/` (300-dpi PNG and vector PDF).
**7 models are compared: 4 classical ML (LR, RF, SVM, XGBoost) and 3 deep learning (MLP, ResNet, FT-Transformer).**

---

## 1. What was done (Methodology section material)

**Data.** Four NHANES cycles: 2011–2012 (G), 2013–2014 (H), 2015–2016 (I) and 2017–2018 (J), downloaded from the CDC.
Files used: DEMO, BPX, BMX, MCQ, TCHOL, HDL, GHB, BIOPRO, SMQ and DIQ. BPQ was used only for the supplementary medication check.
Every variable name was checked to exist with identical coding in all four cycles. 2019–2020 was excluded because COVID-19 interrupted its data collection.

**Population.** Adults aged ≥ 20 who completed the examination (RIDSTATR = 2) and had a determinable outcome.

| Cycle | Participants | Examined adults ≥20 | Outcome determined | CVD cases |
|---|---:|---:|---:|---:|
| 2011–2012 | 9,756 | 5,319 | 5,302 | 550 |
| 2013–2014 | 10,175 | 5,588 | 5,572 | 574 |
| 2015–2016 | 9,971 | 5,474 | 5,448 | 598 |
| 2017–2018 | 9,254 | 5,265 | 5,230 | 647 |
| **Total** | 39,156 | 21,646 | **21,552** | **2,369** |

**Outcome.** CVD = 1 if the participant reported ever being told by a doctor they had congestive heart failure (MCQ160B), coronary heart disease (MCQ160C), angina (MCQ160D), heart attack (MCQ160E) or stroke (MCQ160F). CVD = 0 if "no" to all five. Anyone else was excluded (n = 94). Prevalence: 11.0% unweighted, 8.9% survey-weighted (MEC weight divided by 4).

**Features (20).** Age, sex, race/ethnicity (4 indicators; reference non-Hispanic White), income-to-poverty ratio, education, systolic and diastolic BP (mean of up to 3 readings), pulse rate, BMI, waist circumference, total cholesterol, HDL cholesterol, HbA1c, serum creatinine, uric acid, smoking status and diagnosed diabetes. Medication variables were **excluded** from the inputs, because treatment often follows a CVD diagnosis and would leak the label. The most missing feature was income-to-poverty ratio (9.8%).

**Split and preprocessing.** Stratified 80/20 split: training n = 17,241 (1,895 cases) and test n = 4,311 (474 cases). Median imputation and standardisation were fitted inside each pipeline on training folds only.

**Models and tuning** (5-fold stratified CV on the training set, objective ROC-AUC):

| Model | Family | Search space | Selected |
|---|---|---|---|
| Logistic Regression (L2) | Classical | C ∈ {1e-3 … 100} | C = 0.01 |
| Random Forest (500 trees) | Classical | depth × min leaf × max_features (18 configurations) | depth 12, leaf 20, sqrt |
| SVM (RBF, Platt probabilities) | Classical | C × γ × class_weight (32 configurations) | C = 10, γ = 0.001, balanced |
| XGBoost | Classical | 40-iteration random search (7 hyperparameters) | 502 trees, depth 5, lr 0.0113 |
| MLP | Deep | 5 architectures × dropout × weight decay × lr (60 configurations) | 64-32, dropout 0.5, wd 0.01, lr 1e-3 |
| Tabular ResNet | Deep | width {64,128,256} × blocks {1,2,4} × dropout {0.1,0.3,0.5} × wd (54 configurations) | width 64, 4 blocks, dropout 0.5, wd 0.01 |
| FT-Transformer | Deep | width {64,128} × blocks {2,3} × lr {1e-4,3e-4} (8 configurations) | width 64, 2 blocks, 8 heads, lr 3e-4 |

All three deep models (PyTorch) share one training loop: AdamW optimiser, binary cross-entropy loss, batch size 256, and early stopping on a 15% validation split (patience 15).
- **MLP:** Linear–BatchNorm–ReLU–Dropout blocks; 3,649 parameters.
- **ResNet** (Gorishniy et al., 2021): residual blocks of BN–Linear–ReLU–Dropout–Linear; 68,353 parameters.
- **FT-Transformer** (Gorishniy et al., 2021): a feature tokenizer, a [CLS] token, pre-norm Transformer blocks with multi-head self-attention and ReGLU feed-forward layers; 69,589 parameters.

For ResNet and FT-Transformer, the best configurations sit on a plateau: the top 8 ResNet configurations lie within 0.0016 CV-AUC of each other, and all 8 FT-Transformer configurations within 0.003. Both gaps are well below the CV standard deviation (about 0.006–0.009), so extending the grids would not change the results.

**Decision threshold.** For each model, the threshold maximising F1 on out-of-fold training predictions. The test set was never used for any choice.

**Metrics.** ROC-AUC (1,000-sample bootstrap 95% CI), PR-AUC, precision, recall, specificity, F1, Brier score, ECE (10 quantile bins), calibration slope and intercept, pairwise DeLong tests, survey-weighted test AUC, and tuning, fit, inference and SHAP time.

**SHAP.** The same model-agnostic method for all 7 models: permutation SHAP on the predicted probability, with 300 test rows, 50 background rows and 164 evaluations per row. Validated against exact TreeSHAP for XGBoost: ranking Spearman ρ = **0.989**.

**Stability.**
1. **Across models:** pairwise Spearman correlation of mean |SHAP|, top-5 Jaccard, and Kendall's W overall and within each family.
2. **Within each model:** 5 bootstrap refits of the training set, compared pairwise.

**Ablation.** Leave one feature group out (8 groups), retrain with the tuned settings, and average ΔAUC over 5 random splits. **Robustness:** full-model AUC over the same 5 splits. **Temporal validation:** train on 2011–2016 (n = 16,322), test on 2017–2018 (n = 5,230).

**Compute.** All tuning, final models and reported timings use the CPU. To save time, FT-Transformer's ablation, temporal and bootstrap refits ran on a GPU (RTX 3050). These analyses compare refits of the same model with each other, so the device is consistent within every comparison.

---

## 2. Results

### Table 1: Survey-weighted characteristics (`table1_characteristics.md`)
CVD cases were older (65.0 vs 46.1 years), more often had diagnosed diabetes (33.0% vs 8.3%), and were more often former smokers (38.0% vs 23.1%). Notably, they had **lower** total cholesterol (177.5 vs 193.0 mg/dL) and lower diastolic BP (67.8 vs 71.3 mmHg); see §3.

### Table 2: Test-set performance (n = 4,311; 474 cases)

| Model | Family | CV AUC (SD) | Test AUC [95% CI] | PR-AUC | Precision | Recall | Specificity | F1 | Brier | ECE | Cal. slope |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Classical | 0.846 (0.009) | 0.839 [0.821–0.855] | 0.392 | 0.383 | 0.561 | 0.888 | 0.455 | 0.080 | 0.010 | 1.04 |
| Random Forest | Classical | 0.847 (0.003) | 0.839 [0.821–0.856] | 0.407 | 0.361 | 0.586 | 0.872 | 0.447 | 0.080 | 0.010 | 1.16 |
| SVM (RBF) | Classical | 0.849 (0.007) | 0.839 [0.821–0.854] | 0.396 | 0.362 | 0.584 | 0.873 | 0.447 | 0.080 | 0.007 | 0.97 |
| **XGBoost** | Classical | 0.850 (0.005) | **0.848 [0.829–0.863]** | 0.418 | 0.373 | 0.572 | 0.881 | 0.451 | **0.078** | 0.009 | 1.04 |
| MLP | Deep | 0.849 (0.007) | 0.842 [0.825–0.858] | 0.398 | 0.390 | 0.523 | 0.899 | 0.447 | 0.080 | 0.018 | 0.91 |
| ResNet | Deep | 0.849 (0.009) | 0.837 [0.819–0.855] | 0.402 | 0.377 | 0.527 | 0.892 | 0.439 | 0.080 | 0.021 | 0.87 |
| FT-Transformer | Deep | 0.850 (0.007) | 0.843 [0.825–0.859] | **0.419** | 0.371 | 0.586 | 0.877 | 0.455 | 0.079 | 0.012 | 0.99 |

**Pairwise DeLong p-values** (`table_delong_pairwise.md`):

| | LR | RF | SVM | XGB | MLP | ResNet | FTT |
|---|---|---|---|---|---|---|---|
| LR | – | 0.944 | 0.876 | **0.005** | 0.321 | 0.675 | 0.169 |
| RF | 0.944 | – | 1.000 | **0.001** | 0.460 | 0.643 | 0.239 |
| SVM | 0.876 | 1.000 | – | **0.002** | 0.274 | 0.547 | 0.161 |
| XGB | **0.005** | **0.001** | **0.002** | – | 0.064 | **0.002** | 0.075 |
| MLP | 0.321 | 0.460 | 0.274 | 0.064 | – | **0.031** | 0.682 |
| ResNet | 0.675 | 0.643 | 0.547 | **0.002** | **0.031** | – | 0.075 |
| FTT | 0.169 | 0.239 | 0.161 | 0.075 | 0.682 | 0.075 | – |

- The 7 models span only **0.011 AUC** (0.837–0.848).
- XGBoost had the highest AUC. **After Holm correction for the 21 pairwise comparisons** (`results/delong_pairwise_corrected.csv`), it remained significantly better than RF (adjusted p = 0.012), ResNet (0.036) and SVM (0.040). Its advantage over LR (unadjusted p = 0.005) did not survive correction (adjusted p = 0.095). It was not significantly better than the MLP or FT-Transformer (unadjusted p = 0.064 and 0.075; adjusted 1.000).
- **No deep model was significantly better than any classical model,** with or without correction. Wherever a deep model's AUC was higher than a classical model's (MLP and FT-Transformer vs LR, RF and SVM), unadjusted p ≥ 0.161.
- The MLP vs ResNet difference (unadjusted p = 0.031) did not survive correction (adjusted p = 0.525).
- All models were well calibrated (ECE ≤ 0.021). FT-Transformer had the slope closest to 1 (0.99). ResNet (0.87) and MLP (0.91) were slightly over-confident, and RF (1.16) slightly under-confident.
- Survey-weighted test AUC: 0.851–0.862, with XGBoost again highest.

**Repeated splits (5 seeds, `repeated_split_auc.md`):** mean AUC LR 0.842 ± 0.002, RF 0.841 ± 0.004, SVM 0.843 ± 0.003, **XGB 0.846 ± 0.004**, MLP 0.844 ± 0.003, ResNet 0.841 ± 0.005, FTT 0.845 ± 0.003. XGBoost is highest with FT-Transformer a close second, and all differences are within about one SD.

**Temporal validation (train 2011–2016 → test 2017–2018, `temporal_validation.md`):** AUC LR 0.838, RF 0.837, SVM 0.838, XGB 0.838, MLP 0.839, ResNet 0.840, FTT 0.837; Brier 0.088–0.089 for all. All 7 models converge to 0.837–0.840, so XGBoost's held-out advantage disappears.

### Computational efficiency (`table_efficiency.md`)

| Model | Parameters | Tuning (s) | Final fit (s) | Inference (ms / 1k rows) | SHAP, 300 rows (s) |
|---|---:|---:|---:|---:|---:|
| LR | 21 | 5 | 0.05 | 1.1 | 10 |
| RF | 500 trees | 271 | 1.5 | 24.8 | 120 |
| SVM | – | 224 | 47.4 | 828 | 1,615 |
| XGB | 502 trees | 49 | 0.6 | 2.9 | 10 |
| MLP | 3,649 | 352 | 9.0 | 1.2 | 5 |
| ResNet | 68,353 | 913 | 22.3 | 8.8 | 198 |
| FT-Transformer | 69,589 | **5,779** | **275.6** | 351 | 1,300 |

These are wall-clock times on a 12-core CPU, with tuning run in parallel. FT-Transformer took 119× longer to tune than XGBoost and about 445× longer to fit (5,000× longer than LR), with no significant accuracy gain.

### Figure 2 / Table 3: SHAP importance and stability
- **Age was the #1 feature for all 7 models.** Total cholesterol, smoking, diagnosed diabetes and income-to-poverty ratio complete the top 5 for LR, SVM and all three deep models. RF and XGB bring serum creatinine into the top 5.
- Overall concordance across all 7 models: **Kendall's W = 0.752.**
- **The deep models agree with each other more closely than the classical models do:** Kendall's W = **0.921** among the 3 deep models vs **0.742** among the 4 classical models. The mean pairwise Spearman ρ is 0.882 within the deep family, 0.657 within the classical family and 0.694 between families.
- **Agreement reflects the shape of the decision function, not the classical/deep label.** The tree ensembles agree tightly with each other (RF–XGB ρ = 0.941) and less with everything else. The smooth-function models form a second cluster: ResNet–FTT 0.935, SVM–MLP 0.886, MLP–ResNet 0.863, MLP–FTT 0.848, LR–MLP 0.743. The pairs least in agreement mix trees with smooth models (LR–RF 0.474).
- **Within-model stability (bootstrap refits):** RF ρ = 0.973, XGB 0.921, LR 0.846, MLP 0.805, ResNet 0.804, FTT 0.761, SVM 0.732.
  - Tree models give the most stable full rankings.
  - MLP and SVM keep a perfectly stable top 5 (Jaccard 1.00).
  - FT-Transformer is the least stable (top-5 Jaccard 0.61).
- Tree models give more weight to continuous laboratory and physiological measures (creatinine, HbA1c, systolic BP). The smooth models give more weight to demographics and smoking.

### Figure 4 / Table 4: Leave-one-group-out ablation (ΔAUC, mean ± SD over 5 splits)
- **Removing demographics** caused the largest drop for every model (−0.037 to −0.063), largest for LR (−0.063).
- Lipid, glycaemic, socioeconomic, renal/metabolic and smoking groups each cost 0.001–0.007 AUC.
- **Anthropometric and blood-pressure groups contributed almost nothing** (|ΔAUC| ≤ 0.003 for every model). ResNet's small positive values (+0.003 ± 0.003 and +0.002 ± 0.001) are within noise.
- The ordering of groups is nearly identical across all 7 models, which agrees with SHAP at the group level.

---

## 3. Discussion points (non-causal)

1. **Deep learning gave no measurable advantage.** Neither the MLP nor the two architectures designed for tabular data (ResNet and FT-Transformer) significantly outperformed any classical model. This is consistent with benchmarks showing that tree ensembles match or beat deep models on medium-sized tabular data. Under temporal validation, all 7 models converge to an AUC of about 0.84.
2. **Simplicity is competitive.** LR came within 0.009 AUC of the best model, was well calibrated, and was more than 5,000× cheaper to train than FT-Transformer.
3. **Attribution agreement depends on the shape of the decision function.** Models with similar AUC distribute credit differently. Tree ensembles form one cluster and smooth-function models (LR, SVM and all three deep models) another. The three deep models explain their predictions almost identically (W = 0.92). SHAP from a single model gives an incomplete picture.
4. **Reverse causation from treatment (important caveat).** In every model, SHAP shows *higher* total cholesterol associated with *lower* predicted CVD probability. This is explained by treatment after diagnosis (`06_medication_check.py`): weighted statin use was 58.0% among CVD cases vs 14.2% among non-cases, and among statin users mean cholesterol was 166.4 mg/dL (CVD) vs 183.0 (no CVD). Similarly, 62.8% of cases vs 20.1% of non-cases take BP medication, which likely explains why blood pressure added almost no AUC. Because the outcome is **prevalent, self-reported** CVD measured cross-sectionally, the models partly learn markers of *having been diagnosed and treated*, not *risk of developing* CVD. The attributions describe predictive association only.
5. **Limitations.**
   - The outcome is cross-sectional and self-reported (recall bias; prevalent, not incident).
   - Age is top-coded at 80.
   - Survey weights were used for descriptive statistics and a weighted-AUC check, not in model training.
   - Permutation SHAP is approximate (validated at ρ = 0.989 against exact TreeSHAP).
   - Some selected deep-learning hyperparameters lie at the edges of their grids, but the CV surfaces are flat there (see Methodology).
   - FT-Transformer refits for ablation and stability ran on a GPU.

---

## 4. Results-based abstract (about 190 words)

> Cardiovascular disease (CVD) is a leading cause of death, yet it remains unclear whether deep learning improves CVD classification over classical machine learning on structured clinical data. Using four cycles of the U.S. National Health and Nutrition Examination Survey (NHANES 2011–2018; n = 21,552 adults, 2,369 with CVD), we compared four classical models (Logistic Regression, Random Forest, Support Vector Machine, XGBoost) with three deep learning models (multilayer perceptron, tabular ResNet, FT-Transformer) using 20 demographic, clinical and laboratory features. Models were evaluated for discrimination (ROC-AUC, PR-AUC, DeLong test), calibration (Brier score, expected calibration error) and computational cost, and interpreted with SHAP. All models performed similarly (ROC-AUC 0.837–0.848). XGBoost achieved the highest AUC, and no deep learning model significantly outperformed any classical model, although the FT-Transformer required 119 times longer tuning than XGBoost. Age was the most influential feature for every model. Feature rankings were broadly concordant (Kendall's W = 0.75) but clustered by the shape of the decision function, with the deep models agreeing closely with each other (W = 0.92). Inverse attributions for cholesterol reflected treatment after diagnosis, indicating association rather than causation. For structured health survey data, deep learning offered no measurable advantage over classical machine learning.

---

## 5. Files

| Script | Purpose |
|---|---|
| `00_download_nhanes.py` | Download the NHANES files from the CDC |
| `01_build_dataset.py` | Merge and harmonise 4 cycles → `data/nhanes_cvd_2011_2018.csv`, `data/cohort_flow.csv` |
| `common.py` | Features, pipelines, PyTorch MLP/ResNet/FT-Transformer, metrics, DeLong test, bootstrap |
| `02_train_evaluate.py` | Tuning, OOF thresholds, test metrics, pairwise DeLong → `results/`, `models/` |
| `03_shap_stability.py` | SHAP for 7 models, TreeSHAP check, cross-model and within-model stability |
| `04_ablation_robustness.py` | Group ablation, repeated splits, temporal validation |
| `05_figures_tables.py` | Figures 1–4 and all tables |
| `06_medication_check.py` | Supplementary statin / BP-medication check |

Figures: `fig1_roc_calibration` (ROC and calibration), `fig2_shap_importance_heatmap`, `fig3_shap_beeswarm_XGB`, `fig4_ablation`.
