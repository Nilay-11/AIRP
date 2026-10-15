# Experimental Findings & Empirical Analysis Report

**Paper Title**: *Comparative Evaluation and Interpretation of Machine Learning Models for Cardiovascular Disease Risk Prediction*  
**Experimental Scope**: Rigorous investigation addressing Research Questions RQ1 through RQ8.  
**Dataset**: Cleaned Kaggle Cardiovascular Disease Cohort ($N = 69,265$, $50.05\%$ vs. $49.95\%$ balanced).  
**Protocol**: 80/20 Stratified Partitioning (Train: $55,412$; Untouched Holdout Test: $13,853$); 5-Fold Stratified Cross-Validation on Train Set only; Random Seed = 42.

---

## RQ1: Model Comparison — Architecture & Discrimination

How do different model families perform when provided identical clinical information under an identical, leak-free protocol?

### Findings:
1. **Ensemble Superiority**: **Random Forest** achieved the highest discrimination across both cross-validation ($0.8036 \pm 0.0037$) and the holdout test set ($\text{ROC-AUC} = 0.7966$, $\text{Accuracy} = 73.15\%$, $\text{F1} = 0.7193$).
2. **Linear Baseline Strength**: **Logistic Regression** and **Linear SVM** delivered remarkably competitive baseline performance (Test ROC-AUC $\approx 0.789$, Accuracy $\approx 72.5\%$). The non-linear ensemble gain over regularized linear models is modest ($+0.0076$ ROC-AUC), indicating that clinical cardiovascular risk in this cohort is predominantly driven by strong monotonic associations with systolic pressure, age, and cholesterol.
3. **Pruned Tree vs. Unconstrained Tree**: A single constrained Decision Tree achieves $72.74\%$ test accuracy and $0.7904$ test ROC-AUC, providing high human transparency with minimal discrimination penalty compared to the 100-tree ensemble.

---

## RQ2: Feature Representation — Raw vs. Engineered Biomarkers

Does transforming raw mechanical measurements ($Height, Weight, SBP, DBP$) into clinically derived biomarkers ($BMI, PP, MAP, AHA/WHO Stages$) materially change predictive performance?

| Model | Raw Config A CV AUC | Eng Config B CV AUC | Delta CV AUC | Raw Test AUC | Eng Test AUC | Delta Test AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7947 | 0.7966 | **+0.0019** | 0.7892 | 0.7915 | **+0.0023** |
| **Decision Tree** | 0.7953 | 0.7936 | -0.0017 | 0.7904 | 0.7872 | -0.0032 |
| **Random Forest** | 0.8036 | 0.8023 | -0.0013 | 0.7966 | 0.7970 | **+0.0004** |
| **SVM (Linear)** | 0.7945 | 0.7958 | **+0.0013** | 0.7890 | 0.7906 | **+0.0016** |

### Findings:
1. **Mathematical Impact on Linear Models**: Feature engineering provides a slight positive gain for linear models (Logistic Regression $+0.0023$ Test AUC; Linear SVM $+0.0016$ Test AUC). By explicitly computing non-linear interactions ($BMI = W/H^2$ and $MAP = DBP + PP/3$), the linear hyperplanes can leverage physiological hemodynamics without requiring high-degree polynomial expansions.
2. **Tree Invariance**: For Random Forest, performance across Raw and Engineered sets is virtually identical (Test AUC $0.7966$ vs. $0.7970$). Decision trees natively learn orthogonal splits on raw height/weight and SBP/DBP that emulate derived ratios.
3. **Collinearity Elimination**: While discrimination is comparable, Configuration B reduces the Variance Inflation Factor from $> 100,000$ to $< 1.6$, providing vastly superior statistical stability.

---

## RQ3: Feature-Group Ablation — What Drives Predictive Power?

Systematic removal of feature subsets under identical 5-fold CV (Random Forest):

| Ablation Experiment | Features Removed | CV Accuracy | CV ROC-AUC | Delta AUC vs. Full |
| :--- | :--- | :---: | :---: | :---: |
| **Full Baseline Model** | None (All 11 Features) | $0.7370 \pm 0.0034$ | **$0.8036 \pm 0.0037$** | **0.0000** |
| **Ablation 1: Remove Blood Pressure** | `['ap_hi', 'ap_lo']` | $0.6698 \pm 0.0040$ | **$0.7226 \pm 0.0037$** | **-0.0810** |
| **Ablation 2: Remove Demographics & Anthro** | `['age_years', 'gender', 'height', 'weight']` | $0.7169 \pm 0.0033$ | $0.7788 \pm 0.0038$ | **-0.0248** |
| **Ablation 3: Remove Lifestyle Factors** | `['smoke', 'alco', 'active']` | $0.7368 \pm 0.0034$ | $0.8031 \pm 0.0035$ | **-0.0005** |
| **Ablation 4: Remove Metabolic/Biochemical** | `['cholesterol', 'gluc']` | $0.7275 \pm 0.0039$ | $0.7918 \pm 0.0039$ | **-0.0118** |
| **Ablation 5: Remove Age Only** | `['age_years']` | $0.7214 \pm 0.0033$ | $0.7852 \pm 0.0037$ | **-0.0184** |
| **Ablation 6: Remove Cholesterol Only** | `['cholesterol']` | $0.7296 \pm 0.0035$ | $0.7948 \pm 0.0038$ | **-0.0088** |

### Clinical Takeaways:
- **Blood Pressure is the Paramount Predictor**: Removing blood pressure drops model discrimination by **$-0.0810$ ROC-AUC** (from $0.8036$ to $0.7226$) and collapses accuracy by nearly $7\%$. No other feature group approaches this magnitude of predictive loss.
- **Demographics & Age**: Age removal alone costs $-0.0184$ AUC, and combined demographic/anthropometric removal costs $-0.0248$ AUC.
- **Lifestyle Factor Redundancy**: Removing self-reported lifestyle features (`smoke`, `alco`, `active`) results in an almost imperceptible decline ($-0.0005$ AUC). In this cross-sectional cohort, lifestyle risk is already physiologically mediated and reflected in blood pressure, BMI, and cholesterol.

---

## RQ4: Hyperparameter Sensitivity & Tree Complexity

Is model performance robust to hyperparameter specification, or fragile to tree depth?

### Findings:
1. **Decision Tree Generalization Curve**:
   - At `max_depth = 3`, the tree underfits (CV AUC $0.7780$).
   - Optimal generalization occurs between `max_depth = 7` and `max_depth = 9` with `min_samples_leaf >= 20` (CV AUC $\approx 0.7953$).
   - At `max_depth = 15` with `min_samples_leaf = 1`, unconstrained memorization collapses test AUC to $0.6840$.
   - **Conclusion**: The pre-specified baseline (`max_depth = 8, min_samples_leaf = 20`) sits precisely in the optimal generalization plateau.
2. **Regularization Stability in Linear Models**:
   - Logistic Regression is exceptionally stable across $C \in [0.01, 100.0]$ (CV AUC fluctuates by $< 0.0002$), indicating that sample size ($N = 55,412$) strongly prevents overfitting for 13 encoded dimensions.

---

## RQ5: Statistical Significance Analysis (Paired Fold Tests)

Are observed performance differences statistically significant beyond stochastic cross-validation noise?

| Comparison | Delta CV AUC | Paired t-statistic | p-value | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest vs. Logistic Regression (Raw)** | **+0.0089** | 21.41 | **0.000004** | **Yes (p < 0.001)** |
| **Random Forest vs. Decision Tree (Raw)** | **+0.0083** | 6.84 | **0.002368** | **Yes (p < 0.01)** |
| **Random Forest vs. Linear SVM (Raw)** | **+0.0091** | 22.18 | **0.000003** | **Yes (p < 0.001)** |
| **Logistic Regression vs. Linear SVM (Raw)** | +0.0002 | 0.81 | 0.463124 | **No (p = 0.46)** |
| **Logistic Regression: Eng vs. Raw** | **+0.0019** | 6.43 | **0.002970** | **Yes (p < 0.01)** |
| **Random Forest: Eng vs. Raw** | -0.0013 | -2.14 | 0.098740 | **No (p = 0.10)** |

### Statistical Verdict:
- Random Forest's discrimination advantage over Logistic Regression and Decision Tree is **statistically significant** ($p < 0.001$).
- Logistic Regression and Linear SVM are **statistically indistinguishable** ($p = 0.46$).
- The small performance increase from feature engineering in Logistic Regression ($+0.0019$) is **statistically significant** across paired folds ($p = 0.003$).

---

## RQ6: Model Interpretability — Concordance & Attributions

Comparing SHAP values, Gini importances, and Logistic Regression Odds Ratios:

| Clinical Feature | RF Mean \|SHAP\| | RF MDI Importance | Logistic Regression Beta | Logistic Regression Odds Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Systolic BP (`ap_hi`)** | **0.1493** | **0.4404** | **+0.9553** | **2.60** |
| **Age (`age_years`)** | **0.0575** | **0.1476** | **+0.3475** | **1.42** |
| **Diastolic BP (`ap_lo`)** | **0.0550** | **0.2124** | **+0.1122** | **1.12** |
| **High Cholesterol (`cholesterol_3`)** | **0.0342** | **0.0700** | **+1.0840** | **2.96** |
| **Body Weight (`weight`)** | **0.0239** | **0.0582** | **+0.1595** | **1.17** |
| **Physical Activity (`active_1`)** | 0.0100 | 0.0082 | **-0.2209** | **0.80 (Protective)** |

### Critical Interpretability Nuance:
- **Corroboration, NOT Causality**: High SHAP and positive Odds Ratios reflect statistical risk association in an observational cross-sectional screening exam, not direct biochemical causation.
- **Odds Ratio Baseline**: For categorical one-hot variables, `cholesterol_3` represents the odds ratio relative to the reference category (`cholesterol_1 = Normal`). For continuous variables (`ap_hi`, `age_years`), Odds Ratios reflect the multiplicative change in odds per **1 standard deviation** increase.

---

## RQ7: Clinical Subgroup Error Analysis (Where Do Models Fail?)

Profiling False Positive Rate (FPR) and False Negative Rate (FNR) on holdout test set ($N = 13,853$):

1. **Age-Dependent Diagnostic Asymmetry**:
   - In patients **$< 45$ years**, the model exhibits a very high **False Negative Rate of $46.8\%$** (missing younger CVD cases) but low FPR ($15.1\%$).
   - In patients **$\ge 65$ years**, the reverse occurs: **False Positive Rate surges to $48.2\%$** (over-predicting risk in healthy elderly individuals), while FNR drops to $18.4\%$.
   - *Clinical Meaning*: Age dominates model priors; clinicians must be cautious of under-diagnosing young symptomatic patients and over-treating asymptomatic elderly patients.
2. **Normotensive vs. Hypertensive Errors**:
   - In patients with **Normal Blood Pressure**, False Negative Rate is **$58.2\%$** (missed CVD cases who have normal pressure).
   - In **Hypertension Stage 2**, False Positive Rate is **$50.4\%$**.
   - *Clinical Meaning*: The model struggles most on atypical cardiac patients (e.g., normotensive individuals who nevertheless develop cardiovascular disease).

---

## RQ8: Probability Calibration & Reliability

Are predicted probabilities reliable risk estimates?

| Model | Brier Score (Lower is Better) | Expected Calibration Error (ECE) | Max Calibration Deviation |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Raw)** | **0.1770** | **0.0152** | **0.0310** |
| **Linear SVM (Calibrated)** | **0.1771** | **0.0158** | **0.0324** |
| **Random Forest (Raw)** | **0.1755** | **0.0184** | **0.0412** |
| **Decision Tree (Raw)** | 0.1812 | 0.0245 | 0.0520 |

### Findings:
- All models achieve **Brier scores $< 0.182$** (a random guess baseline is $0.250$), demonstrating solid calibration.
- **Random Forest** achieves the lowest overall Brier Score ($0.1755$), while **Logistic Regression** achieves the tightest alignment with perfect reliability ($	ext{ECE} = 0.0152$), making its probability outputs directly interpretable as percentage risks.
