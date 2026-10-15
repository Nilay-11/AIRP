# Multicollinearity & Feature Architecture Report

**Research Paper**: *Comparative Evaluation and Interpretation of Machine Learning Models for Cardiovascular Disease Risk Prediction Using Clinical Features*  
**Cleaned Cohort Size**: 69,265 patients

---

## 1. Executive Summary & Clinical Recommendation

Derived clinical metrics ($BMI$, $Pulse\ Pressure$, $Mean\ Arterial\ Pressure$) are deterministic mathematical functions of raw physical measurements ($Height$, $Weight$, $Systolic\ BP$, $Diastolic\ BP$). 

This analysis evaluates the degree of **Variance Inflation Factor (VIF)** across different feature subset formulations to establish sound architectural rules for downstream machine learning:

1. **For Linear & Parametric Classifiers (Logistic Regression, ElasticNet, Linear SVM)**:
   - **Must use either Feature Set 1 (Raw Baseline) OR Feature Set 2 (Hemodynamics)**.
   - **Never combine raw and derived metrics simultaneously**; doing so inflates VIF to $> 100,000$ (near-infinite collinearity), destabilizing coefficient signs and inflating standard errors.
2. **For Tree-Based Ensembles (XGBoost, LightGBM, CatBoost, Random Forest)**:
   - Tree partitions can leverage Feature Set 2 or Set 3 because recursive splitting is invariant to monotonic transformations and collinearity.
   - However, for **SHAP Feature Importance & Attribution**, Feature Set 2 provides cleaner interpretability by preventing the dilution of Shapley values across redundant pairs (e.g., splitting credit between SBP and MAP).

---

## 2. Feature Set 1: Raw Clinical Baseline (Non-Redundant)
*Features: Age, Gender, Height, Weight, SBP (ap_hi), DBP (ap_lo), Cholesterol, Glucose, Smoking, Alcohol, Activity*

| Feature | R_Squared | VIF | Collinearity_Status |
| --- | --- | --- | --- |
| age_years | 0.0674 | 1.07 | Low (Safe) |
| gender | 0.3391 | 1.51 | Low (Safe) |
| height | 0.3486 | 1.54 | Low (Safe) |
| weight | 0.1953 | 1.24 | Low (Safe) |
| ap_hi | 0.5713 | 2.33 | Low (Safe) |
| ap_lo | 0.5551 | 2.25 | Low (Safe) |
| cholesterol | 0.2431 | 1.32 | Low (Safe) |
| gluc | 0.206 | 1.26 | Low (Safe) |
| smoke | 0.1985 | 1.25 | Low (Safe) |
| alco | 0.1222 | 1.14 | Low (Safe) |
| active | 0.0018 | 1.0 | Low (Safe) |

> **Conclusion**: All features exhibit low VIF ($< 2.5$). Highly safe and mathematically stable for all model architectures.

---

## 3. Feature Set 2: Hemodynamic / Biomarker Set (Clinically Expressive)
*Features: Age, Gender, BMI, Pulse Pressure (PP), Mean Arterial Pressure (MAP), Cholesterol, Glucose, Smoking, Alcohol, Activity*

| Feature | R_Squared | VIF | Collinearity_Status |
| --- | --- | --- | --- |
| age_years | 0.062 | 1.07 | Low (Safe) |
| gender | 0.1386 | 1.16 | Low (Safe) |
| bmi | 0.1135 | 1.13 | Low (Safe) |
| pulse_pressure | 0.3217 | 1.47 | Low (Safe) |
| mean_arterial_pressure | 0.3638 | 1.57 | Low (Safe) |
| cholesterol | 0.2427 | 1.32 | Low (Safe) |
| gluc | 0.2057 | 1.26 | Low (Safe) |
| smoke | 0.1982 | 1.25 | Low (Safe) |
| alco | 0.1222 | 1.14 | Low (Safe) |
| active | 0.0015 | 1.0 | Low (Safe) |

> **Conclusion**: Highly acceptable collinearity profile ($VIF < 3.5$). Replaces mechanical measurements ($Height, Weight, SBP, DBP$) with systemic cardiovascular biomarkers ($BMI, PP, MAP$).

---

## 4. Feature Set 3: All Combined (Collinearity Stress Test)
*Demonstration of what occurs when raw and derived metrics are blindly trained together:*

| Feature | R_Squared | VIF | Collinearity_Status |
| --- | --- | --- | --- |
| age_years | 0.0681 | 1.07 | Low (Safe) |
| gender | 0.3411 | 1.52 | Low (Safe) |
| height | 0.9685 | 31.71 | Severe (Harmful for Linear Models) |
| weight | 0.9916 | 118.85 | Severe (Harmful for Linear Models) |
| bmi | 0.991 | 110.62 | Severe (Harmful for Linear Models) |
| ap_hi | 1.0 | INF (>100,000) | Severe (Harmful for Linear Models) |
| ap_lo | 1.0 | INF (>100,000) | Severe (Harmful for Linear Models) |
| pulse_pressure | 1.0 | INF (>100,000) | Severe (Harmful for Linear Models) |
| mean_arterial_pressure | 1.0 | INF (>100,000) | Severe (Harmful for Linear Models) |
| cholesterol | 0.2452 | 1.32 | Low (Safe) |
| gluc | 0.2061 | 1.26 | Low (Safe) |
| smoke | 0.1987 | 1.25 | Low (Safe) |
| alco | 0.1224 | 1.14 | Low (Safe) |
| active | 0.0019 | 1.0 | Low (Safe) |

> **Key Finding**: Because $PP = SBP - DBP$ and $MAP = DBP + \frac13PP$, $R^2 \approx 1.0$ and **VIF exceeds 10,000 to infinity**. Linear models trained on this set will suffer from singular matrix problems and erratic weight estimates.

---

## 5. Pearson Correlation Matrix (Target & Key Covariates)

```
                        age_years  gender  height  weight    bmi  ap_hi  ap_lo  pulse_pressure  mean_arterial_pressure  cholesterol   gluc  smoke   alco  active  cardio
age_years                   1.000  -0.022  -0.087   0.055  0.105  0.207  0.154           0.172                   0.192        0.154  0.099 -0.048 -0.030  -0.010   0.238
gender                     -0.022   1.000   0.525   0.164 -0.112  0.064  0.069           0.036                   0.071       -0.035 -0.020  0.338  0.171   0.007   0.008
height                     -0.087   0.525   1.000   0.322 -0.193  0.017  0.034          -0.003                   0.028       -0.056 -0.021  0.197  0.098  -0.009  -0.013
weight                      0.055   0.164   0.322   1.000  0.862  0.272  0.256           0.182                   0.282        0.142  0.105  0.071  0.069  -0.017   0.182
bmi                         0.105  -0.112  -0.193   0.862  1.000  0.273  0.248           0.191                   0.278        0.176  0.118 -0.032  0.018  -0.014   0.195
ap_hi                       0.207   0.064   0.017   0.272  0.273  1.000  0.743           0.835                   0.924        0.196  0.093  0.028  0.036   0.001   0.432
ap_lo                       0.154   0.069   0.034   0.256  0.248  0.743  1.000           0.251                   0.942        0.162  0.073  0.027  0.038   0.002   0.348
pulse_pressure              0.172   0.036  -0.003   0.182  0.191  0.835  0.251           1.000                   0.561        0.150  0.075  0.019  0.021   0.000   0.339
mean_arterial_pressure      0.192   0.071   0.028   0.282  0.278  0.924  0.942           0.561                   1.000        0.191  0.088  0.029  0.040   0.002   0.415
cholesterol                 0.154  -0.035  -0.056   0.142  0.176  0.196  0.162           0.150                   0.191        1.000  0.451  0.011  0.036   0.010   0.221
gluc                        0.099  -0.020  -0.021   0.105  0.118  0.093  0.073           0.075                   0.088        0.451  1.000 -0.005  0.011  -0.007   0.089
smoke                      -0.048   0.338   0.197   0.071 -0.032  0.028  0.027           0.019                   0.029        0.011 -0.005  1.000  0.340   0.025  -0.015
alco                       -0.030   0.171   0.098   0.069  0.018  0.036  0.038           0.021                   0.040        0.036  0.011  0.340  1.000   0.025  -0.007
active                     -0.010   0.007  -0.009  -0.017 -0.014  0.001  0.002           0.000                   0.002        0.010 -0.007  0.025  0.025   1.000  -0.036
cardio                      0.238   0.008  -0.013   0.182  0.195  0.432  0.348           0.339                   0.415        0.221  0.089 -0.015 -0.007  -0.036   1.000
```
