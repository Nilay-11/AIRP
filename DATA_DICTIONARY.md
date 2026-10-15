# Clinical Data Dictionary: Cardiovascular Disease Risk Cohort

**Project**: *Comparative Evaluation and Interpretation of Machine Learning Models for Cardiovascular Disease Risk Prediction Using Clinical Features*  
**Dataset File**: `cardio_cleaned.csv` (also available in `data/cardio_cleaned.csv`)  
**Total Records**: 69,265 patients  
**Total Features**: 20 variables  

---

## 1. Feature Specifications & Metadata

| Column Name | Physical Unit | Variable Type | Source / Derivation | Description & Value Codes |
| :--- | :--- | :--- | :--- | :--- |
| **`id`** | None | Identifier | Original | Unique integer patient examination identifier (excluded from ML modeling). |
| **`age`** | Days | Numerical (Discrete) | Original | Age recorded as total completed days from birth. |
| **`gender`** | None | Categorical (Binary) | Original | **Demographic Gender (Kaggle Standard Convention)**:<br>• `1` = **Female / Women**<br>• `2` = **Male / Men** |
| **`height`** | cm | Numerical (Continuous) | Original (Cleaned) | Patient body height in centimeters. Physiological bounds: $[140, 200]\text{ cm}$. |
| **`weight`** | kg | Numerical (Continuous) | Original (Cleaned) | Patient body weight in kilograms. Physiological bounds: $[40, 160]\text{ kg}$. |
| **`ap_hi_raw`** | mmHg | Numerical | Original (Raw) | Untransformed systolic blood pressure reading as recorded in the raw data source. |
| **`ap_lo_raw`** | mmHg | Numerical | Original (Raw) | Untransformed diastolic blood pressure reading as recorded in the raw data source. |
| **`ap_hi`** | mmHg | Numerical (Continuous) | Verified / Remediated | Systolic Blood Pressure (SBP). Normal clinical range: $[70, 240]\text{ mmHg}$. |
| **`ap_lo`** | mmHg | Numerical (Continuous) | Verified / Remediated | Diastolic Blood Pressure (DBP). Normal clinical range: $[40, 140]\text{ mmHg}$. |
| **`bp_was_corrected`**| None | Boolean (`bool`) | Quality Audit Flag | Audit indicator showing whether blood pressure was modified:<br>• `False` = Pristine recorded measurement<br>• `True` = Remediated (sign flip, 10x scale typo, or transposed pair) |
| **`bp_correction_type`**| None | Categorical (String) | Quality Audit Flag | Exact nature of blood pressure remediation:<br>• `none` = Unmodified<br>• `sign_flipped` = Negative sign removed<br>• `scale_fixed_10x` = Trailing zero removed (e.g., $1200 \rightarrow 120$ mmHg)<br>• `swapped` = Transposed SBP/DBP inverted back<br>• `scale_fixed_10x_and_swapped` = Both fixes applied |
| **`cholesterol`** | Ordinal Level | Categorical (Ordinal) | Original | Total serum cholesterol level:<br>• `1` = Normal ($< 200\text{ mg/dL}$)<br>• `2` = Above Normal ($200 - 239\text{ mg/dL}$)<br>• `3` = Well Above Normal ($\ge 240\text{ mg/dL}$) |
| **`gluc`** | Ordinal Level | Categorical (Ordinal) | Original | Fasting blood glucose level:<br>• `1` = Normal ($< 100\text{ mg/dL}$)<br>• `2` = Above Normal ($100 - 125\text{ mg/dL}$, Pre-diabetes)<br>• `3` = Well Above Normal ($\ge 126\text{ mg/dL}$, Diabetes) |
| **`smoke`** | Binary | Categorical (Binary) | Original | Tobacco smoking status:<br>• `0` = Non-smoker<br>• `1` = Active tobacco smoker |
| **`alco`** | Binary | Categorical (Binary) | Original | Alcohol consumption status:<br>• `0` = Non-drinker<br>• `1` = Active alcohol consumer |
| **`active`** | Binary | Categorical (Binary) | Original | Regular physical activity engagement:<br>• `0` = Inactive / Sedentary lifestyle<br>• `1` = Physically active |
| **`cardio`** | Binary | **Target (Outcome)** | Original | **Cardiovascular Disease Diagnosis (CVD Outcome)**:<br>• `0` = Absence of diagnosed cardiovascular disease<br>• `1` = Presence of diagnosed cardiovascular disease |
| **`age_years`** | Years | Numerical (Continuous) | Calculated | Standardized decimal age: $\frac{\text{age}}{365.25}$. Accounts for leap years over lifespan. |
| **`bmi`** | $\text{kg/m}^2$ | Numerical (Continuous) | Calculated | Body Mass Index: $\frac{\text{weight}}{(\text{height} / 100)^2}$. Physiological bounds: $[16.0, 50.0]\text{ kg/m}^2$. |
| **`pulse_pressure`** | mmHg | Numerical (Continuous) | Calculated | Pulse Pressure: $ap\_hi - ap\_lo$. Established clinical marker of arterial stiffness and vascular aging. |
| **`mean_arterial_pressure`**| mmHg | Numerical (Continuous) | Calculated | Mean Arterial Pressure (MAP): $ap\_lo + \frac{1}{3}(ap\_hi - ap\_lo)$. Quantifies systemic tissue perfusion pressure. |
| **`bp_category`** | Stage Index (0–4) | Categorical (Ordinal) | Calculated | Integer stage index according to 2017 ACC/AHA High Blood Pressure Guidelines (see Section 2). |
| **`bp_stage_label`** | Text Label | Categorical (Nominal) | Calculated | Explicit descriptive string corresponding to `bp_category` (see Section 2). |
| **`bmi_category`** | Class Index (0–4) | Categorical (Ordinal) | Calculated | Integer class index according to World Health Organization (WHO) BMI standards (see Section 3). |
| **`bmi_class_label`**| Text Label | Categorical (Nominal) | Calculated | Explicit descriptive string corresponding to `bmi_category` (see Section 3). |

---

## 2. Blood Pressure Staging Legend (ACC/AHA 2017 Guidelines)

Classification conforms to the *2017 ACC/AHA/AAPA/ABC/ACPM/AGS/APhA/ASH/ASPC/NMA/PCNA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults*:

| `bp_category` | `bp_stage_label` | Systolic Blood Pressure (`ap_hi`) | Operator | Diastolic Blood Pressure (`ap_lo`) |
| :---: | :--- | :--- | :---: | :--- |
| **0** | **Normal** | $< 120\text{ mmHg}$ | **and** | $< 80\text{ mmHg}$ |
| **1** | **Elevated** | $120 - 129\text{ mmHg}$ | **and** | $< 80\text{ mmHg}$ |
| **2** | **Hypertension Stage 1** | $130 - 139\text{ mmHg}$ | **or** | $80 - 89\text{ mmHg}$ |
| **3** | **Hypertension Stage 2** | $\ge 140\text{ mmHg}$ | **or** | $\ge 90\text{ mmHg}$ |
| **4** | **Hypertensive Crisis** | $> 180\text{ mmHg}$ | **and/or**| $> 120\text{ mmHg}$ |

*Clinical Rule*: If systolic and diastolic measurements fall into different stages, the higher risk stage takes precedence.

---

## 3. Body Mass Index Classification Legend (WHO Standard)

Classification conforms to the *World Health Organization (WHO) International BMI Classifications*:

| `bmi_category` | `bmi_class_label` | BMI Range ($\text{kg/m}^2$) | Clinical Description |
| :---: | :--- | :--- | :--- |
| **0** | **Underweight** | $< 18.5$ | Increased risk of nutritional deficiency and osteoporosis. |
| **1** | **Normal weight** | $18.5 - 24.9$ | Optimal cardiovascular risk baseline. |
| **2** | **Overweight** | $25.0 - 29.9$ | Pre-obesity; moderate elevation of cardiovascular risk. |
| **3** | **Obesity Class I/II** | $30.0 - 39.9$ | Moderate to severe obesity; established cardiovascular risk factor. |
| **4** | **Severe Obesity Class III** | $\ge 40.0$ | Morbid obesity; very high risk of cardiometabolic mortality. |

---

## 4. Modeling Guidelines & Feature Collinearity Warning

When designing machine learning models:
1. **Linear & Parametric Models** (e.g., Logistic Regression, SVM with linear kernel, Ridge/Lasso):
   - **Do NOT** include `height`, `weight`, and `bmi` together. Choose either `['height', 'weight']` or `['bmi']`.
   - **Do NOT** include `ap_hi`, `ap_lo`, `pulse_pressure`, and `mean_arterial_pressure` together. Choose either raw BP (`['ap_hi', 'ap_lo']`) or hemodynamics (`['pulse_pressure', 'mean_arterial_pressure']`).
   - Combining them induces high multicollinearity ($VIF > 10$), inflating coefficient variance.
2. **Tree-Based Ensembles** (e.g., Random Forest, XGBoost, LightGBM, CatBoost):
   - Non-parametric tree models handle correlated features gracefully without mathematical breakdown.
   - However, for cleaner SHAP interpretability, non-redundant feature sets prevent dilution of feature attribution across correlated pairs.
