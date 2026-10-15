# Data Cleaning, Clinical Quality Assurance & Audit Trail Report

**Research Paper**: *Comparative Evaluation and Interpretation of Machine Learning Models for Cardiovascular Disease Risk Prediction Using Clinical Features*  
**Source Dataset**: Kaggle Cardiovascular Disease Database (`rp AI healthcare.csv`)

---

## 1. Summary of Data Hygiene & Quality Control

| Metric | Value |
| :--- | :--- |
| **Initial Raw Records** | 70,000 |
| **Duplicate Records Removed** | 24 |
| **Blood Pressure Excluded (Implausible SBP/DBP or Inverted)** | 323 |
| **Implausible Height Excluded (< 140 cm or > 200 cm)** | 151 |
| **Implausible Weight Excluded (< 40 kg or > 160 kg)** | 78 |
| **Extreme BMI Excluded (< 16.0 or > 50.0 kg/m²)** | 159 |
| **Total Cleaned Cohort Retained** | **69,265** |
| **Cohort Retention Rate** | **98.95%** |

---

## 2. Clinical Audit Trail (Blood Pressure Traceability)

To guarantee clinical reproducibility and eliminate silent in-place data modifications, all original values are retained in `ap_hi_raw` and `ap_lo_raw`, and every alteration is flagged:

| Correction Type (`bp_correction_type`) | Record Count |
| :--- | :--- |
| `none` | 68,961 |
| `scale_fixed_10x` | 908 |
| `swapped_and_swapped` | 98 |
| `sign_flipped` | 8 |
| `scale_fixed_10x_and_swapped` | 1 |
| **Total Records with Any BP Remediation (`bp_was_corrected = True`)** | **1,015** |

> **Audit Transparency**: Downstream models can either use the verified remediated values or perform a sensitivity analysis by excluding records where `bp_was_corrected == True`.

---

## 3. Target Variable Balance (`cardio`)

| Class | Count | Percentage |
| :--- | :--- | :--- |
| **0 (No Cardiovascular Disease)** | 34,668 | 50.05% |
| **1 (Cardiovascular Disease Diagnosed)** | 34,597 | 49.95% |

---

## 4. Derived Categorical Legends

### ACC/AHA 2017 Blood Pressure Categories (`bp_category` & `bp_stage_label`)
- `0` = **Normal**: SBP < 120 mmHg and DBP < 80 mmHg
- `1` = **Elevated**: SBP 120–129 mmHg and DBP < 80 mmHg
- `2` = **Hypertension Stage 1**: SBP 130–139 mmHg or DBP 80–89 mmHg
- `3` = **Hypertension Stage 2**: SBP $\ge 140$ mmHg or DBP $\ge 90$ mmHg
- `4` = **Hypertensive Crisis**: SBP > 180 mmHg and/or DBP > 120 mmHg

### WHO BMI Classification (`bmi_category` & `bmi_class_label`)
- `0` = **Underweight**: BMI < 18.5 kg/m²
- `1` = **Normal weight**: BMI 18.5–24.9 kg/m²
- `2` = **Overweight**: BMI 25.0–29.9 kg/m²
- `3` = **Obesity Class I/II**: BMI 30.0–39.9 kg/m²
- `4` = **Severe Obesity Class III**: BMI $\ge 40.0$ kg/m²

### Demographic Variables
- `gender`: Preserved original Kaggle convention (`1` = Female / Women, `2` = Male / Men).

---

## 5. Descriptive Statistics of Cleaned Cohort

```
       age_years    gender    height    weight       bmi     ap_hi     ap_lo  pulse_pressure  mean_arterial_pressure  cholesterol      gluc     smoke      alco   active   cardio
count   69265.00  69265.00  69265.00  69265.00  69265.00  69265.00  69265.00        69265.00                69265.00     69265.00  69265.00  69265.00  69265.00  69265.0  69265.0
mean       53.30      1.35    164.47     74.10     27.42    127.05     81.57           45.47                   96.73         1.37      1.23      0.09      0.05      0.8      0.5
std         6.75      0.48      7.80     13.94      5.04     16.98      9.66           11.75                   11.30         0.68      0.57      0.28      0.23      0.4      0.5
min        29.56      1.00    140.00     40.00     16.00     70.00     40.00            5.00                   50.00         1.00      1.00      0.00      0.00      0.0      0.0
25%        48.36      1.00    159.00     65.00     23.88    120.00     80.00           40.00                   93.33         1.00      1.00      0.00      0.00      1.0      0.0
50%        53.94      1.00    165.00     72.00     26.37    120.00     80.00           40.00                   93.33         1.00      1.00      0.00      0.00      1.0      0.0
75%        58.39      2.00    170.00     82.00     30.12    140.00     90.00           50.00                  103.33         2.00      1.00      0.00      0.00      1.0      1.0
max        64.92      2.00    200.00    156.00     50.00    240.00    140.00          140.00                  173.33         3.00      3.00      1.00      1.00      1.0      1.0
```
