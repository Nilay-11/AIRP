"""
Cross-Verification of Refined Dataset (cardio_cleaned.csv)
Verifies:
1. Formulas for age_years, bmi, pulse_pressure, MAP.
2. Label consistency: bp_stage_label and bmi_class_label vs. integer categories.
3. Full audit trail consistency: bp_was_corrected matches modified rows exactly.
4. Anthropometric outlier fences: all records satisfy strict physiological bounds.
"""

import pandas as pd
import numpy as np

cleaned_path = r"c:\Users\nilay\OneDrive\Desktop\AI RP\cardio_cleaned.csv"
df = pd.read_csv(cleaned_path)

print(f"[INFO] Loaded {len(df):,} records from {cleaned_path}\n")

# 1. Formulas Verification
expected_age = np.round(df["age"] / 365.25, 2)
assert np.abs(df["age_years"] - expected_age).max() == 0.0, "Age mismatch!"
print("[PASSED] Age conversion: 100% exact across all rows.")

expected_bmi = np.round(df["weight"] / ((df["height"] / 100.0) ** 2), 2)
assert np.abs(df["bmi"] - expected_bmi).max() == 0.0, "BMI mismatch!"
print("[PASSED] BMI calculation: 100% exact across all rows.")

expected_pp = df["ap_hi"] - df["ap_lo"]
assert np.abs(df["pulse_pressure"] - expected_pp).max() == 0.0, "Pulse pressure mismatch!"
print("[PASSED] Pulse pressure: 100% exact across all rows.")

expected_map = np.round(df["ap_lo"] + (df["pulse_pressure"] / 3.0), 2)
assert np.abs(df["mean_arterial_pressure"] - expected_map).max() == 0.0, "MAP mismatch!"
print("[PASSED] Mean Arterial Pressure: 100% exact across all rows.")

# 2. Label Consistency Check
bp_label_map = {
    0: "Normal",
    1: "Elevated",
    2: "Hypertension Stage 1",
    3: "Hypertension Stage 2",
    4: "Hypertensive Crisis"
}
expected_bp_labels = df["bp_category"].map(bp_label_map)
assert (df["bp_stage_label"] != expected_bp_labels).sum() == 0, "BP stage label mismatch!"
print("[PASSED] BP Stage Labels match integer categories 100%.")

bmi_label_map = {
    0: "Underweight",
    1: "Normal weight",
    2: "Overweight",
    3: "Obesity Class I/II",
    4: "Severe Obesity Class III"
}
expected_bmi_labels = df["bmi_category"].map(bmi_label_map)
assert (df["bmi_class_label"] != expected_bmi_labels).sum() == 0, "BMI class label mismatch!"
print("[PASSED] BMI Class Labels match integer categories 100%.")

# 3. Audit Trail Check
# A row is altered if ap_hi != ap_hi_raw or ap_lo != ap_lo_raw
physically_altered = (df["ap_hi"] != df["ap_hi_raw"]) | (df["ap_lo"] != df["ap_lo_raw"])
# Flagged in audit
audit_flagged = df["bp_was_corrected"]
discrepancies = (physically_altered != audit_flagged).sum()
assert discrepancies == 0, f"Audit flag discrepancy found: {discrepancies} rows!"
print(f"[PASSED] Audit Trail: Exactly {audit_flagged.sum():,} altered rows correctly flagged. 0 discrepancies.")

# 4. Anthropometric Outlier Boundaries Check
assert df["height"].min() >= 140.0 and df["height"].max() <= 200.0, "Height outlier found!"
assert df["weight"].min() >= 40.0 and df["weight"].max() <= 160.0, "Weight outlier found!"
assert df["bmi"].min() >= 16.0 and df["bmi"].max() <= 50.0, "BMI outlier found!"
print("[PASSED] Anthropometric bounds: Height in [140, 200] cm, Weight in [40, 160] kg, BMI in [16.0, 50.0] kg/m^2 satisfied 100%.")

# 5. Gender check
assert set(df["gender"].unique()) == {1, 2}, "Gender values unexpected!"
print("[PASSED] Gender column preserved untouched with original {1, 2} encoding.")

print("\n" + "="*50)
print("ALL 5 AUDIT AND INTEGRITY CHECKS PASSED WITH ZERO ERRORS")
print("="*50)
