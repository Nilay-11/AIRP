"""
Data Cleaning and Clinical Feature Engineering Pipeline (Refined)
Project: Comparative Evaluation and Interpretation of Machine Learning Models 
         for Cardiovascular Disease Risk Prediction Using Clinical Features
Dataset: Kaggle Cardiovascular Disease Dataset (rp AI healthcare.csv)

Revisions:
1. Outlier consistency: Enforced strict anthropometric bounds (Height in [140, 200] cm, 
   Weight in [40, 160] kg, and BMI in [16.0, 50.0] kg/m^2) eliminating data entry errors
   (e.g., 35 kg at 168 cm, 125 cm height at age 50-60).
2. Audit Trail: Added 'ap_hi_raw', 'ap_lo_raw', 'bp_was_corrected' (boolean), and 
   'bp_correction_type' (string) to ensure complete clinical transparency.
3. Categorical Legends: Added 'bp_stage_label' and 'bmi_class_label' textual columns.
4. Gender: Preserved original 'gender' column (1=Female, 2=Male) with explicit documentation.
"""

import os
import sys
import numpy as np
import pandas as pd


def load_raw_dataset(input_path: str) -> pd.DataFrame:
    """Load the raw dataset with semicolon delimiter."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
    
    print(f"[INFO] Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path, sep=';')
    print(f"[INFO] Loaded shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_and_engineer_features(
    df: pd.DataFrame,
    min_height: float = 140.0,
    max_height: float = 200.0,
    min_weight: float = 40.0,
    max_weight: float = 160.0,
    min_bmi: float = 16.0,
    max_bmi: float = 50.0,
    min_systolic: float = 70.0,
    max_systolic: float = 240.0,
    min_diastolic: float = 40.0,
    max_diastolic: float = 140.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean clinical anomalies with full audit trail and engineer validated clinical features.
    
    Returns:
        cleaned_df: Processed DataFrame
        stats: Dictionary containing before/after numbers and audit breakdown
    """
    initial_count = len(df)
    stats = {"initial_records": initial_count, "filter_breakdown": {}, "audit_breakdown": {}}
    
    # 1. Missing Values & Duplicate check
    missing_count = df.isnull().sum().to_dict()
    stats["missing_values"] = missing_count
    
    # Check duplicate rows excluding 'id'
    dup_mask = df.duplicated(subset=[col for col in df.columns if col != 'id'])
    num_duplicates = int(dup_mask.sum())
    stats["duplicate_records"] = num_duplicates
    df_clean = df[~dup_mask].copy()
    stats["filter_breakdown"]["exact_duplicates_removed"] = num_duplicates
    
    # 2. Age conversion (from days to continuous years using Julian standard 365.25)
    df_clean["age_years"] = np.round(df_clean["age"] / 365.25, 2)
    
    # 3. AUDIT TRAIL PRESERVATION FOR BLOOD PRESSURE
    df_clean["ap_hi_raw"] = df_clean["ap_hi"]
    df_clean["ap_lo_raw"] = df_clean["ap_lo"]
    df_clean["bp_was_corrected"] = False
    df_clean["bp_correction_type"] = "none"
    
    # Cast BP working columns to float
    df_clean["ap_hi"] = df_clean["ap_hi"].astype(float)
    df_clean["ap_lo"] = df_clean["ap_lo"].astype(float)
    
    # Step 3a: Track & flip negative readings (data collection sign artifacts)
    neg_mask = (df_clean["ap_hi"] < 0) | (df_clean["ap_lo"] < 0)
    df_clean.loc[neg_mask, "bp_was_corrected"] = True
    df_clean.loc[neg_mask, "bp_correction_type"] = "sign_flipped"
    df_clean["ap_hi"] = df_clean["ap_hi"].abs()
    df_clean["ap_lo"] = df_clean["ap_lo"].abs()
    
    # Step 3b: Typo correction for trailing zero scaling (e.g. 1100 -> 110, 1400 -> 140)
    mask_hi_typo = (df_clean["ap_hi"] >= 1000) & (df_clean["ap_hi"] <= 2400)
    mask_lo_typo = (df_clean["ap_lo"] >= 1000) & (df_clean["ap_lo"] <= 1400)
    scale_mask = mask_hi_typo | mask_lo_typo
    
    df_clean.loc[mask_hi_typo, "ap_hi"] = (df_clean.loc[mask_hi_typo, "ap_hi"] / 10.0).round()
    df_clean.loc[mask_lo_typo, "ap_lo"] = (df_clean.loc[mask_lo_typo, "ap_lo"] / 10.0).round()
    
    # Update audit flag for scale fix
    df_clean.loc[scale_mask & (~neg_mask), "bp_correction_type"] = "scale_fixed_10x"
    df_clean.loc[scale_mask & neg_mask, "bp_correction_type"] = "sign_flipped_and_scale_fixed"
    df_clean.loc[scale_mask, "bp_was_corrected"] = True
    
    # Step 3c: Swapped systolic/diastolic (Inverted records where ap_lo > ap_hi within plausible bounds)
    swap_mask = (
        (df_clean["ap_lo"] > df_clean["ap_hi"]) &
        (df_clean["ap_lo"] >= min_systolic) & (df_clean["ap_lo"] <= max_systolic) &
        (df_clean["ap_hi"] >= min_diastolic) & (df_clean["ap_hi"] <= max_diastolic)
    )
    # Apply swap
    df_clean.loc[swap_mask, ["ap_hi", "ap_lo"]] = df_clean.loc[swap_mask, ["ap_lo", "ap_hi"]].values
    
    # Update audit flag for swapped records
    df_clean.loc[swap_mask & (df_clean["bp_correction_type"] == "none"), "bp_correction_type"] = "swapped"
    df_clean.loc[swap_mask & (df_clean["bp_correction_type"] != "none"), "bp_correction_type"] += "_and_swapped"
    df_clean.loc[swap_mask, "bp_was_corrected"] = True
    
    # Audit summary counts
    stats["audit_breakdown"]["total_bp_corrected"] = int(df_clean["bp_was_corrected"].sum())
    stats["audit_breakdown"]["correction_type_counts"] = df_clean["bp_correction_type"].value_counts().to_dict()
    
    # 4. Strict Physiological Blood Pressure Filtering
    bp_valid_mask = (
        (df_clean["ap_hi"] >= min_systolic) & (df_clean["ap_hi"] <= max_systolic) &
        (df_clean["ap_lo"] >= min_diastolic) & (df_clean["ap_lo"] <= max_diastolic) &
        (df_clean["ap_hi"] > df_clean["ap_lo"])  # Systolic must strictly exceed diastolic
    )
    dropped_bp = int((~bp_valid_mask).sum())
    stats["filter_breakdown"]["implausible_blood_pressure_removed"] = dropped_bp
    df_clean = df_clean[bp_valid_mask].copy()
    
    # 5. Anthropometric Outlier Filtering (Consistent with BP Treatment)
    # Height filtering: [140, 200] cm
    height_mask = (df_clean["height"] >= min_height) & (df_clean["height"] <= max_height)
    dropped_height = int((~height_mask).sum())
    stats["filter_breakdown"]["implausible_height_removed"] = dropped_height
    df_clean = df_clean[height_mask].copy()
    
    # Weight filtering: [40, 160] kg
    weight_mask = (df_clean["weight"] >= min_weight) & (df_clean["weight"] <= max_weight)
    dropped_weight = int((~weight_mask).sum())
    stats["filter_breakdown"]["implausible_weight_removed"] = dropped_weight
    df_clean = df_clean[weight_mask].copy()
    
    # 6. Feature Engineering: Clinical Biomarkers
    # Body Mass Index (BMI = kg / m^2)
    df_clean["bmi"] = np.round(df_clean["weight"] / ((df_clean["height"] / 100.0) ** 2), 2)
    
    # BMI Physiological Filter: [16.0, 50.0] kg/m^2 (WHO severe starvation to extreme obesity)
    bmi_mask = (df_clean["bmi"] >= min_bmi) & (df_clean["bmi"] <= max_bmi)
    dropped_bmi = int((~bmi_mask).sum())
    stats["filter_breakdown"]["extreme_bmi_removed"] = dropped_bmi
    df_clean = df_clean[bmi_mask].copy()
    
    # Pulse Pressure (PP = SBP - DBP)
    df_clean["pulse_pressure"] = df_clean["ap_hi"] - df_clean["ap_lo"]
    
    # Mean Arterial Pressure (MAP = DBP + 1/3 * (SBP - DBP))
    df_clean["mean_arterial_pressure"] = np.round(
        df_clean["ap_lo"] + (df_clean["pulse_pressure"] / 3.0), 2
    )
    
    # ACC/AHA 2017 Blood Pressure Stage & Text Label:
    # 0 = Normal, 1 = Elevated, 2 = Stage 1, 3 = Stage 2, 4 = Crisis
    def categorize_bp(row):
        sbp = row["ap_hi"]
        dbp = row["ap_lo"]
        if sbp > 180 or dbp > 120:
            return 4, "Hypertensive Crisis"
        elif sbp >= 140 or dbp >= 90:
            return 3, "Hypertension Stage 2"
        elif (sbp >= 130 and sbp < 140) or (dbp >= 80 and dbp < 90):
            return 2, "Hypertension Stage 1"
        elif (sbp >= 120 and sbp < 130) and dbp < 80:
            return 1, "Elevated"
        else:
            return 0, "Normal"

    bp_results = df_clean.apply(categorize_bp, axis=1)
    df_clean["bp_category"] = [res[0] for res in bp_results]
    df_clean["bp_stage_label"] = [res[1] for res in bp_results]
    
    # WHO BMI Classification & Text Label:
    # 0 = Underweight (<18.5), 1 = Normal (18.5-24.9), 2 = Overweight (25-29.9), 
    # 3 = Obese (30-39.9), 4 = Severely Obese (>=40)
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 0, "Underweight"
        elif bmi < 25.0:
            return 1, "Normal weight"
        elif bmi < 30.0:
            return 2, "Overweight"
        elif bmi < 40.0:
            return 3, "Obesity Class I/II"
        else:
            return 4, "Severe Obesity Class III"

    bmi_results = df_clean["bmi"].apply(categorize_bmi)
    df_clean["bmi_category"] = [res[0] for res in bmi_results]
    df_clean["bmi_class_label"] = [res[1] for res in bmi_results]
    
    final_count = len(df_clean)
    retention_rate = np.round((final_count / initial_count) * 100, 2)
    stats["final_records"] = final_count
    stats["total_removed"] = initial_count - final_count
    stats["retention_rate_pct"] = retention_rate
    
    # Target distribution
    stats["target_distribution"] = df_clean["cardio"].value_counts().to_dict()
    stats["target_percentage"] = (df_clean["cardio"].value_counts(normalize=True) * 100).round(2).to_dict()
    
    return df_clean, stats


def generate_markdown_report(stats: dict, summary_df: pd.DataFrame, output_path: str):
    """Write comprehensive cleaning, audit trail & quality report."""
    audit_table = "\n".join([
        f"| `{k}` | {v:,} |" for k, v in stats["audit_breakdown"]["correction_type_counts"].items()
    ])
    
    report = f"""# Data Cleaning, Clinical Quality Assurance & Audit Trail Report

**Research Paper**: *Comparative Evaluation and Interpretation of Machine Learning Models for Cardiovascular Disease Risk Prediction Using Clinical Features*  
**Source Dataset**: Kaggle Cardiovascular Disease Database (`rp AI healthcare.csv`)

---

## 1. Summary of Data Hygiene & Quality Control

| Metric | Value |
| :--- | :--- |
| **Initial Raw Records** | {stats['initial_records']:,} |
| **Duplicate Records Removed** | {stats['filter_breakdown']['exact_duplicates_removed']:,} |
| **Blood Pressure Excluded (Implausible SBP/DBP or Inverted)** | {stats['filter_breakdown']['implausible_blood_pressure_removed']:,} |
| **Implausible Height Excluded (< 140 cm or > 200 cm)** | {stats['filter_breakdown']['implausible_height_removed']:,} |
| **Implausible Weight Excluded (< 40 kg or > 160 kg)** | {stats['filter_breakdown']['implausible_weight_removed']:,} |
| **Extreme BMI Excluded (< 16.0 or > 50.0 kg/m²)** | {stats['filter_breakdown']['extreme_bmi_removed']:,} |
| **Total Cleaned Cohort Retained** | **{stats['final_records']:,}** |
| **Cohort Retention Rate** | **{stats['retention_rate_pct']}%** |

---

## 2. Clinical Audit Trail (Blood Pressure Traceability)

To guarantee clinical reproducibility and eliminate silent in-place data modifications, all original values are retained in `ap_hi_raw` and `ap_lo_raw`, and every alteration is flagged:

| Correction Type (`bp_correction_type`) | Record Count |
| :--- | :--- |
{audit_table}
| **Total Records with Any BP Remediation (`bp_was_corrected = True`)** | **{stats['audit_breakdown']['total_bp_corrected']:,}** |

> **Audit Transparency**: Downstream models can either use the verified remediated values or perform a sensitivity analysis by excluding records where `bp_was_corrected == True`.

---

## 3. Target Variable Balance (`cardio`)

| Class | Count | Percentage |
| :--- | :--- | :--- |
| **0 (No Cardiovascular Disease)** | {stats['target_distribution'].get(0, 0):,} | {stats['target_percentage'].get(0, 0)}% |
| **1 (Cardiovascular Disease Diagnosed)** | {stats['target_distribution'].get(1, 0):,} | {stats['target_percentage'].get(1, 0)}% |

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
{summary_df.to_string()}
```
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Quality report generated at: {output_path}")


def main():
    raw_path = r"C:\Users\nilay\Downloads\rp AI healthcare.csv"
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    
    cleaned_csv_path = os.path.join(output_dir, "cardio_cleaned.csv")
    root_csv_path = os.path.join(os.path.dirname(__file__), "..", "cardio_cleaned.csv")
    
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "data_cleaning_report.md")
    
    df_raw = load_raw_dataset(raw_path)
    df_clean, stats = clean_and_engineer_features(df_raw)
    
    # Save cleaned dataset in both data/ and root for immediate user access
    df_clean.to_csv(cleaned_csv_path, index=False)
    df_clean.to_csv(root_csv_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to: {cleaned_csv_path} and {root_csv_path}")
    
    # Summary stats
    cols_to_summarize = [
        "age_years", "gender", "height", "weight", "bmi",
        "ap_hi", "ap_lo", "pulse_pressure", "mean_arterial_pressure",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
    ]
    summary_df = df_clean[cols_to_summarize].describe().round(2)
    
    generate_markdown_report(stats, summary_df, report_path)
    print("\n--- Cleaning & Audit Summary ---")
    print(f"Original Records : {stats['initial_records']:,}")
    print(f"Cleaned Records  : {stats['final_records']:,}")
    print(f"Retention Rate   : {stats['retention_rate_pct']}%")
    print(f"Total BP Remediated: {stats['audit_breakdown']['total_bp_corrected']:,}")
    print(f"Target Balance   : {stats['target_distribution']}")


if __name__ == "__main__":
    main()
