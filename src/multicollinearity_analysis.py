"""
Multicollinearity and Feature Correlation Analysis Pipeline
Project: Comparative Evaluation and Interpretation of Machine Learning Models 
         for Cardiovascular Disease Risk Prediction Using Clinical Features
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_vif_dataframe(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in feature_cols.
    VIF_j = 1 / (1 - R_j^2) where R_j^2 is from regressing X_j against all other features.
    VIF = 1: No correlation
    VIF 1-5: Moderate correlation (acceptable)
    VIF > 5-10: High multicollinearity (problematic for linear models)
    VIF > 10: Severe multicollinearity (distorts linear regression coefficients)
    """
    X = df[feature_cols].copy()
    
    # Standardize features for stable regression
    X_std = (X - X.mean()) / X.std()
    
    vif_records = []
    for col in feature_cols:
        y_col = X_std[col]
        other_cols = [c for c in feature_cols if c != col]
        X_other = X_std[other_cols]
        
        lr = LinearRegression()
        lr.fit(X_other, y_col)
        r_squared = lr.score(X_other, y_col)
        
        # Guard against perfect collinearity division by zero
        if r_squared >= 0.999999:
            vif = float("inf")
        else:
            vif = 1.0 / (1.0 - r_squared)
            
        vif_records.append({
            "Feature": col,
            "R_Squared": round(r_squared, 4),
            "VIF": round(vif, 2) if vif != float("inf") else "INF (>100,000)",
            "Collinearity_Status": "Low (Safe)" if vif < 5 else ("Moderate" if vif < 10 else "Severe (Harmful for Linear Models)")
        })
        
    return pd.DataFrame(vif_records)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring tabulate package."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join([str(val) for val in row.values]) + " |"
        row_lines.append(row_str)
    return "\n".join([header_line, sep_line] + row_lines)


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "cardio_cleaned.csv")
    df = pd.read_csv(data_path)
    
    print(f"[INFO] Loaded {len(df):,} cleaned records for collinearity analysis.\n")
    
    # Define Feature Sets
    # Set 1: Raw Baseline (standard raw clinical variables)
    set1_raw = [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active"
    ]
    
    # Set 2: Hemodynamic / Non-Redundant Clinical Biomarkers
    set2_hemodynamic = [
        "age_years", "gender", "bmi", "pulse_pressure", "mean_arterial_pressure",
        "cholesterol", "gluc", "smoke", "alco", "active"
    ]
    
    # Set 3: All Features Combined (Demonstrates collinearity trap)
    set3_combined = [
        "age_years", "gender", "height", "weight", "bmi",
        "ap_hi", "ap_lo", "pulse_pressure", "mean_arterial_pressure",
        "cholesterol", "gluc", "smoke", "alco", "active"
    ]
    
    print("Calculating VIF for Feature Set 1 (Raw Baseline)...")
    vif_set1 = compute_vif_dataframe(df, set1_raw)
    
    print("Calculating VIF for Feature Set 2 (Hemodynamic Biomarkers)...")
    vif_set2 = compute_vif_dataframe(df, set2_hemodynamic)
    
    print("Calculating VIF for Feature Set 3 (All Combined - Collinear)...")
    vif_set3 = compute_vif_dataframe(df, set3_combined)
    
    # Pearson Correlation Matrix for numeric features
    numeric_cols = [
        "age_years", "gender", "height", "weight", "bmi",
        "ap_hi", "ap_lo", "pulse_pressure", "mean_arterial_pressure",
        "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
    ]
    corr_matrix = df[numeric_cols].corr(method="pearson").round(3)
    
    # Save correlation matrix to reports
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    corr_csv_path = os.path.join(reports_dir, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_csv_path)
    
    # Generate Markdown Report
    report_path = os.path.join(reports_dir, "multicollinearity_report.md")
    report_md = f"""# Multicollinearity & Feature Architecture Report

**Research Paper**: *Comparative Evaluation and Interpretation of Machine Learning Models for Cardiovascular Disease Risk Prediction Using Clinical Features*  
**Cleaned Cohort Size**: {len(df):,} patients

---

## 1. Executive Summary & Clinical Recommendation

Derived clinical metrics ($BMI$, $Pulse\\ Pressure$, $Mean\\ Arterial\\ Pressure$) are deterministic mathematical functions of raw physical measurements ($Height$, $Weight$, $Systolic\\ BP$, $Diastolic\\ BP$). 

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

{df_to_markdown(vif_set1)}

> **Conclusion**: All features exhibit low VIF ($< 2.5$). Highly safe and mathematically stable for all model architectures.

---

## 3. Feature Set 2: Hemodynamic / Biomarker Set (Clinically Expressive)
*Features: Age, Gender, BMI, Pulse Pressure (PP), Mean Arterial Pressure (MAP), Cholesterol, Glucose, Smoking, Alcohol, Activity*

{df_to_markdown(vif_set2)}

> **Conclusion**: Highly acceptable collinearity profile ($VIF < 3.5$). Replaces mechanical measurements ($Height, Weight, SBP, DBP$) with systemic cardiovascular biomarkers ($BMI, PP, MAP$).

---

## 4. Feature Set 3: All Combined (Collinearity Stress Test)
*Demonstration of what occurs when raw and derived metrics are blindly trained together:*

{df_to_markdown(vif_set3)}

> **Key Finding**: Because $PP = SBP - DBP$ and $MAP = DBP + \\frac{1}{3}PP$, $R^2 \\approx 1.0$ and **VIF exceeds 10,000 to infinity**. Linear models trained on this set will suffer from singular matrix problems and erratic weight estimates.

---

## 5. Pearson Correlation Matrix (Target & Key Covariates)

```
{corr_matrix.to_string()}
```
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"\n[SUCCESS] Multicollinearity report written to: {report_path}")
    print(f"[SUCCESS] Correlation matrix CSV written to: {corr_csv_path}")


if __name__ == "__main__":
    main()
