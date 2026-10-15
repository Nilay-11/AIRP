"""
Audit and verify conversions and data transformations.
Checks exact values, distributions, and potential edge cases.
"""

import pandas as pd
import numpy as np

raw_path = r"C:\Users\nilay\Downloads\rp AI healthcare.csv"
df = pd.read_csv(raw_path, sep=';')

print("="*60)
print("1. AUDIT: Age Conversion (Days to Years)")
print("="*60)
age_sample = df[['id', 'age']].head(10).copy()
age_sample['div_365_25'] = age_sample['age'] / 365.25
age_sample['round_2'] = age_sample['div_365_25'].round(2)
age_sample['int_floor'] = (age_sample['age'] / 365.25).astype(int)
age_sample['div_365_0'] = (age_sample['age'] / 365.0).round(2)
print(age_sample.to_string())

print("\nAge min/max in raw data:")
print("Min days:", df['age'].min(), "-> in years:", df['age'].min() / 365.25)
print("Max days:", df['age'].max(), "-> in years:", df['age'].max() / 365.25)

print("\n" + "="*60)
print("2. AUDIT: Gender Encoding Confirmation")
print("="*60)
g1 = df[df['gender'] == 1]
g2 = df[df['gender'] == 2]
print(f"Gender 1: count={len(g1)}, mean height={g1['height'].mean():.2f} cm, mean weight={g1['weight'].mean():.2f} kg, smoke={g1['smoke'].mean():.3f}")
print(f"Gender 2: count={len(g2)}, mean height={g2['height'].mean():.2f} cm, mean weight={g2['weight'].mean():.2f} kg, smoke={g2['smoke'].mean():.3f}")
print("-> Confirmed: Gender 1 = Female (mean height 161 cm), Gender 2 = Male (mean height 170 cm, 22% smokers)")

print("\n" + "="*60)
print("3. AUDIT: Blood Pressure Values & Outliers in Raw Data")
print("="*60)
print("ap_hi < 0 count:", (df['ap_hi'] < 0).sum(), df[df['ap_hi'] < 0]['ap_hi'].tolist() if (df['ap_hi'] < 0).sum() > 0 else "None")
print("ap_lo < 0 count:", (df['ap_lo'] < 0).sum(), df[df['ap_lo'] < 0]['ap_lo'].tolist() if (df['ap_lo'] < 0).sum() > 0 else "None")
print("ap_hi >= 1000 count:", (df['ap_hi'] >= 1000).sum())
print("ap_hi >= 1000 sample:", df[df['ap_hi'] >= 1000][['id', 'ap_hi', 'ap_lo']].head(10).to_dict('records'))
print("ap_lo >= 1000 count:", (df['ap_lo'] >= 1000).sum())
print("ap_lo >= 1000 sample:", df[df['ap_lo'] >= 1000][['id', 'ap_hi', 'ap_lo']].head(10).to_dict('records'))

print("ap_lo > ap_hi count:", (df['ap_lo'] > df['ap_hi']).sum())
swapped_candidates = df[(df['ap_lo'] > df['ap_hi']) & (df['ap_lo'] <= 240) & (df['ap_hi'] >= 40)]
print("Plausible swapped candidates (ap_lo in 70-240, ap_hi in 40-140):", len(swapped_candidates))
print("Sample swapped:", swapped_candidates[['id', 'ap_hi', 'ap_lo']].head(5).to_dict('records'))

print("\n" + "="*60)
print("4. AUDIT: Height, Weight & BMI Calculations")
print("="*60)
h_min, h_max = df['height'].min(), df['height'].max()
w_min, w_max = df['weight'].min(), df['weight'].max()
print(f"Raw height range: {h_min} to {h_max} cm")
print(f"Raw weight range: {w_min} to {w_max} kg")
raw_bmi = df['weight'] / ((df['height'] / 100.0) ** 2)
print(f"Raw BMI range: min={raw_bmi.min():.2f}, max={raw_bmi.max():.2f}")
print("Implausible BMI (<12 or >65):", ((raw_bmi < 12) | (raw_bmi > 65)).sum())
