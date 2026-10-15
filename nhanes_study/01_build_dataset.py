"""Build the harmonized NHANES 2011-2018 analytic dataset for CVD classification.

Cycles: 2011-2012 (G), 2013-2014 (H), 2015-2016 (I), 2017-2018 (J).
Outcome: self-reported physician diagnosis of any of congestive heart failure (MCQ160B),
coronary heart disease (MCQ160C), angina (MCQ160D), heart attack (MCQ160E) or stroke (MCQ160F).
Population: examined adults aged >= 20 with a determinable outcome.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent / "data_raw"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)
CYCLES = {"G": "2011-2012", "H": "2013-2014", "I": "2015-2016", "J": "2017-2018"}
CVD_ITEMS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]


def read(name, s, cols):
    d = pd.read_sas(RAW / f"{name}_{s}.xpt", format="xport")
    return d[["SEQN"] + cols]


def nan_codes(x, codes=(7, 9, 77, 99, 777, 999)):
    return x.where(~x.isin(codes))


frames, log = [], []
for s, label in CYCLES.items():
    d = read("DEMO", s, ["RIDSTATR", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR",
                         "DMDEDUC2", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"])
    for name, cols in [("BPX", ["BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3", "BPXPLS"]),
                       ("BMX", ["BMXBMI", "BMXWAIST"]), ("MCQ", CVD_ITEMS), ("TCHOL", ["LBXTC"]),
                       ("HDL", ["LBDHDD"]), ("GHB", ["LBXGH"]), ("BIOPRO", ["LBXSCR", "LBXSUA"]),
                       ("SMQ", ["SMQ020", "SMQ040"]), ("DIQ", ["DIQ010"])]:
        d = d.merge(read(name, s, cols), on="SEQN", how="left")
    n0 = len(d)
    d = d[(d.RIDSTATR == 2) & (d.RIDAGEYR >= 20)]
    n1 = len(d)
    # Outcome: 1 if any item == 1 (yes); 0 if all items == 2 (no); otherwise undetermined.
    items = d[CVD_ITEMS]
    yes = (items == 1).any(axis=1)
    all_no = (items == 2).all(axis=1)
    d["cvd"] = np.where(yes, 1, np.where(all_no, 0, np.nan))
    d = d[d.cvd.notna()]
    log.append({"cycle": label, "participants": n0, "examined_adults_20plus": n1,
                "outcome_determined": len(d), "cvd_cases": int(d.cvd.sum())})
    d["cycle"] = label
    frames.append(d)

df = pd.concat(frames, ignore_index=True)

# Harmonized features
X = pd.DataFrame({"SEQN": df.SEQN, "cycle": df.cycle})
X["age"] = df.RIDAGEYR  # top-coded at 80 in all four cycles
X["male"] = (df.RIAGENDR == 1).astype(int)
eth = df.RIDRETH1.map({1: "mexican_american", 2: "other_hispanic", 3: "nh_white", 4: "nh_black", 5: "other_multi"})
for k in ["mexican_american", "other_hispanic", "nh_black", "other_multi"]:  # reference: non-Hispanic white
    X[f"eth_{k}"] = (eth == k).astype(int)
X["income_poverty_ratio"] = df.INDFMPIR
X["education"] = nan_codes(df.DMDEDUC2)  # 1-5 ordinal
sbp = df[["BPXSY1", "BPXSY2", "BPXSY3"]].replace(0, np.nan)
dbp = df[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan)  # DBP of 0 is a recording artefact
X["systolic_bp"] = sbp.mean(axis=1)
X["diastolic_bp"] = dbp.mean(axis=1)
X["pulse_rate"] = df.BPXPLS
X["bmi"] = df.BMXBMI
X["waist_cm"] = df.BMXWAIST
X["total_chol"] = df.LBXTC
X["hdl_chol"] = df.LBDHDD
X["hba1c"] = df.LBXGH
X["creatinine"] = df.LBXSCR
X["uric_acid"] = df.LBXSUA
smq020, smq040 = nan_codes(df.SMQ020), nan_codes(df.SMQ040)
X["smoking"] = np.select([smq020 == 2, smq040.isin([1, 2]), smq040 == 3], [0, 2, 1], default=np.nan)  # 0 never,1 former,2 current
X["diabetes_dx"] = nan_codes(df.DIQ010).map({1: 1, 2: 0, 3: 0})  # borderline counted as no
X["cvd"] = df.cvd.astype(int)
X["WTMEC2YR"], X["SDMVPSU"], X["SDMVSTRA"] = df.WTMEC2YR, df.SDMVPSU, df.SDMVSTRA
X["WTMEC8YR"] = df.WTMEC2YR / 4  # NCHS guidance for combining four 2-year cycles

X.to_csv(OUT / "nhanes_cvd_2011_2018.csv", index=False)
log = pd.DataFrame(log)
log.loc[len(log)] = ["Total"] + log.iloc[:, 1:].sum().tolist()
log.to_csv(OUT / "cohort_flow.csv", index=False)
print(log.to_string(index=False))
feats = [c for c in X.columns if c not in ("SEQN", "cycle", "cvd", "WTMEC2YR", "SDMVPSU", "SDMVSTRA", "WTMEC8YR")]
print("\nMissing % per feature:\n", (X[feats].isna().mean() * 100).round(2).to_string())
print("\nUnweighted prevalence:", round(X.cvd.mean() * 100, 2),
      "% | Weighted prevalence:", round(np.average(X.cvd, weights=X.WTMEC8YR) * 100, 2), "%")
