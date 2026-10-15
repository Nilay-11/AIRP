"""Supplementary: treatment-related reverse causation check (statin / antihypertensive use by CVD status).
Medication variables are NOT model features; this only explains attribution directions."""
import numpy as np
import pandas as pd
from common import *

df = load()
bpq = pd.concat([pd.read_sas(ROOT / "data_raw" / f"BPQ_{c}.xpt", format="xport")[["SEQN", "BPQ100D", "BPQ050A"]]
                 for c in "GHIJ"])
d = df.merge(bpq, on="SEQN", how="left")
d["statin_use"] = (d.BPQ100D == 1).astype(int)       # now taking prescribed cholesterol medicine
d["bp_med_use"] = (d.BPQ050A == 1).astype(int)       # now taking prescribed BP medicine
w = lambda x, g: np.average(x[g.index], weights=g.WTMEC8YR)
rows = {}
for cvd, g in d.groupby("cvd"):
    rows["CVD" if cvd else "No CVD"] = {
        "Statin use, % (weighted)": 100 * np.average(g.statin_use, weights=g.WTMEC8YR),
        "BP medication use, % (weighted)": 100 * np.average(g.bp_med_use, weights=g.WTMEC8YR),
        "Total cholesterol, statin users": g[g.statin_use == 1].total_chol.mean(),
        "Total cholesterol, non-users": g[g.statin_use == 0].total_chol.mean(),
        "Systolic BP, BP-med users": g[g.bp_med_use == 1].systolic_bp.mean(),
        "Systolic BP, non-users": g[g.bp_med_use == 0].systolic_bp.mean(),
    }
out = pd.DataFrame(rows).round(1)
out.to_csv(RES / "tables" / "supp_medication_check.csv")
print(out.to_string())
