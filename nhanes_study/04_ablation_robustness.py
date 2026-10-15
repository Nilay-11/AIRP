"""Feature-group ablation, repeated-split robustness and temporal (2017-2018) validation using tuned hyperparameters."""
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import *

SPLIT_SEEDS = [42, 1, 2, 3, 4]
df = load()
params = json.load(open(RES / "best_params.json"))
for p in params.values():
    p.pop("probability", None)  # AUC only -> SVM decision function is rank-equivalent to Platt probabilities
    if "hidden" in p:
        p["hidden"] = tuple(p["hidden"])


def run(name, feats, seed, ablated):
    train, test = split(df, seed)
    model = make_pipeline(name, **params[name])
    if name == "RF":
        model.set_params(clf__n_jobs=2)
    if name == "XGB":
        model.set_params(clf__n_jobs=1)
    model.fit(train[feats], train.cvd)
    return {"model": name, "seed": seed, "ablated": ablated,
            "auc": roc_auc_score(test.cvd, rank_scores(model, test[feats]))}


old = pd.read_csv(RES / "ablation_raw.csv") if (RES / "ablation_raw.csv").exists() else pd.DataFrame(columns=["model"])
import os
ONLY = os.environ.get("ONLY_MODELS", "").split(",") if os.environ.get("ONLY_MODELS") else MODEL_ORDER
GPU = os.environ.get("TORCH_DEVICE", "cpu") == "cuda"
todo = [m for m in MODEL_ORDER if m not in set(old.model) and m in ONLY]  # reuse stored (seeded, deterministic) results
print("ablation: reusing", sorted(set(old.model)), "| running", todo, flush=True)
tasks = [(m, FEATURES, s, "None (full)") for m in todo for s in SPLIT_SEEDS]
tasks += [(m, [f for f in FEATURES if f not in g], s, gname)
          for gname, g in FEATURE_GROUPS.items() for m in todo for s in SPLIT_SEEDS]
# slow models first so the parallel pool finishes evenly
tasks.sort(key=lambda t: {"FTT": 0, "SVM": 1, "ResNet": 2, "MLP": 3}.get(t[0], 9))
if GPU:  # one GPU, one job at a time
    new = pd.DataFrame([run(*t) for t in tasks])
else:
    new = pd.DataFrame(Parallel(n_jobs=10, verbose=5)(delayed(run)(*t) for t in tasks)) if tasks else pd.DataFrame()
out = pd.concat([old[["model", "seed", "ablated", "auc"]], new], ignore_index=True)
out.to_csv(RES / "ablation_raw.csv", index=False)

full = out[out.ablated == "None (full)"].set_index(["model", "seed"]).auc
out["delta_auc"] = out.apply(lambda r: r.auc - full[(r.model, r.seed)], axis=1)
abl = out[out.ablated != "None (full)"].groupby(["ablated", "model"]).delta_auc.agg(["mean", "std"]).reset_index()
abl.to_csv(RES / "ablation_summary.csv", index=False)
present = [m for m in MODEL_ORDER if m in set(out.model)]
rep = out[out.ablated == "None (full)"].groupby("model").auc.agg(["mean", "std", "min", "max"]).loc[present]
rep.to_csv(RES / "repeated_split_auc.csv")
print(rep.round(4).to_string())
print(abl.pivot(index="ablated", columns="model", values="mean")[present].round(4).to_string())

# ---------------------------------------------------------------- temporal validation
tr, te = df[df.cycle != "2017-2018"], df[df.cycle == "2017-2018"]
old_t = pd.read_csv(RES / "temporal_validation.csv", index_col=0) if (RES / "temporal_validation.csv").exists() else pd.DataFrame()
rows = []
for name in MODEL_ORDER:
    if name in old_t.index:
        rows.append(old_t.loc[name].to_dict() | {"model": name})
        continue
    if name not in ONLY:
        continue
    p_ = dict(params[name], **({"probability": True} if name == "SVM" else {}))
    model = make_pipeline(name, **p_).fit(tr[FEATURES], tr.cvd)
    p = scores(model, te[FEATURES])
    m = all_metrics(te.cvd.values, p, 0.5)
    rows.append({"model": name, "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"],
                 "ece": m["ece"], "calib_slope": m["calib_slope"], "calib_intercept": m["calib_intercept"]})
temporal = pd.DataFrame(rows).set_index("model")
temporal.to_csv(RES / "temporal_validation.csv")
print(f"Temporal: train n={len(tr)} test n={len(te)}\n", temporal.round(4).to_string())
