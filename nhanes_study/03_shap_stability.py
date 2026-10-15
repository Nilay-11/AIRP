"""SHAP attributions for all models with one model-agnostic method, plus cross-model and
within-model (bootstrap refit) feature-importance stability. Stored per-model results are reused."""
import json
import time
import itertools
import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr, kendalltau
from common import *

N_EXPLAIN, N_BACKGROUND, MAX_EVALS, N_BOOT, TOPK, N_EXPLAIN_BOOT = 300, 50, 4 * (2 * len(FEATURES) + 1), 5, 5, 100

df = load()
train, test = split(df)
Xtr, ytr, Xte = train[FEATURES], train.cvd.values, test[FEATURES]
params = json.load(open(RES / "best_params.json"))
for p in params.values():
    if "hidden" in p:
        p["hidden"] = tuple(p["hidden"])

rng = np.random.default_rng(SEED)
explain_idx = rng.choice(len(Xte), N_EXPLAIN, replace=False)
bg_idx = rng.choice(len(Xtr), N_BACKGROUND, replace=False)


def shap_values(model, seed=SEED, n=N_EXPLAIN):
    imp = model.named_steps["impute"]
    bg = pd.DataFrame(imp.transform(Xtr.iloc[bg_idx]), columns=FEATURES)
    xs = pd.DataFrame(imp.transform(Xte.iloc[explain_idx[:n]]), columns=FEATURES)
    f = lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURES))[:, 1]
    expl = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=N_BACKGROUND), seed=seed)
    return expl(xs, max_evals=MAX_EVALS, silent=True).values, xs


def importance(sv):
    return pd.Series(np.abs(sv).mean(0), index=FEATURES)


# ---------------------------------------------------------------- main SHAP on final models
imp_tbl, direction, timing = {}, {}, {}
old_time = pd.read_csv(RES / "shap_time_s.csv", index_col=0).iloc[:, 0] if (RES / "shap_time_s.csv").exists() else pd.Series(dtype=float)
for name in MODEL_ORDER:
    model = joblib.load(MODELS_DIR / f"{name}.joblib")
    if (RES / f"shap_{name}.npy").exists() and name in old_time.index:  # reuse (seeded, deterministic)
        sv = np.load(RES / f"shap_{name}.npy")
        xs = pd.DataFrame(model.named_steps["impute"].transform(Xte.iloc[explain_idx]), columns=FEATURES)
        timing[name] = old_time[name]
    else:
        t0 = time.perf_counter()
        sv, xs = shap_values(model)
        timing[name] = time.perf_counter() - t0
        np.save(RES / f"shap_{name}.npy", sv)
    imp_tbl[name] = importance(sv)
    # direction: Spearman correlation between feature value and its SHAP value
    direction[name] = pd.Series({c: spearmanr(xs[c], sv[:, i])[0] if xs[c].nunique() > 1 else np.nan
                                 for i, c in enumerate(FEATURES)})
    print(f"{name}: SHAP done in {timing[name]:.0f}s; top5 {list(imp_tbl[name].nlargest(TOPK).index)}", flush=True)
pd.DataFrame(imp_tbl).loc[FEATURES].to_csv(RES / "shap_importance.csv")
pd.DataFrame(direction).to_csv(RES / "shap_direction.csv")
pd.DataFrame(Xte.iloc[explain_idx]).to_csv(RES / "shap_explained_rows_raw.csv", index=False)
pd.Series(timing).to_csv(RES / "shap_time_s.csv")

# Sanity check: permutation SHAP vs exact TreeSHAP (probability scale) for XGBoost
xgb = joblib.load(MODELS_DIR / "XGB.joblib")
pre = xgb[:-1]
tree_exp = shap.TreeExplainer(xgb.named_steps["clf"], pre.transform(Xtr.iloc[bg_idx]),
                              feature_perturbation="interventional", model_output="probability")
tree_sv = tree_exp.shap_values(pre.transform(Xte.iloc[explain_idx]))
check = spearmanr(importance(tree_sv), imp_tbl["XGB"])[0]
print(f"XGB permutation-SHAP vs TreeSHAP importance Spearman rho = {check:.3f}", flush=True)

# ---------------------------------------------------------------- cross-model agreement
imp = pd.DataFrame(imp_tbl)
pairs = []
for a, b in itertools.combinations(MODEL_ORDER, 2):
    ta, tb = set(imp[a].nlargest(TOPK).index), set(imp[b].nlargest(TOPK).index)
    pairs.append({"model_a": a, "model_b": b, "spearman": spearmanr(imp[a], imp[b])[0],
                  "kendall": kendalltau(imp[a], imp[b])[0], f"top{TOPK}_jaccard": len(ta & tb) / len(ta | tb)})
pairs = pd.DataFrame(pairs)
ranks = imp.rank(ascending=False)
k, n = ranks.shape[1], ranks.shape[0]
R = ranks.sum(1)
kendall_w = 12 * ((R - R.mean()) ** 2).sum() / (k ** 2 * (n ** 3 - n))
pairs.to_csv(RES / "shap_cross_model_agreement.csv", index=False)
ranks.to_csv(RES / "shap_ranks.csv")

# ---------------------------------------------------------------- within-model stability (bootstrap refits)
import os
ONLY = os.environ.get("ONLY_MODELS", "").split(",") if os.environ.get("ONLY_MODELS") else MODEL_ORDER
boot = {m: [] for m in MODEL_ORDER}
cached = {m for m in MODEL_ORDER if (RES / f"shap_bootstrap_importance_{m}.csv").exists()}
cached |= {m for m in MODEL_ORDER if m not in ONLY}  # skip models outside this run's subset
for m in cached:
    f = RES / f"shap_bootstrap_importance_{m}.csv"
    boot[m] = [r for _, r in pd.read_csv(f)[FEATURES].iterrows()] if f.exists() else []
for b in range(N_BOOT):
    bi = np.random.default_rng(1000 + b).integers(0, len(Xtr), len(Xtr))  # same resamples for every model
    for name in MODEL_ORDER:
        if name in cached:
            continue
        model = make_pipeline(name, **params[name])
        if name in ("RF", "XGB") or name in DEEP:
            model.set_params(clf__random_state=1000 + b)
        model.fit(Xtr.iloc[bi], ytr[bi])
        boot[name].append(importance(shap_values(model, seed=1000 + b, n=N_EXPLAIN_BOOT)[0]))
    print(f"bootstrap {b + 1}/{N_BOOT} done", flush=True)
for name in ONLY:  # save each model's refits as soon as they exist
    if len(boot[name]) == N_BOOT and not (RES / f"shap_bootstrap_importance_{name}.csv").exists():
        pd.DataFrame(boot[name]).to_csv(RES / f"shap_bootstrap_importance_{name}.csv", index=False)
missing = [m for m in MODEL_ORDER if len(boot[m]) != N_BOOT]
if missing:
    print("partial run; bootstrap still missing for", missing, "- summaries not written", flush=True)
    raise SystemExit(0)
within = []
for name in MODEL_ORDER:
    rhos, jac = [], []
    for a, c in itertools.combinations(boot[name], 2):
        rhos.append(spearmanr(a, c)[0])
        ta, tc = set(a.nlargest(TOPK).index), set(c.nlargest(TOPK).index)
        jac.append(len(ta & tc) / len(ta | tc))
    bi = pd.DataFrame(boot[name])
    within.append({"model": name, "spearman_mean": np.mean(rhos), "spearman_min": np.min(rhos),
                   f"top{TOPK}_jaccard_mean": np.mean(jac),
                   "mean_cv_of_importance_top5": (bi.std() / bi.mean())[imp[name].nlargest(TOPK).index].mean()})
    bi.to_csv(RES / f"shap_bootstrap_importance_{name}.csv", index=False)
within = pd.DataFrame(within)
within.to_csv(RES / "shap_within_model_stability.csv", index=False)

def kendall_w_of(models):
    r = imp[models].rank(ascending=False)
    k_, n_ = r.shape[1], r.shape[0]
    s = r.sum(1)
    return 12 * ((s - s.mean()) ** 2).sum() / (k_ ** 2 * (n_ ** 3 - n_))


rho = {(r.model_a, r.model_b): r.spearman for r in pairs.itertuples()}
fam = lambda a, b: ("within_classical" if a in CLASSICAL and b in CLASSICAL else
                    "within_deep" if a in DEEP and b in DEEP else "classical_vs_deep")
fam_mean = pd.Series({k_: v for k_, v in rho.items()}).groupby(lambda k_: fam(*k_)).mean().to_dict()
summary = {"kendall_W_across_models": kendall_w, "kendall_W_classical": kendall_w_of(CLASSICAL),
           "kendall_W_deep": kendall_w_of(DEEP), "mean_spearman_by_family": fam_mean,
           "xgb_perm_vs_tree_spearman": check,
           "n_explain": N_EXPLAIN, "n_background": N_BACKGROUND, "max_evals": MAX_EVALS, "n_boot": N_BOOT, "n_explain_boot": N_EXPLAIN_BOOT}
json.dump(summary, open(RES / "shap_summary.json", "w"), indent=2)
print(imp.round(4).to_string()); print(pairs.round(3).to_string()); print(within.round(3).to_string()); print(summary)
