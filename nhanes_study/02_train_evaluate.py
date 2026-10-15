"""Hyperparameter tuning (5-fold CV on training set), threshold selection, held-out test evaluation.
Models whose results already exist are reused (all runs are seeded and deterministic); pass --force to retrain all."""
import json
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from scipy.stats import randint, uniform, loguniform
from common import *

df = load()
train, test = split(df)
Xtr, ytr, Xte, yte = train[FEATURES], train.cvd.values, test[FEATURES], test.cvd.values
print(f"train n={len(train)} (cases {ytr.sum()}), test n={len(test)} (cases {yte.sum()})")
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

GRIDS = {
    "LR": {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "RF": {"clf__max_depth": [8, 12, None], "clf__min_samples_leaf": [5, 20, 50], "clf__max_features": ["sqrt", 0.5]},
    "SVM": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.001, 0.0003], "clf__class_weight": [None, "balanced"]},
    "MLP": {"clf__hidden": [(16,), (32, 16), (64, 32), (128, 64), (256, 128, 64)], "clf__dropout": [0.1, 0.3, 0.5],
            "clf__weight_decay": [1e-4, 1e-2], "clf__lr": [1e-3, 3e-4]},
    "ResNet": {"clf__d": [64, 128, 256], "clf__n_blocks": [1, 2, 4], "clf__dropout": [0.1, 0.3, 0.5],
               "clf__weight_decay": [1e-4, 1e-2], "clf__lr": [1e-3]},
    "FTT": {"clf__d": [64, 128], "clf__n_blocks": [2, 3], "clf__lr": [1e-4, 3e-4],
            "clf__attn_dropout": [0.2], "clf__ffn_dropout": [0.1], "clf__weight_decay": [1e-5]},
}
XGB_DIST = {"clf__n_estimators": randint(100, 1000), "clf__max_depth": randint(2, 7),
            "clf__learning_rate": loguniform(0.01, 0.2), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 20),
            "clf__reg_lambda": loguniform(0.1, 10)}

force = "--force" in sys.argv
old_res = pd.read_csv(RES / "test_metrics.csv", index_col=0) if (RES / "test_metrics.csv").exists() else pd.DataFrame()
old_pred = pd.read_csv(RES / "test_predictions.csv") if (RES / "test_predictions.csv").exists() else pd.DataFrame()
old_par = json.load(open(RES / "best_params.json")) if (RES / "best_params.json").exists() else {}

rows, tuning, preds = [], {}, {"y": yte}
for name in MODEL_ORDER:
    if not force and name in old_res.index and name in old_pred and name in old_par and (MODELS_DIR / f"{name}.joblib").exists():
        assert (old_pred.y.values == yte).all()
        rows.append(old_res.loc[name].to_dict() | {"model": name})
        preds[name], tuning[name] = old_pred[name].values, old_par[name]
        print(f"{name}: reused stored results (test AUC {old_res.loc[name, 'roc_auc']:.4f})", flush=True)
        continue
    t0 = time.perf_counter()
    if name == "XGB":
        search = RandomizedSearchCV(make_pipeline(name), XGB_DIST, n_iter=40, scoring="roc_auc", cv=cv,
                                    n_jobs=3, random_state=SEED)
    else:
        search = GridSearchCV(make_pipeline(name), GRIDS[name], scoring="roc_auc", cv=cv,
                              n_jobs=1 if name == "RF" else -1)
    search.fit(Xtr, ytr)
    tune_time = time.perf_counter() - t0
    best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    if name == "SVM":
        best["probability"] = True
    cv_sd = search.cv_results_["std_test_score"][search.best_index_]
    print(f"{name}: CV AUC {search.best_score_:.4f}±{cv_sd:.4f} params {best} ({tune_time:.0f}s)", flush=True)

    # threshold maximising F1 on out-of-fold training predictions (test set untouched)
    oof = cross_val_predict(make_pipeline(name, **best), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=5)[:, 1]
    thr = best_f1_threshold(ytr, oof)

    model = make_pipeline(name, **best)
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_time = time.perf_counter() - t0
    t0 = time.perf_counter(); p = scores(model, Xte); infer_time = time.perf_counter() - t0
    m = all_metrics(yte, p, thr)
    lo, hi = bootstrap_ci(yte, p, roc_auc_score)
    m.update(model=name, cv_auc=search.best_score_, cv_auc_sd=cv_sd, auc_ci_low=lo, auc_ci_high=hi,
             weighted_auc=roc_auc_score(yte, p, sample_weight=test.WTMEC8YR), tune_time_s=tune_time,
             fit_time_s=fit_time, infer_ms_per_1k=infer_time / len(Xte) * 1e6)
    if name in DEEP:
        m["mlp_epochs"] = model.named_steps["clf"].n_epochs_
        m["mlp_params"] = sum(p_.numel() for p_ in model.named_steps["clf"].net_.parameters())
    rows.append(m)
    tuning[name] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()}
    preds[name] = p
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    pd.DataFrame(search.cv_results_).to_csv(RES / f"cv_results_{name}.csv", index=False)

res = pd.DataFrame(rows).set_index("model").loc[MODEL_ORDER]
res = res.drop(columns=[c for c in res.columns if c.startswith("delong")])
best_name = res.roc_auc.idxmax()
for name in MODEL_ORDER:
    res.loc[name, "delong_diff_vs_MLP"], res.loc[name, "delong_p_vs_MLP"] = (
        delong_test(yte, preds[name], preds["MLP"]) if name != "MLP" else (0.0, np.nan))
    res.loc[name, "delong_p_vs_best"] = delong_test(yte, preds[name], preds[best_name])[1] if name != best_name else np.nan
pair = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER)
for a in MODEL_ORDER:
    for b in MODEL_ORDER:
        if a != b:
            pair.loc[a, b] = delong_test(yte, preds[a], preds[b])[1]
pair.to_csv(RES / "delong_pairwise_p.csv")
res.to_csv(RES / "test_metrics.csv")
pd.DataFrame(preds).to_csv(RES / "test_predictions.csv", index=False)
json.dump(tuning, open(RES / "best_params.json", "w"), indent=2, default=str)
pd.set_option("display.width", 250)
print(res.round(4).T.to_string())
print("\nPairwise DeLong p-values:\n", pair.round(3).to_string())
