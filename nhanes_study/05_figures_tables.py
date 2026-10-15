"""Paper figures (300 dpi PNG + PDF) and tables (CSV + Markdown)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from common import *

COLORS = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
STYLES = {m: ("-" if m in CLASSICAL else "--") for m in MODEL_ORDER}  # line style encodes family
BLUES = LinearSegmentedColormap.from_list("b", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("d", ["#c23b3a", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png"); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


def md(df, name, floatfmt=".3f"):
    (RES / "tables").mkdir(exist_ok=True)
    df.to_csv(RES / "tables" / f"{name}.csv")
    open(RES / "tables" / f"{name}.md", "w", encoding="utf-8").write(df.to_markdown(floatfmt=floatfmt))


df = load()
metrics = pd.read_csv(RES / "test_metrics.csv", index_col=0).loc[MODEL_ORDER]
preds = pd.read_csv(RES / "test_predictions.csv")
y = preds.y.values

# ---------------------------------------------------------------- Table 1: survey-weighted characteristics
w = df.WTMEC8YR


def wmean(x, wt):
    m = x.notna()
    return np.average(x[m], weights=wt[m])


def wsd(x, wt):
    m = x.notna(); mu = np.average(x[m], weights=wt[m])
    return np.sqrt(np.average((x[m] - mu) ** 2, weights=wt[m]))


rows = {}
cont = ["age", "income_poverty_ratio", "systolic_bp", "diastolic_bp", "pulse_rate", "bmi", "waist_cm",
        "total_chol", "hdl_chol", "hba1c", "creatinine", "uric_acid"]
binary = {"male": "Male sex", "diabetes_dx": "Diagnosed diabetes", "eth_nh_black": "Non-Hispanic Black",
          "eth_mexican_american": "Mexican American"}
groups = {"Overall": df, "No CVD": df[df.cvd == 0], "CVD": df[df.cvd == 1]}
for c in cont:
    rows[f"{PRETTY[c]}, mean (SD)"] = {g: f"{wmean(d[c], d.WTMEC8YR):.1f} ({wsd(d[c], d.WTMEC8YR):.1f})" for g, d in groups.items()}
for c, lab in binary.items():
    rows[f"{lab}, %"] = {g: f"{100 * wmean(d[c], d.WTMEC8YR):.1f}" for g, d in groups.items()}
for v, lab in [(2, "Current smoker, %"), (1, "Former smoker, %")]:
    rows[lab] = {g: f"{100 * wmean((d.smoking == v).astype(float).where(d.smoking.notna()), d.WTMEC8YR):.1f}" for g, d in groups.items()}
t1 = pd.DataFrame(rows).T
t1.loc["Unweighted n"] = {g: f"{len(d):,}" for g, d in groups.items()}
md(t1, "table1_characteristics")

flow = pd.read_csv(ROOT / "data" / "cohort_flow.csv").set_index("cycle")
md(flow, "cohort_flow", ".0f")

# ---------------------------------------------------------------- Table 2: performance
t2 = pd.DataFrame({
    "CV AUC (SD)": pd.Series([f"{a:.3f} ({s:.3f})" for a, s in zip(metrics.cv_auc, metrics.cv_auc_sd)], index=metrics.index),
    "Test AUC [95% CI]": pd.Series([f"{a:.3f} [{l:.3f}-{h:.3f}]" for a, l, h in zip(metrics.roc_auc, metrics.auc_ci_low, metrics.auc_ci_high)], index=metrics.index),
    "PR-AUC": metrics.pr_auc.map("{:.3f}".format), "Precision": metrics.precision.map("{:.3f}".format),
    "Recall": metrics.recall.map("{:.3f}".format), "F1": metrics.f1.map("{:.3f}".format),
    "Brier": metrics.brier.map("{:.4f}".format), "ECE": metrics.ece.map("{:.4f}".format),
    "Cal. slope": metrics.calib_slope.map("{:.2f}".format),
    "DeLong p vs best": metrics.delong_p_vs_best.map(lambda p: "-" if np.isnan(p) else f"{p:.3f}"),
    "Family": pd.Series({m: ("Classical ML" if m in CLASSICAL else "Deep learning") for m in MODEL_ORDER}),
}).rename(index=MODEL_NAMES)
md(t2, "table2_performance")
t_eff = pd.DataFrame({"Tuning time (s)": metrics.tune_time_s, "Final fit time (s)": metrics.fit_time_s,
                      "Inference (ms / 1k samples)": metrics.infer_ms_per_1k,
                      "Survey-weighted test AUC": metrics.weighted_auc,
                      "Threshold (OOF max-F1)": metrics.threshold}).rename(index=MODEL_NAMES)
shap_t = pd.read_csv(RES / "shap_time_s.csv", index_col=0).iloc[:, 0]
t_eff["SHAP time, 300 rows (s)"] = shap_t.rename(index=MODEL_NAMES)
md(t_eff, "table_efficiency", ".3f")

# ---------------------------------------------------------------- Figure 1: ROC + calibration
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
for m in MODEL_ORDER:
    fpr, tpr, _ = roc_curve(y, preds[m])
    ax[0].plot(fpr, tpr, color=COLORS[m], ls=STYLES[m], lw=1.6, label=f"{MODEL_NAMES[m]} ({metrics.roc_auc[m]:.3f})")
    pt, pp = calibration_curve(y, preds[m], n_bins=10, strategy="quantile")
    ax[1].plot(pp, pt, color=COLORS[m], ls=STYLES[m], lw=1.6, marker="o", ms=3, label=MODEL_NAMES[m])
ax[0].plot([0, 1], [0, 1], color="#aaa", lw=0.8)
ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC curves (test AUC)")
ax[0].legend(loc="lower right", fontsize=6.3, handlelength=2.6)
lim = max(preds[MODEL_ORDER].quantile(0.995).max(), 0.6)
ax[1].plot([0, lim], [0, lim], color="#aaa", lw=0.8)
ax[1].set(xlabel="Mean predicted probability", ylabel="Observed CVD fraction", title="(b) Calibration (deciles)",
          xlim=(0, lim), ylim=(0, lim))
ax[1].legend(loc="upper left", fontsize=6.3, handlelength=2.6)
fig.tight_layout(); save(fig, "fig1_roc_calibration")

# ---------------------------------------------------------------- Figure 2: SHAP importance heatmap
imp = pd.read_csv(RES / "shap_importance.csv", index_col=0)[MODEL_ORDER]
ranks = imp.rank(ascending=False).astype(int)
order = imp.div(imp.max()).mean(1).sort_values(ascending=False).index
norm = imp.div(imp.max()).loc[order]
fig, ax = plt.subplots(figsize=(4.2, 4.6))
ax.grid(False)
im = ax.imshow(norm.values, cmap=BLUES, aspect="auto", vmin=0, vmax=1)
for i, f in enumerate(order):
    for j, m in enumerate(MODEL_ORDER):
        ax.text(j, i, ranks.loc[f, m], ha="center", va="center", fontsize=6.5,
                color="white" if norm.iloc[i, j] > 0.55 else "#222")
ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=30, ha="right")
ax.axvline(len(CLASSICAL) - 0.5, color="white", lw=2)
ax.set_yticks(range(len(order)), [PRETTY[f] for f in order])
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Mean |SHAP| / model max", fontsize=7)
ax.set_title("Normalised SHAP importance (cell = rank)")
fig.tight_layout(); save(fig, "fig2_shap_importance_heatmap")

# ---------------------------------------------------------------- Figure 3: SHAP beeswarm for best model
import shap
best = metrics.roc_auc.idxmax()
sv = np.load(RES / f"shap_{best}.npy")
xs = pd.read_csv(RES / "shap_explained_rows_raw.csv")
plt.figure()
shap.summary_plot(sv, xs.rename(columns=PRETTY), max_display=12, show=False, plot_size=(4.6, 3.6))
for a in plt.gcf().axes:
    a.tick_params(labelsize=8); a.yaxis.label.set_size(8)
plt.gcf().axes[0].set_xlabel(f"SHAP value (Δ predicted probability), {MODEL_NAMES[best]}", fontsize=8)
save(plt.gcf(), f"fig3_shap_beeswarm_{best}")

# ---------------------------------------------------------------- Figure 4: ablation heatmap
abl = pd.read_csv(RES / "ablation_summary.csv")
piv = abl.pivot(index="ablated", columns="model", values="mean")[MODEL_ORDER]
sd = abl.pivot(index="ablated", columns="model", values="std")[MODEL_ORDER]
piv = piv.loc[piv.mean(1).sort_values().index]
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.grid(False)
vmax = np.abs(piv.values).max()
im = ax.imshow(piv.values, cmap=DIVERGE, norm=TwoSlopeNorm(0, -vmax, max(vmax * 0.2, 1e-3)), aspect="auto")
for i in range(piv.shape[0]):
    for j in range(len(MODEL_ORDER)):
        v = piv.iloc[i, j]
        ax.text(j, i, f"{100 * v:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v < -0.55 * vmax else "#222")
ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
ax.axvline(len(CLASSICAL) - 0.5, color="white", lw=2)
ax.set_yticks(range(len(piv)), [f"− {g}" for g in piv.index])
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label("Δ test AUC (mean of 5 splits)", fontsize=7)
ax.set_xlabel("Cell value: Δ AUC × 100", fontsize=7)
ax.set_title("Leave-one-group-out ablation")
fig.tight_layout(); save(fig, "fig4_ablation")
t_abl = piv.round(4).astype(str) + " ± " + sd.loc[piv.index].round(4).astype(str)
md(t_abl.rename(columns=MODEL_NAMES), "table4_ablation")

# ---------------------------------------------------------------- Table 3: stability
within = pd.read_csv(RES / "shap_within_model_stability.csv", index_col=0).loc[MODEL_ORDER]
pairs = pd.read_csv(RES / "shap_cross_model_agreement.csv")
summ = json.load(open(RES / "shap_summary.json"))
S = pd.DataFrame(np.eye(len(MODEL_ORDER)), index=MODEL_ORDER, columns=MODEL_ORDER)
for _, r in pairs.iterrows():
    S.loc[r.model_a, r.model_b] = S.loc[r.model_b, r.model_a] = r.spearman
t3 = S.copy()
t3["Mean ρ vs others"] = (S.sum(1) - 1) / (len(MODEL_ORDER) - 1)
t3["Within-model ρ (bootstrap)"] = within.spearman_mean
t3["Within-model top-5 Jaccard"] = within.top5_jaccard_mean
t3["Top-5 features"] = [", ".join(PRETTY[f] for f in imp[m].nlargest(5).index) for m in MODEL_ORDER]
md(t3.rename(index=MODEL_NAMES), "table3_shap_stability")
open(RES / "tables" / "table3_note.txt", "w").write(
    f"Kendall's W across all {len(MODEL_ORDER)} models: {summ['kendall_W_across_models']:.3f}\n"
    f"Kendall's W classical (4): {summ['kendall_W_classical']:.3f}; deep (3): {summ['kendall_W_deep']:.3f}\n"
    f"Mean Spearman by family pair: {summ['mean_spearman_by_family']}\n"
    f"XGB permutation-SHAP vs TreeSHAP rho: {summ['xgb_perm_vs_tree_spearman']:.3f}\n")
pd.read_csv(RES / "delong_pairwise_p.csv", index_col=0).rename(index=MODEL_NAMES, columns=MODEL_NAMES).pipe(
    md, "table_delong_pairwise")
for name in ["repeated_split_auc", "temporal_validation"]:
    md(pd.read_csv(RES / f"{name}.csv", index_col=0).rename(index=MODEL_NAMES), name, ".4f")
print("figures and tables written")
