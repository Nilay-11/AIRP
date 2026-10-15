"""Build the shareable HTML results page from the result files (no hand-copied numbers)."""
import html
import json
import pandas as pd
from common import *

T = RES / "tables"
m = pd.read_csv(RES / "test_metrics.csv", index_col=0).loc[MODEL_ORDER]
rep = pd.read_csv(RES / "repeated_split_auc.csv", index_col=0).loc[MODEL_ORDER]
tmp = pd.read_csv(RES / "temporal_validation.csv", index_col=0).loc[MODEL_ORDER]
flow = pd.read_csv(ROOT / "data" / "cohort_flow.csv")
t1 = pd.read_csv(T / "table1_characteristics.csv", index_col=0)
t3 = pd.read_csv(T / "table3_shap_stability.csv", index_col=0)
abl = pd.read_csv(RES / "ablation_summary.csv")
med = pd.read_csv(T / "supp_medication_check.csv", index_col=0)
summ = json.load(open(RES / "shap_summary.json"))
shap_t = pd.read_csv(RES / "shap_time_s.csv", index_col=0).iloc[:, 0]
params = json.load(open(RES / "best_params.json"))


def table(head, rows, num_from=1, best=None):
    h = "".join(f"<th{' class=n' if i >= num_from else ''}>{html.escape(c)}</th>" for i, c in enumerate(head))
    body = ""
    for r in rows:
        cls = ' class="best"' if best and r[0] == best else ""
        body += f"<tr{cls}>" + "".join(f"<td{' class=n' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


N = MODEL_NAMES
perf = table(["Model", "CV AUC (SD)", "Test AUC [95% CI]", "PR-AUC", "Precision", "Recall", "Specificity", "F1",
              "Brier", "ECE", "Cal. slope", "DeLong p vs best", "Family"],
             [[N[k], f"{r.cv_auc:.3f} ({r.cv_auc_sd:.3f})", f"{r.roc_auc:.3f} [{r.auc_ci_low:.3f}–{r.auc_ci_high:.3f}]",
               f"{r.pr_auc:.3f}", f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.specificity:.3f}", f"{r.f1:.3f}",
               f"{r.brier:.4f}", f"{r.ece:.4f}", f"{r.calib_slope:.2f}",
               "—" if np.isnan(r.delong_p_vs_best) else f"{r.delong_p_vs_best:.3f}",
               "Classical" if k in CLASSICAL else "Deep"] for k, r in m.iterrows()], best=N[m.roc_auc.idxmax()])
pw = pd.read_csv(RES / "delong_pairwise_p.csv", index_col=0)
delong_tbl = table(["Model"] + [N[k] for k in MODEL_ORDER],
                   [[N[a]] + ["—" if a == b else (f"<b>{pw.loc[a, b]:.3f}</b>" if pw.loc[a, b] < 0.05 else f"{pw.loc[a, b]:.3f}")
                              for b in MODEL_ORDER] for a in MODEL_ORDER])
# smallest p among deep-vs-classical pairs where the deep model had the higher AUC
dl_wins = [pw.loc[d, c] for d in DEEP for c in CLASSICAL if m.roc_auc[d] > m.roc_auc[c]]
min_p_dl_better = min(dl_wins) if dl_wins else float("nan")
robust = table(["Model", "Repeated splits AUC (mean ± SD, 5 seeds)", "Range", "Temporal AUC (2017–18)", "Temporal Brier",
                "Temporal cal. slope", "Survey-weighted test AUC"],
               [[N[k], f"{rep.loc[k, 'mean']:.3f} ± {rep.loc[k, 'std']:.3f}", f"{rep.loc[k, 'min']:.3f}–{rep.loc[k, 'max']:.3f}",
                 f"{tmp.loc[k, 'roc_auc']:.3f}", f"{tmp.loc[k, 'brier']:.4f}", f"{tmp.loc[k, 'calib_slope']:.2f}",
                 f"{m.loc[k, 'weighted_auc']:.3f}"] for k in MODEL_ORDER])
eff = table(["Model", "Tuning (s)", "Final fit (s)", "Inference (ms / 1k rows)", "SHAP, 300 rows (s)"],
            [[N[k], f"{m.loc[k, 'tune_time_s']:.0f}", f"{m.loc[k, 'fit_time_s']:.2f}", f"{m.loc[k, 'infer_ms_per_1k']:.1f}",
              f"{shap_t[k]:.0f}"] for k in MODEL_ORDER])
cohort = table(list(flow.columns.str.replace("_", " ").str.capitalize()),
               [[r.iloc[0]] + [f"{int(v):,}" for v in r.iloc[1:]] for _, r in flow.iterrows()])
t1h = table(["Characteristic"] + list(t1.columns), [[i] + list(r) for i, r in t1.iterrows()])
stab = table(["Model"] + [f"ρ vs {k}" for k in MODEL_ORDER] + ["Mean ρ vs others",
              "Within-model ρ (bootstrap)", "Within-model top-5 Jaccard", "Top-5 features"],
             [[i] + [f"{v:.3f}" for v in r.iloc[:-1]] + [r.iloc[-1]] for i, r in t3.iterrows()])
ap = abl.pivot(index="ablated", columns="model", values="mean")[MODEL_ORDER]
asd = abl.pivot(index="ablated", columns="model", values="std")[MODEL_ORDER]
ap = ap.loc[ap.mean(1).sort_values().index]
ablh = table(["Group removed"] + [N[k] for k in MODEL_ORDER],
             [[g] + [f"{ap.loc[g, k]:+.4f} ± {asd.loc[g, k]:.4f}" for k in MODEL_ORDER] for g in ap.index])
medh = table(["Measure"] + list(med.columns), [[i] + [f"{v:.1f}" for v in r] for i, r in med.iterrows()])
groups = "".join(f"<li><b>{g}:</b> {', '.join(PRETTY[f] for f in fs)}</li>" for g, fs in FEATURE_GROUPS.items())
fmt = lambda d: ", ".join(f"{k} = {round(v, 4) if isinstance(v, float) else v}" for k, v in d.items() if k != "probability")
hp = table(["Model", "Selected hyperparameters"], [[N[k], html.escape(fmt(params[k]))] for k in MODEL_ORDER])

tot = flow.iloc[-1]
r = m.roc_auc
res_md = open(ROOT / "RESULTS.md", encoding="utf-8").read()
abstract = res_md[res_md.index("> Cardiovascular disease (CVD) is a leading cause") + 2:res_md.index("\n\n---\n\n## 5. Files")]

page = open(ROOT / "share_template.html", encoding="utf-8").read()
for k, v in {"{{COHORT}}": cohort, "{{T1}}": t1h, "{{PERF}}": perf, "{{ROBUST}}": robust, "{{EFF}}": eff, "{{STAB}}": stab, "{{DELONG}}": delong_tbl, "{{FTT}}": f"{r['FTT']:.3f}",
             "{{ABL}}": ablh, "{{MED}}": medh, "{{GROUPS}}": groups, "{{HP}}": hp, "{{ABSTRACT}}": html.escape(abstract),
             "{{W}}": f"{summ['kendall_W_across_models']:.3f}", "{{TREECHK}}": f"{summ['xgb_perm_vs_tree_spearman']:.3f}",
             "{{N}}": f"{int(tot.outcome_determined):,}", "{{CASES}}": f"{int(tot.cvd_cases):,}",
             "{{XGB}}": f"{r['XGB']:.3f}", "{{MLP}}": f"{r['MLP']:.3f}", "{{LO}}": f"{r.min():.3f}"}.items():
    page = page.replace(k, v)
assert "{{" not in page
(ROOT / "share").mkdir(exist_ok=True)
open(ROOT / "share" / "nhanes_cvd_results.html", "w", encoding="utf-8").write(page)
print("written")
