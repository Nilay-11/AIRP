"""Update build_manuscript.js: Holm-corrected DeLong table/text, methods sentence, abstract wording and an explicit
contribution statement. Table values are generated from nhanes_study/results (no hand-typed numbers)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / "nhanes_study" / "results"
ORDER = ["LR", "RF", "SVM", "XGB", "MLP", "ResNet", "FTT"]
LABEL = {"FTT": "FT-T"}
raw = pd.read_csv(RES / "delong_pairwise_p.csv", index_col=0)
holm = pd.read_csv(RES / "delong_pairwise_p_holm.csv", index_col=0)


def fmt(p):
    s = f"{p:.3f}"
    return f"**{s}**" if p < 0.05 else s


rows = []
for i, a in enumerate(ORDER):
    r = [LABEL.get(a, a)]
    for j, b in enumerate(ORDER):
        r.append("–" if i == j else fmt(raw.loc[a, b] if j > i else holm.loc[a, b]))
    rows.append(r)
table_js = ",\n".join("   " + json.dumps(r, ensure_ascii=False) for r in rows)

js = (ROOT / "build_manuscript.js").read_text(encoding="utf-8")

# 1) results paragraph + table 3
start = js.index('C.push(body("Pairwise DeLong tests')
end = js.index("[1.1, 1, 1, 1, 1, 1, 1, 1]));", start) + len("[1.1, 1, 1, 1, 1, 1, 1, 1]));")
new_block = (
    'C.push(body("Pairwise DeLong tests compared all 21 model pairs (Table 3). After Holm correction for multiple comparisons, '
    'XGBoost remained significantly better than RF (adjusted p = 0.012), ResNet (p = 0.036) and SVM (p = 0.041); its advantage over LR '
    '(unadjusted p = 0.005) was no longer significant (adjusted p = 0.095), and it did not differ significantly from the MLP or the '
    'FT-Transformer (adjusted p = 1.000). No deep learning model was significantly better than any classical model, with or without '
    'correction: wherever a deep model had the higher AUC (the MLP and FT-Transformer versus LR, RF and SVM), unadjusted p ranged from '
    '0.161 to 0.460. The difference between the MLP and ResNet (unadjusted p = 0.031) did not survive correction (adjusted p = 0.525)."));\n'
    'C.push(...table("**Table 3.** Pairwise DeLong test p-values for differences in test ROC-AUC. Upper triangle: unadjusted; '
    'lower triangle: Holm-adjusted for 21 comparisons (bold: p < 0.05)",\n'
    '  ["", "LR", "RF", "SVM", "XGB", "MLP", "ResNet", "FT-T"],\n'
    f"  [\n{table_js}],\n"
    "  [1.1, 1, 1, 1, 1, 1, 1, 1]));"
)
js = js[:start] + new_block + js[end:]

# 2) methods
a = "Paired AUC differences were tested with DeLong’s method (DeLong et al., 1988)."
b = ("Paired AUC differences were tested with DeLong’s method (DeLong et al., 1988), and p-values for the 21 pairwise comparisons "
     "were adjusted with Holm’s step-down procedure (Holm, 1979).")
assert js.count(a) == 1
js = js.replace(a, b)

# 3) abstract
a = "Models were evaluated for discrimination (ROC-AUC, PR-AUC, DeLong test)"
b = "Models were evaluated for discrimination (ROC-AUC, PR-AUC, Holm-corrected DeLong tests)"
assert js.count(a) == 1
js = js.replace(a, b)

# 4) explicit contribution statement (replaces the last Introduction paragraph)
start = js.index('  "This study compares four classical machine learning models with three deep learning models')
end = js.index('",\n', start) + 3
contrib = (
    '  "This study compares four classical machine learning models (LR, RF, SVM, XGBoost) with three deep learning models '
    '(MLP, tabular ResNet, FT-Transformer) for CVD classification on harmonized NHANES 2011–2018 data. Its contributions are fourfold. '
    'First, it provides a controlled comparison in which all seven models share the same data splits, preprocessing and tuning protocol, '
    'with pairwise DeLong tests corrected for multiple comparisons. Second, it evaluates calibration (Brier score, expected calibration '
    'error and calibration slope) alongside discrimination, which most prior CVD studies omit. Third, it quantifies computational cost '
    '(tuning, training and inference time) for every model. Fourth, it measures whether SHAP explanations agree across model families '
    'and remain stable across bootstrap refits, complemented by feature-group ablation and an analysis of treatment-related reverse '
    'causation.",\n'
)
js = js[:start] + contrib + js[end:]
(ROOT / "build_manuscript.js").write_text(js, encoding="utf-8")

# 5) Holm reference (APA)
refs = json.loads((ROOT / "apa_refs.json").read_text(encoding="utf-8"))
refs["holm1979"] = ("Holm, S. (1979). A simple sequentially rejective multiple test procedure. "
                    "*Scandinavian Journal of Statistics*, *6*(2), 65–70.")
(ROOT / "apa_refs.json").write_text(json.dumps(refs, indent=1, ensure_ascii=False), encoding="utf-8")
print("updated; table 3 rows:\n" + table_js)
