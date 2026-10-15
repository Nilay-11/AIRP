# Classical Machine Learning vs. Deep Learning for Cardiovascular Disease Classification (NHANES 2011–2018)

Research project comparing **4 classical machine-learning models** (Logistic Regression, Random Forest, SVM, XGBoost) with **3 deep-learning models** (MLP, tabular ResNet, FT-Transformer) for classifying self-reported cardiovascular disease. The data is four cycles of the U.S. National Health and Nutrition Examination Survey: 21,552 adults, of whom 2,369 have CVD. Models are compared on discrimination, calibration, statistical significance (DeLong test), computational cost and SHAP explanations, including how well the explanations agree across models and how stable they are.

## Main results (held-out test set, n = 4,311)

| Model | Family | Test ROC-AUC [95% CI] | PR-AUC | F1 | Brier | ECE |
|---|---|---|---|---|---|---|
| Logistic Regression | Classical | 0.839 [0.821–0.855] | 0.392 | 0.455 | 0.080 | 0.010 |
| Random Forest | Classical | 0.839 [0.821–0.856] | 0.407 | 0.447 | 0.080 | 0.010 |
| SVM (RBF) | Classical | 0.839 [0.821–0.854] | 0.396 | 0.447 | 0.080 | 0.007 |
| **XGBoost** | Classical | **0.848 [0.829–0.863]** | 0.418 | 0.451 | 0.078 | 0.009 |
| MLP | Deep | 0.842 [0.825–0.858] | 0.398 | 0.447 | 0.080 | 0.018 |
| ResNet | Deep | 0.837 [0.819–0.855] | 0.402 | 0.439 | 0.080 | 0.021 |
| FT-Transformer | Deep | 0.843 [0.825–0.859] | 0.419 | 0.455 | 0.079 | 0.012 |

No deep-learning model significantly outperformed any classical model (DeLong p ≥ 0.064). See `nhanes_study/RESULTS.md` for all results.

## Repository layout

| Path | Contents |
|---|---|
| `nhanes_study/` | The NHANES experiment: code, results, figures, write-ups |
| `nhanes_study/00_download_nhanes.py` … `09_build_summary_pdf.py` | Pipeline scripts, run in numeric order |
| `nhanes_study/common.py` | Features, model definitions (including the PyTorch MLP, ResNet and FT-Transformer), metrics, DeLong test |
| `nhanes_study/results/` | All metrics, predictions, SHAP values, ablation, stability, tables (`results/tables/`) |
| `nhanes_study/figures/` | Paper figures (PNG and vector PDF) |
| `nhanes_study/RESULTS.md` | Full results write-up |
| `nhanes_study/LITERATURE_SYNTHESIS.md` | What the cited papers found and how it relates to our results |
| `nhanes_study/STUDY_SUMMARY_FOR_TEAM.md` | Dataset, features, models, comparison methods, abstract |
| `overleaf_paper/` | IEEE LaTeX paper (`main.tex`, `refs.bib`, figures). Upload it to Overleaf |
| `reference_papers/` | Reference list with links, plus PDFs of the freely available papers (for team use only; this repository is private) |
| `nhanes_study/data_raw/` | Raw NHANES files as downloaded from the CDC |
| `nhanes_study/models/` | The 7 trained models (scikit-learn pipelines saved with joblib) |
| `*.pdf` (root) | Shareable PDF reports |
| `src/`, `reports/`, `results/`, `data/`, `cardio_cleaned.csv`, notebook | Earlier exploratory work on the Kaggle cardiovascular dataset |

## Reproducing the NHANES experiment

```bash
pip install pandas numpy scikit-learn xgboost torch shap scipy matplotlib reportlab pypdf tabulate
cd nhanes_study
python 00_download_nhanes.py        # downloads NHANES files from the CDC
python 01_build_dataset.py          # harmonised analytic dataset
python 02_train_evaluate.py         # tuning + test evaluation (FT-Transformer tuning takes ~1.5 h on CPU)
python 03_shap_stability.py         # SHAP + stability
python 04_ablation_robustness.py    # ablation, repeated splits, temporal validation
python 05_figures_tables.py         # figures and tables
python 06_medication_check.py       # supplementary statin / BP-medication check
```

Everything is seeded (seed 42). All models train on the CPU. For speed, the FT-Transformer's ablation and bootstrap refits were run on a GPU (`TORCH_DEVICE=cuda`).

## Data

NHANES is public-domain data from the U.S. Centers for Disease Control and Prevention, National Center for Health Statistics: https://wwwn.cdc.gov/nchs/nhanes/
