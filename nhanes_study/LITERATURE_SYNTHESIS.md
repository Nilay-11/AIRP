# What the reference papers found, and how it relates to our results

Every number below was taken from the paper's own abstract or full text: the downloaded PDFs, or PubMed/Europe PMC for paywalled papers.
"Our result" refers to our NHANES 2011–2018 experiment (test set n = 4,311).

> **Important when comparing AUCs.** Most CVD papers predict *future* (incident) CVD over 5–12 years. We classify *existing*, self-reported CVD at one point in time. Our AUCs (~0.84) are therefore **not directly comparable** with their AUCs (~0.73–0.77), so don't write "our model outperforms Weng et al." Compare the *pattern* (do ML or DL models beat simpler ones?), not the absolute numbers.

---

## 1. CVD prediction with machine learning

### Weng et al. 2017, PLoS ONE [weng2017]
- **What they did:** 378,256 UK primary-care patients; predicted a first CVD event within 10 years. Compared random forest, logistic regression, gradient boosting and a neural network against the ACC/AHA risk equation.
- **What they found:** ACC/AHA AUC 0.728. RF 0.745, **LR 0.760**, GBM 0.761, **neural network 0.764**. All ML models beat the clinical score.
- **Link to ours:** The best model (a neural network) beat logistic regression by only **0.004 AUC**, and LR beat random forest. The differences between flexible models and LR were tiny, which is the same pattern we see: all our models fall within 0.009 AUC.
- **Sentence you can use:** "Similar to Weng et al., who found that a neural network exceeded logistic regression by only 0.004 AUC, we observed that model choice had little effect on discrimination."

### Alaa et al. 2019, PLoS ONE [alaa2019]
- **What they did:** 423,604 UK Biobank participants, 473 variables, automated ML (AutoPrognosis), 5-year CVD.
- **What they found:** AutoML AUC 0.774 vs Framingham 0.724, Cox model with conventional risk factors 0.734, and Cox with all 473 variables 0.758.
- **Link to ours:** Most of their gain came from using *many more variables*. With the same variables, the gap between the Cox model and AutoML was 0.016. This supports the view that **the features matter more than the algorithm**, which matches our ablation: removing demographics cost 0.04–0.06 AUC, while changing the algorithm changed AUC by ≤0.009.

### Ambale-Venkatesh et al. 2017, Circulation Research [ambale2017]
- **What they did:** 6,814 MESA participants, 735 variables, random survival forests, 6 outcomes over 12 years.
- **What they found:** Brier score improved by 10–25% over established risk scores. **Age was the most important predictor of all-cause mortality.** Imaging and biomarkers (coronary calcium, NT-proBNP, troponin) dominated the other outcomes.
- **Link to ours:** Age is also our #1 feature in every model. Their gains came from rich imaging and biomarker data that NHANES doesn't have, which explains why routine variables plateau at our AUC level.

### Krittanawong et al. 2020, Scientific Reports [krittanawong2020]
- **What they did:** Meta-analysis of 103 cohorts (3,377,318 people).
- **What they found:** Pooled AUC for coronary artery disease: boosting 0.88 and custom algorithms 0.93. For stroke: SVM 0.92, boosting 0.91, CNN 0.90. Confidence intervals overlapped, with **no clear difference between algorithms** and **high heterogeneity** between studies.
- **Link to ours:** Overlapping intervals across algorithm families agree with our finding of no significant difference between most models. The heterogeneity they report is why we compared all models under **one identical pipeline**, which is a strength to state in the paper.

---

## 2. NHANES studies (closest to our paper)

### Dinh et al. 2019, BMC Med Inform Decis Mak [dinh2019]
- **What they did:** NHANES (1999–2014), 131 variables. LR, SVM, RF, gradient boosting and a weighted ensemble for CVD.
- **What they found:** Ensemble AU-ROC **83.1% without laboratory data**. With laboratory data, the abstract reports 83.9% (it calls this "accuracy"). Top CVD predictors: **age, systolic BP, self-reported weight, chest pain, diastolic BP**.
- **Link to ours:** Their AUC is very close to ours (0.839–0.848) on the same survey, although with 131 variables against our 20. **Age is #1 in both studies.** Two differences are worth discussing:
  1. They included **chest pain**, a symptom of the disease itself. We excluded symptoms and medications to avoid label leakage.
  2. BP ranked highly for them but contributed almost nothing for us. Our medication check shows that 62.8% of CVD cases take BP medication, which lowers their measured BP. Their importance measure (information gain from a tree) also differs from SHAP.
- **Sentence you can use:** "Using only 20 routinely collected features and excluding symptom and medication variables, our models achieved discrimination comparable to the 131-variable ensemble of Dinh et al. (AUC 0.831)."

### Martin-Morales et al. 2023, Nutrients [martinmorales2023]
- **What they did:** NHANES 2003–2012, **CVD death** (follow-up through linked death records). LR, SVM, RF, XGBoost, LightGBM, with SHAP.
- **What they found:** Random forest was the most consistent model. SHAP ranked **age and systolic blood pressure** among the most important features.
- **Link to ours:** This is very useful for our Discussion. When the outcome is a *future* event (death), systolic BP is a strong predictor. In our *prevalent* outcome it adds almost nothing. This contrast supports our explanation that treatment after diagnosis hides the BP signal in cross-sectional data.

### Ahiduzzaman & Hasan 2025, PLoS ONE [ahiduzzaman2025]
- **What they did:** NHANES 2017–2023, 12,382 adults, 41 dietary and clinical variables reduced to 30. They oversampled with ROSE and used LR, RF, SVM, XGBoost and LightGBM, with SHAP and LIME.
- **What they found:** RF had the highest AUROC (**0.8139**); XGBoost had the highest accuracy (0.8216) and recall (0.8645). Strongest predictor: **age**, followed by vitamin B12, **total cholesterol**, CRP and waist circumference.
- **Link to ours:**
  - Age #1 and total cholesterol in the top five match our SHAP results.
  - Our AUCs are higher (0.839–0.848). Possible reasons are that we pooled four cycles (21,552 people) and avoided oversampling. Oversampling distorts probability calibration, which we report and they did not.
  - They used no deep learning and no cross-model stability analysis. This is a gap our paper fills.

---

## 3. Classical machine learning vs. deep learning (supports our main conclusion)

### Christodoulou et al. 2019, J Clin Epidemiol [christodoulou2019]
- **What they did:** Systematic review of 71 studies (282 comparisons of LR against ML).
- **What they found:** In **low risk-of-bias** comparisons, the difference in logit(AUC) between ML and LR was **0.00** (95% CI −0.18 to 0.18). ML looked better only in high-bias studies (+0.34). **Calibration was not reported in 79% of studies.**
- **Link to ours:** This directly supports our finding that LR is within 0.009 AUC of the best model. Their 79% figure also justifies our full calibration reporting (ECE, slope, Brier).
- **Sentence you can use:** "Our results are consistent with Christodoulou et al., who found no performance benefit of machine learning over logistic regression in low risk-of-bias comparisons."

### Grinsztajn et al. 2022, NeurIPS [grinsztajn2022]
- **What they did:** Benchmark on 45 table-style datasets with very large hyperparameter searches (20,000 compute hours per model).
- **What they found:** **Tree-based models remain state-of-the-art on medium-sized data (~10K samples)**, even before counting their speed advantage. Neural networks struggle with uninformative features and irregular functions.
- **Link to ours:** Our training set (17,241 samples, 20 features) is exactly this "medium-sized" setting, and XGBoost was our best model. This explains why our deep models didn't win.

### Shwartz-Ziv & Armon 2022, Information Fusion [shwartzziv2022]
- **What they did:** Compared recent deep tabular models (TabNet, NODE, DNF-Net, 1D-CNN) with XGBoost, including on the datasets those models were originally published with.
- **What they found:** **XGBoost outperformed the deep models across datasets**, including their own datasets, and **needed much less tuning**. An ensemble of XGBoost and deep models did best.
- **Link to ours:** This matches our efficiency results. XGBoost tuned in 49 s against 352 s for the MLP, and the FT-Transformer took far longer. Their ensemble result is a good **Future Work** item.

---

## 4. Explainability

### Lundberg & Lee 2017, NeurIPS [lundberg2017]
- **What they did:** Introduced SHAP, showing that six earlier attribution methods are special cases of one framework with a unique solution that has desirable properties.
- **Link to ours:** SHAP is our attribution method. Cite it in Methods.

### Lundberg et al. 2020, Nature Machine Intelligence [lundberg2020]
- **What they did:** Introduced TreeSHAP, the first polynomial-time algorithm for exact SHAP values on tree models. They applied it to medical data, including **mortality risk in the US population (NHANES I)**.
- **Link to ours:** We used TreeSHAP to validate our model-agnostic SHAP estimate (ρ = 0.989). It is also an NHANES precedent for SHAP.

### Krishna et al. 2024, TMLR [krishna2024]
- **What they did:** Compared 6 explanation methods on 6 models and 4 datasets, plus interviews and a user study with data scientists.
- **What they found:** **State-of-the-art explanation methods often disagree**, and practitioners resolve disagreements with ad hoc rules.
- **Link to ours:** They studied disagreement *between explanation methods*. We show the reverse case: **the same method (SHAP) gives different rankings for different models** with near-identical accuracy (tree models vs smooth models). This extends their point and is a novel angle to emphasise.

### Rudin 2019, Nature Machine Intelligence [rudin2019]
- **What they argued:** In high-stakes settings, use inherently interpretable models instead of explaining black boxes, because post-hoc explanations can be unreliable.
- **Link to ours:** Our results support this. Logistic regression (inherently interpretable) is within 0.009 AUC of the best model and well calibrated, so the accuracy cost of interpretability here is very small.

---

## 5. Evaluation and reporting

### Van Calster et al. 2019, BMC Medicine [vancalster2019]
- **What they argued:** Calibration is often ignored, even though poorly calibrated models can mislead clinical decisions even when their AUC is good. They recommend calibration curves and slope or intercept at validation.
- **Link to ours:** We report ECE, calibration slope and intercept, Brier score and calibration curves (Fig. 1b). All models are well calibrated; the MLP is slightly over-confident (slope 0.91).

### Saito & Rehmsmeier 2015, PLoS ONE [saito2015]
- **What they found:** On imbalanced data, ROC plots can look misleadingly good. **Precision-recall** plots better reflect real performance.
- **Link to ours:** Our CVD prevalence is 11%, which justifies reporting PR-AUC (0.39–0.42) alongside ROC-AUC.

### Collins et al. 2024, BMJ (TRIPOD+AI) [collins2024]
- **What it is:** A 27-item reporting checklist for prediction-model studies using regression or ML. It replaces TRIPOD 2015.
- **Link to ours:** State in the paper that reporting follows TRIPOD+AI.

---

## 6. Overall synthesis (for Related Work and Discussion)

| Theme | Literature says | We found | Agreement |
|---|---|---|---|
| ML vs LR | No benefit in low-bias studies [christodoulou2019]; NN beat LR by only 0.004 [weng2017] | LR within 0.009 AUC of the best | ✅ consistent |
| Trees vs deep learning on table-style data | Trees remain best on ~10K samples [grinsztajn2022]; XGBoost beats deep models and needs less tuning [shwartzziv2022] | XGBoost best; MLP not significantly different; deep models far costlier to tune | ✅ consistent |
| Algorithm vs features | Gains come mainly from more or better variables [alaa2019, ambale2017] | Removing demographics cost 0.04–0.06 AUC; algorithm choice ≤0.009 | ✅ consistent |
| Most important predictor | Age #1 [dinh2019, ambale2017, ahiduzzaman2025, martinmorales2023] | Age #1 for every model | ✅ consistent |
| Blood pressure | Strong predictor of *future* CVD or death [dinh2019, martinmorales2023] | ≈0 contribution for *prevalent* CVD | ⚠️ differs, explained by treatment (62.8% of cases on BP medication) |
| Cholesterol | Total cholesterol in top-5 [ahiduzzaman2025] | Top-5, but with **inverse** direction | ⚠️ explained by statin use (58% of cases) |
| Explanation consistency | Explanation methods disagree [krishna2024] | Same method disagrees across model families | ➕ extends literature |
| Calibration | Rarely reported (79% of studies omit it) [christodoulou2019, vancalster2019] | Reported fully; all well calibrated | ➕ addresses a gap |
| Deep learning on NHANES CVD | Not compared in [dinh2019, martinmorales2023, ahiduzzaman2025] | MLP, ResNet and FT-Transformer compared | ➕ new contribution |

**Gap statement (for the end of Related Work):**
Previous NHANES studies of CVD compared classical models only [dinh2019, martinmorales2023, ahiduzzaman2025], reported SHAP for a single model, and rarely assessed calibration [christodoulou2019]. Benchmarks on table-style data suggest deep learning offers little advantage [grinsztajn2022, shwartzziv2022], and explanation methods are known to disagree [krishna2024]. No study has tested whether these findings hold for CVD classification on NHANES, or whether models with similar accuracy produce consistent explanations. Our study addresses both questions.

**Key Discussion points the literature supports:**
1. No meaningful accuracy gain from deep learning, which is expected for medium-sized table-style data [grinsztajn2022, shwartzziv2022].
2. Simple, interpretable LR is competitive [christodoulou2019, weng2017, rudin2019].
3. Differences between studies come from the outcome definition (prevalent vs incident) and the variables used, not from the algorithm [alaa2019, ambale2017].
4. The inverse cholesterol attribution and the missing BP signal are artefacts of treatment in cross-sectional data. Contrast this with prospective NHANES work where BP is predictive [martinmorales2023].
5. Explanation agreement depends on model family, so SHAP from a single model can mislead [krishna2024].
