# References for the paper (IEEE style)

Numbered in the order they would first be cited (Introduction → Related Work → Methodology → Discussion), as IEEE requires.
Each entry was checked against Crossref or the publisher/PubMed Central page in October 2026. DOIs are given wherever one exists.
Entries marked † are conference/JMLR papers that have no DOI; they were checked against the proceedings or journal page.

A 6-page IEEE paper usually has 20–30 references. The **core set** (★) is 25 entries and fits about 0.9 page. The rest are optional extras.

---

## A. Introduction: CVD burden and clinical risk scores

★ [1] C. W. Tsao, A. W. Aday, Z. I. Almarzooq, *et al.*, "Heart disease and stroke statistics—2023 update: A report from the American Heart Association," *Circulation*, vol. 147, no. 8, pp. e93–e621, 2023, doi: 10.1161/CIR.0000000000001123.

★ [2] G. A. Roth, G. A. Mensah, C. O. Johnson, *et al.*, "Global burden of cardiovascular diseases and risk factors, 1990–2019: Update from the GBD 2019 study," *J. Am. Coll. Cardiol.*, vol. 76, no. 25, pp. 2982–3021, 2020, doi: 10.1016/j.jacc.2020.11.010.

★ [3] R. B. D'Agostino Sr., R. S. Vasan, M. J. Pencina, *et al.*, "General cardiovascular risk profile for use in primary care: The Framingham Heart Study," *Circulation*, vol. 117, no. 6, pp. 743–753, 2008, doi: 10.1161/CIRCULATIONAHA.107.699579.

★ [4] D. C. Goff Jr., D. M. Lloyd-Jones, G. Bennett, *et al.*, "2013 ACC/AHA guideline on the assessment of cardiovascular risk: A report of the American College of Cardiology/American Heart Association Task Force on Practice Guidelines," *Circulation*, vol. 129, no. 25, suppl. 2, pp. S49–S73, 2014, doi: 10.1161/01.cir.0000437741.48606.98.

[5] J. Hippisley-Cox, C. Coupland, and P. Brindle, "Development and validation of QRISK3 risk prediction algorithms to estimate future risk of cardiovascular disease: Prospective cohort study," *BMJ*, vol. 357, p. j2099, 2017, doi: 10.1136/bmj.j2099.

## B. Related Work: machine learning for CVD

★ [6] S. F. Weng, J. Reps, J. Kai, J. M. Garibaldi, and N. Qureshi, "Can machine-learning improve cardiovascular risk prediction using routine clinical data?," *PLoS ONE*, vol. 12, no. 4, p. e0174944, 2017, doi: 10.1371/journal.pone.0174944.

★ [7] A. M. Alaa, T. Bolton, E. Di Angelantonio, J. H. F. Rudd, and M. van der Schaar, "Cardiovascular disease risk prediction using automated machine learning: A prospective study of 423,604 UK Biobank participants," *PLoS ONE*, vol. 14, no. 5, p. e0213653, 2019, doi: 10.1371/journal.pone.0213653.

★ [8] B. Ambale-Venkatesh, X. Yang, C. O. Wu, *et al.*, "Cardiovascular event prediction by machine learning: The Multi-Ethnic Study of Atherosclerosis," *Circ. Res.*, vol. 121, no. 9, pp. 1092–1101, 2017, doi: 10.1161/CIRCRESAHA.117.311312.

★ [9] C. Krittanawong, H. U. H. Virk, S. Bangalore, *et al.*, "Machine learning prediction in cardiovascular diseases: A meta-analysis," *Sci. Rep.*, vol. 10, no. 1, p. 16057, 2020, doi: 10.1038/s41598-020-72685-1.

[10] S. Mohan, C. Thirumalai, and G. Srivastava, "Effective heart disease prediction using hybrid machine learning techniques," *IEEE Access*, vol. 7, pp. 81542–81554, 2019, doi: 10.1109/ACCESS.2019.2923707.

### NHANES-based studies (closest to this paper)

★ [11] A. Dinh, S. Miertschin, A. Young, and S. D. Mohanty, "A data-driven approach to predicting diabetes and cardiovascular disease with machine learning," *BMC Med. Inform. Decis. Mak.*, vol. 19, no. 1, p. 211, 2019, doi: 10.1186/s12911-019-0918-5.

★ [12] A. Martin-Morales, M. Yamamoto, M. Inoue, T. Vu, R. Dawadi, and M. Araki, "Predicting cardiovascular disease mortality: Leveraging machine learning for comprehensive assessment of health and nutrition variables," *Nutrients*, vol. 15, no. 18, p. 3937, 2023, doi: 10.3390/nu15183937.

★ [13] M. Ahiduzzaman and M. N. Hasan, "Interpretable machine learning for cardiovascular risk prediction: Insights from NHANES dietary and health data," *PLoS ONE*, vol. 20, no. 11, p. e0335915, 2025, doi: 10.1371/journal.pone.0335915.

### Classical ML vs. deep learning on tabular / clinical data

★ [14] E. Christodoulou, J. Ma, G. S. Collins, E. W. Steyerberg, J. Y. Verbakel, and B. Van Calster, "A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models," *J. Clin. Epidemiol.*, vol. 110, pp. 12–22, 2019, doi: 10.1016/j.jclinepi.2019.02.004.

★ [15] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 35, 2022, pp. 507–520.

★ [16] R. Shwartz-Ziv and A. Armon, "Tabular data: Deep learning is not all you need," *Inf. Fusion*, vol. 81, pp. 84–90, 2022, doi: 10.1016/j.inffus.2021.11.011.

[17] V. Borisov, T. Leemann, K. Seßler, J. Haug, M. Pawelczyk, and G. Kasneci, "Deep neural networks and tabular data: A survey," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 35, no. 6, pp. 7499–7519, 2024, doi: 10.1109/TNNLS.2022.3229161.

### Explainability in healthcare

★ [18] C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nat. Mach. Intell.*, vol. 1, no. 5, pp. 206–215, 2019, doi: 10.1038/s42256-019-0048-x.

[19] G. Stiglic, P. Kocbek, N. Fijacko, M. Zitnik, K. Verbert, and L. Cilar, "Interpretability of machine learning-based prediction models in healthcare," *WIREs Data Min. Knowl. Discov.*, vol. 10, no. 5, p. e1379, 2020, doi: 10.1002/widm.1379.

★ [20] S. Krishna, T. Han, A. Gu, S. Wu, S. Jabbari, and H. Lakkaraju, "The disagreement problem in explainable machine learning: A practitioner's perspective," *Trans. Mach. Learn. Res.*, 2024. [Online]. Available: arXiv:2202.01602 †

## C. Methodology: data, models, tools

★ [21] Centers for Disease Control and Prevention, National Center for Health Statistics, "National Health and Nutrition Examination Survey data, 2011–2018," Hyattsville, MD, USA: U.S. Dept. of Health and Human Services. [Online]. Available: https://wwwn.cdc.gov/nchs/nhanes/

★ [22] C. L. Johnson, R. Paulose-Ram, C. L. Ogden, *et al.*, "National Health and Nutrition Examination Survey: Analytic guidelines, 1999–2010," *Vital Health Stat. 2*, no. 161, pp. 1–24, 2013. (Source for combining cycles and dividing the weights by the number of cycles.)

★ [23] L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

★ [24] C. Cortes and V. Vapnik, "Support-vector networks," *Mach. Learn.*, vol. 20, no. 3, pp. 273–297, 1995, doi: 10.1007/BF00994018.

[25] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in *Advances in Large Margin Classifiers*, A. J. Smola *et al.*, Eds. Cambridge, MA, USA: MIT Press, 1999, pp. 61–74. †

★ [26] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

★ [45] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting deep learning models for tabular data," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 34, 2021, pp. 18932–18943. † (Source of the ResNet and FT-Transformer models; numbered 45 because it was added last)

[27] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929–1958, 2014. †

[28] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. 32nd Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 37, 2015, pp. 448–456. †

[29] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019. † (AdamW optimiser)

[30] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *J. Mach. Learn. Res.*, vol. 13, pp. 281–305, 2012. † (XGBoost random search)

★ [31] F. Pedregosa, G. Varoquaux, A. Gramfort, *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011. †

[32] A. Paszke, S. Gross, F. Massa, *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035. †

### Evaluation metrics and statistics

★ [33] E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988, doi: 10.2307/2531595.

[34] X. Sun and W. Xu, "Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves," *IEEE Signal Process. Lett.*, vol. 21, no. 11, pp. 1389–1393, 2014, doi: 10.1109/LSP.2014.2337313. (The DeLong implementation used in the code)

[35] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, vol. 10, no. 3, p. e0118432, 2015, doi: 10.1371/journal.pone.0118432. (Justifies reporting PR-AUC at 11% prevalence)

[36] G. W. Brier, "Verification of forecasts expressed in terms of probability," *Mon. Weather Rev.*, vol. 78, no. 1, pp. 1–3, 1950, doi: 10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2.

[37] M. P. Naeini, G. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in *Proc. AAAI Conf. Artif. Intell.*, vol. 29, no. 1, 2015, doi: 10.1609/aaai.v29i1.9602. (ECE)

★ [38] B. Van Calster, D. J. McLernon, M. van Smeden, L. Wynants, and E. W. Steyerberg, "Calibration: The Achilles heel of predictive analytics," *BMC Med.*, vol. 17, no. 1, p. 230, 2019, doi: 10.1186/s12916-019-1466-7. (Calibration slope and intercept)

### SHAP and stability

★ [39] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017, pp. 4765–4774. †

★ [40] S. M. Lundberg, G. Erion, H. Chen, *et al.*, "From local explanations to global understanding with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020, doi: 10.1038/s42256-019-0138-9. (TreeSHAP, used for the validation check)

[41] M. G. Kendall and B. Babington Smith, "The problem of m rankings," *Ann. Math. Statist.*, vol. 10, no. 3, pp. 275–287, 1939, doi: 10.1214/aoms/1177732186. (Kendall's W)

[42] G. Hooker, L. Mentch, and S. Zhou, "Unrestricted permutation forces extrapolation: Variable importance requires at least one more model, or there is no free variable importance," *Stat. Comput.*, vol. 31, no. 6, p. 82, 2021, doi: 10.1007/s11222-021-10057-z. (Limitation of permutation-based attributions with correlated features)

## D. Discussion: reporting standards

★ [43] G. S. Collins, K. G. M. Moons, P. Dhiman, *et al.*, "TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ*, vol. 385, p. e078378, 2024, doi: 10.1136/bmj-2023-078378.

[44] E. W. Steyerberg and Y. Vergouwe, "Towards better clinical prediction models: Seven steps for development and an ABCD for validation," *Eur. Heart J.*, vol. 35, no. 29, pp. 1925–1931, 2014, doi: 10.1093/eurheartj/ehu207.

> Cite TRIPOD+AI [43], not the 2015 TRIPOD statement. TRIPOD+AI officially replaces it, and reviewers check this.

---

## Related Work summary table (for Section II)

Your outline asks Related Work to compare 5–8 studies and say what they did not evaluate. This table does that.

| Study | Data | Models | Explainability | What it did **not** do (gap this paper fills) |
|---|---|---|---|---|
| Weng *et al.* 2017 [6] | UK CPRD, 378k patients | LR, RF, GBM, NN vs ACC/AHA score | none | No SHAP; no calibration comparison; no attribution agreement across models |
| Ambale-Venkatesh *et al.* 2017 [8] | MESA, 6.8k | Random survival forest vs Cox | variable importance from one model | One model family; no deep learning comparison |
| Alaa *et al.* 2019 [7] | UK Biobank, 423k | AutoML ensemble vs Framingham | AutoML variable ranking | No head-to-head classical vs MLP; no explanation stability analysis |
| Dinh *et al.* 2019 [11] | NHANES 1999–2014 | LR, SVM, RF, gradient boosting, ensemble | feature importance | No MLP; no calibration analysis; no cross-model attribution agreement |
| Krittanawong *et al.* 2020 [9] | meta-analysis | many | n/a | Pooled AUCs only; heterogeneous outcomes and no shared evaluation |
| Martin-Morales *et al.* 2023 [12] | NHANES 2003–2012, CVD death | LR, SVM, RF, XGB, LightGBM | SHAP | No deep learning; no attribution agreement across models; undersampled data |
| Ahiduzzaman & Hasan 2025 [13] | NHANES 2017–2023 | LR, RF, SVM, XGB, LightGBM | SHAP + LIME | No MLP; no attribution stability across or within models |
| Christodoulou *et al.* 2019 [14] | systematic review, 71 studies | ML vs LR | n/a | Shows ML ≈ LR in clinical data; motivates this comparison |

**Gap statement you can use:** previous NHANES CVD studies compared classical models or reported SHAP for one model. None (a) compared classical models with a tuned MLP under the same pipeline, (b) measured whether SHAP attributions agree across model families and stay stable across refits, or (c) checked calibration and temporal validity together.

## Where to cite what

- **Intro, paragraph 1 (burden):** [1], [2]
- **Intro (risk scores, nonlinearity):** [3], [4], [5]
- **Intro (DL not guaranteed better on tabular data):** [14]–[16]
- **Intro (explainability, attribution ≠ causation):** [18], [39]
- **Methods, data:** [21], [22]
- **Methods, models:** LR + [31]; RF [23]; SVM [24], [25]; XGB [26]; MLP [27]–[29], [32]; tuning [30], [31]
- **Methods, metrics:** [33]–[38]
- **Methods, SHAP:** [39], [40]; Kendall's W [41]
- **Discussion (performance parity):** [14]–[17]
- **Discussion (family-dependent attributions):** [20], [42]
- **Discussion (calibration and reporting):** [38], [43], [44]
