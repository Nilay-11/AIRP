// Builds the ICINCDS 2026 manuscript (.docx) following the official Template.docx:
// US Letter, 1" margins, Times New Roman; title 14 pt bold centred; headings 12 pt bold; body 12 pt justified with
// 0.5" first-line indent; APA references. All numbers come from nhanes_study/results.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, VerticalAlign, LevelFormat,
} = require("docx");

const FIG = path.join(__dirname, "..", "nhanes_study", "figures");
const refs = JSON.parse(fs.readFileSync(path.join(__dirname, "apa_refs.json"), "utf8"));
const FONT = "Times New Roman";
const TEXT_W = 9360; // 6.5" in DXA

// ---- inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), font: FONT, ...base, bold: true }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: FONT, ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const body = (t) => new Paragraph({ children: runs(t), alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 720 } });
const heading = (t) => new Paragraph({ children: [new TextRun({ text: t, bold: true, font: FONT })], spacing: { before: 240 }, keepNext: true });
const subheading = (t) => new Paragraph({ children: [new TextRun({ text: t, bold: true, italics: true, font: FONT })], spacing: { before: 120 }, keepNext: true });
const center = (t, o = {}) => new Paragraph({ children: runs(t, o), alignment: AlignmentType.CENTER, spacing: { after: 60 } });

function figure(file, widthIn, caption) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); // PNG IHDR
  const pxW = Math.round(widthIn * 96), pxH = Math.round(pxW * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: pxW, height: pxH },
        altText: { title: caption.split(".")[0], description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(caption, { size: 20 }) }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(caption, header, rows, colWeights) {
  const total = colWeights.reduce((a, b) => a + b, 0);
  const widths = colWeights.map((w) => Math.floor((TEXT_W * w) / total));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    shading: head ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0, line: 240 },
      children: runs(String(t), head ? { size: 18, bold: true } : { size: 18 }) })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 }, children: runs(caption, { size: 20 }) }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
             ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ content
const C = [];
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Classical Machine Learning vs. Deep Learning for Cardiovascular Disease Classification: An Explainable AI Study Using NHANES Data", bold: true, size: 28, font: FONT })] }));
// TODO (authors): replace placeholders with real names, affiliations and the corresponding author's institutional e-mail.
C.push(center("Author One¹*, Author Two¹, Author Three¹"));
C.push(center("¹Department of Computer Science, University Name, City, Country", { italics: true }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: runs("*Corresponding author: email@university.edu") }));

C.push(heading("Abstract"));
C.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: runs(
  "Cardiovascular disease (CVD) is a leading cause of death, yet it remains unclear whether deep learning improves CVD classification over classical machine learning on structured clinical data. Using four cycles of the U.S. National Health and Nutrition Examination Survey (NHANES 2011–2018; n = 21,552 adults, 2,369 with CVD), this study compared four classical models (Logistic Regression, Random Forest, Support Vector Machine, XGBoost) with three deep learning models (multilayer perceptron, tabular ResNet, FT-Transformer) using 20 demographic, clinical and laboratory features. Models were evaluated for discrimination (ROC-AUC, PR-AUC, DeLong test), calibration (Brier score, expected calibration error) and computational cost, and interpreted with SHAP. All models performed similarly (ROC-AUC 0.837–0.848). XGBoost achieved the highest AUC, and no deep learning model significantly outperformed any classical model, although the FT-Transformer required 119 times longer tuning than XGBoost. Age was the most influential feature for every model. Feature rankings were broadly concordant (Kendall’s W = 0.75) but clustered by the shape of the decision function, with the deep models agreeing closely with each other (W = 0.92). Inverse attributions for cholesterol reflected treatment after diagnosis, indicating association rather than causation. For structured health survey data, deep learning offered no measurable advantage over classical machine learning.") }));
C.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: runs("**Keywords:** cardiovascular disease, machine learning, deep learning, NHANES, explainable artificial intelligence, SHAP") }));

// ---- Introduction
C.push(heading("Introduction"));
[
  "Cardiovascular diseases (CVDs) are a major public health concern associated with demographic, physiological, biochemical and behavioral factors (Roth et al., 2020; Tsao et al., 2023). Clinical variables such as age, blood pressure, cholesterol, glucose concentration and body mass index provide valuable information about cardiovascular health and underpin established risk scores (D’Agostino et al., 2008; Goff et al., 2014). However, relationships among these variables can be nonlinear and interdependent, which motivates the application of data-driven predictive methods.",
  "Classical machine learning algorithms, including Logistic Regression, Support Vector Machines, Random Forests and gradient-boosted decision trees, are widely applicable to structured clinical data. These methods offer different trade-offs between interpretability, computational efficiency and the ability to capture nonlinear relationships. Deep learning models, from multilayer perceptrons to Transformer architectures designed for tabular data (Gorishniy et al., 2021), can learn complex feature interactions but may require careful regularization and sufficient training data. Their greater complexity does not necessarily guarantee superior performance on tabular health datasets (Christodoulou et al., 2019; Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022).",
  "Model evaluation in clinical applications should extend beyond predictive accuracy. Identifying which features influence predictions, determining whether their contributions remain stable across algorithms, and assessing probability calibration are important for judging model reliability (Van Calster et al., 2019). Explainable artificial intelligence techniques, including SHapley Additive exPlanations (SHAP; Lundberg & Lee, 2017), provide a framework for examining feature contributions. However, model attribution indicates predictive influence rather than causation (Rudin, 2019).",
  "The National Health and Nutrition Examination Survey (NHANES), conducted by the United States Centers for Disease Control and Prevention, combines health interviews, physical examinations and laboratory measurements (Centers for Disease Control and Prevention, n.d.). Integrating four survey periods provides an opportunity to compare predictive models on population-level clinical data. Such integration requires consistent variable definitions, appropriate handling of missing data, and consideration of survey design and changes in data collection (Johnson et al., 2013).",
  "This study compares four classical machine learning models with three deep learning models for CVD classification using harmonized NHANES data. It evaluates predictive performance, calibration and computational efficiency, and examines clinical feature contributions through SHAP and feature-ablation experiments. It further investigates whether different model families identify consistent influential features, providing a comparative assessment of predictive effectiveness and interpretability on structured clinical data.",
].forEach((t) => C.push(body(t)));

// ---- Related Work
C.push(heading("Related Work"));
[
  "Machine learning for CVD risk has been studied in large cohorts. Weng et al. (2017) compared logistic regression, random forests, gradient boosting and neural networks with the ACC/AHA risk score in 378,256 UK primary-care patients; the neural network exceeded logistic regression by only 0.004 AUC. Alaa et al. (2019) showed that an automated machine-learning ensemble outperformed the Framingham score in 423,604 UK Biobank participants, largely by using many more variables, and Ambale-Venkatesh et al. (2017) applied random survival forests to the Multi-Ethnic Study of Atherosclerosis. A meta-analysis of 103 cohorts found overlapping performance across algorithm families and substantial heterogeneity between studies (Krittanawong et al., 2020).",
  "Several studies have used NHANES. Dinh et al. (2019) reported an ensemble AUC of 0.831 for CVD using 131 variables. Martin-Morales et al. (2023) predicted CVD mortality and identified age and systolic blood pressure as key SHAP features, and Ahiduzzaman and Hasan (2025) compared five classical models with SHAP and LIME on NHANES 2017–2023. None of these studies evaluated deep learning models, and none measured whether attributions agree across models or remain stable across refits.",
  "Broader evidence suggests that machine learning rarely outperforms logistic regression in clinical prediction (Christodoulou et al., 2019) and that tree ensembles often match or exceed deep learning on medium-sized tabular data (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022). Explanation methods can also disagree with one another (Krishna et al., 2024). The present study addresses these gaps by comparing classical and deep models on identical data splits, with calibration, temporal validation, and cross-model and within-model SHAP stability analyses.",
].forEach((t) => C.push(body(t)));

// ---- Methodology
C.push(heading("Methodology"));
C.push(subheading("Data and cohort"));
C.push(body("Four NHANES cycles (2011–2012, 2013–2014, 2015–2016 and 2017–2018) were combined (Centers for Disease Control and Prevention, n.d.); the 2019–2020 cycle was excluded because data collection was interrupted by COVID-19. The analytic cohort comprised adults aged 20 years or older who completed the physical examination and had a determinable outcome (n = 21,552; 2,369 cases). Prevalence was 11.0% unweighted and 8.9% survey-weighted, using the examination weight divided by four as recommended for combined cycles (Johnson et al., 2013)."));
C.push(subheading("Outcome"));
C.push(body("CVD was defined as a self-reported physician diagnosis of congestive heart failure, coronary heart disease, angina, heart attack or stroke (NHANES items MCQ160B–F). Participants answering “no” to all five items were controls; those with missing or unknown answers were excluded (n = 94)."));
C.push(subheading("Features and preprocessing"));
C.push(body("Twenty features were harmonized across cycles: age, sex, race/ethnicity (four indicators), income-to-poverty ratio, education, systolic and diastolic blood pressure (mean of up to three readings), pulse rate, body mass index, waist circumference, total and HDL cholesterol, HbA1c, serum creatinine, uric acid, smoking status and diagnosed diabetes. Medication variables were excluded because treatment frequently follows a CVD diagnosis and would leak the label. Missingness was at most 9.8%. Median imputation and standardization were fitted inside each model pipeline on training folds only."));
C.push(subheading("Models and training"));
C.push(body("Data were split 80/20 with stratification (training n = 17,241; test n = 4,311). Hyperparameters were tuned by five-fold stratified cross-validation on the training set to maximize ROC-AUC. The classical models were L2-regularized Logistic Regression (LR), Random Forest (RF; Breiman, 2001), a radial-basis-function Support Vector Machine (SVM; Cortes & Vapnik, 1995) with Platt-scaled probabilities, and XGBoost (XGB; Chen & Guestrin, 2016) tuned by random search. The deep models were a multilayer perceptron (MLP) built from Linear–BatchNorm–ReLU–Dropout blocks, a tabular ResNet and an FT-Transformer (Gorishniy et al., 2021), which converts each feature into a token and applies multi-head self-attention. All three deep models were trained with the AdamW optimizer, binary cross-entropy loss and early stopping on a 15% validation split; the selected networks had 3,649, 68,353 and 69,589 parameters, respectively. For ResNet and FT-Transformer the best configurations lay on a plateau, with the top configurations within 0.003 cross-validated AUC of each other, below the cross-validation standard deviation. Models were implemented in scikit-learn, XGBoost and PyTorch. Each model’s decision threshold maximized the F1-score on out-of-fold training predictions."));
C.push(subheading("Evaluation"));
C.push(body("Test-set performance was assessed with ROC-AUC (with 1,000-sample bootstrap confidence intervals), PR-AUC (Saito & Rehmsmeier, 2015), precision, recall, specificity, F1-score, Brier score, expected calibration error (ECE) and calibration slope (Van Calster et al., 2019). Paired AUC differences were tested with DeLong’s method (DeLong et al., 1988). Robustness was assessed over five random splits, by temporal validation (training on 2011–2016 and testing on 2017–2018), and with survey-weighted AUC."));
C.push(subheading("Explainability and stability"));
C.push(body("The same model-agnostic permutation SHAP estimator (Lundberg & Lee, 2017) was applied to every model’s predicted probability (300 test rows, 50 background rows); its rankings agreed with exact TreeSHAP (Lundberg et al., 2020) for XGBoost (Spearman ρ = 0.989). Agreement between models was measured with pairwise Spearman correlation of mean absolute SHAP values and with Kendall’s W; stability within each model was measured across five bootstrap refits. Feature-group ablation removed one of eight feature groups at a time, retrained each model and averaged the change in AUC over five splits. All final models and reported timings used a CPU; FT-Transformer refits for the ablation and stability analyses used a GPU."));

// ---- Results
C.push(heading("Results"));
C.push(subheading("Study population"));
C.push(body("Table 1 compares survey-weighted characteristics by CVD status. Participants with CVD were older (65.0 vs. 46.1 years), more often had diagnosed diabetes (33.0% vs. 8.3%) and were more often former smokers (38.0% vs. 23.1%). They had higher systolic blood pressure, waist circumference, HbA1c and creatinine but, notably, lower total cholesterol (177.5 vs. 193.0 mg/dL) and diastolic blood pressure (67.8 vs. 71.3 mmHg)."));
C.push(...table("**Table 1.** Survey-weighted characteristics of the study population",
  ["Characteristic", "Overall", "No CVD", "CVD"],
  [["Unweighted n", "21,552", "19,183", "2,369"], ["Age, years, mean (SD)", "47.7 (17.1)", "46.1 (16.5)", "65.0 (13.1)"],
   ["Male sex, %", "48.0", "47.5", "54.0"], ["Income-to-poverty ratio, mean (SD)", "3.0 (1.7)", "3.0 (1.7)", "2.5 (1.6)"],
   ["Systolic BP, mmHg, mean (SD)", "122.8 (17.4)", "122.0 (16.8)", "130.4 (20.8)"], ["Diastolic BP, mmHg, mean (SD)", "71.0 (12.0)", "71.3 (11.7)", "67.8 (14.5)"],
   ["BMI, kg/m², mean (SD)", "29.3 (7.0)", "29.1 (7.0)", "30.8 (7.3)"], ["Waist circumference, cm, mean (SD)", "99.9 (16.7)", "99.3 (16.6)", "106.6 (16.2)"],
   ["Total cholesterol, mg/dL, mean (SD)", "191.6 (41.4)", "193.0 (41.0)", "177.5 (43.0)"], ["HDL cholesterol, mg/dL, mean (SD)", "53.9 (16.5)", "54.2 (16.5)", "50.4 (16.4)"],
   ["HbA1c, %, mean (SD)", "5.7 (1.0)", "5.6 (0.9)", "6.2 (1.3)"], ["Serum creatinine, mg/dL, mean (SD)", "0.9 (0.4)", "0.9 (0.3)", "1.1 (0.6)"],
   ["Diagnosed diabetes, %", "10.5", "8.3", "33.0"], ["Current smoker, %", "18.9", "18.5", "22.4"], ["Former smoker, %", "24.5", "23.1", "38.0"]],
  [3.2, 1.3, 1.3, 1.3]));

C.push(subheading("Predictive performance and calibration"));
C.push(body("Table 2 reports test-set performance. Discrimination was remarkably similar: ROC-AUC ranged only from 0.837 (ResNet) to 0.848 (XGBoost), and all 95% confidence intervals overlapped. XGBoost achieved the highest ROC-AUC (0.848, 95% CI 0.829–0.863) and the lowest Brier score (0.078). Among the deep models, the FT-Transformer performed best (ROC-AUC 0.843) and achieved the highest PR-AUC of all models (0.419). Cross-validation AUCs (0.846–0.850) closely matched test AUCs, indicating little overfitting. All models were well calibrated (Fig. 1b): ECE was at most 0.021, the FT-Transformer had the calibration slope closest to one (0.99), ResNet (0.87) and the MLP (0.91) were slightly over-confident, and Random Forest (1.16) was slightly under-confident."));
C.push(...table("**Table 2.** Test-set performance of the seven models (n = 4,311; 474 CVD cases)",
  ["Model", "Family", "ROC-AUC [95% CI]", "PR-AUC", "Precision", "Recall", "F1", "Brier", "ECE", "Slope"],
  [["Logistic Regression", "Classical", "0.839 [0.821–0.855]", "0.392", "0.383", "0.561", "0.455", "0.080", "0.010", "1.04"],
   ["Random Forest", "Classical", "0.839 [0.821–0.856]", "0.407", "0.361", "0.586", "0.447", "0.080", "0.010", "1.16"],
   ["SVM (RBF)", "Classical", "0.839 [0.821–0.854]", "0.396", "0.362", "0.584", "0.447", "0.080", "0.007", "0.97"],
   ["**XGBoost**", "Classical", "**0.848 [0.829–0.863]**", "0.418", "0.373", "0.572", "0.451", "**0.078**", "0.009", "1.04"],
   ["MLP", "Deep", "0.842 [0.825–0.858]", "0.398", "0.390", "0.523", "0.447", "0.080", "0.018", "0.91"],
   ["ResNet", "Deep", "0.837 [0.819–0.855]", "0.402", "0.377", "0.527", "0.439", "0.080", "0.021", "0.87"],
   ["FT-Transformer", "Deep", "0.843 [0.825–0.859]", "**0.419**", "0.371", "0.586", "0.455", "0.079", "0.012", "0.99"]],
  [2.0, 1.15, 2.1, 0.9, 1.0, 0.9, 0.8, 0.8, 0.8, 0.8]));
C.push(...figure("fig1_roc_calibration.png", 6.0, "**Fig. 1.** (a) ROC curves of the seven models on the test set (AUC in legend). (b) Calibration by decile of predicted risk. Solid lines denote classical models and dashed lines deep-learning models."));

C.push(subheading("Statistical comparison"));
C.push(body("Pairwise DeLong tests (Table 3) showed that XGBoost was significantly better than LR (p = 0.005), RF (p = 0.001), SVM (p = 0.002) and ResNet (p = 0.002), but not than the MLP (p = 0.064) or the FT-Transformer (p = 0.075). No deep learning model was significantly better than any classical model: wherever a deep model had the higher AUC (the MLP and FT-Transformer versus LR, RF and SVM), p ranged from 0.161 to 0.460. Within the deep family, the MLP was significantly better than ResNet (p = 0.031)."));
C.push(...table("**Table 3.** Pairwise DeLong test p-values for differences in test ROC-AUC (bold: p < 0.05)",
  ["", "LR", "RF", "SVM", "XGB", "MLP", "ResNet", "FT-T"],
  [["LR", "–", "0.944", "0.876", "**0.005**", "0.321", "0.675", "0.169"],
   ["RF", "0.944", "–", "1.000", "**0.001**", "0.460", "0.643", "0.239"],
   ["SVM", "0.876", "1.000", "–", "**0.002**", "0.274", "0.547", "0.161"],
   ["XGB", "**0.005**", "**0.001**", "**0.002**", "–", "0.064", "**0.002**", "0.075"],
   ["MLP", "0.321", "0.460", "0.274", "0.064", "–", "**0.031**", "0.682"],
   ["ResNet", "0.675", "0.643", "0.547", "**0.002**", "**0.031**", "–", "0.075"],
   ["FT-T", "0.169", "0.239", "0.161", "0.075", "0.682", "0.075", "–"]],
  [1.1, 1, 1, 1, 1, 1, 1, 1]));

C.push(subheading("Robustness and computational cost"));
C.push(body("Across five random splits, XGBoost had the highest mean AUC (0.846), closely followed by the FT-Transformer (0.845), and every model was within about one standard deviation of the others (Table 4). Under temporal validation all seven models converged to an AUC of 0.837–0.840, and survey-weighted test AUCs (0.851–0.862) preserved the ranking. Computational cost, however, differed by orders of magnitude: the FT-Transformer required 5,779 s of tuning, 119 times longer than XGBoost (49 s), and its final training took about 5,000 times longer than that of Logistic Regression."));
C.push(...table("**Table 4.** Robustness of discrimination and computational cost (CPU, wall-clock)",
  ["Model", "AUC, 5 splits (mean ± SD)", "Temporal AUC", "Weighted AUC", "Tuning (s)", "Training (s)", "Inference (ms/1k)"],
  [["Logistic Regression", "0.842 ± 0.002", "0.838", "0.852", "5", "0.05", "1.1"],
   ["Random Forest", "0.841 ± 0.004", "0.837", "0.851", "271", "1.5", "24.8"],
   ["SVM (RBF)", "0.843 ± 0.003", "0.838", "0.853", "224", "47.4", "828"],
   ["XGBoost", "**0.846 ± 0.004**", "0.838", "**0.862**", "49", "0.6", "2.9"],
   ["MLP", "0.844 ± 0.003", "0.839", "0.855", "352", "9.0", "1.2"],
   ["ResNet", "0.841 ± 0.005", "**0.840**", "0.851", "913", "22.3", "8.8"],
   ["FT-Transformer", "0.845 ± 0.003", "0.837", "0.853", "5,779", "275.6", "351"]],
  [2.0, 1.7, 1.1, 1.1, 1.0, 1.0, 1.2]));

C.push(subheading("Feature attributions and stability"));
C.push(body("SHAP analysis identified age as the most influential feature for every model (Fig. 2). Total cholesterol, smoking status, diagnosed diabetes and income-to-poverty ratio completed the top five for LR, SVM and all three deep models, whereas RF and XGBoost ranked serum creatinine among their top five. For XGBoost, older age, smoking, diagnosed diabetes and higher creatinine increased predicted risk, whereas a higher income-to-poverty ratio and, counter-intuitively, higher total cholesterol decreased it."));
C.push(...figure("fig2_shap_importance_heatmap.png", 3.0, "**Fig. 2.** Mean absolute SHAP importance per model, scaled to each model’s top feature. Cell numbers are within-model ranks; the white line separates classical (left) from deep (right) models."));
C.push(body("Feature rankings were broadly concordant across all seven models (Kendall’s W = 0.752), but agreement followed the shape of the decision function rather than the classical/deep distinction (Table 5). The two tree ensembles agreed closely with each other (RF–XGBoost ρ = 0.941), while LR, SVM and the three deep models formed a second cluster (for example ResNet–FT-Transformer ρ = 0.935 and SVM–MLP ρ = 0.886). The deep models were highly concordant with one another (W = 0.921) compared with the classical models (W = 0.742). Across bootstrap refits, the tree ensembles produced the most stable full rankings (RF ρ = 0.973), the SVM and MLP kept a perfectly stable top five, and the FT-Transformer was the least stable (top-five Jaccard index 0.61)."));
C.push(...table("**Table 5.** SHAP ranking agreement between models (Spearman ρ) and within-model stability",
  ["", "LR", "RF", "SVM", "XGB", "MLP", "ResNet", "FT-T"],
  [["RF", "0.474", "", "", "", "", "", ""], ["SVM", "0.656", "0.653", "", "", "", "", ""],
   ["XGB", "0.553", "**0.941**", "0.663", "", "", "", ""], ["MLP", "0.743", "0.629", "**0.886**", "0.633", "", "", ""],
   ["ResNet", "0.666", "0.663", "0.698", "0.717", "0.863", "", ""], ["FT-T", "0.728", "0.597", "0.692", "0.681", "0.848", "**0.935**", ""],
   ["Bootstrap stability (ρ)", "0.846", "0.973", "0.732", "0.921", "0.805", "0.804", "0.761"],
   ["Top-5 stability (Jaccard)", "0.867", "0.767", "1.000", "0.733", "1.000", "0.767", "0.605"]],
  [2.0, 1, 1, 1, 1, 1, 1, 1]));

C.push(subheading("Feature-group ablation"));
C.push(body("Removing one feature group at a time confirmed the SHAP results (Fig. 3). Removing the demographic group caused by far the largest loss for every model (−0.037 to −0.063 AUC), largest for LR. Lipid, glycaemic, socioeconomic, renal/metabolic and smoking groups each cost 0.001–0.007 AUC, whereas removing anthropometric or blood-pressure features changed AUC by at most 0.003, within the variation between splits. The ordering of feature groups was nearly identical across all seven models."));
C.push(...figure("fig4_ablation.png", 4.2, "**Fig. 3.** Change in test ROC-AUC (× 100) when each feature group is removed and the model retrained, averaged over five random splits."));

// ---- Discussion
C.push(heading("Discussion"));
[
  "Neither the MLP nor the architectures designed for tabular data (ResNet and FT-Transformer) offered a measurable advantage over classical models. This agrees with benchmarks on medium-sized tabular data (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) and with clinical prediction studies (Christodoulou et al., 2019; Weng et al., 2017). Logistic Regression came within 0.009 AUC of the best model at a negligible computational cost, supporting the use of inherently interpretable models where their accuracy is comparable (Rudin, 2019).",
  "Similar accuracy did not imply similar explanations. Tree ensembles and smooth-boundary models distributed credit differently among correlated features, so attributions from a single model give an incomplete picture of which clinical factors matter (Krishna et al., 2024). Reporting SHAP results from several model families, together with their stability, offers a more reliable basis for interpretation.",
  "Higher total cholesterol lowered predicted CVD probability in every model, and blood pressure added almost no discriminative information. Both effects reflect treatment after diagnosis: survey-weighted statin use was 58.0% among participants with CVD versus 14.2% among those without, and antihypertensive use was 62.8% versus 20.1%. In contrast, NHANES studies of future CVD mortality find systolic blood pressure predictive (Martin-Morales et al., 2023). Because the outcome in this study is prevalent, self-reported CVD measured cross-sectionally, the models partly learn markers of diagnosis and treatment, and the attributions describe association only.",
  "This study has limitations. The outcome is self-reported and therefore subject to recall bias, age is top-coded at 80 years in NHANES, and survey weights were used for descriptive analysis and a weighted-AUC check rather than for model training. Reporting followed the TRIPOD+AI guidance (Collins et al., 2024).",
].forEach((t) => C.push(body(t)));

// ---- Conclusion
C.push(heading("Conclusion"));
C.push(body("On harmonized NHANES 2011–2018 data, four classical models and three deep learning models achieved nearly identical discrimination and calibration for CVD classification. No deep learning model was significantly better than a classical model, while the FT-Transformer was orders of magnitude more expensive to train. SHAP identified age as the dominant feature for all models, but attribution agreement depended on the shape of the decision function. Future work should use incident outcomes from linked mortality data, model treatment status explicitly, incorporate the survey design into training, and evaluate ensembles of tree-based and deep models."));

// ---- Declarations (Acknowledgments are reserved for funding by the conference guidelines)
C.push(heading("Data Availability"));
C.push(body("NHANES data are publicly available from the U.S. Centers for Disease Control and Prevention (Centers for Disease Control and Prevention, n.d.). Analysis code, random seeds, trained models and result files are available from the corresponding author on reasonable request."));
C.push(heading("Declaration of AI-Assisted Technologies"));
C.push(body("An AI assistant (Claude, Anthropic) was used to help develop the analysis code and draft the manuscript. The authors defined the research question, the study design, the models and the evaluation metrics, ran and verified the analyses, checked all reported results, and take full responsibility for the content of this publication. The AI assistant is not an author."));

// ---- References (APA, alphabetical)
C.push(heading("References"));
const sortKey = (s) => s.replace(/[^A-Za-z]/g, "").toLowerCase();
Object.values(refs).sort((a, b) => sortKey(a).localeCompare(sortKey(b))).forEach((r) =>
  C.push(new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 720, hanging: 720 }, spacing: { after: 80, line: 240 }, children: runs(r) })));

const doc = new Document({
  creator: "ICINCDS 2026 authors",
  title: "Classical Machine Learning vs. Deep Learning for Cardiovascular Disease Classification",
  styles: { default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { after: 160, line: 278 } } } } },
  sections: [{ properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: C }],
});
const out = process.argv[2] || path.join(__dirname, "..", "ICINCDS_2026_Manuscript_NHANES_CVD.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("written", out); });
