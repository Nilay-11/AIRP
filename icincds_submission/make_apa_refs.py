"""Build the APA reference list (template style: up to 7 authors listed in full; otherwise first 6, ..., & last)
from Crossref metadata (DOI entries) plus checked manual entries (no-DOI items). *...* marks italics."""
import json
import re
import sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))


def initials(given):
    parts = re.split(r"[\s]+", given.replace(".", ". ").strip())
    out = []
    for p in parts:
        if not p:
            continue
        if "-" in p.strip("."):
            out.append("-".join(q[0] + "." for q in p.strip(".").split("-") if q))
        else:
            out.append(p[0] + ".")
    return " ".join(out)


def fix_family(f):
    f = f.replace("’", "'")
    return f[:1] + f[1:].lower() if f.isupper() else f


def authors_apa(auth):
    people = [(fix_family(f), initials(g)) for f, g in auth if g]  # drop group-name pseudo-authors
    names = [f"{f}, {g}" for f, g in people]
    if len(names) == 1:
        return names[0]
    if len(names) <= 7:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:6]) + ", ... & " + names[-1]


def sentence_case(t):
    t = re.sub(r"\s+", " ", t).strip().rstrip(".")
    return t


OVERRIDE_AUTH = {  # Crossref quirks corrected
    "kendall1939": "Kendall, M. G., & Babington Smith, B.",
    "dagostino2008": "D'Agostino, R. B., Sr., Vasan, R. S., Pencina, M. J., Wolf, P. A., Cobain, M., Massaro, J. M., & Kannel, W. B.",
    "goff2014": None,
}
refs = {}
for k, m in meta.items():
    a = OVERRIDE_AUTH.get(k) or authors_apa(m["authors"])
    vol = f", *{m['volume']}*" if m["volume"] else ""
    iss = f"({m['issue']})" if m["issue"] and not str(m["issue"]).startswith("jan") else ""
    pg = f", {m['page'].replace('-', '–')}" if m["page"] else ""
    title = sentence_case(m["title"])
    if k == "chen2016":
        refs[k] = (f"{a} ({m['year']}). {title}. In *Proceedings of the 22nd ACM SIGKDD International Conference on "
                   f"Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/{m['doi']}")
        continue
    if k == "naeini2015":
        refs[k] = (f"{a} ({m['year']}). {title}. In *Proceedings of the AAAI Conference on Artificial Intelligence* "
                   f"(Vol. 29, No. 1). https://doi.org/{m['doi']}")
        continue
    cont = m["container"].replace("&amp;", "&")
    refs[k] = f"{a} ({m['year']}). {title}. *{cont}*{vol}{iss}{pg}. https://doi.org/{m['doi']}"

MANUAL = {
    "lundberg2017": "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 4765–4774).",
    "grinsztajn2022": "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 507–520).",
    "gorishniy2021": "Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. In *Advances in Neural Information Processing Systems* (Vol. 34, pp. 18932–18943).",
    "krishna2024": "Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., & Lakkaraju, H. (2024). The disagreement problem in explainable machine learning: A practitioner's perspective. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2202.01602",
    "platt1999": "Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press.",
    "srivastava2014": "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, *15*(56), 1929–1958.",
    "ioffe2015": "Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (PMLR Vol. 37, pp. 448–456).",
    "loshchilov2019": "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations*.",
    "bergstra2012": "Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, *13*, 281–305.",
    "pedregosa2011": "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.",
    "paszke2019": "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems* (Vol. 32, pp. 8024–8035).",
    "johnson2013": "Johnson, C. L., Paulose-Ram, R., Ogden, C. L., Carroll, M. D., Kruszon-Moran, D., Dohrmann, S. M., & Curtin, L. R. (2013). National Health and Nutrition Examination Survey: Analytic guidelines, 1999–2010. *Vital and Health Statistics, Series 2*(161), 1–24.",
    "nhanes": "Centers for Disease Control and Prevention, National Center for Health Statistics. (n.d.). *National Health and Nutrition Examination Survey data, 2011–2018*. U.S. Department of Health and Human Services. https://wwwn.cdc.gov/nchs/nhanes/",
}
refs.update(MANUAL)
if refs.get("goff2014") is None or "goff2014" in OVERRIDE_AUTH:
    m = meta["goff2014"]
    refs["goff2014"] = (f"{authors_apa(m['authors'])} ({m['year']}). 2013 ACC/AHA guideline on the assessment of cardiovascular risk: "
                        f"A report of the American College of Cardiology/American Heart Association Task Force on Practice Guidelines. "
                        f"*Circulation*, *129*(25 Suppl. 2), S49–S73. https://doi.org/{m['doi']}")
json.dump(refs, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k in sorted(refs, key=lambda k: refs[k].lower()):
    print(f"[{k}] {refs[k]}\n")
