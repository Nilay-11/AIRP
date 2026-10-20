"""Convert build_manuscript.js from APA author-year citations to the ICINCDS template style:
numbered in-text citations [1, 2] in order of first appearance, and a numbered reference list (APA-formatted entries)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JS = ROOT / "build_manuscript.js"
s = JS.read_text(encoding="utf-8")

CITE = {  # in-text author-year form -> refs key
    "Roth et al., 2020": "roth2020", "Tsao et al., 2023": "tsao2023", "D’Agostino et al., 2008": "dagostino2008",
    "Goff et al., 2014": "goff2014", "Gorishniy et al., 2021": "gorishniy2021", "Christodoulou et al., 2019": "christodoulou2019",
    "Grinsztajn et al., 2022": "grinsztajn2022", "Shwartz-Ziv & Armon, 2022": "shwartzziv2022", "Van Calster et al., 2019": "vancalster2019",
    "Lundberg & Lee, 2017": "lundberg2017", "Rudin, 2019": "rudin2019", "Centers for Disease Control and Prevention, n.d.": "nhanes",
    "Johnson et al., 2013": "johnson2013", "Krittanawong et al., 2020": "krittanawong2020", "Krishna et al., 2024": "krishna2024",
    "Breiman, 2001": "breiman2001", "Cortes & Vapnik, 1995": "cortes1995", "Chen & Guestrin, 2016": "chen2016",
    "Saito & Rehmsmeier, 2015": "saito2015", "DeLong et al., 1988": "delong1988", "Holm, 1979": "holm1979",
    "Lundberg et al., 2020": "lundberg2020", "Collins et al., 2024": "collins2024", "Weng et al., 2017": "weng2017",
    "Martin-Morales et al., 2023": "martinmorales2023",
}
NARR = {  # narrative form -> (replacement prefix, key)
    "Weng et al. (2017)": ("Weng et al.", "weng2017"), "Alaa et al. (2019)": ("Alaa et al.", "alaa2019"),
    "Ambale-Venkatesh et al. (2017)": ("Ambale-Venkatesh et al.", "ambale2017"), "Dinh et al. (2019)": ("Dinh et al.", "dinh2019"),
    "Martin-Morales et al. (2023)": ("Martin-Morales et al.", "martinmorales2023"),
    "Ahiduzzaman and Hasan (2025)": ("Ahiduzzaman and Hasan", "ahiduzzaman2025"),
}

# body text = everything before the references block
cut = s.index('C.push(heading("References"));')
body, tail = s[:cut], s[cut:]

order = []
def num(k):
    if k not in order:
        order.append(k)
    return order.index(k) + 1

pat_narr = "|".join(re.escape(k) for k in NARR)
pat = re.compile(r"(" + pat_narr + r")|\(([^()]*)\)")

def repl(m):
    if m.group(1):
        pre, k = NARR[m.group(1)]
        return f"{pre} [{num(k)}]"
    parts = [p.strip() for p in m.group(2).split(";")]
    keys = [CITE.get(p) for p in parts]
    if not any(keys):
        return m.group(0)
    labels = [p for p, k in zip(parts, keys) if not k]
    nums = sorted(num(k) for k in keys if k)
    cite = "[" + ", ".join(str(n) for n in nums) + "]"
    return f"({'; '.join(labels)}) {cite}" if labels else cite

body = pat.sub(repl, body)
# a citation that directly follows a word should be separated by exactly one space (it already is); tidy "  ["
body = body.replace("  [", " [")

# references block: numbered list in order of first citation
old_refs = re.search(r'const sortKey = .*?\n(?:.*\n)*?.*runs\(r\) \}\)\)\);\n', tail)
assert old_refs, "reference block not found"
new_refs = (
    "const ORDER = " + json.dumps(order) + ";\n"
    "ORDER.forEach((k, i) => C.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 720, hanging: 720 }, "
    "spacing: { after: 80, line: 240 }, children: runs(`${i + 1}.\\t` + refs[k]) })));\n"
)
tail = tail.replace(old_refs.group(0), new_refs)
JS.write_text(body + tail, encoding="utf-8")

refs = json.loads((ROOT / "apa_refs.json").read_text(encoding="utf-8"))
unused = sorted(set(refs) - set(order))
missing = sorted(set(order) - set(refs))
leftover = re.findall(r"\([^()]*(?:19|20)\d{2}\)|\([^()]*n\.d\.\)", body)
print("references cited:", len(order), "| in apa_refs.json:", len(refs))
print("unused refs:", unused, "| missing refs:", missing)
print("leftover author-year patterns:", leftover[:10])
